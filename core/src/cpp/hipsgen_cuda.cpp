/**
 * HipsGen CUDA - C++版本
 * 完全重写的HiPS生成器，使用CUDA加速双线性插值
 * 避开JNI问题，直接使用C++和CUDA
 */

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <limits>

// CUDA includes
#include <cuda_runtime.h>
#include "bilinear_interpolation_cuda.h"
#include "resource_manager.h"

// HiPS modules
#include "fits_io.h"
#include "coordinate_transform.h"
#include "healpix_util.h"
#include "hpx_finder.h"
#include "hips_tile_generator.h"
#include "fits_writer.h"
#include "allsky_generator.h"
#include "properties_generator.h"
#include "gpu_image_cache.h"
#include "gpu_batch_processor.h"
#include "gpu_full_processor.h"
#include "async_writer.h"

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Overlay modes (matching Java ModeOverlay)
enum class OverlayMode {
    NONE,       // Use first pixel value
    MEAN,       // Calculate mean (AVERAGE)
    FADING,     // Weighted mean based on edge distance
    ADD         // Sum of pixel values
};

// Helper function to get overlay mode name
std::string getOverlayModeName(OverlayMode mode) {
    switch (mode) {
        case OverlayMode::NONE: return "NONE";
        case OverlayMode::MEAN: return "MEAN/AVERAGE";
        case OverlayMode::FADING: return "FADING";
        case OverlayMode::ADD: return "ADD";
        default: return "UNKNOWN";
    }
}

// Configuration
struct Config {
    std::string inputDir;
    std::string outputDir;
    std::string hipsId = "TEST/cuda_hips";
    int orderMax = 3;
    bool autoOrder = true;  // 自动计算推荐order（基于图像分辨率）
    int maxThreads = 4;
    int tileWidth = 512;
    int bitpix = -32;  // float
    double blank = std::numeric_limits<double>::quiet_NaN();  // Use NaN as blank, matching Java
    double bzero = 0.0;
    double bscale = 1.0;
    std::string frame = "ICRS";  // 坐标系统
    
    // Reference image (img= in Java version)
    std::string imgEtalon;     // Reference image for default initializations
    bool hasUserBlank = false; // Whether user explicitly set blank value
    
    // Overlay mode (mode= in Java version)
    OverlayMode overlayMode = OverlayMode::MEAN;  // Default: AVERAGE/MEAN
    
    // Valid pixel range (validRange= in Java version)
    double validMin = -std::numeric_limits<double>::infinity();  // No lower limit by default
    double validMax = std::numeric_limits<double>::infinity();   // No upper limit by default
    bool hasValidRange = false;
    bool autoValidRange = false;  // 自动估计validRange
    double autoSampleRatio = 0.1; // 采样比例（默认10%文件，每文件1%像素）
    
    bool enableIndex = true;   // 是否执行INDEX阶段
    bool enableTiles = true;   // 是否执行TILES阶段
    bool verbose = false;      // 详细输出
    bool forceCPU = false;     // 强制使用CPU版本
    
    // DESI DR10 optimization options
    bool recursiveScan = false;  // 递归扫描子目录
    int maxFiles = 0;            // 限制处理文件数（0=无限制，用于调试）
    bool streamMode = true;      // 流式处理模式（不一次加载所有文件）
    int cacheSize = 100;         // 内存中缓存的FITS文件数
    bool skipErrors = true;      // 跳过损坏的文件继续处理
    std::string resumeFile = ""; // 断点续传文件路径
    int progressInterval = 100;  // 进度报告间隔（文件数）
    std::string filePattern = "";  // 文件名过滤模式（支持通配符，如 "*-image-r.fits.fz"）
};


/**
 * 自动估计validRange
 * 策略：
 * 1. 首先尝试从FITS头读取DATAMIN/DATAMAX
 * 2. 如果头信息不可用，采样部分文件的部分像素进行统计
 * 3. 排除blank值，使用percentile计算有效像素的范围（排除极端值）
 */
struct ValidRangeEstimate {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
    double p1 = 0;   // 1st percentile
    double p99 = 0;  // 99th percentile
    long validPixels = 0;
    long totalPixels = 0;
    bool success = false;
};

ValidRangeEstimate estimateValidRange(
    const std::vector<std::string>& fitsFiles,
    double blankValue,
    double sampleRatio = 0.1,  // 采样10%的文件
    bool verbose = false
) {
    ValidRangeEstimate result;
    
    if (fitsFiles.empty()) return result;
    
    // 增加采样量：至少采样20%的文件
    sampleRatio = std::max(sampleRatio, 0.5);  // Sample at least 50% of files
    int numSampleFiles = std::max(1, (int)(fitsFiles.size() * sampleRatio));
    int fileStep = std::max(1, (int)(fitsFiles.size() / numSampleFiles));
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 收集采样值用于计算percentile
    std::vector<float> sampledValues;
    sampledValues.reserve(100000);  // 预分配
    
    int filesProcessed = 0;
    bool hasHeaderInfo = false;
    
    for (size_t fi = 0; fi < fitsFiles.size(); fi += fileStep) {
        const auto& filename = fitsFiles[fi];
        
        #ifdef USE_CFITSIO
        fitsfile* fptr = nullptr;
        int status = 0;
        
        if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
            continue;
        }
        
        // For .fz compressed files, move to the correct HDU with image data
        int hdunum = 0;
        fits_get_num_hdus(fptr, &hdunum, &status);
        
        // Try to find the first image HDU
        int imageHDU = 1;  // Default to HDU 1
        for (int h = 1; h <= hdunum && h <= 3; h++) {
            int hdutype = 0;
            status = 0;
            fits_movabs_hdu(fptr, h, &hdutype, &status);
            if (status == 0 && hdutype == IMAGE_HDU) {
                // Check if this HDU has actual image data
                int naxis = 0;
                fits_get_img_dim(fptr, &naxis, &status);
                if (status == 0 && naxis >= 2) {
                    imageHDU = h;
                    break;
                }
            }
        }
        
        // Move to the image HDU
        status = 0;
        fits_movabs_hdu(fptr, imageHDU, nullptr, &status);
        
        // 尝试读取DATAMIN/DATAMAX
        double datamin = 0, datamax = 0;
        int minStatus = 0, maxStatus = 0;
        fits_read_key(fptr, TDOUBLE, "DATAMIN", &datamin, nullptr, &minStatus);
        fits_read_key(fptr, TDOUBLE, "DATAMAX", &datamax, nullptr, &maxStatus);
        
        if (minStatus == 0 && maxStatus == 0 && 
            datamin != blankValue && datamax != blankValue &&
            datamax > datamin) {  // 额外验证
            if (datamin < result.minVal) result.minVal = datamin;
            if (datamax > result.maxVal) result.maxVal = datamax;
            hasHeaderInfo = true;
        }
        
        // 始终采样像素（即使有头信息，也收集样本用于验证）
        long naxes[2] = {0, 0};
        int naxis = 0;
        status = 0;
        fits_get_img_dim(fptr, &naxis, &status);
        fits_get_img_size(fptr, 2, naxes, &status);
        
        if (status == 0 && naxis >= 2) {
            long npix = naxes[0] * naxes[1];
            result.totalPixels += npix;
            
            // 采样更多行：5%的行
            int sampleRows = std::max(10, (int)(naxes[1] * 0.1));  // Sample 10% of rows
            int rowStep = std::max(1L, naxes[1] / sampleRows);
            
            std::vector<float> rowData(naxes[0]);
            
            for (long row = 0; row < naxes[1]; row += rowStep) {
                long fpixel[2] = {1, row + 1};
                long lpixel[2] = {naxes[0], row + 1};
                long inc[2] = {1, 1};
                
                status = 0;
                fits_read_subset(fptr, TFLOAT, fpixel, lpixel, inc, 
                                nullptr, rowData.data(), nullptr, &status);
                
                if (status == 0) {
                    for (long x = 0; x < naxes[0]; x++) {
                        float val = rowData[x];
                        if (!std::isnan(val) && val != blankValue) {
                            sampledValues.push_back(val);
                            result.validPixels++;
                        }
                    }
                }
            }
        }
        
        fits_close_file(fptr, &status);
        #endif
        
        filesProcessed++;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // 使用percentile计算范围（排除极端值）
    if (!sampledValues.empty()) {
        std::sort(sampledValues.begin(), sampledValues.end());
        
        size_t n = sampledValues.size();
        size_t p1_idx = n / 100;          // 1st percentile
        size_t p99_idx = n * 99 / 100;    // 99th percentile
        
        result.p1 = sampledValues[p1_idx];
        result.p99 = sampledValues[p99_idx];
        
        // 使用percentile作为validRange（比绝对min/max更稳健）
        if (!hasHeaderInfo || result.minVal <= 0) {
            result.minVal = sampledValues.front();  // Use actual min
            result.maxVal = sampledValues.back();   // Use actual max
        }
        
        result.success = true;
    }
    
    if (verbose) {
        std::cout << "ValidRange estimation: sampled " << filesProcessed << "/" << fitsFiles.size() 
                  << " files in " << duration.count() << " ms" << std::endl;
        std::cout << "  Valid pixels sampled: " << result.validPixels << std::endl;
        if (!sampledValues.empty()) {
            std::cout << "  Sampled range: [" << sampledValues.front() << ", " << sampledValues.back() << "]" << std::endl;
            std::cout << "  1st-99th percentile: [" << result.p1 << ", " << result.p99 << "]" << std::endl;
        }
    }
    
    return result;
}

// Tile任务结构（用于多线程处理）
struct TileTask {
    int order;
    long npix;
    
    TileTask(int o = 0, long n = 0) : order(o), npix(n) {}
};

// 线程安全的任务队列
class TileTaskQueue {
public:
    void push(const TileTask& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(task);
        condition_.notify_one();
    }
    
    bool pop(TileTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !done_) {
            condition_.wait(lock);
        }
        
        if (queue_.empty()) {
            return false;
        }
        
        task = queue_.front();
        queue_.pop();
        return true;
    }
    
    void setDone() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        condition_.notify_all();
    }
    
    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    std::queue<TileTask> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool done_ = false;
};

// 全局统计
struct Statistics {
    std::atomic<long> totalTiles{0};
    std::atomic<long> generatedTiles{0};
    std::atomic<long> skippedTiles{0};
    std::atomic<long> failedTiles{0};
    std::chrono::steady_clock::time_point startTime;
    
    void reset() {
        totalTiles = 0;
        generatedTiles = 0;
        skippedTiles = 0;
        failedTiles = 0;
        startTime = std::chrono::steady_clock::now();
    }
    
    void print() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        
        std::cout << "\n=== 统计信息 ===" << std::endl;
        std::cout << "总tiles数: " << totalTiles << std::endl;
        std::cout << "生成tiles数: " << generatedTiles << std::endl;
        std::cout << "跳过tiles数: " << skippedTiles << std::endl;
        std::cout << "失败tiles数: " << failedTiles << std::endl;
        std::cout << "Duration: " << duration << " seconds" << std::endl;
        if (generatedTiles > 0 && duration > 0) {
            std::cout << "Average speed: " << (generatedTiles / duration) << " tiles/second" << std::endl;
        }
    }
};

static Statistics g_stats;

// Global FITS file cache with LRU eviction
static LRUCache<std::string, FitsData> g_fitsCache(100, 0);  // Will be configured later


/**
 * 解析命令行参数
 */

/**
 * Scan input directory for FITS files
 * Supports recursive scanning and file limiting for DESI DR10
 */
std::vector<std::string> scanFitsFiles(const Config& config) {
    std::vector<std::string> filePaths;
    int fileCount = 0;
    
    std::cout << "Scanning input directory: " << config.inputDir << std::endl;
    if (config.recursiveScan) {
        std::cout << "  Mode: Recursive" << std::endl;
    } else {
        std::cout << "  Mode: Single directory" << std::endl;
    }
    if (config.maxFiles > 0) {
        std::cout << "  Limit: " << config.maxFiles << " files" << std::endl;
    }
    
    try {
        if (config.recursiveScan) {
            for (const auto& entry : fs::recursive_directory_iterator(config.inputDir)) {
                if (config.maxFiles > 0 && fileCount >= config.maxFiles) break;
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    bool isFits = (ext == ".fits" || ext == ".fit" || ext == ".img" || ext == ".fz");
                
                // Apply file pattern filter if specified
                if (isFits && !config.filePattern.empty()) {
                    std::string filename = entry.path().filename().string();
                    // Simple wildcard matching (supports * at start/end)
                    std::string pattern = config.filePattern;
                    bool matches = false;
                    
                    if (pattern.find('*') == 0 && pattern.find('*', 1) == std::string::npos) {
                        // Pattern: *.suffix
                        std::string suffix = pattern.substr(1);
                        matches = (filename.length() >= suffix.length() && 
                                 filename.compare(filename.length() - suffix.length(), suffix.length(), suffix) == 0);
                    } else if (pattern.find('*') == pattern.length() - 1 && pattern.find('*') == pattern.rfind('*')) {
                        // Pattern: prefix*
                        std::string prefix = pattern.substr(0, pattern.length() - 1);
                        matches = (filename.compare(0, prefix.length(), prefix) == 0);
                    } else if (pattern.find('*') != std::string::npos) {
                        // Pattern: prefix*suffix
                        size_t starPos = pattern.find('*');
                        std::string prefix = pattern.substr(0, starPos);
                        std::string suffix = pattern.substr(starPos + 1);
                        matches = (filename.length() >= prefix.length() + suffix.length() &&
                                 filename.compare(0, prefix.length(), prefix) == 0 &&
                                 filename.compare(filename.length() - suffix.length(), suffix.length(), suffix) == 0);
                    } else {
                        // Exact match
                        matches = (filename == pattern);
                    }
                    
                    if (!matches) {
                        isFits = false;  // Skip this file
                    }
                }
                
                if (isFits) {
                        filePaths.push_back(entry.path().string());
                        fileCount++;
                        if (config.progressInterval > 0 && fileCount % config.progressInterval == 0) {
                            std::cout << "  Found " << fileCount << " FITS files..." << std::endl;
                        }
                    }
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(config.inputDir)) {
                if (config.maxFiles > 0 && fileCount >= config.maxFiles) break;
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    bool isFits = (ext == ".fits" || ext == ".fit" || ext == ".img" || ext == ".fz");
                
                // Apply file pattern filter if specified
                if (isFits && !config.filePattern.empty()) {
                    std::string filename = entry.path().filename().string();
                    // Simple wildcard matching (supports * at start/end)
                    std::string pattern = config.filePattern;
                    bool matches = false;
                    
                    if (pattern.find('*') == 0 && pattern.find('*', 1) == std::string::npos) {
                        // Pattern: *.suffix
                        std::string suffix = pattern.substr(1);
                        matches = (filename.length() >= suffix.length() && 
                                 filename.compare(filename.length() - suffix.length(), suffix.length(), suffix) == 0);
                    } else if (pattern.find('*') == pattern.length() - 1 && pattern.find('*') == pattern.rfind('*')) {
                        // Pattern: prefix*
                        std::string prefix = pattern.substr(0, pattern.length() - 1);
                        matches = (filename.compare(0, prefix.length(), prefix) == 0);
                    } else if (pattern.find('*') != std::string::npos) {
                        // Pattern: prefix*suffix
                        size_t starPos = pattern.find('*');
                        std::string prefix = pattern.substr(0, starPos);
                        std::string suffix = pattern.substr(starPos + 1);
                        matches = (filename.length() >= prefix.length() + suffix.length() &&
                                 filename.compare(0, prefix.length(), prefix) == 0 &&
                                 filename.compare(filename.length() - suffix.length(), suffix.length(), suffix) == 0);
                    } else {
                        // Exact match
                        matches = (filename == pattern);
                    }
                    
                    if (!matches) {
                        isFits = false;  // Skip this file
                    }
                }
                
                if (isFits) {
                        filePaths.push_back(entry.path().string());
                        fileCount++;
                        if (config.progressInterval > 0 && fileCount % config.progressInterval == 0) {
                            std::cout << "  Found " << fileCount << " FITS files..." << std::endl;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }
    
    std::cout << "Total found: " << filePaths.size() << " FITS files" << std::endl;
    return filePaths;
}

/**
 * 根据图像分辨率计算推荐的 HEALPix order
 * HiPS tile 在给定 order 下的像素角分辨率 = (512 pixels/tile) 对应的角度
 * 每个 tile 覆盖的天区角度 ≈ (180°/nside) × (1/sqrt(3)) ≈ 58.6°/nside
 * 每像素角分辨率 ≈ 58.6°/(nside×512) = 58.6°/(2^order × 512)
 * 
 * @param pixelScale 图像像素角分辨率（角秒）
 * @param tileWidth HiPS tile 宽度（默认512）
 * @return 推荐的 order 值
 */
int calculateRecommendedOrder(double pixelScale, int tileWidth = 512) {
    // HiPS tile 分辨率公式：
    // tile_resolution (arcsec) = (180 * 3600) / (nside * tileWidth * sqrt(3))
    //                          ≈ 374400 / (nside * tileWidth)  [实际计算]
    //                          = 374400 / (2^order * tileWidth)
    // 
    // 为了匹配图像分辨率，需要：
    // tile_resolution ≤ pixelScale
    // 374400 / (2^order * tileWidth) ≤ pixelScale
    // 2^order ≥ 374400 / (pixelScale * tileWidth)
    // order ≥ log2(374400 / (pixelScale * tileWidth))
    
    const double HEALPIX_FACTOR = 374400.0;  // 约 180 * 3600 / sqrt(3) / π ≈ tile角度因子
    
    double minNside = HEALPIX_FACTOR / (pixelScale * tileWidth);
    int order = static_cast<int>(std::ceil(std::log2(minNside)));
    
    // 限制在合理范围内
    if (order < 0) order = 0;
    if (order > 15) order = 15;  // HiPS 最大支持 order 15
    
    return order;
}

/**
 * 从 FITS 文件中估计像素角分辨率
 * @param fitsFiles FITS 文件列表
 * @param sampleSize 采样数量
 * @return 中位数像素角分辨率（角秒），如果无法计算返回 -1
 */
double estimatePixelScale(const std::vector<std::string>& fitsFiles, int sampleSize = 10) {
    if (fitsFiles.empty()) return -1;
    
    std::vector<double> scales;
    int sampled = 0;
    
    for (const auto& filepath : fitsFiles) {
        if (sampled >= sampleSize) break;
        
        try {
            // 只读取 header，不读取像素数据
            fitsfile* fptr = nullptr;
            int status = 0;
            
            if (fits_open_file(&fptr, filepath.c_str(), READONLY, &status)) {
                continue;
            }
            
            // 检测 .fz 压缩文件，移动到正确的 HDU
            std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
            if (ext == "fz") {
                fits_movabs_hdu(fptr, 2, nullptr, &status);
                status = 0;
            }
            
            // 读取 CD 矩阵或 CDELT
            double cd1_1 = 0, cd1_2 = 0, cd2_1 = 0, cd2_2 = 0;
            double cdelt1 = 0, cdelt2 = 0;
            char comment[80];
            
            bool hasCD = true;
            if (fits_read_key(fptr, TDOUBLE, "CD1_1", &cd1_1, comment, &status)) {
                hasCD = false;
                status = 0;
            }
            
            if (hasCD) {
                fits_read_key(fptr, TDOUBLE, "CD1_2", &cd1_2, comment, &status);
                fits_read_key(fptr, TDOUBLE, "CD2_1", &cd2_1, comment, &status);
                fits_read_key(fptr, TDOUBLE, "CD2_2", &cd2_2, comment, &status);
                status = 0;
                
                // 计算像素尺度: sqrt(|det(CD)|) 或 取对角线平均
                double scale1 = std::sqrt(cd1_1*cd1_1 + cd2_1*cd2_1);
                double scale2 = std::sqrt(cd1_2*cd1_2 + cd2_2*cd2_2);
                double pixelScaleDeg = (scale1 + scale2) / 2.0;  // 度
                
                if (pixelScaleDeg > 0 && pixelScaleDeg < 1) {  // 合理范围检查
                    scales.push_back(pixelScaleDeg * 3600.0);  // 转换为角秒
                    sampled++;
                }
            } else {
                // 尝试读取 CDELT
                status = 0;
                if (fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, comment, &status) == 0 &&
                    fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, comment, &status) == 0) {
                    
                    double pixelScaleDeg = (std::abs(cdelt1) + std::abs(cdelt2)) / 2.0;  // 度
                    
                    if (pixelScaleDeg > 0 && pixelScaleDeg < 1) {
                        scales.push_back(pixelScaleDeg * 3600.0);  // 转换为角秒
                        sampled++;
                    }
                }
            }
            
            fits_close_file(fptr, &status);
            
        } catch (...) {
            continue;
        }
    }
    
    if (scales.empty()) return -1;
    
    // 返回中位数（更稳健）
    std::sort(scales.begin(), scales.end());
    return scales[scales.size() / 2];
}


/**
 * 从缓存获取或加载 FITS 文件
 * 使用 LRU 缓存策略，自动管理内存
 */
FitsData* getCachedFitsFile(const std::string& filepath, bool verbose = false) {
    // 先检查缓存
    FitsData* cached = g_fitsCache.get(filepath);
    if (cached != nullptr) {
        return cached;
    }
    
    // 缓存未命中，加载文件
    FitsData newData = FitsReader::readFitsFile(filepath);
    if (!newData.isValid) {
        if (verbose) {
            std::cerr << "Warning: Failed to load " << filepath << ": " << newData.errorMessage << std::endl;
        }
        return nullptr;
    }
    
    // 计算内存占用
    size_t memSize = newData.pixels.size() * sizeof(float) + sizeof(FitsData);
    
    // 放入缓存
    g_fitsCache.put(filepath, std::move(newData), memSize);
    
    // 返回缓存中的指针
    return g_fitsCache.get(filepath);
}


bool parseArguments(int argc, char* argv[], Config& config) {
    if (argc < 3) {
        std::cout << "Usage: hipsgen_cuda <input_dir> <output_dir> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -order <n>      : Max HEALPix order (default: auto based on pixel scale)" << std::endl;
        std::cout << "  -threads <n>    : Max threads (default: 4)" << std::endl;
        std::cout << "  -bitpix <n>     : BITPIX value (default: -32, float)" << std::endl;
        std::cout << "  -frame <name>   : Coordinate system (ICRS/GAL, default: ICRS)" << std::endl;
        std::cout << "  -img <file>     : Reference image for default initializations" << std::endl;
        std::cout << "  -blank <value>  : Blank pixel value (default: NaN)" << std::endl;
        std::cout << "  -mode <mode>    : Overlay mode (NONE/MEAN/AVERAGE/FADING/ADD, default: MEAN)" << std::endl;
        std::cout << "  -validRange <min> <max> : Valid pixel range (filter out-of-range pixels)" << std::endl;
        std::cout << "  -autoValidRange : Auto-estimate validRange by sampling source data" << std::endl;
        std::cout << "  -sampleRatio <r>: Sample ratio for auto estimation (default: 0.1 = 10%)" << std::endl;
        std::cout << "  -no-index       : Skip INDEX stage" << std::endl;
        std::cout << "  -no-tiles       : Skip TILES stage" << std::endl;
        std::cout << "  -v              : Verbose output" << std::endl;
        std::cout << "  -cpu            : Force CPU mode" << std::endl;
        std::cout << "  -recursive, -r  : Recursively scan subdirectories" << std::endl;
        std::cout << "  -limit <n>      : Limit number of files to process (for testing)" << std::endl;
        std::cout << "  -cache <n>      : Max cached FITS files in memory (default: 100)" << std::endl;
        std::cout << "  -progress <n>   : Progress report interval in files (default: 100)" << std::endl;
        std::cout << "  -pattern <glob> : Filter files by pattern (e.g., \*-image-r.fits.fz)" << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  hipsgen_cuda data/subset hips_output -order 3 -img data/subset/ref.img -blank 0 -mode AVERAGE" << std::endl;
        return false;
    }
    
    config.inputDir = argv[1];
    config.outputDir = argv[2];
    
    // Parse options
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-order" && i + 1 < argc) {
            config.orderMax = std::stoi(argv[++i]);
            config.autoOrder = false;  // 用户手动指定
        } else if (arg == "-threads" && i + 1 < argc) {
            config.maxThreads = std::stoi(argv[++i]);
        } else if (arg == "-bitpix" && i + 1 < argc) {
            config.bitpix = std::stoi(argv[++i]);
        } else if (arg == "-frame" && i + 1 < argc) {
            config.frame = argv[++i];
        } else if (arg == "-img" && i + 1 < argc) {
            config.imgEtalon = argv[++i];
        } else if (arg == "-blank" && i + 1 < argc) {
            config.blank = std::stod(argv[++i]);
            config.hasUserBlank = true;
        } else if (arg == "-validRange" && i + 2 < argc) {
            config.validMin = std::stod(argv[++i]);
            config.validMax = std::stod(argv[++i]);
            config.hasValidRange = true;
        } else if (arg == "-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            // Convert to uppercase for comparison
            std::transform(mode.begin(), mode.end(), mode.begin(), ::toupper);
            if (mode == "NONE") {
                config.overlayMode = OverlayMode::NONE;
            } else if (mode == "MEAN" || mode == "AVERAGE") {
                config.overlayMode = OverlayMode::MEAN;
            } else if (mode == "FADING") {
                config.overlayMode = OverlayMode::FADING;
            } else if (mode == "ADD") {
                config.overlayMode = OverlayMode::ADD;
            } else {
                std::cerr << "Warning: Unknown overlay mode '" << mode << "', using MEAN" << std::endl;
                config.overlayMode = OverlayMode::MEAN;
            }
        } else if (arg == "-autoValidRange") {
            config.autoValidRange = true;
        } else if (arg == "-sampleRatio" && i + 1 < argc) {
            config.autoSampleRatio = std::stod(argv[++i]);
        } else if (arg == "-no-index") {
            config.enableIndex = false;
        } else if (arg == "-no-tiles") {
            config.enableTiles = false;
        } else if (arg == "-v") {
            config.verbose = true;
        } else if (arg == "-cpu") {
            config.forceCPU = true;
        } else if (arg == "-recursive" || arg == "-r") {
            config.recursiveScan = true;
        } else if (arg == "-limit" && i + 1 < argc) {
            config.maxFiles = std::stoi(argv[++i]);
        } else if (arg == "-cache" && i + 1 < argc) {
            config.cacheSize = std::stoi(argv[++i]);
        } else if (arg == "-no-skip-errors") {
            config.skipErrors = false;
        } else if (arg == "-progress" && i + 1 < argc) {
            config.progressInterval = std::stoi(argv[++i]);
        } else if (arg == "-pattern" && i + 1 < argc) {
            config.filePattern = argv[++i];
        }
    }
    
    return true;
}

/**
 * Initialize config from reference image (like Java imgEtalon)
 * Reads FITS header to get bitpix, bzero, bscale, blank values
 */
bool initFromImgEtalon(Config& config) {
    if (config.imgEtalon.empty()) {
        return true;  // No reference image, use defaults
    }
    
    std::cout << "Reading reference image: " << config.imgEtalon << std::endl;
    
    // Read FITS file header
    FitsData fitsData = FitsReader::readFitsFile(config.imgEtalon);
    if (!fitsData.isValid) {
        std::cerr << "WARNING: Failed to read reference image: " << fitsData.errorMessage << std::endl;
        return false;
    }
    
    // Set BITPIX from reference image
    config.bitpix = fitsData.bitpix;
    std::cout << "  BITPIX from reference: " << config.bitpix << std::endl;
    
    // Set BZERO and BSCALE
    config.bzero = fitsData.bzero;
    config.bscale = fitsData.bscale;
    std::cout << "  BZERO: " << config.bzero << ", BSCALE: " << config.bscale << std::endl;
    
    // Set BLANK from reference image if user didn't explicitly set it
    if (!config.hasUserBlank && !std::isnan(fitsData.blank)) {
        config.blank = fitsData.blank;
        std::cout << "  BLANK from reference: " << config.blank << std::endl;
    }
    
    std::cout << "Reference image initialized successfully" << std::endl;
    return true;
}


/**
 * 创建输出目录结构
 */
void createOutputDirectories(const Config& config) {
    fs::create_directories(config.outputDir);
    
    // 创建Norder目录
    for (int order = 0; order <= config.orderMax; order++) {
        // 估算需要的Dir数量
        long nside = HealpixUtil::norderToNside(order);
        long totalPixels = HealpixUtil::getTotalPixelsFromNside(nside);
        int maxDir = (int)(totalPixels / 10000) + 1;
        maxDir = (maxDir < 1000) ? maxDir : 1000;  // Limit to max 1000 Dir
        
        for (int dir = 0; dir < maxDir; dir++) {
            std::ostringstream oss;
            oss << config.outputDir << "/Norder" << order << "/Dir" << dir;
            fs::create_directories(oss.str());
        }
    }
    
    // 创建HpxFinder目录（INDEX阶段需要）
    if (config.enableIndex) {
        std::string hpxFinderPath = config.outputDir + "/HpxFinder";
        for (int order = 0; order <= config.orderMax; order++) {
            long nside = HealpixUtil::norderToNside(order);
            long totalPixels = HealpixUtil::getTotalPixelsFromNside(nside);
            int maxDir = (int)(totalPixels / 100000) + 1;
            maxDir = (maxDir < 1000) ? maxDir : 1000;
            
            for (int dir = 0; dir < maxDir; dir++) {
                std::ostringstream oss;
                oss << hpxFinderPath << "/Norder" << order << "/Dir" << dir;
                fs::create_directories(oss.str());
            }
        }
    }
}

/**
 * INDEX阶段：生成HpxFinder索引
 */
bool runIndexStage(const Config& config) {
    std::cout << "\n=== INDEX阶段：生成HpxFinder索引 ===" << std::endl;
    
    std::string hpxFinderPath = config.outputDir + "/HpxFinder";
    
    std::cout << "输入目录: " << config.inputDir << std::endl;
    std::cout << "HpxFinder路径: " << hpxFinderPath << std::endl;
    std::cout << "最大order: " << config.orderMax << std::endl;
    
    // 生成索引
    bool success = HpxFinder::generateIndex(
        config.inputDir,
        config.outputDir,
        config.orderMax
    );
    
    if (success) {
        std::cout << "INDEX阶段完成" << std::endl;
    } else {
        std::cerr << "ERROR: INDEX阶段失败" << std::endl;
    }
    
    return success;
}

/**
 * 工作线程函数（处理Tile生成任务）
 */
void workerThread(const Config& config, TileTaskQueue& taskQueue) {
    std::string hpxFinderPath = config.outputDir + "/HpxFinder";
    
    TileTask task;
    while (taskQueue.pop(task)) {
        // 查询HpxFinder获取源文件列表
        std::vector<SourceFileInfo> sourceFiles = HpxFinder::queryIndex(hpxFinderPath, task.order, task.npix);
        
        if (sourceFiles.empty()) {
            g_stats.skippedTiles++;
            continue;
        }
        
        // 生成tile
        auto tile = HipsTileGenerator::generateTile(
            config.inputDir,
            hpxFinderPath,
            task.order, task.npix,
            config.tileWidth,
            config.bitpix,
            config.blank
        );
        
        if (!tile) {
            g_stats.skippedTiles++;
            continue;
        }
        
        // 计算tile文件路径
        int dir = HealpixUtil::getDirNumber(task.npix);
        std::ostringstream oss;
        oss << config.outputDir << "/Norder" << task.order << "/Dir" << dir << "/Npix" << task.npix << ".fits";
        std::string tilePath = oss.str();
        
        // 写入FITS文件
        bool success = FitsWriter::writeTileFile(
            tilePath,
            *tile,
            task.order,
            task.npix,
            config.frame
        );
        
        if (success) {
            g_stats.generatedTiles++;
            if (config.verbose && g_stats.generatedTiles % 100 == 0) {
                std::cout << "生成tile: " << tilePath << " (已生成: " << g_stats.generatedTiles << ")" << std::endl;
            }
        } else {
            g_stats.failedTiles++;
            std::cerr << "ERROR: 写入tile失败: " << tilePath << std::endl;
        }
    }
}

/**
 * TILES stage with TRUE GPU acceleration
 * Process ALL tiles in a SINGLE GPU kernel call for maximum efficiency
 */
bool runTilesStageGPU(const Config& config) {
    std::cout << "\n=== TILES Stage (TRUE GPU Batch Processing) ===" << std::endl;
    
    auto totalStartTime = std::chrono::steady_clock::now();
    
    std::string hpxFinderPath = config.outputDir + "/HpxFinder";
    g_stats.reset();
    
    // Step 1: Load all source FITS files
    std::cout << "\nStep 1: Loading source FITS files..." << std::endl;
    std::vector<FitsData> allFitsFiles;
    std::vector<CoordinateTransform> allTransforms;
    
    // Use new scanFitsFiles function
    std::vector<std::string> filePaths = scanFitsFiles(config);
    
    int loadedCount = 0;
    for (const auto& filePath : filePaths) {
        FitsData fits = FitsReader::readFitsFile(filePath);
        if (fits.isValid) {
            allFitsFiles.push_back(fits);
            allTransforms.emplace_back(fits);
            loadedCount++;
            if (config.progressInterval > 0 && loadedCount % config.progressInterval == 0) {
                std::cout << "  Loaded " << loadedCount << " files..." << std::endl;
            }
        } else if (config.skipErrors) {
            if (config.verbose) {
                std::cerr << "Skipping invalid file: " << filePath << std::endl;
            }
        } else {
            std::cerr << "ERROR: Failed to read " << filePath << std::endl;
            return false;
        }
    }
    
    std::cout << "Loaded " << allFitsFiles.size() << " source images" << std::endl;
    if (config.verbose) {
        g_fitsCache.printStatus();
    }
    
    if (allFitsFiles.empty()) {
        std::cerr << "ERROR: No valid FITS files found" << std::endl;
        return false;
    }
    
    // Step 2: Initialize GPU Batch Processor
    std::cout << "\nStep 2: Initializing GPU Batch Processor..." << std::endl;
    GPUBatchProcessor gpuProcessor;
    
    if (!gpuProcessor.initialize(allFitsFiles, allTransforms)) {
        std::cerr << "GPU initialization failed, falling back to CPU" << std::endl;
        return false;
    }
    
    // Step 3: Collect ALL valid tiles
    // Step 3: Collect valid tiles by scanning index directory (OPTIMIZED)
    std::cout << "\nStep 3: Collecting valid tiles (optimized scan)..." << std::endl;
    std::vector<TileInfo> validTiles;
    
    int tileWidth = config.tileWidth;
    int pixelsPerTile = tileWidth * tileWidth;
    
    // Calculate total possible tiles for statistics
    for (int order = 0; order <= config.orderMax; order++) {
        long nside = HealpixUtil::norderToNside(order);
        g_stats.totalTiles += HealpixUtil::getTotalPixelsFromNside(nside);
    }
    
    // Scan HpxFinder directory to find existing index files
    for (int order = 0; order <= config.orderMax; order++) {
        std::string orderPath = hpxFinderPath + "/Norder" + std::to_string(order);
        
        if (!fs::exists(orderPath)) continue;
        
        for (const auto& dirEntry : fs::directory_iterator(orderPath)) {
            if (!dirEntry.is_directory()) continue;
            
            // Iterate index files in Dir*/
            for (const auto& fileEntry : fs::directory_iterator(dirEntry.path())) {
                if (!fileEntry.is_regular_file()) continue;
                
                std::string filename = fileEntry.path().filename().string();
                // Parse Npix from filename (format: NpixXXXXX)
                if (filename.substr(0, 4) != "Npix") continue;
                
                long npix = std::stol(filename.substr(4));
                
                TileInfo info;
                info.order = order;
                info.npix = npix;
                info.startIdx = (int)validTiles.size() * pixelsPerTile;
                validTiles.push_back(info);
            }
        }
    }
    
    g_stats.skippedTiles = g_stats.totalTiles - validTiles.size();
    
    std::cout << "Found " << validTiles.size() << " tiles with data" << std::endl;
    
    if (validTiles.empty()) {
        std::cout << "No tiles to process" << std::endl;
        return true;
    }
    
    // Step 4: Create xy2hpx mapping
    int tileOrder = 0;
    int temp = tileWidth;
    while (temp > 1) { temp >>= 1; tileOrder++; }
    
    std::vector<int> xy2hpx, hpx2xy;
    HipsTileGenerator::createXY2HPXMapping(tileOrder, xy2hpx, hpx2xy);
    
    // Step 5: Process ALL tiles in a single GPU batch
    std::cout << "\nStep 4: Processing ALL tiles in GPU batch..." << std::endl;
    std::vector<std::vector<double>> allResults;
    
    if (!gpuProcessor.processTilesBatch(validTiles, tileWidth, xy2hpx, config.blank, allResults)) {
        std::cerr << "GPU batch processing failed" << std::endl;
        return false;
    }
    
    // Step 6: Write tile files
    std::cout << "\nStep 5: Writing tile files..." << std::endl;
    auto writeStartTime = std::chrono::steady_clock::now();
    
    for (size_t t = 0; t < validTiles.size(); t++) {
        const TileInfo& info = validTiles[t];
        const std::vector<double>& tileResults = allResults[t];
        
        // Check if tile has valid data
        bool hasData = false;
        for (int i = 0; i < pixelsPerTile; i++) {
            if (tileResults[i] != config.blank && !std::isnan(tileResults[i])) {
                hasData = true;
                break;
            }
        }
        
        if (!hasData) {
            g_stats.skippedTiles++;
            continue;
        }
        
        // Create tile data
        TileData tile(tileWidth, tileWidth, config.bitpix);
        // For float output (bitpix<0), output blank should be NaN (match Java)
        tile.blank = (config.bitpix < 0) ? std::numeric_limits<double>::quiet_NaN() : config.blank;
        for (int y = 0; y < tileWidth; y++) {
            for (int x = 0; x < tileWidth; x++) {
                tile.setPixel(x, y, tileResults[y * tileWidth + x]);
            }
        }
        // Write file
        int dir = HealpixUtil::getDirNumber(info.npix);
        std::ostringstream oss;
        oss << config.outputDir << "/Norder" << info.order << "/Dir" << dir << "/Npix" << info.npix << ".fits";
        
        if (FitsWriter::writeTileFile(oss.str(), tile, info.order, info.npix, config.frame)) {
            g_stats.generatedTiles++;
        } else {
            g_stats.failedTiles++;
        }
    }
    
    auto writeEndTime = std::chrono::steady_clock::now();
    auto writeMs = std::chrono::duration_cast<std::chrono::milliseconds>(writeEndTime - writeStartTime).count();
    std::cout << "  File write time: " << writeMs << " ms" << std::endl;
    
    gpuProcessor.release();
    
    auto totalEndTime = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime).count();
    std::cout << "\n=== GPU TILES Stage Complete ==="  << std::endl;
    std::cout << "Total GPU stage time: " << totalMs << " ms" << std::endl;
    
    g_stats.print();
    return true;
}

/**
 * TILES stage with FULL GPU acceleration
 * ALL computation (HEALPix -> Celestial -> WCS -> Interpolation) on GPU
 */
bool runTilesStageFullGPU(Config& config) {
    std::cout << "\n=== TILES Stage (FULL GPU Processing) ===" << std::endl;
    
    auto totalStartTime = std::chrono::steady_clock::now();
    
    std::string hpxFinderPath = config.outputDir + "/HpxFinder";
    g_stats.reset();
    
    // Step 1: Load all source FITS files
    std::cout << "\nStep 1: Loading source FITS files..." << std::endl;
    std::vector<FitsData> allFitsFiles;
    
    
    
    // Use new scanFitsFiles function
    std::vector<std::string> filePaths = scanFitsFiles(config);
    
    int loadedCount = 0;
    for (const auto& filePath : filePaths) {
        // 使用缓存加载
        FitsData* fitsPtr = getCachedFitsFile(filePath, config.verbose);
        if (fitsPtr != nullptr && fitsPtr->isValid) {
            allFitsFiles.push_back(*fitsPtr);
            loadedCount++;
            if (config.progressInterval > 0 && loadedCount % config.progressInterval == 0) {
                std::cout << "  Loaded " << loadedCount << " files..." << std::endl;
            }
        } else if (config.skipErrors) {
            if (config.verbose) {
                std::cerr << "Skipping invalid file: " << filePath << std::endl;
            }
        }
    }
    
    std::cout << "Loaded " << allFitsFiles.size() << " source images" << std::endl;
    if (config.verbose) {
        g_fitsCache.printStatus();
    }
    
    // If user specified blank value, apply to all source images
    if (config.hasUserBlank) {
        for (auto& fits : allFitsFiles) {
            fits.blank = config.blank;
        }
        std::cout << "Applied user blank value " << config.blank << " to all source images" << std::endl;
    }
    
    
    // 自动估计validRange（如果用户请求）
    if (config.autoValidRange && !config.hasValidRange) {
        std::cout << "\n=== Auto-estimating validRange ===" << std::endl;
        
        // 收集所有FITS文件路径
        std::vector<std::string> fitsFilePaths;
        for (const auto& fits : allFitsFiles) {
            fitsFilePaths.push_back(fits.filename);
        }
        
        ValidRangeEstimate estimate = estimateValidRange(
            fitsFilePaths, 
            config.blank, 
            config.autoSampleRatio,
            true  // verbose
        );
        
        if (estimate.success) {
            // 稍微扩展范围以避免边界问题（可选）
            // double margin = (estimate.maxVal - estimate.minVal) * 0.01;
            config.validMin = estimate.minVal;
            config.validMax = estimate.maxVal;
            config.hasValidRange = true;
            
            std::cout << "  Estimated validRange: [" << config.validMin 
                      << ", " << config.validMax << "]" << std::endl;
        } else {
            std::cout << "  Warning: Could not estimate validRange, proceeding without filtering" << std::endl;
        }
    }

    // Apply validRange preprocessing - set pixels outside range to NaN (match Java CacheFits)
    if (config.hasValidRange) {
        int totalFiltered = 0;
        for (auto& fits : allFitsFiles) {
            int filtered = 0;
            for (size_t i = 0; i < fits.pixels.size(); i++) {
                float val = fits.pixels[i];
                if (!std::isnan(val) && (val < config.validMin || val > config.validMax)) {
                    fits.pixels[i] = std::numeric_limits<float>::quiet_NaN();
                    filtered++;
                }
            }
            totalFiltered += filtered;
        }
        std::cout << "Applied validRange [" << config.validMin << ", " << config.validMax 
                  << "] - filtered " << totalFiltered << " pixels to NaN" << std::endl;
    }
    
    // Create filename to index mapping for HpxFinder lookup
    std::map<std::string, int> filenameToIndex;
    for (size_t i = 0; i < allFitsFiles.size(); i++) {
        // Extract just the filename from the full path
        std::string filename = allFitsFiles[i].filename;
        size_t pos = filename.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        filenameToIndex[filename] = (int)i;
    }
    
    // Debug: Print WCS parameters for first few images
    if (config.verbose) {
        std::cout << "\nWCS parameters for first 3 images:" << std::endl;
        size_t maxDebug = allFitsFiles.size() < 3 ? allFitsFiles.size() : 3;
        for (size_t i = 0; i < maxDebug; i++) {
            const auto& f = allFitsFiles[i];
            std::cout << "  Image " << i << " (" << f.filename << "): CRVAL=(" << f.crval1 << ", " << f.crval2 << ")"
                      << ", CD=[[" << f.cd1_1 << ", " << f.cd1_2 << "], ["
                      << f.cd2_1 << ", " << f.cd2_2 << "]]" << std::endl;
        }
    }
    
    if (allFitsFiles.empty()) {
        std::cerr << "ERROR: No valid FITS files found" << std::endl;
        return false;
    }
    
    // Step 2: Initialize Full GPU Processor
    std::cout << "\nStep 2: Initializing Full GPU Processor..." << std::endl;
    GPUFullProcessor gpuProcessor;
    
    if (!gpuProcessor.initialize(allFitsFiles)) {
        std::cerr << "Full GPU initialization failed" << std::endl;
        return false;
    }
    
    // Step 3: Collect valid tiles by scanning index directory (OPTIMIZED)
    // Instead of iterating 4M+ tiles, directly scan existing index files
    std::cout << "\nStep 3: Collecting valid tiles (optimized scan)..." << std::endl;
    std::vector<GPUTileInfo> validTiles;
    
    int tileWidth = config.tileWidth;
    int pixelsPerTile = tileWidth * tileWidth;
    
    // Calculate total possible tiles for statistics
    for (int order = 0; order <= config.orderMax; order++) {
        long nside = HealpixUtil::norderToNside(order);
        g_stats.totalTiles += HealpixUtil::getTotalPixelsFromNside(nside);
    }
    
    // Scan HpxFinder directory to find existing index files
    for (int order = 0; order <= config.orderMax; order++) {
        std::string orderPath = hpxFinderPath + "/Norder" + std::to_string(order);
        
        if (!fs::exists(orderPath)) continue;
        
        for (const auto& dirEntry : fs::directory_iterator(orderPath)) {
            if (!dirEntry.is_directory()) continue;
            
            // Iterate index files in Dir*/
            for (const auto& fileEntry : fs::directory_iterator(dirEntry.path())) {
                if (!fileEntry.is_regular_file()) continue;
                
                std::string filename = fileEntry.path().filename().string();
                // Parse Npix from filename (format: NpixXXXXX)
                if (filename.substr(0, 4) != "Npix") continue;
                
                long npix = std::stol(filename.substr(4));
                
                std::vector<SourceFileInfo> sourceFiles = HpxFinder::readIndexFile(fileEntry.path().string());
                
                if (!sourceFiles.empty()) {
                    GPUTileInfo info;
                    info.order = order;
                    info.npix = npix;
                    
                    for (const auto& sf : sourceFiles) {
                        auto it = filenameToIndex.find(sf.filename);
                        if (it != filenameToIndex.end()) {
                            info.imageIndices.push_back(it->second);
                        }
                    }
                    
                    if (!info.imageIndices.empty()) {
                        validTiles.push_back(info);
                    }
                }
            }
        }
    }
    
    g_stats.skippedTiles = g_stats.totalTiles - validTiles.size();
    
    std::cout << "Found " << validTiles.size() << " tiles with data" << std::endl;
    
    if (validTiles.empty()) {
        std::cout << "No tiles to process" << std::endl;
        return true;
    }
    
    // Step 4: Create xy2hpx mapping
    int tileOrder = 0;
    int temp = tileWidth;
    while (temp > 1) { temp >>= 1; tileOrder++; }
    
    std::vector<int> xy2hpx, hpx2xy;
    HipsTileGenerator::createXY2HPXMapping(tileOrder, xy2hpx, hpx2xy);
    
    // Step 5: Process ALL tiles fully on GPU
    std::cout << "\nStep 4: Processing ALL tiles fully on GPU..." << std::endl;
    std::vector<std::vector<double>> allResults;
    
    // For float output (bitpix<0), output blank should be NaN (match Java)
    double outputBlank = (config.bitpix < 0) ? std::numeric_limits<double>::quiet_NaN() : config.blank;
    
    if (!gpuProcessor.processAllTiles(validTiles, tileWidth, xy2hpx, outputBlank, allResults, config.validMin, config.validMax)) {
        std::cerr << "Full GPU processing failed" << std::endl;
        return false;
    }
    
    // Step 6: Write tile files
    std::cout << "\nStep 5: Writing tile files (parallel)..." << std::endl;
    auto writeStartTime = std::chrono::steady_clock::now();
    
    // Use atomic counters for thread safety
    std::atomic<int> generatedCount(0);
    std::atomic<int> failedCount(0);
    std::atomic<int> progressCount(0);
    int totalTiles = (int)validTiles.size();
    
    // Parallel write using OpenMP
    #pragma omp parallel for schedule(dynamic, 100)
    for (size_t t = 0; t < validTiles.size(); t++) {
        const GPUTileInfo& info = validTiles[t];
        const std::vector<double>& tileResults = allResults[t];
        
        // Create tile data (thread-local)
        TileData tile(tileWidth, tileWidth, config.bitpix);
        tile.blank = (config.bitpix < 0) ? std::numeric_limits<double>::quiet_NaN() : config.blank;
        for (int y = 0; y < tileWidth; y++) {
            for (int x = 0; x < tileWidth; x++) {
                tile.setPixel(x, y, tileResults[y * tileWidth + x]);
            }
        }
        
        // Write file
        int dir = HealpixUtil::getDirNumber(info.npix);
        std::ostringstream oss;
        oss << config.outputDir << "/Norder" << info.order << "/Dir" << dir << "/Npix" << info.npix << ".fits";
        
        if (FitsWriter::writeTileFile(oss.str(), tile, info.order, info.npix, config.frame)) {
            generatedCount++;
        } else {
            failedCount++;
        }
        
        // Progress report (every 1000 tiles)
        int prog = ++progressCount;
        if (prog % 1000 == 0) {
            #pragma omp critical
            {
                std::cout << "  Written " << prog << "/" << totalTiles << " tiles..." << std::endl;
            }
        }
    }
    
    g_stats.generatedTiles = generatedCount;
    g_stats.failedTiles = failedCount;
    
    auto writeEndTime = std::chrono::steady_clock::now();
    auto writeMs = std::chrono::duration_cast<std::chrono::milliseconds>(writeEndTime - writeStartTime).count();
    std::cout << "  File write time: " << writeMs << " ms" << std::endl;
    
    gpuProcessor.release();
    
    auto totalEndTime = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime).count();
    std::cout << "\n=== Full GPU TILES Stage Complete ==="  << std::endl;
    std::cout << "Total Full GPU stage time: " << totalMs << " ms" << std::endl;
    
    g_stats.print();
    return true;
}

/**
 * TILES stage (multi-threaded CPU version, fallback)
 */
bool runTilesStage(const Config& config) {
    std::cout << "\n=== TILES Stage (Multi-threaded) ===" << std::endl;
    
    std::string hpxFinderPath = config.outputDir + "/HpxFinder";
    
    std::cout << "HpxFinder: " << hpxFinderPath << std::endl;
    std::cout << "Output: " << config.outputDir << std::endl;
    std::cout << "Tile size: " << config.tileWidth << "x" << config.tileWidth << std::endl;
    std::cout << "BITPIX: " << config.bitpix << std::endl;
    std::cout << "Threads: " << config.maxThreads << std::endl;
    
    g_stats.reset();
    
    TileTaskQueue taskQueue;
    
    for (int order = 0; order <= config.orderMax; order++) {
        std::cout << "\nProcessing Order " << order << "..." << std::endl;
        
        long nside = HealpixUtil::norderToNside(order);
        long totalPixels = HealpixUtil::getTotalPixelsFromNside(nside);
        
        std::cout << "Nside: " << nside << ", Total tiles: " << totalPixels << std::endl;
        
        g_stats.totalTiles += totalPixels;
        
        for (long npix = 0; npix < totalPixels; npix++) {
            taskQueue.push(TileTask(order, npix));
        }
        
        std::cout << "Added " << totalPixels << " tasks to queue" << std::endl;
    }
    
    std::cout << "\nStarting " << config.maxThreads << " worker threads..." << std::endl;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < config.maxThreads; t++) {
        threads.emplace_back(workerThread, std::cref(config), std::ref(taskQueue));
    }
    
    std::cout << "Waiting for tasks to complete..." << std::endl;
    
    taskQueue.setDone();
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "All threads completed" << std::endl;
    
    g_stats.print();
    return true;
}

/**
 * 主函数
 */
int main(int argc, char* argv[]) {
    std::cout << "=== HipsGen CUDA - GPU Accelerated Version ===" << std::endl;
    std::cout << "Direct CUDA implementation, bypassing JNI" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    Config config;
    if (!parseArguments(argc, argv, config)) {
        return 1;
    }
    
    // Initialize from reference image if specified
    if (!config.imgEtalon.empty()) {
        if (!initFromImgEtalon(config)) {
            std::cerr << "WARNING: Failed to initialize from reference image" << std::endl;
        }
    }
    
    
    // ====== 系统资源检测和自动参数调整 ======
    SystemResources sysRes = ResourceDetector::detect();
    
    // 显示资源信息（verbose 模式）
    if (config.verbose) {
        sysRes.print();
    }
    
    // 如果用户没有手动指定参数，使用系统推荐值
    bool autoThreads = (config.maxThreads == 4);  // 检查是否是默认值
    bool autoCache = (config.cacheSize == 100);   // 检查是否是默认值
    
    if (autoThreads) {
        config.maxThreads = sysRes.recommendedThreads;
    }
    if (autoCache) {
        config.cacheSize = sysRes.recommendedCacheSize;
    }
    
    // 存储推荐的内存限制供后续使用
    size_t recommendedCacheMemory = sysRes.recommendedCacheMemory;
    
    // 配置全局 FITS 缓存
    g_fitsCache.setLimits(config.cacheSize, recommendedCacheMemory);
    
    // 早期扫描文件以用于 auto order 计算
    std::vector<std::string> filePaths;
    bool orderWasAuto = config.autoOrder;
    
    if (config.autoOrder) {
        std::cout << "Scanning files for auto order calculation..." << std::endl;
        filePaths = scanFitsFiles(config);
        std::cout << "Found " << filePaths.size() << " FITS files" << std::endl;
        
        if (!filePaths.empty()) {
            double pixelScale = estimatePixelScale(filePaths, 20);
            if (pixelScale > 0) {
                int recommendedOrder = calculateRecommendedOrder(pixelScale, config.tileWidth);
                config.orderMax = recommendedOrder;
                std::cout << "\n=== Auto Order Calculation ===" << std::endl;
                std::cout << "  Pixel scale: " << std::fixed << std::setprecision(3) << pixelScale << " arcsec" << std::endl;
                std::cout << "  Recommended order: " << recommendedOrder << " (auto)" << std::endl;
                std::cout << "  Tile resolution at order " << recommendedOrder << ": " 
                          << std::fixed << std::setprecision(3)
                          << (411526.0 / (std::pow(2, recommendedOrder) * config.tileWidth)) << " arcsec" << std::endl;
                std::cout << std::endl;
            } else {
                std::cout << "Warning: Could not determine pixel scale, using default order " << config.orderMax << std::endl;
                orderWasAuto = false;
            }
        } else {
            std::cout << "Warning: No files found, using default order " << config.orderMax << std::endl;
            orderWasAuto = false;
        }
    }
    
    std::cout << "System Resources:" << std::endl;
    std::cout << "  CPU Cores: " << sysRes.cpuCores << std::endl;
    std::cout << "  Available Memory: " << (sysRes.availableMemory / (1024*1024*1024)) << " GB" << std::endl;
    if (sysRes.gpuCount > 0) {
        std::cout << "  GPU: " << sysRes.gpuName << " (" 
                  << (sysRes.gpuAvailableMemory / (1024*1024*1024)) << " GB available)" << std::endl;
    }
    std::cout << "  Using threads: " << config.maxThreads << (autoThreads ? " (auto)" : " (user)") << std::endl;
    std::cout << "  Cache size: " << config.cacheSize << " files" << (autoCache ? " (auto)" : " (user)") << std::endl;
    std::cout << "  Cache memory limit: " << (recommendedCacheMemory / (1024*1024)) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Display configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input directory: " << config.inputDir << std::endl;
    std::cout << "  Output directory: " << config.outputDir << std::endl;
    std::cout << "  HiPS ID: " << config.hipsId << std::endl;
    std::cout << "  Max order: " << config.orderMax << (orderWasAuto ? " (auto)" : " (user)") << std::endl;
    std::cout << "  Max threads: " << config.maxThreads << std::endl;
    std::cout << "  Tile size: " << config.tileWidth << "x" << config.tileWidth << std::endl;
    std::cout << "  BITPIX: " << config.bitpix << std::endl;
    std::cout << "  Coordinate system: " << config.frame << std::endl;
    if (!config.imgEtalon.empty()) {
        std::cout << "  Reference image: " << config.imgEtalon << std::endl;
    }
    if (config.hasUserBlank) {
        std::cout << "  BLANK value: " << config.blank << " (user specified)" << std::endl;
    } else if (!std::isnan(config.blank)) {
        std::cout << "  BLANK value: " << config.blank << " (from reference)" << std::endl;
    } else {
        std::cout << "  BLANK value: NaN (default)" << std::endl;
    }
    std::cout << "  Overlay mode: " << getOverlayModeName(config.overlayMode) << std::endl;
    if (config.hasValidRange) {
        std::cout << "  Valid range: [" << config.validMin << ", " << config.validMax << "]" << std::endl;
    }
    std::cout << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "WARNING: CUDA initialization failed!" << std::endl;
        std::cerr << "  Will use CPU mode" << std::endl;
    } else {
        std::cout << "CUDA initialization successful" << std::endl;
    }
    
    // Create output directories
    createOutputDirectories(config);
    
    // Execute INDEX stage
    if (config.enableIndex) {
        if (!runIndexStage(config)) {
            std::cerr << "ERROR: INDEX stage failed, terminating execution" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Skipping INDEX stage" << std::endl;
    }
    
    // Execute TILES stage
    if (config.enableTiles) {
        bool success = false;
        if (config.forceCPU) {
            std::cout << "Forced CPU mode" << std::endl;
            success = runTilesStage(config);
        } else {
            // Try FULL GPU processing first (all computation on GPU)
            success = runTilesStageFullGPU(config);
            if (!success) {
                std::cerr << "ERROR: Full GPU processing failed, trying GPU batch mode..." << std::endl;
                success = runTilesStageGPU(config);
                if (!success) {
                    std::cerr << "ERROR: GPU batch processing also failed!" << std::endl;
                    std::cerr << "Please check GPU memory and reduce batch size or order." << std::endl;
                    // No CPU fallback - fail fast
                }
            }
        }
        if (!success) {
            std::cerr << "ERROR: TILES stage failed" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Skipping TILES stage" << std::endl;
    }
    
    // Generate Allsky preview
    std::cout << "\n=== Generating Allsky preview ===" << std::endl;
    if (!AllskyGenerator::generateAllskyFits(config.outputDir, 3, 64)) {
        std::cerr << "WARNING: Allsky generation failed (not critical)" << std::endl;
    }
    
    // Generate properties metadata file
    std::cout << "\n=== Generating properties metadata ===" << std::endl;
    // Extract input directory name as title
    std::string title = fs::path(config.inputDir).filename().string();
    if (title.empty()) {
        title = "HiPS";
    }
    
    // Use validRange if set, otherwise use defaults
    double pixelCutMin = config.hasValidRange ? config.validMin : -1.0;
    double pixelCutMax = config.hasValidRange ? config.validMax : 1.0;
    
    PropertiesGenerator::generateProperties(
        config.outputDir,
        title,
        config.orderMax,
        config.tileWidth,
        config.bitpix,
        (int)g_stats.generatedTiles,
        pixelCutMin,
        pixelCutMax
    );
    std::cout << "Properties file generated" << std::endl;
    
    // TREE stage disabled - directly generate tiles from source images at all orders
    // (can be enabled later if needed to match Java's hierarchical approach)
    std::cout << "\n=== TREE stage (disabled) ===" << std::endl;
    if (false) {  // DISABLED
    
    // Process from orderMax-1 down to 0
    for (int order = config.orderMax - 1; order >= 0; order--) {
        long nside = HealpixUtil::norderToNside(order);
        long totalPixels = HealpixUtil::getTotalPixelsFromNside(nside);
        int tileWidth = config.tileWidth;
        
        int tilesRebuilt = 0;
        
        for (long npix = 0; npix < totalPixels; npix++) {
            // Build the path for this tile
            int dir = HealpixUtil::getDirNumber(npix);
            std::ostringstream tilePath;
            tilePath << config.outputDir << "/Norder" << order << "/Dir" << dir << "/Npix" << npix << ".fits";
            
            // Check if all 4 child tiles exist
            long childNpix0 = npix * 4;
            bool allChildrenExist = true;
            std::vector<std::string> childPaths(4);
            
            for (int c = 0; c < 4; c++) {
                long childNpix = childNpix0 + c;
                int childDir = HealpixUtil::getDirNumber(childNpix);
                std::ostringstream childPath;
                childPath << config.outputDir << "/Norder" << (order + 1) 
                          << "/Dir" << childDir << "/Npix" << childNpix << ".fits";
                childPaths[c] = childPath.str();
                
                if (!fs::exists(childPaths[c])) {
                    allChildrenExist = false;
                    break;
                }
            }
            
            if (!allChildrenExist) {
                continue;
            }
            
            // Read child tiles and compute average (treeMean)
            std::vector<float> parentPixels(tileWidth * tileWidth, std::numeric_limits<float>::quiet_NaN());
            
            for (int c = 0; c < 4; c++) {
                FitsData childData = FitsReader::readFitsFile(childPaths[c]);
                if (!childData.isValid) continue;
                
                // Each child occupies a quadrant of the parent
                // Quadrant mapping: 0=bottom-left, 1=bottom-right, 2=top-left, 3=top-right
                int offX = (c & 1) ? tileWidth / 2 : 0;
                int offY = (c & 2) ? tileWidth / 2 : 0;
                
                // Average 2x2 pixels from child to 1 pixel in parent
                int halfWidth = tileWidth / 2;
                for (int py = 0; py < halfWidth; py++) {
                    for (int px = 0; px < halfWidth; px++) {
                        double sum = 0;
                        int count = 0;
                        
                        // Average 2x2 block from child
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                int cx = px * 2 + dx;
                                int cy = py * 2 + dy;
                                int childIdx = cy * tileWidth + cx;
                                if (childIdx < (int)childData.pixels.size()) {
                                    float val = childData.pixels[childIdx];
                                    if (!std::isnan(val)) {
                                        sum += val;
                                        count++;
                                    }
                                }
                            }
                        }
                        
                        if (count > 0) {
                            int parentIdx = (offY + py) * tileWidth + (offX + px);
                            parentPixels[parentIdx] = (float)(sum / count);
                        }
                    }
                }
            }
            
            // Write the parent tile
            fs::create_directories(fs::path(tilePath.str()).parent_path());
            FitsWriter::writeFitsFile(tilePath.str(), parentPixels.data(), 
                                      tileWidth, tileWidth, -32);
            tilesRebuilt++;
        }
        
        if (tilesRebuilt > 0) {
            std::cout << "  Order " << order << ": rebuilt " << tilesRebuilt << " tiles from children" << std::endl;
        }
    }
    std::cout << "TREE generation complete" << std::endl;
    }  // END DISABLED TREE
    
    std::cout << "\n=== HiPS Generation Complete! ===" << std::endl;
    std::cout << "Output directory: " << config.outputDir << std::endl;
    
    return 0;
}
