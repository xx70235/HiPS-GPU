/**
 * HpxFinder索引实现
 * 优化版本：GPU加速 + 多线程FITS读取
 */

#include "hpx_finder.h"
#include "index_gpu_processor.h"
#include "fits_header_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <set>
#include <algorithm>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include "coordinate_transform.h"

namespace fs = std::filesystem;

// 全局互斥锁（用于多线程安全）
static std::mutex g_indexMutex;

// 是否启用GPU加速（默认启用）
static bool g_useGPU = true;

/**
 * 生成INDEX阶段的空间索引（GPU加速版本）
 */
bool HpxFinder::generateIndex(const std::string& inputDir, 
                              const std::string& outputDir, 
                              int orderMax) {
    std::string hpxFinderPath = outputDir + "/HpxFinder";
    
    std::cout << "开始生成HpxFinder索引..." << std::endl;
    std::cout << "输入目录: " << inputDir << std::endl;
    std::cout << "输出目录: " << hpxFinderPath << std::endl;
    std::cout << "最大order: " << orderMax << std::endl;
    
    // 创建HpxFinder目录结构
    for (int order = 0; order <= orderMax; order++) {
        long nside = HealpixUtil::norderToNside(order);
        long totalPixels = HealpixUtil::getTotalPixelsFromNside(nside);
        int maxDir = (int)(totalPixels / 100000) + 1;
        maxDir = std::min(maxDir, 1000);
        
        for (int dir = 0; dir < maxDir; dir++) {
            std::ostringstream oss;
            oss << hpxFinderPath << "/Norder" << order << "/Dir" << dir;
            fs::create_directories(oss.str());
        }
    }
    
    // 扫描输入目录中的FITS文件
    std::vector<std::string> fitsFiles;
    if (fs::exists(inputDir) && fs::is_directory(inputDir)) {
        for (const auto& entry : fs::recursive_directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".fits" || ext == ".fit" || ext == ".fts" || ext == ".img" || ext == ".fz") {
                    fitsFiles.push_back(entry.path().string());
                }
            }
        }
    }
    
    std::cout << "找到 " << fitsFiles.size() << " 个FITS文件" << std::endl;
    
    if (fitsFiles.empty()) {
        std::cerr << "ERROR: 没有找到FITS文件" << std::endl;
        return false;
    }
    
    std::map<std::string, std::vector<SourceFileInfo>> tileFileMap;
    
    auto indexStartTime = std::chrono::high_resolution_clock::now();
    
    // =============== GPU加速路径 ===============
    if (g_useGPU) {
        std::cout << "\n使用GPU加速模式..." << std::endl;
        
        auto readStartTime = std::chrono::high_resolution_clock::now();
        
        std::vector<ImageWCSParams> wcsParams(fitsFiles.size());
        std::atomic<int> processedCount(0);
        std::atomic<int> validCount(0);
        
        int numThreads = std::min((int)std::thread::hardware_concurrency(), 16);
        numThreads = std::max(numThreads, 1);
        std::cout << "使用 " << numThreads << " 个线程读取FITS文件头..." << std::endl;
        
        auto readWorker = [&](int threadId) {
            for (size_t i = threadId; i < fitsFiles.size(); i += numThreads) {
                const std::string& fitsPath = fitsFiles[i];
                
                // 使用快速WCS读取器（只读header，不读像素）
                WCSInfo wcs = FastFitsHeaderReader::readWCSInfo(fitsPath);
                
                int count = ++processedCount;
                if (count % 500 == 0) {
                    std::cout << "读取进度: " << count << "/" << fitsFiles.size() << std::endl;
                }
                
                if (!wcs.valid) continue;
                
                std::string relPath = fs::relative(fitsPath, inputDir).string();
                std::replace(relPath.begin(), relPath.end(), '\\', '/');
                
                long cellMem = (long)wcs.width * wcs.height * 4;  // 假设float
                
                wcsParams[i].crval1 = wcs.crval1;
                wcsParams[i].crval2 = wcs.crval2;
                wcsParams[i].crpix1 = wcs.crpix1;
                wcsParams[i].crpix2 = wcs.crpix2;
                wcsParams[i].cd1_1 = wcs.cd1_1;
                wcsParams[i].cd1_2 = wcs.cd1_2;
                wcsParams[i].cd2_1 = wcs.cd2_1;
                wcsParams[i].cd2_2 = wcs.cd2_2;
                wcsParams[i].width = wcs.width;
                wcsParams[i].height = wcs.height;
                wcsParams[i].filepath = relPath;
                wcsParams[i].cellMem = cellMem;
                
                ++validCount;
            }
        };
        
        std::vector<std::thread> threads;
        for (int t = 0; t < numThreads; t++) {
            threads.emplace_back(readWorker, t);
        }
        for (auto& t : threads) {
            t.join();
        }
        
        auto readEndTime = std::chrono::high_resolution_clock::now();
        auto readMs = std::chrono::duration_cast<std::chrono::milliseconds>(readEndTime - readStartTime).count();
        std::cout << "FITS文件头读取完成: " << validCount << "/" << fitsFiles.size() 
                  << " 有效, 耗时 " << readMs << " ms" << std::endl;
        
        std::vector<ImageWCSParams> validParams;
        for (const auto& p : wcsParams) {
            if (p.width > 0 && p.height > 0) {
                validParams.push_back(p);
            }
        }
        
        IndexGPUProcessor gpuProcessor;
        if (gpuProcessor.initialize()) {
            std::map<std::string, std::vector<std::pair<std::string, long>>> gpuTileFileMap;
            
            if (gpuProcessor.computeIndexBatch(validParams, orderMax, gpuTileFileMap)) {
                for (const auto& pair : gpuTileFileMap) {
                    for (const auto& fileInfo : pair.second) {
                        tileFileMap[pair.first].push_back(SourceFileInfo(fileInfo.first, fileInfo.second));
                    }
                }
                std::cout << "GPU索引计算完成" << std::endl;
            } else {
                std::cerr << "GPU索引计算失败，回退到CPU模式" << std::endl;
                g_useGPU = false;
            }
            gpuProcessor.cleanup();
        } else {
            std::cerr << "GPU初始化失败，回退到CPU模式" << std::endl;
            g_useGPU = false;
        }
    }
    
    // =============== CPU回退路径 ===============
    if (!g_useGPU || tileFileMap.empty()) {
        std::cout << "\n使用CPU模式..." << std::endl;
        int processedFiles = 0;
        
        for (const auto& fitsPath : fitsFiles) {
            processedFiles++;
            if (processedFiles % 100 == 0) {
                std::cout << "处理进度: " << processedFiles << "/" << fitsFiles.size() << std::endl;
            }
            
            FitsData fits = FitsReader::readFitsFile(fitsPath);
            if (!fits.isValid) continue;
            
            std::string relPath = fs::relative(fitsPath, inputDir).string();
            std::replace(relPath.begin(), relPath.end(), '\\', '/');
            
            long cellMem = (long)fits.width * fits.height * (abs(fits.bitpix) / 8);
            if (fits.bitpix == 0) cellMem = fits.width * fits.height * 4;
            
            SourceFileInfo srcInfo(relPath, cellMem);
            
            for (int order = 0; order <= orderMax; order++) {
                std::vector<long> coverage = computeCoverage(fits, order);
                
                for (long npix : coverage) {
                    std::ostringstream oss;
                    oss << order << "/" << npix;
                    std::string key = oss.str();
                    tileFileMap[key].push_back(srcInfo);
                }
            }
        }
    }
    
    auto indexEndTime = std::chrono::high_resolution_clock::now();
    auto indexMs = std::chrono::duration_cast<std::chrono::milliseconds>(indexEndTime - indexStartTime).count();
    std::cout << "索引计算总耗时: " << indexMs << " ms" << std::endl;
    
    std::cout << "计算完成，开始写入索引文件..." << std::endl;
    
    int writtenFiles = 0;
    for (const auto& pair : tileFileMap) {
        size_t pos = pair.first.find('/');
        if (pos == std::string::npos) continue;
        
        int order = std::stoi(pair.first.substr(0, pos));
        long npix = std::stol(pair.first.substr(pos + 1));
        
        int dir = HealpixUtil::getDirNumber(npix);
        std::ostringstream oss;
        oss << hpxFinderPath << "/Norder" << order << "/Dir" << dir << "/Npix" << npix;
        std::string indexPath = oss.str();
        
        writeIndexFile(indexPath, pair.second);
        writtenFiles++;
        
        if (writtenFiles % 100 == 0) {
            std::cout << "已写入 " << writtenFiles << " 个索引文件..." << std::endl;
        }
    }
    
    std::cout << "INDEX stage completed!" << std::endl;
    std::cout << "Total written index files: " << writtenFiles << std::endl;
    
    return true;
}

std::vector<SourceFileInfo> HpxFinder::queryIndex(const std::string& hpxFinderPath,
                                                   int order, long npix) {
    std::vector<SourceFileInfo> result;
    
    int dir = HealpixUtil::getDirNumber(npix);
    std::ostringstream oss;
    oss << hpxFinderPath << "/Norder" << order << "/Dir" << dir << "/Npix" << npix;
    std::string indexPath = oss.str();
    
    if (!fs::exists(indexPath)) {
        return result;
    }
    
    return readIndexFile(indexPath);
}

std::vector<long> HpxFinder::computeCoverage(const FitsData& fitsData, int order) {
    std::set<long> npixSet;
    
    if (!fitsData.isValid || fitsData.width <= 0 || fitsData.height <= 0) {
        return std::vector<long>();
    }
    
    if (fitsData.crval1 == 0.0 && fitsData.crval2 == 0.0 && 
        fitsData.cd1_1 == 0.0 && fitsData.cd1_2 == 0.0 && 
        fitsData.cd2_1 == 0.0 && fitsData.cd2_2 == 0.0) {
        return std::vector<long>();
    }
    
    CoordinateTransform transform(fitsData);
    
    double cd_scale = std::max(std::abs(fitsData.cd1_1), std::abs(fitsData.cd2_2));
    if (cd_scale == 0) cd_scale = 0.01;
    
    double healpix_size = HealpixUtil::getPixelSize(order);
    
    int sampleStep = std::max(1, (int)(healpix_size / cd_scale / 4.0));
    sampleStep = std::min(sampleStep, std::max(fitsData.width, fitsData.height) / 20);
    sampleStep = std::max(1, sampleStep);
    
    for (int y = 0; y <= fitsData.height - 1; y += sampleStep) {
        for (int x = 0; x <= fitsData.width - 1; x += sampleStep) {
            Coord pixel(x, y);
            CelestialCoord celestial = transform.pixelToCelestial(pixel);
            long npix = HealpixUtil::celestialToNested(celestial, order);
            if (npix >= 0) npixSet.insert(npix);
        }
        Coord pixel(fitsData.width - 1, y);
        CelestialCoord celestial = transform.pixelToCelestial(pixel);
        long npix = HealpixUtil::celestialToNested(celestial, order);
        if (npix >= 0) npixSet.insert(npix);
    }
    
    for (int x = 0; x <= fitsData.width - 1; x += sampleStep) {
        Coord pixel(x, fitsData.height - 1);
        CelestialCoord celestial = transform.pixelToCelestial(pixel);
        long npix = HealpixUtil::celestialToNested(celestial, order);
        if (npix >= 0) npixSet.insert(npix);
    }
    
    Coord corners[4] = {
        Coord(0, 0),
        Coord(fitsData.width - 1, 0),
        Coord(0, fitsData.height - 1),
        Coord(fitsData.width - 1, fitsData.height - 1)
    };
    for (int i = 0; i < 4; i++) {
        CelestialCoord celestial = transform.pixelToCelestial(corners[i]);
        long npix = HealpixUtil::celestialToNested(celestial, order);
        if (npix >= 0) npixSet.insert(npix);
    }
    
    std::vector<long> coverage;
    coverage.reserve(npixSet.size());
    coverage.assign(npixSet.begin(), npixSet.end());
    
    return coverage;
}

void HpxFinder::writeIndexFile(const std::string& indexPath,
                               const std::vector<SourceFileInfo>& sourceFiles) {
    fs::path path(indexPath);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    
    std::ofstream out(indexPath);
    if (!out.is_open()) {
        std::cerr << "Failed to open index file for writing: " << indexPath << std::endl;
        return;
    }
    
    for (const auto& src : sourceFiles) {
        out << "{\"filepath\":\"" << src.filepath 
            << "\",\"cellmem\":" << src.cellMem << "}\n";
    }
    
    out.close();
}

std::vector<SourceFileInfo> HpxFinder::readIndexFile(const std::string& indexPath) {
    std::vector<SourceFileInfo> result;
    
    std::ifstream in(indexPath);
    if (!in.is_open()) {
        return result;
    }
    
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        
        std::string filepath = extractPathFromJsonLine(line);
        long cellMem = extractCellMemFromJsonLine(line);
        
        if (!filepath.empty()) {
            result.push_back(SourceFileInfo(filepath, cellMem));
        }
    }
    
    in.close();
    return result;
}

std::string HpxFinder::extractPathFromJsonLine(const std::string& line) {
    size_t pos = line.find("\"filepath\":\"");
    if (pos == std::string::npos) return "";
    
    pos += 12;
    size_t end = line.find("\"", pos);
    if (end == std::string::npos) return "";
    
    return line.substr(pos, end - pos);
}

long HpxFinder::extractCellMemFromJsonLine(const std::string& line) {
    size_t pos = line.find("\"cellmem\":");
    if (pos == std::string::npos) return 0;
    
    pos += 10;
    size_t end = line.find_first_of(",}", pos);
    if (end == std::string::npos) return 0;
    
    std::string memStr = line.substr(pos, end - pos);
    try {
        return std::stol(memStr);
    } catch (...) {
        return 0;
    }
}
