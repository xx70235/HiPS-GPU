/**
 * HpxFinder索引实现
 * 优化版本：GPU加速 + 多线程FITS读取
 */

#include "hpx_finder.h"
#include "index_gpu_processor.h"
#include "fits_header_reader.h"
#include "index_merge_utils.h"
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

namespace {

std::string makeStoredSourcePath(const std::string& fitsPath,
                                 const std::string& inputDir,
                                 const std::string& sourceRootDir) {
    fs::path basePath = sourceRootDir.empty() ? fs::path(inputDir) : fs::path(sourceRootDir);
    fs::path inputPath(fitsPath);

    std::string storedPath;
    fs::path lexicalRelative = inputPath.lexically_relative(basePath);
    if (!lexicalRelative.empty()) {
        storedPath = lexicalRelative.generic_string();
        if (storedPath == "." || storedPath.rfind("../", 0) == 0) {
            storedPath.clear();
        }
    }

    if (storedPath.empty()) {
        std::error_code ec;
        fs::path relativePath = fs::relative(inputPath, basePath, ec);
        if (!ec) {
            storedPath = relativePath.generic_string();
            if (storedPath == "." || storedPath.empty() || storedPath.rfind("../", 0) == 0) {
                storedPath.clear();
            }
        }
    }

    if (storedPath.empty()) {
        storedPath = inputPath.filename().generic_string();
    }
    return storedPath;
}

FitsData makeMetadataFitsData(const std::string& filename, const WCSInfo& wcs) {
    FitsData data;
    data.filename = filename;
    data.width = wcs.width;
    data.height = wcs.height;
    data.depth = 1;
    data.crval1 = wcs.crval1;
    data.crval2 = wcs.crval2;
    data.crpix1 = wcs.crpix1;
    data.crpix2 = wcs.crpix2;
    data.cd1_1 = wcs.cd1_1;
    data.cd1_2 = wcs.cd1_2;
    data.cd2_1 = wcs.cd2_1;
    data.cd2_2 = wcs.cd2_2;
    data.blank = std::numeric_limits<double>::quiet_NaN();
    data.isValid = wcs.valid;
    return data;
}

std::vector<std::string> collectFitsFilesRecursive(const std::string& inputDir) {
    std::vector<std::string> fitsFiles;
    if (fs::exists(inputDir) && fs::is_directory(inputDir)) {
        for (const auto& entry : fs::recursive_directory_iterator(inputDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".fits" || ext == ".fit" || ext == ".fts" || ext == ".img" || ext == ".fz") {
                fitsFiles.push_back(entry.path().string());
            }
        }
    }
    return fitsFiles;
}

void ensureOrderDirectories(const std::string& hpxFinderPath, int orderMin, int orderMax) {
    for (int order = orderMin; order <= orderMax; ++order) {
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
}

}  // namespace

/**
 * 生成INDEX阶段的空间索引（GPU加速版本）
 */
bool HpxFinder::generateIndex(const std::string& inputDir, 
                              const std::string& outputDir, 
                              int orderMax,
                              const std::string& sourceRootDir) {
    std::string hpxFinderPath = outputDir + "/HpxFinder";
    
    std::cout << "开始生成HpxFinder索引..." << std::endl;
    std::cout << "输入目录: " << inputDir << std::endl;
    std::cout << "输出目录: " << hpxFinderPath << std::endl;
    std::cout << "最大order: " << orderMax << std::endl;
    if (!sourceRootDir.empty()) {
        std::cout << "源数据根目录: " << sourceRootDir << std::endl;
    }
    
    // 创建HpxFinder目录结构
    ensureOrderDirectories(hpxFinderPath, 0, orderMax);
    
    // 扫描输入目录中的FITS文件
    std::vector<std::string> fitsFiles = collectFitsFilesRecursive(inputDir);
    
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
        std::atomic<int> invalidCount(0);
        
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
                
                if (!wcs.valid) {
                    ++invalidCount;
                    continue;
                }
                
                std::string relPath = makeStoredSourcePath(fitsPath, inputDir, sourceRootDir);
                
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
                  << " 有效, " << invalidCount << " 无效, 耗时 " << readMs << " ms" << std::endl;
        
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
            
            std::string relPath = makeStoredSourcePath(fitsPath, inputDir, sourceRootDir);
            
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

bool HpxFinder::appendIndex(const std::string& inputDir,
                            const std::string& outputDir,
                            int orderMax,
                            const std::string& sourceRootDir,
                            std::vector<long>* dirtyOrderMaxTiles) {
    std::string hpxFinderPath = outputDir + "/HpxFinder";
    std::string effectiveSourceRoot = sourceRootDir.empty() ? inputDir : sourceRootDir;

    std::cout << "开始追加HpxFinder索引..." << std::endl;
    std::cout << "新批次输入目录: " << inputDir << std::endl;
    std::cout << "输出目录: " << hpxFinderPath << std::endl;
    std::cout << "最大order: " << orderMax << std::endl;
    std::cout << "源数据根目录: " << effectiveSourceRoot << std::endl;

    ensureOrderDirectories(hpxFinderPath, orderMax, orderMax);

    std::vector<std::string> fitsFiles = collectFitsFilesRecursive(inputDir);
    std::cout << "找到 " << fitsFiles.size() << " 个待追加FITS文件" << std::endl;

    if (dirtyOrderMaxTiles != nullptr) {
        dirtyOrderMaxTiles->clear();
    }
    if (fitsFiles.empty()) {
        return true;
    }

    auto wcsInfos = FastFitsHeaderReader::readWCSInfoBatch(fitsFiles);
    std::map<long, std::vector<SourceFileInfo>> incomingByTile;
    int validFiles = 0;
    int invalidFiles = 0;

    for (size_t i = 0; i < fitsFiles.size(); ++i) {
        if (i >= wcsInfos.size() || !wcsInfos[i].valid) {
            invalidFiles++;
            continue;
        }

        FitsData meta = makeMetadataFitsData(fitsFiles[i], wcsInfos[i]);
        std::string relPath = makeStoredSourcePath(fitsFiles[i], inputDir, effectiveSourceRoot);
        long cellMem = (long)wcsInfos[i].width * wcsInfos[i].height * 4;
        SourceFileInfo srcInfo(relPath, cellMem);

        std::vector<long> coverage = computeCoverage(meta, orderMax);
        for (long npix : coverage) {
            incomingByTile[npix].push_back(srcInfo);
        }
        validFiles++;
    }

    std::vector<long> dirtyTiles;
    int updatedIndexFiles = 0;
    int unchangedIndexFiles = 0;

    for (auto& entry : incomingByTile) {
        long npix = entry.first;
        int dir = HealpixUtil::getDirNumber(npix);
        std::ostringstream oss;
        oss << hpxFinderPath << "/Norder" << orderMax << "/Dir" << dir << "/Npix" << npix;
        std::string indexPath = oss.str();

        std::vector<SourceFileInfo> existing = readIndexFile(indexPath);
        bool changed = merge_source_files_unique(existing, entry.second);
        if (changed) {
            writeIndexFile(indexPath, existing);
            dirtyTiles.push_back(npix);
            updatedIndexFiles++;
        } else {
            unchangedIndexFiles++;
        }
    }

    std::sort(dirtyTiles.begin(), dirtyTiles.end());
    dirtyTiles.erase(std::unique(dirtyTiles.begin(), dirtyTiles.end()), dirtyTiles.end());
    if (dirtyOrderMaxTiles != nullptr) {
        *dirtyOrderMaxTiles = dirtyTiles;
    }

    std::cout << "追加索引完成: valid=" << validFiles
              << ", invalid=" << invalidFiles
              << ", touched=" << incomingByTile.size()
              << ", updated=" << updatedIndexFiles
              << ", unchanged=" << unchangedIndexFiles
              << ", dirty orderMax tiles=" << dirtyTiles.size() << std::endl;

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
