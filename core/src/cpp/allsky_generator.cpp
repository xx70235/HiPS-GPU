/**
 * Allsky概览图生成器实现 - 优化版本
 * 改进：只遍历实际存在的tiles，而不是所有可能的npix
 */

#include "allsky_generator.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fitsio.h>
#include <cstring>
#include <chrono>
#include <omp.h>

namespace fs = std::filesystem;

/**
 * 生成Allsky.fits文件 - 优化版本
 */
bool AllskyGenerator::generateAllskyFits(
    const std::string& hipsDir,
    int order,
    int outTileWidth
) {
    auto startTime = std::chrono::high_resolution_clock::now();
    std::cout << "生成Allsky.fits (order=" << order << ")..." << std::endl;
    
    // 计算参数
    int nside = 1 << order;
    int nTiles = 12 * nside * nside;
    int nbOutWidth = (int)std::sqrt(nTiles);
    int nbOutHeight = (int)std::ceil((double)nTiles / nbOutWidth);
    int outFileWidth = outTileWidth * nbOutWidth;
    int outFileHeight = outTileWidth * nbOutHeight;
    
    std::cout << "  Tiles数量: " << nTiles 
              << " (" << nbOutWidth << "x" << nbOutHeight << ")"
              << ", 输出尺寸: " << outFileWidth << "x" << outFileHeight 
              << std::endl;
    
    // 创建输出图像
    std::vector<float> allskyPixels(outFileWidth * outFileHeight, 0.0f);
    double blank = 0.0;
    double bzero = 0.0;
    double bscale = 1.0;
    int bitpix = -32;
    bool foundParam = false;
    
    // === 优化：扫描目录找实际存在的tiles ===
    std::string norderDir = hipsDir + "/Norder" + std::to_string(order);
    std::vector<std::pair<long, std::string>> existingTiles;
    
    // 扫描所有Dir子目录
    if (fs::exists(norderDir)) {
        for (const auto& dirEntry : fs::directory_iterator(norderDir)) {
            if (dirEntry.is_directory()) {
                std::string dirName = dirEntry.path().filename().string();
                if (dirName.substr(0, 3) == "Dir") {
                    // 扫描这个Dir下的所有Npix文件
                    for (const auto& fileEntry : fs::directory_iterator(dirEntry.path())) {
                        if (fileEntry.is_regular_file()) {
                            std::string filename = fileEntry.path().filename().string();
                            if (filename.substr(0, 4) == "Npix" && 
                                filename.find(".fits") != std::string::npos) {
                                // 解析npix
                                size_t start = 4;
                                size_t end = filename.find(".");
                                long npix = std::stol(filename.substr(start, end - start));
                                existingTiles.push_back({npix, fileEntry.path().string()});
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "  找到 " << existingTiles.size() << " 个实际存在的tiles" << std::endl;
    
    if (existingTiles.empty()) {
        std::cerr << "警告: 未找到任何Order " << order << " tiles" << std::endl;
        return false;
    }
    
    // 并行读取和处理tiles
    int tilesLoaded = 0;
    
    #pragma omp parallel for schedule(dynamic, 10) reduction(+:tilesLoaded)
    for (size_t i = 0; i < existingTiles.size(); i++) {
        long npix = existingTiles[i].first;
        const std::string& tilePath = existingTiles[i].second;
        
        // 读取tile
        FitsData tile = FitsReader::readFitsFile(tilePath);
        if (!tile.isValid) continue;
        
        tilesLoaded++;
        
        // 获取参数（第一个有效tile）
        #pragma omp critical
        {
            if (!foundParam) {
                bzero = tile.bzero;
                bscale = tile.bscale;
                blank = tile.blank;
                bitpix = tile.bitpix;
                foundParam = true;
            }
        }
        
        // 计算此tile在Allsky中的位置
        int yTile = npix / nbOutWidth;
        int xTile = npix % nbOutWidth;
        
        // 下采样比例
        int gap = tile.width / outTileWidth;
        if (gap < 1) gap = 1;
        
        // 复制像素（带下采样）
        for (int y = 0; y < outTileWidth && y * gap < tile.height; y++) {
            for (int x = 0; x < outTileWidth && x * gap < tile.width; x++) {
                int srcX = x * gap;
                int srcY = tile.height - 1 - y * gap;
                float p = tile.pixels[srcY * tile.width + srcX];
                
                int dstX = xTile * outTileWidth + x;
                int dstY = outFileHeight - 1 - (yTile * outTileWidth + y);
                
                if (dstX >= 0 && dstX < outFileWidth && 
                    dstY >= 0 && dstY < outFileHeight) {
                    // 使用atomic避免竞争（虽然不太可能有冲突）
                    allskyPixels[dstY * outFileWidth + dstX] = p;
                }
            }
        }
    }
    
    std::cout << "  加载了 " << tilesLoaded << " 个tiles" << std::endl;
    
    // 写入Allsky.fits
    std::string allskyPath = norderDir + "/Allsky.fits";
    
    // 使用cfitsio写入
    fitsfile* fptr = nullptr;
    int status = 0;
    
    std::string fileFmt = "!" + allskyPath;
    if (fits_create_file(&fptr, fileFmt.c_str(), &status)) {
        std::cerr << "无法创建Allsky.fits" << std::endl;
        return false;
    }
    
    long naxes[2] = {outFileWidth, outFileHeight};
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    
    // 写入像素数据
    long fpixel[2] = {1, 1};
    fits_write_pix(fptr, TFLOAT, fpixel, allskyPixels.size(), allskyPixels.data(), &status);
    
    // 写入元数据
    char comment[80] = "";
    fits_write_key(fptr, TDOUBLE, "BZERO", &bzero, "Offset", &status);
    fits_write_key(fptr, TDOUBLE, "BSCALE", &bscale, "Scale", &status);
    fits_write_key(fptr, TINT, "HIPSORD", &order, "HiPS order", &status);
    
    fits_close_file(fptr, &status);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "  Allsky.fits已生成: " << allskyPath 
              << " (耗时 " << duration.count() << " ms)" << std::endl;
    
    return status == 0;
}
