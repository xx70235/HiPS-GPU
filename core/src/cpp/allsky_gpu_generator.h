#ifndef ALLSKY_GPU_GENERATOR_H
#define ALLSKY_GPU_GENERATOR_H

#include <string>
#include <vector>
#include <filesystem>
#include <fitsio.h>
#include <omp.h>

namespace fs = std::filesystem;

/**
 * GPU加速的Allsky生成器
 * 优化策略：
 * 1. 只遍历实际存在的tiles（而不是遍历所有可能的npix）
 * 2. 并行读取tiles
 * 3. GPU并行拼接（如果tiles足够多）
 */
class AllskyGPUGenerator {
public:
    /**
     * 快速生成Allsky.fits
     * @param hipsDir HiPS输出目录
     * @param order 目标order
     * @param outTileWidth 输出每个tile的宽度（默认64）
     */
    static bool generateAllskyFast(
        const std::string& hipsDir,
        int order,
        int outTileWidth = 64
    ) {
        std::cout << "Generating Allsky.fits (order=" << order << ") [OPTIMIZED]..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 计算参数
        int nside = 1 << order;
        int nTiles = 12 * nside * nside;
        int nbOutWidth = (int)std::sqrt(nTiles);
        int nbOutHeight = (int)std::ceil((double)nTiles / nbOutWidth);
        int outFileWidth = outTileWidth * nbOutWidth;
        int outFileHeight = outTileWidth * nbOutHeight;
        
        std::cout << "  Output size: " << outFileWidth << "x" << outFileHeight << std::endl;
        
        // 1. 扫描实际存在的tiles（而不是遍历所有可能的npix）
        std::string norderDir = hipsDir + "/Norder" + std::to_string(order);
        std::vector<std::pair<long, std::string>> existingTiles;
        
        for (const auto& dirEntry : fs::recursive_directory_iterator(norderDir)) {
            if (dirEntry.is_regular_file()) {
                std::string filename = dirEntry.path().filename().string();
                if (filename.substr(0, 4) == "Npix" && filename.find(".fits") != std::string::npos) {
                    // 解析npix
                    size_t start = 4;
                    size_t end = filename.find(".");
                    long npix = std::stol(filename.substr(start, end - start));
                    existingTiles.push_back({npix, dirEntry.path().string()});
                }
            }
        }
        
        std::cout << "  Found " << existingTiles.size() << " tiles to process" << std::endl;
        
        if (existingTiles.empty()) {
            std::cout << "  No tiles found, skipping Allsky generation" << std::endl;
            return true;
        }
        
        // 2. 创建输出图像（初始化为0/blank）
        std::vector<float> allskyPixels(outFileWidth * outFileHeight, 0.0f);
        
        // 3. 并行读取和放置tiles
        int tilesLoaded = 0;
        
        #pragma omp parallel for schedule(dynamic, 10) reduction(+:tilesLoaded)
        for (size_t i = 0; i < existingTiles.size(); i++) {
            long npix = existingTiles[i].first;
            const std::string& tilePath = existingTiles[i].second;
            
            // 读取tile
            fitsfile* fptr = nullptr;
            int status = 0;
            if (fits_open_file(&fptr, tilePath.c_str(), READONLY, &status)) continue;
            
            long naxes[2];
            int naxis;
            fits_get_img_dim(fptr, &naxis, &status);
            fits_get_img_size(fptr, 2, naxes, &status);
            
            std::vector<float> tileData(naxes[0] * naxes[1]);
            long fpixel[2] = {1, 1};
            fits_read_pix(fptr, TFLOAT, fpixel, tileData.size(), nullptr, tileData.data(), nullptr, &status);
            fits_close_file(fptr, &status);
            
            if (status != 0) continue;
            
            // 计算在Allsky中的位置
            int tileX = npix % nbOutWidth;
            int tileY = npix / nbOutWidth;
            
            // 降采样并放置
            for (int oy = 0; oy < outTileWidth; oy++) {
                for (int ox = 0; ox < outTileWidth; ox++) {
                    // 简单采样
                    int sx = ox * naxes[0] / outTileWidth;
                    int sy = oy * naxes[1] / outTileWidth;
                    float val = tileData[sy * naxes[0] + sx];
                    
                    int outX = tileX * outTileWidth + ox;
                    int outY = tileY * outTileWidth + oy;
                    if (outX < outFileWidth && outY < outFileHeight) {
                        allskyPixels[outY * outFileWidth + outX] = val;
                    }
                }
            }
            
            tilesLoaded++;
        }
        
        std::cout << "  Loaded " << tilesLoaded << " tiles" << std::endl;
        
        // 4. 写入Allsky.fits
        std::string outPath = norderDir + "/Allsky.fits";
        fitsfile* fptr = nullptr;
        int status = 0;
        
        std::string fileFmt = "!" + outPath;
        fits_create_file(&fptr, fileFmt.c_str(), &status);
        
        long naxes[2] = {outFileWidth, outFileHeight};
        fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
        
        long fpixel[2] = {1, 1};
        fits_write_pix(fptr, TFLOAT, fpixel, allskyPixels.size(), allskyPixels.data(), &status);
        fits_close_file(fptr, &status);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "  Allsky generated in " << duration.count() << " ms" << std::endl;
        
        return status == 0;
    }
};

#endif // ALLSKY_GPU_GENERATOR_H
