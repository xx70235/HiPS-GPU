/**
 * HiPS Tile生成器实现
 * 优化版本：包含缓存、批量处理等效率优化
 */

#include "hips_tile_generator.h"
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <mutex>
#include <functional>

// 缓存已读取的FITS文件（避免重复读取）
static std::unordered_map<std::string, FitsData> fitsCache;
static std::mutex cacheMutex;

/**
 * xy2hpx mapping: Convert tile pixel coordinates to HEALPix index
 * Uses the same recursive quadrant algorithm as Java Aladin
 * This ensures pixel-level compatibility with the Java HiPS generator
 */

// Helper function: recursive fillUp (matches Java Context.fillUp)
static void fillUpXY2HPX(std::vector<int>& npix, int nsize, std::vector<int>* pos) {
    int size = nsize * nsize;
    std::vector<std::vector<int>> fils(4);
    for (int i = 0; i < 4; i++) fils[i].resize(size / 4);
    std::vector<int> nb(4, 0);
    
    for (int i = 0; i < size; i++) {
        // dg: left(0) or right(1) half
        int dg = (i % nsize) < (nsize / 2) ? 0 : 1;
        // bh: top(1) or bottom(0) half
        int bh = i < (size / 2) ? 1 : 0;
        // quad: quadrant number (0-3)
        int quad = (dg << 1) | bh;
        
        int j = (pos == nullptr) ? i : (*pos)[i];
        npix[j] = (npix[j] << 2) | quad;
        fils[quad][nb[quad]++] = j;
    }
    
    if (size > 4) {
        for (int i = 0; i < 4; i++) {
            fillUpXY2HPX(npix, nsize / 2, &fils[i]);
        }
    }
}

void HipsTileGenerator::createXY2HPXMapping(int tileOrder, std::vector<int>& xy2hpx, std::vector<int>& hpx2xy) {
    int nside = 1 << tileOrder;  // tileOrder通常是9（512=2^9）
    if (nside <= 0) nside = 512;
    
    xy2hpx.resize(nside * nside, 0);
    hpx2xy.resize(nside * nside);
    
    // Use the same recursive algorithm as Java Aladin
    fillUpXY2HPX(xy2hpx, nside, nullptr);
    
    // Build the reverse mapping
    for (int i = 0; i < nside * nside; i++) {
        hpx2xy[xy2hpx[i]] = i;
    }
}

/**
 * 生成单个HEALPix tile（优化版本）
 */
std::unique_ptr<TileData> HipsTileGenerator::generateTile(
    const std::string& inputDir,
    const std::string& hpxFinderPath,
    int order, long npix,
    int tileWidth,
    int bitpix,
    double blank
) {
    // 查询HpxFinder获取覆盖该tile的源文件列表
    std::vector<SourceFileInfo> sourceFiles = HpxFinder::queryIndex(hpxFinderPath, order, npix);
    
    if (sourceFiles.empty()) {
        // 没有源文件覆盖此tile
        return nullptr;
    }
    
    // 创建tile数据
    auto tile = std::make_unique<TileData>(tileWidth, tileWidth, bitpix);
    tile->blank = blank;
    
    // 计算tileOrder（通常为9，因为512=2^9）
    int tileOrder = 0;
    int temp = tileWidth;
    while (temp > 1) {
        temp >>= 1;
        tileOrder++;
    }
    
    // 创建xy2hpx映射（缓存，避免重复计算）
    static std::vector<int> xy2hpx;
    static std::vector<int> hpx2xy;
    static int cachedTileOrder = -1;
    
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (cachedTileOrder != tileOrder) {
            createXY2HPXMapping(tileOrder, xy2hpx, hpx2xy);
            cachedTileOrder = tileOrder;
        }
    }
    
    // 计算orderPix（tile内像素对应的HEALPix order）
    int orderPix = order + tileOrder;
    
    // 计算起始HEALPix索引
    long minHPXIndex = npix * tileWidth * tileWidth;
    
    // 预加载和缓存源文件（优化：避免重复读取）
    std::vector<FitsData> cachedFitsFiles;
    std::vector<CoordinateTransform> cachedTransforms;
    
    for (const auto& srcInfo : sourceFiles) {
        std::string filepath = inputDir + "/" + srcInfo.filepath;
        
        // 检查缓存
        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            auto it = fitsCache.find(filepath);
            if (it != fitsCache.end()) {
                cachedFitsFiles.push_back(it->second);
                cachedTransforms.emplace_back(it->second);
                continue;
            }
        }
        
        // 读取文件并加入缓存
        FitsData fits = FitsReader::readFitsFile(filepath);
        if (fits.isValid) {
            {
                std::lock_guard<std::mutex> lock(cacheMutex);
                fitsCache[filepath] = fits;
            }
            cachedFitsFiles.push_back(fits);
            cachedTransforms.emplace_back(fits);
        }
    }
    
    // 批量处理所有像素
    int totalPixels = tileWidth * tileWidth;
    
    // 预计算所有像素的天球坐标
    std::vector<CelestialCoord> celestialCoords(totalPixels);
    for (int y = 0; y < tileWidth; y++) {
        for (int x = 0; x < tileWidth; x++) {
            int pixelIndex = y * tileWidth + x;
            int hpxOffset = xy2hpx[pixelIndex];
            long hpxIndex = minHPXIndex + hpxOffset;
            celestialCoords[pixelIndex] = HealpixUtil::nestedToCelestial(orderPix, hpxIndex);
        }
    }
    
    // 初始化结果数组（用于累加各源文件的贡献）
    std::vector<double> tileValues(totalPixels, 0.0);
    std::vector<double> tileWeights(totalPixels, 0.0);
    std::vector<int> tileCounts(totalPixels, 0);
    
    // 对每个源文件进行批量插值
    for (size_t fi = 0; fi < cachedFitsFiles.size() && fi < cachedTransforms.size(); fi++) {
        const FitsData& fits = cachedFitsFiles[fi];
        const CoordinateTransform& transform = cachedTransforms[fi];
        
        // 批量转换天球坐标到像素坐标
        std::vector<double> coordsX(totalPixels);
        std::vector<double> coordsY(totalPixels);
        std::vector<bool> inBounds(totalPixels, false);
        int validCount = 0;
        
        for (int i = 0; i < totalPixels; i++) {
            Coord pixel = transform.celestialToPixel(celestialCoords[i]);
            coordsX[i] = pixel.x;
            coordsY[i] = pixel.y;
            if (transform.isPixelInBounds(pixel)) {
                inBounds[i] = true;
                validCount++;
            }
        }
        
        // 如果没有有效像素，跳过这个源文件
        if (validCount == 0) continue;
        
        // 批量插值
        std::vector<double> interpResults(totalPixels);
        
        // 自适应选择：当图像较大时使用CUDA，否则用CPU
        // CUDA的GPU内存传输开销较大，只有图像大于某个阈值时CUDA才有优势
        const int CUDA_THRESHOLD = 4096 * 4096;  // 16M像素
        bool useCUDA = (fits.width * fits.height >= CUDA_THRESHOLD);
        
        if (useCUDA) {
            bool cudaSuccess = bilinearInterpolationBatchCUDA(
                fits.pixels.data(),
                fits.width, fits.height,
                coordsX.data(), coordsY.data(),
                interpResults.data(),
                totalPixels,
                (float)fits.blank
            );
            if (!cudaSuccess) {
                useCUDA = false;  // CUDA失败，回退到CPU
            }
        }
        
        if (!useCUDA) {
            bilinearInterpolationBatchCPU(
                fits.pixels.data(),
                fits.width, fits.height,
                coordsX.data(), coordsY.data(),
                interpResults.data(),
                totalPixels,
                (float)fits.blank
            );
        }
        
        // 累加结果（coadd）
        for (int i = 0; i < totalPixels; i++) {
            if (!inBounds[i]) continue;
            
            double value = interpResults[i];
            if (std::isnan(value) || value == blank || value == fits.blank) continue;
            
            double weight = 1.0;
            tileValues[i] += value * weight;
            tileWeights[i] += weight;
            tileCounts[i]++;
        }
    }
    
    // 计算加权平均并设置tile像素
    for (int y = 0; y < tileWidth; y++) {
        for (int x = 0; x < tileWidth; x++) {
            int idx = y * tileWidth + x;
            double finalValue;
            if (tileWeights[idx] > 0.0 && tileCounts[idx] > 0) {
                finalValue = tileValues[idx] / tileWeights[idx];
            } else {
                finalValue = blank;
            }
            tile->setPixel(x, y, finalValue);
        }
    }
    
    return tile;
}

/**
 * 对单个像素进行插值（多文件coadd，优化版本）
 */
double HipsTileGenerator::interpolatePixel(
    const std::vector<FitsData>& cachedFitsFiles,
    const std::vector<CoordinateTransform>& cachedTransforms,
    const CelestialCoord& celestial,
    double blank
) {
    double totalValue = 0.0;
    double totalWeight = 0.0;
    int validCount = 0;
    
    // 遍历所有缓存的源文件
    for (size_t i = 0; i < cachedFitsFiles.size() && i < cachedTransforms.size(); i++) {
        const FitsData& fits = cachedFitsFiles[i];
        const CoordinateTransform& transform = cachedTransforms[i];
        
        // 将天球坐标转换为像素坐标
        Coord pixel = transform.celestialToPixel(celestial);
        
        // 检查是否在图像范围内（快速预检查）
        if (!transform.isPixelInBounds(pixel)) {
            continue;
        }
        
        // 使用CPU进行双线性插值（比CUDA单点插值更高效）
        // TODO: 后续优化为批量CUDA处理
        double value = bilinearInterpolationCPU(
            fits.pixels.data(),
            fits.width, fits.height,
            pixel.x, pixel.y,
            (float)fits.blank
        );
        
        // 检查是否为有效值
        if (std::isnan(value) || value == blank) {
            continue;
        }
        
        // 加权平均（简化：权重为1）
        // 实际可以根据距离中心的位置、图像质量等因素计算权重
        double weight = 1.0;
        
        totalValue += value * weight;
        totalWeight += weight;
        validCount++;
        
        // 优化：如果只需要一个源文件（overlayNone模式），找到第一个有效值后退出
        // 这里简化处理，总是使用所有源文件
    }
    
    // 计算加权平均
    if (totalWeight > 0.0 && validCount > 0) {
        return totalValue / totalWeight;
    }
    
    return blank;  // 没有有效值，返回空白值
}
