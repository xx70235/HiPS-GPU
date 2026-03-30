/**
 * HiPS Tile生成器
 * 实现Tile生成的核心逻辑（对应原版的buildHealpix1）
 */

#ifndef HIPS_TILE_GENERATOR_H
#define HIPS_TILE_GENERATOR_H

#include <vector>
#include <string>
#include <memory>
#include "fits_io.h"
#include "coordinate_transform.h"
#include "healpix_util.h"
#include "hpx_finder.h"
#include "bilinear_interpolation_cuda.h"

// Tile数据
struct TileData {
    int width, height;
    int bitpix;
    std::vector<float> pixels;
    double blank;
    double bzero, bscale;
    
    TileData(int w = 512, int h = 512, int bp = -32)
        : width(w), height(h), bitpix(bp), blank(0.0), bzero(0.0), bscale(1.0) {
        pixels.resize(w * h, 0.0f);
    }
    
    void setPixel(int x, int y, double value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            pixels[y * width + x] = (float)value;
        }
    }
    
    double getPixel(int x, int y) const {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            return pixels[y * width + x];
        }
        return blank;
    }
};

/**
 * HiPS Tile生成器类
 */
class HipsTileGenerator {
public:
    /**
     * 生成单个HEALPix tile
     * 
     * @param inputDir 输入目录（源FITS文件目录）
     * @param hpxFinderPath HpxFinder索引路径
     * @param order HEALPix order
     * @param npix HEALPix pixel index
     * @param tileWidth Tile宽度（默认512）
     * @param bitpix 输出BITPIX
     * @param blank 空白值
     * @return Tile数据，如果生成失败返回nullptr
     */
    static std::unique_ptr<TileData> generateTile(
        const std::string& inputDir,
        const std::string& hpxFinderPath,
        int order, long npix,
        int tileWidth = 512,
        int bitpix = -32,
        double blank = 0.0
    );
    
    /**
     * Create xy2hpx and hpx2xy mapping for tile pixel ordering
     * @param tileOrder The order of the tile (e.g., 9 for 512x512 tiles)
     * @param xy2hpx Output: mapping from x,y index to HEALPix offset
     * @param hpx2xy Output: inverse mapping
     */
    static void createXY2HPXMapping(int tileOrder, std::vector<int>& xy2hpx, std::vector<int>& hpx2xy);
    
private:
    /**
     * 对tile内的单个像素进行插值（多文件coadd，优化版本）
     * 使用预缓存的FITS文件和坐标转换，避免重复读取
     */
    static double interpolatePixel(
        const std::vector<FitsData>& cachedFitsFiles,
        const std::vector<CoordinateTransform>& cachedTransforms,
        const CelestialCoord& celestial,
        double blank
    );
};

#endif // HIPS_TILE_GENERATOR_H
