/**
 * HEALPix网格处理模块
 * HEALPix: Hierarchical Equal Area isoLatitude Pixelization
 * 用于HiPS的sky tessellation（天空分割）
 */

#ifndef HEALPIX_UTIL_H
#define HEALPIX_UTIL_H

#include <cmath>
#include <vector>
#include "coordinate_transform.h"

/**
 * HEALPix参数
 */
struct HealpixParams {
    int norder;        // Order（层级，0-N）
    long npix;         // Pixel index（嵌套编号，0-N）
    int nside;         // Nside（每边的像素数，nside = 2^norder）
    
    HealpixParams(int order = 0, long pixel = 0)
        : norder(order), npix(pixel) {
        nside = 1 << order;  // nside = 2^order
    }
};

/**
 * HEALPix工具类
 * 实现HEALPix网格计算和坐标转换
 */
class HealpixUtil {
public:
    /**
     * 从norder和npix计算nside
     * @param norder HEALPix order (0-N)
     * @return nside (每边的像素数，nside = 2^norder)
     */
    static int norderToNside(int norder) {
        if (norder < 0) return 1;
        return 1 << norder;  // 2^order
    }
    
    /**
     * 从nside计算norder
     * @param nside HEALPix nside
     * @return norder (层级)
     */
    static int nsideToNorder(int nside) {
        if (nside <= 0) return 0;
        int order = 0;
        int temp = nside;
        while (temp > 1) {
            temp >>= 1;
            order++;
        }
        return order;
    }
    
    /**
     * 从norder计算总像素数
     * @param norder HEALPix order
     * @return 总像素数 (12 * nside^2)
     */
    static long getTotalPixels(int norder) {
        int nside = norderToNside(norder);
        return 12L * nside * nside;
    }
    
    /**
     * 从nside计算总像素数
     * @param nside HEALPix nside
     * @return 总像素数 (12 * nside^2)
     */
    static long getTotalPixelsFromNside(int nside) {
        return 12L * nside * nside;
    }
    
    /**
     * 将HEALPix嵌套索引转换为天球坐标（RA/Dec）
     * @param norder HEALPix order
     * @param npix Pixel index (嵌套编号)
     * @return 天球坐标（RA/Dec，度）
     */
    static CelestialCoord nestedToCelestial(int norder, long npix);
    
    /**
     * 将天球坐标（RA/Dec）转换为HEALPix嵌套索引
     * @param celestial 天球坐标（RA/Dec，度）
     * @param norder HEALPix order
     * @return Pixel index (嵌套编号)
     */
    static long celestialToNested(const CelestialCoord& celestial, int norder);
    
    /**
     * 从HEALPix索引计算tile中心的天球坐标
     * @param norder HEALPix order
     * @param npix Pixel index
     * @return 天球坐标（RA/Dec，度）
     */
    static CelestialCoord getPixelCenter(int norder, long npix);
    
    /**
     * 计算HEALPix像素的角直径（度）
     * @param norder HEALPix order
     * @return 像素角直径（度）
     */
    static double getPixelSize(int norder);
    
    /**
     * 检查npix是否在有效范围内
     * @param npix Pixel index
     * @param norder HEALPix order
     * @return true if valid
     */
    static bool isValidPixel(long npix, int norder);
    
    /**
     * 计算HEALPix tile的目录结构路径
     * @param norder HEALPix order
     * @param npix Pixel index
     * @return 目录路径字符串（例如："Norder3/Dir12/Npix12345.fits"）
     */
    static std::string getTilePath(int norder, long npix, const std::string& ext = ".fits");
    
    /**
     * 计算npix对应的Dir编号（用于目录组织）
     * HEALPix tiles按Dir组织：Dir = npix / 10000
     * @param npix Pixel index
     * @return Dir编号
     */
    static int getDirNumber(long npix) {
        return (int)((npix / 10000) * 10000);
    }
    
    /**
     * 计算tile内的像素坐标范围（相对于tile中心的偏移）
     * @param norder HEALPix order
     * @param tileWidth Tile宽度（像素）
     * @return 像素坐标范围（从-tileWidth/2到+tileWidth/2）
     */
    static std::vector<Coord> getTilePixelCoords(int norder, int tileWidth = 512);
    
private:
    // HEALPix常数
    static const int BASE_NSIDE = 1;  // 基础nside
    static const int BASE_NPIX = 12;  // 基础像素数（12个基础像素）
    
    // HEALPix内部辅助函数
    static void nestedToXy(int norder, long npix, int& x, int& y);
    static long xyToNested(int norder, int x, int y);
    static void xyToThetaPhi(int norder, int x, int y, double& theta, double& phi);
    static void thetaPhiToXy(int norder, double theta, double phi, int& x, int& y);
};

#endif // HEALPIX_UTIL_H
