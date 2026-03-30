/**
 * FITS文件写入模块
 * 使用CFITSIO库写入FITS tile文件
 */

#ifndef FITS_WRITER_H
#define FITS_WRITER_H

#include <string>
#include "fits_io.h"
#include "hips_tile_generator.h"

#ifdef USE_CFITSIO
#include "fitsio.h"  // CFITSIO header
#endif

/**
 * FITS文件写入器类
 */
class FitsWriter {
public:
    /**
     * 写入FITS tile文件
     * 
     * @param tilePath 输出文件路径
     * @param tile Tile数据
     * @param order HEALPix order
     * @param npix HEALPix pixel index
     * @param frame 坐标系（ICRS/GAL等，默认ICRS）
     * @return 成功返回true
     */
    static bool writeTileFile(
        const std::string& tilePath,
        const TileData& tile,
        int order,
        long npix,
        const std::string& frame = "ICRS"
    );
    
    /**
     * 写入FITS文件（通用方法）
     * 
     * @param filePath 输出文件路径
     * @param data 像素数据（float数组）
     * @param width 图像宽度
     * @param height 图像高度
     * @param bitpix BITPIX值（-32=float, -64=double, 16=short, 32=int）
     * @param blank 空白值
     * @param bzero BZERO值
     * @param bscale BSCALE值
     * @return 成功返回true
     */
    static bool writeFitsFile(
        const std::string& filePath,
        const float* data,
        int width, int height,
        int bitpix = -32,
        double blank = 0.0,
        double bzero = 0.0,
        double bscale = 1.0
    );
    
private:
#ifdef USE_CFITSIO
    /**
     * 使用CFITSIO写入FITS文件（完整实现）
     */
    static bool writeFitsFileCFITSIO(
        const std::string& filePath,
        const float* data,
        int width, int height,
        int bitpix,
        double blank,
        double bzero,
        double bscale
    );
    
    /**
     * 添加WCS头信息（用于HiPS tile）
     */
    static void addWCSHeaders(fitsfile* fptr, int order, long npix, const std::string& frame);
#else
    /**
     * 简化FITS写入（无CFITSIO时使用，仅支持基本格式）
     */
    static bool writeFitsFileSimple(
        const std::string& filePath,
        const float* data,
        int width, int height,
        int bitpix,
        double blank,
        double bzero,
        double bscale
    );
#endif
};

#endif // FITS_WRITER_H
