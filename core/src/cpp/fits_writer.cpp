/**
 * FITS文件写入实现
 * 使用CFITSIO库写入FITS tile文件
 */

#include "fits_writer.h"
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include "healpix_util.h"

#ifdef USE_CFITSIO
#include "fitsio.h"  // CFITSIO header
#endif

namespace fs = std::filesystem;

/**
 * 写入FITS tile文件
 */
bool FitsWriter::writeTileFile(
    const std::string& tilePath,
    const TileData& tile,
    int order,
    long npix,
    const std::string& frame
) {
    // 确保目录存在
    fs::path path(tilePath);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    
#ifdef USE_CFITSIO
    // 如果使用CFITSIO，先写入基本文件，然后添加WCS头信息
    if (FitsReader::isCFITSIOAvailable()) {
        fitsfile* fptr = nullptr;
        int status = 0;
        
        // 创建或覆盖文件
        std::string fileFmt = "!" + tilePath;
        if (fits_create_file(&fptr, fileFmt.c_str(), &status)) {
            fits_report_error(stderr, status);
            return false;
        }
        
        // 创建primary image HDU
        long naxes[2] = { (long)tile.width, (long)tile.height };
        if (fits_create_img(fptr, tile.bitpix, 2, naxes, &status)) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return false;
        }
        
        // 添加WCS头信息（在写入数据前）
        addWCSHeaders(fptr, order, npix, frame);
        
        // 写入基本关键字
        if (tile.blank != 0.0 && tile.bitpix > 0) {
            long blankValue = (long)tile.blank;
            if (fits_update_key(fptr, TLONG, "BLANK", &blankValue, "Value for blank pixels", &status)) {
                fits_report_error(stderr, status);
            }
        }
        
        if (tile.bzero != 0.0) {
            double bzero_val = tile.bzero;
            if (fits_update_key(fptr, TDOUBLE, "BZERO", &bzero_val, "Zero point for data scaling", &status)) {
                fits_report_error(stderr, status);
            }
        }
        
        if (tile.bscale != 1.0) {
            double bscale_val = tile.bscale;
            if (fits_update_key(fptr, TDOUBLE, "BSCALE", &bscale_val, "Data scaling factor", &status)) {
                fits_report_error(stderr, status);
            }
        }
        
        // 写入像素数据
        long fpixel[2] = {1, 1};
        long nelements = (long)tile.width * tile.height;
        int datatype = (tile.bitpix == -32) ? TFLOAT : ((tile.bitpix == -64) ? TDOUBLE : TFLOAT);
        
        if (fits_write_pix(fptr, datatype, fpixel, nelements, const_cast<float*>(tile.pixels.data()), &status)) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return false;
        }
        
        // 关闭文件
        if (fits_close_file(fptr, &status)) {
            fits_report_error(stderr, status);
            return false;
        }
        
        return true;
    }
#endif
    
    // 使用简化方法
    return writeFitsFile(
        tilePath,
        tile.pixels.data(),
        tile.width, tile.height,
        tile.bitpix,
        tile.blank,
        tile.bzero,
        tile.bscale
    );
}

/**
 * 写入FITS文件（通用方法）
 */
bool FitsWriter::writeFitsFile(
    const std::string& filePath,
    const float* data,
    int width, int height,
    int bitpix,
    double blank,
    double bzero,
    double bscale
) {
#ifdef USE_CFITSIO
    // 优先使用CFITSIO（如果可用）
    if (FitsReader::isCFITSIOAvailable()) {
        return writeFitsFileCFITSIO(filePath, data, width, height, bitpix, blank, bzero, bscale);
    }
#endif
    
#ifndef USE_CFITSIO
    // 简化实现（无CFITSIO时使用）
    return writeFitsFileSimple(filePath, data, width, height, bitpix, blank, bzero, bscale);
#else
    // CFITSIO 定义了但 isCFITSIOAvailable 返回 false 的情况不应该发生
    return false;
#endif
}

#ifdef USE_CFITSIO

/**
 * 使用CFITSIO写入FITS文件（完整实现）
 */
bool FitsWriter::writeFitsFileCFITSIO(
    const std::string& filePath,
    const float* data,
    int width, int height,
    int bitpix,
    double blank,
    double bzero,
    double bscale
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    // 创建或覆盖文件（'!'表示覆盖已存在文件）
    std::string fileFmt = "!" + filePath;
    if (fits_create_file(&fptr, fileFmt.c_str(), &status)) {
        fits_report_error(stderr, status);
        return false;
    }
    
    // 创建primary image HDU
    long naxes[2] = { width, height };
    if (fits_create_img(fptr, bitpix, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return false;
    }
    
    // 写入基本关键字
    if (blank != 0.0 && bitpix > 0) {  // 整数类型支持BLANK
        long blankValue = (long)blank;
        if (fits_set_hdustruc(fptr, &status)) {  // 设置HDU结构
            fits_report_error(stderr, status);
        }
        if (fits_update_key(fptr, TLONG, "BLANK", &blankValue, "Value for blank pixels", &status)) {
            fits_report_error(stderr, status);
        }
    }
    
    // 写入BZERO和BSCALE（如果有）
    if (bzero != 0.0) {
        if (fits_update_key(fptr, TDOUBLE, "BZERO", &bzero, "Zero point for data scaling", &status)) {
            fits_report_error(stderr, status);
        }
    }
    
    if (bscale != 1.0) {
        if (fits_update_key(fptr, TDOUBLE, "BSCALE", &bscale, "Data scaling factor", &status)) {
            fits_report_error(stderr, status);
        }
    }
    
    // 写入像素数据（批量写入，优化性能）
    long fpixel[2] = {1, 1};  // 起始像素（FITS从1开始）
    long nelements = (long)width * height;
    
    // 根据BITPIX选择数据类型
    int datatype = TFLOAT;  // 默认float
    if (bitpix == -64) {
        datatype = TDOUBLE;  // double
    } else if (bitpix == 32) {
        datatype = TINT;  // int
    } else if (bitpix == 16) {
        datatype = TSHORT;  // short
    } else if (bitpix == 8) {
        datatype = TBYTE;  // byte
    }
    
    // 如果有类型转换需要，先转换数据
    if (datatype == TFLOAT) {
        // float类型，直接写入
        if (fits_write_pix(fptr, datatype, fpixel, nelements, const_cast<float*>(data), &status)) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return false;
        }
    } else {
        // 其他类型，需要转换
        // 简化处理：先写入为float，让CFITSIO自动转换
        if (fits_write_pix(fptr, TFLOAT, fpixel, nelements, const_cast<float*>(data), &status)) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return false;
        }
    }
    
    // 关闭文件
    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        return false;
    }
    
    return true;
}

/**
 * Z-Order Curve (Morton code) 解码: hash -> (i, j)
 * 从 Java Tile2HPX.FC.hash2ij 翻译
 */
static void hash2ij(long h, int& i, int& j) {
    h = (0x2222222222222222L & h) <<  1
      | (0x4444444444444444L & h) >> 1
      | (0x9999999999999999L & h);
    h = (0x0C0C0C0C0C0C0C0CL & h) <<  2
      | (0x3030303030303030L & h) >> 2
      | (0xC3C3C3C3C3C3C3C3L & h);
    h = (0x00F000F000F000F0L & h) <<  4
      | (0x0F000F000F000F00L & h) >> 4
      | (0xF00FF00FF00FF00FL & h);
    h = (0x0000FF000000FF00L & h) <<  8
      | (0x00FF000000FF0000L & h) >> 8
      | (0xFF0000FFFF0000FFL & h);
    h = (0x00000000FFFF0000L & h) <<  16
      | (0x0000FFFF00000000L & h) >> 16
      | (0xFFFF00000000FFFFL & h);
    i = (int)h;
    j = (int)(h >> 32);
}

/**
 * 计算 tile 中心在 HPX 投影平面的坐标
 * 从 Java Tile2HPX.center 翻译
 */
static void computeTileCenter(int order, long npix, double& centerX, double& centerY) {
    const double PI_OVER_FOUR = 0.25 * M_PI;
    int nsideTile = 1 << order;
    long xyMask = (1L << (order << 1)) - 1;
    
    // Pull apart the hash elements
    int d0h = (int)(npix >> (order << 1));
    long localHash = npix & xyMask;
    
    int iInD0h, jInD0h;
    hash2ij(localHash, iInD0h, jInD0h);
    
    // Compute coordinates from the center of the base pixel with x-axis = W-->E, y-axis = S-->N
    int lInD0h = iInD0h - jInD0h;
    int hInD0h = iInD0h + jInD0h - (nsideTile - 1);
    
    // Compute coordinates of the base pixel in the projection plane
    int d0hBy4Quotient = d0h >> 2;
    int d0hMod4 = d0h - (d0hBy4Quotient << 2);
    int hD0h = 1 - d0hBy4Quotient;
    int lD0h = d0hMod4 << 1;
    
    if ((hD0h == 0 && (lD0h == 6 || (lD0h == 4 && lInD0h > 0)))   // case equatorial region
        || (hD0h != 0 && ++lD0h > 3)) {                           // case polar caps regions
        lD0h -= 8;
    }
    
    // Finalize computation (in radians)
    centerX = PI_OVER_FOUR * (lD0h + lInD0h / (double)nsideTile);
    centerY = PI_OVER_FOUR * (hD0h + hInD0h / (double)nsideTile);
}

/**
 * 计算 HiPS tile 在 HPX 投影平面中的 CRPIX 值
 * 基于 Java Tile2HPX.toFitsHeader 算法
 * 
 * HPX 投影参数: H=4, K=3, CRVAL=(0,0)
 */
static void computeHpxCRPIX(int order, long npix, int tileWidth, double& crpix1, double& crpix2) {
    // 计算 tile 中心在 HPX 投影平面的坐标 (弧度)
    double centerXRad, centerYRad;
    computeTileCenter(order, npix, centerXRad, centerYRad);
    
    // 转换为度
    double centreX = centerXRad * 180.0 / M_PI;
    double centreY = centerYRad * 180.0 / M_PI;
    
    // 计算像素尺度
    int nsideTile = 1 << order;
    int nsidePix = nsideTile * tileWidth;
    double scale = 45.0 / nsidePix;
    
    // 计算 CRPIX (从 Java Tile2HPX.toFitsHeader)
    // crPix1 = +((inNside + 1) / 2.0) - 0.5 * (-centreX / scale + centreY / scale)
    // crPix2 = +((inNside + 1) / 2.0) - 0.5 * (-centreX / scale - centreY / scale)
    double halfTile = (tileWidth + 1) / 2.0;
    crpix1 = halfTile - 0.5 * (-centreX / scale + centreY / scale);
    crpix2 = halfTile - 0.5 * (-centreX / scale - centreY / scale);
}

/**
 * 添加WCS头信息（用于HiPS tile）
 * 使用标准 HiPS HPX 投影格式
 */
void FitsWriter::addWCSHeaders(fitsfile* fptr, int order, long npix, const std::string& frame) {
    int status = 0;
    int tileWidth = 512;
    
    // 计算 CRPIX
    double crpix1, crpix2;
    computeHpxCRPIX(order, npix, tileWidth, crpix1, crpix2);
    
    // CRVAL 固定为 (0, 0)
    double crval1 = 0.0;
    double crval2 = 0.0;
    
    // CD 矩阵: 45度旋转 + 缩放
    int nside = 1 << order;
    double cdelt = 45.0 / (nside * tileWidth);
    double cd1_1 = -cdelt;
    double cd1_2 = -cdelt;
    double cd2_1 = cdelt;
    double cd2_2 = -cdelt;
    
    // 写入 WCS 关键字
    int order_key = order;
    long npix_key = npix;
    fits_update_key(fptr, TINT, "ORDER", &order_key, "HEALPix order", &status);
    fits_update_key(fptr, TLONG, "NPIX", &npix_key, "HEALPix pixel index", &status);
    
    fits_update_key(fptr, TDOUBLE, "CRPIX1", &crpix1, nullptr, &status);
    fits_update_key(fptr, TDOUBLE, "CRPIX2", &crpix2, nullptr, &status);
    
    fits_update_key(fptr, TDOUBLE, "CD1_1", &cd1_1, nullptr, &status);
    fits_update_key(fptr, TDOUBLE, "CD1_2", &cd1_2, nullptr, &status);
    fits_update_key(fptr, TDOUBLE, "CD2_1", &cd2_1, nullptr, &status);
    fits_update_key(fptr, TDOUBLE, "CD2_2", &cd2_2, nullptr, &status);
    
    char ctype1[] = "RA---HPX";
    char ctype2[] = "DEC--HPX";
    fits_update_key(fptr, TSTRING, "CTYPE1", ctype1, nullptr, &status);
    fits_update_key(fptr, TSTRING, "CTYPE2", ctype2, nullptr, &status);
    
    fits_update_key(fptr, TDOUBLE, "CRVAL1", &crval1, nullptr, &status);
    fits_update_key(fptr, TDOUBLE, "CRVAL2", &crval2, nullptr, &status);
    
    int pv2_1 = 4;
    int pv2_2 = 3;
    fits_update_key(fptr, TINT, "PV2_1", &pv2_1, nullptr, &status);
    fits_update_key(fptr, TINT, "PV2_2", &pv2_2, nullptr, &status);
}

#else  // 无CFITSIO，使用简化实现

/**
 * 简化FITS写入（无CFITSIO时使用，仅支持基本格式）
 */
bool FitsWriter::writeFitsFileSimple(
    const std::string& filePath,
    const float* data,
    int width, int height,
    int bitpix,
    double blank,
    double bzero,
    double bscale
) {
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filePath << std::endl;
        return false;
    }
    
    // 简化实现：写入基本FITS格式
    // 注意：这是一个非常简化的实现，仅用于测试
    // 实际生产环境应该使用CFITSIO
    
    // FITS头（2880字节）
    char header[2880];
    memset(header, ' ', 2880);  // 用空格填充
    
    // 辅助函数：写入一个80字节的FITS卡片
    auto writeCard = [&header](int pos, const char* keyword, const char* value, const char* comment) {
        char card[81];
        memset(card, ' ', 80);
        card[80] = '\0';
        
        // 写入关键字（前8个字符）
        int keyLen = strlen(keyword);
        memcpy(card, keyword, keyLen < 8 ? keyLen : 8);
        
        // 写入等号（第9个字符）
        card[8] = '=';
        card[9] = ' ';
        
        // 写入值（右对齐到第30个字符）
        int valLen = strlen(value);
        int valStart = 30 - valLen;
        if (valStart < 10) valStart = 10;
        memcpy(card + valStart, value, valLen);
        
        // 写入注释（第32个字符开始）
        if (comment && strlen(comment) > 0) {
            card[31] = '/';
            card[32] = ' ';
            int cmtLen = strlen(comment);
            if (cmtLen > 47) cmtLen = 47;
            memcpy(card + 33, comment, cmtLen);
        }
        
        memcpy(header + pos, card, 80);
    };
    
    int pos = 0;
    
    // SIMPLE = T
    writeCard(pos, "SIMPLE", "T", "file does conform to FITS standard");
    pos += 80;
    
    // BITPIX
    char bitpixStr[32];
    snprintf(bitpixStr, sizeof(bitpixStr), "%d", bitpix);
    writeCard(pos, "BITPIX", bitpixStr, "number of bits per data pixel");
    pos += 80;
    
    // NAXIS
    writeCard(pos, "NAXIS", "2", "number of data axes");
    pos += 80;
    
    // NAXIS1
    char widthStr[32];
    snprintf(widthStr, sizeof(widthStr), "%d", width);
    writeCard(pos, "NAXIS1", widthStr, "length of data axis 1");
    pos += 80;
    
    // NAXIS2
    char heightStr[32];
    snprintf(heightStr, sizeof(heightStr), "%d", height);
    writeCard(pos, "NAXIS2", heightStr, "length of data axis 2");
    pos += 80;
    
    // END
    memcpy(header + pos, "END", 3);
    
    // 写入头（补足到2880字节）
    out.write(header, 2880);
    
    // 写入像素数据（简化：只支持float）
    // 注意：FITS标准要求大端序（big-endian）
    if (bitpix == -32) {
        // 转换字节序：小端序 -> 大端序
        int npixels = width * height;
        std::vector<char> bigEndianData(npixels * sizeof(float));
        
        for (int i = 0; i < npixels; i++) {
            // 获取当前float的字节
            const unsigned char* src = reinterpret_cast<const unsigned char*>(&data[i]);
            unsigned char* dst = reinterpret_cast<unsigned char*>(&bigEndianData[i * sizeof(float)]);
            
            // 反转字节序（小端序 -> 大端序）
            dst[0] = src[3];
            dst[1] = src[2];
            dst[2] = src[1];
            dst[3] = src[0];
        }
        
        out.write(bigEndianData.data(), npixels * sizeof(float));
    } else {
        // 其他格式需要转换（简化实现不支持）
        std::cerr << "Warning: Simplified FITS writer only supports BITPIX=-32 (float)" << std::endl;
        out.close();
        return false;
    }
    
    // 补足到2880字节边界
    long dataSize = width * height * sizeof(float);
    long padding = (2880 - (dataSize % 2880)) % 2880;
    if (padding > 0) {
        char pad[2880] = {0};
        out.write(pad, padding);
    }
    
    out.close();
    return true;
}

#endif  // USE_CFITSIO
