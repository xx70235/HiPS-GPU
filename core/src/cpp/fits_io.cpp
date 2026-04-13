/**
 * FITS文件读取实现
 * 支持CFITSIO库和简化实现
 */

#include "fits_io.h"
#include "fits_hdu_utils.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>

// 尝试包含CFITSIO头文件（如果可用）
#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

// FITS文件格式常量
const int FITS_BLOCK_SIZE = 2880;  // FITS块大小（字节）
const int FITS_CARD_SIZE = 80;     // FITS卡片大小（80字节）

/**
 * 检查CFITSIO库是否可用
 */
bool FitsReader::isCFITSIOAvailable() {
#ifdef USE_CFITSIO
    return true;
#else
    return false;
#endif
}

/**
 * 读取FITS文件（自动选择最佳方法）
 */
FitsData FitsReader::readFitsFile(const std::string& filename, int hdu) {
    // 优先使用CFITSIO（如果可用）
    if (isCFITSIOAvailable()) {
        return readFitsFileCFITSIO(filename, hdu);
    } else {
        // 使用简化实现
        return readFitsFileSimple(filename, hdu);
    }
}

/**
 * 使用CFITSIO库读取FITS文件
 */
FitsData FitsReader::readFitsFileCFITSIO(const std::string& filename, int hdu) {
    FitsData data;
    data.filename = filename;
    
#ifdef USE_CFITSIO
    fitsfile* fptr = nullptr;
    int status = 0;
    int hdutype = 0;
    long naxes[3] = {1, 1, 1};
    int naxis = 0;
    
    
    // 打开FITS文件
    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        data.errorMessage = "Failed to open FITS file";
        data.isValid = false;
        return data;
    }
    
    HduSelectionResult selection = select_valid_image_hdu(fptr, filename, hdu);
    if (!selection.found) {
        fits_close_file(fptr, &status);
        data.errorMessage = selection.error;
        data.isValid = false;
        return data;
    }
    
    // 移动到找到的有效 HDU
    status = 0;
    if (fits_movabs_hdu(fptr, selection.hdu_index_1_based, &hdutype, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to move to HDU";
        data.isValid = false;
        return data;
    }
    
    // 读取图像尺寸
    if (fits_get_img_dim(fptr, &naxis, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to get image dimensions";
        data.isValid = false;
        return data;
    }
    
    if (fits_get_img_size(fptr, 3, naxes, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to get image size";
        data.isValid = false;
        return data;
    }
    
    if (naxis >= 2) {
        data.width = (int)naxes[0];
        data.height = (int)naxes[1];
        data.depth = (naxis >= 3) ? (int)naxes[2] : 1;
    }
    
    // 读取BITPIX
    int bitpix = 0;
    if (fits_get_img_type(fptr, &bitpix, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to get BITPIX";
        data.isValid = false;
        return data;
    }
    data.bitpix = bitpix;
    
    // 读取WCS参数 - 使用单独的 status 避免错误累积
    double crval1 = 0.0, crval2 = 0.0;
    double crpix1 = 0.0, crpix2 = 0.0;
    double cd1_1 = 0.0, cd1_2 = 0.0, cd2_1 = 0.0, cd2_2 = 0.0;
    char ctype1[80] = "", ctype2[80] = "";
    int wcsStatus = 0;
    
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRVAL2", &crval2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRPIX2", &crpix2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD1_1", &cd1_1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD1_2", &cd1_2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD2_1", &cd2_1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD2_2", &cd2_2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TSTRING, "CTYPE1", ctype1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TSTRING, "CTYPE2", ctype2, nullptr, &wcsStatus);
    
    data.crval1 = crval1;
    data.crval2 = crval2;
    data.crpix1 = crpix1;
    data.crpix2 = crpix2;
    data.cd1_1 = cd1_1;
    data.cd1_2 = cd1_2;
    data.cd2_1 = cd2_1;
    data.cd2_2 = cd2_2;
    data.ctype1 = std::string(ctype1);
    data.ctype2 = std::string(ctype2);
    
    // 如果 CD 矩阵为零，尝试从 CDELT 和 CROTA 推导
    if (data.cd1_1 == 0.0 && data.cd1_2 == 0.0 && data.cd2_1 == 0.0 && data.cd2_2 == 0.0) {
        double cdelt1 = 0.0, cdelt2 = 0.0, crota2 = 0.0;
        wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, nullptr, &wcsStatus);
        wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, nullptr, &wcsStatus);
        wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CROTA2", &crota2, nullptr, &wcsStatus);
        if (wcsStatus != 0) {
            wcsStatus = 0;
            fits_read_key(fptr, TDOUBLE, "CROTA1", &crota2, nullptr, &wcsStatus);
        }
        if (cdelt1 != 0.0 && cdelt2 != 0.0) {
            double crota_rad = crota2 * M_PI / 180.0;
            data.cd1_1 = cdelt1 * cos(crota_rad);
            data.cd1_2 = -cdelt2 * sin(crota_rad);
            data.cd2_1 = cdelt1 * sin(crota_rad);
            data.cd2_2 = cdelt2 * cos(crota_rad);
        }
    }
    
    // 读取BZERO和BSCALE（可选字段，使用单独的 status）
    int bzeroStatus = 0;
    fits_read_key(fptr, TDOUBLE, "BZERO", &data.bzero, nullptr, &bzeroStatus);
    bzeroStatus = 0;
    fits_read_key(fptr, TDOUBLE, "BSCALE", &data.bscale, nullptr, &bzeroStatus);
    if (data.bscale == 0.0) data.bscale = 1.0;
    
    // 重置 status 以便读取像素数据
    status = 0;
    
    // 读取像素数据
    long npixels = (long)data.width * data.height * data.depth;
    std::vector<double> tempPixels(npixels);
    
    // Always use TDOUBLE to read into double array - cfitsio will auto-convert
    if (fits_read_img(fptr, TDOUBLE, 1, npixels, nullptr, tempPixels.data(), nullptr, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to read image data";
        data.isValid = false;
        return data;
    }
    
    // 转换为float数组
    data.pixels.resize(npixels);
    for (long i = 0; i < npixels; i++) {
        data.pixels[i] = (float)tempPixels[i];
    }
    
    // 应用BZERO和BSCALE
    if (data.bzero != 0.0 || data.bscale != 1.0) {
        for (size_t i = 0; i < data.pixels.size(); i++) {
            data.pixels[i] = data.pixels[i] * data.bscale + data.bzero;
        }
    }
    
    fits_close_file(fptr, &status);
    data.isValid = true;
    
#else
    // CFITSIO不可用，使用简化实现
    data = readFitsFileSimple(filename, hdu);
#endif
    
    return data;
}

FitsData FitsReader::readFitsCutout(const std::string& filename, const PixelBounds& bounds, int hdu) {
    FitsData data;
    data.filename = filename;

    if (bounds.width <= 0 || bounds.height <= 0) {
        data.isValid = false;
        data.errorMessage = "Invalid cutout bounds";
        return data;
    }

#ifdef USE_CFITSIO
    fitsfile* fptr = nullptr;
    int status = 0;
    int hdutype = 0;
    long naxes[3] = {1, 1, 1};
    int naxis = 0;

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        data.errorMessage = "Failed to open FITS file";
        data.isValid = false;
        return data;
    }

    HduSelectionResult selection = select_valid_image_hdu(fptr, filename, hdu);
    if (!selection.found) {
        fits_close_file(fptr, &status);
        data.errorMessage = selection.error;
        data.isValid = false;
        return data;
    }

    status = 0;
    if (fits_movabs_hdu(fptr, selection.hdu_index_1_based, &hdutype, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to move to HDU";
        data.isValid = false;
        return data;
    }

    if (fits_get_img_dim(fptr, &naxis, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to get image dimensions";
        data.isValid = false;
        return data;
    }

    if (fits_get_img_size(fptr, 3, naxes, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to get image size";
        data.isValid = false;
        return data;
    }

    if (naxis < 2) {
        fits_close_file(fptr, &status);
        data.errorMessage = "FITS image has fewer than 2 dimensions";
        data.isValid = false;
        return data;
    }

    const int fullWidth = static_cast<int>(naxes[0]);
    const int fullHeight = static_cast<int>(naxes[1]);
    const int depth = (naxis >= 3) ? static_cast<int>(naxes[2]) : 1;

    const int x0 = std::max(0, bounds.x0);
    const int y0 = std::max(0, bounds.y0);
    const int x1 = std::min(fullWidth, bounds.x0 + bounds.width);
    const int y1 = std::min(fullHeight, bounds.y0 + bounds.height);

    if (x0 >= x1 || y0 >= y1) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Cutout bounds outside image";
        data.isValid = false;
        return data;
    }

    data.width = x1 - x0;
    data.height = y1 - y0;
    data.depth = depth;

    int bitpix = 0;
    if (fits_get_img_type(fptr, &bitpix, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to get BITPIX";
        data.isValid = false;
        return data;
    }
    data.bitpix = bitpix;

    double crval1 = 0.0, crval2 = 0.0;
    double crpix1 = 0.0, crpix2 = 0.0;
    double cd1_1 = 0.0, cd1_2 = 0.0, cd2_1 = 0.0, cd2_2 = 0.0;
    char ctype1[80] = "", ctype2[80] = "";
    int wcsStatus = 0;

    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRVAL2", &crval2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CRPIX2", &crpix2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD1_1", &cd1_1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD1_2", &cd1_2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD2_1", &cd2_1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CD2_2", &cd2_2, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TSTRING, "CTYPE1", ctype1, nullptr, &wcsStatus);
    wcsStatus = 0; fits_read_key(fptr, TSTRING, "CTYPE2", ctype2, nullptr, &wcsStatus);

    data.crval1 = crval1;
    data.crval2 = crval2;
    data.crpix1 = crpix1 - x0;
    data.crpix2 = crpix2 - y0;
    data.cd1_1 = cd1_1;
    data.cd1_2 = cd1_2;
    data.cd2_1 = cd2_1;
    data.cd2_2 = cd2_2;
    data.ctype1 = std::string(ctype1);
    data.ctype2 = std::string(ctype2);

    if (data.cd1_1 == 0.0 && data.cd1_2 == 0.0 && data.cd2_1 == 0.0 && data.cd2_2 == 0.0) {
        double cdelt1 = 0.0, cdelt2 = 0.0, crota2 = 0.0;
        wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, nullptr, &wcsStatus);
        wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, nullptr, &wcsStatus);
        wcsStatus = 0; fits_read_key(fptr, TDOUBLE, "CROTA2", &crota2, nullptr, &wcsStatus);
        if (wcsStatus != 0) {
            wcsStatus = 0;
            fits_read_key(fptr, TDOUBLE, "CROTA1", &crota2, nullptr, &wcsStatus);
        }
        if (cdelt1 != 0.0 && cdelt2 != 0.0) {
            double crota_rad = crota2 * M_PI / 180.0;
            data.cd1_1 = cdelt1 * cos(crota_rad);
            data.cd1_2 = -cdelt2 * sin(crota_rad);
            data.cd2_1 = cdelt1 * sin(crota_rad);
            data.cd2_2 = cdelt2 * cos(crota_rad);
        }
    }

    int bzeroStatus = 0;
    fits_read_key(fptr, TDOUBLE, "BZERO", &data.bzero, nullptr, &bzeroStatus);
    bzeroStatus = 0;
    fits_read_key(fptr, TDOUBLE, "BSCALE", &data.bscale, nullptr, &bzeroStatus);
    if (data.bscale == 0.0) data.bscale = 1.0;

    status = 0;
    long fpixel[3] = {static_cast<long>(x0 + 1), static_cast<long>(y0 + 1), 1};
    long lpixel[3] = {static_cast<long>(x1), static_cast<long>(y1), static_cast<long>(depth)};
    long inc[3] = {1, 1, 1};
    long npixels = static_cast<long>(data.width) * data.height * data.depth;
    std::vector<double> tempPixels(npixels);

    if (fits_read_subset(fptr, TDOUBLE, fpixel, lpixel, inc, nullptr, tempPixels.data(), nullptr, &status)) {
        fits_close_file(fptr, &status);
        data.errorMessage = "Failed to read cutout image data";
        data.isValid = false;
        return data;
    }

    data.pixels.resize(npixels);
    for (long i = 0; i < npixels; ++i) {
        data.pixels[i] = static_cast<float>(tempPixels[i]);
    }

    if (data.bzero != 0.0 || data.bscale != 1.0) {
        for (size_t i = 0; i < data.pixels.size(); ++i) {
            data.pixels[i] = data.pixels[i] * data.bscale + data.bzero;
        }
    }

    fits_close_file(fptr, &status);
    data.isValid = true;
    return data;
#else
    FitsData full = readFitsFileSimple(filename, hdu);
    if (!full.isValid) {
        return full;
    }

    const int x0 = std::max(0, bounds.x0);
    const int y0 = std::max(0, bounds.y0);
    const int x1 = std::min(full.width, bounds.x0 + bounds.width);
    const int y1 = std::min(full.height, bounds.y0 + bounds.height);
    if (x0 >= x1 || y0 >= y1) {
        full.isValid = false;
        full.errorMessage = "Cutout bounds outside image";
        return full;
    }

    FitsData cutout = full;
    cutout.width = x1 - x0;
    cutout.height = y1 - y0;
    cutout.crpix1 = full.crpix1 - x0;
    cutout.crpix2 = full.crpix2 - y0;
    cutout.pixels.assign(static_cast<size_t>(cutout.width) * cutout.height * cutout.depth, 0.0f);

    for (int z = 0; z < cutout.depth; ++z) {
        for (int y = 0; y < cutout.height; ++y) {
            for (int x = 0; x < cutout.width; ++x) {
                size_t srcIdx = (static_cast<size_t>(z) * full.height + (y0 + y)) * full.width + (x0 + x);
                size_t dstIdx = (static_cast<size_t>(z) * cutout.height + y) * cutout.width + x;
                cutout.pixels[dstIdx] = full.pixels[srcIdx];
            }
        }
    }
    return cutout;
#endif
}

/**
 * 使用简化实现读取FITS文件
 * 注意：这是一个简化实现，仅支持基本的FITS格式
 */
FitsData FitsReader::readFitsFileSimple(const std::string& filename, int hdu) {
    FitsData data;
    data.filename = filename;
    
    // 读取FITS头
    if (readFitsHeaderSimple(filename, data) != 0) {
        data.isValid = false;
        data.errorMessage = "Failed to read FITS header";
        return data;
    }
    
    // 读取FITS数据
    if (readFitsDataSimple(filename, data, data.pixels) != 0) {
        data.isValid = false;
        data.errorMessage = "Failed to read FITS data";
        return data;
    }
    
    data.isValid = true;
    return data;
}

/**
 * 简化实现：读取FITS头
 */
int FitsReader::readFitsHeaderSimple(const std::string& filename, FitsData& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return -1;
    }
    
    // 读取FITS主头（可能多个块，直到遇到END关键字）
    // FITS头由多个2880字节的块组成，每个关键字80字节
    std::string headerStr;
    std::vector<char> headerBlock(FITS_BLOCK_SIZE);
    bool foundEnd = false;
    int blocksRead = 0;
    // 增大MAX_HEADER_BLOCKS以支持更大的FITS头
    // 某些FITS文件（如EP卫星数据）的WCS关键字可能在第13-14个块中
    const int MAX_HEADER_BLOCKS = 50;  // 最多读取50个块（约1800个关键字）
    
    while (!foundEnd && blocksRead < MAX_HEADER_BLOCKS) {
        file.read(headerBlock.data(), FITS_BLOCK_SIZE);
        std::streamsize bytesRead = file.gcount();
        
        if (bytesRead < FITS_BLOCK_SIZE) {
            break;  // 文件结束
        }
        
        // 追加到headerStr
        std::string blockStr(headerBlock.begin(), headerBlock.end());
        headerStr.append(blockStr);
        
        // 检查是否包含END关键字（80字节行格式）
        for (size_t i = 0; i < blockStr.length(); i += FITS_CARD_SIZE) {
            if (i + 8 <= blockStr.length()) {
                std::string key = blockStr.substr(i, 8);
                // 去除尾部空格
                while (!key.empty() && key.back() == ' ') {
                    key.pop_back();
                }
                if (key == "END") {
                    foundEnd = true;
                    // 截取到END关键字的位置（包括END行）
                    size_t endPos = headerStr.find("END     ");
                    if (endPos != std::string::npos) {
                        headerStr = headerStr.substr(0, endPos + FITS_CARD_SIZE);
                    }
                    break;
                }
            }
        }
        
        if (foundEnd) break;
        blocksRead++;
    }
    
    // 保存主头的块数（用于后续计算主头大小）
    int mainHeaderBlocks = blocksRead;
    
    // 保存主头字符串（用于后续提取基本信息）
    std::string mainHeaderStr = headerStr;
    
    // 先提取基本参数（用于后续可能的数据区域跳过）
    std::string naxisStr = getKeywordValue(mainHeaderStr, "NAXIS");
    int naxis = !naxisStr.empty() ? std::stoi(naxisStr) : 0;
    
    // 检查主头是否有WCS
    std::string crval1Str = getKeywordValue(mainHeaderStr, "CRVAL1");
    std::string cd1_1Str = getKeywordValue(mainHeaderStr, "CD1_1");
    std::string cdelt1Str = getKeywordValue(mainHeaderStr, "CDELT1");
    bool hasWCS = !crval1Str.empty() || !cd1_1Str.empty() || !cdelt1Str.empty();
    
    // 如果主头没有WCS，尝试读取扩展头（HDU 1）
    if (!hasWCS && naxis > 0) {
        // 计算主头数据区域的大小并跳过
        int bitpix = -32;  // 默认
        std::string bitpixStr = getKeywordValue(mainHeaderStr, "BITPIX");
        if (!bitpixStr.empty()) {
            bitpix = std::stoi(bitpixStr);
        }
        
        int bytesPerPixel = (bitpix > 0) ? (bitpix / 8) : ((-bitpix) / 8);
        if (bytesPerPixel == 0) bytesPerPixel = 4;  // 默认float
        
        long naxis1 = 0, naxis2 = 0;
        std::string naxis1Str = getKeywordValue(mainHeaderStr, "NAXIS1");
        std::string naxis2Str = getKeywordValue(mainHeaderStr, "NAXIS2");
        if (!naxis1Str.empty()) naxis1 = std::stol(naxis1Str);
        if (!naxis2Str.empty()) naxis2 = std::stol(naxis2Str);
        
        // 计算主头的实际长度（已读取的块数 * 2880）
        // 注意：我们已经读取了mainHeaderBlocks个2880字节的块
        // mainHeaderBlocks是在读取主头时累计的块数
        long mainHeaderSize = mainHeaderBlocks * FITS_BLOCK_SIZE;
        
        // 但headerStr可能已经截取到END关键字，所以需要对齐到2880字节边界
        // 实际上，我们已经读取了完整的块，所以mainHeaderSize就是已读取的字节数
        // 如果headerStr被截取，我们需要确保mainHeaderSize至少包含END关键字所在的位置
        if (foundEnd) {
            // 找到END，确保mainHeaderSize至少包含END所在的位置
            long headerStrLen = mainHeaderStr.length();
            long headerStrSize = ((headerStrLen + FITS_BLOCK_SIZE - 1) / FITS_BLOCK_SIZE) * FITS_BLOCK_SIZE;
            // 使用较大的值（应该相同，但以防万一）
            if (headerStrSize > mainHeaderSize) {
                mainHeaderSize = headerStrSize;
            }
        }
        
        // 计算数据大小（字节数，向上取整到2880的倍数）
        // 参考原版moveOnHUD的逻辑：taille = naxis1 * naxis2 * ... * bitpix/8
        long dataSize = naxis1 * naxis2 * bytesPerPixel;
        long paddedDataSize = ((dataSize + FITS_BLOCK_SIZE - 1) / FITS_BLOCK_SIZE) * FITS_BLOCK_SIZE;
        
        // 跳过主头数据区域，准备读取扩展头
        // 文件当前位置：已经读取了mainHeaderSize字节（主头）
        // 需要跳过：主头数据区域（paddedDataSize）
        // 目标位置：文件开头 + mainHeaderSize + paddedDataSize
        // 但注意：我们已经读取了mainHeaderSize字节，所以当前位置可能已经在mainHeaderSize之后
        // 需要重新定位到主头之后的数据区域末尾
        long targetPos = mainHeaderSize + paddedDataSize;
        file.seekg(targetPos, std::ios::beg);
        
        // 尝试读取扩展头（HDU 1）
        if (file.good() && !file.eof()) {
            std::string extHeaderStr;
            foundEnd = false;
            int extBlocksRead = 0;  // 扩展头的块计数器
            
            while (!foundEnd && extBlocksRead < MAX_HEADER_BLOCKS) {
                file.read(headerBlock.data(), FITS_BLOCK_SIZE);
                std::streamsize bytesRead = file.gcount();
                
                if (bytesRead < FITS_BLOCK_SIZE) {
                    break;  // 文件结束
                }
                
                std::string blockStr(headerBlock.begin(), headerBlock.end());
                extHeaderStr.append(blockStr);
                
                // 检查END关键字
                for (size_t i = 0; i < blockStr.length(); i += FITS_CARD_SIZE) {
                    if (i + 8 <= blockStr.length()) {
                        std::string key = blockStr.substr(i, 8);
                        while (!key.empty() && key.back() == ' ') {
                            key.pop_back();
                        }
                        if (key == "END") {
                            foundEnd = true;
                            size_t endPos = extHeaderStr.find("END     ");
                            if (endPos != std::string::npos) {
                                extHeaderStr = extHeaderStr.substr(0, endPos + FITS_CARD_SIZE);
                            }
                            break;
                        }
                    }
                }
                
                if (foundEnd) break;
                extBlocksRead++;
            }
            
            // 如果扩展头有WCS关键字，使用扩展头的头信息
            if (!extHeaderStr.empty()) {
                std::string extCrval1Str = getKeywordValue(extHeaderStr, "CRVAL1");
                std::string extCd1_1Str = getKeywordValue(extHeaderStr, "CD1_1");
                std::string extCdelt1Str = getKeywordValue(extHeaderStr, "CDELT1");
                
                if (!extCrval1Str.empty() || !extCd1_1Str.empty() || !extCdelt1Str.empty()) {
                    // 扩展头有WCS，将扩展头的WCS关键字追加到主头字符串中
                    // 这样后续可以从headerStr中读取WCS参数，同时保持主头的基本信息
                    // 由于扩展头的WCS关键字优先级更高，我们直接合并两个头字符串
                    // 但要注意：提取参数时会先查找，所以扩展头的关键字会覆盖主头的
                    // 简单做法：在headerStr中追加扩展头的WCS相关行
                    headerStr += extHeaderStr;
                    hasWCS = true;
                }
            }
        }
    }
    
    file.close();
    
    // 提取基本参数
    data.width = std::stoi(getKeywordValue(headerStr, "NAXIS1"));
    data.height = std::stoi(getKeywordValue(headerStr, "NAXIS2"));
    
    std::string naxis3 = getKeywordValue(headerStr, "NAXIS3");
    if (naxis3.empty()) naxis3 = getKeywordValue(mainHeaderStr, "NAXIS3");
    if (!naxis3.empty()) {
        data.depth = std::stoi(naxis3);
    }
    
    std::string bitpixStr = getKeywordValue(headerStr, "BITPIX");
    if (bitpixStr.empty()) bitpixStr = getKeywordValue(mainHeaderStr, "BITPIX");
    if (!bitpixStr.empty()) {
        data.bitpix = std::stoi(bitpixStr);
    }
    
    // 提取WCS参数（使用最终的headerStr，可能是主头或扩展头）
    std::string crval1StrFinal = getKeywordValue(headerStr, "CRVAL1");
    if (!crval1StrFinal.empty()) data.crval1 = std::stod(crval1StrFinal);
    
    std::string crval2Str = getKeywordValue(headerStr, "CRVAL2");
    if (!crval2Str.empty()) data.crval2 = std::stod(crval2Str);
    
    std::string crpix1Str = getKeywordValue(headerStr, "CRPIX1");
    if (!crpix1Str.empty()) data.crpix1 = std::stod(crpix1Str);
    
    std::string crpix2Str = getKeywordValue(headerStr, "CRPIX2");
    if (!crpix2Str.empty()) data.crpix2 = std::stod(crpix2Str);
    
    // 提取CD矩阵
    std::string cd1_1StrFinal = getKeywordValue(headerStr, "CD1_1");
    if (!cd1_1StrFinal.empty()) data.cd1_1 = std::stod(cd1_1StrFinal);
    
    std::string cd1_2Str = getKeywordValue(headerStr, "CD1_2");
    if (!cd1_2Str.empty()) data.cd1_2 = std::stod(cd1_2Str);
    
    std::string cd2_1Str = getKeywordValue(headerStr, "CD2_1");
    if (!cd2_1Str.empty()) data.cd2_1 = std::stod(cd2_1Str);
    
    std::string cd2_2Str = getKeywordValue(headerStr, "CD2_2");
    if (!cd2_2Str.empty()) data.cd2_2 = std::stod(cd2_2Str);
    
    // 如果CD矩阵为零，尝试从CDELT和CROTA推导
    if (data.cd1_1 == 0.0 && data.cd1_2 == 0.0 && data.cd2_1 == 0.0 && data.cd2_2 == 0.0) {
        std::string cdelt1Str = getKeywordValue(headerStr, "CDELT1");
        std::string cdelt2Str = getKeywordValue(headerStr, "CDELT2");
        std::string crota2Str = getKeywordValue(headerStr, "CROTA2");
        if (crota2Str.empty()) {
            crota2Str = getKeywordValue(headerStr, "CROTA1");
        }
        
        double cdelt1 = 0.0, cdelt2 = 0.0, crota = 0.0;
        if (!cdelt1Str.empty()) cdelt1 = std::stod(cdelt1Str);
        if (!cdelt2Str.empty()) cdelt2 = std::stod(cdelt2Str);
        if (!crota2Str.empty()) crota = std::stod(crota2Str);
        
        // 如果CDELT存在，构建CD矩阵（考虑CROTA旋转）
        if (cdelt1 != 0.0 || cdelt2 != 0.0) {
            // CD矩阵公式：
            // CD1_1 = CDELT1 * cos(CROTA)
            // CD1_2 = -CDELT2 * sin(CROTA)
            // CD2_1 = CDELT1 * sin(CROTA)
            // CD2_2 = CDELT2 * cos(CROTA)
            double crota_rad = crota * M_PI / 180.0;
            double cos_crota = cos(crota_rad);
            double sin_crota = sin(crota_rad);
            
            data.cd1_1 = cdelt1 * cos_crota;
            data.cd1_2 = -cdelt2 * sin_crota;
            data.cd2_1 = cdelt1 * sin_crota;
            data.cd2_2 = cdelt2 * cos_crota;
        }
    }
    
    // 提取CTYPE
    data.ctype1 = trim(getKeywordValue(headerStr, "CTYPE1"));
    data.ctype2 = trim(getKeywordValue(headerStr, "CTYPE2"));
    
    // 提取BZERO和BSCALE
    std::string bzeroStr = getKeywordValue(headerStr, "BZERO");
    if (!bzeroStr.empty()) data.bzero = std::stod(bzeroStr);
    
    std::string bscaleStr = getKeywordValue(headerStr, "BSCALE");
    if (!bscaleStr.empty()) data.bscale = std::stod(bscaleStr);
    if (data.bscale == 0.0) data.bscale = 1.0;
    
    // 提取BLANK (空白像素值)
    // 注意：data.blank默认为NaN，只有在FITS头中显式设置时才覆盖
    std::string blankStr = getKeywordValue(headerStr, "BLANK");
    if (!blankStr.empty()) {
        data.blank = std::stod(blankStr);
    }
    // 如果BLANK未设置，data.blank保持为NaN
    
    file.close();
    return 0;
}

/**
 * 简化实现：读取FITS数据
 * 注意：这是一个简化的实现，仅支持基本格式
 */
int FitsReader::readFitsDataSimple(const std::string& filename, const FitsData& header, std::vector<float>& pixels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return -1;
    }
    
    // 先读取头部找到END关键字，确定头部大小
    std::vector<char> block(FITS_BLOCK_SIZE);
    long headerSize = 0;
    bool foundEnd = false;
    
    while (!foundEnd && headerSize < FITS_BLOCK_SIZE * 100) {  // 最多读取100个块
        file.read(block.data(), FITS_BLOCK_SIZE);
        if (file.gcount() < FITS_BLOCK_SIZE) break;
        headerSize += FITS_BLOCK_SIZE;
        
        // 检查END关键字
        for (int i = 0; i < FITS_BLOCK_SIZE; i += 80) {
            if (block[i] == 'E' && block[i+1] == 'N' && block[i+2] == 'D' && 
                (block[i+3] == ' ' || block[i+3] == '\0')) {
                foundEnd = true;
                break;
            }
        }
    }
    
    if (!foundEnd) {
        file.close();
        return -1;  // 找不到END
    }
    
    // 数据区域从head头结束后开始（已经对齐到FITS_BLOCK_SIZE）
    // file已经在数据区域的开始位置
    
    // 计算像素数量
    long npixels = (long)header.width * header.height * header.depth;
    pixels.resize(npixels);
    
    // 根据BITPIX读取数据
    int bytesPerPixel = std::abs(header.bitpix) / 8;
    if (bytesPerPixel == 0) bytesPerPixel = 4;  // 默认float
    
    // 读取原始数据
    std::vector<char> rawData(npixels * bytesPerPixel);
    file.read(rawData.data(), rawData.size());
    
    if (file.gcount() < (std::streamsize)rawData.size()) {
        file.close();
        return -1;
    }
    
    // 转换为float（处理字节序：FITS是大端序）
    if (header.bitpix == -32) {  // float (big-endian)
        unsigned char* src = reinterpret_cast<unsigned char*>(rawData.data());
        for (long i = 0; i < npixels; i++) {
            // 大端序 -> 小端序
            unsigned char temp[4];
            temp[0] = src[i*4 + 3];
            temp[1] = src[i*4 + 2];
            temp[2] = src[i*4 + 1];
            temp[3] = src[i*4 + 0];
            memcpy(&pixels[i], temp, 4);
        }
    } else {
        // 其他格式需要更复杂的转换（暂时不支持）
        for (long i = 0; i < npixels; i++) {
            pixels[i] = 0.0f;
        }
    }
    
    // 应用BZERO和BSCALE
    if (header.bzero != 0.0 || header.bscale != 1.0) {
        for (size_t i = 0; i < pixels.size(); i++) {
            pixels[i] = pixels[i] * (float)header.bscale + (float)header.bzero;
        }
    }
    
    file.close();
    return 0;
}

/**
 * 从FITS头字符串中提取关键字值
 */
std::string FitsReader::getKeywordValue(const std::string& header, const std::string& keyword) {
    // FITS关键字格式：KEYWORD = 'value' / comment
    // 每个关键字80字节，需要逐行查找
    size_t headerLen = header.length();
    size_t keywordLen = keyword.length();
    
    // 逐行查找（每80字节一行）
    for (size_t i = 0; i < headerLen; i += FITS_CARD_SIZE) {
        // 确保不会越界
        if (i + FITS_CARD_SIZE > headerLen) break;
        
        // 提取当前关键字卡片（80字节）
        std::string card = header.substr(i, FITS_CARD_SIZE);
        
        // 检查关键字名称（前8个字符）
        std::string cardKeyword = card.substr(0, 8);
        
        // 移除尾部空格
        while (!cardKeyword.empty() && cardKeyword.back() == ' ') {
            cardKeyword.pop_back();
        }
        
        // 检查是否匹配
        if (cardKeyword == keyword) {
            // 找到匹配的关键字，提取值
            // FITS格式：KEYWORD = 'value' / comment
            // 第9个字符（索引8）应该是'='
            
            // 检查等号
            if (card.length() < 10 || card[8] != '=') {
                continue;  // 没有等号，继续查找
            }
            
            // 提取值（从第10个字符开始，索引9）
            size_t valueStart = 10;
            // 跳过等号后的空格
            while (valueStart < card.length() && card[valueStart] == ' ') {
                valueStart++;
            }
            
            if (valueStart >= card.length()) {
                return "";  // 没有值
            }
            
            // 查找值的结束
            size_t valueEnd = valueStart;
            bool inQuotes = false;
            
            // 检查是否以引号开始（字符串值）
            if (valueStart < card.length() && card[valueStart] == '\'') {
                inQuotes = true;
                valueStart++;  // 跳过开始引号
                valueEnd = valueStart;
                
                // 查找结束引号
                while (valueEnd < card.length() && card[valueEnd] != '\'') {
                    valueEnd++;
                }
            } else {
                // 数值或布尔值：找到第一个空格或斜杠
                while (valueEnd < card.length()) {
                    if (card[valueEnd] == ' ' || card[valueEnd] == '/') {
                        break;
                    }
                    valueEnd++;
                }
            }
            
            if (valueEnd <= valueStart) {
                return "";  // 空值
            }
            
            std::string value = card.substr(valueStart, valueEnd - valueStart);
            return trim(value);
        }
    }
    
    // 未找到关键字
    return "";
}

/**
 * 去除字符串首尾空格
 */
std::string FitsReader::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

/**
 * 解析FITS头关键字
 */
void FitsReader::parseFitsHeader(const std::string& header, FitsData& data) {
    // 这个函数可以用于更复杂的头解析
    // 目前基本功能已在readFitsHeaderSimple中实现
}
