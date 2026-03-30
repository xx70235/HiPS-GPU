#ifndef FITS_HEADER_READER_H
#define FITS_HEADER_READER_H

#include <string>
#include <vector>
#include <fitsio.h>

/**
 * 快速FITS头读取器 - 只读取WCS信息，不读取像素数据
 * 用于INDEX阶段的优化
 */
struct WCSInfo {
    double crval1 = 0, crval2 = 0;  // 参考点天球坐标
    double crpix1 = 0, crpix2 = 0;  // 参考点像素坐标
    double cd1_1 = 0, cd1_2 = 0;    // CD矩阵
    double cd2_1 = 0, cd2_2 = 0;
    int width = 0, height = 0;       // 图像尺寸
    bool valid = false;
};

class FastFitsHeaderReader {
public:
    /**
     * 快速读取FITS文件的WCS信息（不读取像素数据）
     * 比readFitsFile快10-100倍
     */
    static WCSInfo readWCSInfo(const std::string& filename) {
        WCSInfo info;
        fitsfile* fptr = nullptr;
        int status = 0;
        
        // 打开文件
        if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
            return info;
        }
        
        // 对于.fz压缩文件，移动到HDU 2
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        if (ext == "fz") {
            int hdutype;
            fits_movabs_hdu(fptr, 2, &hdutype, &status);
            if (status) {
                status = 0;
                fits_movabs_hdu(fptr, 1, &hdutype, &status);
            }
        }
        
        // 读取图像尺寸
        long naxes[2] = {0, 0};
        int naxis = 0;
        fits_get_img_dim(fptr, &naxis, &status);
        if (naxis >= 2) {
            fits_get_img_size(fptr, 2, naxes, &status);
            info.width = naxes[0];
            info.height = naxes[1];
        }
        
        // 读取WCS关键字
        char comment[FLEN_COMMENT];
        fits_read_key(fptr, TDOUBLE, "CRVAL1", &info.crval1, comment, &status); status = 0;
        fits_read_key(fptr, TDOUBLE, "CRVAL2", &info.crval2, comment, &status); status = 0;
        fits_read_key(fptr, TDOUBLE, "CRPIX1", &info.crpix1, comment, &status); status = 0;
        fits_read_key(fptr, TDOUBLE, "CRPIX2", &info.crpix2, comment, &status); status = 0;
        
        // 尝试读取CD矩阵
        fits_read_key(fptr, TDOUBLE, "CD1_1", &info.cd1_1, comment, &status);
        if (status) {
            // 如果没有CD矩阵，尝试CDELT
            status = 0;
            double cdelt1 = 0, cdelt2 = 0, crota2 = 0;
            fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, comment, &status); status = 0;
            fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, comment, &status); status = 0;
            fits_read_key(fptr, TDOUBLE, "CROTA2", &crota2, comment, &status); status = 0;
            
            double cos_r = cos(crota2 * M_PI / 180.0);
            double sin_r = sin(crota2 * M_PI / 180.0);
            info.cd1_1 = cdelt1 * cos_r;
            info.cd1_2 = -cdelt2 * sin_r;
            info.cd2_1 = cdelt1 * sin_r;
            info.cd2_2 = cdelt2 * cos_r;
        } else {
            status = 0;
            fits_read_key(fptr, TDOUBLE, "CD1_2", &info.cd1_2, comment, &status); status = 0;
            fits_read_key(fptr, TDOUBLE, "CD2_1", &info.cd2_1, comment, &status); status = 0;
            fits_read_key(fptr, TDOUBLE, "CD2_2", &info.cd2_2, comment, &status); status = 0;
        }
        
        fits_close_file(fptr, &status);
        
        // 验证WCS有效性
        if (info.width > 0 && info.height > 0 && 
            (info.cd1_1 != 0 || info.cd1_2 != 0 || info.cd2_1 != 0 || info.cd2_2 != 0)) {
            info.valid = true;
        }
        
        return info;
    }
    
    /**
     * 批量读取多个文件的WCS信息（OpenMP并行）
     */
    static std::vector<WCSInfo> readWCSInfoBatch(const std::vector<std::string>& files) {
        std::vector<WCSInfo> results(files.size());
        
        #pragma omp parallel for schedule(dynamic, 10)
        for (size_t i = 0; i < files.size(); i++) {
            results[i] = readWCSInfo(files[i]);
        }
        
        return results;
    }
};

#endif // FITS_HEADER_READER_H
