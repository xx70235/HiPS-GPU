/**
 * FITS文件读取接口
 * 支持CFITSIO库和简化实现
 */

#ifndef FITS_IO_H
#define FITS_IO_H

#include <string>
#include <vector>
#include <limits>

// FITS文件数据结构
struct FitsData {
    std::string filename;
    
    // 图像尺寸
    int width = 0;
    int height = 0;
    int depth = 1;  // 颜色通道数（通常为1）
    
    // 像素数据类型
    int bitpix = -32;  // -32=float, -64=double, 8=byte, 16=short, 32=int
    
    // 像素数据
    std::vector<float> pixels;  // 存储为float数组
    
    // WCS参数（世界坐标系统）
    double crval1 = 0.0;  // RA (度)
    double crval2 = 0.0;  // Dec (度)
    double crpix1 = 0.0;  // 参考像素X
    double crpix2 = 0.0;  // 参考像素Y
    double cd1_1 = 0.0;   // CD矩阵元素
    double cd1_2 = 0.0;
    double cd2_1 = 0.0;
    double cd2_2 = 0.0;
    
    // 投影类型
    std::string ctype1 = "RA---TAN";  // 投影类型1
    std::string ctype2 = "DEC--TAN";  // 投影类型2
    
    // 其他参数
    double bzero = 0.0;   // BZERO
    double bscale = 1.0;  // BSCALE
    double blank = std::numeric_limits<double>::quiet_NaN();   // 空白值 (NaN表示未设置)
    
    // 坐标系
    std::string radesys = "ICRS";  // 坐标系统
    
    // 状态标志
    bool isValid = false;
    std::string errorMessage;
    
    /**
     * 获取像素值（double格式）
     */
    double getPixelDouble(int x, int y, int z = 0) const {
        if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
            return blank;
        }
        int idx = (z * height + y) * width + x;
        if (idx < 0 || idx >= (int)pixels.size()) {
            return blank;
        }
        return (double)pixels[idx];
    }
    
    /**
     * 检查坐标是否在图像范围内
     */
    bool isInBounds(double x, double y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }
};

/**
 * FITS文件读取类
 * 
 * 支持两种模式：
 * 1. 使用CFITSIO库（完整功能，推荐）
 * 2. 简化实现（基本功能，不依赖外部库）
 */
class FitsReader {
public:
    /**
     * 读取FITS文件
     * @param filename FITS文件路径
     * @param hdu HDU编号（默认为0，即主HDU）
     * @return FitsData结构，包含所有读取的数据
     */
    static FitsData readFitsFile(const std::string& filename, int hdu = 0);
    
    /**
     * 使用CFITSIO库读取FITS文件（如果可用）
     */
    static FitsData readFitsFileCFITSIO(const std::string& filename, int hdu = 0);
    
    /**
     * 使用简化实现读取FITS文件（不依赖外部库）
     * 仅支持基本的FITS格式
     */
    static FitsData readFitsFileSimple(const std::string& filename, int hdu = 0);
    
    /**
     * 检查CFITSIO库是否可用
     */
    static bool isCFITSIOAvailable();
    
    /**
     * 解析FITS头关键字
     */
    static void parseFitsHeader(const std::string& header, FitsData& data);
    
private:
    // 内部辅助函数
    static std::string trim(const std::string& str);
    static std::string getKeywordValue(const std::string& header, const std::string& keyword);
    static int readFitsHeaderSimple(const std::string& filename, FitsData& data);
    static int readFitsDataSimple(const std::string& filename, const FitsData& header, std::vector<float>& pixels);
};

#endif // FITS_IO_H
