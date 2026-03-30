/**
 * 坐标转换模块
 * 实现天球坐标（RA/Dec）与像素坐标（X/Y）之间的转换
 * 基于WCS（World Coordinate System）变换
 */

#ifndef COORDINATE_TRANSFORM_H
#define COORDINATE_TRANSFORM_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <string>
#include "fits_io.h"

// 坐标点结构
struct Coord {
    double x, y;  // 像素坐标或天球坐标
    
    Coord(double x = 0.0, double y = 0.0) : x(x), y(y) {}
};

// 天球坐标（RA/Dec）
struct CelestialCoord {
    double ra;   // 赤经（度）
    double dec;  // 赤纬（度）
    
    CelestialCoord(double ra = 0.0, double dec = 0.0) : ra(ra), dec(dec) {}
};

/**
 * 坐标转换类
 * 基于WCS参数实现坐标变换
 */
class CoordinateTransform {
public:
    /**
     * 构造函数
     * @param fitsData FITS文件数据，包含WCS参数
     */
    CoordinateTransform(const FitsData& fitsData);
    
    /**
     * 从天球坐标（RA/Dec）转换为像素坐标（X/Y）
     * @param celestial 天球坐标（RA/Dec，单位：度）
     * @return 像素坐标（X/Y）
     */
    Coord celestialToPixel(const CelestialCoord& celestial) const;
    
    /**
     * 从像素坐标（X/Y）转换为天球坐标（RA/Dec）
     * @param pixel 像素坐标（X/Y）
     * @return 天球坐标（RA/Dec，单位：度）
     */
    CelestialCoord pixelToCelestial(const Coord& pixel) const;
    
    /**
     * 检查像素坐标是否在图像范围内
     */
    bool isPixelInBounds(const Coord& pixel) const;
    
    /**
     * 检查天球坐标是否在图像覆盖范围内（近似检查）
     */
    bool isCelestialInBounds(const CelestialCoord& celestial) const;
    
private:
    // WCS参数
    double crval1, crval2;  // RA/Dec参考值（度）
    double crpix1, crpix2;  // 参考像素坐标
    double cd1_1, cd1_2;    // CD矩阵元素
    double cd2_1, cd2_2;
    
    std::string ctype1, ctype2;  // 投影类型
    
    // 图像尺寸
    int width, height;
    
    // 投影类型常量
    static const std::string PROJ_TAN;  // TAN (gnomonic/tangential)
    static const std::string PROJ_SIN;  // SIN (orthographic/slant orthographic)
    static const std::string PROJ_ARC;  // ARC (zenithal equidistant)
    static const std::string PROJ_AIT;  // AIT (Hammer-Aitoff)
    static const std::string PROJ_STG;  // STG (stereographic)
    
    // 投影相关辅助函数
    void tangentProjection(const CelestialCoord& celestial, double& x, double& y) const;
    void tangentProjectionInverse(double x, double y, CelestialCoord& celestial) const;
    
    void applyCDMatrix(double dx, double dy, double& x, double& y) const;
    void applyCDMatrixInverse(double x, double y, double& dx, double& dy) const;
    
    // 度与弧度的转换
    static double degToRad(double deg) { return deg * 3.14159265358979323846 / 180.0; }
    static double radToDeg(double rad) { return rad * 180.0 / 3.14159265358979323846; }
    
    // 三角函数（使用弧度）
    static double sind(double deg) { return sin(degToRad(deg)); }
    static double cosd(double deg) { return cos(degToRad(deg)); }
    static double tand(double deg) { return tan(degToRad(deg)); }
    static double asind(double val) { return radToDeg(asin(val)); }
    static double acosd(double val) { return radToDeg(acos(val)); }
    static double atand(double val) { return radToDeg(atan(val)); }
    static double atan2d(double y, double x) { return radToDeg(atan2(y, x)); }
};

#endif // COORDINATE_TRANSFORM_H
