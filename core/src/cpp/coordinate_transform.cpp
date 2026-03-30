/**
 * 坐标转换实现 - 修复版
 * 基于WCS参数实现天球坐标与像素坐标之间的转换
 * TAN投影公式参考WCSLIB实现
 */

#include "coordinate_transform.h"
#include <cmath>
#include <algorithm>

using std::string;

const string CoordinateTransform::PROJ_TAN = "TAN";
const string CoordinateTransform::PROJ_SIN = "SIN";
const string CoordinateTransform::PROJ_ARC = "ARC";
const string CoordinateTransform::PROJ_AIT = "AIT";
const string CoordinateTransform::PROJ_STG = "STG";

/**
 * 构造函数
 */
CoordinateTransform::CoordinateTransform(const FitsData& fitsData)
    : crval1(fitsData.crval1)
    , crval2(fitsData.crval2)
    , crpix1(fitsData.crpix1)
    , crpix2(fitsData.crpix2)
    , cd1_1(fitsData.cd1_1)
    , cd1_2(fitsData.cd1_2)
    , cd2_1(fitsData.cd2_1)
    , cd2_2(fitsData.cd2_2)
    , ctype1(fitsData.ctype1)
    , ctype2(fitsData.ctype2)
    , width(fitsData.width)
    , height(fitsData.height)
{
    // 如果CD矩阵为零，尝试从CDELT推导（简化处理）
    if (cd1_1 == 0.0 && cd1_2 == 0.0 && cd2_1 == 0.0 && cd2_2 == 0.0) {
        double pixelScale = 1.0 / 3600.0;  // 假设1 arcsec/pixel
        cd1_1 = pixelScale;
        cd2_2 = pixelScale;
    }
}

/**
 * 从天球坐标转换为像素坐标
 */
Coord CoordinateTransform::celestialToPixel(const CelestialCoord& celestial) const {
    // 应用投影变换（TAN投影）
    double x, y;
    tangentProjection(celestial, x, y);
    
    // 应用CD矩阵逆（从投影坐标转换为像素偏移）
    double dx, dy;
    applyCDMatrixInverse(x, y, dx, dy);
    
    // 加上参考像素偏移 (CRPIX是1-based，这里转为0-based)
    double pixelX = crpix1 - 1.0 + dx;
    double pixelY = crpix2 - 1.0 + dy;
    
    return Coord(pixelX, pixelY);
}

/**
 * 从像素坐标转换为天球坐标
 */
CelestialCoord CoordinateTransform::pixelToCelestial(const Coord& pixel) const {
    // 计算相对于参考像素的偏移 (转换为1-based再计算)
    double dx = pixel.x + 1.0 - crpix1;
    double dy = pixel.y + 1.0 - crpix2;
    
    // 应用CD矩阵（从像素偏移转换为投影坐标）
    double x, y;
    applyCDMatrix(dx, dy, x, y);
    
    // 应用投影逆变换（从投影坐标转换为天球坐标）
    CelestialCoord celestial;
    tangentProjectionInverse(x, y, celestial);
    
    return celestial;
}

/**
 * TAN投影（正切投影/gnomonic投影）
 * 从天球坐标转换为投影坐标
 * 参考: WCSLIB prj.c tans2x()
 */
void CoordinateTransform::tangentProjection(const CelestialCoord& celestial, double& x, double& y) const {
    double ra = celestial.ra;
    double dec = celestial.dec;
    
    // 转换为弧度 - 使用绝对坐标
    double ra_rad = degToRad(ra);
    double dec_rad = degToRad(dec);
    double ra0_rad = degToRad(crval1);
    double dec0_rad = degToRad(crval2);
    
    // TAN投影公式 (gnomonic projection)
    double cos_dec = cos(dec_rad);
    double sin_dec = sin(dec_rad);
    double cos_dec0 = cos(dec0_rad);
    double sin_dec0 = sin(dec0_rad);
    double delta_ra = ra_rad - ra0_rad;
    double cos_delta_ra = cos(delta_ra);
    double sin_delta_ra = sin(delta_ra);
    
    // denom = sin(dec0)*sin(dec) + cos(dec0)*cos(dec)*cos(delta_ra)
    double denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_delta_ra;
    
    if (fabs(denom) < 1e-10) {
        // 点在切平面后面
        x = 0.0;
        y = 0.0;
        return;
    }
    
    // 投影坐标（弧度）
    // x = cos(dec)*sin(delta_ra) / denom
    // y = (cos(dec0)*sin(dec) - sin(dec0)*cos(dec)*cos(delta_ra)) / denom
    x = cos_dec * sin_delta_ra / denom;
    y = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_delta_ra) / denom;
    
    // 转换为度
    x = radToDeg(x);
    y = radToDeg(y);
}

/**
 * TAN投影逆变换
 * 从投影坐标转换为天球坐标
 * 参考: WCSLIB prj.c tanx2s()
 */
void CoordinateTransform::tangentProjectionInverse(double x, double y, CelestialCoord& celestial) const {
    // 转换为弧度
    double x_rad = degToRad(x);
    double y_rad = degToRad(y);
    double dec0_rad = degToRad(crval2);
    
    double cos_dec0 = cos(dec0_rad);
    double sin_dec0 = sin(dec0_rad);
    
    // TAN投影逆变换公式
    double rho = sqrt(x_rad * x_rad + y_rad * y_rad);
    
    if (rho < 1e-10) {
        // 接近参考点
        celestial.ra = crval1;
        celestial.dec = crval2;
        return;
    }
    
    double c = atan(rho);
    double sin_c = sin(c);
    double cos_c = cos(c);
    
    double dec_rad = asin(cos_c * sin_dec0 + y_rad * sin_c * cos_dec0 / rho);
    double ra_offset = atan2(x_rad * sin_c, rho * cos_dec0 * cos_c - y_rad * sin_dec0 * sin_c);
    
    celestial.dec = radToDeg(dec_rad);
    celestial.ra = crval1 + radToDeg(ra_offset);
    
    // 归一化RA到[0, 360)
    while (celestial.ra < 0.0) celestial.ra += 360.0;
    while (celestial.ra >= 360.0) celestial.ra -= 360.0;
}

/**
 * 应用CD矩阵（从像素偏移到投影坐标）
 */
void CoordinateTransform::applyCDMatrix(double dx, double dy, double& x, double& y) const {
    // CD矩阵：从像素偏移到投影坐标（度）
    // [x]   [CD1_1  CD1_2] [dx]
    // [y] = [CD2_1  CD2_2] [dy]
    x = cd1_1 * dx + cd1_2 * dy;
    y = cd2_1 * dx + cd2_2 * dy;
}

/**
 * 应用CD矩阵逆变换（从投影坐标到像素偏移）
 */
void CoordinateTransform::applyCDMatrixInverse(double x, double y, double& dx, double& dy) const {
    // CD矩阵逆：从投影坐标到像素偏移
    // 计算行列式
    double det = cd1_1 * cd2_2 - cd1_2 * cd2_1;
    
    if (fabs(det) < 1e-10) {
        // 奇异矩阵，使用近似
        dx = x / cd1_1;
        dy = y / cd2_2;
        return;
    }
    
    // 逆矩阵
    double inv_det = 1.0 / det;
    dx = (cd2_2 * x - cd1_2 * y) * inv_det;
    dy = (-cd2_1 * x + cd1_1 * y) * inv_det;
}

/**
 * 检查像素坐标是否在图像范围内
 */
bool CoordinateTransform::isPixelInBounds(const Coord& pixel) const {
    return pixel.x >= 0 && pixel.x < width && pixel.y >= 0 && pixel.y < height;
}

/**
 * 检查天球坐标是否在图像覆盖范围内（近似检查）
 */
bool CoordinateTransform::isCelestialInBounds(const CelestialCoord& celestial) const {
    // 将坐标转换为像素坐标
    Coord pixel = celestialToPixel(celestial);
    
    // 检查是否在图像范围内（带一些容差）
    double tolerance = 10.0;  // 10像素容差
    return pixel.x >= -tolerance && pixel.x < width + tolerance &&
           pixel.y >= -tolerance && pixel.y < height + tolerance;
}
