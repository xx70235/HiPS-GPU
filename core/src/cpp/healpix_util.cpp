/**
 * HEALPix网格处理实现
 * 实现HEALPix嵌套索引与天球坐标的转换
 * 
 * 基于标准HEALPix算法实现，参考：
 * - HEALPix官方C++实现
 * - esheldon/healpix_util的C实现
 * - CDS HEALPix Java库
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "healpix_util.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

using std::string;

// HEALPix常量
static const double TWOPI = 2.0 * M_PI;
static const double HALFPI = M_PI / 2.0;
static const double TWOTHIRD = 2.0 / 3.0;

// 12个基础像素的face编号到xy坐标的映射
// jrll: 基础像素的"环"坐标 (1-4表示北极，5-8表示赤道，9-12表示南极)
static const int jrll[] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};  // 行号
static const int jpll[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};  // 列号

// 查找表：用于Morton编码/解码
// 这些表将位交错/解交错
static int pix2x_[1024];
static int pix2y_[1024];
static int x2pix_[128];
static int y2pix_[128];
static bool tables_initialized = false;

/**
 * 初始化Morton编码查找表
 */
static void initTables() {
    if (tables_initialized) return;
    
    // 构建 x2pix 和 y2pix 表
    for (int i = 0; i < 128; i++) {
        int x = i;
        int y = i;
        int xi = 0, yi = 0;
        
        for (int k = 0; k < 7; k++) {
            xi |= ((x & 1) << (2 * k));
            yi |= ((y & 1) << (2 * k + 1));
            x >>= 1;
            y >>= 1;
        }
        x2pix_[i] = xi;
        y2pix_[i] = yi;
    }
    
    // 构建 pix2x 和 pix2y 表
    for (int i = 0; i < 1024; i++) {
        int x = 0, y = 0;
        for (int k = 0; k < 10; k++) {
            if (k % 2 == 0) {
                x |= ((i >> k) & 1) << (k / 2);
            } else {
                y |= ((i >> k) & 1) << (k / 2);
            }
        }
        pix2x_[i] = x;
        pix2y_[i] = y;
    }
    
    tables_initialized = true;
}

/**
 * Morton编码：将(x,y)坐标转换为嵌套索引中的位置部分
 * 位交错：将x和y的位交替放置
 */
static long xy2pix(int x, int y) {
    initTables();
    return x2pix_[x & 127] | y2pix_[y & 127] |
           (x2pix_[(x >> 7) & 127] | y2pix_[(y >> 7) & 127]) << 14 |
           (long)(x2pix_[(x >> 14) & 127] | y2pix_[(y >> 14) & 127]) << 28;
}

/**
 * Morton解码：从嵌套索引中的位置部分提取(x,y)坐标
 */
static void pix2xy(long ipf, int& x, int& y) {
    initTables();
    
    int v = (int)(ipf & 0x3FF);
    x = pix2x_[v];
    y = pix2y_[v];
    
    v = (int)((ipf >> 10) & 0x3FF);
    x |= pix2x_[v] << 5;
    y |= pix2y_[v] << 5;
    
    v = (int)((ipf >> 20) & 0x3FF);
    x |= pix2x_[v] << 10;
    y |= pix2y_[v] << 10;
    
    v = (int)((ipf >> 30) & 0x3FF);
    x |= pix2x_[v] << 15;
    y |= pix2y_[v] << 15;
}

/**
 * 将角度坐标(theta, phi)转换为嵌套索引
 * @param nside HEALPix分辨率参数
 * @param theta 余纬（colatitude），0到PI
 * @param phi 经度，0到2PI
 * @return 嵌套像素索引
 */
static long ang2pix_nest_internal(long nside, double theta, double phi) {
    initTables();
    
    double z = cos(theta);
    double za = fabs(z);
    
    // 归一化phi到[0, 2PI)
    double tt = fmod(phi, TWOPI);
    if (tt < 0) tt += TWOPI;
    tt = tt / HALFPI;  // tt在[0, 4)
    
    long face_num, ix, iy;
    
    if (za <= TWOTHIRD) {
        // 赤道区域
        double temp1 = nside * (0.5 + tt - z * 0.75);
        double temp2 = nside * (0.5 + tt + z * 0.75);
        
        long jp = (long)temp1;  // 上升边线索引
        long jm = (long)temp2;  // 下降边线索引
        
        long ifp = jp / nside;  // 上升边线的face
        long ifm = jm / nside;  // 下降边线的face
        
        if (ifp == ifm) {
            // 在同一face内
            face_num = (ifp == 4) ? 4 : ifp + 4;
        } else if (ifp < ifm) {
            face_num = ifp + 4;
        } else {
            face_num = ifm + 8;
        }
        
        ix = jm & (nside - 1);
        iy = nside - (jp & (nside - 1)) - 1;
    } else {
        // 极区
        long ntt = (long)tt;
        if (ntt >= 4) ntt = 3;
        double tp = tt - ntt;
        
        double tmp;
        if (za < 0.99) {
            tmp = nside * sqrt(3.0 * (1.0 - za));
        } else {
            // 接近极点时使用更精确的公式
            double sa = sqrt((1.0 - za) * 2.0);
            tmp = nside * sqrt(3.0) * sa / sqrt(2.0);
        }
        
        long jp = (long)(tp * tmp);
        long jm = (long)((1.0 - tp) * tmp);
        
        if (jp >= nside) jp = nside - 1;
        if (jm >= nside) jm = nside - 1;
        
        if (z >= 0) {
            // 北极区
            face_num = ntt;
            ix = nside - jm - 1;
            iy = nside - jp - 1;
        } else {
            // 南极区
            face_num = ntt + 8;
            ix = jp;
            iy = jm;
        }
    }
    
    // 计算最终的嵌套索引
    long npface = nside * nside;
    return face_num * npface + xy2pix((int)ix, (int)iy);
}

/**
 * 将嵌套索引转换为角度坐标(theta, phi)
 * @param nside HEALPix分辨率参数
 * @param ipix 嵌套像素索引
 * @param theta 输出余纬（colatitude），0到PI
 * @param phi 输出经度，0到2PI
 */
static void pix2ang_nest_internal(long nside, long ipix, double& theta, double& phi) {
    initTables();
    
    long npface = nside * nside;
    long nl4 = 4 * nside;
    
    // 找到face编号和face内的像素编号
    long face_num = ipix / npface;
    long ipf = ipix % npface;
    
    // 从Morton码提取x,y坐标
    int ix, iy;
    pix2xy(ipf, ix, iy);
    
    // 计算jr（环索引，从1开始）
    long jrt = jrll[face_num] * nside - ix - iy - 1;
    
    long nr, kshift, jp;
    double z;
    
    if (jrt < nside) {
        // 北极区
        nr = jrt;
        z = 1.0 - nr * nr * 2.0 / (3.0 * nside * nside);
        kshift = 0;
        jp = (jpll[face_num] * nr + ix - iy + 1) / 2;
        if (jp > nl4) jp -= nl4;
        if (jp < 1) jp += nl4;
    } else if (jrt > 3 * nside) {
        // 南极区
        nr = nl4 - jrt;
        z = -(1.0 - nr * nr * 2.0 / (3.0 * nside * nside));
        kshift = 0;
        jp = (jpll[face_num] * nr + ix - iy + 1) / 2;
        if (jp > nl4) jp -= nl4;
        if (jp < 1) jp += nl4;
    } else {
        // 赤道区域
        nr = nside;
        long jpt = (jpll[face_num] * nr + ix - iy + 1 + ((jrt - nside) & 1));
        jp = jpt / 2;
        if (jp > nl4) jp -= nl4;
        if (jp < 1) jp += nl4;
        z = (2 * nside - jrt) * 2.0 / (3.0 * nside);
        kshift = (jrt - nside) & 1;
    }
    
    theta = acos(z);
    phi = (jp - (kshift + 1) * 0.5) * HALFPI / nr;
    if (phi < 0) phi += TWOPI;
    if (phi >= TWOPI) phi -= TWOPI;
}

/**
 * 将HEALPix嵌套索引转换为天球坐标
 */
CelestialCoord HealpixUtil::nestedToCelestial(int norder, long npix) {
    // 计算nside
    int nside = norderToNside(norder);
    
    // 验证npix范围
    long maxPix = getTotalPixelsFromNside(nside) - 1;
    if (npix < 0 || npix > maxPix) {
        // 无效索引，返回默认值
        return CelestialCoord(0.0, 0.0);
    }
    
    // 转换为theta, phi
    double theta, phi;
    pix2ang_nest_internal(nside, npix, theta, phi);
    
    // 转换为RA/Dec (度)
    // theta是余纬（colatitude），theta = PI/2 - Dec
    // phi是经度（longitude），phi = RA
    double dec_deg = (HALFPI - theta) * 180.0 / M_PI;
    double ra_deg = phi * 180.0 / M_PI;
    
    // 归一化RA到[0, 360)
    while (ra_deg < 0.0) ra_deg += 360.0;
    while (ra_deg >= 360.0) ra_deg -= 360.0;
    
    return CelestialCoord(ra_deg, dec_deg);
}

/**
 * 将天球坐标转换为HEALPix嵌套索引
 */
long HealpixUtil::celestialToNested(const CelestialCoord& celestial, int norder) {
    // 计算nside
    int nside = norderToNside(norder);
    
    // 转换RA/Dec(度)为theta/phi(弧度)
    // theta是余纬（colatitude），theta = PI/2 - Dec
    // phi是经度（longitude），phi = RA
    double ra_rad = celestial.ra * M_PI / 180.0;
    double dec_rad = celestial.dec * M_PI / 180.0;
    
    double theta = HALFPI - dec_rad;  // 余纬
    double phi = ra_rad;              // 经度
    
    // 归一化phi到[0, 2PI)
    while (phi < 0.0) phi += TWOPI;
    while (phi >= TWOPI) phi -= TWOPI;
    
    // 确保theta在有效范围[0, PI]
    if (theta < 0.0) theta = 0.0;
    if (theta > M_PI) theta = M_PI;
    
    return ang2pix_nest_internal(nside, theta, phi);
}

/**
 * 获取HEALPix像素中心的天球坐标
 */
CelestialCoord HealpixUtil::getPixelCenter(int norder, long npix) {
    return nestedToCelestial(norder, npix);
}

/**
 * 计算HEALPix像素的角直径（度）
 */
double HealpixUtil::getPixelSize(int norder) {
    int nside = norderToNside(norder);
    // HEALPix像素角直径 ≈ sqrt(4*PI / (12*nside^2)) ≈ sqrt(PI/(3)) / nside (弧度)
    double pixelSize_rad = sqrt(4.0 * M_PI / (12.0 * nside * nside));
    return pixelSize_rad * 180.0 / M_PI;  // 转换为度
}

/**
 * 检查npix是否在有效范围内
 */
bool HealpixUtil::isValidPixel(long npix, int norder) {
    if (norder < 0 || npix < 0) return false;
    long maxPix = getTotalPixels(norder) - 1;
    return npix <= maxPix;
}

/**
 * 计算HEALPix tile路径
 */
string HealpixUtil::getTilePath(int norder, long npix, const string& ext) {
    int dirNum = getDirNumber(npix);
    
    std::ostringstream oss;
    oss << "Norder" << norder << "/Dir" << dirNum << "/Npix" << npix << ext;
    return oss.str();
}

/**
 * 从嵌套索引转换为x, y坐标
 * HEALPix使用嵌套排序（Morton曲线/Z-order曲线）
 */
void HealpixUtil::nestedToXy(int norder, long npix, int& x, int& y) {
    int nside = norderToNside(norder);
    long npface = (long)nside * nside;
    long ipf = npix % npface;
    pix2xy(ipf, x, y);
}

/**
 * 从x, y坐标转换为嵌套索引
 */
long HealpixUtil::xyToNested(int norder, int x, int y) {
    int nside = norderToNside(norder);
    long npface = (long)nside * nside;
    // 注意：这里假设face_num为0，实际使用时需要正确的face信息
    return xy2pix(x, y);
}

/**
 * 从x, y坐标转换为theta, phi（球坐标）
 * 注意：此函数已不再使用，保留接口兼容性
 */
void HealpixUtil::xyToThetaPhi(int norder, int x, int y, double& theta, double& phi) {
    // 此函数不再使用简化实现
    // 正确的转换应该通过pix2ang_nest_internal实现
    theta = 0;
    phi = 0;
}

/**
 * 从theta, phi转换为x, y坐标
 * 注意：此函数已不再使用，保留接口兼容性
 */
void HealpixUtil::thetaPhiToXy(int norder, double theta, double phi, int& x, int& y) {
    // 此函数不再使用简化实现
    // 正确的转换应该通过ang2pix_nest_internal实现
    x = 0;
    y = 0;
}

/**
 * 计算tile内的像素坐标范围
 */
std::vector<Coord> HealpixUtil::getTilePixelCoords(int norder, int tileWidth) {
    std::vector<Coord> coords;
    coords.reserve(tileWidth * tileWidth);
    
    double halfWidth = tileWidth / 2.0;
    
    for (int y = 0; y < tileWidth; y++) {
        for (int x = 0; x < tileWidth; x++) {
            // 相对于tile中心的坐标
            double dx = (x - halfWidth) + 0.5;
            double dy = (y - halfWidth) + 0.5;
            
            coords.push_back(Coord(dx, dy));
        }
    }
    
    return coords;
}
