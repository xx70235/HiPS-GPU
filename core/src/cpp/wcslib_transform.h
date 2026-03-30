/**
 * WCS坐标转换模块 - 基于WCSLIB
 * 使用专业天文库实现精确的WCS坐标转换
 */

#ifndef WCSLIB_TRANSFORM_H
#define WCSLIB_TRANSFORM_H

#include <wcslib/wcs.h>
#include <string>
#include <memory>
#include "fits_io.h"

/**
 * 基于WCSLIB的WCS坐标转换类
 */
class WCSLibTransform {
public:
    WCSLibTransform(const FitsData& fitsData);
    ~WCSLibTransform();
    
    int worldToPixel(double ra, double dec, double& x, double& y) const;
    int pixelToWorld(double x, double y, double& ra, double& dec) const;
    int worldToPixelBatch(int ncoord, const double* world, double* pixcrd, int* stat) const;
    
    bool isValid() const { return valid_; }
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    
private:
    struct wcsprm* wcs_;
    bool valid_;
    int width_;
    int height_;
    
    WCSLibTransform(const WCSLibTransform&) = delete;
    WCSLibTransform& operator=(const WCSLibTransform&) = delete;
};

#endif // WCSLIB_TRANSFORM_H
