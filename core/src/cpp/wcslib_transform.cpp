/**
 * WCS坐标转换实现 - 基于WCSLIB
 */

#include "wcslib_transform.h"
#include <wcslib/wcs.h>
#include <wcslib/wcsfix.h>
#include <cstring>
#include <cstdio>
#include <cmath>

WCSLibTransform::WCSLibTransform(const FitsData& fitsData)
    : wcs_(nullptr)
    , valid_(false)
    , width_(fitsData.width)
    , height_(fitsData.height)
{
    wcs_ = (struct wcsprm*)calloc(1, sizeof(struct wcsprm));
    if (!wcs_) {
        fprintf(stderr, "WCSLibTransform: Failed to allocate wcsprm\n");
        return;
    }
    
    wcs_->flag = -1;
    int status = wcsini(1, 2, wcs_);
    if (status) {
        fprintf(stderr, "WCSLibTransform: wcsini error %d\n", status);
        free(wcs_);
        wcs_ = nullptr;
        return;
    }
    
    // Set WCS parameters
    wcs_->crpix[0] = fitsData.crpix1;
    wcs_->crpix[1] = fitsData.crpix2;
    wcs_->crval[0] = fitsData.crval1;
    wcs_->crval[1] = fitsData.crval2;
    
    strncpy(wcs_->ctype[0], fitsData.ctype1.c_str(), 72);
    strncpy(wcs_->ctype[1], fitsData.ctype2.c_str(), 72);
    
    if (fitsData.cd1_1 != 0.0 || fitsData.cd1_2 != 0.0 ||
        fitsData.cd2_1 != 0.0 || fitsData.cd2_2 != 0.0) {
        wcs_->altlin = 2;
        wcs_->cd = (double*)calloc(4, sizeof(double));
        if (wcs_->cd) {
            wcs_->cd[0] = fitsData.cd1_1;
            wcs_->cd[1] = fitsData.cd1_2;
            wcs_->cd[2] = fitsData.cd2_1;
            wcs_->cd[3] = fitsData.cd2_2;
        }
    } else {
        wcs_->cdelt[0] = 1.0;
        wcs_->cdelt[1] = 1.0;
        wcs_->pc[0] = 1.0;
        wcs_->pc[1] = 0.0;
        wcs_->pc[2] = 0.0;
        wcs_->pc[3] = 1.0;
    }
    
    strcpy(wcs_->radesys, "ICRS");
    wcs_->equinox = 2000.0;
    
    status = wcsset(wcs_);
    if (status) {
        fprintf(stderr, "WCSLibTransform: wcsset error %d: %s\n", 
                status, wcs_errmsg[status]);
        wcsfree(wcs_);
        free(wcs_);
        wcs_ = nullptr;
        return;
    }
    
    valid_ = true;
}

WCSLibTransform::~WCSLibTransform() {
    if (wcs_) {
        wcsfree(wcs_);
        free(wcs_);
        wcs_ = nullptr;
    }
}

int WCSLibTransform::worldToPixel(double ra, double dec, double& x, double& y) const {
    if (!valid_ || !wcs_) return 1;
    
    double world[2] = {ra, dec};
    double imgcrd[2];
    double phi, theta;
    double pixcrd[2];
    int stat;
    
    int status = wcss2p(wcs_, 1, 2, world, &phi, &theta, imgcrd, pixcrd, &stat);
    
    if (status || stat) return status ? status : stat;
    
    // WCSLIB returns 1-based, convert to 0-based
    x = pixcrd[0] - 1.0;
    y = pixcrd[1] - 1.0;
    
    return 0;
}

int WCSLibTransform::pixelToWorld(double x, double y, double& ra, double& dec) const {
    if (!valid_ || !wcs_) return 1;
    
    // Convert to 1-based
    double pixcrd[2] = {x + 1.0, y + 1.0};
    double imgcrd[2];
    double phi, theta;
    double world[2];
    int stat;
    
    int status = wcsp2s(wcs_, 1, 2, pixcrd, imgcrd, &phi, &theta, world, &stat);
    
    if (status || stat) return status ? status : stat;
    
    ra = world[0];
    dec = world[1];
    
    return 0;
}

int WCSLibTransform::worldToPixelBatch(int ncoord, const double* world, 
                                        double* pixcrd, int* stat) const {
    if (!valid_ || !wcs_) return 1;
    
    double* imgcrd = new double[ncoord * 2];
    double* phi = new double[ncoord];
    double* theta = new double[ncoord];
    
    int status = wcss2p(wcs_, ncoord, 2, world, phi, theta, imgcrd, pixcrd, stat);
    
    for (int i = 0; i < ncoord; i++) {
        pixcrd[i*2] -= 1.0;
        pixcrd[i*2+1] -= 1.0;
    }
    
    delete[] imgcrd;
    delete[] phi;
    delete[] theta;
    
    int failures = 0;
    for (int i = 0; i < ncoord; i++) {
        if (stat[i]) failures++;
    }
    
    return status ? status : failures;
}
