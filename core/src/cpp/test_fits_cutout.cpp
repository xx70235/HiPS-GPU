#include "fits_io.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <unistd.h>

#include <fitsio.h>

static std::string make_test_fits() {
    char tmpTemplate[] = "/tmp/test_fits_cutout_XXXXXX.fits";
    int fd = mkstemps(tmpTemplate, 5);
    assert(fd >= 0);
    close(fd);

    std::string path = tmpTemplate;
    std::remove(path.c_str());
    std::string createPath = "!" + path;

    fitsfile* fptr = nullptr;
    int status = 0;

    fits_create_file(&fptr, createPath.c_str(), &status);
    long imageAxes[2] = {6, 4};
    fits_create_img(fptr, FLOAT_IMG, 2, imageAxes, &status);

    double crval1 = 15.0;
    double crval2 = -2.0;
    double crpix1 = 5.0;
    double crpix2 = 5.0;
    double cd11 = -0.0001;
    double cd12 = 0.0;
    double cd21 = 0.0;
    double cd22 = 0.0001;
    fits_write_key(fptr, TDOUBLE, "CRVAL1", &crval1, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CRVAL2", &crval2, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CRPIX1", &crpix1, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CRPIX2", &crpix2, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CD1_1", &cd11, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CD1_2", &cd12, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CD2_1", &cd21, nullptr, &status);
    fits_write_key(fptr, TDOUBLE, "CD2_2", &cd22, nullptr, &status);

    float pixels[24];
    for (int i = 0; i < 24; ++i) {
        pixels[i] = static_cast<float>(i);
    }

    long firstPixel = 1;
    fits_write_img(fptr, TFLOAT, firstPixel, 24, pixels, &status);
    fits_close_file(fptr, &status);

    assert(status == 0);
    return path;
}

int main() {
    std::string path = make_test_fits();

    PixelBounds bounds;
    bounds.x0 = 2;
    bounds.y0 = 1;
    bounds.width = 3;
    bounds.height = 2;

    FitsData cutout = FitsReader::readFitsCutout(path, bounds, 0);
    assert(cutout.isValid);
    assert(cutout.width == 3);
    assert(cutout.height == 2);
    assert(cutout.crpix1 == 3.0);
    assert(cutout.crpix2 == 4.0);
    assert(cutout.pixels.size() == 6);
    assert(cutout.pixels[0] == 8.0f);
    assert(cutout.pixels[5] == 16.0f);

    std::remove(path.c_str());
    return 0;
}
