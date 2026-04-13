#include "fits_hdu_utils.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <unistd.h>

#include <fitsio.h>

static std::string make_extension_image_fits() {
    char tmpTemplate[] = "/tmp/test_hdu_selection_XXXXXX.fits";
    int fd = mkstemps(tmpTemplate, 5);
    assert(fd >= 0);
    close(fd);

    std::string path = tmpTemplate;
    std::remove(path.c_str());
    std::string createPath = "!" + path;

    fitsfile* fptr = nullptr;
    int status = 0;

    fits_create_file(&fptr, createPath.c_str(), &status);
    long primaryAxes[2] = {0, 0};
    fits_create_img(fptr, BYTE_IMG, 0, primaryAxes, &status);

    long imageAxes[2] = {8, 6};
    fits_create_img(fptr, FLOAT_IMG, 2, imageAxes, &status);

    float pixels[48];
    for (int i = 0; i < 48; ++i) {
        pixels[i] = static_cast<float>(i);
    }

    long firstPixel = 1;
    fits_write_img(fptr, TFLOAT, firstPixel, 48, pixels, &status);
    fits_close_file(fptr, &status);

    assert(status == 0);
    return path;
}

int main() {
    std::string path = make_extension_image_fits();

    fitsfile* fptr = nullptr;
    int status = 0;
    fits_open_file(&fptr, path.c_str(), READONLY, &status);
    assert(status == 0);

    HduSelectionResult result = select_valid_image_hdu(fptr, path, 0);
    assert(result.found);
    assert(result.hdu_index_1_based == 2);
    assert(result.width == 8);
    assert(result.height == 6);

    fits_close_file(fptr, &status);
    std::remove(path.c_str());
    return 0;
}
