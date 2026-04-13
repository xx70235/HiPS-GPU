#ifndef FITS_HDU_UTILS_H
#define FITS_HDU_UTILS_H

#include <string>

#include <fitsio.h>

struct HduSelectionResult {
    bool found = false;
    int hdu_index_1_based = 0;
    int hdutype = 0;
    int naxis = 0;
    long width = 0;
    long height = 0;
    std::string error;
};

HduSelectionResult select_valid_image_hdu(
    fitsfile* fptr,
    const std::string& filename,
    int preferred_zero_based_hdu
);

#endif
