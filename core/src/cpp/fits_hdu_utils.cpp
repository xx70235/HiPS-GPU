#include "fits_hdu_utils.h"

#include <algorithm>
#include <cctype>

namespace {

std::string get_lower_extension(const std::string& filename) {
    size_t lastDot = filename.find_last_of('.');
    if (lastDot == std::string::npos || lastDot + 1 >= filename.size()) {
        return "";
    }

    std::string ext = filename.substr(lastDot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

bool probe_image_hdu(fitsfile* fptr, int hduIndex1Based, HduSelectionResult& result) {
    int status = 0;
    int hdutype = 0;
    if (fits_movabs_hdu(fptr, hduIndex1Based, &hdutype, &status) != 0) {
        return false;
    }
    if (hdutype != IMAGE_HDU) {
        return false;
    }

    int naxis = 0;
    status = 0;
    if (fits_get_img_dim(fptr, &naxis, &status) != 0 || naxis <= 0) {
        return false;
    }

    long naxes[2] = {0, 0};
    status = 0;
    if (fits_get_img_size(fptr, 2, naxes, &status) != 0) {
        return false;
    }

    result.found = true;
    result.hdu_index_1_based = hduIndex1Based;
    result.hdutype = hdutype;
    result.naxis = naxis;
    result.width = naxes[0];
    result.height = (naxis >= 2) ? naxes[1] : 1;
    return true;
}

}  // namespace

HduSelectionResult select_valid_image_hdu(
    fitsfile* fptr,
    const std::string& filename,
    int preferred_zero_based_hdu
) {
    HduSelectionResult result;

    if (fptr == nullptr) {
        result.error = "Null FITS handle";
        return result;
    }

    int numHdus = 1;
    int status = 0;
    fits_get_num_hdus(fptr, &numHdus, &status);
    status = 0;

    int preferredHdu = preferred_zero_based_hdu + 1;
    if (preferred_zero_based_hdu == 0 && get_lower_extension(filename) == "fz") {
        preferredHdu = 2;
    }

    if (preferredHdu >= 1 && preferredHdu <= numHdus && probe_image_hdu(fptr, preferredHdu, result)) {
        return result;
    }

    for (int hduIndex = 1; hduIndex <= numHdus; ++hduIndex) {
        if (hduIndex == preferredHdu) {
            continue;
        }
        if (probe_image_hdu(fptr, hduIndex, result)) {
            return result;
        }
    }

    result.error = "No valid image HDU found in file";
    return result;
}
