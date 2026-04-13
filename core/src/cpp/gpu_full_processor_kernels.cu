/**
 * GPU Full Processing Kernels
 * Uses precomputed celestial coordinates from official HEALPix library
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

__constant__ double d_PI = 3.1415926535897932384626433832795;

/**
 * Device function: TAN projection + CD matrix inverse
 * Convert celestial (RA, Dec) to pixel coordinates for a specific image
 * Uses Java Aladin TAN projection formulas (Calib.java GetXYstand)
 */
__device__ void celestialToPixel_device(
    double ra, double dec,
    double crval1, double crval2,
    double crpix1, double crpix2,
    double cd1_1, double cd1_2,
    double cd2_1, double cd2_2,
    double& pixelX, double& pixelY
) {
    // Convert all to radians
    double ra_rad = ra * d_PI / 180.0;
    double dec_rad = dec * d_PI / 180.0;
    double ra0_rad = crval1 * d_PI / 180.0;
    double dec0_rad = crval2 * d_PI / 180.0;
    
    // Precompute trig
    double cos_dec = cos(dec_rad);
    double sin_dec = sin(dec_rad);
    double cos_dec0 = cos(dec0_rad);
    double sin_dec0 = sin(dec0_rad);
    double delta_ra = ra_rad - ra0_rad;
    double cos_delta_ra = cos(delta_ra);
    double sin_delta_ra = sin(delta_ra);
    
    // Java Aladin TAN projection - matches Calib.java GetXYstand()
    // x_tet_phi = cos_del * sin_dalpha
    // y_tet_phi = sin_del * cdelz - cos_del * sdelz * cos_dalpha
    // den = sin_del * sdelz + cos_del * cdelz  (NOTE: no cos_dalpha!)
    double x_tet_phi = cos_dec * sin_delta_ra;
    double y_tet_phi = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra;
    double denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra;  // Java includes cos_delta_ra!
    
    if (fabs(denom) < 1e-10) {
        pixelX = -1e10;
        pixelY = -1e10;
        return;
    }
    
    double x = x_tet_phi / denom;
    double y = y_tet_phi / denom;
    
    // Convert to degrees
    x = x * 180.0 / d_PI;
    y = y * 180.0 / d_PI;
    
    // Apply CD matrix inverse
    double det = cd1_1 * cd2_2 - cd1_2 * cd2_1;
    double dx, dy;
    if (fabs(det) < 1e-10) {
        dx = x / cd1_1;
        dy = y / cd2_2;
    } else {
        double inv_det = 1.0 / det;
        dx = (cd2_2 * x - cd1_2 * y) * inv_det;
        dy = (-cd2_1 * x + cd1_1 * y) * inv_det;
    }
    
    pixelX = crpix1 + dx;
    pixelY = crpix2 + dy;
}

/**
 * Device function: Check if a value is blank
 */
__device__ bool isBlank_device(float value, float blank) {
    if (isnan(value)) return true;
    if (isnan(blank)) return isnan(value);
    return value == blank;
}

/**
 * Device function: Bilinear interpolation (Java compatible)
 */
__device__ double bilinearInterp_device(
    const float* pixels,
    int width, int height,
    double x, double y,
    float blank
) {
    // Standard boundary check for bilinear interpolation
    if (x < -0.5 || x >= width - 0.5 || y < -0.5 || y >= height - 0.5) {
        return NAN;
    }
    
    x = fmax(0.0, fmin(x, (double)(width - 1) - 0.001));
    y = fmax(0.0, fmin(y, (double)(height - 1) - 0.001));
    
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float v00 = pixels[y0 * width + x0];  // a0
    float v10 = pixels[y0 * width + x1];  // a1
    float v01 = pixels[y1 * width + x0];  // a2
    float v11 = pixels[y1 * width + x1];  // a3
    
    bool b0 = isBlank_device(v00, blank);
    bool b1 = isBlank_device(v10, blank);
    bool b2 = isBlank_device(v01, blank);
    bool b3 = isBlank_device(v11, blank);
    
    if (b0 && b1 && b2 && b3) {
        return NAN;
    }
    
    if (b0 || b1 || b2 || b3) {
        double fillValue = !b0 ? v00 : !b1 ? v10 : !b2 ? v01 : v11;
        if (b0) v00 = fillValue;
        if (b1) v10 = fillValue;
        if (b2) v01 = fillValue;
        if (b3) v11 = fillValue;
    }
    
    // Java-style inverse distance weighted interpolation (bilineaire in ThreadBuilderTile.java)
    double d0, d1, d2, d3, pA, pB;
    
    if (x == x0) { d0 = 1.0; d1 = 0.0; }
    else if (x == (x0 + 1)) { d0 = 0.0; d1 = 1.0; }
    else { d0 = 1.0 / (x - x0); d1 = 1.0 / ((x0 + 1) - x); }
    
    if (y == y0) { d2 = 1.0; d3 = 0.0; }
    else if (y == (y0 + 1)) { d2 = 0.0; d3 = 1.0; }
    else { d2 = 1.0 / (y - y0); d3 = 1.0 / ((y0 + 1) - y); }
    
    pA = (v00 * d0 + v10 * d1) / (d0 + d1);
    pB = (v01 * d0 + v11 * d1) / (d0 + d1);
    
    return (pA * d2 + pB * d3) / (d2 + d3);
}
/**
 * Main kernel: Process all tiles with precomputed coordinates
 * Each thread handles one output pixel
 */
template <typename ResultT>
__global__ void fullGPUTileKernelWithCoordsTyped(
    // Source images
    const float* __restrict__ allPixels,
    const size_t* __restrict__ imageOffsets,
    // WCS parameters
    const double* __restrict__ crval1, const double* __restrict__ crval2,
    const double* __restrict__ crpix1, const double* __restrict__ crpix2,
    const double* __restrict__ cd1_1, const double* __restrict__ cd1_2,
    const double* __restrict__ cd2_1, const double* __restrict__ cd2_2,
    const int* __restrict__ widths, const int* __restrict__ heights,
    const float* __restrict__ blanks,
    int numImages,
    // Precomputed celestial coordinates (from official HEALPix library)
    const double* __restrict__ raCoords,   // [numTiles * pixelsPerTile]
    const double* __restrict__ decCoords,  // [numTiles * pixelsPerTile]
    // Tile parameters
    const int* __restrict__ imageMasks,  // [numTiles * numImages]
    int tileWidth,
    int numTiles,
    // Output
    ResultT* __restrict__ results,
    double outputBlank
) {
    size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelsPerTile = tileWidth * tileWidth;
    size_t totalPixels = (size_t)numTiles * pixelsPerTile;

    if (globalIdx >= totalPixels) return;

    int tileIdx = globalIdx / pixelsPerTile;

    // Get precomputed celestial coordinates
    double ra = raCoords[globalIdx];
    double dec = decCoords[globalIdx];

    // WCS transform + interpolation + weighted coadd for each source image
    // Matches Java ThreadBuilderTile weighted average calculation
    double sumWeightedValue = 0.0;
    double totalCoef = 0.0;

    for (int img = 0; img < numImages; img++) {
        if (imageMasks[tileIdx * numImages + img] == 0) {
            continue;
        }

        double pixelX, pixelY;
        celestialToPixel_device(
            ra, dec,
            crval1[img], crval2[img],
            crpix1[img], crpix2[img],
            cd1_1[img], cd1_2[img],
            cd2_1[img], cd2_2[img],
            pixelX, pixelY
        );

        // Y-flip and 0-based conversion (match Java)
        double adjustedY = pixelY - 1;  // Java-compatible: no Y-flip, just 0-based
        pixelX = pixelX - 1;

        // Standard boundary check
        if (pixelX < -0.5 || pixelX >= widths[img] - 0.5 ||
            adjustedY < -0.5 || adjustedY >= heights[img] - 0.5) {
            continue;
        }

        const float* imgPixels = allPixels + imageOffsets[img];
        double value = bilinearInterp_device(
            imgPixels,
            widths[img], heights[img],
            pixelX, adjustedY,
            blanks[img]
        );

        if (!isBlank_device((float)value, blanks[img])) {
            double coef = 1.0;
            int x1 = (int)pixelX;
            int y1 = (int)adjustedY;
            int w = widths[img];
            int h = heights[img];

            if (x1 > 0 && x1 < w - 1 && (pixelX <= 0.5 || pixelX >= w - 1.5)) {
                coef *= 0.5;
            }
            if (y1 > 0 && y1 < h - 1 && (adjustedY <= 0.5 || adjustedY >= h - 1.5)) {
                coef *= 0.5;
            }

            totalCoef += coef;
            sumWeightedValue += value * coef;
        }
    }

    if (totalCoef > 0) {
        results[globalIdx] = static_cast<ResultT>(sumWeightedValue / totalCoef);
    } else {
        results[globalIdx] = static_cast<ResultT>(outputBlank);
    }
}

extern "C" void launchFullGPUTileKernelWithCoordsFloat(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    int numImages,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_imageMasks,
    int tileWidth,
    int numTiles,
    float* d_results,
    double outputBlank
) {
    int pixelsPerTile = tileWidth * tileWidth;
    size_t totalPixels = (size_t)numTiles * pixelsPerTile;

    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    fullGPUTileKernelWithCoordsTyped<float><<<numBlocks, blockSize>>>(
        d_allPixels, d_imageOffsets,
        d_crval1, d_crval2,
        d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2,
        d_cd2_1, d_cd2_2,
        d_widths, d_heights, d_blanks,
        numImages,
        d_raCoords, d_decCoords,
        d_imageMasks,
        tileWidth, numTiles,
        d_results, outputBlank
    );
}

extern "C" void launchFullGPUTileKernelWithCoordsDouble(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    int numImages,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_imageMasks,
    int tileWidth,
    int numTiles,
    double* d_results,
    double outputBlank
) {
    int pixelsPerTile = tileWidth * tileWidth;
    size_t totalPixels = (size_t)numTiles * pixelsPerTile;

    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    fullGPUTileKernelWithCoordsTyped<double><<<numBlocks, blockSize>>>(
        d_allPixels, d_imageOffsets,
        d_crval1, d_crval2,
        d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2,
        d_cd2_1, d_cd2_2,
        d_widths, d_heights, d_blanks,
        numImages,
        d_raCoords, d_decCoords,
        d_imageMasks,
        tileWidth, numTiles,
        d_results, outputBlank
    );
}

extern "C" void launchFullGPUTileKernelWithCoords(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    int numImages,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_imageMasks,
    int tileWidth,
    int numTiles,
    double* d_results,
    double outputBlank
) {
    launchFullGPUTileKernelWithCoordsDouble(
        d_allPixels, d_imageOffsets,
        d_crval1, d_crval2,
        d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2,
        d_cd2_1, d_cd2_2,
        d_widths, d_heights, d_blanks,
        numImages,
        d_raCoords, d_decCoords,
        d_imageMasks,
        tileWidth, numTiles,
        d_results, outputBlank
    );
}

extern "C" void launchFullGPUTileKernelWithCoordsSparseFloat(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_tileImageOffsets,
    const int* d_tileImageIndices,
    int tileWidth,
    int numTiles,
    float* d_results,
    double outputBlank
);


extern "C" void launchFullGPUTileKernelWithCoordsSparseAccum(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_tileImageOffsets,
    const int* d_tileImageIndices,
    int tileWidth,
    int numTiles,
    double* d_weightedSums,
    double* d_totalWeights
);

template <typename ResultT>
__global__ void fullGPUTileKernelWithCoordsSparseTyped(
    const float* __restrict__ allPixels,
    const size_t* __restrict__ imageOffsets,
    const double* __restrict__ crval1, const double* __restrict__ crval2,
    const double* __restrict__ crpix1, const double* __restrict__ crpix2,
    const double* __restrict__ cd1_1, const double* __restrict__ cd1_2,
    const double* __restrict__ cd2_1, const double* __restrict__ cd2_2,
    const int* __restrict__ widths, const int* __restrict__ heights,
    const float* __restrict__ blanks,
    const double* __restrict__ raCoords,
    const double* __restrict__ decCoords,
    const int* __restrict__ tileImageOffsets,
    const int* __restrict__ tileImageIndices,
    int tileWidth,
    int numTiles,
    ResultT* __restrict__ results,
    double outputBlank
) {
    size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    const int pixelsPerTile = tileWidth * tileWidth;
    const size_t totalPixels = (size_t)numTiles * pixelsPerTile;
    if (globalIdx >= totalPixels) return;

    const int tileIdx = globalIdx / pixelsPerTile;
    const int tileStart = tileImageOffsets[tileIdx];
    const int tileEnd = tileImageOffsets[tileIdx + 1];

    const double ra = raCoords[globalIdx];
    const double dec = decCoords[globalIdx];

    double sumWeightedValue = 0.0;
    double totalCoef = 0.0;

    for (int refIdx = tileStart; refIdx < tileEnd; ++refIdx) {
        const int img = tileImageIndices[refIdx];

        double pixelX, pixelY;
        celestialToPixel_device(
            ra, dec,
            crval1[img], crval2[img],
            crpix1[img], crpix2[img],
            cd1_1[img], cd1_2[img],
            cd2_1[img], cd2_2[img],
            pixelX, pixelY
        );

        double adjustedY = pixelY - 1;
        pixelX = pixelX - 1;

        if (pixelX < -0.5 || pixelX >= widths[img] - 0.5 ||
            adjustedY < -0.5 || adjustedY >= heights[img] - 0.5) {
            continue;
        }

        const float* imgPixels = allPixels + imageOffsets[img];
        const double value = bilinearInterp_device(
            imgPixels,
            widths[img], heights[img],
            pixelX, adjustedY,
            blanks[img]
        );

        if (!isBlank_device((float)value, blanks[img])) {
            double coef = 1.0;
            const int x1 = (int)pixelX;
            const int y1 = (int)adjustedY;
            const int w = widths[img];
            const int h = heights[img];

            if (x1 > 0 && x1 < w - 1 && (pixelX <= 0.5 || pixelX >= w - 1.5)) {
                coef *= 0.5;
            }
            if (y1 > 0 && y1 < h - 1 && (adjustedY <= 0.5 || adjustedY >= h - 1.5)) {
                coef *= 0.5;
            }

            totalCoef += coef;
            sumWeightedValue += value * coef;
        }
    }

    if (totalCoef > 0) {
        results[globalIdx] = static_cast<ResultT>(sumWeightedValue / totalCoef);
    } else {
        results[globalIdx] = static_cast<ResultT>(outputBlank);
    }
}


__global__ void fullGPUTileKernelWithCoordsSparseAccumKernel(
    const float* __restrict__ allPixels,
    const size_t* __restrict__ imageOffsets,
    const double* __restrict__ crval1, const double* __restrict__ crval2,
    const double* __restrict__ crpix1, const double* __restrict__ crpix2,
    const double* __restrict__ cd1_1, const double* __restrict__ cd1_2,
    const double* __restrict__ cd2_1, const double* __restrict__ cd2_2,
    const int* __restrict__ widths, const int* __restrict__ heights,
    const float* __restrict__ blanks,
    const double* __restrict__ raCoords,
    const double* __restrict__ decCoords,
    const int* __restrict__ tileImageOffsets,
    const int* __restrict__ tileImageIndices,
    int tileWidth,
    int numTiles,
    double* __restrict__ weightedSums,
    double* __restrict__ totalWeights
) {
    size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    const int pixelsPerTile = tileWidth * tileWidth;
    const size_t totalPixels = (size_t)numTiles * pixelsPerTile;
    if (globalIdx >= totalPixels) return;

    const int tileIdx = globalIdx / pixelsPerTile;
    const int tileStart = tileImageOffsets[tileIdx];
    const int tileEnd = tileImageOffsets[tileIdx + 1];

    const double ra = raCoords[globalIdx];
    const double dec = decCoords[globalIdx];

    double sumWeightedValue = 0.0;
    double totalCoef = 0.0;

    for (int refIdx = tileStart; refIdx < tileEnd; ++refIdx) {
        const int img = tileImageIndices[refIdx];

        double pixelX, pixelY;
        celestialToPixel_device(
            ra, dec,
            crval1[img], crval2[img],
            crpix1[img], crpix2[img],
            cd1_1[img], cd1_2[img],
            cd2_1[img], cd2_2[img],
            pixelX, pixelY
        );

        double adjustedY = pixelY - 1;
        pixelX = pixelX - 1;

        if (pixelX < -0.5 || pixelX >= widths[img] - 0.5 ||
            adjustedY < -0.5 || adjustedY >= heights[img] - 0.5) {
            continue;
        }

        const float* imgPixels = allPixels + imageOffsets[img];
        const double value = bilinearInterp_device(
            imgPixels,
            widths[img], heights[img],
            pixelX, adjustedY,
            blanks[img]
        );

        if (!isBlank_device((float)value, blanks[img])) {
            double coef = 1.0;
            const int x1 = (int)pixelX;
            const int y1 = (int)adjustedY;
            const int w = widths[img];
            const int h = heights[img];

            if (x1 > 0 && x1 < w - 1 && (pixelX <= 0.5 || pixelX >= w - 1.5)) {
                coef *= 0.5;
            }
            if (y1 > 0 && y1 < h - 1 && (adjustedY <= 0.5 || adjustedY >= h - 1.5)) {
                coef *= 0.5;
            }

            totalCoef += coef;
            sumWeightedValue += value * coef;
        }
    }

    weightedSums[globalIdx] = sumWeightedValue;
    totalWeights[globalIdx] = totalCoef;
}

extern "C" void launchFullGPUTileKernelWithCoordsSparseAccum(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_tileImageOffsets,
    const int* d_tileImageIndices,
    int tileWidth,
    int numTiles,
    double* d_weightedSums,
    double* d_totalWeights
) {
    const int pixelsPerTile = tileWidth * tileWidth;
    const size_t totalPixels = (size_t)numTiles * pixelsPerTile;
    const int blockSize = 256;
    const int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    fullGPUTileKernelWithCoordsSparseAccumKernel<<<numBlocks, blockSize>>>(
        d_allPixels, d_imageOffsets,
        d_crval1, d_crval2,
        d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2,
        d_cd2_1, d_cd2_2,
        d_widths, d_heights, d_blanks,
        d_raCoords, d_decCoords,
        d_tileImageOffsets, d_tileImageIndices,
        tileWidth, numTiles,
        d_weightedSums, d_totalWeights
    );
}

extern "C" void launchFullGPUTileKernelWithCoordsSparseFloat(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_tileImageOffsets,
    const int* d_tileImageIndices,
    int tileWidth,
    int numTiles,
    float* d_results,
    double outputBlank
) {
    const int pixelsPerTile = tileWidth * tileWidth;
    const size_t totalPixels = (size_t)numTiles * pixelsPerTile;
    const int blockSize = 256;
    const int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    fullGPUTileKernelWithCoordsSparseTyped<float><<<numBlocks, blockSize>>>(
        d_allPixels, d_imageOffsets,
        d_crval1, d_crval2,
        d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2,
        d_cd2_1, d_cd2_2,
        d_widths, d_heights, d_blanks,
        d_raCoords, d_decCoords,
        d_tileImageOffsets, d_tileImageIndices,
        tileWidth, numTiles,
        d_results, outputBlank
    );
}

extern "C" void launchFullGPUTileKernelWithCoordsSparseDouble(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    const double* d_raCoords,
    const double* d_decCoords,
    const int* d_tileImageOffsets,
    const int* d_tileImageIndices,
    int tileWidth,
    int numTiles,
    double* d_results,
    double outputBlank
) {
    const int pixelsPerTile = tileWidth * tileWidth;
    const size_t totalPixels = (size_t)numTiles * pixelsPerTile;
    const int blockSize = 256;
    const int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    fullGPUTileKernelWithCoordsSparseTyped<double><<<numBlocks, blockSize>>>(
        d_allPixels, d_imageOffsets,
        d_crval1, d_crval2,
        d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2,
        d_cd2_1, d_cd2_2,
        d_widths, d_heights, d_blanks,
        d_raCoords, d_decCoords,
        d_tileImageOffsets, d_tileImageIndices,
        tileWidth, numTiles,
        d_results, outputBlank
    );
}

// Keep old launcher for backward compatibility (will be removed later)
extern "C" void launchFullGPUTileKernel(
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    int numImages,
    const int* d_xy2hpx,
    const int* d_tileOrders,
    const long* d_tileNpixs,
    const int* d_imageMasks,
    int tileWidth,
    int tileOrder,
    int numTiles,
    double* d_results,
    double outputBlank,
    double validMin,
    double validMax
) {
    // This is a stub - the old kernel has bugs in HEALPix conversion
    // Use launchFullGPUTileKernelWithCoords instead
    fprintf(stderr, "WARNING: launchFullGPUTileKernel is deprecated. Use launchFullGPUTileKernelWithCoords.\n");
}

