/**
 * GPU Image Cache CUDA Kernels
 * Multi-image batch interpolation for HiPS tile generation
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * Device function: Check if a value is blank (handles NaN blank)
 */
__device__ inline bool isBlank_cache(float value, float blank) {
    if (isnan(blank)) {
        return isnan(value);
    }
    return isnan(value) || value == blank;
}

/**
 * Bilinear interpolation device function
 */
__device__ double bilinearInterp(
    const float* pixels,
    int width, int height,
    double x, double y,
    float blank
) {
    // Check bounds
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
        return blank;
    }
    
    int ix = (int)x;
    int iy = (int)y;
    
    double fx = x - ix;
    double fy = y - iy;
    
    // Get four corner values
    float v00 = pixels[iy * width + ix];
    float v10 = pixels[iy * width + (ix + 1)];
    float v01 = pixels[(iy + 1) * width + ix];
    float v11 = pixels[(iy + 1) * width + (ix + 1)];
    
    // Check for blank values (handles NaN blank)
    if (isBlank_cache(v00, blank)) return blank;
    if (isBlank_cache(v10, blank)) return blank;
    if (isBlank_cache(v01, blank)) return blank;
    if (isBlank_cache(v11, blank)) return blank;
    
    // Bilinear interpolation
    double v0 = v00 * (1.0 - fx) + v10 * fx;
    double v1 = v01 * (1.0 - fx) + v11 * fx;
    return v0 * (1.0 - fy) + v1 * fy;
}

/**
 * Multi-image batch interpolation kernel
 * Each thread handles one output pixel, iterating through all source images
 */
__global__ void multiImageInterpolationKernel(
    const float* __restrict__ allPixels,
    const int* __restrict__ imageWidths,
    const int* __restrict__ imageHeights,
    const size_t* __restrict__ imageOffsets,
    const float* __restrict__ imageBlanks,
    int numImages,
    const double* __restrict__ coordsX,    // [numImages][numCoords] flattened
    const double* __restrict__ coordsY,
    const int* __restrict__ inBoundsFlags, // [numImages][numCoords] flattened
    double* __restrict__ results,
    int numCoords,
    double outputBlank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCoords) return;
    
    double totalValue = 0.0;
    double totalWeight = 0.0;
    int validCount = 0;
    
    // Iterate through all source images
    for (int img = 0; img < numImages; img++) {
        int coordIdx = img * numCoords + idx;
        
        // Check if this coordinate is in bounds for this image
        if (inBoundsFlags[coordIdx] == 0) continue;
        
        double x = coordsX[coordIdx];
        double y = coordsY[coordIdx];
        
        // Get image info
        int width = imageWidths[img];
        int height = imageHeights[img];
        float blank = imageBlanks[img];
        size_t offset = imageOffsets[img];
        
        // Get pixel pointer for this image
        const float* imagePixels = allPixels + offset;
        
        // Interpolate
        double value = bilinearInterp(imagePixels, width, height, x, y, blank);
        
        // Accumulate if valid (handles NaN blank)
        if (!isBlank_cache((float)value, blank) && value != outputBlank) {
            totalValue += value;
            totalWeight += 1.0;
            validCount++;
        }
    }
    
    // Compute average
    if (validCount > 0 && totalWeight > 0.0) {
        results[idx] = totalValue / totalWeight;
    } else {
        results[idx] = outputBlank;
    }
}

/**
 * Launch the multi-image interpolation kernel
 */
extern "C" void launchMultiImageInterpolationKernel(
    const float* d_allPixels,
    const int* d_imageWidths,
    const int* d_imageHeights,
    const size_t* d_imageOffsets,
    const float* d_imageBlanks,
    int numImages,
    const double* d_coordsX,
    const double* d_coordsY,
    const int* d_inBounds,
    double* d_results,
    int numCoords,
    double outputBlank
) {
    int blockSize = 256;
    int numBlocks = (numCoords + blockSize - 1) / blockSize;
    
    multiImageInterpolationKernel<<<numBlocks, blockSize>>>(
        d_allPixels,
        d_imageWidths,
        d_imageHeights,
        d_imageOffsets,
        d_imageBlanks,
        numImages,
        d_coordsX,
        d_coordsY,
        d_inBounds,
        d_results,
        numCoords,
        outputBlank
    );
}
