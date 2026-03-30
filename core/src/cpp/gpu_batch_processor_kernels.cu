/**
 * GPU Batch Processor CUDA Kernels
 * Optimized for processing ALL tile coordinates in a single kernel call
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * Device function: Check if a value is blank (handles NaN blank)
 */
__device__ inline bool isBlank_batch(float value, float blank) {
    if (isnan(blank)) {
        return isnan(value);
    }
    return isnan(value) || value == blank;
}

/**
 * Device function: Bilinear interpolation
 */
__device__ inline double bilinearInterpDevice(
    const float* __restrict__ pixels,
    int width, int height,
    double x, double y,
    float blank
) {
    if (x < 0.0 || x >= (double)(width - 1) || y < 0.0 || y >= (double)(height - 1)) {
        return (double)blank;
    }
    
    int ix = (int)x;
    int iy = (int)y;
    
    double fx = x - (double)ix;
    double fy = y - (double)iy;
    
    float v00 = pixels[iy * width + ix];
    float v10 = pixels[iy * width + (ix + 1)];
    float v01 = pixels[(iy + 1) * width + ix];
    float v11 = pixels[(iy + 1) * width + (ix + 1)];
    
    // Check blank values (handles NaN blank)
    if (isBlank_batch(v00, blank)) return (double)blank;
    if (isBlank_batch(v10, blank)) return (double)blank;
    if (isBlank_batch(v01, blank)) return (double)blank;
    if (isBlank_batch(v11, blank)) return (double)blank;
    
    double v0 = (double)v00 * (1.0 - fx) + (double)v10 * fx;
    double v1 = (double)v01 * (1.0 - fx) + (double)v11 * fx;
    return v0 * (1.0 - fy) + v1 * fy;
}

/**
 * Main batch interpolation kernel
 * Each thread processes ONE output coordinate (one pixel in one tile)
 * Iterates through all source images to compute the coadd value
 * 
 * Memory layout:
 * - coordsX/Y/inBounds: [coord0_img0, coord0_img1, ..., coord1_img0, coord1_img1, ...]
 *   i.e., for coordinate i and image j: index = i * numImages + j
 */
__global__ void batchTileInterpolationKernel(
    const float* __restrict__ allPixels,
    const int* __restrict__ imageWidths,
    const int* __restrict__ imageHeights,
    const size_t* __restrict__ imageOffsets,
    const float* __restrict__ imageBlanks,
    int numImages,
    const double* __restrict__ coordsX,
    const double* __restrict__ coordsY,
    const int* __restrict__ inBounds,
    double* __restrict__ results,
    size_t totalCoords,
    double outputBlank
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalCoords) return;
    
    double totalValue = 0.0;
    int validCount = 0;
    
    // Base index for this coordinate's image data
    size_t baseIdx = idx * numImages;
    
    // Iterate through all source images
    for (int img = 0; img < numImages; img++) {
        size_t coordIdx = baseIdx + img;
        
        // Skip if not in bounds
        if (inBounds[coordIdx] == 0) continue;
        
        double x = coordsX[coordIdx];
        double y = coordsY[coordIdx];
        
        int width = imageWidths[img];
        int height = imageHeights[img];
        float blank = imageBlanks[img];
        size_t offset = imageOffsets[img];
        
        const float* imagePixels = allPixels + offset;
        
        double value = bilinearInterpDevice(imagePixels, width, height, x, y, blank);
        
        // Check if value is valid (not blank)
        if (!isBlank_batch((float)value, blank) && value != outputBlank) {
            totalValue += value;
            validCount++;
        }
    }
    
    // Compute mean
    if (validCount > 0) {
        results[idx] = totalValue / (double)validCount;
    } else {
        results[idx] = outputBlank;
    }
}

/**
 * Kernel launcher
 */
extern "C" void launchBatchTileInterpolationKernel(
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
    size_t totalCoords,
    double outputBlank
) {
    if (totalCoords == 0) return;
    
    int blockSize = 256;
    size_t numBlocks = (totalCoords + blockSize - 1) / blockSize;
    
    // Limit to max grid size
    if (numBlocks > 65535) numBlocks = 65535;
    
    batchTileInterpolationKernel<<<(int)numBlocks, blockSize>>>(
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
        totalCoords,
        outputBlank
    );
}
