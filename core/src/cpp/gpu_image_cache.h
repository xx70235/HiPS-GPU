/**
 * GPU Image Cache Manager
 * Preload all source images to GPU memory for efficient batch processing
 */

#ifndef GPU_IMAGE_CACHE_H
#define GPU_IMAGE_CACHE_H

#include <vector>
#include <string>
#include <unordered_map>
#include "fits_io.h"

/**
 * GPU cached image info
 */
struct GPUImageInfo {
    int width;
    int height;
    float blank;
    float* d_pixels;  // Device pointer
    size_t offset;    // Offset in unified buffer
};

/**
 * GPU Image Cache - manages all source images on GPU
 */
class GPUImageCache {
public:
    GPUImageCache();
    ~GPUImageCache();
    
    /**
     * Load all source images to GPU memory
     * @param fitsFiles Vector of FitsData from source files
     * @return true if successful
     */
    bool loadToGPU(const std::vector<FitsData>& fitsFiles);
    
    /**
     * Batch interpolate from all cached images
     * For each output coordinate, find the best value from all source images
     * 
     * @param celestialRA RA coordinates (radians)
     * @param celestialDec Dec coordinates (radians)
     * @param pixelCoordsX Output: pixel X coords for each image (numImages * numCoords)
     * @param pixelCoordsY Output: pixel Y coords for each image
     * @param numCoords Number of output coordinates
     * @param results Output values
     * @param blank Blank value for output
     */
    bool batchInterpolate(
        const double* celestialRA,
        const double* celestialDec,
        const std::vector<std::vector<double>>& pixelCoordsX,
        const std::vector<std::vector<double>>& pixelCoordsY,
        const std::vector<std::vector<bool>>& inBounds,
        int numCoords,
        double* results,
        double blank
    );
    
    /**
     * Release GPU memory
     */
    void release();
    
    /**
     * Check if GPU cache is available
     */
    bool isAvailable() const { return m_available; }
    
    /**
     * Get number of cached images
     */
    int getImageCount() const { return (int)m_images.size(); }
    
    /**
     * Get total GPU memory used (bytes)
     */
    size_t getGPUMemoryUsed() const { return m_totalMemory; }

private:
    std::vector<GPUImageInfo> m_images;
    float* m_d_allPixels;      // Unified device buffer for all images
    size_t m_totalMemory;
    bool m_available;
};

/**
 * Launch batch interpolation kernel for multiple images
 * Each thread processes one output coordinate, iterating through all images
 */
extern "C" void launchMultiImageInterpolationKernel(
    const float* d_allPixels,
    const int* d_imageWidths,
    const int* d_imageHeights,
    const size_t* d_imageOffsets,
    const float* d_imageBlanks,
    int numImages,
    const double* d_coordsX,  // numImages * numCoords
    const double* d_coordsY,
    const int* d_inBounds,    // 1 if in bounds, 0 otherwise
    double* d_results,
    int numCoords,
    double outputBlank
);

#endif // GPU_IMAGE_CACHE_H
