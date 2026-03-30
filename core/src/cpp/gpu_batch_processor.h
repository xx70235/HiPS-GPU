/**
 * GPU Batch Processor for HiPS Tile Generation
 * Process ALL tiles in a single GPU kernel call for maximum efficiency
 */

#ifndef GPU_BATCH_PROCESSOR_H
#define GPU_BATCH_PROCESSOR_H

#include <vector>
#include <string>
#include "fits_io.h"
#include "coordinate_transform.h"
#include "hips_tile_generator.h"

/**
 * Tile info for batch processing
 */
struct TileInfo {
    int order;
    long npix;
    int startIdx;  // Start index in the global results array
};

/**
 * GPU Batch Processor
 * Processes all valid tiles in minimal GPU operations
 */
class GPUBatchProcessor {
public:
    GPUBatchProcessor();
    ~GPUBatchProcessor();
    
    /**
     * Initialize with source images
     * @param fitsFiles All source FITS files
     * @param transforms Coordinate transforms for each image
     * @return true if successful
     */
    bool initialize(
        const std::vector<FitsData>& fitsFiles,
        const std::vector<CoordinateTransform>& transforms
    );
    
    /**
     * Process all tiles in batch
     * @param validTiles List of tiles to process
     * @param tileWidth Tile width (512)
     * @param xy2hpx Pixel to HEALPix mapping
     * @param blank Blank value
     * @param results Output: results for each tile [numTiles][tileWidth*tileWidth]
     * @return true if successful
     */
    bool processTilesBatch(
        const std::vector<TileInfo>& validTiles,
        int tileWidth,
        const std::vector<int>& xy2hpx,
        double blank,
        std::vector<std::vector<double>>& results
    );
    
    /**
     * Release GPU resources
     */
    void release();
    
    /**
     * Check if GPU is available
     */
    bool isGPUAvailable() const { return m_gpuAvailable; }

private:
    // Source images on GPU
    float* m_d_allPixels;
    int* m_d_imageWidths;
    int* m_d_imageHeights;
    size_t* m_d_imageOffsets;
    float* m_d_imageBlanks;
    
    // Host copies for coordinate transforms
    std::vector<CoordinateTransform> m_transforms;
    std::vector<int> m_imageWidths;
    std::vector<int> m_imageHeights;
    
    int m_numImages;
    size_t m_totalImagePixels;
    bool m_gpuAvailable;
    bool m_initialized;
};

/**
 * CUDA kernel launcher for batch tile processing
 */
extern "C" void launchBatchTileInterpolationKernel(
    const float* d_allPixels,
    const int* d_imageWidths,
    const int* d_imageHeights,
    const size_t* d_imageOffsets,
    const float* d_imageBlanks,
    int numImages,
    const double* d_coordsX,      // [totalCoords * numImages]
    const double* d_coordsY,
    const int* d_inBounds,
    double* d_results,            // [totalCoords]
    size_t totalCoords,
    double outputBlank
);

#endif // GPU_BATCH_PROCESSOR_H
