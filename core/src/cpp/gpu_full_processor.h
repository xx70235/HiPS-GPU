/**
 * GPU Full Processor for HiPS Tile Generation
 * Uses official HEALPix library (healpix_cxx) for coordinate computation
 * GPU handles WCS projection and interpolation
 */

#ifndef GPU_FULL_PROCESSOR_H
#define GPU_FULL_PROCESSOR_H

#include <vector>
#include <string>
#include "fits_io.h"
#include "coordinate_transform.h"

/**
 * WCS parameters for GPU (plain struct, no methods)
 */
struct WCSParams {
    double crval1, crval2;  // RA/Dec reference (degrees)
    double crpix1, crpix2;  // Reference pixel
    double cd1_1, cd1_2;    // CD matrix
    double cd2_1, cd2_2;
    int width, height;      // Image dimensions
    float blank;            // Blank value
};

/**
 * Tile info for processing
 */
struct GPUTileInfo {
    int order;
    long npix;
    std::vector<int> imageIndices;  // Indices of source images that cover this tile
};

/**
 * GPU Full Processor
 * HEALPix coordinates computed on CPU using official library
 * WCS projection and interpolation computed on GPU
 */
class GPUFullProcessor {
public:
    GPUFullProcessor();
    ~GPUFullProcessor();
    
    /**
     * Initialize with source images and WCS parameters
     * @param fitsFiles All source FITS files
     * @return true if successful
     */
    bool initialize(const std::vector<FitsData>& fitsFiles);
    
    /**
     * Process all tiles using GPU for WCS projection and interpolation
     * @param tiles List of tiles to process
     * @param tileWidth Tile width (512)
     * @param xy2hpx Pixel to HEALPix mapping
     * @param blank Output blank value
     * @param results Output: results for each tile
     * @return true if successful
     */
    bool processAllTiles(
        const std::vector<GPUTileInfo>& tiles,
        int tileWidth,
        const std::vector<int>& xy2hpx,
        double blank,
        std::vector<std::vector<double>>& results,
        double validMin = -std::numeric_limits<double>::infinity(),
        double validMax = std::numeric_limits<double>::infinity()
    );
    
    void release();
    
private:
    /**
     * Compute celestial coordinates for all pixels using official HEALPix library
     * This ensures coordinates match the Java reference implementation
     */
    void computeCelestialCoords(
        const std::vector<GPUTileInfo>& tiles,
        int tileWidth,
        const std::vector<int>& xy2hpx,
        std::vector<double>& raCoords,
        std::vector<double>& decCoords
    );

    // GPU memory for source images
    float* m_d_allPixels;
    size_t* m_d_imageOffsets;
    
    // GPU memory for WCS parameters (as arrays)
    double* m_d_crval1;
    double* m_d_crval2;
    double* m_d_crpix1;
    double* m_d_crpix2;
    double* m_d_cd1_1;
    double* m_d_cd1_2;
    double* m_d_cd2_1;
    double* m_d_cd2_2;
    int* m_d_widths;
    int* m_d_heights;
    float* m_d_blanks;
    
    // Host data
    std::vector<WCSParams> m_wcsParams;
    int m_numImages;
    size_t m_totalPixels;
    bool m_initialized;
};

/**
 * CUDA kernel launcher with precomputed coordinates
 * Uses official HEALPix library coordinates from CPU
 */
extern "C" void launchFullGPUTileKernelWithCoords(
    // Source images
    const float* d_allPixels,
    const size_t* d_imageOffsets,
    // WCS parameters (per image)
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_widths, const int* d_heights,
    const float* d_blanks,
    int numImages,
    // Precomputed celestial coordinates (from official HEALPix library)
    const double* d_raCoords,   // [numTiles * tileWidth * tileWidth]
    const double* d_decCoords,  // [numTiles * tileWidth * tileWidth]
    // Tile parameters
    const int* d_imageMasks,    // [numTiles * numImages] - which images to process for each tile
    int tileWidth,
    int numTiles,
    // Output
    double* d_results,          // [numTiles * tileWidth * tileWidth]
    double outputBlank
);

// Legacy kernel launcher (deprecated - has bugs in HEALPix conversion)
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
);

#endif // GPU_FULL_PROCESSOR_H
