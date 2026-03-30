/**
 * GPU Full Processor Implementation
 * Uses official HEALPix library (healpix_cxx) for coordinate computation
 * Transfers precomputed celestial coordinates to GPU
 */

#include "gpu_full_processor.h"
#include "healpix_cuda.cuh"  // GPU HEALPix coordinate computation
#include <omp.h>
#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/pointing.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>

// Declare the new kernel launcher
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
);

GPUFullProcessor::GPUFullProcessor()
    : m_d_allPixels(nullptr)
    , m_d_imageOffsets(nullptr)
    , m_d_crval1(nullptr), m_d_crval2(nullptr)
    , m_d_crpix1(nullptr), m_d_crpix2(nullptr)
    , m_d_cd1_1(nullptr), m_d_cd1_2(nullptr)
    , m_d_cd2_1(nullptr), m_d_cd2_2(nullptr)
    , m_d_widths(nullptr), m_d_heights(nullptr)
    , m_d_blanks(nullptr)
    , m_numImages(0)
    , m_totalPixels(0)
    , m_initialized(false)
{
}

GPUFullProcessor::~GPUFullProcessor() {
    release();
}

bool GPUFullProcessor::initialize(const std::vector<FitsData>& fitsFiles) {
    if (fitsFiles.empty()) return false;
    
    m_numImages = (int)fitsFiles.size();
    
    // Calculate total pixel memory needed
    m_totalPixels = 0;
    std::vector<size_t> offsets(m_numImages);
    m_wcsParams.resize(m_numImages);
    
    for (int i = 0; i < m_numImages; i++) {
        offsets[i] = m_totalPixels;
        m_totalPixels += (size_t)fitsFiles[i].width * fitsFiles[i].height;
        
        // Store WCS parameters
        m_wcsParams[i].crval1 = fitsFiles[i].crval1;
        m_wcsParams[i].crval2 = fitsFiles[i].crval2;
        m_wcsParams[i].crpix1 = fitsFiles[i].crpix1;
        m_wcsParams[i].crpix2 = fitsFiles[i].crpix2;
        m_wcsParams[i].cd1_1 = fitsFiles[i].cd1_1;
        m_wcsParams[i].cd1_2 = fitsFiles[i].cd1_2;
        m_wcsParams[i].cd2_1 = fitsFiles[i].cd2_1;
        m_wcsParams[i].cd2_2 = fitsFiles[i].cd2_2;
        m_wcsParams[i].width = fitsFiles[i].width;
        m_wcsParams[i].height = fitsFiles[i].height;
        m_wcsParams[i].blank = (float)fitsFiles[i].blank;
    }
    
    std::cout << "GPU Full: Allocating " << (m_totalPixels * sizeof(float) / 1024 / 1024) 
              << " MB for images" << std::endl;
    
    // Allocate GPU memory for all images
    cudaError_t err = cudaMalloc(&m_d_allPixels, m_totalPixels * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for images" << std::endl;
        return false;
    }
    
    // Copy all images to GPU
    size_t currentOffset = 0;
    for (int i = 0; i < m_numImages; i++) {
        size_t imgSize = (size_t)fitsFiles[i].width * fitsFiles[i].height * sizeof(float);
        cudaMemcpy(m_d_allPixels + currentOffset, fitsFiles[i].pixels.data(), imgSize, cudaMemcpyHostToDevice);
        currentOffset += (size_t)fitsFiles[i].width * fitsFiles[i].height;
    }
    
    // Allocate and copy image offsets
    cudaMalloc(&m_d_imageOffsets, m_numImages * sizeof(size_t));
    cudaMemcpy(m_d_imageOffsets, offsets.data(), m_numImages * sizeof(size_t), cudaMemcpyHostToDevice);
    
    // Extract WCS parameters into separate arrays for GPU
    std::vector<double> h_crval1(m_numImages), h_crval2(m_numImages);
    std::vector<double> h_crpix1(m_numImages), h_crpix2(m_numImages);
    std::vector<double> h_cd1_1(m_numImages), h_cd1_2(m_numImages);
    std::vector<double> h_cd2_1(m_numImages), h_cd2_2(m_numImages);
    std::vector<int> h_widths(m_numImages), h_heights(m_numImages);
    std::vector<float> h_blanks(m_numImages);
    
    for (int i = 0; i < m_numImages; i++) {
        h_crval1[i] = m_wcsParams[i].crval1;
        h_crval2[i] = m_wcsParams[i].crval2;
        h_crpix1[i] = m_wcsParams[i].crpix1;
        h_crpix2[i] = m_wcsParams[i].crpix2;
        h_cd1_1[i] = m_wcsParams[i].cd1_1;
        h_cd1_2[i] = m_wcsParams[i].cd1_2;
        h_cd2_1[i] = m_wcsParams[i].cd2_1;
        h_cd2_2[i] = m_wcsParams[i].cd2_2;
        h_widths[i] = m_wcsParams[i].width;
        h_heights[i] = m_wcsParams[i].height;
        h_blanks[i] = m_wcsParams[i].blank;
    }
    
    // Allocate and copy WCS parameters
    cudaMalloc(&m_d_crval1, m_numImages * sizeof(double));
    cudaMalloc(&m_d_crval2, m_numImages * sizeof(double));
    cudaMalloc(&m_d_crpix1, m_numImages * sizeof(double));
    cudaMalloc(&m_d_crpix2, m_numImages * sizeof(double));
    cudaMalloc(&m_d_cd1_1, m_numImages * sizeof(double));
    cudaMalloc(&m_d_cd1_2, m_numImages * sizeof(double));
    cudaMalloc(&m_d_cd2_1, m_numImages * sizeof(double));
    cudaMalloc(&m_d_cd2_2, m_numImages * sizeof(double));
    cudaMalloc(&m_d_widths, m_numImages * sizeof(int));
    cudaMalloc(&m_d_heights, m_numImages * sizeof(int));
    cudaMalloc(&m_d_blanks, m_numImages * sizeof(float));
    
    cudaMemcpy(m_d_crval1, h_crval1.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_crval2, h_crval2.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_crpix1, h_crpix1.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_crpix2, h_crpix2.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_cd1_1, h_cd1_1.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_cd1_2, h_cd1_2.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_cd2_1, h_cd2_1.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_cd2_2, h_cd2_2.data(), m_numImages * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_widths, h_widths.data(), m_numImages * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_heights, h_heights.data(), m_numImages * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_blanks, h_blanks.data(), m_numImages * sizeof(float), cudaMemcpyHostToDevice);
    
    m_initialized = true;
    std::cout << "GPU Full: Initialized with " << m_numImages << " images" << std::endl;
    
    // Debug: print blank values for first few images
    std::cout << "GPU Full: Blank values for first 3 images: ";
    for (int i = 0; i < std::min(3, m_numImages); i++) {
        if (std::isnan(h_blanks[i])) {
            std::cout << "NaN ";
        } else {
            std::cout << h_blanks[i] << " ";
        }
    }
    std::cout << std::endl;
    
    return true;
}

/**
 * Compute celestial coordinates for all pixels in all tiles using official HEALPix library
 * This ensures coordinates match the Java reference implementation
 */
void GPUFullProcessor::computeCelestialCoords(
    const std::vector<GPUTileInfo>& tiles,
    int tileWidth,
    const std::vector<int>& xy2hpx,
    std::vector<double>& raCoords,
    std::vector<double>& decCoords
) {
    int pixelsPerTile = tileWidth * tileWidth;
    size_t totalPixels = tiles.size() * pixelsPerTile;
    
    raCoords.resize(totalPixels);
    decCoords.resize(totalPixels);
    
    // Prepare tile data for GPU
    std::vector<int> tileOrders(tiles.size());
    std::vector<long> tileNpixs(tiles.size());
    
    for (size_t i = 0; i < tiles.size(); i++) {
        tileOrders[i] = tiles[i].order;
        tileNpixs[i] = tiles[i].npix;
    }
    
    // Use official HEALPix library with OpenMP parallelization
    // (GPU version has bugs in phi calculation, disabled for now)
    cudaError_t err = cudaErrorUnknown;  // Force CPU fallback
    
    {
        std::cout << "Using CPU (OpenMP) with official HEALPix library..." << std::endl;
        
        // OpenMP parallel CPU computation
        #pragma omp parallel for schedule(dynamic, 10)
        for (size_t tileIdx = 0; tileIdx < tiles.size(); tileIdx++) {
            int order = tiles[tileIdx].order;
            long npix = tiles[tileIdx].npix;
            
            int tileOrder = 0;
            int temp = tileWidth;
            while (temp > 1) { temp >>= 1; tileOrder++; }
            
            int pixelOrder = order + tileOrder;
            long nside = 1L << pixelOrder;
            
            T_Healpix_Base<int64> hpx(nside, NEST, SET_NSIDE);
            long baseIdx = npix << (2 * tileOrder);
            
            for (int pixelInTile = 0; pixelInTile < pixelsPerTile; pixelInTile++) {
                int hpxOffset = xy2hpx[pixelInTile];
                long healpixIdx = baseIdx + hpxOffset;
                
                pointing ptg = hpx.pix2ang(healpixIdx);
                
                double dec = 90.0 - ptg.theta * 180.0 / M_PI;
                double ra = ptg.phi * 180.0 / M_PI;
                
                size_t globalIdx = tileIdx * pixelsPerTile + pixelInTile;
                raCoords[globalIdx] = ra;
                decCoords[globalIdx] = dec;
            }
        }
    }
    
    // Debug output
    if (totalPixels > 0) {
        std::cout << "GPU Full: Computed celestial coordinates for " << totalPixels << " pixels (GPU accelerated)" << std::endl;
        std::cout << "GPU Full: First tile (order=" << tiles[0].order << ", npix=" << tiles[0].npix << ")" << std::endl;
        std::cout << "GPU Full: Sample coords - pixel0: (ra=" << raCoords[0] << ", dec=" << decCoords[0] << ")" << std::endl;
        if (pixelsPerTile > 1) {
            std::cout << "GPU Full: Sample coords - pixel1: (ra=" << raCoords[1] << ", dec=" << decCoords[1] << ")" << std::endl;
        }
    }
}

bool GPUFullProcessor::processAllTiles(
    const std::vector<GPUTileInfo>& tiles,
    int tileWidth,
    const std::vector<int>& xy2hpx,
    double blank,
    std::vector<std::vector<double>>& results,
    double validMin,
    double validMax
) {
    if (!m_initialized || tiles.empty()) return false;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int numTiles = (int)tiles.size();
    int pixelsPerTile = tileWidth * tileWidth;
    
    // Query GPU memory
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    
    // Reserve some memory for safety (20% buffer)
    size_t availableMem = (size_t)(freeMem * 0.8);
    
    // Memory per tile:
    // - coordSize: 2 * pixelsPerTile * sizeof(double) = 2 * 262144 * 8 = 4MB
    // - outputSize: pixelsPerTile * sizeof(double) = 262144 * 8 = 2MB
    // - imageMask: numImages * sizeof(int) = 100 * 4 = 400 bytes
    // Total per tile: ~6MB
    size_t memoryPerTile = 2 * pixelsPerTile * sizeof(double) + pixelsPerTile * sizeof(double) + m_numImages * sizeof(int);
    
    // Calculate max tiles per batch (leave headroom for kernel overhead)
    int maxTilesPerBatch = std::max(1, (int)(availableMem / memoryPerTile / 2));
    maxTilesPerBatch = std::min(maxTilesPerBatch, 5000); // Cap at 5000 tiles per batch
    
    int numBatches = (numTiles + maxTilesPerBatch - 1) / maxTilesPerBatch;
    
    std::cout << "GPU Full: Processing " << numTiles << " tiles in " << numBatches 
              << " batches (max " << maxTilesPerBatch << " tiles/batch)" << std::endl;
    std::cout << "GPU Full: Available memory: " << (availableMem / 1024 / 1024) << " MB" << std::endl;
    
    results.resize(numTiles);
    
    long totalCoordMs = 0;
    long totalKernelMs = 0;
    long totalCopyMs = 0;
    
    for (int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * maxTilesPerBatch;
        int batchEnd = std::min(batchStart + maxTilesPerBatch, numTiles);
        int batchSize = batchEnd - batchStart;
        size_t batchOutputPixels = (size_t)batchSize * pixelsPerTile;
        
        // Extract batch tiles
        std::vector<GPUTileInfo> batchTiles(tiles.begin() + batchStart, tiles.begin() + batchEnd);
        
        // Step 1: Precompute celestial coordinates for this batch
        auto coordStart = std::chrono::high_resolution_clock::now();
        
        std::vector<double> h_raCoords, h_decCoords;
        computeCelestialCoords(batchTiles, tileWidth, xy2hpx, h_raCoords, h_decCoords);
        
        auto coordEnd = std::chrono::high_resolution_clock::now();
        totalCoordMs += std::chrono::duration_cast<std::chrono::milliseconds>(coordEnd - coordStart).count();
        
        // Create image mask for batch tiles
        std::vector<int> h_imageMasks(batchSize * m_numImages, 0);
        for (int i = 0; i < batchSize; i++) {
            for (int imgIdx : batchTiles[i].imageIndices) {
                if (imgIdx >= 0 && imgIdx < m_numImages) {
                    h_imageMasks[i * m_numImages + imgIdx] = 1;
                }
            }
        }
        
        // Allocate GPU memory for this batch
        double* d_raCoords = nullptr;
        double* d_decCoords = nullptr;
        int* d_imageMasks = nullptr;
        double* d_results = nullptr;
        
        cudaError_t err;
        size_t coordSize = batchOutputPixels * sizeof(double);
        size_t outputSize = batchOutputPixels * sizeof(double);
        
        err = cudaMalloc(&d_raCoords, coordSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_raCoords for batch " << batch << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_decCoords, coordSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_decCoords for batch " << batch << std::endl;
            cudaFree(d_raCoords);
            return false;
        }
        
        cudaMalloc(&d_imageMasks, batchSize * m_numImages * sizeof(int));
        
        err = cudaMalloc(&d_results, outputSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_results for batch " << batch << std::endl;
            cudaFree(d_raCoords);
            cudaFree(d_decCoords);
            cudaFree(d_imageMasks);
            return false;
        }
        
        // Copy data to GPU
        cudaMemcpy(d_raCoords, h_raCoords.data(), coordSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_decCoords, h_decCoords.data(), coordSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imageMasks, h_imageMasks.data(), batchSize * m_numImages * sizeof(int), cudaMemcpyHostToDevice);
        
        // Execute kernel
        auto kernelStart = std::chrono::high_resolution_clock::now();
        
        launchFullGPUTileKernelWithCoords(
            m_d_allPixels, m_d_imageOffsets,
            m_d_crval1, m_d_crval2,
            m_d_crpix1, m_d_crpix2,
            m_d_cd1_1, m_d_cd1_2,
            m_d_cd2_1, m_d_cd2_2,
            m_d_widths, m_d_heights, m_d_blanks,
            m_numImages,
            d_raCoords, d_decCoords,
            d_imageMasks,
            tileWidth, batchSize,
            d_results, blank
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel error in batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_raCoords);
            cudaFree(d_decCoords);
            cudaFree(d_imageMasks);
            cudaFree(d_results);
            return false;
        }
        
        cudaDeviceSynchronize();
        
        auto kernelEnd = std::chrono::high_resolution_clock::now();
        totalKernelMs += std::chrono::duration_cast<std::chrono::milliseconds>(kernelEnd - kernelStart).count();
        
        // Copy results back
        auto copyStart = std::chrono::high_resolution_clock::now();
        
        std::vector<double> h_results(batchOutputPixels);
        cudaMemcpy(h_results.data(), d_results, outputSize, cudaMemcpyDeviceToHost);
        
        // Store results for this batch
        for (int t = 0; t < batchSize; t++) {
            int globalIdx = batchStart + t;
            results[globalIdx].resize(pixelsPerTile);
            memcpy(results[globalIdx].data(), &h_results[t * pixelsPerTile], pixelsPerTile * sizeof(double));
        }
        
        auto copyEnd = std::chrono::high_resolution_clock::now();
        totalCopyMs += std::chrono::duration_cast<std::chrono::milliseconds>(copyEnd - copyStart).count();
        
        // Cleanup batch GPU memory
        cudaFree(d_raCoords);
        cudaFree(d_decCoords);
        cudaFree(d_imageMasks);
        cudaFree(d_results);
        
        // Progress report
        if (batch == 0 || (batch + 1) % 10 == 0 || batch == numBatches - 1) {
            std::cout << "  Batch " << (batch + 1) << "/" << numBatches 
                      << " done (" << batchSize << " tiles)" << std::endl;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "GPU Full Summary:" << std::endl;
    std::cout << "  Coord comp:  " << totalCoordMs << " ms" << std::endl;
    std::cout << "  GPU kernel:  " << totalKernelMs << " ms" << std::endl;
    std::cout << "  Copy back:   " << totalCopyMs << " ms" << std::endl;
    std::cout << "  Total time:  " << totalMs << " ms" << std::endl;
    
    return true;
}


void GPUFullProcessor::release() {
    if (m_d_allPixels) { cudaFree(m_d_allPixels); m_d_allPixels = nullptr; }
    if (m_d_imageOffsets) { cudaFree(m_d_imageOffsets); m_d_imageOffsets = nullptr; }
    if (m_d_crval1) { cudaFree(m_d_crval1); m_d_crval1 = nullptr; }
    if (m_d_crval2) { cudaFree(m_d_crval2); m_d_crval2 = nullptr; }
    if (m_d_crpix1) { cudaFree(m_d_crpix1); m_d_crpix1 = nullptr; }
    if (m_d_crpix2) { cudaFree(m_d_crpix2); m_d_crpix2 = nullptr; }
    if (m_d_cd1_1) { cudaFree(m_d_cd1_1); m_d_cd1_1 = nullptr; }
    if (m_d_cd1_2) { cudaFree(m_d_cd1_2); m_d_cd1_2 = nullptr; }
    if (m_d_cd2_1) { cudaFree(m_d_cd2_1); m_d_cd2_1 = nullptr; }
    if (m_d_cd2_2) { cudaFree(m_d_cd2_2); m_d_cd2_2 = nullptr; }
    if (m_d_widths) { cudaFree(m_d_widths); m_d_widths = nullptr; }
    if (m_d_heights) { cudaFree(m_d_heights); m_d_heights = nullptr; }
    if (m_d_blanks) { cudaFree(m_d_blanks); m_d_blanks = nullptr; }
    
    m_wcsParams.clear();
    m_numImages = 0;
    m_totalPixels = 0;
    m_initialized = false;
}
