/**
 * GPU Batch Processor Implementation
 * Processes ALL tiles in minimal GPU operations for maximum efficiency
 */

#include "gpu_batch_processor.h"
#include "healpix_util.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>

GPUBatchProcessor::GPUBatchProcessor()
    : m_d_allPixels(nullptr)
    , m_d_imageWidths(nullptr)
    , m_d_imageHeights(nullptr)
    , m_d_imageOffsets(nullptr)
    , m_d_imageBlanks(nullptr)
    , m_numImages(0)
    , m_totalImagePixels(0)
    , m_gpuAvailable(false)
    , m_initialized(false)
{
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    m_gpuAvailable = (err == cudaSuccess && deviceCount > 0);
}

GPUBatchProcessor::~GPUBatchProcessor() {
    release();
}

bool GPUBatchProcessor::initialize(
    const std::vector<FitsData>& fitsFiles,
    const std::vector<CoordinateTransform>& transforms
) {
    if (!m_gpuAvailable) {
        std::cerr << "GPU not available" << std::endl;
        return false;
    }
    
    if (fitsFiles.empty()) {
        return false;
    }
    
    release();
    
    m_numImages = (int)fitsFiles.size();
    m_transforms = transforms;
    
    // Calculate total memory needed and prepare metadata
    m_totalImagePixels = 0;
    m_imageWidths.resize(m_numImages);
    m_imageHeights.resize(m_numImages);
    std::vector<size_t> offsets(m_numImages);
    std::vector<float> blanks(m_numImages);
    
    for (int i = 0; i < m_numImages; i++) {
        m_imageWidths[i] = fitsFiles[i].width;
        m_imageHeights[i] = fitsFiles[i].height;
        offsets[i] = m_totalImagePixels;
        blanks[i] = (float)fitsFiles[i].blank;
        m_totalImagePixels += (size_t)fitsFiles[i].width * fitsFiles[i].height;
    }
    
    size_t imageMemory = m_totalImagePixels * sizeof(float);
    std::cout << "GPU Batch: Allocating " << (imageMemory / 1024 / 1024) << " MB for images" << std::endl;
    
    cudaError_t err;
    
    // Allocate GPU memory for images
    err = cudaMalloc(&m_d_allPixels, imageMemory);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for images: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Copy all images to GPU
    size_t currentOffset = 0;
    for (int i = 0; i < m_numImages; i++) {
        size_t imgSize = (size_t)fitsFiles[i].width * fitsFiles[i].height * sizeof(float);
        err = cudaMemcpy(m_d_allPixels + currentOffset, fitsFiles[i].pixels.data(), 
                         imgSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy image " << i << " to GPU" << std::endl;
            release();
            return false;
        }
        currentOffset += (size_t)fitsFiles[i].width * fitsFiles[i].height;
    }
    
    // Allocate and copy metadata
    err = cudaMalloc(&m_d_imageWidths, m_numImages * sizeof(int));
    if (err != cudaSuccess) { release(); return false; }
    cudaMemcpy(m_d_imageWidths, m_imageWidths.data(), m_numImages * sizeof(int), cudaMemcpyHostToDevice);
    
    err = cudaMalloc(&m_d_imageHeights, m_numImages * sizeof(int));
    if (err != cudaSuccess) { release(); return false; }
    cudaMemcpy(m_d_imageHeights, m_imageHeights.data(), m_numImages * sizeof(int), cudaMemcpyHostToDevice);
    
    err = cudaMalloc(&m_d_imageOffsets, m_numImages * sizeof(size_t));
    if (err != cudaSuccess) { release(); return false; }
    cudaMemcpy(m_d_imageOffsets, offsets.data(), m_numImages * sizeof(size_t), cudaMemcpyHostToDevice);
    
    err = cudaMalloc(&m_d_imageBlanks, m_numImages * sizeof(float));
    if (err != cudaSuccess) { release(); return false; }
    cudaMemcpy(m_d_imageBlanks, blanks.data(), m_numImages * sizeof(float), cudaMemcpyHostToDevice);
    
    m_initialized = true;
    std::cout << "GPU Batch: Initialized with " << m_numImages << " images" << std::endl;
    
    return true;
}

bool GPUBatchProcessor::processTilesBatch(
    const std::vector<TileInfo>& validTiles,
    int tileWidth,
    const std::vector<int>& xy2hpx,
    double blank,
    std::vector<std::vector<double>>& results
) {
    if (!m_initialized || validTiles.empty()) {
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int numTiles = (int)validTiles.size();
    int pixelsPerTile = tileWidth * tileWidth;
    
    // Limit batch size to avoid GPU memory overflow
    // Memory per tile: pixelsPerTile * numImages * (8+8+4) bytes = 5.2MB per tile for 40 images
    // With 6GB usable GPU memory, we can safely process ~50 tiles at a time
    const int MAX_TILES_PER_BATCH = 50;
    
    results.resize(numTiles);
    for (int t = 0; t < numTiles; t++) {
        results[t].resize(pixelsPerTile, blank);
    }
    
    // Calculate tile order for HEALPix conversion
    int tileOrder = 0;
    int temp = tileWidth;
    while (temp > 1) { temp >>= 1; tileOrder++; }
    
    int numBatches = (numTiles + MAX_TILES_PER_BATCH - 1) / MAX_TILES_PER_BATCH;
    std::cout << "GPU Batch: Processing " << numTiles << " tiles in " << numBatches << " batches" << std::endl;
    
    long totalCoordPrecompute = 0, totalTransfer = 0, totalKernel = 0, totalCopyBack = 0;
    
    for (int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * MAX_TILES_PER_BATCH;
        int batchEnd = std::min(batchStart + MAX_TILES_PER_BATCH, numTiles);
        int batchSize = batchEnd - batchStart;
        size_t batchCoords = (size_t)batchSize * pixelsPerTile;
        
        // Step 1: Precompute coordinates for this batch
        auto step1Start = std::chrono::high_resolution_clock::now();
        
        size_t coordArraySize = batchCoords * m_numImages;
        std::vector<double> h_coordsX(coordArraySize);
        std::vector<double> h_coordsY(coordArraySize);
        std::vector<int> h_inBounds(coordArraySize, 0);
        
        // Parallel coordinate computation within batch
        #pragma omp parallel for schedule(dynamic, 1)
        for (int bt = 0; bt < batchSize; bt++) {
            int t = batchStart + bt;
            int order = validTiles[t].order;
            long npix = validTiles[t].npix;
            int orderPix = order + tileOrder;
            long minHPXIndex = npix * pixelsPerTile;
            
            for (int p = 0; p < pixelsPerTile; p++) {
                size_t coordIdx = (size_t)bt * pixelsPerTile + p;
                
                int hpxOffset = xy2hpx[p];
                long hpxIndex = minHPXIndex + hpxOffset;
                
                CelestialCoord celestial = HealpixUtil::nestedToCelestial(orderPix, hpxIndex);
                
                for (int img = 0; img < m_numImages; img++) {
                    size_t idx = coordIdx * m_numImages + img;
                    
                    Coord pixel = m_transforms[img].celestialToPixel(celestial);
                    h_coordsX[idx] = pixel.x;
                    h_coordsY[idx] = pixel.y;
                    h_inBounds[idx] = m_transforms[img].isPixelInBounds(pixel) ? 1 : 0;
                }
            }
        }
        
        auto step1End = std::chrono::high_resolution_clock::now();
        totalCoordPrecompute += std::chrono::duration_cast<std::chrono::milliseconds>(step1End - step1Start).count();
        
        // Step 2: Transfer to GPU
        auto step2Start = std::chrono::high_resolution_clock::now();
        
        double* d_coordsX = nullptr;
        double* d_coordsY = nullptr;
        int* d_inBounds = nullptr;
        double* d_results = nullptr;
        
        cudaError_t err;
        size_t coordMemory = coordArraySize * sizeof(double);
        size_t boundsMemory = coordArraySize * sizeof(int);
        size_t resultMemory = batchCoords * sizeof(double);
        
        err = cudaMalloc(&d_coordsX, coordMemory);
        if (err != cudaSuccess) { 
            std::cerr << "cudaMalloc failed for coordsX: " << cudaGetErrorString(err) << std::endl; 
            return false; 
        }
        
        err = cudaMalloc(&d_coordsY, coordMemory);
        if (err != cudaSuccess) { cudaFree(d_coordsX); return false; }
        
        err = cudaMalloc(&d_inBounds, boundsMemory);
        if (err != cudaSuccess) { cudaFree(d_coordsX); cudaFree(d_coordsY); return false; }
        
        err = cudaMalloc(&d_results, resultMemory);
        if (err != cudaSuccess) { cudaFree(d_coordsX); cudaFree(d_coordsY); cudaFree(d_inBounds); return false; }
        
        cudaMemcpy(d_coordsX, h_coordsX.data(), coordMemory, cudaMemcpyHostToDevice);
        cudaMemcpy(d_coordsY, h_coordsY.data(), coordMemory, cudaMemcpyHostToDevice);
        cudaMemcpy(d_inBounds, h_inBounds.data(), boundsMemory, cudaMemcpyHostToDevice);
        
        auto step2End = std::chrono::high_resolution_clock::now();
        totalTransfer += std::chrono::duration_cast<std::chrono::milliseconds>(step2End - step2Start).count();
        
        // Step 3: Execute kernel
        auto step3Start = std::chrono::high_resolution_clock::now();
        
        launchBatchTileInterpolationKernel(
            m_d_allPixels,
            m_d_imageWidths,
            m_d_imageHeights,
            m_d_imageOffsets,
            m_d_imageBlanks,
            m_numImages,
            d_coordsX,
            d_coordsY,
            d_inBounds,
            d_results,
            batchCoords,
            blank
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_coordsX); cudaFree(d_coordsY); cudaFree(d_inBounds); cudaFree(d_results);
            return false;
        }
        
        cudaDeviceSynchronize();
        
        auto step3End = std::chrono::high_resolution_clock::now();
        totalKernel += std::chrono::duration_cast<std::chrono::milliseconds>(step3End - step3Start).count();
        
        // Step 4: Copy results back
        auto step4Start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> h_results(batchCoords);
        cudaMemcpy(h_results.data(), d_results, resultMemory, cudaMemcpyDeviceToHost);
        
        // Copy results to output
        for (int bt = 0; bt < batchSize; bt++) {
            int t = batchStart + bt;
            size_t startIdx = (size_t)bt * pixelsPerTile;
            std::memcpy(results[t].data(), &h_results[startIdx], pixelsPerTile * sizeof(double));
        }
        
        auto step4End = std::chrono::high_resolution_clock::now();
        totalCopyBack += std::chrono::duration_cast<std::chrono::milliseconds>(step4End - step4Start).count();
        
        // Cleanup batch GPU memory
        cudaFree(d_coordsX);
        cudaFree(d_coordsY);
        cudaFree(d_inBounds);
        cudaFree(d_results);
        
        if (batch == 0 || (batch + 1) % 2 == 0) {
            std::cout << "  Batch " << (batch + 1) << "/" << numBatches << " done (" 
                      << batchSize << " tiles)" << std::endl;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "GPU Batch Summary:" << std::endl;
    std::cout << "  Coord precompute: " << totalCoordPrecompute << " ms" << std::endl;
    std::cout << "  GPU transfer:     " << totalTransfer << " ms" << std::endl;
    std::cout << "  GPU kernel:       " << totalKernel << " ms" << std::endl;
    std::cout << "  Result copy:      " << totalCopyBack << " ms" << std::endl;
    std::cout << "  Total time:       " << totalMs << " ms" << std::endl;
    
    return true;
}

void GPUBatchProcessor::release() {
    if (m_d_allPixels) { cudaFree(m_d_allPixels); m_d_allPixels = nullptr; }
    if (m_d_imageWidths) { cudaFree(m_d_imageWidths); m_d_imageWidths = nullptr; }
    if (m_d_imageHeights) { cudaFree(m_d_imageHeights); m_d_imageHeights = nullptr; }
    if (m_d_imageOffsets) { cudaFree(m_d_imageOffsets); m_d_imageOffsets = nullptr; }
    if (m_d_imageBlanks) { cudaFree(m_d_imageBlanks); m_d_imageBlanks = nullptr; }
    
    m_transforms.clear();
    m_imageWidths.clear();
    m_imageHeights.clear();
    m_numImages = 0;
    m_totalImagePixels = 0;
    m_initialized = false;
}
