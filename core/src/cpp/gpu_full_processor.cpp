/**
 * GPU Full Processor Implementation
 * Uses official HEALPix library (healpix_cxx) for coordinate computation
 * Transfers precomputed celestial coordinates to GPU
 */

#include "gpu_full_processor.h"
#include "tile_image_index.h"
#include "tile_coord_cache.h"
#include "healpix_cuda.cuh"  // GPU HEALPix coordinate computation
#include <omp.h>
#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/pointing.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <memory>

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

GPUFullProcessor::GPUFullProcessor()
    : m_d_allPixels(nullptr)
    , m_d_imageOffsets(nullptr)
    , m_d_crval1(nullptr), m_d_crval2(nullptr)
    , m_d_crpix1(nullptr), m_d_crpix2(nullptr)
    , m_d_cd1_1(nullptr), m_d_cd1_2(nullptr)
    , m_d_cd2_1(nullptr), m_d_cd2_2(nullptr)
    , m_d_widths(nullptr), m_d_heights(nullptr)
    , m_d_blanks(nullptr)
    , m_d_raCoords(nullptr)
    , m_d_decCoords(nullptr)
    , m_d_results(nullptr)
    , m_d_imageMasks(nullptr)
    , m_allocatedOutputPixels(0)
    , m_allocatedMaskInts(0)
    , m_resultElementBytes(0)
    , m_useFloatResults(false)
    , m_coordinateCacheRoot()
    , m_numImages(0)
    , m_totalPixels(0)
    , m_initialized(false)
{
}

GPUFullProcessor::~GPUFullProcessor() {
    release();
}

void GPUFullProcessor::setCoordinateCacheRoot(const std::string& outputDir) {
    m_coordinateCacheRoot = outputDir;
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
    const int pixelsPerTile = tileWidth * tileWidth;
    const std::size_t totalPixels = tiles.size() * static_cast<std::size_t>(pixelsPerTile);

    raCoords.resize(totalPixels);
    decCoords.resize(totalPixels);

    std::vector<std::size_t> missTileIndices;
    missTileIndices.reserve(tiles.size());

    std::size_t cacheHits = 0;
    const bool useCoordCache = !m_coordinateCacheRoot.empty();

    for (std::size_t tileIdx = 0; tileIdx < tiles.size(); ++tileIdx) {
        const std::size_t globalIdx = tileIdx * static_cast<std::size_t>(pixelsPerTile);
        bool cacheHit = false;
        if (useCoordCache) {
            cacheHit = read_tile_coord_cache(
                m_coordinateCacheRoot,
                tiles[tileIdx].order,
                tiles[tileIdx].npix,
                tileWidth,
                raCoords.data() + globalIdx,
                decCoords.data() + globalIdx,
                static_cast<std::size_t>(pixelsPerTile)
            );
        }
        if (cacheHit) {
            cacheHits++;
        } else {
            missTileIndices.push_back(tileIdx);
        }
    }

    const int tileOrder = [&]() {
        int order = 0;
        int temp = tileWidth;
        while (temp > 1) {
            temp >>= 1;
            order++;
        }
        return order;
    }();

    if (!missTileIndices.empty()) {
        std::cout << "Using CPU (OpenMP) with official HEALPix library..." << std::endl;

        #pragma omp parallel
        {
            int cachedPixelOrder = -1;
            std::unique_ptr<T_Healpix_Base<int64>> hpx;

            #pragma omp for schedule(dynamic, 10)
            for (std::size_t missIdx = 0; missIdx < missTileIndices.size(); ++missIdx) {
                const std::size_t tileIdx = missTileIndices[missIdx];
                const int order = tiles[tileIdx].order;
                const long npix = tiles[tileIdx].npix;
                const int pixelOrder = order + tileOrder;

                if (!hpx || pixelOrder != cachedPixelOrder) {
                    const long nside = 1L << pixelOrder;
                    hpx = std::make_unique<T_Healpix_Base<int64>>(nside, NEST, SET_NSIDE);
                    cachedPixelOrder = pixelOrder;
                }

                const long baseIdx = npix << (2 * tileOrder);
                const std::size_t globalBase = tileIdx * static_cast<std::size_t>(pixelsPerTile);

                for (int pixelInTile = 0; pixelInTile < pixelsPerTile; ++pixelInTile) {
                    const int hpxOffset = xy2hpx[pixelInTile];
                    const long healpixIdx = baseIdx + hpxOffset;
                    const pointing ptg = hpx->pix2ang(healpixIdx);

                    raCoords[globalBase + static_cast<std::size_t>(pixelInTile)] = ptg.phi * 180.0 / M_PI;
                    decCoords[globalBase + static_cast<std::size_t>(pixelInTile)] = 90.0 - ptg.theta * 180.0 / M_PI;
                }
            }
        }

        if (useCoordCache) {
            int cacheWriteFailures = 0;
            for (std::size_t missIdx = 0; missIdx < missTileIndices.size(); ++missIdx) {
                const std::size_t tileIdx = missTileIndices[missIdx];
                const std::size_t globalBase = tileIdx * static_cast<std::size_t>(pixelsPerTile);
                if (!write_tile_coord_cache(
                        m_coordinateCacheRoot,
                        tiles[tileIdx].order,
                        tiles[tileIdx].npix,
                        tileWidth,
                        raCoords.data() + globalBase,
                        decCoords.data() + globalBase,
                        static_cast<std::size_t>(pixelsPerTile))) {
                    cacheWriteFailures++;
                }
            }
            if (cacheWriteFailures > 0) {
                std::cout << "GPU Full: Coordinate cache write failures: " << cacheWriteFailures << std::endl;
            }
        }
    }

    if (useCoordCache) {
        std::cout << "GPU Full: Coord cache hits=" << cacheHits
                  << ", misses=" << missTileIndices.size() << std::endl;
    }

    if (totalPixels > 0) {
        std::cout << "GPU Full: Computed celestial coordinates for " << totalPixels << " pixels (GPU accelerated)" << std::endl;
        std::cout << "GPU Full: First tile (order=" << tiles[0].order << ", npix=" << tiles[0].npix << ")" << std::endl;
        std::cout << "GPU Full: Sample coords - pixel0: (ra=" << raCoords[0] << ", dec=" << decCoords[0] << ")" << std::endl;
        if (pixelsPerTile > 1) {
            std::cout << "GPU Full: Sample coords - pixel1: (ra=" << raCoords[1] << ", dec=" << decCoords[1] << ")" << std::endl;
        }
    }
}

bool GPUFullProcessor::ensureBatchWorkspace(const BatchWorkspacePlan& plan) {
    m_useFloatResults = plan.use_float_results;

    const bool needReallocate =
        (m_d_raCoords == nullptr) ||
        (plan.total_output_pixels > m_allocatedOutputPixels) ||
        (plan.mask_ints > m_allocatedMaskInts) ||
        (plan.result_element_bytes != m_resultElementBytes);

    if (!needReallocate) {
        return true;
    }

    releaseBatchWorkspace();

    if (plan.total_output_pixels == 0) {
        m_resultElementBytes = plan.result_element_bytes;
        return true;
    }

    cudaError_t err = cudaMalloc(&m_d_raCoords, plan.total_output_pixels * plan.coord_element_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate reusable RA workspace" << std::endl;
        releaseBatchWorkspace();
        return false;
    }

    err = cudaMalloc(&m_d_decCoords, plan.total_output_pixels * plan.coord_element_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate reusable Dec workspace" << std::endl;
        releaseBatchWorkspace();
        return false;
    }

    if (plan.mask_ints > 0) {
        err = cudaMalloc(&m_d_imageMasks, plan.mask_ints * sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate reusable image-mask workspace" << std::endl;
            releaseBatchWorkspace();
            return false;
        }
    }

    err = cudaMalloc(&m_d_results, plan.total_output_pixels * plan.result_element_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate reusable result workspace" << std::endl;
        releaseBatchWorkspace();
        return false;
    }

    m_allocatedOutputPixels = plan.total_output_pixels;
    m_allocatedMaskInts = plan.mask_ints;
    m_resultElementBytes = plan.result_element_bytes;
    return true;
}

void GPUFullProcessor::releaseBatchWorkspace() {
    if (m_d_raCoords) { cudaFree(m_d_raCoords); m_d_raCoords = nullptr; }
    if (m_d_decCoords) { cudaFree(m_d_decCoords); m_d_decCoords = nullptr; }
    if (m_d_results) { cudaFree(m_d_results); m_d_results = nullptr; }
    if (m_d_imageMasks) { cudaFree(m_d_imageMasks); m_d_imageMasks = nullptr; }

    m_allocatedOutputPixels = 0;
    m_allocatedMaskInts = 0;
    m_resultElementBytes = 0;
    m_useFloatResults = false;
}

bool GPUFullProcessor::processAllTiles(
    const std::vector<GPUTileInfo>& tiles,
    int tileWidth,
    const std::vector<int>& xy2hpx,
    double blank,
    std::vector<std::vector<double>>& results,
    int outputBitpix,
    double validMin,
    double validMax
) {
    (void)validMin;
    (void)validMax;

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
    // - coordSize: 2 * pixelsPerTile * sizeof(double)
    // - outputSize: pixelsPerTile * sizeof(double)
    // - imageMask: numImages * sizeof(int)
    size_t memoryPerTile = 2 * pixelsPerTile * sizeof(double) + pixelsPerTile * sizeof(double) + m_numImages * sizeof(int);

    // Calculate max tiles per batch (leave headroom for kernel overhead)
    int maxTilesPerBatch = std::max(1, (int)(availableMem / memoryPerTile / 2));
    maxTilesPerBatch = std::min(maxTilesPerBatch, 5000); // Cap at 5000 tiles per batch

    int numBatches = (numTiles + maxTilesPerBatch - 1) / maxTilesPerBatch;

    BatchWorkspacePlan workspacePlan = make_batch_workspace_plan(maxTilesPerBatch, tileWidth, m_numImages, outputBitpix);
    if (workspacePlan.use_float_results && (numBatches > 1 || m_numImages > 128 || workspacePlan.total_output_pixels > 30000000u)) {
        workspacePlan.use_float_results = false;
        workspacePlan.result_element_bytes = sizeof(double);
        std::cout << "GPU Full: Using double batch results for this large run to avoid the unstable float-output path" << std::endl;
    }
    if (!ensureBatchWorkspace(workspacePlan)) {
        return false;
    }

    std::cout << "GPU Full: Processing " << numTiles << " tiles in " << numBatches
              << " batches (max " << maxTilesPerBatch << " tiles/batch)" << std::endl;
    std::cout << "GPU Full: Available memory: " << (availableMem / 1024 / 1024) << " MB" << std::endl;

    results.resize(numTiles);

    long totalCoordMs = 0;
    long totalKernelMs = 0;
    long totalCopyMs = 0;

    std::vector<double> h_raCoords;
    std::vector<double> h_decCoords;
    std::vector<float> h_resultsFloat;
    std::vector<double> h_resultsDouble;

    h_raCoords.reserve(workspacePlan.total_output_pixels);
    h_decCoords.reserve(workspacePlan.total_output_pixels);
    if (workspacePlan.use_float_results) {
        h_resultsFloat.reserve(workspacePlan.total_output_pixels);
    } else {
        h_resultsDouble.reserve(workspacePlan.total_output_pixels);
    }

    for (int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * maxTilesPerBatch;
        int batchEnd = std::min(batchStart + maxTilesPerBatch, numTiles);
        int batchSize = batchEnd - batchStart;
        size_t batchOutputPixels = (size_t)batchSize * pixelsPerTile;

        // Extract batch tiles
        std::vector<GPUTileInfo> batchTiles(tiles.begin() + batchStart, tiles.begin() + batchEnd);

        // Step 1: Precompute celestial coordinates for this batch
        auto coordStart = std::chrono::high_resolution_clock::now();

        computeCelestialCoords(batchTiles, tileWidth, xy2hpx, h_raCoords, h_decCoords);

        auto coordEnd = std::chrono::high_resolution_clock::now();
        totalCoordMs += std::chrono::duration_cast<std::chrono::milliseconds>(coordEnd - coordStart).count();

        SparseTileImageIndex sparseIndex = build_sparse_tile_image_index(batchTiles, 0);
        int* d_tileImageOffsets = nullptr;
        int* d_tileImageIndices = nullptr;
        auto releaseSparseBuffers = [&]() {
            if (d_tileImageOffsets) { cudaFree(d_tileImageOffsets); d_tileImageOffsets = nullptr; }
            if (d_tileImageIndices) { cudaFree(d_tileImageIndices); d_tileImageIndices = nullptr; }
        };

        size_t coordSize = batchOutputPixels * sizeof(double);
        size_t outputSize = batchOutputPixels * m_resultElementBytes;
        size_t tileOffsetBytes = sparseIndex.tile_offsets.size() * sizeof(int);
        size_t tileIndexBytes = sparseIndex.image_indices.size() * sizeof(int);

        // Copy data to reusable GPU buffers
        cudaError_t err = cudaMemcpy(m_d_raCoords, h_raCoords.data(), coordSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy RA coordinates for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        err = cudaMemcpy(m_d_decCoords, h_decCoords.data(), coordSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy Dec coordinates for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        err = cudaMalloc(&d_tileImageOffsets, tileOffsetBytes);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate tile offsets for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            releaseSparseBuffers();
            return false;
        }
        err = cudaMemcpy(d_tileImageOffsets, sparseIndex.tile_offsets.data(), tileOffsetBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy tile offsets for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            releaseSparseBuffers();
            return false;
        }

        if (tileIndexBytes > 0) {
            err = cudaMalloc(&d_tileImageIndices, tileIndexBytes);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate tile-image indices for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
                releaseSparseBuffers();
                return false;
            }
            err = cudaMemcpy(d_tileImageIndices, sparseIndex.image_indices.data(), tileIndexBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy tile-image indices for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
                releaseSparseBuffers();
                return false;
            }
        }

        // Execute kernel
        auto kernelStart = std::chrono::high_resolution_clock::now();

        if (m_useFloatResults) {
            launchFullGPUTileKernelWithCoordsSparseFloat(
                m_d_allPixels, m_d_imageOffsets,
                m_d_crval1, m_d_crval2,
                m_d_crpix1, m_d_crpix2,
                m_d_cd1_1, m_d_cd1_2,
                m_d_cd2_1, m_d_cd2_2,
                m_d_widths, m_d_heights, m_d_blanks,
                m_d_raCoords, m_d_decCoords,
                d_tileImageOffsets, d_tileImageIndices,
                tileWidth, batchSize,
                static_cast<float*>(m_d_results), blank
            );
        } else {
            launchFullGPUTileKernelWithCoordsSparseDouble(
                m_d_allPixels, m_d_imageOffsets,
                m_d_crval1, m_d_crval2,
                m_d_crpix1, m_d_crpix2,
                m_d_cd1_1, m_d_cd1_2,
                m_d_cd2_1, m_d_cd2_2,
                m_d_widths, m_d_heights, m_d_blanks,
                m_d_raCoords, m_d_decCoords,
                d_tileImageOffsets, d_tileImageIndices,
                tileWidth, batchSize,
                static_cast<double*>(m_d_results), blank
            );
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel error in batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            releaseSparseBuffers();
            return false;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Kernel synchronize error in batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
            releaseSparseBuffers();
            return false;
        }

        auto kernelEnd = std::chrono::high_resolution_clock::now();
        totalKernelMs += std::chrono::duration_cast<std::chrono::milliseconds>(kernelEnd - kernelStart).count();

        // Copy results back
        auto copyStart = std::chrono::high_resolution_clock::now();

        if (m_useFloatResults) {
            h_resultsFloat.resize(batchOutputPixels);
            err = cudaMemcpy(h_resultsFloat.data(), m_d_results, outputSize, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy float results for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
                releaseSparseBuffers();
                return false;
            }

            for (int t = 0; t < batchSize; t++) {
                int globalIdx = batchStart + t;
                results[globalIdx].resize(pixelsPerTile);
                const float* src = h_resultsFloat.data() + (size_t)t * pixelsPerTile;
                for (int p = 0; p < pixelsPerTile; ++p) {
                    results[globalIdx][p] = src[p];
                }
            }
        } else {
            h_resultsDouble.resize(batchOutputPixels);
            err = cudaMemcpy(h_resultsDouble.data(), m_d_results, outputSize, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy double results for batch " << batch << ": " << cudaGetErrorString(err) << std::endl;
                releaseSparseBuffers();
                return false;
            }

            for (int t = 0; t < batchSize; t++) {
                int globalIdx = batchStart + t;
                results[globalIdx].resize(pixelsPerTile);
                std::memcpy(
                    results[globalIdx].data(),
                    h_resultsDouble.data() + (size_t)t * pixelsPerTile,
                    (size_t)pixelsPerTile * sizeof(double)
                );
            }
        }

        auto copyEnd = std::chrono::high_resolution_clock::now();
        totalCopyMs += std::chrono::duration_cast<std::chrono::milliseconds>(copyEnd - copyStart).count();

        releaseSparseBuffers();

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


bool GPUFullProcessor::processAllTilesWeightedStats(
    const std::vector<GPUTileInfo>& tiles,
    int tileWidth,
    const std::vector<int>& xy2hpx,
    std::vector<std::vector<double>>& weightedSums,
    std::vector<std::vector<double>>& totalWeights
) {
    if (!m_initialized || tiles.empty()) return false;

    auto startTime = std::chrono::high_resolution_clock::now();

    int numTiles = (int)tiles.size();
    int pixelsPerTile = tileWidth * tileWidth;

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t availableMem = (size_t)(freeMem * 0.8);

    size_t memoryPerTile =
        2 * pixelsPerTile * sizeof(double) +  // coords
        2 * pixelsPerTile * sizeof(double) +  // weighted sum + total weight
        m_numImages * sizeof(int);

    int maxTilesPerBatch = std::max(1, (int)(availableMem / memoryPerTile / 2));
    maxTilesPerBatch = std::min(maxTilesPerBatch, 5000);

    int numBatches = (numTiles + maxTilesPerBatch - 1) / maxTilesPerBatch;

    BatchWorkspacePlan workspacePlan = make_batch_workspace_plan(maxTilesPerBatch, tileWidth, m_numImages, -64);
    workspacePlan.use_float_results = false;
    workspacePlan.result_element_bytes = sizeof(double);
    if (!ensureBatchWorkspace(workspacePlan)) {
        return false;
    }

    std::cout << "GPU Full: Processing weighted stats for " << numTiles << " tiles in "
              << numBatches << " batches (max " << maxTilesPerBatch << " tiles/batch)" << std::endl;

    weightedSums.resize(numTiles);
    totalWeights.resize(numTiles);

    std::vector<double> h_raCoords;
    std::vector<double> h_decCoords;
    std::vector<double> h_weightedSums;
    std::vector<double> h_totalWeights;

    h_raCoords.reserve(workspacePlan.total_output_pixels);
    h_decCoords.reserve(workspacePlan.total_output_pixels);
    h_weightedSums.reserve(workspacePlan.total_output_pixels);
    h_totalWeights.reserve(workspacePlan.total_output_pixels);

    long totalCoordMs = 0;
    long totalKernelMs = 0;
    long totalCopyMs = 0;

    for (int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * maxTilesPerBatch;
        int batchEnd = std::min(batchStart + maxTilesPerBatch, numTiles);
        int batchSize = batchEnd - batchStart;
        size_t batchOutputPixels = (size_t)batchSize * pixelsPerTile;

        std::vector<GPUTileInfo> batchTiles(tiles.begin() + batchStart, tiles.begin() + batchEnd);

        auto coordStart = std::chrono::high_resolution_clock::now();
        computeCelestialCoords(batchTiles, tileWidth, xy2hpx, h_raCoords, h_decCoords);
        auto coordEnd = std::chrono::high_resolution_clock::now();
        totalCoordMs += std::chrono::duration_cast<std::chrono::milliseconds>(coordEnd - coordStart).count();

        SparseTileImageIndex sparseIndex = build_sparse_tile_image_index(batchTiles, 0);
        int* d_tileImageOffsets = nullptr;
        int* d_tileImageIndices = nullptr;
        double* d_weightedSums = nullptr;
        double* d_totalWeights = nullptr;
        auto releaseBuffers = [&]() {
            if (d_tileImageOffsets) { cudaFree(d_tileImageOffsets); d_tileImageOffsets = nullptr; }
            if (d_tileImageIndices) { cudaFree(d_tileImageIndices); d_tileImageIndices = nullptr; }
            if (d_weightedSums) { cudaFree(d_weightedSums); d_weightedSums = nullptr; }
            if (d_totalWeights) { cudaFree(d_totalWeights); d_totalWeights = nullptr; }
        };

        size_t coordSize = batchOutputPixels * sizeof(double);
        size_t tileOffsetBytes = sparseIndex.tile_offsets.size() * sizeof(int);
        size_t tileIndexBytes = sparseIndex.image_indices.size() * sizeof(int);
        size_t accumBytes = batchOutputPixels * sizeof(double);

        cudaError_t err = cudaMemcpy(m_d_raCoords, h_raCoords.data(), coordSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy RA coordinates for weighted-stats batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        err = cudaMemcpy(m_d_decCoords, h_decCoords.data(), coordSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy Dec coordinates for weighted-stats batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        err = cudaMalloc(&d_tileImageOffsets, tileOffsetBytes);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate tile offsets for weighted-stats batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }
        err = cudaMemcpy(d_tileImageOffsets, sparseIndex.tile_offsets.data(), tileOffsetBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy tile offsets for weighted-stats batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }

        if (tileIndexBytes > 0) {
            err = cudaMalloc(&d_tileImageIndices, tileIndexBytes);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate tile-image indices for weighted-stats batch " << batch
                          << ": " << cudaGetErrorString(err) << std::endl;
                releaseBuffers();
                return false;
            }
            err = cudaMemcpy(d_tileImageIndices, sparseIndex.image_indices.data(), tileIndexBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy tile-image indices for weighted-stats batch " << batch
                          << ": " << cudaGetErrorString(err) << std::endl;
                releaseBuffers();
                return false;
            }
        }

        err = cudaMalloc(&d_weightedSums, accumBytes);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate weighted sums for batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }
        err = cudaMalloc(&d_totalWeights, accumBytes);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate total weights for batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }

        auto kernelStart = std::chrono::high_resolution_clock::now();
        launchFullGPUTileKernelWithCoordsSparseAccum(
            m_d_allPixels, m_d_imageOffsets,
            m_d_crval1, m_d_crval2,
            m_d_crpix1, m_d_crpix2,
            m_d_cd1_1, m_d_cd1_2,
            m_d_cd2_1, m_d_cd2_2,
            m_d_widths, m_d_heights, m_d_blanks,
            m_d_raCoords, m_d_decCoords,
            d_tileImageOffsets, d_tileImageIndices,
            tileWidth, batchSize,
            d_weightedSums, d_totalWeights
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Weighted-stats kernel error in batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Weighted-stats synchronize error in batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }
        auto kernelEnd = std::chrono::high_resolution_clock::now();
        totalKernelMs += std::chrono::duration_cast<std::chrono::milliseconds>(kernelEnd - kernelStart).count();

        auto copyStart = std::chrono::high_resolution_clock::now();
        h_weightedSums.resize(batchOutputPixels);
        h_totalWeights.resize(batchOutputPixels);

        err = cudaMemcpy(h_weightedSums.data(), d_weightedSums, accumBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy weighted sums for batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }
        err = cudaMemcpy(h_totalWeights.data(), d_totalWeights, accumBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy total weights for batch " << batch
                      << ": " << cudaGetErrorString(err) << std::endl;
            releaseBuffers();
            return false;
        }

        for (int t = 0; t < batchSize; t++) {
            int globalIdx = batchStart + t;
            weightedSums[globalIdx].resize(pixelsPerTile);
            totalWeights[globalIdx].resize(pixelsPerTile);
            std::memcpy(
                weightedSums[globalIdx].data(),
                h_weightedSums.data() + (size_t)t * pixelsPerTile,
                (size_t)pixelsPerTile * sizeof(double)
            );
            std::memcpy(
                totalWeights[globalIdx].data(),
                h_totalWeights.data() + (size_t)t * pixelsPerTile,
                (size_t)pixelsPerTile * sizeof(double)
            );
        }

        auto copyEnd = std::chrono::high_resolution_clock::now();
        totalCopyMs += std::chrono::duration_cast<std::chrono::milliseconds>(copyEnd - copyStart).count();

        releaseBuffers();

        if (batch == 0 || (batch + 1) % 10 == 0 || batch == numBatches - 1) {
            std::cout << "  Weighted-stats batch " << (batch + 1) << "/" << numBatches
                      << " done (" << batchSize << " tiles)" << std::endl;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "GPU Full Weighted-Stats Summary:" << std::endl;
    std::cout << "  Coord comp:  " << totalCoordMs << " ms" << std::endl;
    std::cout << "  GPU kernel:  " << totalKernelMs << " ms" << std::endl;
    std::cout << "  Copy back:   " << totalCopyMs << " ms" << std::endl;
    std::cout << "  Total time:  " << totalMs << " ms" << std::endl;

    return true;
}

void GPUFullProcessor::release() {
    releaseBatchWorkspace();

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
