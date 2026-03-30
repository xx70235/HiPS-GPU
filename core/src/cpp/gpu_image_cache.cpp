/**
 * GPU Image Cache Manager Implementation
 */

#include "gpu_image_cache.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

GPUImageCache::GPUImageCache()
    : m_d_allPixels(nullptr)
    , m_totalMemory(0)
    , m_available(false)
{
}

GPUImageCache::~GPUImageCache() {
    release();
}

bool GPUImageCache::loadToGPU(const std::vector<FitsData>& fitsFiles) {
    if (fitsFiles.empty()) {
        return false;
    }
    
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "CUDA not available for GPU cache" << std::endl;
        return false;
    }
    
    // Release any existing data
    release();
    
    // Calculate total memory needed
    size_t totalPixels = 0;
    for (const auto& fits : fitsFiles) {
        if (fits.isValid) {
            totalPixels += (size_t)fits.width * fits.height;
        }
    }
    
    m_totalMemory = totalPixels * sizeof(float);
    
    std::cout << "GPU Cache: Loading " << fitsFiles.size() << " images, "
              << (m_totalMemory / 1024 / 1024) << " MB" << std::endl;
    
    // Allocate unified GPU buffer
    err = cudaMalloc((void**)&m_d_allPixels, m_totalMemory);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        m_d_allPixels = nullptr;
        m_totalMemory = 0;
        return false;
    }
    
    // Copy all images to GPU
    size_t currentOffset = 0;
    for (const auto& fits : fitsFiles) {
        if (!fits.isValid) continue;
        
        GPUImageInfo info;
        info.width = fits.width;
        info.height = fits.height;
        info.blank = (float)fits.blank;
        info.offset = currentOffset;
        info.d_pixels = m_d_allPixels + currentOffset;
        
        size_t imageSize = (size_t)fits.width * fits.height * sizeof(float);
        
        err = cudaMemcpy(info.d_pixels, fits.pixels.data(), imageSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy image to GPU: " << cudaGetErrorString(err) << std::endl;
            release();
            return false;
        }
        
        m_images.push_back(info);
        currentOffset += (size_t)fits.width * fits.height;
    }
    
    m_available = true;
    std::cout << "GPU Cache: Successfully loaded " << m_images.size() << " images" << std::endl;
    
    return true;
}

bool GPUImageCache::batchInterpolate(
    const double* celestialRA,
    const double* celestialDec,
    const std::vector<std::vector<double>>& pixelCoordsX,
    const std::vector<std::vector<double>>& pixelCoordsY,
    const std::vector<std::vector<bool>>& inBounds,
    int numCoords,
    double* results,
    double blank
) {
    if (!m_available || m_images.empty() || numCoords == 0) {
        return false;
    }
    
    int numImages = (int)m_images.size();
    cudaError_t err;
    
    // Prepare host data arrays
    std::vector<int> h_widths(numImages);
    std::vector<int> h_heights(numImages);
    std::vector<size_t> h_offsets(numImages);
    std::vector<float> h_blanks(numImages);
    
    for (int i = 0; i < numImages; i++) {
        h_widths[i] = m_images[i].width;
        h_heights[i] = m_images[i].height;
        h_offsets[i] = m_images[i].offset;
        h_blanks[i] = m_images[i].blank;
    }
    
    // Flatten coordinate arrays [numImages][numCoords] -> [numImages * numCoords]
    size_t coordsSize = (size_t)numImages * numCoords;
    std::vector<double> h_coordsX(coordsSize);
    std::vector<double> h_coordsY(coordsSize);
    std::vector<int> h_inBounds(coordsSize);
    
    for (int img = 0; img < numImages; img++) {
        for (int c = 0; c < numCoords; c++) {
            size_t idx = (size_t)img * numCoords + c;
            h_coordsX[idx] = pixelCoordsX[img][c];
            h_coordsY[idx] = pixelCoordsY[img][c];
            h_inBounds[idx] = inBounds[img][c] ? 1 : 0;
        }
    }
    
    // Allocate device memory for metadata and coordinates
    int* d_widths = nullptr;
    int* d_heights = nullptr;
    size_t* d_offsets = nullptr;
    float* d_blanks = nullptr;
    double* d_coordsX = nullptr;
    double* d_coordsY = nullptr;
    int* d_inBounds = nullptr;
    double* d_results = nullptr;
    
    bool success = true;
    
    // Allocate
    err = cudaMalloc((void**)&d_widths, numImages * sizeof(int));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_heights, numImages * sizeof(int));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_offsets, numImages * sizeof(size_t));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_blanks, numImages * sizeof(float));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_coordsX, coordsSize * sizeof(double));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_coordsY, coordsSize * sizeof(double));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_inBounds, coordsSize * sizeof(int));
    if (err != cudaSuccess) success = false;
    
    err = cudaMalloc((void**)&d_results, numCoords * sizeof(double));
    if (err != cudaSuccess) success = false;
    
    if (!success) {
        std::cerr << "Failed to allocate GPU memory for batch interpolation" << std::endl;
        goto cleanup;
    }
    
    // Copy data to device
    cudaMemcpy(d_widths, h_widths.data(), numImages * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heights, h_heights.data(), numImages * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), numImages * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blanks, h_blanks.data(), numImages * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coordsX, h_coordsX.data(), coordsSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coordsY, h_coordsY.data(), coordsSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inBounds, h_inBounds.data(), coordsSize * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    launchMultiImageInterpolationKernel(
        m_d_allPixels,
        d_widths,
        d_heights,
        d_offsets,
        d_blanks,
        numImages,
        d_coordsX,
        d_coordsY,
        d_inBounds,
        d_results,
        numCoords,
        blank
    );
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        success = false;
        goto cleanup;
    }
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(results, d_results, numCoords * sizeof(double), cudaMemcpyDeviceToHost);
    
cleanup:
    if (d_widths) cudaFree(d_widths);
    if (d_heights) cudaFree(d_heights);
    if (d_offsets) cudaFree(d_offsets);
    if (d_blanks) cudaFree(d_blanks);
    if (d_coordsX) cudaFree(d_coordsX);
    if (d_coordsY) cudaFree(d_coordsY);
    if (d_inBounds) cudaFree(d_inBounds);
    if (d_results) cudaFree(d_results);
    
    return success;
}

void GPUImageCache::release() {
    if (m_d_allPixels) {
        cudaFree(m_d_allPixels);
        m_d_allPixels = nullptr;
    }
    m_images.clear();
    m_totalMemory = 0;
    m_available = false;
}
