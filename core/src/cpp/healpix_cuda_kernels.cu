/**
 * GPU-accelerated HEALPix coordinate computation - CUDA kernels
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Constants
#define HEALPIX_PI 3.14159265358979323846
#define HEALPIX_TWOTHIRD 0.6666666666666666

// NESTED scheme lookup tables
__constant__ int d_jrll[12] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
__constant__ int d_jpll[12] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

/**
 * Convert NESTED HEALPix index to (theta, phi) on GPU
 */
__device__ void pix2ang_nest_device(long nside, long ipix, double* theta, double* phi) {
    long npface = nside * nside;
    long nl4 = 4 * nside;
    
    int face = ipix / npface;
    long ipf = ipix % npface;
    
    // Bit interleaving to get (ix, iy)
    long ix = 0, iy = 0;
    long t = ipf;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ix |= ((t & 1) << i);
        t >>= 1;
        iy |= ((t & 1) << i);
        t >>= 1;
    }
    
    int jrt = d_jrll[face];
    int jpt = d_jpll[face];
    
    long jr = (long)(jrt * nside) - ix - iy - 1;
    
    double z;
    long nr;
    bool have_sth = false;
    double sth = 0.0;
    
    if (jr < nside) {
        nr = jr;
        z = 1.0 - (double)(nr * nr) / (3.0 * npface);
        have_sth = true;
        sth = sqrt((1.0 - z) * (1.0 + z));
    } else if (jr > 3 * nside) {
        nr = nl4 - jr;
        z = -1.0 + (double)(nr * nr) / (3.0 * npface);
        have_sth = true;
        sth = sqrt((1.0 - z) * (1.0 + z));
    } else {
        nr = nside;
        z = (double)(2 * nside - jr) * HEALPIX_TWOTHIRD / nside;
    }
    
    *theta = have_sth ? atan2(sth, z) : acos(z);
    
    long jp = (jpt * nr + ix - iy + 1 + nl4) % nl4;
    if (jp < 1) jp += nl4;
    
    double shift = 0.0;
    if (jr < nside || jr > 3 * nside) {
        shift = 0.5;
    } else {
        shift = ((jr - nside) & 1) ? 0.0 : 0.5;
    }
    
    *phi = (jp - shift) * HEALPIX_PI / (2.0 * nr);
}

/**
 * Kernel to compute HEALPix coordinates for all pixels
 */
__global__ void computeHEALPixCoords_kernel(
    const int* __restrict__ tileOrders,
    const long* __restrict__ tileNpixs,
    const int* __restrict__ xy2hpx,
    int tileWidth,
    int numTiles,
    double* __restrict__ raCoords,
    double* __restrict__ decCoords
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long pixelsPerTile = tileWidth * tileWidth;
    long totalPixels = (long)numTiles * pixelsPerTile;
    
    if (idx >= totalPixels) return;
    
    int tileIdx = idx / pixelsPerTile;
    int pixelInTile = idx % pixelsPerTile;
    
    int order = tileOrders[tileIdx];
    long npix = tileNpixs[tileIdx];
    
    // Calculate tileOrder = log2(tileWidth)
    int tileOrder = 0;
    int temp = tileWidth;
    while (temp > 1) { temp >>= 1; tileOrder++; }
    
    int pixelOrder = order + tileOrder;
    long nside = 1L << pixelOrder;
    long baseIdx = npix << (2 * tileOrder);
    
    int hpxOffset = xy2hpx[pixelInTile];
    long healpixIdx = baseIdx + hpxOffset;
    
    double theta, phi;
    pix2ang_nest_device(nside, healpixIdx, &theta, &phi);
    
    raCoords[idx] = phi * 180.0 / HEALPIX_PI;
    decCoords[idx] = 90.0 - theta * 180.0 / HEALPIX_PI;
}

// C++ wrapper function (extern "C" for linking)
extern "C" cudaError_t computeHEALPixCoordsGPU(
    const int* h_tileOrders,
    const long* h_tileNpixs,
    const int* h_xy2hpx,
    int numTiles,
    int tileWidth,
    double* h_raCoords,
    double* h_decCoords
) {
    int pixelsPerTile = tileWidth * tileWidth;
    size_t totalPixels = (size_t)numTiles * pixelsPerTile;
    
    int* d_tileOrders;
    long* d_tileNpixs;
    int* d_xy2hpx;
    double* d_raCoords;
    double* d_decCoords;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_tileOrders, numTiles * sizeof(int));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_tileNpixs, numTiles * sizeof(long));
    if (err != cudaSuccess) { cudaFree(d_tileOrders); return err; }
    
    err = cudaMalloc(&d_xy2hpx, pixelsPerTile * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_tileOrders); cudaFree(d_tileNpixs); return err; }
    
    err = cudaMalloc(&d_raCoords, totalPixels * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_tileOrders); cudaFree(d_tileNpixs); cudaFree(d_xy2hpx); return err; }
    
    err = cudaMalloc(&d_decCoords, totalPixels * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_tileOrders); cudaFree(d_tileNpixs); cudaFree(d_xy2hpx); cudaFree(d_raCoords); return err; }
    
    cudaMemcpy(d_tileOrders, h_tileOrders, numTiles * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tileNpixs, h_tileNpixs, numTiles * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xy2hpx, h_xy2hpx, pixelsPerTile * sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    
    computeHEALPixCoords_kernel<<<numBlocks, blockSize>>>(
        d_tileOrders, d_tileNpixs, d_xy2hpx,
        tileWidth, numTiles,
        d_raCoords, d_decCoords
    );
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_tileOrders);
        cudaFree(d_tileNpixs);
        cudaFree(d_xy2hpx);
        cudaFree(d_raCoords);
        cudaFree(d_decCoords);
        return err;
    }
    
    cudaMemcpy(h_raCoords, d_raCoords, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_decCoords, d_decCoords, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_tileOrders);
    cudaFree(d_tileNpixs);
    cudaFree(d_xy2hpx);
    cudaFree(d_raCoords);
    cudaFree(d_decCoords);
    
    return cudaSuccess;
}
