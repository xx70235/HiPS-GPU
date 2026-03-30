/**
 * GPU-accelerated HEALPix coordinate computation - Header
 */

#ifndef HEALPIX_CUDA_CUH
#define HEALPIX_CUDA_CUH

#include <cuda_runtime.h>
#include <vector>

// External C function declaration
extern "C" cudaError_t computeHEALPixCoordsGPU(
    const int* h_tileOrders,
    const long* h_tileNpixs,
    const int* h_xy2hpx,
    int numTiles,
    int tileWidth,
    double* h_raCoords,
    double* h_decCoords
);

#endif // HEALPIX_CUDA_CUH
