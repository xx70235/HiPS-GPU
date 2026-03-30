/**
 * CUDA双线性插值Kernel实现
 * 直接使用现有的CUDA代码，作为独立的CUDA源文件
 * 可以被C++程序直接链接，无需JNI
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define TILE_SIZE 16

// Device function: Get pixel value with bounds checking
__device__ __forceinline__ double getPixelBounds(const double *image, int x, int y, int width, int height, double myBlank) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return myBlank;
    }
    return image[y * width + x];
}

// Shared memory optimized bilinear interpolation kernel
__global__ void bilinearInterpolationSharedMemKernel(
    const double *inputImage, double *outputImage,
    int width, int height,
    const double *coordsX, const double *coordsY,
    int imageWidth, int imageHeight,
    double myBlank
) {
    // Shared memory declaration
    __shared__ double sharedData[(TILE_SIZE + 1) * (TILE_SIZE + 1)];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x >= imageWidth || y >= imageHeight) return;

    int idx = y * imageWidth + x;

    // Load data to shared memory
    int sharedIdx = ty * (TILE_SIZE + 1) + tx;
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        double coordX = coordsX[idx];
        double coordY = coordsY[idx];
        
        int srcX = __double2int_rd(coordX);
        int srcY = __double2int_rd(coordY);
        
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            sharedData[sharedIdx] = inputImage[srcY * width + srcX];
        } else {
            sharedData[sharedIdx] = myBlank;
        }
    }
    
    __syncthreads();

    // Perform bilinear interpolation
    double coordX = coordsX[idx];
    double coordY = coordsY[idx];

    if (coordX < 0 || coordX >= width - 1 || coordY < 0 || coordY >= height - 1) {
        outputImage[idx] = myBlank;
        return;
    }

    int ix1 = __double2int_rd(coordX);
    int iy1 = __double2int_rd(coordY);
    int ix2 = ix1 + 1;
    int iy2 = iy1 + 1;

    double a0 = getPixelBounds(inputImage, ix1, iy1, width, height, myBlank);
    double a1 = getPixelBounds(inputImage, ix2, iy1, width, height, myBlank);
    double a2 = getPixelBounds(inputImage, ix1, iy2, width, height, myBlank);
    double a3 = getPixelBounds(inputImage, ix2, iy2, width, height, myBlank);

    bool b0 = isnan(a0) || a0 == myBlank;
    bool b1 = isnan(a1) || a1 == myBlank;
    bool b2 = isnan(a2) || a2 == myBlank;
    bool b3 = isnan(a3) || a3 == myBlank;

    if (b0 && b1 && b2 && b3) {
        outputImage[idx] = myBlank;
        return;
    }

    if (b0 || b1 || b2 || b3) {
        double replacement = !b0 ? a0 : (!b1 ? a1 : (!b2 ? a2 : a3));
        if (b0) a0 = replacement;
        if (b1) a1 = replacement;
        if (b2) a2 = replacement;
        if (b3) a3 = replacement;
    }

    double wx = coordX - ix1;
    double wy = coordY - iy1;

    double interpolatedValue = (1.0 - wx) * (1.0 - wy) * a0 +
                               wx * (1.0 - wy) * a1 +
                               (1.0 - wx) * wy * a2 +
                               wx * wy * a3;

    outputImage[idx] = interpolatedValue;
}

// Batch bilinear interpolation kernel - process multiple coordinates at once
__global__ void bilinearInterpolationBatchKernel(
    const double *inputImage, 
    double *output,
    const double *coordsX,
    const double *coordsY,
    int numCoords,
    int width, int height,
    double myBlank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCoords) return;
    
    double coordX = coordsX[idx];
    double coordY = coordsY[idx];
    
    // Check bounds
    if (coordX < 0 || coordX >= width - 1 || coordY < 0 || coordY >= height - 1) {
        output[idx] = myBlank;
        return;
    }

    int ix1 = __double2int_rd(coordX);
    int iy1 = __double2int_rd(coordY);
    int ix2 = ix1 + 1;
    int iy2 = iy1 + 1;

    double a0 = getPixelBounds(inputImage, ix1, iy1, width, height, myBlank);
    double a1 = getPixelBounds(inputImage, ix2, iy1, width, height, myBlank);
    double a2 = getPixelBounds(inputImage, ix1, iy2, width, height, myBlank);
    double a3 = getPixelBounds(inputImage, ix2, iy2, width, height, myBlank);

    bool b0 = isnan(a0) || a0 == myBlank;
    bool b1 = isnan(a1) || a1 == myBlank;
    bool b2 = isnan(a2) || a2 == myBlank;
    bool b3 = isnan(a3) || a3 == myBlank;

    if (b0 && b1 && b2 && b3) {
        output[idx] = myBlank;
        return;
    }

    if (b0 || b1 || b2 || b3) {
        double replacement = !b0 ? a0 : (!b1 ? a1 : (!b2 ? a2 : a3));
        if (b0) a0 = replacement;
        if (b1) a1 = replacement;
        if (b2) a2 = replacement;
        if (b3) a3 = replacement;
    }

    double wx = coordX - ix1;
    double wy = coordY - iy1;

    output[idx] = (1.0 - wx) * (1.0 - wy) * a0 +
                  wx * (1.0 - wy) * a1 +
                  (1.0 - wx) * wy * a2 +
                  wx * wy * a3;
}

// Single point interpolation kernel (simplified version)
__global__ void bilinearInterpolationSingleKernel(
    const double *inputImage, 
    double *output,
    int width, int height,
    double coordX, double coordY,
    double myBlank
) {
    if (coordX < 0 || coordX >= width - 1 || coordY < 0 || coordY >= height - 1) {
        *output = myBlank;
        return;
    }

    int ix1 = __double2int_rd(coordX);
    int iy1 = __double2int_rd(coordY);
    int ix2 = ix1 + 1;
    int iy2 = iy1 + 1;

    double a0 = getPixelBounds(inputImage, ix1, iy1, width, height, myBlank);
    double a1 = getPixelBounds(inputImage, ix2, iy1, width, height, myBlank);
    double a2 = getPixelBounds(inputImage, ix1, iy2, width, height, myBlank);
    double a3 = getPixelBounds(inputImage, ix2, iy2, width, height, myBlank);

    bool b0 = isnan(a0) || a0 == myBlank;
    bool b1 = isnan(a1) || a1 == myBlank;
    bool b2 = isnan(a2) || a2 == myBlank;
    bool b3 = isnan(a3) || a3 == myBlank;

    if (b0 && b1 && b2 && b3) {
        *output = myBlank;
        return;
    }

    if (b0 || b1 || b2 || b3) {
        double replacement = !b0 ? a0 : (!b1 ? a1 : (!b2 ? a2 : a3));
        if (b0) a0 = replacement;
        if (b1) a1 = replacement;
        if (b2) a2 = replacement;
        if (b3) a3 = replacement;
    }

    double wx = coordX - ix1;
    double wy = coordY - iy1;

    double interpolatedValue = (1.0 - wx) * (1.0 - wy) * a0 +
                               wx * (1.0 - wy) * a1 +
                               (1.0 - wx) * wy * a2 +
                               wx * wy * a3;

    *output = interpolatedValue;
}

// Host包装函数：从C++代码调用CUDA kernel
extern "C" void launchBilinearInterpolationSingleKernel(
    const double* inputImage,
    double* output,
    int width, int height,
    double coordX, double coordY,
    double myBlank
) {
    bilinearInterpolationSingleKernel<<<1, 1>>>(
        inputImage, output, width, height, coordX, coordY, myBlank
    );
}

// 批量插值启动函数
extern "C" void launchBilinearInterpolationBatchKernel(
    const double* inputImage,
    double* output,
    const double* coordsX,
    const double* coordsY,
    int numCoords,
    int width, int height,
    double myBlank
) {
    // 计算线程块和网格大小
    int blockSize = 256;
    int numBlocks = (numCoords + blockSize - 1) / blockSize;
    
    bilinearInterpolationBatchKernel<<<numBlocks, blockSize>>>(
        inputImage, output, coordsX, coordsY, numCoords, width, height, myBlank
    );
}
