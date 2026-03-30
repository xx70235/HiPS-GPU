/**
 * CUDA双线性插值C++包装
 * 将CUDA实现包装为C++函数，直接使用，无需JNI
 */

#include "bilinear_interpolation_cuda.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <cmath>

// CUDA kernel声明（在bilinear_interpolation_cuda_kernels.cu中实现）
extern "C" void bilinearInterpolationSharedMemKernel(
    const double* d_inputImage,
    double* d_outputImage,
    int width, int height,
    const double* d_coordsX,
    const double* d_coordsY,
    int imageWidth, int imageHeight,
    double myBlank
);

// Host包装函数（在bilinear_interpolation_cuda_kernels.cu中实现）
extern "C" void launchBilinearInterpolationSingleKernel(
    const double* inputImage,
    double* output,
    int width, int height,
    double coordX, double coordY,
    double myBlank
);

// 批量插值启动函数（在bilinear_interpolation_cuda_kernels.cu中实现）
extern "C" void launchBilinearInterpolationBatchKernel(
    const double* inputImage,
    double* output,
    const double* coordsX,
    const double* coordsY,
    int numCoords,
    int width, int height,
    double myBlank
);

/**
 * CUDA双线性插值实现
 * 直接调用CUDA kernel，无需JNI层
 */
double bilinearInterpolationCUDAImpl(
    const float* inputImage,
    int width, int height,
    double x, double y,
    float blankValue,
    int bitpix
) {
    // 检查CUDA是否可用
    cudaError_t cudaStatus;
    int deviceCount = 0;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        // CUDA不可用，使用CPU实现作为fallback
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    // 边界检查
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return blankValue;
    }
    
    // 将float转换为double（CUDA kernel使用double）
    size_t imageSize = width * height * sizeof(double);
    double* h_inputImage = new double[width * height];
    for (int i = 0; i < width * height; i++) {
        h_inputImage[i] = (double)inputImage[i];
    }
    
    // 分配GPU内存
    double* d_inputImage = nullptr;
    double* d_output = nullptr;
    
    cudaStatus = cudaMalloc((void**)&d_inputImage, imageSize);
    if (cudaStatus != cudaSuccess) {
        delete[] h_inputImage;
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    cudaStatus = cudaMalloc((void**)&d_output, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        delete[] h_inputImage;
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    // 复制输入数据到GPU
    cudaStatus = cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        cudaFree(d_output);
        delete[] h_inputImage;
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    // 启动CUDA kernel（单点插值）- 通过host包装函数调用
    launchBilinearInterpolationSingleKernel(
        d_inputImage, d_output, width, height, x, y, (double)blankValue
    );
    
    // 检查kernel执行错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        cudaFree(d_output);
        delete[] h_inputImage;
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    // 等待kernel完成
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        cudaFree(d_output);
        delete[] h_inputImage;
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    // 复制结果回CPU
    double result;
    cudaStatus = cudaMemcpy(&result, d_output, sizeof(double), cudaMemcpyDeviceToHost);
    
    // 清理GPU内存
    cudaFree(d_inputImage);
    cudaFree(d_output);
    delete[] h_inputImage;
    
    if (cudaStatus != cudaSuccess) {
        return bilinearInterpolationCPU(inputImage, width, height, x, y, blankValue);
    }
    
    return result;
}

/**
 * CPU fallback实现（当CUDA不可用时使用）
 */
double bilinearInterpolationCPU(
    const float* inputImage,
    int width, int height,
    double x, double y,
    float blankValue
) {
    // 边界检查
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return blankValue;
    }
    
    // 转换为整数坐标
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    
    // 确保在边界内
    if (ix < 0) ix = 0;
    if (ix >= width - 1) ix = width - 2;
    if (iy < 0) iy = 0;
    if (iy >= height - 1) iy = height - 2;
    
    // 计算小数部分
    double fx = x - ix;
    double fy = y - iy;
    
    // 获取四个角的像素值
    double v00 = inputImage[iy * width + ix];
    double v10 = inputImage[iy * width + (ix + 1)];
    double v01 = inputImage[(iy + 1) * width + ix];
    double v11 = inputImage[(iy + 1) * width + (ix + 1)];
    
    // 检查空白值
    if ((v00 == blankValue || std::isnan(v00)) &&
        (v10 == blankValue || std::isnan(v10)) &&
        (v01 == blankValue || std::isnan(v01)) &&
        (v11 == blankValue || std::isnan(v11))) {
        return blankValue;
    }
    
    // 双线性插值
    double v0 = v00 * (1 - fx) + v10 * fx;
    double v1 = v01 * (1 - fx) + v11 * fx;
    double result = v0 * (1 - fy) + v1 * fy;
    
    return result;
}

/**
 * 批量CUDA双线性插值实现
 * 一次性处理多个坐标点，大幅提升性能
 */
bool bilinearInterpolationBatchCUDA(
    const float* inputImage,
    int width, int height,
    const double* coordsX,
    const double* coordsY,
    double* results,
    int numCoords,
    float blankValue
) {
    // 检查CUDA是否可用
    cudaError_t cudaStatus;
    int deviceCount = 0;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        return false;  // CUDA不可用
    }
    
    if (numCoords <= 0) {
        return true;  // 没有要处理的坐标
    }
    
    // 将float转换为double（CUDA kernel使用double）
    size_t imageSize = (size_t)width * height * sizeof(double);
    double* h_inputImage = new double[width * height];
    for (int i = 0; i < width * height; i++) {
        h_inputImage[i] = (double)inputImage[i];
    }
    
    // 分配GPU内存
    double* d_inputImage = nullptr;
    double* d_output = nullptr;
    double* d_coordsX = nullptr;
    double* d_coordsY = nullptr;
    
    bool success = true;
    
    // 分配图像内存
    cudaStatus = cudaMalloc((void**)&d_inputImage, imageSize);
    if (cudaStatus != cudaSuccess) {
        delete[] h_inputImage;
        return false;
    }
    
    // 分配输出内存
    cudaStatus = cudaMalloc((void**)&d_output, numCoords * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        delete[] h_inputImage;
        return false;
    }
    
    // 分配坐标内存
    cudaStatus = cudaMalloc((void**)&d_coordsX, numCoords * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        cudaFree(d_output);
        delete[] h_inputImage;
        return false;
    }
    
    cudaStatus = cudaMalloc((void**)&d_coordsY, numCoords * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_inputImage);
        cudaFree(d_output);
        cudaFree(d_coordsX);
        delete[] h_inputImage;
        return false;
    }
    
    // 复制数据到GPU
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coordsX, coordsX, numCoords * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coordsY, coordsY, numCoords * sizeof(double), cudaMemcpyHostToDevice);
    
    // 启动批量CUDA kernel
    launchBilinearInterpolationBatchKernel(
        d_inputImage, d_output, d_coordsX, d_coordsY,
        numCoords, width, height, (double)blankValue
    );
    
    // 检查kernel执行错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        success = false;
    }
    
    // 等待kernel完成
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        success = false;
    }
    
    // 复制结果回CPU
    if (success) {
        cudaMemcpy(results, d_output, numCoords * sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    // 清理GPU内存
    cudaFree(d_inputImage);
    cudaFree(d_output);
    cudaFree(d_coordsX);
    cudaFree(d_coordsY);
    delete[] h_inputImage;
    
    return success;
}

/**
 * CPU批量双线性插值实现（当CUDA不可用时使用）
 */
void bilinearInterpolationBatchCPU(
    const float* inputImage,
    int width, int height,
    const double* coordsX,
    const double* coordsY,
    double* results,
    int numCoords,
    float blankValue
) {
    for (int i = 0; i < numCoords; i++) {
        results[i] = bilinearInterpolationCPU(
            inputImage, width, height,
            coordsX[i], coordsY[i],
            blankValue
        );
    }
}
