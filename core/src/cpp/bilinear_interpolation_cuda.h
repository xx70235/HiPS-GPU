/**
 * CUDA双线性插值C++接口
 * 直接使用CUDA实现，无需JNI层
 */

#ifndef BILINEAR_INTERPOLATION_CUDA_H
#define BILINEAR_INTERPOLATION_CUDA_H

#include <vector>

/**
 * CUDA双线性插值实现
 * 直接调用CUDA kernel，无需JNI包装
 * 
 * @param inputImage 输入图像数据（float数组）
 * @param width 图像宽度
 * @param height 图像高度
 * @param x 插值点x坐标（像素坐标）
 * @param y 插值点y坐标（像素坐标）
 * @param blankValue 空白值
 * @param bitpix FITS BITPIX值（目前未使用，保留用于未来扩展）
 * @return 插值结果
 */
double bilinearInterpolationCUDAImpl(
    const float* inputImage,
    int width, int height,
    double x, double y,
    float blankValue,
    int bitpix
);

/**
 * CPU fallback双线性插值实现
 * 当CUDA不可用时使用
 */
double bilinearInterpolationCPU(
    const float* inputImage,
    int width, int height,
    double x, double y,
    float blankValue
);

/**
 * 批量CUDA双线性插值实现
 * 一次性处理多个坐标点，大幅提升性能
 * 
 * @param inputImage 输入图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param coordsX X坐标数组
 * @param coordsY Y坐标数组
 * @param results 输出结果数组
 * @param numCoords 坐标点数量
 * @param blankValue 空白值
 * @return 成功返回true，CUDA不可用返回false
 */
bool bilinearInterpolationBatchCUDA(
    const float* inputImage,
    int width, int height,
    const double* coordsX,
    const double* coordsY,
    double* results,
    int numCoords,
    float blankValue
);

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
);

#endif // BILINEAR_INTERPOLATION_CUDA_H
