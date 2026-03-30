/**
 * INDEX阶段GPU加速处理器
 * 使用HEALPix C++库进行精确的tile覆盖检测
 */

#ifndef INDEX_GPU_PROCESSOR_H
#define INDEX_GPU_PROCESSOR_H

#include <vector>
#include <string>
#include <map>
#include <set>
#include "fits_io.h"

/**
 * 图像WCS参数结构（用于GPU传输）
 */
struct ImageWCSParams {
    double crval1, crval2;
    double crpix1, crpix2;
    double cd1_1, cd1_2;
    double cd2_1, cd2_2;
    int width, height;
    std::string filepath;
    long cellMem;
    
    ImageWCSParams() : crval1(0), crval2(0), crpix1(0), crpix2(0),
                       cd1_1(0), cd1_2(0), cd2_1(0), cd2_2(0),
                       width(0), height(0), cellMem(0) {}
};

/**
 * GPU索引处理器类
 */
class IndexGPUProcessor {
public:
    IndexGPUProcessor();
    ~IndexGPUProcessor();
    
    bool initialize();
    void cleanup();
    
    bool computeIndexBatch(
        const std::vector<ImageWCSParams>& wcsParams,
        int orderMax,
        std::map<std::string, std::vector<std::pair<std::string, long>>>& tileFileMap
    );
    
    static std::vector<std::pair<int, int>> generateSamplePoints(
        int width, int height, double cd_scale, int order
    );
    
    // HEALPix polygon query for accurate tile coverage detection
    static std::set<long> queryPolygonInclusive(
        const ImageWCSParams& params, int order
    );
    
private:
    bool m_initialized;
    
    double* m_d_crval1;
    double* m_d_crval2;
    double* m_d_crpix1;
    double* m_d_crpix2;
    double* m_d_cd1_1;
    double* m_d_cd1_2;
    double* m_d_cd2_1;
    double* m_d_cd2_2;
    
    int* m_d_sampleX;
    int* m_d_sampleY;
    int* m_d_imageIndices;
    long* m_d_npixResults;
    
    size_t m_allocatedImages;
    size_t m_allocatedSamples;
    
    bool allocateGPUMemory(size_t numImages, size_t numSamples, int orderMax);
    void freeGPUMemory();
};

#endif // INDEX_GPU_PROCESSOR_H
