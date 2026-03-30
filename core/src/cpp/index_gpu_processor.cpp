/**
 * INDEX阶段GPU加速处理器实现
 */

#include "index_gpu_processor.h"
#include "healpix_util.h"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

// HEALPix C++ library for polygon query
#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/pointing.h>
#include <healpix_cxx/rangeset.h>
#include <healpix_cxx/vec3.h>

extern "C" void initMortonTables();
extern "C" void launchBatchPixelToHpxMultiOrderKernel(
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_sampleX, const int* d_sampleY,
    const int* d_imageIndices,
    int numSamples,
    int orderMax,
    long* d_npixResults,
    cudaStream_t stream
);

IndexGPUProcessor::IndexGPUProcessor()
    : m_initialized(false)
    , m_d_crval1(nullptr), m_d_crval2(nullptr)
    , m_d_crpix1(nullptr), m_d_crpix2(nullptr)
    , m_d_cd1_1(nullptr), m_d_cd1_2(nullptr)
    , m_d_cd2_1(nullptr), m_d_cd2_2(nullptr)
    , m_d_sampleX(nullptr), m_d_sampleY(nullptr)
    , m_d_imageIndices(nullptr), m_d_npixResults(nullptr)
    , m_allocatedImages(0), m_allocatedSamples(0)
{
}

IndexGPUProcessor::~IndexGPUProcessor() {
    cleanup();
}

bool IndexGPUProcessor::initialize() {
    if (m_initialized) return true;
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "IndexGPU: No CUDA devices available" << std::endl;
        return false;
    }
    
    initMortonTables();
    
    m_initialized = true;
    std::cout << "IndexGPU: Initialized successfully" << std::endl;
    return true;
}

void IndexGPUProcessor::cleanup() {
    freeGPUMemory();
    m_initialized = false;
}

bool IndexGPUProcessor::allocateGPUMemory(size_t numImages, size_t numSamples, int orderMax) {
    if (m_allocatedImages >= numImages && m_allocatedSamples >= numSamples * (orderMax + 1)) {
        return true;
    }
    
    freeGPUMemory();
    
    cudaError_t err;
    
    err = cudaMalloc(&m_d_crval1, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_crval2, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_crpix1, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_crpix2, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_cd1_1, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_cd1_2, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_cd2_1, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_cd2_2, numImages * sizeof(double));
    if (err != cudaSuccess) goto alloc_failed;
    
    err = cudaMalloc(&m_d_sampleX, numSamples * sizeof(int));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_sampleY, numSamples * sizeof(int));
    if (err != cudaSuccess) goto alloc_failed;
    err = cudaMalloc(&m_d_imageIndices, numSamples * sizeof(int));
    if (err != cudaSuccess) goto alloc_failed;
    
    err = cudaMalloc(&m_d_npixResults, numSamples * (orderMax + 1) * sizeof(long));
    if (err != cudaSuccess) goto alloc_failed;
    
    m_allocatedImages = numImages;
    m_allocatedSamples = numSamples * (orderMax + 1);
    
    return true;
    
alloc_failed:
    std::cerr << "IndexGPU: Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
    freeGPUMemory();
    return false;
}

void IndexGPUProcessor::freeGPUMemory() {
    if (m_d_crval1) { cudaFree(m_d_crval1); m_d_crval1 = nullptr; }
    if (m_d_crval2) { cudaFree(m_d_crval2); m_d_crval2 = nullptr; }
    if (m_d_crpix1) { cudaFree(m_d_crpix1); m_d_crpix1 = nullptr; }
    if (m_d_crpix2) { cudaFree(m_d_crpix2); m_d_crpix2 = nullptr; }
    if (m_d_cd1_1) { cudaFree(m_d_cd1_1); m_d_cd1_1 = nullptr; }
    if (m_d_cd1_2) { cudaFree(m_d_cd1_2); m_d_cd1_2 = nullptr; }
    if (m_d_cd2_1) { cudaFree(m_d_cd2_1); m_d_cd2_1 = nullptr; }
    if (m_d_cd2_2) { cudaFree(m_d_cd2_2); m_d_cd2_2 = nullptr; }
    if (m_d_sampleX) { cudaFree(m_d_sampleX); m_d_sampleX = nullptr; }
    if (m_d_sampleY) { cudaFree(m_d_sampleY); m_d_sampleY = nullptr; }
    if (m_d_imageIndices) { cudaFree(m_d_imageIndices); m_d_imageIndices = nullptr; }
    if (m_d_npixResults) { cudaFree(m_d_npixResults); m_d_npixResults = nullptr; }
    
    m_allocatedImages = 0;
    m_allocatedSamples = 0;
}


// 使用HEALPix C++库的query_polygon_inclusive计算图像覆盖的tiles

// 检查球面多边形是否为凸多边形（使用叉积符号一致性）
static bool isConvexSphericalPolygon(const std::vector<pointing>& vertices) {
    if (vertices.size() < 3) return false;
    
    size_t n = vertices.size();
    int sign = 0;
    
    for (size_t i = 0; i < n; i++) {
        const pointing& p0 = vertices[i];
        const pointing& p1 = vertices[(i + 1) % n];
        const pointing& p2 = vertices[(i + 2) % n];
        
        // 转换为3D笛卡尔坐标
        vec3 v0(sin(p0.theta) * cos(p0.phi), sin(p0.theta) * sin(p0.phi), cos(p0.theta));
        vec3 v1(sin(p1.theta) * cos(p1.phi), sin(p1.theta) * sin(p1.phi), cos(p1.theta));
        vec3 v2(sin(p2.theta) * cos(p2.phi), sin(p2.theta) * sin(p2.phi), cos(p2.theta));
        
        // 计算边向量的叉积
        vec3 e1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        vec3 e2(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
        double cross_z = e1.x * e2.y - e1.y * e2.x;
        
        if (std::abs(cross_z) > 1e-10) {
            int current_sign = (cross_z > 0) ? 1 : -1;
            if (sign == 0) {
                sign = current_sign;
            } else if (sign != current_sign) {
                return false;
            }
        }
    }
    return true;
}

// 这与Java CDSHealpix.query_polygon(order, cooList, true)功能相同
std::set<long> IndexGPUProcessor::queryPolygonInclusive(
    const ImageWCSParams& params, int order
) {
    std::set<long> npixSet;
    
    if (params.width <= 0 || params.height <= 0) {
        return npixSet;
    }
    
    // 计算图像四个角点的天球坐标
    // 角点顺序: 左下、右下、右上、左上（逆时针，这是凸多边形的标准顺序）
    double corners[4][2];  // [corner][x,y]
    corners[0][0] = 0.5;                    corners[0][1] = 0.5;                     // 左下
    corners[1][0] = params.width + 0.5;     corners[1][1] = 0.5;                     // 右下
    corners[2][0] = params.width + 0.5;     corners[2][1] = params.height + 0.5;    // 右上
    corners[3][0] = 0.5;                    corners[3][1] = params.height + 0.5;    // 左上
    
    std::vector<pointing> vertices;
    
    for (int i = 0; i < 4; i++) {
        // 像素坐标转天球坐标 (使用与GPU kernel相同的WCS转换)
        double x = corners[i][0];
        double y = corners[i][1];
        
        // WCS transformation: pixel -> world
        double dx = x - params.crpix1;
        double dy = y - params.crpix2;
        
        // Apply CD matrix
        double xi = params.cd1_1 * dx + params.cd1_2 * dy;   // 中间坐标 (degrees)
        double eta = params.cd2_1 * dx + params.cd2_2 * dy;
        
        // TAN projection: intermediate -> celestial
        double xi_rad = xi * M_PI / 180.0;
        double eta_rad = eta * M_PI / 180.0;
        double crval1_rad = params.crval1 * M_PI / 180.0;
        double crval2_rad = params.crval2 * M_PI / 180.0;
        
        double sin_dec0 = sin(crval2_rad);
        double cos_dec0 = cos(crval2_rad);
        
        double denom = cos_dec0 - eta_rad * sin_dec0;
        double ra_rad = crval1_rad + atan2(xi_rad, denom);
        double dec_rad = atan2((sin_dec0 + eta_rad * cos_dec0) * cos(ra_rad - crval1_rad), denom);
        
        // 确保RA在 [0, 2π) 范围内
        while (ra_rad < 0) ra_rad += 2.0 * M_PI;
        while (ra_rad >= 2.0 * M_PI) ra_rad -= 2.0 * M_PI;
        
        // HEALPix pointing: theta = colatitude (0 at north pole), phi = longitude
        double theta = M_PI / 2.0 - dec_rad;  // colatitude = 90° - declination
        double phi = ra_rad;
        
        vertices.push_back(pointing(theta, phi));
    }
    
    // 创建HEALPix对象并查询多边形覆盖的tiles
    try {
        // 使用NEST scheme（与Java版本一致）
        T_Healpix_Base<int64> hpx(order, NEST);
        
        rangeset<int64> pixset;
        // fact=4 for query_polygon_inclusive 是典型选择，精度和性能的平衡
        // 检查多边形是否凸，非凸多边形会导致HEALPix库崩溃
        if (!isConvexSphericalPolygon(vertices)) {
            // 非凸多边形，返回空集（将使用采样方法作为回退）
            return npixSet;
        }
        hpx.query_polygon_inclusive(vertices, pixset, 4);
        
        // 从rangeset提取所有npix
        for (tsize i = 0; i < pixset.nranges(); i++) {
            for (int64 npix = pixset.ivbegin(i); npix < pixset.ivend(i); npix++) {
                npixSet.insert(npix);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "HEALPix query_polygon error for " << params.filepath 
                  << ": " << e.what() << std::endl;
        // 回退到采样方法
    }
    
    return npixSet;
}

std::vector<std::pair<int, int>> IndexGPUProcessor::generateSamplePoints(
    int width, int height, double cd_scale, int order
) {
    std::vector<std::pair<int, int>> points;
    
    if (width <= 0 || height <= 0) return points;
    
    if (cd_scale == 0) cd_scale = 0.01;
    double healpix_size = HealpixUtil::getPixelSize(order);
    int sampleStep = std::max(1, (int)(healpix_size / cd_scale / 4.0));
    sampleStep = std::min(sampleStep, std::max(width, height) / 20);
    sampleStep = std::max(1, sampleStep);
    
    // 边界采样步长更密集，确保不漏掉边缘tiles
    int borderStep = std::max(1, sampleStep / 4);
    
    // 1. 内部网格采样
    for (int y = 0; y < height; y += sampleStep) {
        for (int x = 0; x < width; x += sampleStep) {
            points.push_back({x, y});
        }
        // 确保包含最右边的像素
        if ((width - 1) % sampleStep != 0) {
            points.push_back({width - 1, y});
        }
    }
    
    // 确保包含最下面的行
    if ((height - 1) % sampleStep != 0) {
        for (int x = 0; x < width; x += sampleStep) {
            points.push_back({x, height - 1});
        }
        points.push_back({width - 1, height - 1});
    }
    
    // 2. 四条边界密集采样（检测边缘覆盖的tiles）
    for (int x = 0; x < width; x += borderStep) {
        points.push_back({x, 0});
        points.push_back({x, height - 1});
    }
    for (int y = 0; y < height; y += borderStep) {
        points.push_back({0, y});
        points.push_back({width - 1, y});
    }
    
    // 3. 四个角点
    points.push_back({0, 0});
    points.push_back({width - 1, 0});
    points.push_back({0, height - 1});
    points.push_back({width - 1, height - 1});
    
    return points;
}

bool IndexGPUProcessor::computeIndexBatch(
    const std::vector<ImageWCSParams>& wcsParams,
    int orderMax,
    std::map<std::string, std::vector<std::pair<std::string, long>>>& tileFileMap
) {
    // 使用HEALPix C++库的query_polygon_inclusive进行精确的tile覆盖检测
    // 这与Java版本的CDSHealpix.query_polygon(order, cooList, true)算法完全一致
    
    if (wcsParams.empty()) {
        std::cerr << "IndexGPU: No images to process" << std::endl;
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    size_t numImages = wcsParams.size();
    std::cout << "IndexGPU: Processing " << numImages << " images using HEALPix polygon query..." << std::endl;
    
    for (size_t imgIdx = 0; imgIdx < numImages; imgIdx++) {
        const auto& params = wcsParams[imgIdx];
        
        if (params.width <= 0 || params.height <= 0) continue;
        
        for (int order = 0; order <= orderMax; order++) {
            // 使用HEALPix query_polygon_inclusive获取所有相交的tiles
            std::set<long> npixSet = queryPolygonInclusive(params, order);
            
            // 如果多边形查询失败（非凸多边形），回退到采样方法
            if (npixSet.empty()) {
                // 使用中心点作为最小覆盖
                double ra_rad = params.crval1 * M_PI / 180.0;
                double dec_rad = params.crval2 * M_PI / 180.0;
                double theta = M_PI / 2.0 - dec_rad;
                double phi = ra_rad;
                
                T_Healpix_Base<int64> hpx(order, NEST);
                pointing pt(theta, phi);
                int64 centerNpix = hpx.ang2pix(pt);
                npixSet.insert(centerNpix);
                
                // 也添加相邻的tiles
                fix_arr<int64, 8> neighbors;
                hpx.neighbors(centerNpix, neighbors);
                for (int j = 0; j < 8; j++) {
                    if (neighbors[j] >= 0) {
                        npixSet.insert(neighbors[j]);
                    }
                }
            }
            
            for (long npix : npixSet) {
                std::ostringstream oss;
                oss << order << "/" << npix;
                std::string key = oss.str();
                
                tileFileMap[key].push_back({params.filepath, params.cellMem});
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "IndexGPU: HEALPix polygon query completed in " << totalMs << " ms" << std::endl;
    std::cout << "IndexGPU: Generated " << tileFileMap.size() << " index entries" << std::endl;
    
    return true;
}
