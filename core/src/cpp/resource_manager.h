/**
 * 资源管理器 - 自动检测硬件资源并管理内存/显存使用
 * 实现灵活的缓存策略，根据可用资源动态调整
 */

#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <list>
#include <mutex>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <iostream>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

/**
 * 系统资源信息
 */
struct SystemResources {
    // CPU
    int cpuCores = 1;
    
    // 内存 (bytes)
    size_t totalMemory = 0;
    size_t availableMemory = 0;
    
    // GPU
    int gpuCount = 0;
    int currentGpu = 0;
    size_t gpuTotalMemory = 0;
    size_t gpuAvailableMemory = 0;
    std::string gpuName;
    
    // 推荐参数
    int recommendedThreads = 4;
    size_t recommendedCacheSize = 100;      // 文件缓存数量
    size_t recommendedCacheMemory = 0;      // 缓存使用的内存限制 (bytes)
    size_t recommendedGpuBatchSize = 1000;  // GPU 批处理大小
    
    void print() const {
        std::cout << "\n=== System Resources ===" << std::endl;
        std::cout << "CPU Cores: " << cpuCores << std::endl;
        std::cout << "Total Memory: " << (totalMemory / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "Available Memory: " << (availableMemory / (1024*1024*1024)) << " GB" << std::endl;
        
        if (gpuCount > 0) {
            std::cout << "GPU Count: " << gpuCount << std::endl;
            std::cout << "Current GPU: " << currentGpu << " (" << gpuName << ")" << std::endl;
            std::cout << "GPU Total Memory: " << (gpuTotalMemory / (1024*1024*1024)) << " GB" << std::endl;
            std::cout << "GPU Available Memory: " << (gpuAvailableMemory / (1024*1024*1024)) << " GB" << std::endl;
        } else {
            std::cout << "GPU: Not available" << std::endl;
        }
        
        std::cout << "\n=== Recommended Parameters ===" << std::endl;
        std::cout << "Threads: " << recommendedThreads << std::endl;
        std::cout << "Cache Size: " << recommendedCacheSize << " files" << std::endl;
        std::cout << "Cache Memory Limit: " << (recommendedCacheMemory / (1024*1024)) << " MB" << std::endl;
        std::cout << "GPU Batch Size: " << recommendedGpuBatchSize << std::endl;
    }
};

/**
 * 资源检测器
 */
class ResourceDetector {
public:
    /**
     * 检测所有系统资源
     */
    static SystemResources detect() {
        SystemResources res;
        
        // 检测 CPU
        detectCPU(res);
        
        // 检测内存
        detectMemory(res);
        
        // 检测 GPU
        detectGPU(res);
        
        // 计算推荐参数
        calculateRecommendations(res);
        
        return res;
    }
    
private:
    static void detectCPU(SystemResources& res) {
#ifdef __linux__
        res.cpuCores = sysconf(_SC_NPROCESSORS_ONLN);
#else
        res.cpuCores = std::thread::hardware_concurrency();
#endif
        if (res.cpuCores <= 0) res.cpuCores = 4;
    }
    
    static void detectMemory(SystemResources& res) {
#ifdef __linux__
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            res.totalMemory = si.totalram * si.mem_unit;
            res.availableMemory = si.freeram * si.mem_unit;
            
            // 也检查 /proc/meminfo 获取更准确的可用内存
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemAvailable:") == 0) {
                    std::istringstream iss(line);
                    std::string key;
                    size_t value;
                    iss >> key >> value;
                    res.availableMemory = value * 1024;  // KB to bytes
                    break;
                }
            }
        }
#endif
    }
    
    static void detectGPU(SystemResources& res) {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        
        if (err != cudaSuccess || deviceCount == 0) {
            res.gpuCount = 0;
            return;
        }
        
        res.gpuCount = deviceCount;
        
        // 获取当前 GPU
        cudaGetDevice(&res.currentGpu);
        
        // 获取 GPU 属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, res.currentGpu);
        res.gpuName = prop.name;
        res.gpuTotalMemory = prop.totalGlobalMem;
        
        // 获取可用显存
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        res.gpuAvailableMemory = free;
    }
    
    static void calculateRecommendations(SystemResources& res) {
        // 推荐线程数：CPU核心数的一半到全部，取决于是否有GPU
        if (res.gpuCount > 0) {
            // 有 GPU 时，CPU 主要负责 I/O，用一半核心
            res.recommendedThreads = std::max(2, res.cpuCores / 2);
        } else {
            // 纯 CPU 模式，用全部核心
            res.recommendedThreads = std::max(2, res.cpuCores - 1);
        }
        
        // 推荐缓存内存：可用内存的 30%
        res.recommendedCacheMemory = res.availableMemory * 3 / 10;
        
        // 假设每个 FITS 文件平均 50 MB
        const size_t avgFileSize = 50 * 1024 * 1024;
        res.recommendedCacheSize = res.recommendedCacheMemory / avgFileSize;
        res.recommendedCacheSize = std::max((size_t)10, std::min(res.recommendedCacheSize, (size_t)10000));
        
        // GPU 批处理大小：根据显存计算
        if (res.gpuCount > 0 && res.gpuAvailableMemory > 0) {
            // 每个坐标点约需要 32 bytes，加上图像数据
            // 保守估计，使用可用显存的 50%
            size_t usableGpuMem = res.gpuAvailableMemory / 2;
            res.recommendedGpuBatchSize = usableGpuMem / (1024 * 32);  // 假设每批处理1K个点
            res.recommendedGpuBatchSize = std::max((size_t)1000, std::min(res.recommendedGpuBatchSize, (size_t)100000));
        }
    }
};

/**
 * LRU 缓存模板类
 * 用于缓存 FITS 文件数据，自动管理内存
 */
template<typename Key, typename Value>
class LRUCache {
public:
    LRUCache(size_t maxItems = 100, size_t maxMemory = 0)
        : maxItems_(maxItems), maxMemory_(maxMemory), currentMemory_(0) {}
    
    /**
     * 设置缓存限制
     */
    void setLimits(size_t maxItems, size_t maxMemory) {
        std::lock_guard<std::mutex> lock(mutex_);
        maxItems_ = maxItems;
        maxMemory_ = maxMemory;
        evictIfNeeded();
    }
    
    /**
     * 检查是否存在
     */
    bool contains(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.find(key) != cache_.end();
    }
    
    /**
     * 获取值（如果存在）
     * 返回 nullptr 表示不存在
     */
    Value* get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return nullptr;
        }
        // 移动到 LRU 列表头部
        lruList_.erase(it->second.lruIt);
        lruList_.push_front(key);
        it->second.lruIt = lruList_.begin();
        return &(it->second.value);
    }
    
    /**
     * 插入值
     * @param key 键
     * @param value 值
     * @param size 值占用的内存大小（用于内存限制）
     */
    void put(const Key& key, Value value, size_t size = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 如果已存在，更新
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            currentMemory_ -= it->second.size;
            currentMemory_ += size;
            it->second.value = std::move(value);
            it->second.size = size;
            // 移动到 LRU 头部
            lruList_.erase(it->second.lruIt);
            lruList_.push_front(key);
            it->second.lruIt = lruList_.begin();
            return;
        }
        
        // 新插入，先检查是否需要淘汰
        currentMemory_ += size;
        evictIfNeeded();
        
        // 插入新项
        lruList_.push_front(key);
        CacheEntry entry;
        entry.value = std::move(value);
        entry.size = size;
        entry.lruIt = lruList_.begin();
        cache_[key] = std::move(entry);
    }
    
    /**
     * 删除指定项
     */
    void remove(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            currentMemory_ -= it->second.size;
            lruList_.erase(it->second.lruIt);
            cache_.erase(it);
        }
    }
    
    /**
     * 清空缓存
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        lruList_.clear();
        currentMemory_ = 0;
    }
    
    /**
     * 获取当前状态
     */
    size_t size() const { return cache_.size(); }
    size_t memoryUsage() const { return currentMemory_; }
    
    void printStatus() const {
        std::cout << "Cache: " << cache_.size() << "/" << maxItems_ << " items, "
                  << (currentMemory_ / (1024*1024)) << "/" << (maxMemory_ / (1024*1024)) << " MB"
                  << std::endl;
    }
    
private:
    struct CacheEntry {
        Value value;
        size_t size;
        typename std::list<Key>::iterator lruIt;
    };
    
    void evictIfNeeded() {
        // 按数量淘汰
        while (cache_.size() >= maxItems_ && !lruList_.empty()) {
            evictOne();
        }
        
        // 按内存淘汰
        while (maxMemory_ > 0 && currentMemory_ > maxMemory_ && !lruList_.empty()) {
            evictOne();
        }
    }
    
    void evictOne() {
        if (lruList_.empty()) return;
        Key lruKey = lruList_.back();
        lruList_.pop_back();
        auto it = cache_.find(lruKey);
        if (it != cache_.end()) {
            currentMemory_ -= it->second.size;
            cache_.erase(it);
        }
    }
    
    size_t maxItems_;
    size_t maxMemory_;
    size_t currentMemory_;
    std::map<Key, CacheEntry> cache_;
    std::list<Key> lruList_;
    mutable std::mutex mutex_;
};

#endif // RESOURCE_MANAGER_H
