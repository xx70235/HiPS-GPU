#pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <vector>
#include <chrono>
#include <iostream>
#include <sstream>
#include "fits_writer.h"
#include "healpix_util.h"

/**
 * 异步文件写入器
 * - 使用生产者-消费者模式
 * - 后台线程池并行写入
 * - GPU计算不阻塞等待I/O
 */
class AsyncWriter {
public:
    struct WriteTask {
        std::string filepath;
        std::vector<double> data;
        int width;
        int height;
        int bitpix;
        int order;
        long npix;
        std::string frame;
        double blank;
    };

private:
    std::queue<WriteTask> taskQueue_;
    std::mutex queueMutex_;
    std::condition_variable cvNotEmpty_;
    std::condition_variable cvNotFull_;
    
    std::vector<std::thread> workers_;
    std::atomic<bool> shutdown_;
    std::atomic<int> successCount_;
    std::atomic<int> failCount_;
    std::atomic<int> pendingCount_;
    
    size_t maxQueueSize_;
    int numThreads_;
    
    // 统计
    std::atomic<long long> totalWriteTimeNs_;
    std::atomic<int> writeCount_;

    void workerThread() {
        while (true) {
            WriteTask task;
            
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                cvNotEmpty_.wait(lock, [this] {
                    return !taskQueue_.empty() || shutdown_;
                });
                
                if (shutdown_ && taskQueue_.empty()) {
                    return;
                }
                
                task = std::move(taskQueue_.front());
                taskQueue_.pop();
                cvNotFull_.notify_one();
            }
            
            // 执行写入（不持有锁）
            auto startTime = std::chrono::steady_clock::now();
            
            TileData tile(task.width, task.height, task.bitpix);
            tile.blank = task.blank;
            for (int y = 0; y < task.height; y++) {
                for (int x = 0; x < task.width; x++) {
                    tile.setPixel(x, y, task.data[y * task.width + x]);
                }
            }
            
            bool success = FitsWriter::writeTileFile(task.filepath, tile, task.order, task.npix, task.frame);
            
            auto endTime = std::chrono::steady_clock::now();
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
            totalWriteTimeNs_ += ns;
            writeCount_++;
            
            if (success) {
                successCount_++;
            } else {
                failCount_++;
            }
            pendingCount_--;
        }
    }

public:
    // numThreads: 0=自动（NFS优化：16-32线程，本地SSD可更多）
    AsyncWriter(int numThreads = 0, size_t maxQueueSize = 10000)
        : shutdown_(false), successCount_(0), failCount_(0), pendingCount_(0),
          maxQueueSize_(maxQueueSize), totalWriteTimeNs_(0), writeCount_(0)
    {
        // NFS优化：限制并发写入线程数
        // 太多线程会导致NFS锁竞争，反而变慢
        if (numThreads <= 0) {
            int cpuCores = (int)std::thread::hardware_concurrency();
            // NFS场景：16-32线程通常最优
            // 本地SSD：可以用更多线程
            numThreads = std::min(32, std::max(8, cpuCores / 8));
        }
        numThreads_ = numThreads;
    }
    
    void start() {
        shutdown_ = false;
        for (int i = 0; i < numThreads_; i++) {
            workers_.emplace_back(&AsyncWriter::workerThread, this);
        }
        std::cout << "AsyncWriter: Started " << numThreads_ << " writer threads" << std::endl;
    }
    
    // 提交写入任务（可能阻塞如果队列满）
    void submit(WriteTask&& task) {
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            cvNotFull_.wait(lock, [this] {
                return taskQueue_.size() < maxQueueSize_ || shutdown_;
            });
            
            if (shutdown_) return;
            
            taskQueue_.push(std::move(task));
            pendingCount_++;
        }
        cvNotEmpty_.notify_one();
    }
    
    // 便捷方法：从GPU结果直接提交
    void submitTile(const std::string& outputDir, int order, long npix,
                    const std::vector<double>& data, int width, int height,
                    int bitpix, const std::string& frame, double blank) {
        WriteTask task;
        int dir = HealpixUtil::getDirNumber(npix);
        std::ostringstream oss;
        oss << outputDir << "/Norder" << order << "/Dir" << dir << "/Npix" << npix << ".fits";
        task.filepath = oss.str();
        task.data = data;  // 拷贝数据
        task.width = width;
        task.height = height;
        task.bitpix = bitpix;
        task.order = order;
        task.npix = npix;
        task.frame = frame;
        task.blank = blank;
        submit(std::move(task));
    }
    
    // 等待所有任务完成
    void waitAll() {
        while (pendingCount_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    // 关闭写入器
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            shutdown_ = true;
        }
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    ~AsyncWriter() {
        shutdown();
    }
    
    int getSuccessCount() const { return successCount_; }
    int getFailCount() const { return failCount_; }
    int getPendingCount() const { return pendingCount_; }
    int getQueueSize() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex_));
        return (int)taskQueue_.size();
    }
    
    void printStats() const {
        double avgWriteMs = writeCount_ > 0 ? (totalWriteTimeNs_ / 1e6) / writeCount_ : 0;
        std::cout << "AsyncWriter stats: " << successCount_ << " success, " 
                  << failCount_ << " failed, avg write time: " << avgWriteMs << " ms/tile" << std::endl;
    }
};
