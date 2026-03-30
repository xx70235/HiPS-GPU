#ifndef BATCH_FITS_WRITER_H
#define BATCH_FITS_WRITER_H

#include <string>
#include <vector>
#include <fitsio.h>
#include <omp.h>
#include <filesystem>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace fs = std::filesystem;

/**
 * 批量FITS文件写入器
 * 优化策略：
 * 1. 预先创建所有目录结构（避免并发mkdir冲突）
 * 2. 使用多线程池写入
 * 3. 每个线程独立的cfitsio实例（避免锁竞争）
 */
class BatchFitsWriter {
public:
    struct TileToWrite {
        std::string path;
        std::vector<float> data;
        int width, height;
        int order;
        long npix;
    };
    
    /**
     * 批量写入tiles
     * @param tiles 要写入的tiles列表
     * @param numThreads 并行线程数
     * @return 成功写入的数量
     */
    static int writeTilesBatch(std::vector<TileToWrite>& tiles, int numThreads = 0) {
        if (tiles.empty()) return 0;
        
        if (numThreads <= 0) {
            numThreads = omp_get_max_threads();
        }
        
        // 1. 预先创建所有目录
        std::set<std::string> dirs;
        for (const auto& tile : tiles) {
            fs::path p(tile.path);
            if (p.has_parent_path()) {
                dirs.insert(p.parent_path().string());
            }
        }
        for (const auto& dir : dirs) {
            fs::create_directories(dir);
        }
        
        // 2. 并行写入
        std::atomic<int> successCount(0);
        std::atomic<int> progressCount(0);
        int totalTiles = tiles.size();
        
        #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 50)
        for (size_t i = 0; i < tiles.size(); i++) {
            const auto& tile = tiles[i];
            
            if (writeSingleTile(tile)) {
                successCount++;
            }
            
            int prog = ++progressCount;
            if (prog % 500 == 0) {
                #pragma omp critical
                {
                    std::cout << "  Written " << prog << "/" << totalTiles << " tiles..." << std::endl;
                }
            }
        }
        
        return successCount.load();
    }
    
private:
    static bool writeSingleTile(const TileToWrite& tile) {
        fitsfile* fptr = nullptr;
        int status = 0;
        
        // 创建文件
        std::string fileFmt = "!" + tile.path;  // ! 表示覆盖
        if (fits_create_file(&fptr, fileFmt.c_str(), &status)) {
            return false;
        }
        
        // 创建图像
        long naxes[2] = {tile.width, tile.height};
        if (fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status)) {
            fits_close_file(fptr, &status);
            return false;
        }
        
        // 写入数据
        long fpixel[2] = {1, 1};
        if (fits_write_pix(fptr, TFLOAT, fpixel, tile.data.size(), 
                          (void*)tile.data.data(), &status)) {
            fits_close_file(fptr, &status);
            return false;
        }
        
        // 写入HiPS关键字
        char comment[80] = "";
        fits_write_key(fptr, TINT, "HIPSORD", (void*)&tile.order, "HiPS order", &status);
        fits_write_key(fptr, TLONG, "HIPSNPIX", (void*)&tile.npix, "HiPS npix", &status);
        
        fits_close_file(fptr, &status);
        return status == 0;
    }
};

#endif // BATCH_FITS_WRITER_H
