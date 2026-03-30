/**
 * HpxFinder索引模块
 * 用于生成和查询HpxFinder空间索引
 * HpxFinder是HiPS生成过程中的关键索引，用于快速查找覆盖某个天球区域的源文件
 */

#ifndef HPX_FINDER_H
#define HPX_FINDER_H

#include <string>
#include <vector>
#include <map>
#include "fits_io.h"
#include "healpix_util.h"

// 源文件信息（用于HpxFinder索引）
struct SourceFileInfo {
    std::string filepath;  // 源文件路径（相对于输入目录）
    long cellMem;          // 单元格内存大小（字节）
    std::string filename;  // 文件名（不含路径）
    
    SourceFileInfo(const std::string& path = "", long mem = 0)
        : filepath(path), cellMem(mem) {
        // 从路径中提取文件名
        size_t pos = path.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = path.substr(pos + 1);
        } else {
            filename = path;
        }
    }
};

/**
 * HpxFinder索引生成和查询类
 */
class HpxFinder {
public:
    /**
     * 生成INDEX阶段的空间索引
     * 扫描输入目录中的FITS文件，为每个HEALPix tile生成索引文件
     * 
     * @param inputDir 输入目录（包含FITS文件）
     * @param outputDir 输出目录（HpxFinder将生成在outputDir/HpxFinder）
     * @param orderMax 最大HEALPix order
     * @return 成功返回true
     */
    static bool generateIndex(const std::string& inputDir, 
                              const std::string& outputDir, 
                              int orderMax);
    
    /**
     * 查询HpxFinder索引，获取覆盖指定HEALPix tile的源文件列表
     * 
     * @param hpxFinderPath HpxFinder目录路径
     * @param order HEALPix order
     * @param npix HEALPix pixel index
     * @return 源文件列表
     */
    static std::vector<SourceFileInfo> queryIndex(const std::string& hpxFinderPath,
                                                   int order, long npix);
    
    /**
     * 计算FITS文件覆盖的HEALPix像素列表
     * 
     * @param fitsData FITS文件数据
     * @param order HEALPix order
     * @return 覆盖的HEALPix像素索引列表
     */
    static std::vector<long> computeCoverage(const FitsData& fitsData, int order);
    
    /**
     * 写入HpxFinder索引文件（JSON格式，简化版）
     * 
     * @param indexPath 索引文件路径
     * @param sourceFiles 源文件列表
     */
    static void writeIndexFile(const std::string& indexPath,
                               const std::vector<SourceFileInfo>& sourceFiles);
    
    /**
     * 读取HpxFinder索引文件
     * 
     * @param indexPath 索引文件路径
     * @return 源文件列表
     */
    static std::vector<SourceFileInfo> readIndexFile(const std::string& indexPath);
    
private:
    /**
     * 简化版JSON解析（用于读取索引文件）
     */
    static std::string extractPathFromJsonLine(const std::string& line);
    static long extractCellMemFromJsonLine(const std::string& line);
};

#endif // HPX_FINDER_H
