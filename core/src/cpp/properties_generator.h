/**
 * HiPS properties元数据文件生成器
 */

#ifndef PROPERTIES_GENERATOR_H
#define PROPERTIES_GENERATOR_H

#include <string>

/**
 * Properties生成器类
 */
class PropertiesGenerator {
public:
    /**
     * 生成properties文件
     * 
     * @param hipsDir HiPS输出目录
     * @param title 数据集标题
     * @param order 最大HEALPix order
     * @param tileWidth Tile宽度
     * @param bitpix FITS BITPIX
     * @param tilesCount 生成的tiles数量
     * @param pixelCutMin 像素值最小值
     * @param pixelCutMax 像素值最大值
     * @return 成功返回true
     */
    static bool generateProperties(
        const std::string& hipsDir,
        const std::string& title,
        int order,
        int tileWidth,
        int bitpix,
        int tilesCount,
        double pixelCutMin,
        double pixelCutMax
    );
};

#endif // PROPERTIES_GENERATOR_H
