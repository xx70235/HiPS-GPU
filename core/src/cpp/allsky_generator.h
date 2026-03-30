/**
 * Allsky概览图生成器
 * 将Order 3的768个tiles拼接成低分辨率的全天图
 */

#ifndef ALLSKY_GENERATOR_H
#define ALLSKY_GENERATOR_H

#include <string>
#include <vector>
#include "fits_io.h"

/**
 * Allsky生成器类
 */
class AllskyGenerator {
public:
    /**
     * 生成Allsky.fits文件
     * 
     * @param hipsDir HiPS输出目录
     * @param order HEALPix order (通常为3)
     * @param outTileWidth 每个tile在Allsky中的宽度（默认64，即512的1/8下采样）
     * @return 成功返回true
     */
    static bool generateAllskyFits(
        const std::string& hipsDir,
        int order = 3,
        int outTileWidth = 64
    );
    
private:
    /**
     * 对单个tile进行下采样
     */
    static void downsampleTile(
        const std::vector<float>& srcPixels, int srcWidth, int srcHeight,
        std::vector<float>& dstPixels, int dstWidth, int dstHeight,
        double blank
    );
};

#endif // ALLSKY_GENERATOR_H
