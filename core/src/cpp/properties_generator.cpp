/**
 * HiPS properties元数据文件生成器实现
 */

#include "properties_generator.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cmath>

/**
 * 生成properties文件
 */
bool PropertiesGenerator::generateProperties(
    const std::string& hipsDir,
    const std::string& title,
    int order,
    int tileWidth,
    int bitpix,
    int tilesCount,
    double pixelCutMin,
    double pixelCutMax
) {
    std::string propertiesPath = hipsDir + "/properties";
    
    std::ofstream outFile(propertiesPath);
    if (!outFile) {
        return false;
    }
    
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm* gm_time = std::gmtime(&time_t_now);
    
    std::stringstream dateStream;
    dateStream << std::put_time(gm_time, "%Y-%m-%dT%H:%MZ");
    std::string releaseDate = dateStream.str();
    
    // 计算像素比例（度/像素）
    // Order 3, tileWidth 512: 像素大小约为 180/(4*8*512) ≈ 0.01099度
    double hipsPixelScale = 180.0 / (4.0 * (1 << order) * tileWidth);
    
    // 计算MOC覆盖率估计
    int totalTiles = 12 * (1 << order) * (1 << order);
    double mocSkyFraction = (double)tilesCount / totalTiles;
    
    // 写入properties
    outFile << "creator_did          = ivo://CUDA/P/hips" << std::endl;
    outFile << "#hips_creator        = HiPS creator (institute or person)" << std::endl;
    outFile << "#hips_copyright      = Copyright mention of the HiPS" << std::endl;
    outFile << "obs_title            = " << title << std::endl;
    outFile << "#obs_collection      = Dataset collection name" << std::endl;
    outFile << "#obs_description     = Dataset text description" << std::endl;
    outFile << "#obs_ack             = Acknowledgement mention" << std::endl;
    outFile << "#prov_progenitor     = Provenance of the original data (free text)" << std::endl;
    outFile << "#bib_reference       = Bibcode for bibliographic reference" << std::endl;
    outFile << "#bib_reference_url   = URL to bibliographic reference" << std::endl;
    outFile << "#obs_copyright       = Copyright mention of the original data" << std::endl;
    outFile << "#obs_copyright_url   = URL to copyright page of the original data" << std::endl;
    outFile << "#t_min               = Start time in MJD" << std::endl;
    outFile << "#t_max               = Stop time in MJD" << std::endl;
    outFile << "#obs_regime          = Waveband keyword" << std::endl;
    outFile << "#em_min              = Start in spectral coordinates in meters" << std::endl;
    outFile << "#em_max              = Stop in spectral coordinates in meters" << std::endl;
    outFile << "hips_builder         = HipsGen-CUDA/1.0" << std::endl;
    outFile << "hips_version         = 1.4" << std::endl;
    outFile << "hips_frame           = equatorial" << std::endl;
    outFile << "hips_order           = " << order << std::endl;
    outFile << "hips_order_min       = 0" << std::endl;
    outFile << "hips_tile_width      = " << tileWidth << std::endl;
    outFile << "#hips_service_url    = ex: http://yourHipsServer/hips" << std::endl;
    outFile << "hips_status          = public master clonableOnce" << std::endl;
    outFile << std::fixed << std::setprecision(5);
    outFile << "hips_pixel_scale     = " << hipsPixelScale << std::endl;
    outFile << "dataproduct_type     = image" << std::endl;
    outFile << "hips_pixel_bitpix    = " << bitpix << std::endl;
    outFile << "data_pixel_bitpix    = " << bitpix << std::endl;
    outFile << "hips_sampling        = bilinear" << std::endl;
    outFile << "hips_overlay         = overlayMean mergeOverwriteTile treeMean" << std::endl;
    outFile << "hips_release_date    = " << releaseDate << std::endl;
    outFile << "hips_tile_format     = fits" << std::endl;
    outFile << std::scientific << std::setprecision(4);
    outFile << "hips_pixel_cut       = " << pixelCutMin << " " << pixelCutMax << std::endl;
    outFile << std::fixed << std::setprecision(4);
    outFile << "moc_sky_fraction     = " << mocSkyFraction << std::endl;
    outFile << "hipsgen_date         = " << releaseDate << std::endl;
    
    outFile.close();
    
    return true;
}
