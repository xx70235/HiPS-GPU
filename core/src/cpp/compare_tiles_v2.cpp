/**
 * Compare tiles - exclude blank values
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include "fits_io.h"

namespace fs = std::filesystem;
const int TILE_SIZE = 512;

double computeCorrelation(const std::vector<float>& java, const std::vector<float>& gpu) {
    // Exclude blank values
    std::vector<double> jVals, gVals;
    for (size_t i = 0; i < java.size(); i++) {
        float jv = java[i];
        float gv = gpu[i];
        if (!std::isnan(jv) && !std::isnan(gv) && jv != -1.0f && gv != -1.0f) {
            jVals.push_back(jv);
            gVals.push_back(gv);
        }
    }
    
    if (jVals.size() < 10) return 0;
    
    double sumJ = 0, sumG = 0;
    for (size_t i = 0; i < jVals.size(); i++) {
        sumJ += jVals[i];
        sumG += gVals[i];
    }
    double meanJ = sumJ / jVals.size();
    double meanG = sumG / gVals.size();
    
    double cov = 0, varJ = 0, varG = 0;
    for (size_t i = 0; i < jVals.size(); i++) {
        double dJ = jVals[i] - meanJ;
        double dG = gVals[i] - meanG;
        cov += dJ * dG;
        varJ += dJ * dJ;
        varG += dG * dG;
    }
    
    if (varJ == 0 || varG == 0) return 0;
    return cov / std::sqrt(varJ * varG);
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    std::string javaDir = "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0";
    std::string gpuDir = "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0";
    
    // Tiles to compare
    int testTiles[] = {499, 498, 500, 502, 503};
    
    for (int npix : testTiles) {
        std::string javaPath = javaDir + "/Npix" + std::to_string(npix) + ".fits";
        std::string gpuPath = gpuDir + "/Npix" + std::to_string(npix) + ".fits";
        
        if (!fs::exists(javaPath) || !fs::exists(gpuPath)) continue;
        
        FitsData java = FitsReader::readFitsFile(javaPath);
        FitsData gpu = FitsReader::readFitsFile(gpuPath);
        
        if (!java.isValid || !gpu.isValid) continue;
        
        // Count valid (non-blank) pixels
        int jValid = 0, gValid = 0, bothValid = 0;
        double maxDiff = 0, sumDiff = 0;
        int diffCount = 0;
        
        for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
            float jv = java.pixels[i];
            float gv = gpu.pixels[i];
            
            if (!std::isnan(jv) && jv != -1.0f) jValid++;
            if (!std::isnan(gv) && gv != -1.0f) gValid++;
            
            if (!std::isnan(jv) && jv != -1.0f && !std::isnan(gv) && gv != -1.0f) {
                bothValid++;
                double diff = std::abs(jv - gv);
                maxDiff = std::max(maxDiff, diff);
                sumDiff += diff;
                diffCount++;
            }
        }
        
        double corr = computeCorrelation(java.pixels, gpu.pixels);
        
        std::cout << "\n=== Npix" << npix << ".fits ===" << std::endl;
        std::cout << "Valid (non-blank): java=" << jValid << ", gpu=" << gValid << ", both=" << bothValid << std::endl;
        std::cout << "Max diff: " << maxDiff << ", mean diff: " << (diffCount > 0 ? sumDiff/diffCount : 0) << std::endl;
        std::cout << "Correlation (non-blank): " << corr << std::endl;
    }
    
    return 0;
}
