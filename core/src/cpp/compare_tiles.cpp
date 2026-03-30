#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

struct TileData {
    std::vector<float> pixels;
    int width, height;
};

bool readFitsTile(const std::string& filename, TileData& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    
    char header[2880];
    file.read(header, 2880);
    
    data.width = 512;
    data.height = 512;
    int bitpix = -32;
    
    for (int i = 0; i < 2880; i += 80) {
        std::string card(header + i, 80);
        if (card.substr(0, 6) == "NAXIS1") {
            data.width = std::stoi(card.substr(10, 20));
        } else if (card.substr(0, 6) == "NAXIS2") {
            data.height = std::stoi(card.substr(10, 20));
        }
    }
    
    int npix = data.width * data.height;
    data.pixels.resize(npix);
    
    std::vector<char> buffer(npix * 4);
    file.read(buffer.data(), npix * 4);
    
    // FITS uses big-endian
    for (int i = 0; i < npix; i++) {
        char bytes[4];
        bytes[0] = buffer[i*4 + 3];
        bytes[1] = buffer[i*4 + 2];
        bytes[2] = buffer[i*4 + 1];
        bytes[3] = buffer[i*4 + 0];
        memcpy(&data.pixels[i], bytes, 4);
    }
    
    return true;
}

void compareTiles(const std::string& javaFile, const std::string& gpuFile) {
    TileData java, gpu;
    if (!readFitsTile(javaFile, java)) {
        std::cerr << "Failed to read: " << javaFile << std::endl;
        return;
    }
    if (!readFitsTile(gpuFile, gpu)) {
        std::cerr << "Failed to read: " << gpuFile << std::endl;
        return;
    }
    
    int javaValid = 0, gpuValid = 0, bothValid = 0;
    double sumDiff = 0, maxDiff = 0;
    double sumJava = 0, sumGpu = 0, sumJJ = 0, sumGG = 0, sumJG = 0;
    
    for (size_t i = 0; i < java.pixels.size(); i++) {
        bool jv = !std::isnan(java.pixels[i]);
        bool gv = !std::isnan(gpu.pixels[i]);
        if (jv) javaValid++;
        if (gv) gpuValid++;
        if (jv && gv) {
            bothValid++;
            double diff = std::abs(java.pixels[i] - gpu.pixels[i]);
            sumDiff += diff;
            if (diff > maxDiff) maxDiff = diff;
            
            sumJava += java.pixels[i];
            sumGpu += gpu.pixels[i];
            sumJJ += java.pixels[i] * java.pixels[i];
            sumGG += gpu.pixels[i] * gpu.pixels[i];
            sumJG += java.pixels[i] * gpu.pixels[i];
        }
    }
    
    double corr = 0;
    if (bothValid > 1) {
        double meanJ = sumJava / bothValid;
        double meanG = sumGpu / bothValid;
        double varJ = sumJJ / bothValid - meanJ * meanJ;
        double varG = sumGG / bothValid - meanG * meanG;
        double cov = sumJG / bothValid - meanJ * meanG;
        if (varJ > 0 && varG > 0) {
            corr = cov / sqrt(varJ * varG);
        }
    }
    
    std::cout << "Valid: java=" << javaValid << ", gpu=" << gpuValid 
              << ", both=" << bothValid << std::endl;
    std::cout << "Max diff: " << maxDiff << ", mean diff: " << (bothValid > 0 ? sumDiff/bothValid : 0) << std::endl;
    std::cout << "Correlation: " << corr << std::endl;
}

int main(int argc, char* argv[]) {
    std::string javaBase = "data/hips_output_java";
    std::string gpuBase = "hips_output_gpu";
    
    // 测试几个特定tiles
    std::vector<std::string> testTiles = {
        "Norder3/Dir0/Npix499.fits",
        "Norder3/Dir0/Npix498.fits",
        "Norder3/Dir0/Npix500.fits"
    };
    
    int goodCount = 0, badCount = 0;
    double minCorr = 999;
    std::string worstTile;
    
    for (const auto& tile : testTiles) {
        std::string javaFile = javaBase + "/" + tile;
        std::string gpuFile = gpuBase + "/" + tile;
        
        TileData java, gpu;
        if (!readFitsTile(javaFile, java) || !readFitsTile(gpuFile, gpu)) {
            continue;
        }
        
        std::cout << "\n=== " << tile << " ===" << std::endl;
        compareTiles(javaFile, gpuFile);
    }
    
    return 0;
}
