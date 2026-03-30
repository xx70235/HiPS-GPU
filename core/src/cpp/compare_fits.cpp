#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fitsio.h>

struct FitsData {
    std::vector<float> pixels;
    int width, height;
};

FitsData readFitsFile(const std::string& filename) {
    FitsData data;
    fitsfile* fptr;
    int status = 0;
    
    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        std::cerr << "Error opening " << filename << std::endl;
        return data;
    }
    
    long naxes[2];
    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    fits_get_img_size(fptr, 2, naxes, &status);
    
    data.width = naxes[0];
    data.height = naxes[1];
    data.pixels.resize(data.width * data.height);
    
    long fpixel[2] = {1, 1};
    fits_read_pix(fptr, TFLOAT, fpixel, data.width * data.height, nullptr, data.pixels.data(), nullptr, &status);
    
    fits_close_file(fptr, &status);
    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: compare_fits <java_file> <cuda_file>" << std::endl;
        return 1;
    }
    
    std::string javaFile = argv[1];
    std::string cudaFile = argv[2];
    
    FitsData java = readFitsFile(javaFile);
    FitsData cuda = readFitsFile(cudaFile);
    
    std::cout << "Java file: " << javaFile << std::endl;
    std::cout << "CUDA file: " << cudaFile << std::endl;
    std::cout << "Dimensions: " << java.width << "x" << java.height << std::endl;
    
    // Count valid pixels
    int javaValid = 0, cudaValid = 0;
    for (float v : java.pixels) if (!std::isnan(v)) javaValid++;
    for (float v : cuda.pixels) if (!std::isnan(v)) cudaValid++;
    
    std::cout << "\nValid pixels: Java=" << javaValid << ", CUDA=" << cudaValid << std::endl;
    
    // Compare where both are valid
    int bothValid = 0;
    double maxDiff = 0, sumDiff = 0;
    int closeMatches = 0;
    
    std::vector<std::pair<int, double>> largeDiffs;  // position, diff
    
    for (size_t i = 0; i < java.pixels.size(); i++) {
        float jv = java.pixels[i];
        float cv = cuda.pixels[i];
        
        if (!std::isnan(jv) && !std::isnan(cv)) {
            bothValid++;
            double diff = std::abs(jv - cv);
            maxDiff = std::max(maxDiff, diff);
            sumDiff += diff;
            
            // Close match (relative tolerance 1e-4)
            double tol = std::abs(jv) * 1e-4 + 1e-8;
            if (diff <= tol) closeMatches++;
            
            if (diff > 1e-6 && largeDiffs.size() < 10) {
                largeDiffs.push_back({i, diff});
            }
        }
    }
    
    std::cout << "\nBoth valid at same position: " << bothValid << std::endl;
    
    if (bothValid > 0) {
        std::cout << "Max absolute difference: " << maxDiff << std::endl;
        std::cout << "Mean absolute difference: " << (sumDiff / bothValid) << std::endl;
        std::cout << "Close matches: " << closeMatches << "/" << bothValid 
                  << " (" << (100.0 * closeMatches / bothValid) << "%)" << std::endl;
    }
    
    // Show some large differences
    if (!largeDiffs.empty()) {
        std::cout << "\nSample large differences:" << std::endl;
        for (auto& p : largeDiffs) {
            int idx = p.first;
            int x = idx % java.width;
            int y = idx / java.width;
            std::cout << "  (x=" << x << ", y=" << y << "): Java=" << java.pixels[idx] 
                      << ", CUDA=" << cuda.pixels[idx] << ", diff=" << p.second << std::endl;
        }
    }
    
    // Count Java-only and CUDA-only
    int javaOnly = 0, cudaOnly = 0;
    for (size_t i = 0; i < java.pixels.size(); i++) {
        bool jValid = !std::isnan(java.pixels[i]);
        bool cValid = !std::isnan(cuda.pixels[i]);
        if (jValid && !cValid) javaOnly++;
        if (!jValid && cValid) cudaOnly++;
    }
    std::cout << "\nJava only: " << javaOnly << ", CUDA only: " << cudaOnly << std::endl;
    
    return 0;
}
