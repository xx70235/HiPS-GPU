/**
 * Detailed comparison: GPU output vs Java output
 * Focus on understanding the pixel mapping
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "fits_io.h"

const int TILE_SIZE = 512;

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    // Load Java tile
    std::string javaTilePath = "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits";
    FitsData javaTile = FitsReader::readFitsFile(javaTilePath);
    if (!javaTile.isValid) {
        std::cerr << "Failed to load Java tile" << std::endl;
        return 1;
    }
    
    // Load GPU tile
    std::string gpuTilePath = "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0/Npix499.fits";
    FitsData gpuTile = FitsReader::readFitsFile(gpuTilePath);
    if (!gpuTile.isValid) {
        std::cerr << "Failed to load GPU tile" << std::endl;
        return 1;
    }
    
    std::cout << "=== Tile statistics ===" << std::endl;
    
    // Count valid pixels
    int javaValid = 0, gpuValid = 0, bothValid = 0;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        bool jv = !std::isnan(javaTile.pixels[i]);
        bool gv = !std::isnan(gpuTile.pixels[i]);
        if (jv) javaValid++;
        if (gv) gpuValid++;
        if (jv && gv) bothValid++;
    }
    std::cout << "Java valid pixels: " << javaValid << std::endl;
    std::cout << "GPU valid pixels: " << gpuValid << std::endl;
    std::cout << "Both valid pixels: " << bothValid << std::endl;
    
    // Sample specific pixels to see the pattern
    std::cout << "\n=== Sample pixels (position, java_val, gpu_val) ===" << std::endl;
    for (int y = 0; y < 30; y += 5) {
        for (int x = 0; x < 30; x += 5) {
            int idx = y * TILE_SIZE + x;
            float jv = javaTile.pixels[idx];
            float gv = gpuTile.pixels[idx];
            if (!std::isnan(jv) || !std::isnan(gv)) {
                std::cout << "(" << x << "," << y << "): Java=" << jv << " GPU=" << gv << std::endl;
            }
        }
    }
    
    // Look for pattern: for each Java valid pixel, where is a similar GPU value?
    std::cout << "\n=== Pattern search: Java pixel value locations vs GPU ===" << std::endl;
    for (int jy = 0; jy < 20; jy++) {
        for (int jx = 0; jx < 20; jx++) {
            int jIdx = jy * TILE_SIZE + jx;
            float jVal = javaTile.pixels[jIdx];
            if (std::isnan(jVal)) continue;
            
            // Search for this value in GPU tile
            int bestGx = -1, bestGy = -1;
            float bestDiff = 1e30;
            for (int gy = 0; gy < TILE_SIZE; gy++) {
                for (int gx = 0; gx < TILE_SIZE; gx++) {
                    int gIdx = gy * TILE_SIZE + gx;
                    float gVal = gpuTile.pixels[gIdx];
                    if (std::isnan(gVal)) continue;
                    float diff = std::abs(jVal - gVal);
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        bestGx = gx;
                        bestGy = gy;
                    }
                }
            }
            
            if (bestDiff < 0.001) {
                std::cout << "Java(" << jx << "," << jy << ")=" << jVal 
                          << " -> GPU(" << bestGx << "," << bestGy << ") diff=" << bestDiff << std::endl;
            }
        }
    }
    
    return 0;
}
