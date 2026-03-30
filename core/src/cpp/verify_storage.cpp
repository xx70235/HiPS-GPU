/**
 * Verify exact storage pattern
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include "fits_io.h"

const int TILE_SIZE = 512;

// Morton encoding
int mortonEncode(int x, int y) {
    int result = 0;
    for (int i = 0; i < 9; i++) {
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return result;
}

int main() {
    std::cout << std::fixed << std::setprecision(8);
    
    FitsData javaTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits");
    FitsData gpuTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0/Npix499.fits");
    
    // Find a unique non-blank value in Java tile
    std::cout << "Looking for unique values to trace..." << std::endl;
    
    for (int y = 0; y < 50; y++) {
        for (int x = 0; x < 50; x++) {
            int linearIdx = y * TILE_SIZE + x;
            float jVal = javaTile.pixels[linearIdx];
            
            if (std::isnan(jVal) || jVal == -1.0f || jVal == 0.0f) continue;
            
            // Count occurrences in Java tile
            int jCount = 0;
            for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
                if (std::abs(javaTile.pixels[i] - jVal) < 0.0001f) jCount++;
            }
            
            if (jCount > 5) continue;  // Skip non-unique values
            
            // Find this value in GPU tile
            int mortonIdx = mortonEncode(x, y);
            float gpuAtLinear = gpuTile.pixels[linearIdx];
            float gpuAtMorton = gpuTile.pixels[mortonIdx];
            
            std::cout << "Java(" << x << "," << y << ") linear=" << linearIdx 
                      << " morton=" << mortonIdx << " value=" << jVal
                      << " | GPU@linear=" << gpuAtLinear 
                      << " GPU@morton=" << gpuAtMorton;
            
            if (std::abs(gpuAtLinear - jVal) < 0.001f) {
                std::cout << " [MATCH at linear]";
            } else if (std::abs(gpuAtMorton - jVal) < 0.001f) {
                std::cout << " [MATCH at morton]";
            }
            std::cout << std::endl;
            
            if (jCount == 1) break;  // Found unique value
        }
    }
    
    return 0;
}
