/**
 * Analyze the coordinate mapping between Java and GPU output
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>
#include "fits_io.h"

const int TILE_SIZE = 512;

// Morton encoding (xy to hpx index)
int xy2hpx(int x, int y) {
    int result = 0;
    for (int i = 0; i < 9; i++) {
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return result;
}

// Morton decoding (hpx to xy)
void hpx2xy(int hpx, int& x, int& y) {
    x = 0;
    y = 0;
    for (int i = 0; i < 9; i++) {
        x |= ((hpx >> (2*i)) & 1) << i;
        y |= ((hpx >> (2*i + 1)) & 1) << i;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    // Load tiles
    FitsData javaTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits");
    FitsData gpuTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0/Npix499.fits");
    
    if (!javaTile.isValid || !gpuTile.isValid) {
        std::cerr << "Failed to load tiles" << std::endl;
        return 1;
    }
    
    // Build a map of unique GPU values to their positions
    std::map<int, std::pair<int,int>> gpuValueToPos;  // Use scaled int as key
    for (int gy = 0; gy < TILE_SIZE; gy++) {
        for (int gx = 0; gx < TILE_SIZE; gx++) {
            float gVal = gpuTile.pixels[gy * TILE_SIZE + gx];
            if (!std::isnan(gVal) && gVal != -1.0f) {
                int key = (int)(gVal * 1000000);
                gpuValueToPos[key] = {gx, gy};
            }
        }
    }
    
    std::cout << "=== Mapping Analysis ===" << std::endl;
    std::cout << "Format: Java(jx,jy) -> GPU(gx,gy) | morton(jx,jy)=" << std::endl;
    
    int matchCount = 0;
    for (int jy = 0; jy < 100; jy++) {
        for (int jx = 0; jx < 100; jx++) {
            float jVal = javaTile.pixels[jy * TILE_SIZE + jx];
            if (std::isnan(jVal) || jVal == -1.0f) continue;
            
            int key = (int)(jVal * 1000000);
            if (gpuValueToPos.count(key)) {
                int gx = gpuValueToPos[key].first;
                int gy = gpuValueToPos[key].second;
                
                // Calculate morton indices
                int jMorton = xy2hpx(jx, jy);
                int gMorton = xy2hpx(gx, gy);
                
                // Also try reverse morton
                int jReverseMortonX, jReverseMortonY;
                hpx2xy(jy * TILE_SIZE + jx, jReverseMortonX, jReverseMortonY);
                
                std::cout << "Java(" << jx << "," << jy << ") -> GPU(" << gx << "," << gy << ")"
                          << " | jMorton=" << jMorton << " gMorton=" << gMorton 
                          << " | jLinear=" << (jy * TILE_SIZE + jx) 
                          << " gLinear=" << (gy * TILE_SIZE + gx) << std::endl;
                
                matchCount++;
                if (matchCount >= 30) break;
            }
        }
        if (matchCount >= 30) break;
    }
    
    // Check specific hypothesis: GPU stores at Morton index, Java stores at linear
    std::cout << "\n=== Hypothesis Test: GPU at morton(x,y), Java at linear(x,y) ===" << std::endl;
    int hypothesis1Match = 0, hypothesis1Mismatch = 0;
    for (int y = 0; y < 100; y++) {
        for (int x = 0; x < 100; x++) {
            int jIdx = y * TILE_SIZE + x;  // Java linear index
            int gIdx = xy2hpx(x, y);       // GPU morton index
            
            float jVal = javaTile.pixels[jIdx];
            float gVal = gpuTile.pixels[gIdx];
            
            if (std::isnan(jVal) || jVal == -1.0f) continue;
            if (std::isnan(gVal) || gVal == -1.0f) continue;
            
            if (std::abs(jVal - gVal) < 0.001) {
                hypothesis1Match++;
            } else {
                hypothesis1Mismatch++;
            }
        }
    }
    std::cout << "Hypothesis 1 (GPU uses morton): Match=" << hypothesis1Match 
              << " Mismatch=" << hypothesis1Mismatch << std::endl;
    
    // Check reverse: Java stores at Morton, GPU at linear
    std::cout << "\n=== Hypothesis Test: Java at morton(x,y), GPU at linear(x,y) ===" << std::endl;
    int hypothesis2Match = 0, hypothesis2Mismatch = 0;
    for (int y = 0; y < 100; y++) {
        for (int x = 0; x < 100; x++) {
            int jIdx = xy2hpx(x, y);       // Java morton index
            int gIdx = y * TILE_SIZE + x;  // GPU linear index
            
            float jVal = javaTile.pixels[jIdx];
            float gVal = gpuTile.pixels[gIdx];
            
            if (std::isnan(jVal) || jVal == -1.0f) continue;
            if (std::isnan(gVal) || gVal == -1.0f) continue;
            
            if (std::abs(jVal - gVal) < 0.001) {
                hypothesis2Match++;
            } else {
                hypothesis2Mismatch++;
            }
        }
    }
    std::cout << "Hypothesis 2 (Java uses morton): Match=" << hypothesis2Match 
              << " Mismatch=" << hypothesis2Mismatch << std::endl;
    
    return 0;
}
