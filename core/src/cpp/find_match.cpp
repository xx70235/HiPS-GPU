/**
 * Find matching values between Java and GPU tiles
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include "fits_io.h"

const int TILE_SIZE = 512;

int mortonEncode(int x, int y) {
    int result = 0;
    for (int i = 0; i < 9; i++) {
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return result;
}

void mortonDecode(int morton, int& x, int& y) {
    x = 0;
    y = 0;
    for (int i = 0; i < 9; i++) {
        x |= ((morton >> (2*i)) & 1) << i;
        y |= ((morton >> (2*i + 1)) & 1) << i;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(8);
    
    FitsData javaTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits");
    FitsData gpuTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0/Npix499.fits");
    
    // Build GPU value -> position map
    std::map<int, int> gpuValueToIdx;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        float gv = gpuTile.pixels[i];
        if (!std::isnan(gv) && gv != -1.0f && gv > -0.9f) {  // Unique-ish values
            int key = (int)(gv * 100000);
            gpuValueToIdx[key] = i;
        }
    }
    
    // For each Java non-blank position, find matching GPU position
    std::cout << "Finding matching values..." << std::endl;
    int foundCount = 0;
    for (int jy = 0; jy < 100 && foundCount < 30; jy++) {
        for (int jx = 0; jx < 100 && foundCount < 30; jx++) {
            int jLinear = jy * TILE_SIZE + jx;
            float jVal = javaTile.pixels[jLinear];
            
            if (std::isnan(jVal) || jVal == -1.0f || jVal <= -0.9f) continue;
            
            int key = (int)(jVal * 100000);
            auto it = gpuValueToIdx.find(key);
            if (it != gpuValueToIdx.end()) {
                int gLinear = it->second;
                int gx = gLinear % TILE_SIZE;
                int gy = gLinear / TILE_SIZE;
                
                int jMorton = mortonEncode(jx, jy);
                int gMorton = mortonEncode(gx, gy);
                
                // Also check: what if gLinear is the morton of jx,jy?
                int expectedGLinear = mortonEncode(jx, jy);
                
                std::cout << "Java(" << jx << "," << jy << ") linear=" << jLinear 
                          << " | GPU(" << gx << "," << gy << ") linear=" << gLinear
                          << " | jMorton=" << jMorton << " gMorton=" << gMorton
                          << " | value=" << jVal;
                
                if (gLinear == jLinear) std::cout << " [SAME LINEAR]";
                else if (gLinear == jMorton) std::cout << " [GPU linear = Java morton]";
                else if (jLinear == gMorton) std::cout << " [Java linear = GPU morton]";
                
                std::cout << std::endl;
                foundCount++;
            }
        }
    }
    
    return 0;
}
