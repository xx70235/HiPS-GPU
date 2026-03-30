/**
 * Find unique values to trace the mapping
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <set>
#include "fits_io.h"

const int TILE_SIZE = 512;

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
    
    // Find unique Java values (precision to 1e-5)
    std::map<int, int> javaValueCount;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        float jv = javaTile.pixels[i];
        if (!std::isnan(jv) && jv != -1.0f) {
            int key = (int)(jv * 100000);
            javaValueCount[key]++;
        }
    }
    
    // Build GPU value -> position map
    std::map<int, int> gpuValueToIdx;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        float gv = gpuTile.pixels[i];
        if (!std::isnan(gv) && gv != -1.0f) {
            int key = (int)(gv * 100000);
            gpuValueToIdx[key] = i;
        }
    }
    
    std::cout << "Finding truly unique value matches..." << std::endl;
    
    int foundCount = 0;
    int linearMatch = 0, mortonMatch = 0, otherMatch = 0;
    
    for (int jy = 0; jy < TILE_SIZE && foundCount < 50; jy++) {
        for (int jx = 0; jx < TILE_SIZE && foundCount < 50; jx++) {
            int jLinear = jy * TILE_SIZE + jx;
            float jVal = javaTile.pixels[jLinear];
            
            if (std::isnan(jVal) || jVal == -1.0f) continue;
            
            int key = (int)(jVal * 100000);
            
            // Only consider unique values
            if (javaValueCount[key] != 1) continue;
            
            auto it = gpuValueToIdx.find(key);
            if (it != gpuValueToIdx.end()) {
                int gLinear = it->second;
                int gx = gLinear % TILE_SIZE;
                int gy = gLinear / TILE_SIZE;
                
                int jMorton = mortonEncode(jx, jy);
                
                std::cout << "Java(" << jx << "," << jy << ") linear=" << jLinear 
                          << " -> GPU(" << gx << "," << gy << ") linear=" << gLinear
                          << " | jMorton=" << jMorton 
                          << " | value=" << jVal;
                
                if (gLinear == jLinear) {
                    std::cout << " [SAME LINEAR]";
                    linearMatch++;
                } else if (gLinear == jMorton) {
                    std::cout << " [GPU linear = Java morton]";
                    mortonMatch++;
                } else {
                    std::cout << " [OTHER]";
                    otherMatch++;
                }
                
                std::cout << std::endl;
                foundCount++;
            }
        }
    }
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Linear matches: " << linearMatch << std::endl;
    std::cout << "  Morton matches (GPU@jMorton = Java@jLinear): " << mortonMatch << std::endl;
    std::cout << "  Other: " << otherMatch << std::endl;
    
    return 0;
}
