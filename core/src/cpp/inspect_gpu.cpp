/**
 * Inspect GPU tile values
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include "fits_io.h"

const int TILE_SIZE = 512;

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    FitsData javaTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits");
    FitsData gpuTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0/Npix499.fits");
    
    // Value distribution
    std::map<int, int> javaHist, gpuHist;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        int jKey = (int)(javaTile.pixels[i] * 10);
        int gKey = (int)(gpuTile.pixels[i] * 10);
        javaHist[jKey]++;
        gpuHist[gKey]++;
    }
    
    std::cout << "=== Java value distribution (binned at 0.1) ===" << std::endl;
    for (auto& p : javaHist) {
        if (p.second > 100) {
            std::cout << "  " << (p.first/10.0) << ": " << p.second << " pixels" << std::endl;
        }
    }
    
    std::cout << "\n=== GPU value distribution (binned at 0.1) ===" << std::endl;
    for (auto& p : gpuHist) {
        if (p.second > 100) {
            std::cout << "  " << (p.first/10.0) << ": " << p.second << " pixels" << std::endl;
        }
    }
    
    // Sample some GPU non-blank values
    std::cout << "\n=== Sample GPU non-blank values ===" << std::endl;
    int count = 0;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE && count < 20; i++) {
        float gv = gpuTile.pixels[i];
        if (!std::isnan(gv) && gv != -1.0f && gv != -0.5f) {
            int x = i % TILE_SIZE;
            int y = i / TILE_SIZE;
            std::cout << "GPU[" << x << "," << y << "] linear=" << i << ": " << gv << std::endl;
            count++;
        }
    }
    
    return 0;
}
