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
    
    // Analyze mismatches
    int match = 0, mismatch = 0;
    int javaNegOne = 0, gpuNegOne = 0;
    int javaNegOneGpuValid = 0, gpuNegOneJavaValid = 0;
    double sumDiff = 0;
    
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        float jv = javaTile.pixels[i];
        float gv = gpuTile.pixels[i];
        
        if (std::isnan(jv) || std::isnan(gv)) continue;
        
        bool jIsBlank = (jv == -1.0f);
        bool gIsBlank = (gv == -1.0f);
        
        if (jIsBlank) javaNegOne++;
        if (gIsBlank) gpuNegOne++;
        
        if (jIsBlank && !gIsBlank) javaNegOneGpuValid++;
        if (!jIsBlank && gIsBlank) gpuNegOneJavaValid++;
        
        if (!jIsBlank && !gIsBlank) {
            double diff = std::abs(jv - gv);
            sumDiff += diff;
            if (diff < 0.001) {
                match++;
            } else {
                mismatch++;
                if (mismatch <= 10) {
                    int x = i % TILE_SIZE;
                    int y = i / TILE_SIZE;
                    std::cout << "Mismatch (" << x << "," << y << "): Java=" << jv << " GPU=" << gv << " diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "\nSummary (non-blank):" << std::endl;
    std::cout << "  Match: " << match << std::endl;
    std::cout << "  Mismatch: " << mismatch << std::endl;
    std::cout << "  Avg diff: " << sumDiff / (match + mismatch) << std::endl;
    
    std::cout << "\nBlank analysis:" << std::endl;
    std::cout << "  Java -1 count: " << javaNegOne << std::endl;
    std::cout << "  GPU -1 count: " << gpuNegOne << std::endl;
    std::cout << "  Java -1 but GPU valid: " << javaNegOneGpuValid << std::endl;
    std::cout << "  GPU -1 but Java valid: " << gpuNegOneJavaValid << std::endl;
    
    return 0;
}
