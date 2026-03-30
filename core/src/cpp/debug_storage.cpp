/**
 * Debug: verify storage order
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include "fits_io.h"

const int TILE_SIZE = 512;

// Morton encoding
int xy2hpx(int x, int y) {
    int result = 0;
    for (int i = 0; i < 9; i++) {
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return result;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    // Load tiles
    FitsData javaTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits");
    FitsData gpuTile = FitsReader::readFitsFile(
        "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output_gpu/Norder3/Dir0/Npix499.fits");
    
    // Test: if GPU stores at linear position what it computed for morton HEALPix
    // and Java stores at linear position what it computed for that same HEALPix
    // then they should match at the SAME linear position (both should be computing
    // for the same HEALPix given the same linear position)
    
    // The code: for linearIdx i, compute morton = xy2hpx[i], sample HEALPix at base+morton, store at results[i]
    // So results[i] contains the sample for HEALPix index = base + xy2hpx[i]
    
    // When writing: tile.setPixel(x, y, results[y*512+x])
    // So FITS pixel at (x,y) = results[y*512+x] = sample for HEALPix base + xy2hpx[y*512+x]
    
    // This means: FITS pixel (x,y) has the value for HEALPix = base + morton(x,y)
    // Which is CORRECT per HiPS spec!
    
    // BUT if Java also does this correctly, why mismatch?
    
    // Let me test exact match at same position
    std::cout << "=== Direct comparison at same linear position ===" << std::endl;
    int exactMatch = 0, exactMismatch = 0;
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        float jv = javaTile.pixels[i];
        float gv = gpuTile.pixels[i];
        
        if (std::isnan(jv) || std::isnan(gv)) continue;
        if (jv == -1.0f && gv == -1.0f) continue;  // Skip blank
        
        if (std::abs(jv - gv) < 0.001) {
            exactMatch++;
        } else {
            exactMismatch++;
            if (exactMismatch <= 5) {
                int x = i % TILE_SIZE;
                int y = i / TILE_SIZE;
                std::cout << "Mismatch at linear " << i << " (x=" << x << ",y=" << y << "): Java=" << jv << " GPU=" << gv << std::endl;
            }
        }
    }
    std::cout << "Exact position match: " << exactMatch << " mismatch: " << exactMismatch << std::endl;
    
    // Test: if we read GPU at morton position for Java's linear position
    std::cout << "\n=== Test: Java[linear] vs GPU[morton of linear's (x,y)] ===" << std::endl;
    int mortonMatch = 0, mortonMismatch = 0;
    for (int linearIdx = 0; linearIdx < TILE_SIZE * TILE_SIZE; linearIdx++) {
        int x = linearIdx % TILE_SIZE;
        int y = linearIdx / TILE_SIZE;
        int mortonIdx = xy2hpx(x, y);
        
        float jv = javaTile.pixels[linearIdx];
        float gv = gpuTile.pixels[mortonIdx];
        
        if (std::isnan(jv) || std::isnan(gv)) continue;
        if (jv == -1.0f) continue;
        
        if (std::abs(jv - gv) < 0.001) {
            mortonMatch++;
        } else {
            mortonMismatch++;
        }
    }
    std::cout << "Java[linear] vs GPU[morton]: match=" << mortonMatch << " mismatch=" << mortonMismatch << std::endl;
    
    return 0;
}
