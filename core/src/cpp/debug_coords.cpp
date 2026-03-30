/**
 * Debug tool: Compare coordinate calculations for specific tiles
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/pointing.h>

// Correct recursive fillUp (matches Java Context.fillUp exactly)
static void fillUpRecursive(std::vector<int>& npix, int nsize, std::vector<int>* pos) {
    int size = nsize * nsize;
    std::vector<std::vector<int>> fils(4);
    for (int i = 0; i < 4; i++) fils[i].resize(size / 4);
    std::vector<int> nb(4, 0);
    
    for (int i = 0; i < size; i++) {
        int dg = (i % nsize) < (nsize / 2) ? 0 : 1;
        int bh = i < (size / 2) ? 1 : 0;
        int quad = (dg << 1) | bh;
        
        int j = (pos == nullptr) ? i : (*pos)[i];
        npix[j] = (npix[j] << 2) | quad;
        fils[quad][nb[quad]++] = j;
    }
    
    if (size > 4) {
        for (int i = 0; i < 4; i++) {
            fillUpRecursive(npix, nsize / 2, &fils[i]);
        }
    }
}

std::vector<int> createXY2HPX(int tileWidth) {
    std::vector<int> xy2hpx(tileWidth * tileWidth, 0);
    fillUpRecursive(xy2hpx, tileWidth, nullptr);
    return xy2hpx;
}

int main() {
    int tileWidth = 512;
    int tileOrder = 9;  // log2(512)
    
    // Test cases: order, npix pairs
    std::vector<std::pair<int, long>> testCases = {
        {0, 3},   // Norder0/Npix3
        {0, 4},   // Norder0/Npix4 (reportedly good match)
        {1, 15},  // Norder1/Npix15
        {2, 62}   // Norder2/Npix62
    };
    
    std::vector<int> xy2hpx = createXY2HPX(tileWidth);
    
    std::cout << std::fixed << std::setprecision(6);
    
    for (auto& tc : testCases) {
        int order = tc.first;
        long npix = tc.second;
        
        int pixelOrder = order + tileOrder;
        long nside = 1L << pixelOrder;
        T_Healpix_Base<int64> hpx(nside, NEST, SET_NSIDE);
        
        // baseIdx calculation
        long baseIdx = npix << (2 * tileOrder);
        
        std::cout << "\n=== Order=" << order << ", Npix=" << npix << " ===" << std::endl;
        std::cout << "pixelOrder=" << pixelOrder << ", nside=" << nside << std::endl;
        std::cout << "baseIdx=" << baseIdx << std::endl;
        
        // Check a few sample pixels
        std::vector<std::pair<int, int>> samples = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {255, 255}, {256, 256},
            {511, 0}, {0, 511}, {511, 511}
        };
        
        std::cout << "\nSample pixel coordinates (x, y) -> (ra, dec):" << std::endl;
        std::cout << "  x    y     xyIdx    hpxOff   healpixIdx          ra           dec" << std::endl;
        
        for (auto& s : samples) {
            int x = s.first;
            int y = s.second;
            
            // Java uses: index = min + context.xy2hpx[y * out.width + x]
            int xyIdx = y * tileWidth + x;
            int hpxOffset = xy2hpx[xyIdx];
            long healpixIdx = baseIdx + hpxOffset;
            
            pointing ptg = hpx.pix2ang(healpixIdx);
            double dec = 90.0 - ptg.theta * 180.0 / M_PI;
            double ra = ptg.phi * 180.0 / M_PI;
            
            std::cout << std::setw(4) << x << " " 
                      << std::setw(4) << y << " "
                      << std::setw(8) << xyIdx << " "
                      << std::setw(8) << hpxOffset << " "
                      << std::setw(14) << healpixIdx << " "
                      << std::setw(12) << ra << " "
                      << std::setw(12) << dec << std::endl;
        }
    }
    
    std::cout << "\n\n=== xy2hpx mapping verification (using recursive method) ===" << std::endl;
    std::cout << "First 20 entries of xy2hpx:" << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << "  xy2hpx[" << i << "] = " << xy2hpx[i] << std::endl;
    }
    
    return 0;
}
