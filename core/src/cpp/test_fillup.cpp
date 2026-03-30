#include <iostream>
#include <vector>

// Method 1: From debug_coords.cpp (direct calculation)
std::vector<int> fillUpMethod1(int tileWidth) {
    std::vector<int> xy2hpx(tileWidth * tileWidth);
    std::vector<int> hpx2xy(tileWidth * tileWidth);
    
    int quart = tileWidth / 2;
    for (int i = 0; i < tileWidth * tileWidth / 4; i++) {
        int x = 0, y = 0;
        int c = i;
        for (int k = 1; k < tileWidth; k *= 2) {
            int nx = c % 2;
            c /= 2;
            int ny = c % 2;
            c /= 2;
            x += nx * k;
            y += ny * k;
        }
        hpx2xy[i] = x * tileWidth + y;
        hpx2xy[i + 1 * tileWidth * tileWidth / 4] = (quart + y) * tileWidth + (quart - x - 1);
        hpx2xy[i + 2 * tileWidth * tileWidth / 4] = (tileWidth - x - 1) * tileWidth + (tileWidth - y - 1);
        hpx2xy[i + 3 * tileWidth * tileWidth / 4] = (quart - y - 1) * tileWidth + (quart + x);
    }
    
    for (int i = 0; i < tileWidth * tileWidth; i++) {
        xy2hpx[hpx2xy[i]] = i;
    }
    
    return xy2hpx;
}

// Method 2: From hips_tile_generator.cpp (recursive)
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

std::vector<int> fillUpMethod2(int tileWidth) {
    std::vector<int> xy2hpx(tileWidth * tileWidth, 0);
    fillUpRecursive(xy2hpx, tileWidth, nullptr);
    return xy2hpx;
}

int main() {
    int tileWidth = 512;
    
    std::vector<int> method1 = fillUpMethod1(tileWidth);
    std::vector<int> method2 = fillUpMethod2(tileWidth);
    
    std::cout << "Comparing two fillUp methods for tileWidth=" << tileWidth << std::endl;
    
    // Check first 20 entries
    std::cout << "\nFirst 20 entries:" << std::endl;
    std::cout << "Index    Method1    Method2    Match" << std::endl;
    int mismatches = 0;
    for (int i = 0; i < 20; i++) {
        bool match = (method1[i] == method2[i]);
        std::cout << i << "        " << method1[i] << "          " << method2[i] << "        " << (match ? "Yes" : "NO!") << std::endl;
        if (!match) mismatches++;
    }
    
    // Count total mismatches
    for (int i = 20; i < tileWidth * tileWidth; i++) {
        if (method1[i] != method2[i]) mismatches++;
    }
    
    std::cout << "\nTotal mismatches: " << mismatches << " out of " << (tileWidth * tileWidth) << std::endl;
    
    return 0;
}
