#include <iostream>
#include <vector>

// Inline Morton encoding
int mortonEncode(int x, int y, int bits) {
    int hpx = 0;
    for (int i = 0; i < bits; i++) {
        int bitX = (x >> i) & 1;
        int bitY = (y >> i) & 1;
        hpx |= (bitX << (2 * i)) | (bitY << (2 * i + 1));
    }
    return hpx;
}

void createXY2HPXMapping(int tileOrder, std::vector<int>& xy2hpx, std::vector<int>& hpx2xy) {
    int size = 1 << tileOrder;  // 512 for tileOrder=9
    xy2hpx.resize(size * size);
    hpx2xy.resize(size * size);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int idx = y * size + x;
            int hpx = mortonEncode(x, y, tileOrder);
            xy2hpx[idx] = hpx;
            hpx2xy[hpx] = idx;
        }
    }
}

int main() {
    std::vector<int> xy2hpx, hpx2xy;
    createXY2HPXMapping(9, xy2hpx, hpx2xy);
    
    std::cout << "Testing xy2hpx mapping:" << std::endl;
    
    // Linear index 1 -> (x=1, y=0) -> morton(1,0) = 1
    std::cout << "xy2hpx[1] = " << xy2hpx[1] << " (expected morton(1,0) = 1)" << std::endl;
    
    // Linear index 512 -> (x=0, y=1) -> morton(0,1) = 2
    std::cout << "xy2hpx[512] = " << xy2hpx[512] << " (expected morton(0,1) = 2)" << std::endl;
    
    // Linear index 513 -> (x=1, y=1) -> morton(1,1) = 3
    std::cout << "xy2hpx[513] = " << xy2hpx[513] << " (expected morton(1,1) = 3)" << std::endl;
    
    // Linear index 10 -> (x=10, y=0)
    // x=10: bits = 1010
    // morton: bit0(x)=0, bit1(y)=0, bit2(x)=1, bit3(y)=0, bit4(x)=0, bit5(y)=0, bit6(x)=1, bit7(y)=0
    // = 0b01000100 = 68
    std::cout << "xy2hpx[10] = " << xy2hpx[10] << " (expected morton(10,0) = 68)" << std::endl;
    
    // Direct test
    std::cout << "\nDirect morton(10,0) = " << mortonEncode(10, 0, 9) << std::endl;
    std::cout << "Direct morton(0,10) = " << mortonEncode(0, 10, 9) << std::endl;
    
    // Verify hpx2xy is the inverse
    std::cout << "\nVerify inverse:" << std::endl;
    std::cout << "hpx2xy[1] = " << hpx2xy[1] << " (expected 1)" << std::endl;
    std::cout << "hpx2xy[2] = " << hpx2xy[2] << " (expected 512)" << std::endl;
    std::cout << "hpx2xy[3] = " << hpx2xy[3] << " (expected 513)" << std::endl;
    std::cout << "hpx2xy[68] = " << hpx2xy[68] << " (expected 10)" << std::endl;
    
    return 0;
}
