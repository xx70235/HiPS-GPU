#include <iostream>
#include <vector>

void createXY2HPXMapping(int tileOrder, std::vector<int>& xy2hpx) {
    int size = 1 << tileOrder;
    xy2hpx.resize(size * size);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int idx = y * size + x;
            int hpx = 0;
            for (int i = 0; i < tileOrder; i++) {
                int bitX = (x >> i) & 1;
                int bitY = (y >> i) & 1;
                hpx |= (bitX << (2 * i)) | (bitY << (2 * i + 1));
            }
            xy2hpx[idx] = hpx;
        }
    }
}

int main() {
    int tileOrder = 9;  // 512 = 2^9
    std::vector<int> xy2hpx;
    createXY2HPXMapping(tileOrder, xy2hpx);
    
    // 检查几个点
    std::cout << "Testing Morton encoding for " << (1<<tileOrder) << "x" << (1<<tileOrder) << " tile:" << std::endl;
    
    // (0,0) -> Morton应该是0
    std::cout << "(0,0) -> Morton = " << xy2hpx[0*512+0] << " (expected 0)" << std::endl;
    // (1,0) -> Morton应该是1
    std::cout << "(1,0) -> Morton = " << xy2hpx[0*512+1] << " (expected 1)" << std::endl;
    // (0,1) -> Morton应该是2
    std::cout << "(0,1) -> Morton = " << xy2hpx[1*512+0] << " (expected 2)" << std::endl;
    // (1,1) -> Morton应该是3
    std::cout << "(1,1) -> Morton = " << xy2hpx[1*512+1] << " (expected 3)" << std::endl;
    
    // (256,256) 中心点
    int centerIdx = 256 * 512 + 256;
    std::cout << "(256,256) -> Morton = " << xy2hpx[centerIdx] << std::endl;
    
    return 0;
}
