/**
 * Debug: examine actual pixel values
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include "fits_io.h"
#include "coordinate_transform.h"
#include "healpix_util.h"

const int TILE_SIZE = 512;

int getMortonIndex(int x, int y) {
    int result = 0;
    for (int i = 0; i < 9; i++) {
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return result;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::string fitsPath = "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/data/subset/ep06800000874wxt13_50-100-flux.img";
    
    FitsData fitsData = FitsReader::readFitsFile(fitsPath);
    if (!fitsData.isValid) {
        std::cerr << "Failed to open FITS file" << std::endl;
        return 1;
    }
    
    std::cout << "=== Image Info ===" << std::endl;
    std::cout << "Size: " << fitsData.width << " x " << fitsData.height << std::endl;
    std::cout << "BLANK: " << fitsData.blank << std::endl;
    
    // Count values
    int nanCount = 0, negOneCount = 0, validCount = 0;
    float minVal = 1e30, maxVal = -1e30;
    for (const auto& v : fitsData.pixels) {
        if (std::isnan(v)) nanCount++;
        else if (v == -1.0f) negOneCount++;
        else {
            validCount++;
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
    }
    std::cout << "Pixel stats:" << std::endl;
    std::cout << "  NaN count: " << nanCount << std::endl;
    std::cout << "  -1 count: " << negOneCount << std::endl;
    std::cout << "  Valid count: " << validCount << std::endl;
    if (validCount > 0) {
        std::cout << "  Value range: [" << minVal << ", " << maxVal << "]" << std::endl;
    }
    
    CoordinateTransform transform(fitsData);
    
    // Test specific pixel
    int tileOrder = 3;
    int pixelOrder = tileOrder + 9;
    int tileNpix = 499;
    long baseHpxIndex = (long)tileNpix * TILE_SIZE * TILE_SIZE;
    
    // Test tile pixel (10, 0)
    int tileX = 10, tileY = 0;
    int mortonOffset = getMortonIndex(tileX, tileY);
    long hpxIndex = baseHpxIndex + mortonOffset;
    
    CelestialCoord radec = HealpixUtil::nestedToCelestial(pixelOrder, hpxIndex);
    std::cout << "\n=== Test tile pixel (" << tileX << "," << tileY << ") ===" << std::endl;
    std::cout << "HEALPix index: " << hpxIndex << std::endl;
    std::cout << "RA, Dec: (" << radec.ra << ", " << radec.dec << ")" << std::endl;
    
    Coord pixel = transform.celestialToPixel(radec);
    std::cout << "celestialToPixel: (" << pixel.x << ", " << pixel.y << ")" << std::endl;
    
    // Y flip
    double flippedY = fitsData.height - pixel.y - 1;
    double adjustedX = pixel.x - 1;
    std::cout << "After flip: (" << adjustedX << ", " << flippedY << ")" << std::endl;
    
    int ix = (int)adjustedX;
    int iy = (int)flippedY;
    std::cout << "Integer coords: (" << ix << ", " << iy << ")" << std::endl;
    
    // Get corner values
    int width = fitsData.width;
    const float* data = fitsData.pixels.data();
    
    float v00 = data[iy * width + ix];
    float v10 = data[iy * width + ix + 1];
    float v01 = data[(iy + 1) * width + ix];
    float v11 = data[(iy + 1) * width + ix + 1];
    
    std::cout << "Corner values:" << std::endl;
    std::cout << "  v00 [" << iy << "][" << ix << "] = " << v00 << std::endl;
    std::cout << "  v10 [" << iy << "][" << ix+1 << "] = " << v10 << std::endl;
    std::cout << "  v01 [" << iy+1 << "][" << ix << "] = " << v01 << std::endl;
    std::cout << "  v11 [" << iy+1 << "][" << ix+1 << "] = " << v11 << std::endl;
    
    // Java tile
    std::string javaTilePath = "/mnt/c/Users/xuyunfei/Documents/workspace/01key_project/01HiPS-Research/aladin/hips_output/Norder3/Dir0/Npix499.fits";
    FitsData javaTile = FitsReader::readFitsFile(javaTilePath);
    if (javaTile.isValid) {
        int javaIdx = tileY * TILE_SIZE + tileX;
        std::cout << "\nJava tile pixel (" << tileX << "," << tileY << "): " << javaTile.pixels[javaIdx] << std::endl;
    }
    
    return 0;
}
