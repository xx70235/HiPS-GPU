#include <iostream>
#include <iomanip>
#include <cmath>
#include "healpix_util.h"
#include "coordinate_transform.h"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    // Tile 499
    long tile_npix = 499;
    int tile_order = 3;
    int tile_size = 512;  // 512x512 pixels
    int pixel_order_diff = (int)log2(tile_size);  // 9 for 512
    int pixel_order = tile_order + pixel_order_diff;  // 3 + 9 = 12
    
    std::cout << "Tile " << tile_npix << " at order " << tile_order << std::endl;
    std::cout << "Tile size: " << tile_size << "x" << tile_size << std::endl;
    std::cout << "Pixel order diff: log2(" << tile_size << ") = " << pixel_order_diff << std::endl;
    std::cout << "Pixel order: " << tile_order << " + " << pixel_order_diff << " = " << pixel_order << std::endl;
    
    int nside_tile = 1 << tile_order;  // 8
    int nside_pixel = 1 << pixel_order;  // 4096
    long pixels_per_tile = (long)tile_size * tile_size;  // 262144
    long min_hpx = tile_npix * pixels_per_tile;
    
    std::cout << "nside (tile): " << nside_tile << std::endl;
    std::cout << "nside (pixel): " << nside_pixel << std::endl;
    std::cout << "Pixels per tile: " << pixels_per_tile << std::endl;
    std::cout << "Min HEALPix index: " << min_hpx << std::endl;
    
    long total_pixels_at_pixel_order = HealpixUtil::getTotalPixels(pixel_order);
    std::cout << "Total pixels at order " << pixel_order << ": " << total_pixels_at_pixel_order << std::endl;
    std::cout << "npix < total: " << (min_hpx < total_pixels_at_pixel_order ? "yes" : "NO!") << std::endl;
    
    // 测试几个像素
    std::vector<long> test_hpx = {min_hpx, min_hpx + 100, min_hpx + 1000};
    
    for (long hpx : test_hpx) {
        CelestialCoord coord = HealpixUtil::nestedToCelestial(pixel_order, hpx);
        std::cout << "HEALPix " << hpx << " -> RA=" << coord.ra << ", Dec=" << coord.dec << std::endl;
    }
    
    return 0;
}
