#include "tile_coord_cache.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

int main() {
    const fs::path root = fs::temp_directory_path() / "hips_gpu_coord_cache_test";
    fs::remove_all(root);
    fs::create_directories(root);

    const int tile_width = 2;
    const std::size_t pixel_count = 4;
    std::vector<double> ra = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> dec = {-1.0, -2.0, -3.0, -4.0};
    std::vector<double> loaded_ra(pixel_count, std::nan(""));
    std::vector<double> loaded_dec(pixel_count, std::nan(""));

    assert(!read_tile_coord_cache(root.string(), 5, 12345, tile_width, loaded_ra.data(), loaded_dec.data(), pixel_count));
    assert(write_tile_coord_cache(root.string(), 5, 12345, tile_width, ra.data(), dec.data(), pixel_count));
    assert(read_tile_coord_cache(root.string(), 5, 12345, tile_width, loaded_ra.data(), loaded_dec.data(), pixel_count));
    assert(loaded_ra == ra);
    assert(loaded_dec == dec);
    assert(!read_tile_coord_cache(root.string(), 5, 12345, tile_width + 1, loaded_ra.data(), loaded_dec.data(), pixel_count));

    const std::string cache_path = build_tile_coord_cache_path(root.string(), 5, 12345);
    assert(cache_path.find("celestial_coords") != std::string::npos);
    assert(cache_path.find("Norder5") != std::string::npos);
    assert(cache_path.find("Npix12345") != std::string::npos);

    fs::remove_all(root);
    return 0;
}
