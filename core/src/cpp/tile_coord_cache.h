#ifndef TILE_COORD_CACHE_H
#define TILE_COORD_CACHE_H

#include <cstddef>
#include <string>

std::string build_tile_coord_cache_path(
    const std::string& output_dir,
    int order,
    long npix
);

bool read_tile_coord_cache(
    const std::string& output_dir,
    int order,
    long npix,
    int tile_width,
    double* ra_out,
    double* dec_out,
    std::size_t pixel_count
);

bool write_tile_coord_cache(
    const std::string& output_dir,
    int order,
    long npix,
    int tile_width,
    const double* ra_values,
    const double* dec_values,
    std::size_t pixel_count
);

#endif  // TILE_COORD_CACHE_H
