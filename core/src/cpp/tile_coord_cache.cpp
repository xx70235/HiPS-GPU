#include "tile_coord_cache.h"

#include "healpix_util.h"

#include <cstdint>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kMagic = 0x48434331u;  // HCC1

struct CoordCacheHeader {
    std::uint32_t magic;
    std::uint32_t tile_width;
    std::uint64_t pixel_count;
};

}  // namespace

std::string build_tile_coord_cache_path(
    const std::string& output_dir,
    int order,
    long npix
) {
    int dir = HealpixUtil::getDirNumber(npix);
    return output_dir + "/.hips_gpu_cache/celestial_coords/Norder" +
           std::to_string(order) + "/Dir" + std::to_string(dir) +
           "/Npix" + std::to_string(npix) + ".bin";
}

bool read_tile_coord_cache(
    const std::string& output_dir,
    int order,
    long npix,
    int tile_width,
    double* ra_out,
    double* dec_out,
    std::size_t pixel_count
) {
    const std::string path = build_tile_coord_cache_path(output_dir, order, npix);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    CoordCacheHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kMagic ||
        header.tile_width != static_cast<std::uint32_t>(tile_width) ||
        header.pixel_count != static_cast<std::uint64_t>(pixel_count)) {
        return false;
    }

    in.read(reinterpret_cast<char*>(ra_out), static_cast<std::streamsize>(pixel_count * sizeof(double)));
    in.read(reinterpret_cast<char*>(dec_out), static_cast<std::streamsize>(pixel_count * sizeof(double)));
    return static_cast<bool>(in);
}

bool write_tile_coord_cache(
    const std::string& output_dir,
    int order,
    long npix,
    int tile_width,
    const double* ra_values,
    const double* dec_values,
    std::size_t pixel_count
) {
    const std::string path = build_tile_coord_cache_path(output_dir, order, npix);
    fs::create_directories(fs::path(path).parent_path());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    const CoordCacheHeader header{
        kMagic,
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint64_t>(pixel_count),
    };
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(ra_values), static_cast<std::streamsize>(pixel_count * sizeof(double)));
    out.write(reinterpret_cast<const char*>(dec_values), static_cast<std::streamsize>(pixel_count * sizeof(double)));
    return static_cast<bool>(out);
}
