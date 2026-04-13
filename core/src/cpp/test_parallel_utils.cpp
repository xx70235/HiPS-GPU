#include "parallel_utils.h"

#include <cassert>
#include <set>
#include <string>
#include <vector>

int main() {
    assert(choose_parallel_chunk_size(0, 32) == 1);
    assert(choose_parallel_chunk_size(46, 32) == 1);
    assert(choose_parallel_chunk_size(512, 32) == 4);
    assert(choose_parallel_chunk_size(4096, 32) == 32);
    assert(choose_parallel_chunk_size(100000, 32) == 64);

    std::vector<std::string> filePaths = {
        "/tmp/out/Norder3/Dir0/Npix1.fits",
        "/tmp/out/Norder3/Dir0/Npix2.fits",
        "/tmp/out/.hips_gpu_cache/weighted_sum/Norder3/Dir0/Npix1.fits",
        "/tmp/out/.hips_gpu_cache/weighted_sum/Norder3/Dir0/Npix1.fits",
        "relative/Norder4/Dir100/Npix1024.fits",
    };
    std::vector<std::string> dirs = collect_unique_parent_dirs(filePaths);
    std::set<std::string> dirSet(dirs.begin(), dirs.end());
    assert(dirs.size() == dirSet.size());
    assert(dirSet.count("/tmp/out/Norder3/Dir0") == 1);
    assert(dirSet.count("/tmp/out/.hips_gpu_cache/weighted_sum/Norder3/Dir0") == 1);
    assert(dirSet.count("relative/Norder4/Dir100") == 1);
    return 0;
}
