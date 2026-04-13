#include "tile_image_index.h"

#include <cassert>
#include <vector>

int main() {
    std::vector<GPUTileInfo> tiles = {
        {3, 100, {1, 4, 7}},
        {3, 101, {4}},
        {3, 102, {2, 5, 6, 8, 9}},
    };

    SparseTileImageIndex index = build_sparse_tile_image_index(tiles, 2);
    assert(index.tile_offsets.size() == 4);
    assert(index.tile_offsets[0] == 0);
    assert(index.tile_offsets[1] == 3);
    assert(index.tile_offsets[2] == 4);
    assert(index.tile_offsets[3] == 9);
    assert(index.image_indices.size() == 9);
    assert(index.chunk_pass_offsets.size() == 4);
    assert(index.chunk_pass_offsets[0] == 0);
    assert(index.chunk_pass_offsets[1] == 2);
    assert(index.chunk_pass_offsets[2] == 4);
    assert(index.chunk_pass_offsets[3] == 5);
    return 0;
}
