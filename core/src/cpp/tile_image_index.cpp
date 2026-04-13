#include "tile_image_index.h"

#include <algorithm>

SparseTileImageIndex build_sparse_tile_image_index(
    const std::vector<GPUTileInfo>& tiles,
    int max_overlay_per_pass
) {
    SparseTileImageIndex index;
    index.tile_offsets.reserve(tiles.size() + 1);
    index.tile_offsets.push_back(0);

    for (const auto& tile : tiles) {
        const int count = static_cast<int>(tile.imageIndices.size());
        index.max_images_per_tile = std::max(index.max_images_per_tile, count);
        index.image_indices.insert(index.image_indices.end(), tile.imageIndices.begin(), tile.imageIndices.end());
        index.tile_offsets.push_back(static_cast<int>(index.image_indices.size()));
    }

    const int pass_width = max_overlay_per_pass > 0 ? max_overlay_per_pass : std::max(1, index.max_images_per_tile);
    index.chunk_pass_offsets.push_back(0);
    if (index.max_images_per_tile == 0) {
        index.chunk_pass_offsets.push_back(0);
    } else {
        for (int offset = pass_width; offset < index.max_images_per_tile; offset += pass_width) {
            index.chunk_pass_offsets.push_back(offset);
        }
        index.chunk_pass_offsets.push_back(index.max_images_per_tile);
    }

    return index;
}
