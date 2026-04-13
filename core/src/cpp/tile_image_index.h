#ifndef TILE_IMAGE_INDEX_H
#define TILE_IMAGE_INDEX_H

#include <vector>

#include "gpu_full_processor.h"

struct SparseTileImageIndex {
    std::vector<int> tile_offsets;
    std::vector<int> image_indices;
    std::vector<int> chunk_pass_offsets;
    int max_images_per_tile = 0;
};

SparseTileImageIndex build_sparse_tile_image_index(
    const std::vector<GPUTileInfo>& tiles,
    int max_overlay_per_pass
);

#endif
