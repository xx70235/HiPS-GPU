#ifndef GPU_FULL_BATCH_PLAN_H
#define GPU_FULL_BATCH_PLAN_H

#include <cstddef>

struct BatchWorkspacePlan {
    size_t total_output_pixels = 0;
    size_t coord_element_bytes = sizeof(double);
    size_t result_element_bytes = sizeof(double);
    size_t mask_ints = 0;
    bool use_float_results = false;
};

BatchWorkspacePlan make_batch_workspace_plan(
    int batch_tiles,
    int tile_width,
    int num_images,
    int bitpix
);

#endif
