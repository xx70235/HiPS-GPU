#include "gpu_full_batch_plan.h"

BatchWorkspacePlan make_batch_workspace_plan(
    int batch_tiles,
    int tile_width,
    int num_images,
    int bitpix
) {
    BatchWorkspacePlan plan;

    const size_t safe_batch_tiles = batch_tiles > 0 ? static_cast<size_t>(batch_tiles) : 0u;
    const size_t safe_tile_width = tile_width > 0 ? static_cast<size_t>(tile_width) : 0u;
    const size_t safe_num_images = num_images > 0 ? static_cast<size_t>(num_images) : 0u;

    plan.total_output_pixels = safe_batch_tiles * safe_tile_width * safe_tile_width;
    plan.mask_ints = safe_batch_tiles * safe_num_images;
    plan.use_float_results = (bitpix == -32);
    plan.result_element_bytes = plan.use_float_results ? sizeof(float) : sizeof(double);

    return plan;
}
