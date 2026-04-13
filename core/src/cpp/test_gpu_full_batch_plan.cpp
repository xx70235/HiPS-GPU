#include "gpu_full_batch_plan.h"

#include <cassert>

int main() {
    BatchWorkspacePlan plan = make_batch_workspace_plan(/*batch_tiles=*/85, /*tile_width=*/512, /*num_images=*/42, /*bitpix=*/-32);
    assert(plan.use_float_results);
    assert(plan.result_element_bytes == sizeof(float));
    assert(plan.coord_element_bytes == sizeof(double));
    assert(plan.total_output_pixels == 85u * 512u * 512u);
    return 0;
}
