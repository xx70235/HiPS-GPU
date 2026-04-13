#include "pipeline_policy.h"

bool should_run_tree_rebuild(
    bool append_mode,
    bool enable_tiles,
    int order_max,
    bool enable_tree_rebuild
) {
    return enable_tree_rebuild && enable_tiles && !append_mode && order_max > 0;
}

int compute_output_tile_order_start(
    bool append_mode,
    int order_max,
    bool enable_tree_rebuild
) {
    return (append_mode || !enable_tree_rebuild) ? order_max : 0;
}

bool should_defer_preview_until_tree(
    int order_max,
    bool run_tree_rebuild
) {
    return run_tree_rebuild && order_max > 3;
}
