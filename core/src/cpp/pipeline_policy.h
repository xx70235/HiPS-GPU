#ifndef PIPELINE_POLICY_H
#define PIPELINE_POLICY_H

bool should_run_tree_rebuild(
    bool append_mode,
    bool enable_tiles,
    int order_max,
    bool enable_tree_rebuild
);

int compute_output_tile_order_start(
    bool append_mode,
    int order_max,
    bool enable_tree_rebuild
);

bool should_defer_preview_until_tree(
    int order_max,
    bool run_tree_rebuild
);

#endif  // PIPELINE_POLICY_H
