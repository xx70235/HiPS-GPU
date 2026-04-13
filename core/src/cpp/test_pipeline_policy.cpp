#include "pipeline_policy.h"

#include <cassert>

int main() {
    assert(!should_run_tree_rebuild(false, true, 13, false));
    assert(should_run_tree_rebuild(false, true, 13, true));
    assert(!should_run_tree_rebuild(true, true, 13, true));
    assert(!should_run_tree_rebuild(false, false, 13, true));

    assert(compute_output_tile_order_start(false, 13, false) == 13);
    assert(compute_output_tile_order_start(false, 13, true) == 0);
    assert(compute_output_tile_order_start(true, 13, true) == 13);

    assert(!should_defer_preview_until_tree(13, false));
    assert(should_defer_preview_until_tree(13, true));
    assert(!should_defer_preview_until_tree(3, true));
    return 0;
}
