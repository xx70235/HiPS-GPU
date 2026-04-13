#include "cutout_planner.h"

#include <cassert>
#include <vector>

int main() {
    std::vector<PixelBounds> requests = {
        {10, 10, 20, 20},
        {18, 18, 20, 10},
        {80, 16, 8, 8}
    };

    auto cells = snap_requests_to_partition_cells(requests, 32, 128, 128);
    assert(cells.size() == 3);
    assert(cells[0].bounds.x0 == 0);
    assert(cells[0].bounds.y0 == 0);
    assert(cells[0].bounds.width == 32);
    assert(cells[0].bounds.height == 32);
    assert(cells[1].bounds.x0 == 32);
    assert(cells[1].bounds.y0 == 0);
    assert(cells[2].bounds.x0 == 64);
    assert(cells[2].bounds.y0 == 0);

    PlannedCutoutCells sparsePlan = plan_cutout_cells(requests, 32, 128, 128);
    assert(!sparsePlan.use_full_image);
    assert(sparsePlan.cells.size() == 3);
    assert(sparsePlan.planned_pixels == 3u * 32u * 32u);
    assert(sparsePlan.image_pixels == 128u * 128u);
    assert(sparsePlan.effective_partition_size == 32);

    std::vector<PixelBounds> denseRequests = {
        {0, 0, 128, 128}
    };
    PlannedCutoutCells densePlan = plan_cutout_cells(denseRequests, 32, 128, 128);
    assert(densePlan.use_full_image);
    assert(densePlan.cells.size() == 1);
    assert(densePlan.cells[0].bounds.x0 == 0);
    assert(densePlan.cells[0].bounds.y0 == 0);
    assert(densePlan.cells[0].bounds.width == 128);
    assert(densePlan.cells[0].bounds.height == 128);
    assert(densePlan.planned_fraction == 1.0);
    assert(densePlan.effective_partition_size == 128);

    std::vector<FitsData> sourceMetas(3);
    sourceMetas[0].width = 128;
    sourceMetas[0].height = 128;
    sourceMetas[0].isValid = true;
    sourceMetas[1].width = 256;
    sourceMetas[1].height = 128;
    sourceMetas[1].isValid = true;
    sourceMetas[2].width = 64;
    sourceMetas[2].height = 64;
    sourceMetas[2].isValid = false;

    std::vector<std::vector<PixelBounds>> sourceRequests(3);
    sourceRequests[0] = {
        {10, 10, 20, 20},
        {80, 16, 8, 8}
    };
    sourceRequests[1] = {
        {0, 0, 256, 128}
    };
    sourceRequests[2] = {
        {0, 0, 64, 64}
    };

    CutoutLoadPlan loadPlan = build_cutout_load_plan(sourceRequests, sourceMetas, 32, 0.85);
    assert(loadPlan.partition_sizes.size() == 3);
    assert(loadPlan.partition_sizes[0] == 32);
    assert(loadPlan.partition_sizes[1] == 256);
    assert(loadPlan.partition_sizes[2] == 0);
    assert(loadPlan.high_coverage_full_image_count == 1);
    assert(loadPlan.jobs.size() == 3);
    assert(loadPlan.key_to_job_index.size() == 3);

    assert(loadPlan.jobs[0].key.source_index == 0);
    assert(loadPlan.jobs[0].key.cell_x == 0);
    assert(loadPlan.jobs[0].key.cell_y == 0);
    assert(!loadPlan.jobs[0].read_full_image);

    assert(loadPlan.jobs[1].key.source_index == 0);
    assert(loadPlan.jobs[1].key.cell_x == 2);
    assert(loadPlan.jobs[1].key.cell_y == 0);
    assert(!loadPlan.jobs[1].read_full_image);

    assert(loadPlan.jobs[2].key.source_index == 1);
    assert(loadPlan.jobs[2].key.cell_x == 0);
    assert(loadPlan.jobs[2].key.cell_y == 0);
    assert(loadPlan.jobs[2].read_full_image);
    assert(loadPlan.jobs[2].bounds.x0 == 0);
    assert(loadPlan.jobs[2].bounds.y0 == 0);
    assert(loadPlan.jobs[2].bounds.width == 256);
    assert(loadPlan.jobs[2].bounds.height == 128);

    CutoutLoadKey firstKey;
    firstKey.source_index = 0;
    firstKey.cell_x = 0;
    firstKey.cell_y = 0;
    assert(loadPlan.key_to_job_index.at(firstKey) == 0);

    CutoutLoadKey secondKey;
    secondKey.source_index = 0;
    secondKey.cell_x = 2;
    secondKey.cell_y = 0;
    assert(loadPlan.key_to_job_index.at(secondKey) == 1);

    CutoutLoadKey fullKey;
    fullKey.source_index = 1;
    fullKey.cell_x = 0;
    fullKey.cell_y = 0;
    assert(loadPlan.key_to_job_index.at(fullKey) == 2);
    return 0;
}
