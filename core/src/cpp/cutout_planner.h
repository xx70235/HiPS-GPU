#ifndef CUTOUT_PLANNER_H
#define CUTOUT_PLANNER_H

#include <cstddef>
#include <map>
#include <vector>

#include "fits_io.h"

struct PartitionCell {
    PixelBounds bounds;
    int cell_x = 0;
    int cell_y = 0;
};

struct PlannedCutoutCells {
    std::vector<PartitionCell> cells;
    bool use_full_image = false;
    size_t planned_pixels = 0;
    size_t image_pixels = 0;
    double planned_fraction = 0.0;
    int effective_partition_size = 0;
};

struct CutoutLoadKey {
    int source_index = -1;
    int cell_x = 0;
    int cell_y = 0;

    bool operator<(const CutoutLoadKey& other) const {
        if (source_index != other.source_index) return source_index < other.source_index;
        if (cell_y != other.cell_y) return cell_y < other.cell_y;
        return cell_x < other.cell_x;
    }
};

struct CutoutLoadJob {
    CutoutLoadKey key;
    PixelBounds bounds;
    bool read_full_image = false;
};

struct CutoutLoadPlan {
    std::vector<CutoutLoadJob> jobs;
    std::vector<int> partition_sizes;
    std::map<CutoutLoadKey, int> key_to_job_index;
    int high_coverage_full_image_count = 0;
};

std::vector<PartitionCell> snap_requests_to_partition_cells(
    const std::vector<PixelBounds>& requests,
    int partition_size,
    int image_width,
    int image_height
);

PlannedCutoutCells plan_cutout_cells(
    const std::vector<PixelBounds>& requests,
    int partition_size,
    int image_width,
    int image_height,
    double full_image_threshold = 0.85
);

CutoutLoadPlan build_cutout_load_plan(
    const std::vector<std::vector<PixelBounds>>& source_requests,
    const std::vector<FitsData>& source_metas,
    int default_partition_size = 512,
    double full_image_threshold = 0.85
);

#endif // CUTOUT_PLANNER_H
