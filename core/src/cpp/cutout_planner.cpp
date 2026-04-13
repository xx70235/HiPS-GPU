#include "cutout_planner.h"

#include <algorithm>
#include <set>
#include <utility>

namespace {

PixelBounds clip_bounds(const PixelBounds& bounds, int image_width, int image_height) {
    PixelBounds clipped;
    clipped.x0 = std::max(0, bounds.x0);
    clipped.y0 = std::max(0, bounds.y0);
    int x1 = std::min(image_width, bounds.x0 + bounds.width);
    int y1 = std::min(image_height, bounds.y0 + bounds.height);
    clipped.width = std::max(0, x1 - clipped.x0);
    clipped.height = std::max(0, y1 - clipped.y0);
    return clipped;
}

int choose_partition_size(const FitsData& image_meta, int default_partition_size) {
    const int max_dim = std::max(image_meta.width, image_meta.height);
    if (max_dim <= 0) {
        return 0;
    }
    return std::min(default_partition_size, max_dim);
}

}  // namespace

std::vector<PartitionCell> snap_requests_to_partition_cells(
    const std::vector<PixelBounds>& requests,
    int partition_size,
    int image_width,
    int image_height
) {
    std::vector<PartitionCell> result;
    if (partition_size <= 0 || image_width <= 0 || image_height <= 0) {
        return result;
    }

    std::set<std::pair<int, int>> seen;
    for (const PixelBounds& request : requests) {
        PixelBounds clipped = clip_bounds(request, image_width, image_height);
        if (clipped.width <= 0 || clipped.height <= 0) {
            continue;
        }

        int x0_cell = clipped.x0 / partition_size;
        int y0_cell = clipped.y0 / partition_size;
        int x1_cell = (clipped.x0 + clipped.width - 1) / partition_size;
        int y1_cell = (clipped.y0 + clipped.height - 1) / partition_size;

        for (int cell_y = y0_cell; cell_y <= y1_cell; ++cell_y) {
            for (int cell_x = x0_cell; cell_x <= x1_cell; ++cell_x) {
                if (!seen.insert({cell_y, cell_x}).second) {
                    continue;
                }

                PartitionCell cell;
                cell.cell_x = cell_x;
                cell.cell_y = cell_y;
                cell.bounds.x0 = cell_x * partition_size;
                cell.bounds.y0 = cell_y * partition_size;
                cell.bounds.width = std::min(partition_size, image_width - cell.bounds.x0);
                cell.bounds.height = std::min(partition_size, image_height - cell.bounds.y0);
                result.push_back(cell);
            }
        }
    }

    return result;
}

PlannedCutoutCells plan_cutout_cells(
    const std::vector<PixelBounds>& requests,
    int partition_size,
    int image_width,
    int image_height,
    double full_image_threshold
) {
    PlannedCutoutCells plan;
    if (image_width <= 0 || image_height <= 0) {
        return plan;
    }

    plan.image_pixels = static_cast<size_t>(image_width) * static_cast<size_t>(image_height);
    if (plan.image_pixels == 0) {
        return plan;
    }

    plan.effective_partition_size = partition_size;
    plan.cells = snap_requests_to_partition_cells(requests, partition_size, image_width, image_height);
    for (const PartitionCell& cell : plan.cells) {
        plan.planned_pixels += static_cast<size_t>(cell.bounds.width) *
                               static_cast<size_t>(cell.bounds.height);
    }
    plan.planned_fraction = static_cast<double>(plan.planned_pixels) /
                            static_cast<double>(plan.image_pixels);

    if (!plan.cells.empty() && plan.planned_fraction >= full_image_threshold) {
        plan.use_full_image = true;
        plan.cells.clear();

        PartitionCell full;
        full.bounds.x0 = 0;
        full.bounds.y0 = 0;
        full.bounds.width = image_width;
        full.bounds.height = image_height;
        plan.cells.push_back(full);
        plan.planned_pixels = plan.image_pixels;
        plan.planned_fraction = 1.0;
        plan.effective_partition_size = std::max(image_width, image_height);
    }

    return plan;
}

CutoutLoadPlan build_cutout_load_plan(
    const std::vector<std::vector<PixelBounds>>& source_requests,
    const std::vector<FitsData>& source_metas,
    int default_partition_size,
    double full_image_threshold
) {
    CutoutLoadPlan load_plan;
    load_plan.partition_sizes.resize(source_metas.size(), 0);

    const std::size_t source_count = std::min(source_requests.size(), source_metas.size());
    for (std::size_t source_index = 0; source_index < source_count; ++source_index) {
        if (source_requests[source_index].empty()) {
            continue;
        }

        const FitsData& image_meta = source_metas[source_index];
        if (!image_meta.isValid || image_meta.width <= 0 || image_meta.height <= 0) {
            continue;
        }

        int partition_size = choose_partition_size(image_meta, default_partition_size);
        PlannedCutoutCells plan = plan_cutout_cells(
            source_requests[source_index],
            partition_size,
            image_meta.width,
            image_meta.height,
            full_image_threshold
        );

        bool read_full_image = plan.use_full_image;
        if (plan.use_full_image) {
            load_plan.high_coverage_full_image_count++;
        }
        if (plan.effective_partition_size > 0) {
            partition_size = plan.effective_partition_size;
        }

        std::vector<PartitionCell> cells = std::move(plan.cells);
        if (cells.empty()) {
            PartitionCell full;
            full.bounds.x0 = 0;
            full.bounds.y0 = 0;
            full.bounds.width = image_meta.width;
            full.bounds.height = image_meta.height;
            cells.push_back(full);
            partition_size = std::max(image_meta.width, image_meta.height);
            read_full_image = true;
        }

        load_plan.partition_sizes[source_index] = partition_size;

        for (const PartitionCell& cell : cells) {
            if (cell.bounds.width <= 0 || cell.bounds.height <= 0) {
                continue;
            }

            CutoutLoadJob job;
            job.key.source_index = static_cast<int>(source_index);
            job.key.cell_x = cell.cell_x;
            job.key.cell_y = cell.cell_y;
            job.bounds = cell.bounds;
            job.read_full_image = read_full_image;

            load_plan.key_to_job_index[job.key] = static_cast<int>(load_plan.jobs.size());
            load_plan.jobs.push_back(job);
        }
    }

    return load_plan;
}
