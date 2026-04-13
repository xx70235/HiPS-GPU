#ifndef PARALLEL_UTILS_H
#define PARALLEL_UTILS_H

#include <cstddef>
#include <string>
#include <vector>

int choose_parallel_chunk_size(
    std::size_t task_count,
    int worker_count,
    int target_chunks_per_worker = 4,
    int max_chunk_size = 64
);

std::vector<std::string> collect_unique_parent_dirs(
    const std::vector<std::string>& file_paths
);

#endif  // PARALLEL_UTILS_H
