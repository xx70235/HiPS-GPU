#include "parallel_utils.h"

#include <algorithm>
#include <filesystem>
#include <unordered_set>

namespace fs = std::filesystem;

int choose_parallel_chunk_size(
    std::size_t task_count,
    int worker_count,
    int target_chunks_per_worker,
    int max_chunk_size
) {
    if (task_count == 0) {
        return 1;
    }

    const int safe_workers = std::max(1, worker_count);
    const int safe_chunks_per_worker = std::max(1, target_chunks_per_worker);
    const int safe_max_chunk_size = std::max(1, max_chunk_size);
    const std::size_t target_chunks = static_cast<std::size_t>(safe_workers) * static_cast<std::size_t>(safe_chunks_per_worker);
    const std::size_t raw_chunk = (task_count + target_chunks - 1) / target_chunks;
    return static_cast<int>(std::min<std::size_t>(std::max<std::size_t>(1, raw_chunk), static_cast<std::size_t>(safe_max_chunk_size)));
}

std::vector<std::string> collect_unique_parent_dirs(const std::vector<std::string>& file_paths) {
    std::vector<std::string> dirs;
    std::unordered_set<std::string> seen;
    dirs.reserve(file_paths.size());
    for (const auto& file_path : file_paths) {
        fs::path path(file_path);
        if (!path.has_parent_path()) {
            continue;
        }
        std::string parent = path.parent_path().string();
        if (seen.insert(parent).second) {
            dirs.push_back(parent);
        }
    }
    return dirs;
}
