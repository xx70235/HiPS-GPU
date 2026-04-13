#include "index_merge_utils.h"

#include <algorithm>
#include <unordered_map>

namespace {

std::string normalize_source_key(const SourceFileInfo& info) {
    std::string key = info.filepath.empty() ? info.filename : info.filepath;
    std::replace(key.begin(), key.end(), '\\', '/');
    return key;
}

}  // namespace

bool merge_source_files_unique(std::vector<SourceFileInfo>& existing,
                               const std::vector<SourceFileInfo>& incoming) {
    std::unordered_map<std::string, size_t> existing_positions;
    existing_positions.reserve(existing.size());

    for (size_t i = 0; i < existing.size(); ++i) {
        std::string key = normalize_source_key(existing[i]);
        if (!key.empty()) {
            existing_positions[key] = i;
        }
    }

    bool changed = false;
    for (const auto& item : incoming) {
        std::string key = normalize_source_key(item);
        if (key.empty()) {
            continue;
        }

        auto it = existing_positions.find(key);
        if (it == existing_positions.end()) {
            existing.push_back(item);
            existing_positions[key] = existing.size() - 1;
            changed = true;
            continue;
        }

        SourceFileInfo& current = existing[it->second];
        if (current.cellMem != item.cellMem) {
            current.cellMem = item.cellMem;
            changed = true;
        }
        if (current.filepath.empty() && !item.filepath.empty()) {
            current.filepath = item.filepath;
            current.filename = item.filename;
            changed = true;
        }
    }

    return changed;
}
