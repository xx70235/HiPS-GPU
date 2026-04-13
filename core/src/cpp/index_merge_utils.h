#ifndef INDEX_MERGE_UTILS_H
#define INDEX_MERGE_UTILS_H

#include <vector>

#include "hpx_finder.h"

bool merge_source_files_unique(std::vector<SourceFileInfo>& existing,
                               const std::vector<SourceFileInfo>& incoming);

#endif  // INDEX_MERGE_UTILS_H
