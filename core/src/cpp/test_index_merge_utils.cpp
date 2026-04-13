#include "hpx_finder.h"
#include "index_merge_utils.h"

#include <cassert>
#include <vector>

int main() {
    std::vector<SourceFileInfo> existing = {
        SourceFileInfo("a.fits", 10),
        SourceFileInfo("b.fits", 20)
    };
    std::vector<SourceFileInfo> incoming = {
        SourceFileInfo("b.fits", 20),
        SourceFileInfo("c.fits", 30)
    };

    bool changed = merge_source_files_unique(existing, incoming);
    assert(changed);
    assert(existing.size() == 3);
    assert(existing[0].filepath == "a.fits");
    assert(existing[1].filepath == "b.fits");
    assert(existing[2].filepath == "c.fits");

    changed = merge_source_files_unique(existing, incoming);
    assert(!changed);
    assert(existing.size() == 3);
    return 0;
}
