#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def tile_map(root: Path):
    tiles = {}
    for path in root.rglob("Npix*.fits"):
        rel = path.relative_to(root).as_posix()
        tiles[rel] = path
    return tiles


def valid_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data)


def count_valid_pixels(path: Path) -> int:
    data = fits.getdata(path)
    return int(valid_mask(data).sum())


def main():
    if len(sys.argv) != 4:
        print("Usage: compare_hips_outputs.py <java_out> <gpu_out> <json_out>", file=sys.stderr)
        sys.exit(2)

    java_root = Path(sys.argv[1])
    gpu_root = Path(sys.argv[2])
    json_out = Path(sys.argv[3])

    java_tiles = tile_map(java_root)
    gpu_tiles = tile_map(gpu_root)

    common = sorted(set(java_tiles) & set(gpu_tiles))
    java_only = sorted(set(java_tiles) - set(gpu_tiles))
    gpu_only = sorted(set(gpu_tiles) - set(java_tiles))

    per_tile_close = []
    per_tile_median_abs = []
    global_max_abs = 0.0
    total_common_valid = 0
    total_close_1e3 = 0
    total_close_1e2 = 0

    for rel in common:
        j = fits.getdata(java_tiles[rel]).astype(np.float64, copy=False)
        g = fits.getdata(gpu_tiles[rel]).astype(np.float64, copy=False)

        mask = valid_mask(j) & valid_mask(g)
        n = int(mask.sum())
        if n == 0:
            continue

        diff = np.abs(j[mask] - g[mask])
        total_common_valid += n
        total_close_1e3 += int((diff < 1e-3).sum())
        total_close_1e2 += int((diff < 1e-2).sum())
        tile_close = float((diff < 1e-3).sum() / n * 100.0)
        per_tile_close.append(tile_close)
        med = float(np.median(diff))
        per_tile_median_abs.append(med)
        tile_max = float(np.max(diff))
        if tile_max > global_max_abs:
            global_max_abs = tile_max

    gpu_only_valid = [count_valid_pixels(gpu_tiles[rel]) for rel in gpu_only[:2000]]
    java_only_valid = [count_valid_pixels(java_tiles[rel]) for rel in java_only[:2000]]

    result = {
        "java_tile_count": len(java_tiles),
        "gpu_tile_count": len(gpu_tiles),
        "common_tile_count": len(common),
        "java_only_tile_count": len(java_only),
        "gpu_only_tile_count": len(gpu_only),
        "total_common_valid_pixels": total_common_valid,
        "pct_common_pixels_abs_lt_1e3": (total_close_1e3 / total_common_valid * 100.0) if total_common_valid else math.nan,
        "pct_common_pixels_abs_lt_1e2": (total_close_1e2 / total_common_valid * 100.0) if total_common_valid else math.nan,
        "median_per_tile_abs_diff": float(np.median(per_tile_median_abs)) if per_tile_median_abs else math.nan,
        "median_per_tile_pct_abs_lt_1e3": float(np.median(per_tile_close)) if per_tile_close else math.nan,
        "global_max_abs_diff": global_max_abs,
        "gpu_only_valid_pixel_median": float(np.median(gpu_only_valid)) if gpu_only_valid else math.nan,
        "gpu_only_valid_pixel_max": int(max(gpu_only_valid)) if gpu_only_valid else 0,
        "java_only_valid_pixel_median": float(np.median(java_only_valid)) if java_only_valid else math.nan,
        "java_only_valid_pixel_max": int(max(java_only_valid)) if java_only_valid else 0,
        "sample_gpu_only_tiles": gpu_only[:10],
        "sample_java_only_tiles": java_only[:10],
    }

    json_out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
