#!/usr/bin/env python3
"""
Measure a practical latency budget for the HpxFinder-guided workflow.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "runtime") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "runtime"))

from project_paths import DATA_ROOT, OUTPUT_ROOT

from hips_processor import diff_hips, run_local_hips  # type: ignore
from hips_runtime import query_hpxfinder_sources_for_sky  # type: ignore
from photometry import run_photometry, combine_flux_measurements  # type: ignore


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0


def main():
    base = DATA_ROOT / "e2e_20260321a"
    cands = json.load(open(base / "A" / "diff" / "candidates.json"))
    # pick a stable recovered source around S/N=10
    target = next(c for c in cands if c["snr_ref"] == 10 and c["recovered"])

    # 1. epoch HiPS build latency from a single image
    tmp = OUTPUT_ROOT / "validation_runs" / "latency_budget"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    _, t_hips = timed(
        run_local_hips,
        str(base / "A" / "epoch_input"),
        str(tmp / "epoch_hips"),
        order=7,
        mode="MEAN",
        timeout=300,
    )

    # 2. HpxFinder lookup latency
    lookup_times = []
    for _ in range(100):
        _, dt = timed(
            query_hpxfinder_sources_for_sky,
            str(base / "A" / "epoch_hips"),
            float(target["ra"]),
            float(target["dec"]),
            7,
        )
        lookup_times.append(dt)

    # 3. original-domain differencing latency (single candidate)
    cand_pos = [{
        "truth_id": target["truth_id"],
        "ra": target["ra"],
        "dec": target["dec"],
        "snr_ref": target["snr_ref"],
        "flux": 0.0,
    }]
    _, t_diff = timed(
        diff_hips,
        event_id="SIM",
        tid="A",
        night="20260321",
        band="r",
        candidate_positions=cand_pos,
        reference_local_dir=str(base / "A" / "reference_hips"),
        epoch_local_dir=str(base / "A" / "epoch_hips"),
        reference_source_dir=str(base / "A" / "reference_input"),
        epoch_source_dir=str(base / "A" / "epoch_input"),
        output_local_dir=str(tmp / "diff_out"),
        threshold_sigma=5.0,
        cutout_half_size=25,
        detection_radius=4,
        cleanup=False,
    )

    # 4. single-image forced photometry latency
    data = fits.getdata(base / "A" / "epoch_input" / "A_epoch_r.fits")
    hdr = fits.getheader(base / "A" / "epoch_input" / "A_epoch_r.fits")
    phot_times = []
    phot_results = []
    for tel in ["A", "B", "C"]:
        data = fits.getdata(base / tel / "epoch_input" / f"{tel}_epoch_r.fits")
        hdr = fits.getheader(base / tel / "epoch_input" / f"{tel}_epoch_r.fits")
        res, dt = timed(
            run_photometry,
            data, hdr,
            float(target["ra"]), float(target["dec"]),
            "r", None, False,
        )
        phot_times.append(dt)
        phot_results.append(res)

    # 5. flux-space combination latency
    _, t_combine = timed(combine_flux_measurements, phot_results)

    out = {
        "epoch_hips_generation_s": t_hips,
        "hpxfinder_lookup_ms_median": statistics.median(lookup_times) * 1000.0,
        "original_domain_difference_s": t_diff,
        "forced_photometry_ms_median": statistics.median(phot_times) * 1000.0,
        "flux_combination_ms": t_combine * 1000.0,
        "end_to_end_s": t_hips + statistics.median(lookup_times) + t_diff + statistics.median(phot_times) + t_combine,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
