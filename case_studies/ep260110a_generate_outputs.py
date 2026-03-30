#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DATA_ROOT, OUTPUT_ROOT, ensure_dir

INPUT_ROOT = DATA_ROOT / "ep260110a"
CASE = ensure_dir(Path(os.environ.get("HIPS_GPU_EP260110A_OUTPUT_DIR", OUTPUT_ROOT / "case_studies" / "EP260110a")))
TRIGGER = datetime.fromisoformat("2026-01-10T11:58:23")


plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def hours_from_trigger(date_obs: str) -> float:
    dt = datetime.fromisoformat(date_obs)
    return (dt - TRIGGER).total_seconds() / 3600.0


def main():
    summary = json.load(open(INPUT_ROOT / "summary.json"))
    gcn_rows = list(csv.DictReader(open(INPUT_ROOT / "ep260110a_gcn_followup_summary.csv")))

    xl = summary["xl100_case"]["phot_rows"]
    trt = summary["trt_single_case"]["phot_rows"]
    trt_stack = summary["trt_stack_case"]["phot_rows"][0]

    # XL100 multi-night figure
    t = np.array([hours_from_trigger(r["date_obs"]) for r in xl])
    diff_snr = np.array([r["diff_snr"] for r in xl], dtype=float)
    lim = np.array([r["lim_mag_3sigma"] for r in xl], dtype=float)
    mag = np.array([r["mag"] if r["mag"] is not None else np.nan for r in xl], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.8), sharex=True)
    ax1.axhline(5.0, color="red", linestyle="--", linewidth=1, label="5σ")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.plot(t, diff_snr, "o-", color="#1f77b4", label="Target-centered diff S/N")
    ax1.set_ylabel("Diff S/N")
    ax1.set_title("EP260110a: XL100 Multi-night Constraints")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)

    ax2.plot(t, lim, "s-", color="#444444", label="3σ upper limit")
    good = np.isfinite(mag)
    if np.any(good):
        ax2.scatter(t[good], mag[good], marker="o", color="#ff7f0e", label="Calibrated target mag")
    ax2.set_xlabel("Hours since trigger")
    ax2.set_ylabel("Magnitude (smaller = brighter)")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False)

    out1 = CASE / "fig_ep260110a_xl100_multinight.png"
    fig.tight_layout()
    fig.savefig(out1)
    plt.close(fig)

    # TRT single-frame vs stack figure
    t_trt = np.array([hours_from_trigger(r["date_obs"]) for r in trt])
    snr_trt = np.array([r["snr"] for r in trt], dtype=float)
    diff_trt = np.array([r["diff_snr"] for r in trt], dtype=float)
    lim_trt = np.array([r["lim_mag_3sigma"] for r in trt], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    ax1.plot(t_trt, snr_trt, "o-", color="#1f77b4", label="Single-frame phot S/N")
    ax1.plot(t_trt, diff_trt, "s--", color="#ff7f0e", label="Single-frame diff S/N")
    ax1.axhline(trt_stack["snr"], color="#2ca02c", linestyle="-", label="Stacked phot S/N")
    ax1.axhline(trt_stack["diff_snr"], color="#2ca02c", linestyle="--", label="Stacked diff S/N")
    ax1.set_xlabel("Hours since trigger")
    ax1.set_ylabel("S/N")
    ax1.set_title("TRT Single-frame vs Stacked")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)

    labels = ["median single", "best single", "stacked", "GCN 43390"]
    values = [
        float(np.median(lim_trt)),
        float(np.max(lim_trt)),
        float(trt_stack["lim_mag_3sigma"]),
        19.6,
    ]
    ax2.bar(labels, values, color=["#999999", "#1f77b4", "#2ca02c", "#d62728"])
    ax2.set_ylabel("3σ / reported limit magnitude")
    ax2.set_title("TRT Limiting-Magnitude Comparison")
    ax2.invert_yaxis()
    ax2.grid(True, axis="y", alpha=0.25)

    out2 = CASE / "fig_ep260110a_trt_single_vs_stack.png"
    fig.tight_layout()
    fig.savefig(out2)
    plt.close(fig)

    # Formal GCN consistency table
    out_table = CASE / "ep260110a_formal_consistency_table.csv"
    rows = []
    rows.append({
        "dataset": "XL100 nightly sequence",
        "band": "g",
        "time_window_hr": f"{t.min():.2f}--{t.max():.2f}",
        "our_result": f"target-centered diff S/N median={np.median(diff_snr):.2f}, max={np.max(diff_snr):.2f}; 3σ limits span {np.min(lim):.2f}--{np.max(lim):.2f} mag",
        "published_reference": "No published g-band detection found in inspected GCN sample",
        "consistency_note": "Consistent with a non-detection / weak-constraint optical scenario",
    })
    rows.append({
        "dataset": "TRT-SRO single frames",
        "band": "R",
        "time_window_hr": f"{t_trt.min():.2f}--{t_trt.max():.2f}",
        "our_result": f"single-frame diff S/N median={np.median(diff_trt):.2f}, max={np.max(diff_trt):.2f}; 3σ limit median={np.median(lim_trt):.2f} mag",
        "published_reference": "GCN 43390: stacked R-band limits >19.6, >20.9, >21.0 (Vega)",
        "consistency_note": "Our single-frame data are shallower than the later stacked limits, but remain consistent with non-detection",
    })
    rows.append({
        "dataset": "TRT-SRO stacked",
        "band": "R",
        "time_window_hr": f"{hours_from_trigger(trt_stack['date_obs']):.2f}",
        "our_result": f"stacked diff S/N={trt_stack['diff_snr']:.2f}; 3σ limit={trt_stack['lim_mag_3sigma']:.2f} mag",
        "published_reference": "GCN 43390 first stacked limit: R > 19.6 (Vega)",
        "consistency_note": "Directly comparable same-facility stacked result; our 3σ limit is numerically consistent with the earliest reported TRT constraint",
    })
    rows.append({
        "dataset": "Deep external follow-up",
        "band": "r/z/H",
        "time_window_hr": "13.8--39.4",
        "our_result": "No strong late-time counterpart in current two-station dataset",
        "published_reference": "GCN 43391: NOT/GTC upper limits r>23.5, z>22.3, H>23.3, later r>24.2, z>22.7",
        "consistency_note": "External deeper upper limits reinforce the null-result interpretation",
    })
    with open(out_table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({
        "figure_xl100": str(out1),
        "figure_trt": str(out2),
        "consistency_table": str(out_table),
    }, indent=2))


if __name__ == "__main__":
    main()
