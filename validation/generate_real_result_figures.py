#!/usr/bin/env python3
"""
Generate figures from the real benchmark and end-to-end validation results.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DATA_ROOT, DOCS_ROOT, ensure_dir

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


DOCS = ensure_dir(DOCS_ROOT)
BENCH = DATA_ROOT / "paper_benchmarks_20260318a" / "benchmark_results_20260318a.csv"
E2E = DATA_ROOT / "paper_benchmarks_20260321a"


def load_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def median_wall(rows, case_name, tool):
    vals = [float(r["wall_s"]) for r in rows if r["case_name"] == case_name and r["tool"] == tool]
    return float(np.median(vals))


def fig_runtime_regimes():
    rows = load_csv(BENCH)
    desi_cases = ["desi100_o5", "desi100_o7", "desi100_o9"]
    desi_orders = [5, 7, 9]
    java_desi = [median_wall(rows, c, "java") for c in desi_cases]
    gpu_desi = [median_wall(rows, c, "gpu") for c in desi_cases]

    inc_cases = ["desi1_o7", "desi4_o7", "desi16_o7"]
    inc_labels = [1, 4, 16]
    java_inc = [median_wall(rows, c, "java") for c in inc_cases]
    gpu_inc = [median_wall(rows, c, "gpu") for c in inc_cases]

    ep_java = median_wall(rows, "ep1000_o3", "java")
    ep_gpu = median_wall(rows, "ep1000_o3", "gpu")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Bulk DESI
    ax = axes[0]
    x = np.arange(len(desi_orders))
    w = 0.36
    ax.bar(x - w/2, java_desi, width=w, color="#d87c7c", label="HipsGen")
    ax.bar(x + w/2, gpu_desi, width=w, color="#5b8fd1", label="GPU")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o}" for o in desi_orders])
    ax.set_xlabel("DESI Order")
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("Bulk Optical Generation")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    for i, (j, g) in enumerate(zip(java_desi, gpu_desi)):
        ax.text(i, max(j, g) + 0.8, f"{j/g:.2f}x", ha="center", va="bottom", fontsize=10)

    # Incremental latency
    ax = axes[1]
    x = np.arange(len(inc_labels))
    ax.plot(inc_labels, java_inc, "o-", color="#d87c7c", linewidth=2, label="HipsGen")
    ax.plot(inc_labels, gpu_inc, "s-", color="#5b8fd1", linewidth=2, label="GPU")
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("Incremental Epoch Latency")
    ax.grid(True, alpha=0.25)
    ax.set_xscale("log", base=2)
    ax.set_xticks(inc_labels)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    for n, j, g in zip(inc_labels, java_inc, gpu_inc):
        ax.text(n, max(j, g) + 0.25, f"{j/g:.2f}x", ha="center", va="bottom", fontsize=10)

    # EP summary
    ax = axes[2]
    labels = ["HipsGen", "GPU"]
    vals = [ep_java, ep_gpu]
    colors = ["#d87c7c", "#5b8fd1"]
    ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("High-Overlap EP WXT")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.5, max(vals)*1.08, f"{ep_java/ep_gpu:.1f}x", ha="center", va="bottom", fontsize=11)

    fig.suptitle("Measured Runtime Regimes of GPU HiPS", y=1.03, fontsize=15)
    fig.tight_layout()
    out = DOCS / "fig_runtime_regimes_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


def fig_end_to_end_validation():
    recovery_rows = load_csv(E2E / "recovery_summary.csv")
    phot_rows = load_csv(E2E / "photometry_summary.csv")
    snr_rows = load_csv(E2E / "combined_snr_summary.csv")

    snr = np.array([int(r["snr_ref"]) for r in recovery_rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Recovery curves
    ax = axes[0]
    for key, color, marker in [("A", "#1f77b4", "o"), ("B", "#ff7f0e", "s"), ("C", "#2ca02c", "^"), ("any_telescope", "#444444", "d")]:
        vals = [float(r[key]) for r in recovery_rows]
        ax.plot(snr, vals, marker=marker, color=color, linewidth=2, label=key.replace("_", " "))
    ax.set_xlabel("Injected Reference S/N")
    ax.set_ylabel("Recovery Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Original-Domain Recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    # Phot residuals
    ax = axes[1]
    tel = [r["telescope"] for r in phot_rows]
    med = np.array([float(r["median_residual"]) for r in phot_rows])
    sca = np.array([float(r["robust_scatter"]) for r in phot_rows])
    ax.bar(tel, med, yerr=sca, capsize=4, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(r"$(f_{\rm meas}-f_{\rm true})/f_{\rm true}$")
    ax.set_title("Forced-Photometry Residuals")
    ax.grid(True, axis="y", alpha=0.25)

    # Combined SNR
    ax = axes[2]
    snr2 = np.array([int(r["snr_ref"]) for r in snr_rows])
    for key, color, marker in [("A_only", "#1f77b4", "o"), ("A+B", "#ff7f0e", "s"), ("A+C", "#2ca02c", "^"), ("A+B+C", "#444444", "d")]:
        vals = [float(r[key]) for r in snr_rows]
        ax.plot(snr2, vals, marker=marker, color=color, linewidth=2, label=key.replace("_", " "))
    ax.set_xlabel("Injected Reference S/N")
    ax.set_ylabel("Measured Median S/N")
    ax.set_title("Flux-Space Combination")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle("End-to-End Validation with HpxFinder-Guided Original-Image Analysis", y=1.03, fontsize=15)
    fig.tight_layout()
    out = DOCS / "fig_end_to_end_validation_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


def fig_recovery_vs_snr():
    recovery_rows = load_csv(E2E / "recovery_summary.csv")
    snr = np.array([int(r["snr_ref"]) for r in recovery_rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for key, color, marker in [("A", "#1f77b4", "o"), ("B", "#ff7f0e", "s"), ("C", "#2ca02c", "^"), ("any_telescope", "#444444", "d")]:
        vals = [float(r[key]) for r in recovery_rows]
        ax.plot(snr, vals, marker=marker, color=color, linewidth=2, label=key.replace("_", " "))
    ax.set_xlabel("Injected Reference S/N")
    ax.set_ylabel("Recovery Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Recovery Fraction versus Injected S/N")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    out = DOCS / "fig_recovery_vs_snr_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


def fig_two_layer_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", linewidth=1.8)
        ax.add_patch(rect)
        ax.text(
            x + w/2,
            y + h/2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            linespacing=1.15,
        )

    # Inputs
    box(0.6, 7.7, 2.3, 1.2, "Reference Images", "#e8f1ff")
    box(0.6, 5.8, 2.3, 1.2, "Epoch Images", "#fff2df")
    box(0.6, 3.9, 2.3, 1.2, "GPU HiPS Generation", "#e8ffe8")

    # Layer 1
    box(3.6, 7.0, 2.6, 1.5, "Reference HiPS", "#dce9ff")
    box(3.6, 5.0, 2.6, 1.5, "Epoch HiPS", "#ffe7c7")
    box(6.9, 6.0, 3.0, 1.8, "Layer 1:\nSpatial Scoping\nwith HiPS/HpxFinder", "#f7f7f7")
    box(10.4, 6.0, 2.8, 1.8, "Original-Image\nDifference Cutouts", "#ffe6e6")

    # Layer 2
    box(3.6, 2.0, 2.6, 1.5, "HpxFinder Lookup", "#eef7ff")
    box(6.9, 2.0, 3.0, 1.5, "Layer 2:\nForced Photometry\non Original Images", "#eef9ee")
    box(10.4, 2.0, 2.8, 1.5, "Flux-Space\nCombination", "#fdf0ff")

    # Output
    box(6.2, 0.3, 4.0, 1.0, "Candidates, Light Curves, Multi-band Measurements", "#f4f4f4")

    arrows = [
        (2.9, 8.3, 0.7, -0.5),
        (2.9, 6.4, 0.7, -0.1),
        (2.0, 5.8, 0.0, -0.7),
        (6.2, 7.7, 0.7, -0.6),
        (6.2, 5.7, 0.7, 0.6),
        (9.7, 6.9, 0.7, 0.0),
        (8.3, 5.9, -0.8, -2.4),
        (3.0, 6.0, 0.6, 0.0),
        (6.2, 2.75, 0.7, 0.0),
        (9.7, 2.75, 0.7, 0.0),
        (8.3, 2.0, 0.0, -0.7),
    ]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))

    ax.text(8.3, 8.9, "Common sky grid for indexing and provenance", ha="center", fontsize=11)
    ax.text(8.3, 4.6, "Science-domain differencing remains in original images", ha="center", fontsize=11)

    out = DOCS / "fig_two_layer_architecture_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


def fig_difference_example():
    import json
    from astropy.io import fits
    from astropy.wcs import WCS

    rows = json.load(open(DATA_ROOT / "e2e_20260321a" / "photometry_records.json"))
    target = next(r for r in rows if r["telescope"] == "A" and r["snr_ref"] == 10 and r["recovered"])
    truth_id = target["truth_id"]

    summary = json.load(open(DATA_ROOT / "e2e_20260321a" / "recovery_records.json"))
    # Load original inputs
    ref_path = DATA_ROOT / "e2e_20260321a" / "A" / "reference_input" / "A_reference_r.fits"
    epoch_path = DATA_ROOT / "e2e_20260321a" / "A" / "epoch_input" / "A_epoch_r.fits"
    ref_data = fits.getdata(ref_path).astype(float)
    epoch_data = fits.getdata(epoch_path).astype(float)
    hdr = fits.getheader(epoch_path)
    wcs = WCS(hdr)

    # Find sky position from diff candidates JSON
    cands = json.load(open(DATA_ROOT / "e2e_20260321a" / "A" / "diff" / "candidates.json"))
    cand = next(c for c in cands if c["truth_id"] == truth_id)
    x, y = wcs.all_world2pix([[cand["ra"], cand["dec"]]], 0)[0]
    x = int(round(x))
    y = int(round(y))
    hs = 25
    ref_cut = ref_data[y-hs:y+hs+1, x-hs:x+hs+1]
    epoch_cut = epoch_data[y-hs:y+hs+1, x-hs:x+hs+1]
    diff_cut = epoch_cut - ref_cut

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    panels = [
        (ref_cut, "Reference Image"),
        (epoch_cut, "Epoch Image"),
        (diff_cut, "Original-Domain Difference"),
    ]
    for ax, (img, title) in zip(axes, panels):
        v1, v2 = np.percentile(img[np.isfinite(img)], [5, 99]) if np.isfinite(img).any() else (0, 1)
        ax.imshow(img, origin="lower", cmap="gray", vmin=v1, vmax=v2)
        ax.scatter([hs], [hs], marker="+", s=120, c="red", linewidths=2)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    out = DOCS / "fig_difference_example_real.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


def fig_photometry_residual_distribution():
    import json
    rows = json.load(open(DATA_ROOT / "e2e_20260321a" / "photometry_records.json"))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    bins = np.linspace(-1.0, 1.0, 40)
    for tel in ["A", "B", "C"]:
        vals = [r["flux_residual"] for r in rows if r["telescope"] == tel and r["flux_residual"] is not None and abs(r["flux_residual"]) < 1.0]
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, label=tel, color=colors[tel], density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(r"$(f_{\rm meas}-f_{\rm true})/f_{\rm true}$")
    ax.set_ylabel("Normalized Count")
    ax.set_title("Photometric Residual Distributions")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    out = DOCS / "fig_photometry_residual_distribution_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


def fig_multistation_snr_gain():
    snr_rows = load_csv(E2E / "combined_snr_summary.csv")
    snr = np.array([int(r["snr_ref"]) for r in snr_rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for key, color, marker in [("A_only", "#1f77b4", "o"), ("A+B", "#ff7f0e", "s"), ("A+C", "#2ca02c", "^"), ("A+B+C", "#444444", "d")]:
        vals = [float(r[key]) for r in snr_rows]
        ax.plot(snr, vals, marker=marker, color=color, linewidth=2, label=key.replace("_", " "))
    ax.set_xlabel("Injected Reference S/N")
    ax.set_ylabel("Measured Median S/N")
    ax.set_title("Multi-telescope S/N Enhancement")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    out = DOCS / "fig_multistation_snr_gain_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


if __name__ == "__main__":
    fig_two_layer_architecture()
    fig_runtime_regimes()
    fig_difference_example()
    fig_recovery_vs_snr()
    fig_end_to_end_validation()
    fig_photometry_residual_distribution()
    fig_multistation_snr_gain()
