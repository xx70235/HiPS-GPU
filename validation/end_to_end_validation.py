#!/usr/bin/env python3
"""
End-to-end validation for the Transient Hub + hips-gpu workflow.

Design principle:
  - HiPS/HpxFinder are used for spatial indexing and provenance lookup.
  - Difference imaging and photometry are evaluated in the original-image domain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "runtime") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "runtime"))

from project_paths import DATA_ROOT, OUTPUT_ROOT

from hips_processor import run_local_hips, diff_hips
from photometry import run_photometry, combine_flux_measurements


@dataclass
class TelescopeConfig:
    name: str
    blur_sigma_pix: float
    extra_noise_sigma: float


def robust_scatter(values: np.ndarray) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def solve_flux_for_snr(target_snr: float, sigma_bkg: float, aperture_radius: float = 8.0) -> float:
    n_ap = math.pi * aperture_radius ** 2
    a = 1.0
    b = -(target_snr ** 2)
    c = -(target_snr ** 2) * n_ap * (sigma_bkg ** 2)
    disc = b * b - 4 * a * c
    return float(0.5 * (-b + math.sqrt(max(disc, 0.0))))


def choose_positions(data: np.ndarray, n_total: int, rng: np.random.Generator,
                     margin: int = 40, min_sep: int = 24) -> list[tuple[float, float]]:
    ny, nx = data.shape
    median = np.median(data)
    std = robust_scatter(data.astype(float))
    ok = np.abs(data - median) < 1.5 * max(std, 1e-6)
    coords = np.argwhere(ok)
    coords = coords[
        (coords[:, 0] > margin) & (coords[:, 0] < ny - margin) &
        (coords[:, 1] > margin) & (coords[:, 1] < nx - margin)
    ]
    if len(coords) < n_total:
        coords = np.argwhere(np.ones_like(data, dtype=bool))
        coords = coords[
            (coords[:, 0] > margin) & (coords[:, 0] < ny - margin) &
            (coords[:, 1] > margin) & (coords[:, 1] < nx - margin)
        ]

    rng.shuffle(coords)
    chosen = []
    for y, x in coords:
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_sep ** 2 for cx, cy in chosen):
            chosen.append((float(x), float(y)))
            if len(chosen) == n_total:
                break
    if len(chosen) < n_total:
        raise RuntimeError(f"Could not place {n_total} separated injection positions")
    return chosen


def choose_positions_per_snr(data: np.ndarray, snr_levels: list[int], n_per_snr: int,
                             rng: np.random.Generator, margin: int = 30,
                             min_sep: int = 14) -> list[tuple[float, float]]:
    """
    A denser placement strategy for larger validation sets.
    Positions remain separated, but the minimum spacing is relaxed enough
    to allow a few hundred injections on a 512x512 frame.
    """
    out = []
    used = []
    ny, nx = data.shape
    median = np.median(data)
    std = robust_scatter(data.astype(float))
    ok = np.abs(data - median) < 1.8 * max(std, 1e-6)
    coords = np.argwhere(ok)
    coords = coords[
        (coords[:, 0] > margin) & (coords[:, 0] < ny - margin) &
        (coords[:, 1] > margin) & (coords[:, 1] < nx - margin)
    ]
    if len(coords) == 0:
        raise RuntimeError("No valid pixels available for injection placement")

    for _snr in snr_levels:
        rng.shuffle(coords)
        chosen = 0
        for y, x in coords:
            if all((x - cx) ** 2 + (y - cy) ** 2 >= min_sep ** 2 for cx, cy in used):
                used.append((float(x), float(y)))
                out.append((float(x), float(y)))
                chosen += 1
                if chosen == n_per_snr:
                    break
        if chosen < n_per_snr:
            raise RuntimeError(
                f"Could not place {n_per_snr} positions for one S/N level "
                f"with margin={margin}, min_sep={min_sep}"
            )
    return out


def inject_gaussian_sources(data: np.ndarray, sources: list[dict], psf_sigma_pix: float) -> np.ndarray:
    out = data.copy()
    yy, xx = np.indices(data.shape)
    norm = 1.0 / (2.0 * math.pi * psf_sigma_pix ** 2)
    for src in sources:
        amp = src["flux"] * norm
        g = amp * np.exp(-((xx - src["x"]) ** 2 + (yy - src["y"]) ** 2) / (2 * psf_sigma_pix ** 2))
        out += g
    return out


def make_variant(base_data: np.ndarray, cfg: TelescopeConfig, rng: np.random.Generator) -> np.ndarray:
    out = base_data.astype(np.float64, copy=True)
    if cfg.blur_sigma_pix > 0:
        kernel = Gaussian2DKernel(cfg.blur_sigma_pix)
        out = convolve(out, kernel, boundary="extend", normalize_kernel=True)
    if cfg.extra_noise_sigma > 0:
        out = out + rng.normal(0.0, cfg.extra_noise_sigma, size=out.shape)
    return out


def write_fits(path: Path, data: np.ndarray, header):
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(path, overwrite=True)


def aggregate_recovery(records: list[dict], tel_names: list[str], snr_levels: list[int]) -> list[dict]:
    out = []
    for snr in snr_levels:
        rec = {"snr_ref": snr}
        any_count = 0
        total = 0
        for tel in tel_names:
            subset = [r for r in records if r["telescope"] == tel and r["snr_ref"] == snr]
            total = len(subset)
            k = sum(1 for r in subset if r["recovered"])
            val = k / total if total else math.nan
            lo, hi = wilson_interval(k, total)
            rec[tel] = val
            rec[f"{tel}_ci_low"] = lo
            rec[f"{tel}_ci_high"] = hi
        truth_ids = sorted({r["truth_id"] for r in records if r["snr_ref"] == snr})
        for tid in truth_ids:
            per_src = [r for r in records if r["snr_ref"] == snr and r["truth_id"] == tid]
            if any(r["recovered"] for r in per_src):
                any_count += 1
        any_val = any_count / len(truth_ids) if truth_ids else math.nan
        rec["any_telescope"] = any_val
        lo, hi = wilson_interval(any_count, len(truth_ids))
        rec["any_ci_low"] = lo
        rec["any_ci_high"] = hi
        out.append(rec)
    return out


def run_false_positive_validation(base_data: np.ndarray, base_hdr, bkg_rms: float,
                                  outdir: Path, order: int, seed: int,
                                  n_trials: int, tel_configs: list[TelescopeConfig]) -> list[dict]:
    """
    Run no-injection validation to estimate false detections.
    Each trial uses a fresh reference and epoch realization with no injected sources.
    """
    results = []
    for cfg in tel_configs:
        tel_dir = outdir / f"falsepos_{cfg.name}"
        for i in range(n_trials):
            ref_input = tel_dir / f"trial_{i}" / "reference_input"
            epoch_input = tel_dir / f"trial_{i}" / "epoch_input"
            ref_hips = tel_dir / f"trial_{i}" / "reference_hips"
            epoch_hips = tel_dir / f"trial_{i}" / "epoch_hips"
            diff_out = tel_dir / f"trial_{i}" / "diff"
            ref_input.mkdir(parents=True, exist_ok=True)
            epoch_input.mkdir(parents=True, exist_ok=True)

            ref_data = make_variant(base_data, cfg, np.random.default_rng(seed + 1000 + i * 17 + hash(cfg.name) % 1000))
            epoch_data = make_variant(base_data, cfg, np.random.default_rng(seed + 2000 + i * 19 + hash(cfg.name) % 1000))

            ref_path = ref_input / f"{cfg.name}_reference_r.fits"
            epoch_path = epoch_input / f"{cfg.name}_epoch_r.fits"
            write_fits(ref_path, ref_data, base_hdr)
            write_fits(epoch_path, epoch_data, base_hdr)

            run_local_hips(str(ref_input), str(ref_hips), order=order, mode="MEAN", timeout=300)
            run_local_hips(str(epoch_input), str(epoch_hips), order=order, mode="MEAN", timeout=300)

            # Choose random blank-ish positions to test for false detections
            positions = choose_positions(base_data, n_total=10, rng=np.random.default_rng(seed + 3000 + i), margin=40, min_sep=20)
            candidate_positions = []
            wcs = WCS(base_hdr)
            for j, (x, y) in enumerate(positions):
                ra, dec = wcs.all_pix2world([[x, y]], 0)[0]
                candidate_positions.append({
                    "truth_id": j,
                    "x": x,
                    "y": y,
                    "ra": float(ra),
                    "dec": float(dec),
                    "snr_ref": -1,
                    "flux": 0.0,
                })

            diff_res = diff_hips(
                event_id="SIMFP",
                tid=cfg.name,
                night="20260321",
                band="r",
                candidate_positions=candidate_positions,
                reference_local_dir=str(ref_hips),
                epoch_local_dir=str(epoch_hips),
                reference_source_dir=str(ref_input),
                epoch_source_dir=str(epoch_input),
                output_local_dir=str(diff_out),
                threshold_sigma=5.0,
                cutout_half_size=25,
                detection_radius=4,
                cleanup=False,
            )
            false_count = sum(1 for c in diff_res["candidates"] if c.get("recovered"))
            results.append({
                "telescope": cfg.name,
                "trial": i,
                "n_test_positions": len(candidate_positions),
                "n_false_detections": false_count,
                "false_detection_rate": false_count / len(candidate_positions),
            })
    return results


def main():
    p = argparse.ArgumentParser(description="Run end-to-end original-domain validation")
    p.add_argument(
        "--base-fits",
        default=str(DATA_ROOT / "e2e_20260321a" / "A" / "reference_input" / "A_reference_r.fits"),
    )
    p.add_argument("--outdir", default=str(OUTPUT_ROOT / "validation_runs" / "20260319a"))
    p.add_argument("--order", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-snr", type=int, default=10)
    p.add_argument("--false-positive-trials", type=int, default=10)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    base_data = fits.getdata(args.base_fits).astype(np.float64)
    base_hdr = fits.getheader(args.base_fits)
    wcs = WCS(base_hdr)
    _, bkg_med, bkg_rms = sigma_clipped_stats(base_data, sigma=3.0)
    base_data = base_data - bkg_med

    snr_levels = [3, 5, 7, 10, 20, 50]
    n_total = len(snr_levels) * args.n_per_snr
    positions = choose_positions_per_snr(base_data, snr_levels, args.n_per_snr, rng=rng)

    psf_sigma_pix = 1.4
    flux_lookup = {snr: solve_flux_for_snr(snr, bkg_rms, aperture_radius=8.0) for snr in snr_levels}
    truth_sources = []
    for i, (x, y) in enumerate(positions):
        snr_ref = snr_levels[i // args.n_per_snr]
        ra, dec = wcs.all_pix2world([[x, y]], 0)[0]
        truth_sources.append({
            "truth_id": i,
            "x": x,
            "y": y,
            "ra": float(ra),
            "dec": float(dec),
            "snr_ref": snr_ref,
            "flux": flux_lookup[snr_ref],
        })

    tel_configs = [
        TelescopeConfig("A", blur_sigma_pix=0.0, extra_noise_sigma=float(bkg_rms)),
        TelescopeConfig("B", blur_sigma_pix=0.8, extra_noise_sigma=float(1.5 * bkg_rms)),
        TelescopeConfig("C", blur_sigma_pix=0.0, extra_noise_sigma=float(2.5 * bkg_rms)),
    ]

    recovery_records = []
    phot_records = []
    snr_rows = []

    for cfg in tel_configs:
        tel_dir = outdir / cfg.name
        ref_input = tel_dir / "reference_input"
        epoch_input = tel_dir / "epoch_input"
        ref_hips = tel_dir / "reference_hips"
        epoch_hips = tel_dir / "epoch_hips"
        diff_out = tel_dir / "diff"
        ref_input.mkdir(parents=True, exist_ok=True)
        epoch_input.mkdir(parents=True, exist_ok=True)

        ref_data = make_variant(base_data, cfg, np.random.default_rng(args.seed + hash(cfg.name) % 1000 + 1))
        epoch_data = make_variant(base_data, cfg, np.random.default_rng(args.seed + hash(cfg.name) % 1000 + 2))
        epoch_data = inject_gaussian_sources(epoch_data, truth_sources, psf_sigma_pix=psf_sigma_pix + cfg.blur_sigma_pix)

        ref_path = ref_input / f"{cfg.name}_reference_r.fits"
        epoch_path = epoch_input / f"{cfg.name}_epoch_r.fits"
        write_fits(ref_path, ref_data, base_hdr)
        write_fits(epoch_path, epoch_data, base_hdr)

        run_local_hips(str(ref_input), str(ref_hips), order=args.order, mode="MEAN", timeout=300)
        run_local_hips(str(epoch_input), str(epoch_hips), order=args.order, mode="MEAN", timeout=300)

        diff_res = diff_hips(
            event_id="SIM",
            tid=cfg.name,
            night="20260319",
            band="r",
            candidate_positions=truth_sources,
            reference_local_dir=str(ref_hips),
            epoch_local_dir=str(epoch_hips),
            reference_source_dir=str(ref_input),
            epoch_source_dir=str(epoch_input),
            output_local_dir=str(diff_out),
            threshold_sigma=5.0,
            cutout_half_size=25,
            detection_radius=4,
            cleanup=False,
        )

        by_truth = {r["truth_id"]: r for r in diff_res["candidates"]}
        for truth in truth_sources:
            rec = by_truth[truth["truth_id"]]
            recovery_records.append({
                "telescope": cfg.name,
                "truth_id": truth["truth_id"],
                "snr_ref": truth["snr_ref"],
                "recovered": bool(rec.get("recovered")),
                "peak_snr": rec.get("peak_snr"),
                "aperture_snr": rec.get("aperture_snr"),
                "peak": rec.get("peak"),
                "sigma": rec.get("sigma"),
                "det_ra": rec.get("det_ra"),
                "det_dec": rec.get("det_dec"),
            })

            phot = run_photometry(
                epoch_data, base_hdr,
                target_ra=truth["ra"],
                target_dec=truth["dec"],
                filter_name="r",
                seeing=None,
                calibrate=False,
            )
            residual = None
            if phot.flux is not None and truth["flux"] > 0:
                residual = (phot.flux - truth["flux"]) / truth["flux"]
            phot_records.append({
                "telescope": cfg.name,
                "truth_id": truth["truth_id"],
                "snr_ref": truth["snr_ref"],
                "flux_true": truth["flux"],
                "flux_measured": phot.flux,
                "flux_err": phot.flux_err,
                "snr_measured": phot.snr,
                "flux_residual": residual,
                "recovered": bool(rec.get("recovered")),
            })

    # Combined SNR tables
    by_truth_tel = {}
    for row in phot_records:
        by_truth_tel.setdefault(row["truth_id"], {})[row["telescope"]] = row

    combos = {
        "A_only": ["A"],
        "A+B": ["A", "B"],
        "A+C": ["A", "C"],
        "A+B+C": ["A", "B", "C"],
    }
    combined_summary = []
    for snr in snr_levels:
        rows = {"snr_ref": snr}
        truth_ids = [t["truth_id"] for t in truth_sources if t["snr_ref"] == snr]
        for label, members in combos.items():
            vals = []
            for tid in truth_ids:
                res = []
                for tel in members:
                    r = by_truth_tel.get(tid, {}).get(tel)
                    if r and r["flux_measured"] is not None and r["flux_err"] is not None and r["flux_err"] > 0:
                        class _Obj:
                            flux = r["flux_measured"]
                            flux_err = r["flux_err"]
                        res.append(_Obj())
                comb = combine_flux_measurements(res)
                if comb.snr is not None:
                    vals.append(comb.snr)
            rows[label] = float(np.median(vals)) if vals else math.nan
        combined_summary.append(rows)

    phot_summary = []
    for tel in [c.name for c in tel_configs]:
        vals = [r["flux_residual"] for r in phot_records if r["telescope"] == tel and r["flux_residual"] is not None]
        phot_summary.append({
            "telescope": tel,
            "median_residual": float(np.median(vals)),
            "robust_scatter": robust_scatter(np.array(vals, dtype=float)),
        })

    recovery_summary = aggregate_recovery(recovery_records, [c.name for c in tel_configs], snr_levels)

    summary = {
        "config": {
            "base_fits": args.base_fits,
            "order": args.order,
            "seed": args.seed,
            "n_per_snr": args.n_per_snr,
            "snr_levels": snr_levels,
            "background_rms": float(bkg_rms),
            "psf_sigma_pix": psf_sigma_pix,
        },
        "recovery_summary": recovery_summary,
        "photometry_summary": phot_summary,
        "combined_snr_summary": combined_summary,
    }

    false_positive_records = run_false_positive_validation(
        base_data, base_hdr, bkg_rms, outdir, args.order, args.seed, args.false_positive_trials, tel_configs
    )
    fp_summary = []
    for tel in [c.name for c in tel_configs]:
        vals = [r["false_detection_rate"] for r in false_positive_records if r["telescope"] == tel]
        fp_summary.append({
            "telescope": tel,
            "mean_false_detection_rate": float(np.mean(vals)),
            "median_false_detection_rate": float(np.median(vals)),
            "max_false_detection_rate": float(np.max(vals)),
        })
    summary["false_positive_summary"] = fp_summary

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "recovery_records.json").write_text(json.dumps(recovery_records, indent=2))
    (outdir / "photometry_records.json").write_text(json.dumps(phot_records, indent=2))
    (outdir / "false_positive_records.json").write_text(json.dumps(false_positive_records, indent=2))

    with (outdir / "recovery_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "snr_ref",
                "A", "A_ci_low", "A_ci_high",
                "B", "B_ci_low", "B_ci_high",
                "C", "C_ci_low", "C_ci_high",
                "any_telescope", "any_ci_low", "any_ci_high",
            ],
        )
        w.writeheader()
        w.writerows(recovery_summary)

    with (outdir / "photometry_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["telescope", "median_residual", "robust_scatter"])
        w.writeheader()
        w.writerows(phot_summary)

    with (outdir / "combined_snr_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["snr_ref", "A_only", "A+B", "A+C", "A+B+C"])
        w.writeheader()
        w.writerows(combined_summary)

    with (outdir / "false_positive_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["telescope", "mean_false_detection_rate", "median_false_detection_rate", "max_false_detection_rate"])
        w.writeheader()
        w.writerows(fp_summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
