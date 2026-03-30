#!/usr/bin/env python3
"""
EP260110a real-event case study.

Goal:
  - organize two-station local data (XL100 and TRT-SRO),
  - build matched public references,
  - run HpxFinder-guided original-domain differencing and photometry,
  - compare TRT single-frame vs stacked performance,
  - produce paper-ready tables and plots.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "runtime") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "runtime"))

from project_paths import DATA_ROOT, OUTPUT_ROOT, ensure_dir

from hips_processor import run_local_hips, diff_hips  # type: ignore
from photometry import run_photometry, aperture_phot, estimate_background  # type: ignore


EVENT_ID = "EP260110a"
# Use the refined EP/FXT position reported in GCN 43367
RA = 199.4321
DEC = 65.8490
TRIGGER_UTC = "2026-01-10T11:58:23"

LOCAL_XL100 = Path(os.environ.get("HIPS_GPU_EP260110A_XL100_DIR", DATA_ROOT / "ep260110a_raw" / "XL100"))
LOCAL_TRT = Path(os.environ.get("HIPS_GPU_EP260110A_TRT_DIR", DATA_ROOT / "ep260110a_raw" / "TRT-SRO"))
OUT = Path(os.environ.get("HIPS_GPU_EP260110A_OUTPUT_DIR", OUTPUT_ROOT / "case_studies" / "EP260110a"))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path):
    ensure_dir(dest.parent)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def query_ps1_http(ra: float, dec: float, radius_deg: float, band: str):
    col = {"g": "gMeanPSFMag", "r": "rMeanPSFMag", "R": "rMeanPSFMag"}.get(band, "rMeanPSFMag")
    url = (
        "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv?"
        + urllib.parse.urlencode(
            {
                "ra": ra,
                "dec": dec,
                "radius": radius_deg,
                "nDetections.gte": 1,
                "columns": f"[raMean,decMean,{col}]",
            }
        )
    )
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    out = []
    for r in rows:
        try:
            mag = float(r[col])
            ra0 = float(r["raMean"])
            dec0 = float(r["decMean"])
        except Exception:
            continue
        if not np.isfinite(mag) or mag <= 0 or mag > 21 or mag < 14:
            continue
        out.append({"ra": ra0, "dec": dec0, "mag": mag})
    return out


def estimate_zeropoint_http(data: np.ndarray, header, band: str, ra0: float, dec0: float):
    stars = query_ps1_http(ra0, dec0, 0.15, band)
    if not stars:
        return None, 0
    w = WCS(header)
    data_sub, _, _ = estimate_background(data.astype(float))
    zps = []
    for star in stars:
        try:
            x, y = w.all_world2pix([[star["ra"], star["dec"]]], 0)[0]
        except Exception:
            continue
        if not (12 < x < data.shape[1] - 12 and 12 < y < data.shape[0] - 12):
            continue
        flux, _ = aperture_phot(data_sub, x, y)
        if flux is None or not np.isfinite(flux) or flux <= 0:
            continue
        inst_mag = -2.5 * np.log10(flux)
        zps.append(star["mag"] - inst_mag)
    if len(zps) < 3:
        return None, len(zps)
    zps = np.array(zps, dtype=float)
    med = np.median(zps)
    mad = 1.4826 * np.median(np.abs(zps - med))
    good = np.abs(zps - med) < 3 * max(mad, 0.05)
    zps = zps[good]
    if len(zps) < 3:
        return float(np.median(zps)), len(zps)
    return float(np.median(zps)), len(zps)


def robust_stats(values):
    vals = np.array(values, dtype=float)
    med = float(np.median(vals))
    mad = float(1.4826 * np.median(np.abs(vals - med)))
    return med, mad


def bilinear_sample(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    valid = (x >= 0) & (x < data.shape[1] - 1) & (y >= 0) & (y < data.shape[0] - 1)
    if not np.any(valid):
        return out
    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(int)
    y0 = np.floor(yv).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xv - x0
    wy = yv - y0
    v00 = data[y0, x0]
    v10 = data[y0, x1]
    v01 = data[y1, x0]
    v11 = data[y1, x1]
    out[valid] = (
        (1 - wx) * (1 - wy) * v00 +
        wx * (1 - wy) * v10 +
        (1 - wx) * wy * v01 +
        wx * wy * v11
    )
    return out


def target_centered_difference(epoch_file: Path, reference_file: Path, ra: float, dec: float, half_size: int = 25):
    epoch_data = fits.getdata(epoch_file).astype(float)
    ref_data = fits.getdata(reference_file).astype(float)
    eh = fits.getheader(epoch_file)
    rh = fits.getheader(reference_file)
    ew = WCS(eh)
    rw = WCS(rh)
    ex, ey = ew.all_world2pix([[ra, dec]], 0)[0]
    cx = int(round(ex))
    cy = int(round(ey))
    x0 = max(0, cx - half_size)
    x1 = min(epoch_data.shape[1], cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(epoch_data.shape[0], cy + half_size + 1)
    epoch_cut = epoch_data[y0:y1, x0:x1]

    yy, xx = np.indices(epoch_cut.shape, dtype=float)
    world = ew.all_pix2world(np.column_stack([(x0 + xx).ravel(), (y0 + yy).ravel()]), 0)
    ref_xy = rw.all_world2pix(world, 0)
    ref_cut = bilinear_sample(ref_data, ref_xy[:, 0].reshape(epoch_cut.shape), ref_xy[:, 1].reshape(epoch_cut.shape))
    diff = epoch_cut - ref_cut
    diff_sub, _, _ = estimate_background(diff)
    tx = ex - x0
    ty = ey - y0
    flux, ferr = aperture_phot(diff_sub, tx, ty)
    snr = flux / ferr if ferr and ferr > 0 else None
    return {
        "diff_flux": float(flux) if flux is not None else None,
        "diff_flux_err": float(ferr) if ferr is not None else None,
        "diff_snr": float(snr) if snr is not None else None,
    }


def build_references(refdir: Path):
    ensure_dir(refdir)
    urls = {
        "desi_g_ref.fits": f"https://www.legacysurvey.org/viewer/fits-cutout?ra={RA}&dec={DEC}&layer=ls-dr10&pixscale=0.262&bands=g",
        "desi_r_ref.fits": f"https://www.legacysurvey.org/viewer/fits-cutout?ra={RA}&dec={DEC}&layer=ls-dr10&pixscale=0.262&bands=r",
        "ps1_g_ref.fits": "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra=199.4060&dec=65.8450&size=240&format=fits&red=/rings.v3.skycell/2521/048/rings.v3.skycell.2521.048.stk.g.unconv.fits",
        "ps1_r_ref.fits": "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra=199.4060&dec=65.8450&size=240&format=fits&red=/rings.v3.skycell/2521/048/rings.v3.skycell.2521.048.stk.r.unconv.fits",
    }
    for name, url in urls.items():
        path = refdir / name
        if not path.exists():
            download(url, path)
    return {
        "XL100": refdir / "desi_g_ref.fits",
        "TRT-SRO": refdir / "desi_r_ref.fits",
    }


def inventory_station(station: str, station_dir: Path):
    rows = []
    for f in sorted(station_dir.glob("*.fits")):
        h = fits.getheader(f)
        d = fits.getdata(f)
        w = WCS(h)
        x, y = w.all_world2pix([[RA, DEC]], 0)[0]
        rows.append({
            "station": station,
            "file": f.name,
            "date_obs": h.get("DATE-OBS"),
            "filter": h.get("FILTER") or h.get("FILTNAM") or h.get("FILTER1"),
            "shape_x": d.shape[1],
            "shape_y": d.shape[0],
            "x": float(x),
            "y": float(y),
            "in_fov": bool(0 <= x < d.shape[1] and 0 <= y < d.shape[0]),
            "ra_center": h.get("CRVAL1"),
            "dec_center": h.get("CRVAL2"),
            "exptime": h.get("EXPTIME"),
        })
    return rows


def stack_images(files: list[Path], out_path: Path):
    datas = []
    hdr0 = None
    for f in files:
        if hdr0 is None:
            hdr0 = fits.getheader(f)
        datas.append(fits.getdata(f).astype(np.float64))
    stack = np.median(np.array(datas), axis=0)
    fits.PrimaryHDU(data=stack.astype(np.float32), header=hdr0).writeto(out_path, overwrite=True)


def measure_latency(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_station_case(station: str, epoch_files: list[Path], ref_file: Path, outdir: Path):
    ensure_dir(outdir)
    epoch_input = outdir / "epoch_input"
    reference_input = outdir / "reference_input"
    epoch_hips = outdir / "epoch_hips"
    reference_hips = outdir / "reference_hips"
    diff_dir = outdir / "diff"
    ensure_dir(epoch_input)
    ensure_dir(reference_input)

    for f in epoch_files:
        shutil.copy2(f, epoch_input / f.name)
    shutil.copy2(ref_file, reference_input / ref_file.name)

    _, t_epoch_hips = measure_latency(run_local_hips, str(epoch_input), str(epoch_hips), order=7, mode="MEAN", timeout=300)
    _, t_ref_hips = measure_latency(run_local_hips, str(reference_input), str(reference_hips), order=7, mode="MEAN", timeout=300)

    candidate_positions = [{"truth_id": 0, "ra": RA, "dec": DEC, "snr_ref": 0, "flux": 0.0}]
    diff_res, t_diff = measure_latency(
        diff_hips,
        event_id=EVENT_ID,
        tid=station,
        night="case",
        band="g" if station == "XL100" else "r",
        candidate_positions=candidate_positions,
        reference_local_dir=str(reference_hips),
        epoch_local_dir=str(epoch_hips),
        reference_source_dir=str(reference_input),
        epoch_source_dir=str(epoch_input),
        output_local_dir=str(diff_dir),
        threshold_sigma=5.0,
        cutout_half_size=25,
        detection_radius=4,
        cleanup=False,
    )

    phot_rows = []
    phot_times = []
    for f in epoch_files:
        data = fits.getdata(f)
        hdr = fits.getheader(f)
        band = "g" if station == "XL100" else "r"
        res, dt = measure_latency(run_photometry, data, hdr, RA, DEC, band, None, False)
        diff_stats = target_centered_difference(f, ref_file, RA, DEC)
        zp, n_ref = estimate_zeropoint_http(data, hdr, band, RA, DEC)
        mag = None
        lim_mag = None
        if zp is not None and res.flux is not None and np.isfinite(res.flux) and res.flux > 0:
            mag = -2.5 * np.log10(res.flux) + zp
        if zp is not None and res.flux_err is not None and np.isfinite(res.flux_err) and res.flux_err > 0:
            lim_mag = -2.5 * np.log10(3.0 * res.flux_err) + zp
        phot_times.append(dt)
        phot_rows.append({
            "station": station,
            "file": f.name,
            "date_obs": hdr.get("DATE-OBS"),
            "flux": res.flux,
            "flux_err": res.flux_err,
            "snr": res.snr,
            "mag": mag,
            "mag_err": res.mag_err,
            "lim_mag_3sigma": lim_mag,
            "zp": zp,
            "n_ref_stars": n_ref,
            **diff_stats,
        })

    med_phot_ms = float(np.median(phot_times) * 1000.0) if phot_times else None
    candidate = diff_res["candidates"][0]
    diff_snrs = [r["diff_snr"] for r in phot_rows if r.get("diff_snr") is not None]
    return {
        "station": station,
        "n_epoch_files": len(epoch_files),
        "reference_file": ref_file.name,
        "epoch_hips_s": t_epoch_hips,
        "reference_hips_s": t_ref_hips,
        "diff_s": t_diff,
        "median_phot_ms": med_phot_ms,
        "recovered": candidate.get("recovered"),
        "peak_snr": candidate.get("peak_snr"),
        "aperture_snr": candidate.get("aperture_snr"),
        "target_centered_diff_snr_median": float(np.median(diff_snrs)) if diff_snrs else None,
        "target_centered_diff_snr_max": float(np.max(diff_snrs)) if diff_snrs else None,
        "target_centered_detected": bool(any(s is not None and s >= 5.0 for s in diff_snrs)),
        "det_ra": candidate.get("det_ra"),
        "det_dec": candidate.get("det_dec"),
        "phot_rows": phot_rows,
    }


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dir(OUT)

    refs = build_references(OUT / "reference")

    inventory = []
    inventory.extend(inventory_station("XL100", LOCAL_XL100))
    inventory.extend(inventory_station("TRT-SRO", LOCAL_TRT))

    with (OUT / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
        w.writeheader()
        w.writerows(inventory)

    # XL100: night-by-night single-epoch handling
    xl_files = sorted(LOCAL_XL100.glob("*.fits"))
    xl_case = run_station_case("XL100", xl_files, refs["XL100"], OUT / "XL100_case")

    # TRT: single-frame and stacked comparison
    trt_files = sorted(LOCAL_TRT.glob("*.fits"))
    trt_single = run_station_case("TRT-SRO", trt_files, refs["TRT-SRO"], OUT / "TRT_single_case")
    stacked = OUT / "TRT_stack.fits"
    stack_images(trt_files, stacked)
    trt_stack = run_station_case("TRT-SRO_stack", [stacked], refs["TRT-SRO"], OUT / "TRT_stack_case")

    summary = {
        "event_id": EVENT_ID,
        "ra_deg": RA,
        "dec_deg": DEC,
        "trigger_utc": TRIGGER_UTC,
        "inventory_rows": len(inventory),
        "reference_files": {k: str(v) for k, v in refs.items()},
        "xl100_case": xl_case,
        "trt_single_case": trt_single,
        "trt_stack_case": trt_stack,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
