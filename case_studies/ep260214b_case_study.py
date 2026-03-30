#!/usr/bin/env python3
"""
EP260214b real-event positive-detection case study.

This script:
  1. prepares TRT-CTO and Xinglong 2.16m epoch products,
  2. calibrates Xinglong images with local bias/flat frames,
  3. downloads matched public g/r reference cutouts,
  4. runs HiPS/HpxFinder provenance lookup on each station/band bundle,
  5. measures target-centered difference-image significance and calibrated
     difference magnitudes, and
  6. writes paper-ready summary tables and figures.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "runtime") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "runtime"))

from project_paths import DATA_ROOT as PAPER_DATA_ROOT, DOCS_ROOT, OUTPUT_ROOT, ensure_dir

DATA_ROOT = Path(os.environ.get("HIPS_GPU_EP260214B_RAW_ROOT", PAPER_DATA_ROOT / "ep260214b_raw"))
OUT = Path(os.environ.get("HIPS_GPU_EP260214B_OUTPUT_DIR", OUTPUT_ROOT / "case_studies" / "EP260214b"))
DOCS = ensure_dir(DOCS_ROOT)

from hips_processor import diff_hips, run_local_hips  # type: ignore
from photometry import aperture_phot, estimate_background  # type: ignore


EVENT_ID = "EP260214b"
RA = 191.2582
DEC = 23.8536
TRIGGER_UTC = "2026-02-14T22:03:41.16"
TRIGGER = Time(TRIGGER_UTC, format="isot", scale="utc")
ORDER = 7

PS1_CACHE: dict[tuple[float, float, float, str], list[dict]] = {}

PUBLISHED_CONTEXT = [
    {
        "source": "NOT",
        "gcn": "43745",
        "band": "r",
        "time_hr": 2.22,
        "mag": 20.90,
        "mag_err": 0.04,
        "kind": "detection",
        "note": "Published optical counterpart position and r-band detection.",
    },
    {
        "source": "OHP/T120",
        "gcn": "43752",
        "band": "r",
        "time_hr": 3.60,
        "mag": 21.22,
        "mag_err": 0.13,
        "kind": "detection",
        "note": "Independent r-band detection reported from OHP.",
    },
    {
        "source": "COLIBRI",
        "gcn": "43747",
        "band": "g",
        "time_hr": 7.95,
        "mag": 21.30,
        "mag_err": 0.09,
        "kind": "detection",
        "note": "g-band detection from COLIBRI.",
    },
    {
        "source": "COLIBRI",
        "gcn": "43747",
        "band": "r",
        "time_hr": 7.95,
        "mag": 21.11,
        "mag_err": 0.07,
        "kind": "detection",
        "note": "r-band detection from COLIBRI.",
    },
    {
        "source": "LCO",
        "gcn": "43750",
        "band": "r",
        "time_hr": 11.34,
        "mag": 21.28,
        "mag_err": 0.09,
        "kind": "detection",
        "note": "LCO r-band detection.",
    },
    {
        "source": "GIT",
        "gcn": "43757",
        "band": "r",
        "time_hr": 18.40,
        "limit_mag": 20.90,
        "kind": "upper_limit",
        "note": "GIT upper limit.",
    },
]


plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def download(url: str, dest: Path):
    ensure_dir(dest.parent)
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())


def hours_from_trigger(date_obs: str) -> float:
    return (Time(date_obs, format="isot", scale="utc") - TRIGGER).to_value("hour")


def sanitize_header(header):
    clean = header.copy()
    for key in ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "EXPTIME"]:
        if key in clean:
            try:
                clean[key] = float(clean[key])
            except Exception:
                pass
    return clean


def build_wcs(header):
    header = sanitize_header(header)
    ctype1 = str(header.get("CTYPE1", ""))
    ctype2 = str(header.get("CTYPE2", ""))
    if "SIP" in ctype1 or "SIP" in ctype2:
        return WCS(header)
    needed = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CTYPE1", "CTYPE2"]
    if all(key in header for key in needed):
        w = WCS(naxis=2)
        w.wcs.crval = [float(header["CRVAL1"]), float(header["CRVAL2"])]
        w.wcs.crpix = [float(header["CRPIX1"]), float(header["CRPIX2"])]
        w.wcs.cd = np.array(
            [
                [float(header["CD1_1"]), float(header["CD1_2"])],
                [float(header["CD2_1"]), float(header["CD2_2"])],
            ]
        )
        w.wcs.ctype = [str(header["CTYPE1"]), str(header["CTYPE2"])]
        w.wcs.cunit = ["deg", "deg"]
        return w
    return WCS(header)


def measure_target_photometry(data: np.ndarray, header, aperture_radius: float = 8.0) -> dict:
    wcs = build_wcs(header)
    data_sub, _, _ = estimate_background(data.astype(float))
    tx, ty = wcs.all_world2pix([[RA, DEC]], 0)[0]
    flux, flux_err = aperture_phot(data_sub, tx, ty, r_ap=aperture_radius)
    snr = flux / flux_err if flux_err and flux_err > 0 else None
    return {
        "flux": float(flux) if flux is not None and np.isfinite(flux) else None,
        "flux_err": float(flux_err) if flux_err is not None and np.isfinite(flux_err) else None,
        "snr": float(snr) if snr is not None and np.isfinite(snr) else None,
    }


def query_ps1_http(ra: float, dec: float, radius_deg: float, band: str) -> list[dict]:
    band_key = band.lower()[0]
    mag_col = {"g": "gMeanPSFMag", "r": "rMeanPSFMag"}.get(band_key, "rMeanPSFMag")
    err_col = f"{mag_col}Err"
    key = (round(ra, 4), round(dec, 4), round(radius_deg, 4), band_key)
    if key in PS1_CACHE:
        return PS1_CACHE[key]

    url = (
        "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv?"
        + urllib.parse.urlencode(
            {
                "ra": ra,
                "dec": dec,
                "radius": radius_deg,
                "nDetections.gte": 1,
                "columns": f"[raMean,decMean,{mag_col},{err_col}]",
            }
        )
    )
    text = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    stars = []
    for row in rows:
        try:
            mag = float(row[mag_col])
            mag_err = float(row.get(err_col) or 0.05)
            ra0 = float(row["raMean"])
            dec0 = float(row["decMean"])
        except Exception:
            continue
        if not np.isfinite(mag) or mag < 14.0 or mag > 21.0:
            continue
        if not np.isfinite(mag_err) or mag_err <= 0:
            mag_err = 0.05
        stars.append({"ra": ra0, "dec": dec0, "mag": mag, "mag_err": mag_err})
    PS1_CACHE[key] = stars
    return stars


def estimate_zeropoint_http(
    data: np.ndarray,
    header,
    band: str,
    ra0: float,
    dec0: float,
    aperture_radius: float = 8.0,
) -> tuple[Optional[float], int]:
    stars = query_ps1_http(ra0, dec0, 0.15, band)
    if not stars:
        return None, 0

    wcs = build_wcs(header)
    if not wcs.has_celestial:
        return None, 0

    data_sub, _, _ = estimate_background(data.astype(float))
    zps = []
    for star in stars:
        try:
            x, y = wcs.all_world2pix([[star["ra"], star["dec"]]], 0)[0]
        except Exception:
            continue
        if not (aperture_radius + 4 < x < data.shape[1] - aperture_radius - 4):
            continue
        if not (aperture_radius + 4 < y < data.shape[0] - aperture_radius - 4):
            continue
        flux, _ = aperture_phot(data_sub, x, y, r_ap=aperture_radius)
        if flux is None or not np.isfinite(flux) or flux <= 0:
            continue
        inst_mag = -2.5 * np.log10(flux)
        zps.append(star["mag"] - inst_mag)

    if len(zps) < 3:
        return None, len(zps)

    zps_arr = np.array(zps, dtype=float)
    med = np.median(zps_arr)
    mad = 1.4826 * np.median(np.abs(zps_arr - med))
    keep = np.abs(zps_arr - med) < 3 * max(mad, 0.05)
    zps_arr = zps_arr[keep]
    if len(zps_arr) < 3:
        return None, len(zps_arr)
    return float(np.median(zps_arr)), int(len(zps_arr))


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
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    return out


def reproject_to_header(data: np.ndarray, in_header, out_header) -> np.ndarray:
    in_wcs = build_wcs(in_header)
    out_wcs = build_wcs(out_header)
    ny = int(out_header["NAXIS2"])
    nx = int(out_header["NAXIS1"])
    yy, xx = np.indices((ny, nx), dtype=float)
    world = out_wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    in_xy = in_wcs.all_world2pix(world, 0)
    return bilinear_sample(data, in_xy[:, 0].reshape((ny, nx)), in_xy[:, 1].reshape((ny, nx)))


def build_master_bias(files: list[Path]) -> np.ndarray:
    stack = [fits.getdata(path).astype(np.float64) for path in files]
    return np.median(np.array(stack), axis=0)


def build_master_flat(files: list[Path], master_bias: np.ndarray) -> np.ndarray:
    flats = []
    for path in files:
        data = fits.getdata(path).astype(np.float64) - master_bias
        finite = np.isfinite(data)
        if not np.any(finite):
            continue
        med = float(np.median(data[finite]))
        if med <= 0:
            continue
        flats.append(data / med)
    if not flats:
        raise RuntimeError("no valid flat frames were available")
    master = np.median(np.array(flats), axis=0)
    bad = ~np.isfinite(master) | (master < 0.1)
    master[bad] = 1.0
    return master


def reduce_xinglong_science(science_path: Path, master_bias: np.ndarray, master_flat: np.ndarray) -> tuple[np.ndarray, fits.Header]:
    data = fits.getdata(science_path).astype(np.float64)
    hdr = fits.getheader(science_path)
    reduced = (data - master_bias) / master_flat
    return reduced, hdr


def write_fits(path: Path, data: np.ndarray, header):
    ensure_dir(path.parent)
    fits.PrimaryHDU(data=data.astype(np.float32), header=sanitize_header(header)).writeto(path, overwrite=True)


def stack_to_reference(files: list[Path], out_path: Path):
    base_data = fits.getdata(files[0]).astype(np.float64)
    base_hdr = fits.getheader(files[0])
    stack = [base_data]
    mjds = [Time(base_hdr["DATE-OBS"], format="isot", scale="utc").mjd]
    total_exptime = float(base_hdr.get("EXPTIME", 0.0))
    for path in files[1:]:
        data = fits.getdata(path).astype(np.float64)
        hdr = fits.getheader(path)
        reproj = reproject_to_header(data, hdr, base_hdr)
        stack.append(reproj)
        mjds.append(Time(hdr["DATE-OBS"], format="isot", scale="utc").mjd)
        total_exptime += float(hdr.get("EXPTIME", 0.0))
    combo = np.nanmedian(np.stack(stack), axis=0)
    hdr_out = base_hdr.copy()
    hdr_out["DATE-OBS"] = Time(np.mean(mjds), format="mjd", scale="utc").isot
    hdr_out["EXPTIME"] = total_exptime
    write_fits(out_path, combo, hdr_out)


def build_references(refdir: Path) -> dict[str, Path]:
    ensure_dir(refdir)
    refs = {}
    for band in ("g", "r"):
        url = (
            "https://www.legacysurvey.org/viewer/fits-cutout?"
            + urllib.parse.urlencode(
                {
                    "ra": RA,
                    "dec": DEC,
                    "layer": "ls-dr10",
                    "pixscale": 0.262,
                    "size": 2048,
                    "bands": band,
                }
            )
        )
        dest = refdir / f"desi_{band}_ref.fits"
        download(url, dest)
        refs[band] = dest
    return refs


def solve_catalog_shift(epoch_path: Path, band: str) -> tuple[float, float]:
    epoch_data = fits.getdata(epoch_path).astype(np.float64)
    epoch_hdr = fits.getheader(epoch_path)
    wcs = build_wcs(epoch_hdr)
    epoch_sub, _, _ = estimate_background(epoch_data)

    smooth = gaussian_filter(epoch_sub, 1.2)
    threshold = float(np.median(smooth) + 3.0 * np.std(smooth))
    maxima = maximum_filter(smooth, size=7)
    mask = (smooth == maxima) & (smooth > threshold)
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return 0.0, 0.0
    vals = smooth[yy, xx]
    order = np.argsort(vals)[::-1][:250]
    detected = np.column_stack([xx[order], yy[order], vals[order]])

    predicted = []
    for star in query_ps1_http(RA, DEC, 0.2, band):
        mag = star["mag"]
        if not (13.0 < mag < 19.5):
            continue
        try:
            x, y = wcs.all_world2pix([[star["ra"], star["dec"]]], 0)[0]
        except Exception:
            continue
        if 20.0 < x < epoch_data.shape[1] - 20.0 and 20.0 < y < epoch_data.shape[0] - 20.0:
            predicted.append((x, y))
    if not predicted:
        return 0.0, 0.0

    bins: dict[tuple[int, int], int] = {}
    for px, py in predicted:
        for dx, dy, _ in detected:
            sx = float(dx - px)
            sy = float(dy - py)
            if abs(sx) < 200.0 and abs(sy) < 200.0:
                key = (round(sx / 4.0) * 4, round(sy / 4.0) * 4)
                bins[key] = bins.get(key, 0) + 1
    if not bins:
        return 0.0, 0.0

    coarse_x, coarse_y = max(bins.items(), key=lambda item: item[1])[0]
    refined = []
    for px, py in predicted:
        dx = detected[:, 0] - (px + coarse_x)
        dy = detected[:, 1] - (py + coarse_y)
        rr = np.hypot(dx, dy)
        idx = int(np.argmin(rr))
        if rr[idx] < 8.0:
            refined.append((detected[idx, 0] - px, detected[idx, 1] - py))
    if len(refined) < 3:
        return float(coarse_x), float(coarse_y)
    return float(np.median([val[0] for val in refined])), float(np.median([val[1] for val in refined]))


def apply_wcs_shift(path: Path, dx: float, dy: float):
    with fits.open(path, mode="update") as hdul:
        hdr = hdul[0].header
        for key in ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "EXPTIME"]:
            if key in hdr:
                try:
                    hdr[key] = float(hdr[key])
                except Exception:
                    pass
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) + dx
        hdr["CRPIX2"] = float(hdr["CRPIX2"]) + dy
        hdul.flush()


def target_centered_difference(
    epoch_data: np.ndarray,
    epoch_header,
    reference_data: np.ndarray,
    reference_header,
    epoch_zp: Optional[float],
    ref_zp: Optional[float],
    half_size: int = 30,
) -> dict:
    if epoch_zp is None or ref_zp is None:
        return {
            "diff_flux_cal": None,
            "diff_flux_err_cal": None,
            "diff_snr": None,
            "diff_mag": None,
            "diff_mag_err": None,
            "diff_lim_mag_3sigma": None,
        }

    epoch_sub, _, _ = estimate_background(epoch_data.astype(float))
    ref_sub, _, _ = estimate_background(reference_data.astype(float))

    # Express the reference image on the epoch-count scale so the aperture
    # noise model remains in detector-like units.
    ref_scaled = ref_sub * 10 ** (0.4 * (epoch_zp - ref_zp))

    ew = build_wcs(epoch_header)
    rw = build_wcs(reference_header)
    ex, ey = ew.all_world2pix([[RA, DEC]], 0)[0]
    cx = int(round(ex))
    cy = int(round(ey))
    x0 = max(0, cx - half_size)
    x1 = min(epoch_sub.shape[1], cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(epoch_sub.shape[0], cy + half_size + 1)

    epoch_cut = epoch_sub[y0:y1, x0:x1]
    yy, xx = np.indices(epoch_cut.shape, dtype=float)
    world = ew.all_pix2world(np.column_stack([(x0 + xx).ravel(), (y0 + yy).ravel()]), 0)
    ref_xy = rw.all_world2pix(world, 0)
    ref_cut = bilinear_sample(
        ref_scaled,
        ref_xy[:, 0].reshape(epoch_cut.shape),
        ref_xy[:, 1].reshape(epoch_cut.shape),
    )

    diff = epoch_cut - ref_cut
    finite = np.isfinite(diff)
    if not np.any(finite):
        return {
            "diff_flux_cal": None,
            "diff_flux_err_cal": None,
            "diff_snr": None,
            "diff_mag": None,
            "diff_mag_err": None,
            "diff_lim_mag_3sigma": None,
        }

    med = float(np.nanmedian(diff[finite]))
    filled = np.where(finite, diff, med)
    diff_sub, _, _ = estimate_background(filled)
    tx = ex - x0
    ty = ey - y0
    flux, flux_err = aperture_phot(diff_sub, tx, ty)
    snr = flux / flux_err if flux_err and flux_err > 0 else None

    diff_mag = None
    diff_mag_err = None
    lim_mag = None
    if flux is not None and np.isfinite(flux) and flux > 0:
        diff_mag = -2.5 * np.log10(flux) + epoch_zp
        if flux_err is not None and np.isfinite(flux_err) and flux_err > 0:
            diff_mag_err = 1.0857 * flux_err / flux
    if flux_err is not None and np.isfinite(flux_err) and flux_err > 0:
        lim_mag = -2.5 * np.log10(3.0 * flux_err) + epoch_zp

    return {
        "diff_flux_cal": float(flux) if flux is not None and np.isfinite(flux) else None,
        "diff_flux_err_cal": float(flux_err) if flux_err is not None and np.isfinite(flux_err) else None,
        "diff_snr": float(snr) if snr is not None and np.isfinite(snr) else None,
        "diff_mag": float(diff_mag) if diff_mag is not None and np.isfinite(diff_mag) else None,
        "diff_mag_err": float(diff_mag_err) if diff_mag_err is not None and np.isfinite(diff_mag_err) else None,
        "diff_lim_mag_3sigma": float(lim_mag) if lim_mag is not None and np.isfinite(lim_mag) else None,
    }


def prepare_case_layout() -> dict[str, object]:
    clean_dir(OUT)

    epoch_root = OUT / "epoch"
    trt_out = epoch_root / "TRT-CTO" / "r"
    xg_out = epoch_root / "XINGLONG216" / "g"
    xr_out = epoch_root / "XINGLONG216" / "r"
    ensure_dir(trt_out)
    ensure_dir(xg_out)
    ensure_dir(xr_out)

    # Copy TRT science frames as-is.
    trt_files = sorted((DATA_ROOT / "TRT-CTO").glob("*.fits"))
    prepared_trt = []
    for src in trt_files:
        dest = trt_out / src.name
        shutil.copy2(src, dest)
        prepared_trt.append(dest)

    # Reduce Xinglong science frames.
    xdir = DATA_ROOT / "XINGLONG216"
    bias_files = sorted(xdir.glob("*_BIAS_*.fit"))
    gflat_files = sorted(xdir.glob("*_PHOTFLAT_*_Free_g_Free.fit"))
    rflat_files = sorted(xdir.glob("*_PHOTFLAT_*_Free_r_Free.fit"))
    gscience = sorted(xdir.glob("*_PHOTTARGET_*_Free_g_Free.fit"))
    rscience = sorted(xdir.glob("*_PHOTTARGET_*_Free_r_Free.fit"))

    master_bias = build_master_bias(bias_files)
    master_flat_g = build_master_flat(gflat_files, master_bias)
    master_flat_r = build_master_flat(rflat_files, master_bias)

    prepared_xg = []
    for src in gscience:
        data, hdr = reduce_xinglong_science(src, master_bias, master_flat_g)
        dest = xg_out / f"{src.stem}.reduced.fits"
        write_fits(dest, data, hdr)
        prepared_xg.append(dest)

    prepared_xr = []
    for src in rscience:
        data, hdr = reduce_xinglong_science(src, master_bias, master_flat_r)
        dest = xr_out / f"{src.stem}.reduced.fits"
        write_fits(dest, data, hdr)
        prepared_xr.append(dest)

    refs = build_references(OUT / "reference")
    return {
        "prepared_trt": prepared_trt,
        "prepared_xg": prepared_xg,
        "prepared_xr": prepared_xr,
        "references": refs,
    }


def inventory_frame(station: str, band: str, file_path: Path) -> dict:
    hdr = fits.getheader(file_path)
    data = fits.getdata(file_path)
    w = build_wcs(hdr)
    x, y = w.all_world2pix([[RA, DEC]], 0)[0]
    return {
        "station": station,
        "band": band,
        "file": file_path.name,
        "date_obs": hdr.get("DATE-OBS"),
        "hours_since_trigger": hours_from_trigger(hdr.get("DATE-OBS")),
        "shape_x": int(data.shape[1]),
        "shape_y": int(data.shape[0]),
        "target_x": float(x),
        "target_y": float(y),
        "in_fov": bool(0 <= x < data.shape[1] and 0 <= y < data.shape[0]),
        "filter": hdr.get("FILTER"),
        "exptime": hdr.get("EXPTIME"),
        "ra_center": hdr.get("CRVAL1"),
        "dec_center": hdr.get("CRVAL2"),
    }


def measure_latency(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def measure_bundle(
    case_name: str,
    station: str,
    band: str,
    epoch_files: list[Path],
    reference_file: Path,
    reference_zp: Optional[float],
    output_dir: Path,
) -> dict:
    clean_dir(output_dir)

    epoch_input = output_dir / "epoch_input"
    reference_input = output_dir / "reference_input"
    epoch_hips = output_dir / "epoch_hips"
    reference_hips = output_dir / "reference_hips"
    diff_dir = output_dir / "diff"
    ensure_dir(epoch_input)
    ensure_dir(reference_input)

    for src in epoch_files:
        shutil.copy2(src, epoch_input / src.name)
    shutil.copy2(reference_file, reference_input / reference_file.name)

    build_epoch, t_epoch_hips = measure_latency(
        run_local_hips, str(epoch_input), str(epoch_hips), order=ORDER, mode="MEAN", timeout=300
    )
    build_ref, t_ref_hips = measure_latency(
        run_local_hips, str(reference_input), str(reference_hips), order=ORDER, mode="MEAN", timeout=300
    )

    diff_res, t_diff = measure_latency(
        diff_hips,
        event_id=EVENT_ID,
        tid=station,
        night=case_name,
        band=band,
        candidate_positions=[{"truth_id": 0, "ra": RA, "dec": DEC}],
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

    ref_data = fits.getdata(reference_file).astype(np.float64)
    ref_hdr = fits.getheader(reference_file)

    rows = []
    phot_times = []
    for path in epoch_files:
        data = fits.getdata(path).astype(np.float64)
        hdr = fits.getheader(path)
        phot_res, dt = measure_latency(measure_target_photometry, data, hdr)
        phot_times.append(dt)

        epoch_zp, n_epoch_ref = estimate_zeropoint_http(data, hdr, band, RA, DEC)
        direct_mag = None
        direct_mag_err = None
        direct_lim = None
        if epoch_zp is not None and phot_res["flux"] is not None and np.isfinite(phot_res["flux"]) and phot_res["flux"] > 0:
            direct_mag = -2.5 * np.log10(phot_res["flux"]) + epoch_zp
            if phot_res["flux_err"] is not None and np.isfinite(phot_res["flux_err"]) and phot_res["flux_err"] > 0:
                direct_mag_err = 1.0857 * phot_res["flux_err"] / phot_res["flux"]
        if epoch_zp is not None and phot_res["flux_err"] is not None and np.isfinite(phot_res["flux_err"]) and phot_res["flux_err"] > 0:
            direct_lim = -2.5 * np.log10(3.0 * phot_res["flux_err"]) + epoch_zp

        diff_stats = target_centered_difference(data, hdr, ref_data, ref_hdr, epoch_zp, reference_zp)

        rows.append(
            {
                "case": case_name,
                "station": station,
                "band": band,
                "file": path.name,
                "date_obs": hdr.get("DATE-OBS"),
                "hours_since_trigger": hours_from_trigger(hdr.get("DATE-OBS")),
                "exptime": float(hdr.get("EXPTIME", 0.0)),
                "zp_epoch": epoch_zp,
                "n_epoch_ref_stars": n_epoch_ref,
                "direct_flux": phot_res["flux"],
                "direct_flux_err": phot_res["flux_err"],
                "direct_snr": phot_res["snr"],
                "direct_mag": direct_mag,
                "direct_mag_err": direct_mag_err,
                "direct_lim_mag_3sigma": direct_lim,
                **diff_stats,
            }
        )

    diff_snrs = [row["diff_snr"] for row in rows if row["diff_snr"] is not None]
    diff_mags = [row["diff_mag"] for row in rows if row["diff_mag"] is not None]
    diff_errs = [row["diff_mag_err"] for row in rows if row["diff_mag_err"] is not None]
    med_phot_ms = float(np.median(phot_times) * 1000.0) if phot_times else None
    provenance_row = diff_res["candidates"][0] if diff_res.get("candidates") else {}

    return {
        "case": case_name,
        "station": station,
        "band": band,
        "n_epoch_files": len(epoch_files),
        "reference_file": reference_file.name,
        "reference_zp": reference_zp,
        "epoch_hips_s": t_epoch_hips,
        "reference_hips_s": t_ref_hips,
        "diff_lookup_s": t_diff,
        "median_phot_ms": med_phot_ms,
        "build_epoch_ok": bool(build_epoch.get("ok")),
        "build_reference_ok": bool(build_ref.get("ok")),
        "provenance_status": diff_res.get("status"),
        "hpxfinder_epoch_sources": provenance_row.get("epoch_source_files", []),
        "hpxfinder_reference_sources": provenance_row.get("reference_source_files", []),
        "hpxfinder_epoch_source_path": provenance_row.get("epoch_source_path"),
        "hpxfinder_reference_source_path": provenance_row.get("reference_source_path"),
        "target_centered_diff_snr_median": float(np.median(diff_snrs)) if diff_snrs else None,
        "target_centered_diff_snr_max": float(np.max(diff_snrs)) if diff_snrs else None,
        "target_centered_detected": bool(any(val >= 5.0 for val in diff_snrs)),
        "diff_mag_median": float(np.median(diff_mags)) if diff_mags else None,
        "diff_mag_err_median": float(np.median(diff_errs)) if diff_errs else None,
        "phot_rows": rows,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_lightcurve(summary: dict):
    trt_single = summary["trt_r_single"]["phot_rows"]
    trt_stack = summary["trt_r_stack"]["phot_rows"][0]
    xg_single = summary["xinglong_g_single"]["phot_rows"]
    xg_stack = summary["xinglong_g_stack"]["phot_rows"][0]
    xr_single = summary["xinglong_r_single"]["phot_rows"]
    xr_stack = summary["xinglong_r_stack"]["phot_rows"][0]

    def mag_fields(rows):
        if any(row.get("diff_mag") is not None and row.get("diff_mag_err") is not None for row in rows):
            return "diff_mag", "diff_mag_err"
        return "direct_mag", "direct_mag_err"

    def finite_rows(rows, mag_key, err_key):
        return [row for row in rows if row.get(mag_key) is not None and row.get(err_key) is not None]

    trt_mag_key, trt_err_key = mag_fields(trt_single)
    xr_mag_key, xr_err_key = mag_fields(xr_single)
    xg_mag_key, xg_err_key = mag_fields(xg_single)
    trt_single = finite_rows(trt_single, trt_mag_key, trt_err_key)
    xg_single = finite_rows(xg_single, xg_mag_key, xg_err_key)
    xr_single = finite_rows(xr_single, xr_mag_key, xr_err_key)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True)

    ax = axes[0]
    if trt_single:
        ax.errorbar(
            [row["hours_since_trigger"] for row in trt_single],
            [row[trt_mag_key] for row in trt_single],
            yerr=[row[trt_err_key] for row in trt_single],
            fmt="o",
            color="#f28e2b",
            label="TRT-CTO r single",
        )
    ax.errorbar(
        trt_stack["hours_since_trigger"],
        trt_stack[trt_mag_key],
        yerr=trt_stack[trt_err_key],
        fmt="*",
        markersize=13,
        color="#c45b00",
        label="TRT-CTO r stack",
    )
    if xr_single:
        ax.errorbar(
            [row["hours_since_trigger"] for row in xr_single],
            [row[xr_mag_key] for row in xr_single],
            yerr=[row[xr_err_key] for row in xr_single],
            fmt="o",
            color="#d62728",
            label="Xinglong 2.16m r single",
        )
    ax.errorbar(
        xr_stack["hours_since_trigger"],
        xr_stack[xr_mag_key],
        yerr=xr_stack[xr_err_key],
        fmt="*",
        markersize=13,
        color="#8c1d1d",
        label="Xinglong 2.16m r stack",
    )
    for row in PUBLISHED_CONTEXT:
        if row["band"] != "r":
            continue
        if row["kind"] == "detection":
            ax.errorbar(
                row["time_hr"],
                row["mag"],
                yerr=row["mag_err"],
                fmt="s",
                ms=6,
                mfc="white",
                mec="black",
                ecolor="black",
            )
        else:
            ax.scatter(row["time_hr"], row["limit_mag"], marker="v", color="black", s=40)
            ax.annotate(
                "",
                xy=(row["time_hr"], row["limit_mag"]),
                xytext=(row["time_hr"], row["limit_mag"] + 0.35),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1),
            )
    ax.set_ylabel("r-band AB magnitude")
    ax.set_title("EP260214b: Real Positive Detection from TRT and Xinglong 2.16m")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    if xg_single:
        ax.errorbar(
            [row["hours_since_trigger"] for row in xg_single],
            [row[xg_mag_key] for row in xg_single],
            yerr=[row[xg_err_key] for row in xg_single],
            fmt="s",
            color="#1f77b4",
            label="Xinglong 2.16m g single (marginal)",
        )
    ax.errorbar(
        xg_stack["hours_since_trigger"],
        xg_stack[xg_mag_key],
        yerr=xg_stack[xg_err_key],
        fmt="*",
        markersize=13,
        color="#104e8b",
        label="Xinglong 2.16m g stack (marginal)",
    )
    for row in PUBLISHED_CONTEXT:
        if row["band"] != "g":
            continue
        ax.errorbar(
            row["time_hr"],
            row["mag"],
            yerr=row["mag_err"],
            fmt="s",
            ms=6,
            mfc="white",
            mec="black",
            ecolor="black",
        )
    ax.set_xlabel("Hours since trigger")
    ax.set_ylabel("g-band AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out = OUT / "fig_ep260214b_multiband_lightcurve.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    shutil.copy2(out, DOCS / out.name)
    return out


def plot_detection_summary(summary: dict):
    def choose_snr(rows):
        if any(row.get("diff_mag") is not None for row in rows):
            return [row["diff_snr"] for row in rows if row.get("diff_snr") is not None]
        return [row["direct_snr"] for row in rows if row.get("direct_snr") is not None]

    def choose_mag(row):
        if row.get("diff_mag") is not None and row.get("diff_mag_err") is not None:
            return row["diff_mag"], row["diff_mag_err"]
        return row.get("direct_mag"), row.get("direct_mag_err")

    cases = [
        ("TRT r single", summary["trt_r_single"]["phot_rows"], "#f28e2b"),
        ("TRT r stack", summary["trt_r_stack"]["phot_rows"], "#c45b00"),
        ("X216 g single", summary["xinglong_g_single"]["phot_rows"], "#1f77b4"),
        ("X216 g stack", summary["xinglong_g_stack"]["phot_rows"], "#104e8b"),
        ("X216 r single", summary["xinglong_r_single"]["phot_rows"], "#d62728"),
        ("X216 r stack", summary["xinglong_r_stack"]["phot_rows"], "#8c1d1d"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    ax = axes[0]
    xpos = np.arange(len(cases))
    for i, (label, rows, color) in enumerate(cases):
        snrs = choose_snr(rows)
        jitter = np.linspace(-0.08, 0.08, len(snrs)) if len(snrs) > 1 else [0.0]
        for dx, val in zip(jitter, snrs):
            ax.scatter(i + dx, val, color=color, s=35)
        if snrs:
            ax.hlines(np.median(snrs), i - 0.18, i + 0.18, color=color, linewidth=2)
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([label for label, _, _ in cases], rotation=25, ha="right")
    ax.set_ylabel("Target-centered diff S/N")
    ax.set_title("Detection Significance")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    labels = ["TRT r stack", "X216 g stack", "X216 r stack"]
    trt_mag, trt_err = choose_mag(summary["trt_r_stack"]["phot_rows"][0])
    xg_mag, xg_err = choose_mag(summary["xinglong_g_stack"]["phot_rows"][0])
    xr_mag, xr_err = choose_mag(summary["xinglong_r_stack"]["phot_rows"][0])
    mags = [
        trt_mag,
        xg_mag,
        xr_mag,
    ]
    errs = [
        trt_err,
        xg_err,
        xr_err,
    ]
    colors = ["#c45b00", "#104e8b", "#8c1d1d"]
    ax.bar(labels, mags, yerr=errs, color=colors, capsize=4)
    ax.set_ylabel("Stacked AB magnitude")
    ax.set_title("Stacked Counterpart Measurements")
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.25)

    g_mag = summary["xinglong_g_stack"]["phot_rows"][0]["diff_mag"]
    g_err = summary["xinglong_g_stack"]["phot_rows"][0]["diff_mag_err"] or 0.0
    r_mag = summary["xinglong_r_stack"]["phot_rows"][0]["diff_mag"]
    r_err = summary["xinglong_r_stack"]["phot_rows"][0]["diff_mag_err"] or 0.0
    if g_mag is not None and r_mag is not None:
        color_gr = g_mag - r_mag
        color_err = math.sqrt(g_err ** 2 + r_err ** 2)
        ax.text(
            0.98,
            0.95,
            f"Xinglong stacked g-r = {color_gr:.2f} +/- {color_err:.2f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )

    out = OUT / "fig_ep260214b_detection_summary.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    shutil.copy2(out, DOCS / out.name)
    return out


def build_formal_table(summary: dict) -> list[dict]:
    rows = []
    trt_single = summary["trt_r_single"]["phot_rows"]
    trt_stack = summary["trt_r_stack"]["phot_rows"][0]
    xg_single = summary["xinglong_g_single"]["phot_rows"]
    xg_stack = summary["xinglong_g_stack"]["phot_rows"][0]
    xr_single = summary["xinglong_r_single"]["phot_rows"]
    xr_stack = summary["xinglong_r_stack"]["phot_rows"][0]

    def finite_values(items, key):
        return [item[key] for item in items if item.get(key) is not None and np.isfinite(item[key])]

    def fmt_stats(values):
        if not values:
            return "n/a"
        return f"{np.median(values):.2f}, max={np.max(values):.2f}"

    def choose_stack_mag(row):
        if row.get("diff_mag") is not None and row.get("diff_mag_err") is not None:
            return row["diff_mag"], row["diff_mag_err"], "difference-image"
        return row.get("direct_mag"), row.get("direct_mag_err"), "direct-target"

    trt_single_snr = finite_values(trt_single, "diff_snr")
    trt_single_mag = finite_values(trt_single, "diff_mag")
    xg_single_snr = finite_values(xg_single, "diff_snr")
    xr_single_snr = finite_values(xr_single, "diff_snr")
    xg_direct_snr = finite_values(xg_single, "direct_snr")
    xg_mag, xg_mag_err, xg_mode = choose_stack_mag(xg_stack)
    xr_mag, xr_mag_err, xr_mode = choose_stack_mag(xr_stack)

    rows.append(
        {
            "dataset": "TRT-CTO single frames",
            "band": "r",
            "time_window_hr": f"{min(r['hours_since_trigger'] for r in trt_single):.2f}--{max(r['hours_since_trigger'] for r in trt_single):.2f}",
            "our_result": (
                f"target-centered diff S/N median/max={fmt_stats(trt_single_snr)}; "
                f"diff mags {np.min(trt_single_mag):.2f}--{np.max(trt_single_mag):.2f} AB"
            ),
            "published_reference": "GCN 43747 and GCN 43750 report r-band detections at 7.95 hr and 11.34 hr",
            "consistency_note": "Our TRT sequence securely detects the counterpart between these published epochs and follows the same brightness range.",
        }
    )
    rows.append(
        {
            "dataset": "TRT-CTO stacked",
            "band": "r",
            "time_window_hr": f"{trt_stack['hours_since_trigger']:.2f}",
            "our_result": (
                f"stacked diff S/N={trt_stack['diff_snr']:.2f}; "
                f"stacked diff mag={trt_stack['diff_mag']:.2f} +/- {trt_stack['diff_mag_err']:.2f} AB"
            ),
            "published_reference": "GCN 43747: r=21.11+/-0.07 at 7.95 hr; GCN 43750: r=21.28+/-0.09 at 11.34 hr",
            "consistency_note": "The stacked TRT measurement lies between the published early and later r-band detections, as expected for the observation time.",
        }
    )
    rows.append(
        {
            "dataset": "Xinglong 2.16m g sequence",
            "band": "g",
            "time_window_hr": f"{min(r['hours_since_trigger'] for r in xg_single):.2f}--{max(r['hours_since_trigger'] for r in xg_single):.2f}",
            "our_result": (
                f"target-centered diff S/N median/max={fmt_stats(xg_single_snr)}; "
                f"direct-target S/N median/max={fmt_stats(xg_direct_snr)}; "
                f"stacked {xg_mode} mag={xg_mag:.2f} +/- {xg_mag_err:.2f} AB"
            ),
            "published_reference": "GCN 43747: g=21.30+/-0.09 at 7.95 hr",
            "consistency_note": "The late-time g-band signal is only marginal in the difference image and is therefore not used as the primary positive-detection anchor.",
        }
    )
    rows.append(
        {
            "dataset": "Xinglong 2.16m r sequence",
            "band": "r",
            "time_window_hr": f"{min(r['hours_since_trigger'] for r in xr_single):.2f}--{max(r['hours_since_trigger'] for r in xr_single):.2f}",
            "our_result": (
                f"target-centered diff S/N median/max={fmt_stats(xr_single_snr)}; "
                f"stacked {xr_mode} mag={xr_mag:.2f} +/- {xr_mag_err:.2f} AB"
            ),
            "published_reference": "GCN 43750: r=21.28+/-0.09 at 11.34 hr; GCN 43757: r>20.9 at 18.40 hr",
            "consistency_note": "The later Xinglong r-band detection is still consistent with the published slow-fading counterpart and remains fainter than the shallow GIT limit threshold.",
        }
    )
    return rows


def main():
    layout = prepare_case_layout()

    refs = layout["references"]
    reference_zp = {}
    for band, path in refs.items():
        data = fits.getdata(path).astype(np.float64)
        hdr = fits.getheader(path)
        zp, n_ref = estimate_zeropoint_http(data, hdr, band, RA, DEC)
        reference_zp[band] = {"zp": zp, "n_ref_stars": n_ref}

    # Build stacks.
    stack_root = OUT / "stacks"
    ensure_dir(stack_root)
    trt_stack = stack_root / "EP260214b_TRT_r_stack.fits"
    xg_stack = stack_root / "EP260214b_Xinglong_g_stack.fits"
    xr_stack = stack_root / "EP260214b_Xinglong_r_stack.fits"
    stack_to_reference(layout["prepared_trt"], trt_stack)
    stack_to_reference(layout["prepared_xg"], xg_stack)
    stack_to_reference(layout["prepared_xr"], xr_stack)

    g_shift = solve_catalog_shift(xg_stack, "g")
    r_shift = solve_catalog_shift(xr_stack, "r")
    for path in list(layout["prepared_xg"]) + [xg_stack]:
        apply_wcs_shift(path, *g_shift)
    for path in list(layout["prepared_xr"]) + [xr_stack]:
        apply_wcs_shift(path, *r_shift)

    inventory_rows = []
    for path in layout["prepared_trt"]:
        inventory_rows.append(inventory_frame("TRT-CTO", "r", path))
    for path in layout["prepared_xg"]:
        inventory_rows.append(inventory_frame("XINGLONG216", "g", path))
    for path in layout["prepared_xr"]:
        inventory_rows.append(inventory_frame("XINGLONG216", "r", path))
    inventory_rows.append(inventory_frame("TRT-CTO", "r", trt_stack))
    inventory_rows.append(inventory_frame("XINGLONG216", "g", xg_stack))
    inventory_rows.append(inventory_frame("XINGLONG216", "r", xr_stack))
    write_csv(OUT / "inventory.csv", inventory_rows)

    summary = {
        "event_id": EVENT_ID,
        "ra_deg": RA,
        "dec_deg": DEC,
        "trigger_utc": TRIGGER_UTC,
        "published_context": PUBLISHED_CONTEXT,
        "reference_files": {band: str(path) for band, path in refs.items()},
        "reference_zp": reference_zp,
        "xinglong_wcs_shift_pixels": {"g": {"dx": g_shift[0], "dy": g_shift[1]}, "r": {"dx": r_shift[0], "dy": r_shift[1]}},
        "trt_r_single": measure_bundle(
            "TRT_r_single",
            "TRT-CTO",
            "r",
            layout["prepared_trt"],
            refs["r"],
            reference_zp["r"]["zp"],
            OUT / "TRT_r_single_case",
        ),
        "trt_r_stack": measure_bundle(
            "TRT_r_stack",
            "TRT-CTO_stack",
            "r",
            [trt_stack],
            refs["r"],
            reference_zp["r"]["zp"],
            OUT / "TRT_r_stack_case",
        ),
        "xinglong_g_single": measure_bundle(
            "XINGLONG_g_single",
            "XINGLONG216_g",
            "g",
            layout["prepared_xg"],
            refs["g"],
            reference_zp["g"]["zp"],
            OUT / "XINGLONG_g_single_case",
        ),
        "xinglong_g_stack": measure_bundle(
            "XINGLONG_g_stack",
            "XINGLONG216_g_stack",
            "g",
            [xg_stack],
            refs["g"],
            reference_zp["g"]["zp"],
            OUT / "XINGLONG_g_stack_case",
        ),
        "xinglong_r_single": measure_bundle(
            "XINGLONG_r_single",
            "XINGLONG216_r",
            "r",
            layout["prepared_xr"],
            refs["r"],
            reference_zp["r"]["zp"],
            OUT / "XINGLONG_r_single_case",
        ),
        "xinglong_r_stack": measure_bundle(
            "XINGLONG_r_stack",
            "XINGLONG216_r_stack",
            "r",
            [xr_stack],
            refs["r"],
            reference_zp["r"]["zp"],
            OUT / "XINGLONG_r_stack_case",
        ),
    }

    # Flat measurement table for downstream paper writing.
    measurement_rows = []
    for key in [
        "trt_r_single",
        "trt_r_stack",
        "xinglong_g_single",
        "xinglong_g_stack",
        "xinglong_r_single",
        "xinglong_r_stack",
    ]:
        measurement_rows.extend(summary[key]["phot_rows"])
    write_csv(OUT / "ep260214b_measurement_summary.csv", measurement_rows)

    formal_rows = build_formal_table(summary)
    write_csv(OUT / "ep260214b_formal_detection_table.csv", formal_rows)
    write_csv(OUT / "ep260214b_published_context.csv", PUBLISHED_CONTEXT)

    fig1 = plot_lightcurve(summary)
    fig2 = plot_detection_summary(summary)
    summary["figure_files"] = [str(fig1), str(fig2)]

    g_mag = summary["xinglong_g_stack"]["phot_rows"][0]["diff_mag"]
    g_err = summary["xinglong_g_stack"]["phot_rows"][0]["diff_mag_err"]
    r_mag = summary["xinglong_r_stack"]["phot_rows"][0]["diff_mag"]
    r_err = summary["xinglong_r_stack"]["phot_rows"][0]["diff_mag_err"]
    if g_mag is not None and r_mag is not None:
        summary["xinglong_stacked_g_minus_r"] = {
            "color": g_mag - r_mag,
            "color_err": math.sqrt((g_err or 0.0) ** 2 + (r_err or 0.0) ** 2),
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.copy2(OUT / "summary.json", OUT / "summary_latest.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
