"""
Utilities for working with HiPS/HpxFinder products at runtime.

The main design principle is:
  HiPS/HpxFinder locates the relevant original images,
  while science-domain differencing and photometry remain in the original-image domain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def highest_hips_order(hips_dir: str) -> int:
    orders = []
    for path in Path(hips_dir).glob("Norder*"):
        if path.is_dir() and path.name[6:].isdigit():
            orders.append(int(path.name[6:]))
    if not orders:
        raise FileNotFoundError(f"No Norder* directories found under {hips_dir}")
    return max(orders)


def iter_tiles_for_order(hips_dir: str, order: int):
    root = Path(hips_dir) / f"Norder{order}"
    if not root.exists():
        return
    for path in root.rglob("Npix*.fits"):
        yield path


def parse_order_npix(tile_path: str | Path) -> tuple[int, int]:
    p = Path(tile_path)
    order = int(p.parent.parent.name.replace("Norder", ""))
    npix = int(p.stem.replace("Npix", ""))
    return order, npix


def hpxfinder_index_path(hips_dir: str, order: int, npix: int) -> Path:
    dir_base = (int(npix) // 10000) * 10000
    return Path(hips_dir) / "HpxFinder" / f"Norder{order}" / f"Dir{dir_base}" / f"Npix{npix}"


def read_hpxfinder_sources(index_path: str | Path) -> list[str]:
    p = Path(index_path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = rec.get("filepath")
            if fp:
                out.append(os.path.basename(fp))
    return out


def find_tile_for_sky(hips_dir: str, ra: float, dec: float, order: Optional[int] = None):
    order = highest_hips_order(hips_dir) if order is None else order
    for tile_path in iter_tiles_for_order(hips_dir, order):
        hdr = fits.getheader(tile_path)
        wcs = WCS(hdr)
        try:
            x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
        except Exception:
            continue
        if -0.5 <= x < hdr.get("NAXIS1", 512) - 0.5 and -0.5 <= y < hdr.get("NAXIS2", 512) - 0.5:
            _, npix = parse_order_npix(tile_path)
            return {
                "tile_path": str(tile_path),
                "order": int(order),
                "npix": int(npix),
                "x": float(x),
                "y": float(y),
            }
    return None


def query_hpxfinder_sources_for_sky(hips_dir: str, ra: float, dec: float, order: Optional[int] = None):
    tile = find_tile_for_sky(hips_dir, ra, dec, order=order)
    if tile is None:
        return None
    idx = hpxfinder_index_path(hips_dir, tile["order"], tile["npix"])
    return {
        **tile,
        "index_path": str(idx),
        "source_files": read_hpxfinder_sources(idx),
    }


def build_source_map(source_dir: str) -> dict[str, str]:
    mapping = {}
    for path in Path(source_dir).iterdir():
        if path.is_file():
            mapping[path.name] = str(path)
    return mapping


def select_source_image_for_sky(source_map: dict[str, str], source_files: list[str],
                                ra: float, dec: float) -> tuple[Optional[str], Optional[float], Optional[float]]:
    for name in source_files:
        path = source_map.get(os.path.basename(name))
        if not path or not os.path.exists(path):
            continue
        hdr = fits.getheader(path)
        wcs = WCS(hdr)
        try:
            x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
        except Exception:
            continue
        ny = hdr.get("NAXIS2", 0)
        nx = hdr.get("NAXIS1", 0)
        if 0 <= x < nx and 0 <= y < ny:
            return path, float(x), float(y)
    return None, None, None


def _extract_cutout(data: np.ndarray, x: float, y: float, half_size: int) -> tuple[np.ndarray, int, int]:
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - half_size)
    x1 = min(data.shape[1], cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(data.shape[0], cy + half_size + 1)
    return data[y0:y1, x0:x1].astype(float, copy=False), x0, y0


def _bilinear_sample(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling on a 2D image for arrays of floating-point coordinates.
    Samples outside the image are returned as NaN.
    """
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

    samp = (
        (1 - wx) * (1 - wy) * v00 +
        wx * (1 - wy) * v10 +
        (1 - wx) * wy * v01 +
        wx * wy * v11
    )
    out[valid] = samp
    return out


def original_domain_difference(epoch_path: str, reference_path: str,
                               ra: float, dec: float,
                               cutout_half_size: int = 25,
                               detection_radius: int = 4,
                               threshold_sigma: float = 5.0) -> dict:
    """
    Difference two original images around a sky position and test whether
    a positive transient-like residual is detected near that position.
    """
    epoch_data = fits.getdata(epoch_path).astype(np.float64, copy=False)
    epoch_hdr = fits.getheader(epoch_path)
    ref_data = fits.getdata(reference_path).astype(np.float64, copy=False)
    ref_hdr = fits.getheader(reference_path)

    epoch_wcs = WCS(epoch_hdr)
    ref_wcs = WCS(ref_hdr)
    ex, ey = epoch_wcs.all_world2pix([[ra, dec]], 0)[0]
    rx, ry = ref_wcs.all_world2pix([[ra, dec]], 0)[0]

    epoch_cut, ex0, ey0 = _extract_cutout(epoch_data, ex, ey, cutout_half_size)

    # Reproject the reference image onto the epoch cutout pixel grid before differencing.
    yy, xx = np.indices(epoch_cut.shape, dtype=float)
    epoch_x = ex0 + xx
    epoch_y = ey0 + yy
    world = epoch_wcs.all_pix2world(
        np.column_stack([epoch_x.ravel(), epoch_y.ravel()]),
        0
    )
    ref_xy = ref_wcs.all_world2pix(world, 0)
    ref_x = ref_xy[:, 0].reshape(epoch_cut.shape)
    ref_y = ref_xy[:, 1].reshape(epoch_cut.shape)
    ref_cut = _bilinear_sample(ref_data, ref_x, ref_y)

    diff = epoch_cut - ref_cut

    finite = np.isfinite(diff)
    if not finite.any():
        return {
            "recovered": False,
            "sigma": None,
            "peak": None,
            "peak_snr": None,
            "det_ra": None,
            "det_dec": None,
        }

    vals = diff[finite]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(vals))
    sigma = max(sigma, 1e-6)

    cx = int(round(ex)) - ex0
    cy = int(round(ey)) - ey0
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    region = finite & (rr2 <= detection_radius ** 2)
    if not region.any():
        return {
            "recovered": False,
            "sigma": sigma,
            "peak": None,
            "peak_snr": 0.0,
            "det_ra": None,
            "det_dec": None,
        }

    local_vals = diff[region]
    peak = float(np.max(local_vals))
    signal = float(np.sum(local_vals))
    noise = float(sigma * np.sqrt(np.sum(region)))
    peak_idx = np.argmax(np.where(region, diff, -np.inf))
    py, px = np.unravel_index(int(peak_idx), diff.shape)
    det_x = ex0 + px
    det_y = ey0 + py
    det_ra, det_dec = epoch_wcs.all_pix2world([[det_x, det_y]], 0)[0]
    peak_snr = peak / sigma
    aperture_snr = signal / noise if noise > 0 else 0.0

    return {
        "recovered": bool(aperture_snr >= threshold_sigma),
        "sigma": sigma,
        "peak": peak,
        "peak_snr": peak_snr,
        "signal": signal,
        "aperture_snr": aperture_snr,
        "det_ra": float(det_ra),
        "det_dec": float(det_dec),
        "epoch_x": float(ex),
        "epoch_y": float(ey),
        "reference_x": float(rx),
        "reference_y": float(ry),
    }
