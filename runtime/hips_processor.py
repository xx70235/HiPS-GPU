"""
GPU HiPS generation for Transient Hub.

Given a (event_id, tid, night, band) group, this module:
1. Downloads all science FITS for the group from OSS
2. Optionally applies flat/bias calibration
3. Runs hipsgen_cuda to build epoch HiPS tiles
4. Uploads tiles to OSS under {event}/hips/epochs/{tid}/{night}/{band}/
5. (Future) Difference imaging against reference HiPS

Requires:
  - hipsgen_cuda binary
  - s3.py  (from api/)
  - calibrate.py (from pipeline/)
"""
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from project_paths import OUTPUT_ROOT, default_hipsgen_bin

from hips_runtime import (
    build_source_map,
    highest_hips_order,
    original_domain_difference,
    query_hpxfinder_sources_for_sky,
    select_source_image_for_sky,
)

try:
    from models import SessionLocal, Observation, Event
except Exception:  # pragma: no cover - optional in local validation mode
    SessionLocal = None
    Observation = None
    Event = None

try:
    from s3 import (
        get_s3_client, S3_BUCKET, build_hips_key,
        upload_directory, download_prefix,
    )
except Exception:  # pragma: no cover - optional in local validation mode
    get_s3_client = None
    S3_BUCKET = None
    build_hips_key = None
    upload_directory = None
    download_prefix = None

log = logging.getLogger("hips_processor")

HIPSGEN = default_hipsgen_bin()
# Scratch space on fast local storage
SCRATCH_BASE = os.environ.get("HIPS_SCRATCH", str(OUTPUT_ROOT / "scratch"))
DEFAULT_ORDER = 7  # HiPS tile order; covers ~0.8° field well


# ─────────────────── helpers ───────────────────

def _download_from_oss(key: str, dest: str):
    if get_s3_client is None or S3_BUCKET is None:
        raise RuntimeError("S3 dependencies are unavailable in this environment")
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3.download_file(S3_BUCKET, key, dest)


def _group_observations(event_id: str, tid: str, night: str, band: str):
    """Return list of Observation rows for this group (status=done, science)."""
    if SessionLocal is None or Observation is None:
        raise RuntimeError("Database dependencies are unavailable in this environment")
    db = SessionLocal()
    try:
        return (
            db.query(Observation)
            .filter(
                Observation.event_id == event_id,
                Observation.telescope == tid,
                Observation.night == night,
                Observation.filter_name == band,
                Observation.is_calibration == False,
                Observation.obs_type == "imaging",
                Observation.status == "done",
            )
            .all()
        )
    finally:
        db.close()


def auto_order(n_pixels: int) -> int:
    """Pick HiPS order based on image size (rough heuristic)."""
    if n_pixels < 2048:
        return 5
    elif n_pixels < 4096:
        return 6
    elif n_pixels < 8192:
        return 7
    else:
        return 8


def run_local_hips(input_dir: str, output_dir: str,
                   order: int | None = None,
                   mode: str = "MEAN",
                   timeout: int = 300) -> dict:
    """
    Run hipsgen_cuda directly on a local directory.
    Used by the worker after download and by validation scripts.
    """
    os.makedirs(output_dir, exist_ok=True)
    if order is None:
        sample = fits.getheader(os.path.join(input_dir, os.listdir(input_dir)[0]))
        npix = max(sample.get("NAXIS1", 2048), sample.get("NAXIS2", 2048))
        order = auto_order(npix)

    cmd = [HIPSGEN, input_dir, output_dir, "-order", str(order), "-mode", mode]
    log.info("Running local HiPS build: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("hipsgen_cuda failed:\nstdout: %s\nstderr: %s",
                  result.stdout, result.stderr)
        return {"ok": False, "order": order, "error": result.stderr or result.stdout}

    n_tiles = sum(1 for _ in Path(output_dir).rglob("Npix*.fits"))
    return {
        "ok": True,
        "order": order,
        "n_tiles": n_tiles,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": output_dir,
    }


# ─────────────────── main entry ───────────────────

def generate_epoch_hips(event_id: str, tid: str, night: str, band: str,
                        order: int | None = None, cleanup: bool = True) -> dict:
    """
    Build epoch HiPS for one (event, telescope, night, band) group.

    Returns dict with keys: n_frames, n_tiles_uploaded, oss_prefix, order
    """
    tag = f"{event_id}/{tid}/{night}/{band}"
    log.info("HiPS start: %s", tag)
    if upload_directory is None or build_hips_key is None:
        return {"n_frames": 0, "n_tiles_uploaded": 0, "error": "S3 dependencies unavailable"}

    obs_list = _group_observations(event_id, tid, night, band)
    if not obs_list:
        log.warning("No done observations for group %s", tag)
        return {"n_frames": 0, "n_tiles_uploaded": 0, "error": "no observations"}

    # ── scratch dirs ──
    work = os.path.join(SCRATCH_BASE, f"hips_{event_id}_{tid}_{night}_{band}")
    input_dir = os.path.join(work, "input")
    output_dir = os.path.join(work, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ── 1. Download FITS ──
        for obs in obs_list:
            dest = os.path.join(input_dir, os.path.basename(obs.fits_key))
            _download_from_oss(obs.fits_key, dest)
            log.info("Downloaded %s", obs.fits_key)

        # ── 2. Determine order ──
        if order is None:
            sample = fits.getheader(os.path.join(input_dir, os.listdir(input_dir)[0]))
            npix = max(sample.get("NAXIS1", 2048), sample.get("NAXIS2", 2048))
            order = auto_order(npix)
        log.info("Using HiPS order=%d for %d frames", order, len(obs_list))

        # ── 3. Run hipsgen_cuda ──
        build = run_local_hips(input_dir, output_dir, order=order, mode="MEAN", timeout=300)
        if not build.get("ok"):
            return {"n_frames": len(obs_list), "n_tiles_uploaded": 0,
                    "error": build.get("error", "hips build failed")}
        log.info("hipsgen_cuda finished OK")

        # ── 4. Upload tiles to OSS ──
        prefix = build_hips_key(event_id, "epochs", tid=tid, night=night, band=band)
        n_uploaded = upload_directory(output_dir, prefix)
        log.info("Uploaded %d tiles → %s", n_uploaded, prefix)

        return {
            "n_frames": len(obs_list),
            "n_tiles_uploaded": n_uploaded,
            "oss_prefix": prefix,
            "order": order,
        }

    finally:
        if cleanup and os.path.exists(work):
            shutil.rmtree(work, ignore_errors=True)
            log.debug("Cleaned scratch %s", work)


def diff_hips(event_id: str, tid: str, night: str, band: str,
              candidate_positions: Optional[list[dict]] = None,
              reference_event_id: Optional[str] = None,
              reference_local_dir: Optional[str] = None,
              epoch_local_dir: Optional[str] = None,
              reference_source_dir: Optional[str] = None,
              epoch_source_dir: Optional[str] = None,
              output_local_dir: Optional[str] = None,
              threshold_sigma: float = 5.0,
              cutout_half_size: int = 25,
              detection_radius: int = 4,
              cleanup: bool = True) -> dict:
    """
    Use HiPS/HpxFinder for provenance lookup, then perform difference imaging
    in the original-image domain around supplied sky positions.

    This implements the science-safe path:
      HiPS/HpxFinder for localization/provenance,
      original images for differencing and measurement.

    candidate_positions:
      list of dicts with at least {ra, dec}. Optional fields are preserved.
    """
    tag = f"{event_id}/{tid}/{night}/{band}"
    work = os.path.join(SCRATCH_BASE, f"diff_{event_id}_{tid}_{night}_{band}")
    epoch_dir = epoch_local_dir
    ref_dir = reference_local_dir

    if not candidate_positions:
        return {
            "status": "error",
            "error": "candidate_positions required for original-domain diff validation",
            "candidates": [],
        }

    try:
        if epoch_dir is None:
            if download_prefix is None or build_hips_key is None:
                raise RuntimeError("S3 dependencies unavailable for remote diff mode")
            epoch_dir = os.path.join(work, "epoch")
            prefix = build_hips_key(event_id, "epochs", tid=tid, night=night, band=band)
            n = download_prefix(prefix, epoch_dir)
            log.info("Downloaded %d epoch HiPS files for %s", n, tag)

        if ref_dir is None:
            if download_prefix is None or build_hips_key is None:
                raise RuntimeError("S3 dependencies unavailable for remote diff mode")
            ref_dir = os.path.join(work, "reference")
            ref_event = reference_event_id or event_id
            prefix = build_hips_key(ref_event, "reference", band=band)
            n = download_prefix(prefix, ref_dir)
            log.info("Downloaded %d reference HiPS files for %s", n, tag)

        if output_local_dir is None:
            output_local_dir = os.path.join(work, "diff_output")
        os.makedirs(output_local_dir, exist_ok=True)

        if epoch_source_dir is None:
            raise ValueError("epoch_source_dir is required for original-image differencing")
        if reference_source_dir is None:
            raise ValueError("reference_source_dir is required for original-image differencing")

        epoch_sources = build_source_map(epoch_source_dir)
        ref_sources = build_source_map(reference_source_dir)

        results = []
        order = highest_hips_order(epoch_dir)
        for idx, pos in enumerate(candidate_positions):
            ra = float(pos["ra"])
            dec = float(pos["dec"])
            epoch_hit = query_hpxfinder_sources_for_sky(epoch_dir, ra, dec, order=order)
            ref_hit = query_hpxfinder_sources_for_sky(ref_dir, ra, dec, order=order)

            rec = {
                **pos,
                "candidate_id": int(idx),
                "order": order,
                "epoch_tile_npix": epoch_hit["npix"] if epoch_hit else None,
                "reference_tile_npix": ref_hit["npix"] if ref_hit else None,
                "epoch_source_files": epoch_hit["source_files"] if epoch_hit else [],
                "reference_source_files": ref_hit["source_files"] if ref_hit else [],
            }

            if not epoch_hit or not ref_hit:
                rec.update({"recovered": False, "reason": "no_hpxfinder_hit"})
                results.append(rec)
                continue

            epoch_path, _, _ = select_source_image_for_sky(
                epoch_sources, epoch_hit["source_files"], ra, dec)
            ref_path, _, _ = select_source_image_for_sky(
                ref_sources, ref_hit["source_files"], ra, dec)

            rec["epoch_source_path"] = epoch_path
            rec["reference_source_path"] = ref_path
            if not epoch_path or not ref_path:
                rec.update({"recovered": False, "reason": "no_source_image"})
                results.append(rec)
                continue

            diff = original_domain_difference(
                epoch_path, ref_path, ra, dec,
                cutout_half_size=cutout_half_size,
                detection_radius=detection_radius,
                threshold_sigma=threshold_sigma,
            )
            rec.update(diff)
            results.append(rec)

        out_json = os.path.join(output_local_dir, "candidates.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return {
            "status": "ok",
            "n_candidates": len(results),
            "n_recovered": int(sum(1 for r in results if r.get("recovered"))),
            "order": order,
            "candidates_json": out_json,
            "candidates": results,
        }
    finally:
        if cleanup and not epoch_local_dir and not reference_local_dir and os.path.exists(work):
            shutil.rmtree(work, ignore_errors=True)


# ─────────────────── CLI for manual runs ───────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Generate epoch HiPS for a group")
    p.add_argument("event_id")
    p.add_argument("tid")
    p.add_argument("night")
    p.add_argument("band")
    p.add_argument("--order", type=int, default=None)
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--input-dir", type=str, default=None,
                   help="Optional local input directory for direct HiPS generation")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Optional local output directory for direct HiPS generation")
    args = p.parse_args()

    if args.input_dir and args.output_dir:
        res = run_local_hips(args.input_dir, args.output_dir,
                             order=args.order, mode="MEAN")
    else:
        res = generate_epoch_hips(args.event_id, args.tid, args.night, args.band,
                                  order=args.order, cleanup=not args.no_cleanup)
    print(json.dumps(res, indent=2))
