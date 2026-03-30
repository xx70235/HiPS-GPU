"""
calibrate.py — Basic CCD image calibration.

Given a science frame and available calibration frames (bias/flat)
from the same telescope+night, produce a reduced image.

    reduced = (raw - master_bias) / normalized_master_flat
"""
import numpy as np
from astropy.io import fits
import logging, os, tempfile

log = logging.getLogger("pipeline.calibrate")


def make_master(frames: list[str], method: str = "median") -> np.ndarray | None:
    """Combine multiple calibration frames into a master."""
    if not frames:
        return None
    stack = []
    for f in frames:
        try:
            data = fits.getdata(f, ext=0).astype(np.float64)
            stack.append(data)
        except Exception as e:
            log.warning(f"Cannot read {f}: {e}")
    if not stack:
        return None
    cube = np.array(stack)
    if method == "median":
        master = np.median(cube, axis=0)
    else:
        master = np.mean(cube, axis=0)
    log.info(f"Master {method} from {len(stack)} frames, shape={master.shape}")
    return master


def reduce_science(science_path: str,
                   bias_frames: list[str] = None,
                   flat_frames: list[str] = None) -> tuple[np.ndarray, fits.Header]:
    """
    Apply calibration to a science frame.

    Returns (data, header) — calibrated 2D array and original header.
    """
    hdr = fits.getheader(science_path, ext=0)
    data = fits.getdata(science_path, ext=0).astype(np.float64)
    log.info(f"Science frame: {os.path.basename(science_path)}, "
             f"shape={data.shape}, median={np.median(data):.1f}")

    # Bias subtraction
    if bias_frames:
        master_bias = make_master(bias_frames, "median")
        if master_bias is not None and master_bias.shape == data.shape:
            data = data - master_bias
            log.info(f"Bias subtracted (master median={np.median(master_bias):.1f})")

    # Flat division
    if flat_frames:
        master_flat = make_master(flat_frames, "median")
        if master_flat is not None and master_flat.shape == data.shape:
            # Normalize flat to median=1
            norm = np.median(master_flat)
            if norm > 0:
                master_flat = master_flat / norm
                # Avoid division by very small values
                master_flat[master_flat < 0.1] = 1.0
                data = data / master_flat
                log.info(f"Flat divided (flat norm={norm:.1f})")

    return data, hdr
