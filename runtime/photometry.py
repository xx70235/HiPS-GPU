"""
photometry.py — Aperture photometry + PS1 zero-point calibration.

Pipeline:
  1. Background estimation + subtraction
  2. Source detection (DAOStarFinder)
  3. WCS → pixel coord for target (ra, dec)
  4. Aperture photometry at target position
  5. Field star cross-match with PS1 for zero point
  6. Return calibrated magnitude

Returns a dict: {mag, mag_err, flux, flux_err, zp, lim_mag, n_ref_stars, calibrated}
"""
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Iterable

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

try:
    from photutils.detection import DAOStarFinder
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from photutils.background import Background2D, MedianBackground
    HAS_PHOTUTILS = True
except Exception:  # pragma: no cover - fallback for minimal environments
    DAOStarFinder = None
    CircularAperture = None
    CircularAnnulus = None
    aperture_photometry = None
    Background2D = None
    MedianBackground = None
    HAS_PHOTUTILS = False

log = logging.getLogger("pipeline.photometry")

# ── Config ──
APERTURE_RADIUS = 8.0      # pixels (will adapt to seeing if available)
ANNULUS_INNER = 12.0
ANNULUS_OUTER = 18.0
DETECTION_FWHM = 5.0
DETECTION_THRESHOLD = 5.0  # sigma above background
PS1_SEARCH_RADIUS = 0.15   # degrees
PS1_MAG_BRIGHT = 14.0      # min mag for reference stars
PS1_MAG_FAINT = 20.0       # max mag for reference stars
MATCH_RADIUS = 2.0         # arcsec for cross-match


@dataclass
class PhotometryResult:
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    flux: Optional[float] = None
    flux_err: Optional[float] = None
    zp: Optional[float] = None
    lim_mag: Optional[float] = None
    n_ref_stars: int = 0
    calibrated: bool = False
    mjd: Optional[float] = None
    snr: Optional[float] = None


def combine_flux_measurements(results: Iterable[PhotometryResult]) -> PhotometryResult:
    """
    Inverse-variance combine flux measurements from multiple images.

    Only results with finite positive flux_err are used.
    """
    good = [
        r for r in results
        if r.flux is not None and r.flux_err is not None
        and np.isfinite(r.flux) and np.isfinite(r.flux_err)
        and r.flux_err > 0
    ]
    out = PhotometryResult()
    if not good:
        return out

    weights = np.array([1.0 / (r.flux_err ** 2) for r in good], dtype=float)
    fluxes = np.array([r.flux for r in good], dtype=float)
    out.flux = float(np.sum(weights * fluxes) / np.sum(weights))
    out.flux_err = float(1.0 / np.sqrt(np.sum(weights)))
    out.snr = float(out.flux / out.flux_err) if out.flux_err > 0 else None
    return out


def estimate_background(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Estimate and subtract 2D background."""
    try:
        if not HAS_PHOTUTILS:
            raise RuntimeError("photutils unavailable")
        bkg = Background2D(data, (64, 64),
                           filter_size=(3, 3),
                           sigma_clip=SigmaClip(sigma=3.0),
                           bkg_estimator=MedianBackground())
        return data - bkg.background, bkg.background_median, bkg.background_rms_median
    except Exception:
        # Fallback: simple sigma-clipped stats
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        return data - median, median, std


def detect_sources(data: np.ndarray, bkg_rms: float,
                   fwhm: float = DETECTION_FWHM) -> list:
    """Detect point sources."""
    if not HAS_PHOTUTILS:
        log.warning("photutils unavailable: detect_sources fallback returns []")
        return []
    finder = DAOStarFinder(fwhm=fwhm, threshold=DETECTION_THRESHOLD * bkg_rms)
    sources = finder(data)
    if sources is None:
        return []
    log.info(f"Detected {len(sources)} sources")
    return sources


def aperture_phot(data: np.ndarray, x: float, y: float,
                  r_ap: float = APERTURE_RADIUS,
                  r_in: float = ANNULUS_INNER,
                  r_out: float = ANNULUS_OUTER) -> tuple[float, float]:
    """Aperture photometry at (x,y) with local sky annulus."""
    if not HAS_PHOTUTILS:
        yy, xx = np.indices(data.shape)
        rr2 = (xx - x) ** 2 + (yy - y) ** 2
        ap_mask = rr2 <= r_ap ** 2
        ann_mask = (rr2 >= r_in ** 2) & (rr2 <= r_out ** 2)
        if not np.any(ap_mask):
            return 0.0, float("inf")
        ann_vals = data[ann_mask]
        sky_per_pixel = float(np.median(ann_vals)) if ann_vals.size else 0.0
        ap_vals = data[ap_mask]
        net_flux = float(np.sum(ap_vals - sky_per_pixel))
        noise = float(np.sqrt(np.abs(net_flux) + ap_mask.sum() * max(np.var(ann_vals), 0.0)))
        if noise <= 0:
            noise = float(np.sqrt(ap_mask.sum()) * max(np.std(ann_vals), 1e-6))
        return net_flux, noise

    aperture = CircularAperture((x, y), r=r_ap)
    annulus = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)

    phot = aperture_photometry(data, [aperture, annulus])
    # Local sky per pixel
    ann_area = annulus.area
    sky_per_pixel = phot["aperture_sum_1"][0] / ann_area
    # Net flux
    net_flux = phot["aperture_sum_0"][0] - sky_per_pixel * aperture.area
    # Noise: Poisson + sky + readnoise (approximate)
    noise = np.sqrt(np.abs(net_flux) + aperture.area * sky_per_pixel)
    return float(net_flux), float(noise)


def query_ps1(ra: float, dec: float, radius_deg: float,
              filter_name: str) -> list[dict]:
    """Query PS1 DR2 for reference stars near (ra, dec)."""
    try:
        from astroquery.vizier import Vizier
        # PS1 DR2 catalog
        v = Vizier(columns=["RAJ2000", "DEJ2000",
                            "gmag", "rmag", "imag", "zmag", "ymag",
                            "e_gmag", "e_rmag", "e_imag", "e_zmag", "e_ymag"],
                   row_limit=500)
        coord = SkyCoord(ra, dec, unit="deg", frame="icrs")
        result = v.query_region(coord, radius=radius_deg * u.deg,
                                catalog="II/349/ps1")
        if not result:
            log.warning("PS1 query returned no results")
            return []
        tab = result[0]

        # Map filter to PS1 column
        filt_map = {"g": "gmag", "r": "rmag", "i": "imag",
                    "z": "zmag", "y": "ymag",
                    "G": "gmag", "R": "rmag", "I": "imag",
                    "B": "gmag", "V": "rmag"}  # rough mapping for B/V
        mag_col = filt_map.get(filter_name)
        err_col = f"e_{mag_col}" if mag_col else None
        if not mag_col or mag_col not in tab.colnames:
            log.warning(f"No PS1 column for filter '{filter_name}'")
            return []

        stars = []
        for row in tab:
            m = float(row[mag_col])
            if np.isnan(m) or m < PS1_MAG_BRIGHT or m > PS1_MAG_FAINT:
                continue
            me = float(row[err_col]) if err_col and err_col in tab.colnames else 0.01
            if np.isnan(me):
                me = 0.05
            stars.append({
                "ra": float(row["RAJ2000"]),
                "dec": float(row["DEJ2000"]),
                "mag": m,
                "mag_err": me,
            })
        log.info(f"PS1: {len(stars)} reference stars in {filter_name}-band")
        return stars
    except Exception as e:
        log.warning(f"PS1 query failed: {e}")
        return []


def compute_zeropoint(data: np.ndarray, wcs: WCS,
                      ref_stars: list[dict],
                      r_ap: float = APERTURE_RADIUS) -> tuple[float, float, int]:
    """
    Cross-match detected sources with PS1 reference stars,
    compute instrumental magnitude and derive zero point.

    Returns (zp, zp_err, n_matched)
    """
    zps = []
    for star in ref_stars:
        try:
            px, py = wcs.all_world2pix(star["ra"], star["dec"], 0)
            px, py = float(px), float(py)
        except Exception:
            continue
        # Check within image
        ny, nx = data.shape
        if px < r_ap or px > nx - r_ap or py < r_ap or py > ny - r_ap:
            continue
        flux, noise = aperture_phot(data, px, py, r_ap)
        if flux <= 0:
            continue
        inst_mag = -2.5 * np.log10(flux)
        zp = star["mag"] - inst_mag
        zps.append(zp)

    if len(zps) < 3:
        return 0.0, 0.0, len(zps)

    zps = np.array(zps)
    # Sigma-clip outliers
    med = np.median(zps)
    mad = np.median(np.abs(zps - med))
    mask = np.abs(zps - med) < 3 * max(mad, 0.05)
    zps = zps[mask]

    if len(zps) < 3:
        return float(med), float(mad), len(zps)

    return float(np.median(zps)), float(np.std(zps) / np.sqrt(len(zps))), len(zps)


def run_photometry(data: np.ndarray, header: fits.Header,
                   target_ra: float, target_dec: float,
                   filter_name: str = "",
                   seeing: float = None,
                   calibrate: bool = True) -> PhotometryResult:
    """
    Full photometry pipeline on a single frame.

    Args:
        data: calibrated 2D image
        header: FITS header (needs WCS and DATE-OBS)
        target_ra, target_dec: event coordinates
        filter_name: filter band for PS1 cross-match
        seeing: FWHM in arcsec (optional, for aperture sizing)
        calibrate: when False, skip external catalog zeropoint calibration
    """
    result = PhotometryResult()

    # MJD from DATE-OBS
    date_obs = header.get("DATE-OBS", "")
    if date_obs:
        try:
            result.mjd = Time(date_obs, format="isot").mjd
        except Exception:
            pass

    # WCS
    try:
        wcs = WCS(header)
        if not wcs.has_celestial:
            log.warning("No valid WCS — cannot do positional photometry")
            return result
    except Exception as e:
        log.warning(f"WCS parse failed: {e}")
        return result

    # Adapt aperture to seeing
    r_ap = APERTURE_RADIUS
    if seeing and seeing > 0:
        # plate scale from WCS
        try:
            ps = np.abs(wcs.wcs.cdelt[0]) * 3600  # arcsec/pix
            if ps > 0:
                r_ap = max(3.0, 1.5 * seeing / ps)  # 1.5× FWHM
                log.info(f"Aperture adapted: r={r_ap:.1f} pix "
                         f"(seeing={seeing:.1f}\", ps={ps:.2f}\"/pix)")
        except Exception:
            pass

    # Background
    data_sub, bkg_med, bkg_rms = estimate_background(data)
    log.info(f"Background: median={bkg_med:.1f}, rms={bkg_rms:.2f}")

    # Target pixel coordinates
    try:
        tx, ty = wcs.all_world2pix(target_ra, target_dec, 0)
        tx, ty = float(tx), float(ty)
    except Exception:
        log.warning("Cannot convert target (ra,dec) to pixel")
        return result

    ny, nx = data_sub.shape
    if tx < 0 or tx >= nx or ty < 0 or ty >= ny:
        log.warning(f"Target pixel ({tx:.0f},{ty:.0f}) outside image {nx}×{ny}")
        return result

    # Aperture photometry on target
    flux, flux_err = aperture_phot(data_sub, tx, ty, r_ap)
    result.flux = flux
    result.flux_err = flux_err
    result.snr = flux / flux_err if flux_err > 0 else 0

    zp_err = 0.0
    ref_stars = []
    if calibrate:
        # Zero point from PS1
        ref_stars = query_ps1(target_ra, target_dec, PS1_SEARCH_RADIUS, filter_name)
        if ref_stars:
            zp, zp_err, n_matched = compute_zeropoint(data_sub, wcs, ref_stars, r_ap)
            result.n_ref_stars = n_matched
            if n_matched >= 3 and zp != 0:
                result.zp = zp
                result.calibrated = True
                if flux_err > 0:
                    result.lim_mag = -2.5 * np.log10(5 * flux_err) + zp

    if flux <= 0:
        log.warning(f"Target flux non-positive: {flux:.1f}")
        if result.calibrated and result.lim_mag is not None:
            log.info(f"Calibrated non-detection: lim_mag={result.lim_mag:.3f} "
                     f"(ZP={result.zp:.3f}, {result.n_ref_stars} ref stars)")
        return result

    inst_mag = -2.5 * np.log10(flux)
    if result.calibrated and result.zp is not None:
        # Error propagation: sqrt(flux_err² + zp_err²)
        mag_err_phot = 1.0857 * flux_err / flux
        zp_term = zp_err if ref_stars else 0.0
        result.mag = inst_mag + result.zp
        result.mag_err = np.sqrt(mag_err_phot**2 + zp_term**2)
        log.info(f"Calibrated: mag={result.mag:.3f}±{result.mag_err:.3f} "
                 f"(ZP={result.zp:.3f}±{zp_term:.3f}, {result.n_ref_stars} ref stars)")
        return result

    log.info(f"Flux measured without zeropoint calibration: "
             f"flux={flux:.1f}±{flux_err:.1f}, SNR={result.snr:.1f}")
    return result
