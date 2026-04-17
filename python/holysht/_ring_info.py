"""HEALPix ring geometry, matching +holysht/private/healpix_ring_info.m."""

import numpy as np
from dataclasses import dataclass


@dataclass
class RingInfo:
    theta: np.ndarray      # colatitudes (radians)
    phi0: np.ndarray       # azimuth of first pixel per ring
    nphi: np.ndarray       # pixels per ring (int64)
    ringstart: np.ndarray  # 0-based start index per ring (int64)
    weight: float          # quadrature weight = 4*pi / npix_full
    npix: int              # total pixels in selected rings


def healpix_ring_info(nside, lat_range=None):
    """Compute HEALPix ring geometry, optionally restricted to a latitude band.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    lat_range : (float, float) or None
        If given, (lat_min, lat_max) in degrees (+90 = north pole).
        Only rings within this latitude band are returned.

    Returns
    -------
    RingInfo
        Ring geometry struct with fields theta, phi0, nphi, ringstart,
        weight, npix.
    """
    nside = int(nside)
    npix0 = 12 * nside * nside
    nring = 4 * nside - 1
    rings = np.arange(1, nring + 1, dtype=np.float64)

    # "North-folded" ring index
    northrings = rings.copy()
    south = rings > 2 * nside
    northrings[south] = 4 * nside - rings[south]

    theta = np.zeros(nring)
    phi0 = np.zeros(nring)
    nphi = np.zeros(nring, dtype=np.int64)

    # Polar cap rings (northring < nside)
    cap = northrings < nside
    theta[cap] = 2.0 * np.arcsin(northrings[cap] / (np.sqrt(6.0) * nside))
    nphi[cap] = (4 * northrings[cap]).astype(np.int64)
    phi0[cap] = np.pi / (4.0 * northrings[cap])

    # Equatorial band rings
    rest = ~cap
    theta[rest] = np.arccos((2.0 * nside - northrings[rest]) * (8.0 * nside / npix0))
    nphi[rest] = 4 * nside
    phi0[rest] = (np.pi / (4.0 * nside)) * ((northrings[rest] - nside) % 2 == 0).astype(np.float64)

    # Southern hemisphere: theta -> pi - theta
    theta[south] = np.pi - theta[south]

    weight = 4.0 * np.pi / npix0

    # Latitude filtering
    if lat_range is not None:
        lat_min, lat_max = lat_range
        if lat_min >= lat_max:
            raise ValueError("lat_range must have lat_range[0] < lat_range[1]")
        if lat_min < -90 or lat_max > 90:
            raise ValueError("lat_range entries must lie in [-90, 90]")
        theta_min = (90.0 - lat_max) * np.pi / 180.0
        theta_max = (90.0 - lat_min) * np.pi / 180.0
        sel = (theta >= theta_min) & (theta <= theta_max)
        theta = theta[sel]
        phi0 = phi0[sel]
        nphi = nphi[sel]
        if len(theta) == 0:
            raise ValueError(f"lat_range selects zero HEALPix rings at nside={nside}")

    # Repack ringstart contiguously
    ringstart = np.zeros(len(nphi), dtype=np.int64)
    if len(nphi) > 1:
        ringstart[1:] = np.cumsum(nphi[:-1])

    total_npix = int(np.sum(nphi))

    return RingInfo(
        theta=theta,
        phi0=phi0,
        nphi=nphi,
        ringstart=ringstart,
        weight=weight,
        npix=total_npix,
    )
