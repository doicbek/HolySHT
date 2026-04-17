"""User-facing alm2map / map2alm, mirroring the MATLAB +holysht API."""

import numpy as np
from ._ring_info import healpix_ring_info


def _get_core():
    """Lazy import of the C++ extension."""
    from . import _holysht_core
    return _holysht_core


def alm2map(alm, spin, nside, lmax, *, lat_range=None, nthreads=0):
    """Synthesise a HEALPix map from spherical-harmonic coefficients.

    Parameters
    ----------
    alm : complex ndarray, shape (ncomp, nalm) or (N, ncomp, nalm)
        Spherical-harmonic coefficients in DUCC mstart ordering.
        ncomp = 1 for spin-0, 2 for spin-2.
        nalm = (lmax+1)*(lmax+2)/2.
    spin : {0, 2}
        Spin of the transform.
    nside : int
        HEALPix resolution parameter.
    lmax : int
        Maximum multipole order.
    lat_range : (float, float) or None
        Latitude band in degrees (+90 = north pole).
    nthreads : int
        Thread count (0 = auto).

    Returns
    -------
    map : real ndarray, shape (ncomp, npix) or (N, ncomp, npix)
        Precision matches input alm.
    """
    if spin not in (0, 2):
        raise ValueError("spin must be 0 or 2")
    nside = int(nside)
    lmax = int(lmax)
    if nside < 0:
        raise ValueError("nside must be non-negative")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")

    info = healpix_ring_info(nside, lat_range)
    ncomp = 1 if spin == 0 else 2
    nalm = (lmax + 1) * (lmax + 2) // 2

    alm = np.asarray(alm)
    if not np.iscomplexobj(alm):
        alm = alm.astype(np.complex128)

    # Reshape 1-D vector to (ncomp, nalm)
    if alm.ndim == 1:
        alm = alm.reshape(ncomp, nalm)

    is_single = alm.dtype in (np.complex64,)

    core = _get_core()
    if is_single:
        alm = np.ascontiguousarray(alm, dtype=np.complex64)
        return core.alm2map_f32(
            alm, lmax, spin,
            info.theta, info.nphi, info.phi0, info.ringstart,
            nthreads)
    else:
        alm = np.ascontiguousarray(alm, dtype=np.complex128)
        return core.alm2map_f64(
            alm, lmax, spin,
            info.theta, info.nphi, info.phi0, info.ringstart,
            nthreads)


def map2alm(map, spin, nside, lmax, *, n_iter=3, lat_range=None, nthreads=0):
    """Compute spherical-harmonic coefficients from a HEALPix map.

    Uses adjoint synthesis + Jacobi (Richardson) iteration.

    Parameters
    ----------
    map : real ndarray, shape (ncomp, npix) or (N, ncomp, npix)
        HEALPix map(s). ncomp = 1 for spin-0, 2 for spin-2.
    spin : {0, 2}
        Spin of the transform.
    nside : int
        HEALPix resolution parameter.
    lmax : int
        Maximum multipole order.
    n_iter : int
        Number of Jacobi refinement iterations (default 3).
    lat_range : (float, float) or None
        Latitude band in degrees (+90 = north pole).
    nthreads : int
        Thread count (0 = auto).

    Returns
    -------
    alm : complex ndarray, shape (ncomp, nalm) or (N, ncomp, nalm)
        nalm = (lmax+1)*(lmax+2)/2.  Precision matches input map.
    """
    if spin not in (0, 2):
        raise ValueError("spin must be 0 or 2")
    nside = int(nside)
    lmax = int(lmax)
    if nside < 0:
        raise ValueError("nside must be non-negative")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")

    info = healpix_ring_info(nside, lat_range)
    ncomp = 1 if spin == 0 else 2

    map = np.asarray(map, dtype=np.float64 if not hasattr(map, 'dtype') else None)

    # Reshape 1-D vector to (ncomp, npix)
    if map.ndim == 1:
        map = map.reshape(ncomp, info.npix)

    is_single = map.dtype == np.float32

    core = _get_core()
    if is_single:
        map = np.ascontiguousarray(map, dtype=np.float32)
        return core.map2alm_f32(
            map, lmax, spin,
            info.theta, info.nphi, info.phi0, info.ringstart,
            info.weight, n_iter, nthreads)
    else:
        map = np.ascontiguousarray(map, dtype=np.float64)
        return core.map2alm_f64(
            map, lmax, spin,
            info.theta, info.nphi, info.phi0, info.ringstart,
            info.weight, n_iter, nthreads)
