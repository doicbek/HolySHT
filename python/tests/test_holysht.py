"""Tests mirroring test/test_holysht.m."""

import numpy as np
import holysht


def mstart(lmax):
    """Return mstart array for DUCC ordering."""
    return np.array([m * (2 * lmax + 1 - m) // 2 for m in range(lmax + 1)])


def random_alm(ncomp, lmax, rng, dtype=np.complex128):
    """Random alm with real m=0 modes (physical convention for real maps)."""
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = (rng.standard_normal((ncomp, nalm)) +
           1j * rng.standard_normal((ncomp, nalm))).astype(dtype)
    # m=0 modes must be real for spin-0 real-valued maps
    ms = mstart(lmax)
    for l in range(lmax + 1):
        alm[:, ms[0] + l] = alm[:, ms[0] + l].real
    return alm


def relerr_l2(a, b):
    return np.linalg.norm(a.ravel() - b.ravel()) / np.linalg.norm(a.ravel())


def test_spin0_roundtrip():
    """Spin-0 alm -> map -> alm round-trip, L2 relative error < 1e-10."""
    nside, lmax = 64, 128
    rng = np.random.default_rng(42)
    alm = random_alm(1, lmax, rng)

    map_ = holysht.alm2map(alm, 0, nside, lmax)
    alm2 = holysht.map2alm(map_, 0, nside, lmax, n_iter=10)

    err = relerr_l2(alm, alm2)
    print(f"  spin-0 round-trip L2 error = {err:.2e}")
    assert err < 1e-10, f"Spin-0 round-trip error {err:.2e}"


def test_spin2_roundtrip():
    """Spin-2 round-trip with l<2 zeroed, L2 relative error < 1e-9."""
    nside, lmax = 64, 128
    rng = np.random.default_rng(123)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = (rng.standard_normal((2, nalm)) +
           1j * rng.standard_normal((2, nalm)))

    # Zero out l < |spin| = 2
    ms = mstart(lmax)
    for l in range(2):
        for m in range(l + 1):
            alm[:, ms[m] + l] = 0

    # m=0 modes must be real for real-valued Q/U maps
    for l in range(lmax + 1):
        alm[:, ms[0] + l] = alm[:, ms[0] + l].real

    map_ = holysht.alm2map(alm, 2, nside, lmax)
    alm2 = holysht.map2alm(map_, 2, nside, lmax, n_iter=10)

    err = relerr_l2(alm, alm2)
    print(f"  spin-2 round-trip L2 error = {err:.2e}")
    assert err < 1e-9, f"Spin-2 round-trip error {err:.2e}"


def test_batch_consistency():
    """Batch alm2map matches looping over single calls."""
    nside, lmax = 32, 64
    rng = np.random.default_rng(7)
    N = 4
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm_batch = rng.standard_normal((N, 1, nalm)) + 1j * rng.standard_normal((N, 1, nalm))

    map_batch = holysht.alm2map(alm_batch, 0, nside, lmax)

    for i in range(N):
        map_single = holysht.alm2map(alm_batch[i], 0, nside, lmax)
        np.testing.assert_allclose(map_batch[i], map_single, atol=1e-14)


def test_lat_range_size_reduction():
    """lat_range reduces the number of pixels."""
    nside = 64
    info_full = holysht.healpix_ring_info(nside)
    info_part = holysht.healpix_ring_info(nside, lat_range=(-30, 30))
    assert info_part.npix < info_full.npix
    print(f"  full npix = {info_full.npix}, band npix = {info_part.npix}")


def test_partial_sky():
    """Partial-sky round-trip produces finite values."""
    nside, lmax = 64, 128
    lat_range = (-30, 30)
    rng = np.random.default_rng(99)
    alm = random_alm(1, lmax, rng)

    map_ = holysht.alm2map(alm, 0, nside, lmax, lat_range=lat_range)
    alm2 = holysht.map2alm(map_, 0, nside, lmax, lat_range=lat_range)

    assert np.all(np.isfinite(map_))
    assert np.all(np.isfinite(alm2))


def test_single_precision():
    """Single-precision round-trip, L2 error < 1e-3."""
    nside, lmax = 64, 128
    rng = np.random.default_rng(55)
    alm = random_alm(1, lmax, rng, dtype=np.complex64)

    map_ = holysht.alm2map(alm, 0, nside, lmax)
    assert map_.dtype == np.float32
    alm2 = holysht.map2alm(map_, 0, nside, lmax, n_iter=5)
    assert alm2.dtype == np.complex64

    err = relerr_l2(alm.astype(np.complex128), alm2.astype(np.complex128))
    print(f"  single-precision L2 error = {err:.2e}")
    assert err < 1e-3, f"Single-precision error {err:.2e}"


def test_error_handling():
    """Bad spin and bad lat_range should raise ValueError."""
    try:
        holysht.alm2map(np.zeros((1, 3), dtype=complex), 1, 4, 2)
        assert False, "spin=1 should have errored"
    except ValueError:
        pass

    try:
        holysht.healpix_ring_info(64, lat_range=(30, -30))
        assert False, "inverted lat_range should have errored"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        ("Spin-0 round-trip", test_spin0_roundtrip),
        ("Spin-2 round-trip", test_spin2_roundtrip),
        ("Batch consistency", test_batch_consistency),
        ("lat_range size reduction", test_lat_range_size_reduction),
        ("Partial sky", test_partial_sky),
        ("Single precision", test_single_precision),
        ("Error handling", test_error_handling),
    ]
    for name, fn in tests:
        print(f"=== {name} ===")
        try:
            fn()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
    print("\nDone.")
