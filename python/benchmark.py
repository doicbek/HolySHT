"""
Benchmark: HolySHT vs healpy for alm2map and map2alm.

Compares wall-clock time across:
  1. nside (problem size) at fixed batch size N=1
  2. batch size N at fixed nside (HolySHT only, healpy has no native batch)

healpy uses its own alm ordering (l-major); HolySHT uses DUCC mstart
(m-major).  We generate random alm in each convention separately so the
benchmark measures pure transform time, not reordering overhead.
"""

import time
import sys
import numpy as np

# --------------- helpers ------------------------------------------------

def random_alm_healpy(lmax, rng):
    """Random alm in healpy ordering (l-major), m=0 modes real."""
    import healpy
    nalm = healpy.Alm.getsize(lmax)
    alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    # m=0 modes real
    for l in range(lmax + 1):
        idx = healpy.Alm.getidx(lmax, l, 0)
        alm[idx] = alm[idx].real
    return alm


def mstart(lmax):
    return np.array([m * (2 * lmax + 1 - m) // 2 for m in range(lmax + 1)])


def random_alm_holysht(ncomp, lmax, rng):
    """Random alm in DUCC mstart ordering, m=0 modes real."""
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.standard_normal((ncomp, nalm)) + 1j * rng.standard_normal((ncomp, nalm))
    ms = mstart(lmax)
    for l in range(lmax + 1):
        alm[:, ms[0] + l] = alm[:, ms[0] + l].real
    return alm


def bench(fn, n_warmup=1, n_repeat=5):
    """Time fn(), returning median wall-clock seconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)

# --------------- benchmark 1: nside scaling -----------------------------

def bench_nside(nthreads):
    import healpy
    sys.path.insert(0, '.')
    sys.path.insert(0, 'build')
    import holysht

    nsides = [64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(42)

    results = []
    print(f"\n{'='*70}")
    print(f"  nside scaling (spin-0, single map, nthreads={nthreads})")
    print(f"{'='*70}")
    print(f"{'nside':>6}  {'lmax':>6}  {'npix':>10}  "
          f"{'healpy a2m':>10}  {'holy a2m':>10}  {'ratio':>6}  "
          f"{'healpy m2a':>10}  {'holy m2a':>10}  {'ratio':>6}")
    print("-" * 90)

    for nside in nsides:
        lmax = 2 * nside
        npix = 12 * nside**2

        # healpy alm
        alm_hp = random_alm_healpy(lmax, rng)
        # holysht alm
        alm_hs = random_alm_holysht(1, lmax, rng)

        # --- alm2map ---
        t_hp_a2m = bench(lambda: healpy.alm2map(alm_hp, nside, lmax=lmax, verbose=False))
        t_hs_a2m = bench(lambda: holysht.alm2map(alm_hs, 0, nside, lmax, nthreads=nthreads))

        # Generate maps for map2alm
        map_hp = healpy.alm2map(alm_hp, nside, lmax=lmax, verbose=False)
        map_hs = holysht.alm2map(alm_hs, 0, nside, lmax, nthreads=nthreads)

        # --- map2alm ---
        t_hp_m2a = bench(lambda: healpy.map2alm(map_hp, lmax=lmax, iter=3, verbose=False))
        t_hs_m2a = bench(lambda: holysht.map2alm(map_hs, 0, nside, lmax, n_iter=3, nthreads=nthreads))

        r_a2m = t_hp_a2m / t_hs_a2m
        r_m2a = t_hp_m2a / t_hs_m2a

        print(f"{nside:>6}  {lmax:>6}  {npix:>10,}  "
              f"{t_hp_a2m:>10.4f}  {t_hs_a2m:>10.4f}  {r_a2m:>5.1f}x  "
              f"{t_hp_m2a:>10.4f}  {t_hs_m2a:>10.4f}  {r_m2a:>5.1f}x")

        results.append(dict(nside=nside, lmax=lmax, npix=npix,
                            healpy_a2m=t_hp_a2m, holysht_a2m=t_hs_a2m,
                            healpy_m2a=t_hp_m2a, holysht_m2a=t_hs_m2a))

    return results

# --------------- benchmark 2: batch size scaling ------------------------

def bench_batch(nthreads):
    sys.path.insert(0, '.')
    sys.path.insert(0, 'build')
    import holysht

    nside = 256
    lmax = 2 * nside
    rng = np.random.default_rng(99)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    results = []
    print(f"\n{'='*70}")
    print(f"  batch size scaling (nside={nside}, lmax={lmax}, nthreads={nthreads})")
    print(f"{'='*70}")
    print(f"{'N':>5}  {'a2m total':>10}  {'a2m/map':>10}  "
          f"{'m2a total':>10}  {'m2a/map':>10}  {'a2m speedup':>11}  {'m2a speedup':>11}")
    print("-" * 80)

    t_a2m_1 = None
    t_m2a_1 = None

    for N in batch_sizes:
        alm_batch = random_alm_holysht(1, lmax, rng).reshape(1, 1, -1)
        alm_batch = np.tile(alm_batch, (N, 1, 1))

        t_a2m = bench(lambda: holysht.alm2map(alm_batch, 0, nside, lmax, nthreads=nthreads))

        map_batch = holysht.alm2map(alm_batch, 0, nside, lmax, nthreads=nthreads)
        t_m2a = bench(lambda: holysht.map2alm(map_batch, 0, nside, lmax, n_iter=3, nthreads=nthreads))

        if t_a2m_1 is None:
            t_a2m_1 = t_a2m
            t_m2a_1 = t_m2a

        # ideal = N * single-map time; speedup = ideal / actual
        speedup_a2m = (N * t_a2m_1) / t_a2m
        speedup_m2a = (N * t_m2a_1) / t_m2a

        print(f"{N:>5}  {t_a2m:>10.4f}  {t_a2m/N:>10.5f}  "
              f"{t_m2a:>10.4f}  {t_m2a/N:>10.5f}  "
              f"{speedup_a2m:>10.2f}x  {speedup_m2a:>10.2f}x")

        results.append(dict(N=N, a2m_total=t_a2m, a2m_per_map=t_a2m/N,
                            m2a_total=t_m2a, m2a_per_map=t_m2a/N,
                            a2m_speedup=speedup_a2m, m2a_speedup=speedup_m2a))

    return results

# --------------- plotting -----------------------------------------------

def make_plots(nside_results, batch_results, nthreads):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Plot 1: alm2map nside scaling ---
    ax = axes[0, 0]
    nsides = [r['nside'] for r in nside_results]
    ax.loglog(nsides, [r['healpy_a2m'] for r in nside_results], 'o-', label='healpy', color='C0')
    ax.loglog(nsides, [r['holysht_a2m'] for r in nside_results], 's-', label='HolySHT', color='C1')
    ax.set_xlabel('nside')
    ax.set_ylabel('Time (s)')
    ax.set_title('alm2map (spin-0, single map)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: map2alm nside scaling ---
    ax = axes[0, 1]
    ax.loglog(nsides, [r['healpy_m2a'] for r in nside_results], 'o-', label='healpy', color='C0')
    ax.loglog(nsides, [r['holysht_m2a'] for r in nside_results], 's-', label='HolySHT', color='C1')
    ax.set_xlabel('nside')
    ax.set_ylabel('Time (s)')
    ax.set_title('map2alm (spin-0, n_iter=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: batch alm2map per-map time ---
    ax = axes[1, 0]
    Ns = [r['N'] for r in batch_results]
    ax.semilogx(Ns, [r['a2m_per_map'] for r in batch_results], 's-', color='C1')
    ax.axhline(batch_results[0]['a2m_per_map'], ls='--', color='gray', alpha=0.5, label='N=1 baseline')
    ax.set_xlabel('Batch size N')
    ax.set_ylabel('Time per map (s)')
    ax.set_title(f'alm2map per-map cost (nside=256)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: batch map2alm per-map time ---
    ax = axes[1, 1]
    ax.semilogx(Ns, [r['m2a_per_map'] for r in batch_results], 's-', color='C1')
    ax.axhline(batch_results[0]['m2a_per_map'], ls='--', color='gray', alpha=0.5, label='N=1 baseline')
    ax.set_xlabel('Batch size N')
    ax.set_ylabel('Time per map (s)')
    ax.set_title(f'map2alm per-map cost (nside=256, n_iter=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'HolySHT benchmark (nthreads={nthreads})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    outpath = 'benchmark_results.png'
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")


# --------------- main ---------------------------------------------------

if __name__ == "__main__":
    import os
    nthreads = int(os.environ.get("HOLYSHT_NTHREADS", 0))
    if nthreads == 0:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
    print(f"Using nthreads={nthreads}")

    nside_results = bench_nside(nthreads)
    batch_results = bench_batch(nthreads)
    make_plots(nside_results, batch_results, nthreads)
