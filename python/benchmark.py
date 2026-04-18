"""
Benchmark: HolySHT vs healpy for alm2map and map2alm.

Compares wall-clock time across:
  1. nside (problem size) at fixed batch size N=1, at 1 and N threads
  2. batch size N at fixed nside (HolySHT only, healpy has no native batch)

healpy uses its own alm ordering (l-major); HolySHT uses DUCC mstart
(m-major).  We generate random alm in each convention separately so the
benchmark measures pure transform time, not reordering overhead.

healpy (>= 1.16) uses DUCC internally and auto-threads via
DUCC0_NUM_THREADS / OMP_NUM_THREADS.
"""

import time
import sys
import os
import numpy as np

# --------------- helpers ------------------------------------------------

def random_alm_healpy(lmax, rng):
    """Random alm in healpy ordering (l-major), m=0 modes real."""
    import healpy
    nalm = healpy.Alm.getsize(lmax)
    alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
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


def set_healpy_threads(n):
    """Control healpy/DUCC thread count via environment variables.
    Must be called before healpy is imported, OR use ducc0 directly."""
    try:
        import ducc0
        ducc0.misc.set_thread_count(n)
    except (ImportError, AttributeError):
        # Fallback: env vars (only effective before first DUCC call)
        os.environ['DUCC0_NUM_THREADS'] = str(n)
        os.environ['OMP_NUM_THREADS'] = str(n)


# --------------- benchmark 1: nside scaling -----------------------------

def bench_nside(nthreads):
    import healpy
    sys.path.insert(0, '.')
    sys.path.insert(0, 'build')
    import holysht

    nsides = [64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(42)

    results_1t = []
    results_nt = []

    for label, nt in [("1", 1), (str(nthreads), nthreads)]:
        results = results_1t if nt == 1 else results_nt
        set_healpy_threads(nt)

        print(f"\n{'='*70}")
        print(f"  nside scaling (spin-0, single map, nthreads={nt})")
        print(f"{'='*70}")
        print(f"{'nside':>6}  {'lmax':>6}  {'npix':>10}  "
              f"{'healpy a2m':>10}  {'holy a2m':>10}  {'ratio':>6}  "
              f"{'healpy m2a':>10}  {'holy m2a':>10}  {'ratio':>6}")
        print("-" * 90)

        for nside in nsides:
            lmax = 2 * nside
            npix = 12 * nside**2

            alm_hp = random_alm_healpy(lmax, rng)
            alm_hs = random_alm_holysht(1, lmax, rng)

            # --- alm2map ---
            t_hp_a2m = bench(lambda: healpy.alm2map(alm_hp, nside, lmax=lmax, verbose=False))
            t_hs_a2m = bench(lambda: holysht.alm2map(alm_hs, 0, nside, lmax, nthreads=nt))

            # Generate maps for map2alm
            map_hp = healpy.alm2map(alm_hp, nside, lmax=lmax, verbose=False)
            map_hs = holysht.alm2map(alm_hs, 0, nside, lmax, nthreads=nt)

            # --- map2alm ---
            t_hp_m2a = bench(lambda: healpy.map2alm(map_hp, lmax=lmax, iter=3, verbose=False))
            t_hs_m2a = bench(lambda: holysht.map2alm(map_hs, 0, nside, lmax, n_iter=3, nthreads=nt))

            r_a2m = t_hp_a2m / t_hs_a2m
            r_m2a = t_hp_m2a / t_hs_m2a

            print(f"{nside:>6}  {lmax:>6}  {npix:>10,}  "
                  f"{t_hp_a2m:>10.4f}  {t_hs_a2m:>10.4f}  {r_a2m:>5.1f}x  "
                  f"{t_hp_m2a:>10.4f}  {t_hs_m2a:>10.4f}  {r_m2a:>5.1f}x")

            results.append(dict(nside=nside, lmax=lmax, npix=npix,
                                healpy_a2m=t_hp_a2m, holysht_a2m=t_hs_a2m,
                                healpy_m2a=t_hp_m2a, holysht_m2a=t_hs_m2a))

    return results_1t, results_nt

# --------------- benchmark 2: batch size scaling ------------------------

def bench_batch(nthreads):
    sys.path.insert(0, '.')
    sys.path.insert(0, 'build')
    import holysht

    nside = 256
    lmax = 2 * nside
    rng = np.random.default_rng(99)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    all_results = {}

    for label, nt in [("1", 1), (str(nthreads), nthreads)]:
        results = []
        all_results[nt] = results

        print(f"\n{'='*70}")
        print(f"  batch size scaling (nside={nside}, lmax={lmax}, nthreads={nt})")
        print(f"{'='*70}")
        print(f"{'N':>5}  {'a2m total':>10}  {'a2m/map':>10}  "
              f"{'m2a total':>10}  {'m2a/map':>10}  {'a2m speedup':>11}  {'m2a speedup':>11}")
        print("-" * 80)

        t_a2m_1 = None
        t_m2a_1 = None

        for N in batch_sizes:
            alm_batch = random_alm_holysht(1, lmax, rng).reshape(1, 1, -1)
            alm_batch = np.tile(alm_batch, (N, 1, 1))

            t_a2m = bench(lambda: holysht.alm2map(alm_batch, 0, nside, lmax, nthreads=nt))

            map_batch = holysht.alm2map(alm_batch, 0, nside, lmax, nthreads=nt)
            t_m2a = bench(lambda: holysht.map2alm(map_batch, 0, nside, lmax, n_iter=3, nthreads=nt))

            if t_a2m_1 is None:
                t_a2m_1 = t_a2m
                t_m2a_1 = t_m2a

            speedup_a2m = (N * t_a2m_1) / t_a2m
            speedup_m2a = (N * t_m2a_1) / t_m2a

            print(f"{N:>5}  {t_a2m:>10.4f}  {t_a2m/N:>10.5f}  "
                  f"{t_m2a:>10.4f}  {t_m2a/N:>10.5f}  "
                  f"{speedup_a2m:>10.2f}x  {speedup_m2a:>10.2f}x")

            results.append(dict(N=N, a2m_total=t_a2m, a2m_per_map=t_a2m/N,
                                m2a_total=t_m2a, m2a_per_map=t_m2a/N,
                                a2m_speedup=speedup_a2m, m2a_speedup=speedup_m2a))

    return all_results

# --------------- plotting -----------------------------------------------

def make_plots(nside_1t, nside_nt, batch_results, nthreads):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
    })

    C_HP = '#4477AA'
    C_HS = '#EE6677'
    C_BAR_A2M = '#EE6677'
    C_BAR_M2A = '#CCBB44'

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    nsides = [r['nside'] for r in nside_1t]

    # ====== Row 1: nside scaling ======

    # -- alm2map --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(nsides, [r['healpy_a2m'] for r in nside_1t],
               'o-', color=C_HP, label='healpy (1 thr)', lw=2, ms=6)
    ax1.loglog(nsides, [r['healpy_a2m'] for r in nside_nt],
               'o--', color=C_HP, label=f'healpy ({nthreads} thr)', lw=1.5, ms=5, alpha=0.6)
    ax1.loglog(nsides, [r['holysht_a2m'] for r in nside_1t],
               's-', color=C_HS, label='HolySHT (1 thr)', lw=2, ms=6)
    ax1.loglog(nsides, [r['holysht_a2m'] for r in nside_nt],
               's--', color=C_HS, label=f'HolySHT ({nthreads} thr)', lw=1.5, ms=5, alpha=0.6)
    ax1.set_xlabel('nside')
    ax1.set_ylabel('Wall time (s)')
    ax1.set_title('alm2map')
    ax1.legend(loc='upper left', fontsize=8.5)
    ax1.grid(True, alpha=0.2, which='both')
    ax1.set_xticks(nsides)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # -- map2alm --
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(nsides, [r['healpy_m2a'] for r in nside_1t],
               'o-', color=C_HP, label='healpy (1 thr)', lw=2, ms=6)
    ax2.loglog(nsides, [r['healpy_m2a'] for r in nside_nt],
               'o--', color=C_HP, label=f'healpy ({nthreads} thr)', lw=1.5, ms=5, alpha=0.6)
    ax2.loglog(nsides, [r['holysht_m2a'] for r in nside_1t],
               's-', color=C_HS, label='HolySHT (1 thr)', lw=2, ms=6)
    ax2.loglog(nsides, [r['holysht_m2a'] for r in nside_nt],
               's--', color=C_HS, label=f'HolySHT ({nthreads} thr)', lw=1.5, ms=5, alpha=0.6)
    ax2.set_xlabel('nside')
    ax2.set_ylabel('Wall time (s)')
    ax2.set_title('map2alm (n_iter=3)')
    ax2.legend(loc='upper left', fontsize=8.5)
    ax2.grid(True, alpha=0.2, which='both')
    ax2.set_xticks(nsides)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # -- Speedup bar chart (1-thread) --
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(nsides))
    w = 0.35
    speedup_a2m = [h['healpy_a2m'] / h['holysht_a2m'] for h in nside_1t]
    speedup_m2a = [h['healpy_m2a'] / h['holysht_m2a'] for h in nside_1t]
    bars1 = ax3.bar(x - w/2, speedup_a2m, w, label='alm2map', color=C_BAR_A2M, alpha=0.8)
    bars2 = ax3.bar(x + w/2, speedup_m2a, w, label='map2alm', color=C_BAR_M2A, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(n) for n in nsides])
    ax3.set_xlabel('nside')
    ax3.set_ylabel('Speedup vs healpy')
    ax3.set_title('Speedup (single-threaded)')
    ax3.axhline(1, color='gray', ls='--', lw=0.8)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.2, axis='y')
    for bar in bars1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                 f'{h:.1f}x', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                 f'{h:.1f}x', ha='center', va='bottom', fontsize=8)

    # ====== Row 2: batch scaling ======
    batch_1t = batch_results[1]
    batch_nt = batch_results[nthreads]
    batch_N = [r['N'] for r in batch_1t]

    # -- alm2map per-map --
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(batch_N, [r['a2m_per_map'] * 1000 for r in batch_1t],
             's-', color=C_HS, lw=2, ms=6, label='1 thread')
    ax4.plot(batch_N, [r['a2m_per_map'] * 1000 for r in batch_nt],
             's--', color=C_HS, lw=1.5, ms=5, alpha=0.6, label=f'{nthreads} threads')
    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Batch size N')
    ax4.set_ylabel('Time per map (ms)')
    ax4.set_title('alm2map batch (nside=256)')
    ax4.set_xticks(batch_N)
    ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax4.legend()
    ax4.grid(True, alpha=0.2)

    # -- map2alm per-map --
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(batch_N, [r['m2a_per_map'] * 1000 for r in batch_1t],
             's-', color=C_HS, lw=2, ms=6, label='1 thread')
    ax5.plot(batch_N, [r['m2a_per_map'] * 1000 for r in batch_nt],
             's--', color=C_HS, lw=1.5, ms=5, alpha=0.6, label=f'{nthreads} threads')
    ax5.set_xscale('log', base=2)
    ax5.set_xlabel('Batch size N')
    ax5.set_ylabel('Time per map (ms)')
    ax5.set_title('map2alm batch (nside=256, n_iter=3)')
    ax5.set_xticks(batch_N)
    ax5.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax5.legend()
    ax5.grid(True, alpha=0.2)

    # -- Batch throughput --
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(batch_N, [1.0 / r['a2m_per_map'] for r in batch_1t],
             's-', color=C_BAR_A2M, lw=2, ms=6, label='alm2map (1 thr)')
    ax6.plot(batch_N, [1.0 / r['a2m_per_map'] for r in batch_nt],
             's--', color=C_BAR_A2M, lw=1.5, ms=5, alpha=0.6, label=f'alm2map ({nthreads} thr)')
    ax6.plot(batch_N, [1.0 / r['m2a_per_map'] for r in batch_1t],
             'o-', color=C_BAR_M2A, lw=2, ms=6, label='map2alm (1 thr)')
    ax6.plot(batch_N, [1.0 / r['m2a_per_map'] for r in batch_nt],
             'o--', color=C_BAR_M2A, lw=1.5, ms=5, alpha=0.6, label=f'map2alm ({nthreads} thr)')
    ax6.set_xscale('log', base=2)
    ax6.set_xlabel('Batch size N')
    ax6.set_ylabel('Maps / second')
    ax6.set_title('Batch throughput (nside=256)')
    ax6.set_xticks(batch_N)
    ax6.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax6.legend(loc='upper left', fontsize=8.5)
    ax6.grid(True, alpha=0.2)

    fig.suptitle(f'HolySHT benchmark  (spin-0, Intel Xeon Platinum 8358)',
                 fontsize=13, fontweight='bold')

    outpath = 'benchmark_results.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {outpath}')


# --------------- main ---------------------------------------------------

if __name__ == "__main__":
    nthreads = int(os.environ.get("HOLYSHT_NTHREADS", 0))
    if nthreads == 0:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
    print(f"Using nthreads={nthreads}")

    nside_1t, nside_nt = bench_nside(nthreads)
    batch_results = bench_batch(nthreads)
    make_plots(nside_1t, nside_nt, batch_results, nthreads)
