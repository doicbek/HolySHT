"""Generate a clean benchmark figure for the README."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Data from benchmark runs (Intel Xeon Platinum 8358, 8 cores) ---

nsides = [64, 128, 256, 512, 1024, 2048]

# Single-threaded
hp_a2m_1t  = [0.0040, 0.0111, 0.0598, 0.3285,  2.0354, 14.4786]
hs_a2m_1t  = [0.0041, 0.0062, 0.0309, 0.1747,  0.8195,  4.9374]
hp_m2a_1t  = [0.0128, 0.0702, 0.4080, 2.1604, 14.2342, 100.7568]
hs_m2a_1t  = [0.0102, 0.0446, 0.2394, 1.0569,  5.9377, 34.5346]

# 8-threaded
hp_a2m_8t  = [0.0028, 0.0098, 0.0518, 0.3168,  2.0222, 13.6937]
hs_a2m_8t  = [0.0016, 0.0035, 0.0079, 0.0244,  0.1334,  0.6731]
hp_m2a_8t  = [0.0106, 0.0653, 0.3581, 2.0887, 13.5289, 97.0095]
hs_m2a_8t  = [0.0092, 0.0268, 0.0859, 0.2596,  1.2271,  6.5380]

# Batch scaling (nside=256, 1 thread)
batch_N        = [1, 2, 4, 8, 16, 32, 64, 128]
b_a2m_per_1t   = [0.02798, 0.02638, 0.02603, 0.02679, 0.02681, 0.02630, 0.02550, 0.02551]
b_m2a_per_1t   = [0.19844, 0.18977, 0.18795, 0.18544, 0.18430, 0.17722, 0.17673, 0.17671]

# Batch scaling (nside=256, 8 threads)
b_a2m_per_8t   = [0.02044, 0.01284, 0.00744, 0.00449, 0.00433, 0.00430, 0.00424, 0.00422]
b_m2a_per_8t   = [0.15055, 0.10522, 0.07287, 0.05378, 0.05129, 0.04867, 0.04714, 0.04609]

# --- Styling ---

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'figure.facecolor': 'white',
})

C_HP  = '#4477AA'
C_HS  = '#EE6677'
C_BAR_A2M = '#EE6677'
C_BAR_M2A = '#CCBB44'

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

# ====== Row 1: nside scaling ======

# -- alm2map --
ax1 = fig.add_subplot(gs[0, 0])
ax1.loglog(nsides, hp_a2m_1t, 'o-',  color=C_HP, label='healpy (1 thr)', lw=2, ms=6)
ax1.loglog(nsides, hp_a2m_8t, 'o--', color=C_HP, label='healpy (8 thr)', lw=1.5, ms=5, alpha=0.6)
ax1.loglog(nsides, hs_a2m_1t, 's-',  color=C_HS, label='HolySHT (1 thr)', lw=2, ms=6)
ax1.loglog(nsides, hs_a2m_8t, 's--', color=C_HS, label='HolySHT (8 thr)', lw=1.5, ms=5, alpha=0.6)
ax1.set_xlabel('nside')
ax1.set_ylabel('Wall time (s)')
ax1.set_title('alm2map')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.2, which='both')
ax1.set_xticks(nsides)
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

# -- map2alm --
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(nsides, hp_m2a_1t, 'o-',  color=C_HP, label='healpy (1 thr)', lw=2, ms=6)
ax2.loglog(nsides, hp_m2a_8t, 'o--', color=C_HP, label='healpy (8 thr)', lw=1.5, ms=5, alpha=0.6)
ax2.loglog(nsides, hs_m2a_1t, 's-',  color=C_HS, label='HolySHT (1 thr)', lw=2, ms=6)
ax2.loglog(nsides, hs_m2a_8t, 's--', color=C_HS, label='HolySHT (8 thr)', lw=1.5, ms=5, alpha=0.6)
ax2.set_xlabel('nside')
ax2.set_ylabel('Wall time (s)')
ax2.set_title('map2alm (n_iter=3)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.2, which='both')
ax2.set_xticks(nsides)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

# -- Speedup bar chart (8-threaded) --
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(nsides))
w = 0.35
speedup_a2m = [h / s for h, s in zip(hp_a2m_8t, hs_a2m_8t)]
speedup_m2a = [h / s for h, s in zip(hp_m2a_8t, hs_m2a_8t)]
bars1 = ax3.bar(x - w/2, speedup_a2m, w, label='alm2map', color=C_BAR_A2M, alpha=0.8)
bars2 = ax3.bar(x + w/2, speedup_m2a, w, label='map2alm', color=C_BAR_M2A, alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels([str(n) for n in nsides])
ax3.set_xlabel('nside')
ax3.set_ylabel('Speedup vs healpy')
ax3.set_title('Speedup (8 threads)')
ax3.axhline(1, color='gray', ls='--', lw=0.8)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2, axis='y')
for bar in bars1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.3,
             f'{h:.1f}x', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.3,
             f'{h:.1f}x', ha='center', va='bottom', fontsize=8)

# ====== Row 2: batch scaling ======

# -- alm2map per-map (both thread counts) --
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(batch_N, [t * 1000 for t in b_a2m_per_1t], 's-', color=C_HS, lw=2, ms=6, label='1 thread')
ax4.plot(batch_N, [t * 1000 for t in b_a2m_per_8t], 's--', color=C_HS, lw=1.5, ms=5, alpha=0.6, label='8 threads')
ax4.set_xscale('log', base=2)
ax4.set_xlabel('Batch size N')
ax4.set_ylabel('Time per map (ms)')
ax4.set_title('alm2map batch (nside=256)')
ax4.set_xticks(batch_N)
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.legend()
ax4.grid(True, alpha=0.2)

# -- map2alm per-map (both thread counts) --
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(batch_N, [t * 1000 for t in b_m2a_per_1t], 's-', color=C_HS, lw=2, ms=6, label='1 thread')
ax5.plot(batch_N, [t * 1000 for t in b_m2a_per_8t], 's--', color=C_HS, lw=1.5, ms=5, alpha=0.6, label='8 threads')
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
tput_a2m_1t = [1.0 / t for t in b_a2m_per_1t]
tput_m2a_1t = [1.0 / t for t in b_m2a_per_1t]
tput_a2m_8t = [1.0 / t for t in b_a2m_per_8t]
tput_m2a_8t = [1.0 / t for t in b_m2a_per_8t]
ax6.plot(batch_N, tput_a2m_1t, 's-',  color=C_BAR_A2M, lw=2, ms=6, label='alm2map (1 thr)')
ax6.plot(batch_N, tput_a2m_8t, 's--', color=C_BAR_A2M, lw=1.5, ms=5, alpha=0.6, label='alm2map (8 thr)')
ax6.plot(batch_N, tput_m2a_1t, 'o-',  color=C_BAR_M2A, lw=2, ms=6, label='map2alm (1 thr)')
ax6.plot(batch_N, tput_m2a_8t, 'o--', color=C_BAR_M2A, lw=1.5, ms=5, alpha=0.6, label='map2alm (8 thr)')
ax6.set_xscale('log', base=2)
ax6.set_xlabel('Batch size N')
ax6.set_ylabel('Maps / second')
ax6.set_title('Batch throughput (nside=256)')
ax6.set_xticks(batch_N)
ax6.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax6.legend(loc='upper left', fontsize=8.5)
ax6.grid(True, alpha=0.2)

fig.suptitle('HolySHT benchmark  (spin-0, Intel Xeon Platinum 8358, 8 cores)',
             fontsize=13, fontweight='bold')

outpath = '../benchmark.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
