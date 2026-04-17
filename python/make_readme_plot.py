"""Generate a clean benchmark figure for the README."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Data from benchmark runs (Intel Xeon Platinum 8358) ---

nsides = [64, 128, 256, 512, 1024, 2048]

# Single-threaded
hp_a2m_1t  = [0.0036, 0.0092, 0.0527, 0.2669,  1.8950, 13.7188]
hs_a2m_1t  = [0.0023, 0.0045, 0.0197, 0.0850,  0.4877,  2.9109]
hp_m2a_1t  = [0.0118, 0.0632, 0.3716, 1.8201, 12.8307, 94.0643]
hs_m2a_1t  = [0.0078, 0.0315, 0.1431, 0.5914,  3.4236, 20.8730]

# 8-threaded
hp_a2m_8t  = [0.0017, 0.0087, 0.0517, 0.3389,  2.3144, 16.7529]
hs_a2m_8t  = [0.0013, 0.0045, 0.0209, 0.1164,  0.6665,  3.9848]
hp_m2a_8t  = [0.0114, 0.0580, 0.3415, 2.1765, 14.7382, 106.0183]
hs_m2a_8t  = [0.0082, 0.0306, 0.1404, 0.8175,  4.3926, 26.4800]

# Batch scaling (nside=256, 1 thread)
batch_N        = [1, 2, 4, 8, 16, 32, 64, 128]
b_a2m_per_1t   = [0.01490, 0.01451, 0.01580, 0.01777, 0.01814, 0.01837, 0.01842, 0.01853]
b_m2a_per_1t   = [0.10541, 0.08759, 0.08873, 0.08523, 0.08705, 0.08614, 0.08687, 0.08549]

# Batch scaling (nside=256, 8 threads)
b_a2m_per_8t   = [0.02130, 0.02093, 0.02229, 0.02381, 0.02381, 0.02377, 0.02372, 0.02370]
b_m2a_per_8t   = [0.14239, 0.12306, 0.12358, 0.11319, 0.11041, 0.10813, 0.10698, 0.10650]

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

# ====== Row 1: nside scaling (1-thread) ======

# -- alm2map --
ax1 = fig.add_subplot(gs[0, 0])
ax1.loglog(nsides, hp_a2m_1t, 'o-',  color=C_HP, label='healpy', lw=2, ms=6)
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
ax2.loglog(nsides, hp_m2a_1t, 'o-',  color=C_HP, label='healpy', lw=2, ms=6)
ax2.loglog(nsides, hs_m2a_1t, 's-',  color=C_HS, label='HolySHT (1 thr)', lw=2, ms=6)
ax2.loglog(nsides, hs_m2a_8t, 's--', color=C_HS, label='HolySHT (8 thr)', lw=1.5, ms=5, alpha=0.6)
ax2.set_xlabel('nside')
ax2.set_ylabel('Wall time (s)')
ax2.set_title('map2alm (n_iter=3)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.2, which='both')
ax2.set_xticks(nsides)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

# -- Speedup bar chart --
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(nsides))
w = 0.35
speedup_a2m = [h / s for h, s in zip(hp_a2m_1t, hs_a2m_1t)]
speedup_m2a = [h / s for h, s in zip(hp_m2a_1t, hs_m2a_1t)]
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

fig.suptitle('HolySHT benchmark  (spin-0, Intel Xeon Platinum 8358)',
             fontsize=13, fontweight='bold')

outpath = '../benchmark.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
