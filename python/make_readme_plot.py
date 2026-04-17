"""Generate a clean benchmark figure for the README."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Data from single-threaded benchmark run ---

nsides      = [64,    128,   256,   512,    1024,   2048]
lmaxes      = [128,   256,   512,   1024,   2048,   4096]
hp_a2m      = [0.0036, 0.0092, 0.0527, 0.2669, 1.8950, 13.7188]
hs_a2m      = [0.0023, 0.0045, 0.0197, 0.0850, 0.4877,  2.9109]
hp_m2a      = [0.0118, 0.0632, 0.3716, 1.8201, 12.8307, 94.0643]
hs_m2a      = [0.0078, 0.0315, 0.1431, 0.5914,  3.4236, 20.8730]

batch_N     = [1, 2, 4, 8, 16, 32, 64, 128]
b_a2m_per   = [0.01490, 0.01451, 0.01580, 0.01777, 0.01814, 0.01837, 0.01842, 0.01853]
b_m2a_per   = [0.10541, 0.08759, 0.08873, 0.08523, 0.08705, 0.08614, 0.08687, 0.08549]

# --- Styling ---

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})

C_HP  = '#4477AA'   # blue for healpy
C_HS  = '#EE6677'   # red-pink for HolySHT
C_BAR = '#228833'   # green for speedup bars

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

# ====== Row 1: nside scaling ======

# -- alm2map --
ax1 = fig.add_subplot(gs[0, 0])
ax1.loglog(nsides, hp_a2m, 'o-', color=C_HP, label='healpy', lw=2, ms=6)
ax1.loglog(nsides, hs_a2m, 's-', color=C_HS, label='HolySHT', lw=2, ms=6)
ax1.set_xlabel('nside')
ax1.set_ylabel('Wall time (s)')
ax1.set_title('alm2map')
ax1.legend()
ax1.grid(True, alpha=0.2, which='both')
ax1.set_xticks(nsides)
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

# -- map2alm --
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(nsides, hp_m2a, 'o-', color=C_HP, label='healpy', lw=2, ms=6)
ax2.loglog(nsides, hs_m2a, 's-', color=C_HS, label='HolySHT', lw=2, ms=6)
ax2.set_xlabel('nside')
ax2.set_ylabel('Wall time (s)')
ax2.set_title('map2alm (n_iter=3)')
ax2.legend()
ax2.grid(True, alpha=0.2, which='both')
ax2.set_xticks(nsides)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

# -- Speedup bar chart --
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(nsides))
w = 0.35
speedup_a2m = [h / s for h, s in zip(hp_a2m, hs_a2m)]
speedup_m2a = [h / s for h, s in zip(hp_m2a, hs_m2a)]
bars1 = ax3.bar(x - w/2, speedup_a2m, w, label='alm2map', color=C_HS, alpha=0.8)
bars2 = ax3.bar(x + w/2, speedup_m2a, w, label='map2alm', color='#CCBB44', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels([str(n) for n in nsides])
ax3.set_xlabel('nside')
ax3.set_ylabel('Speedup vs healpy')
ax3.set_title('HolySHT / healpy speedup')
ax3.axhline(1, color='gray', ls='--', lw=0.8)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2, axis='y')
# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.08,
             f'{h:.1f}x', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.08,
             f'{h:.1f}x', ha='center', va='bottom', fontsize=8)

# ====== Row 2: batch scaling ======

# -- alm2map per-map --
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(batch_N, [t * 1000 for t in b_a2m_per], 's-', color=C_HS, lw=2, ms=6)
ax4.axhline(b_a2m_per[0] * 1000, ls='--', color='gray', alpha=0.5, lw=1)
ax4.set_xscale('log', base=2)
ax4.set_xlabel('Batch size N')
ax4.set_ylabel('Time per map (ms)')
ax4.set_title('alm2map batch (nside=256)')
ax4.set_xticks(batch_N)
ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.grid(True, alpha=0.2)

# -- map2alm per-map --
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(batch_N, [t * 1000 for t in b_m2a_per], 's-', color=C_HS, lw=2, ms=6)
ax5.axhline(b_m2a_per[0] * 1000, ls='--', color='gray', alpha=0.5, lw=1,
            label='N=1 baseline')
ax5.set_xscale('log', base=2)
ax5.set_xlabel('Batch size N')
ax5.set_ylabel('Time per map (ms)')
ax5.set_title('map2alm batch (nside=256, n_iter=3)')
ax5.set_xticks(batch_N)
ax5.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.2)

# -- Batch throughput --
ax6 = fig.add_subplot(gs[1, 2])
tput_a2m = [N / t for N, t in zip(batch_N,
            [b_a2m_per[i] * batch_N[i] for i in range(len(batch_N))])]
tput_m2a = [N / t for N, t in zip(batch_N,
            [b_m2a_per[i] * batch_N[i] for i in range(len(batch_N))])]
ax6.plot(batch_N, tput_a2m, 's-', color=C_HS, lw=2, ms=6, label='alm2map')
ax6.plot(batch_N, tput_m2a, 'o-', color='#CCBB44', lw=2, ms=6, label='map2alm')
ax6.set_xscale('log', base=2)
ax6.set_xlabel('Batch size N')
ax6.set_ylabel('Maps / second')
ax6.set_title('Batch throughput (nside=256)')
ax6.set_xticks(batch_N)
ax6.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax6.legend()
ax6.grid(True, alpha=0.2)

fig.suptitle('HolySHT benchmark  (spin-0, single-threaded, Intel Xeon 8358)',
             fontsize=13, fontweight='bold')

outpath = '../benchmark.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
