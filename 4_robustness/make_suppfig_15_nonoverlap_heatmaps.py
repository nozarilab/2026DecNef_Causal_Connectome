"""Supplementary Fig. 15: control-hub analysis on non-overlapping graphs, heatmap view.

Uses only graphs from disjoint run pairs (NF runs 1-2, 3-4, 5-6, 7-8; baseline 1-2, 3-4, 5-6):
(a) NF vs baseline modulation, (b) score correlation, (c) their intersection.
Output: Results/figures_supplementary/SuppFig15_nonoverlap_heatmaps.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from make_suppfigs_22_25_alpha import two_step, VIS
from make_fig3_alpha import per_graph_scores

SWEEP = 'Results/glm_revision/alpha_sweep/subnet_abar_0.05.npz'
FINAL = 'Results/figures_supplementary'
CTRLIDX = [6, 7]
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'pdf.fonttype': 42})


def net_axes(ax):
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels(VIS, rotation=90, fontsize=9); ax.set_yticklabels(VIS, fontsize=9)
    for c in CTRLIDX:                                       # mark the control rows/cols
        ax.axhline(c - .5, color='k', lw=.4, alpha=.3); ax.axvline(c - .5, color='k', lw=.4, alpha=.3)
        ax.axhline(c + .5, color='k', lw=.4, alpha=.3); ax.axvline(c + .5, color='k', lw=.4, alpha=.3)


def main():
    d = np.load(SWEEP)
    nf, dc = d['nf_weighted'].astype(float), d['dc_weighted'].astype(float)
    nfidx, dcidx = d['nf_idx'], d['dc_idx']
    sc = np.array([per_graph_scores().get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])
    nfk = np.isin(nfidx[:, 3], [0, 2, 4, 6]); dck = np.isin(dcidx[:, 3], [0, 2, 4])
    nf, dc, sc = nf[nfk], dc[dck], sc[nfk]
    ok = np.isfinite(sc)

    stronger, poscorr, hub, nfdc, scorr = two_step(nf, dc, nf[ok], sc[ok])

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2), layout='constrained')

    v1 = np.nanpercentile(np.abs(nfdc[nfdc != 0]), 99) if (nfdc != 0).any() else 1
    im = ax[0].imshow(nfdc, cmap='coolwarm', vmin=-v1, vmax=v1)
    yi, xi = np.where(hub)
    ax[0].scatter(xi, yi, s=170, facecolors='none', edgecolors='lime', linewidths=2.6)
    fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04).set_label('rank-sum stat (NF $-$ baseline)', fontsize=11)
    ax[0].set_title('(a)', loc='left', fontsize=20, fontweight='bold')
    print(f'(a) NF>baseline: {int(stronger.sum())} edges')

    v2 = np.nanpercentile(np.abs(scorr[scorr != 0]), 99) if (scorr != 0).any() else .1
    im = ax[1].imshow(scorr, cmap='coolwarm', vmin=-v2, vmax=v2)
    fig.colorbar(im, ax=ax[1], fraction=.046, pad=.04).set_label('Spearman $\\rho$ (edge vs score)', fontsize=11)
    ax[1].set_title('(b)', loc='left', fontsize=20, fontweight='bold')
    print(f'(b) score corr: {int(poscorr.sum())}+ / {int((scorr < 0).sum())}-')

    cmap = mcolors.ListedColormap(['white', '#1b7837'])
    ax[2].imshow(hub.astype(int), cmap=cmap, vmin=0, vmax=1)
    ax[2].set_title('(c)', loc='left', fontsize=20, fontweight='bold')
    print(f'(c) surviving hub: {int(hub.sum())} edge(s)')
    for c in range(3):
        net_axes(ax[c])

    os.makedirs(FINAL, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'SuppFig15_nonoverlap_heatmaps.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    hub_edges = [f'{VIS[i]}->{VIS[j]}' for i, j in zip(*np.where(hub))]
    print(f'stronger-in-NF={int(stronger.sum())}  +score={int(poscorr.sum())}  hub={int(hub.sum())}  {hub_edges}')
    print('saved', os.path.join(FINAL, 'SuppFig15_nonoverlap_heatmaps.png'))


if __name__ == '__main__':
    main()
