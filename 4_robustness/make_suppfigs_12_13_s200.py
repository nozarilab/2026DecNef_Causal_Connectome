"""Supplementary Figs. 12-13: control hub at Schaefer-200 + Tian S1 (216 parcels).

  SuppFig12_s200_windowing_design   how the eight runs of a session are grouped into graphs
  SuppFig13_s200_windowing_maps     two-step control-hub analysis for the two four-run designs
Output folder: Results/figures_supplementary/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from make_s200_windowing_maps import two_step_maps, load_case, CASES, CLO, CHI
from compare_s200_windowing import NET, CTRL_NET, CTRL_LABELS

OUT = 'Results/figures_supplementary'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
    'legend.fontsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 130, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200)
    plt.close(fig)
    print('saved', os.path.join(OUT, name + '.pdf'))


# -------------------------------------------------- design schematic
def fig_windowing_design():
    """8 runs per session, tiled into a graph in five ways. Each window is one bar."""
    N_RUN = 8
    schemes = [
        ('2-run (paper design)', 2, list(range(0, 7)), '#6a51a3',
         '7 windows/session, overlap 1 run\n945 graphs'),
        ('4-run, stride-1 (overlap 3 runs)', 4, [0, 1, 2, 3, 4], '#2171b5',
         '5 windows/session, overlap 3 runs\n675 graphs'),
        ('4-run, stride-2 (overlap 2 runs)', 4, [0, 2, 4], '#238b45',
         '3 windows/session, overlap 2 runs\n405 graphs'),
    ]
    row_h = [len(s[2]) for s in schemes]
    fig_h = 1.6 + 0.32 * sum(row_h) + 0.9 * len(schemes)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(-0.4, N_RUN + 3.4)
    y = 0.0
    band_tops = []

    # draw each scheme from bottom to top
    for title, W, starts, color, note in reversed(schemes):
        y0 = y
        for k, s in enumerate(reversed(starts)):
            yy = y + k * 1.0
            ax.add_patch(FancyBboxPatch((s + 0.08, yy + 0.12), W - 0.16, 0.76,
                                        boxstyle='round,pad=0.02,rounding_size=0.12',
                                        fc=color, ec='none', alpha=0.85))
            ax.text(s + W / 2, yy + 0.5, f'runs {s + 1}–{s + W}', ha='center', va='center',
                    fontsize=10, color='white', weight='bold')
        band_top = y + len(starts) * 1.0
        # scheme label + note to the right
        ax.text(N_RUN + 0.25, (y0 + band_top) / 2 + 0.25, title, ha='left', va='center',
                fontsize=13, weight='bold', color=color)
        ax.text(N_RUN + 0.25, (y0 + band_top) / 2 - 0.55, note, ha='left', va='center',
                fontsize=11, color='0.25')
        band_tops.append(band_top)
        y = band_top + 0.9

    # run ruler along the top
    top = y
    for r in range(N_RUN):
        ax.add_patch(Rectangle((r + 0.05, top + 0.1), 0.9, 0.7, fc='0.9', ec='0.5', lw=1))
        ax.text(r + 0.5, top + 0.45, f'run\n{r + 1}', ha='center', va='center', fontsize=10.5, color='0.2')
    ax.text(N_RUN / 2, top + 1.15, 'one neurofeedback session = 8 runs (per subject, per session)',
            ha='center', va='center', fontsize=13.5, weight='bold')

    ax.set_ylim(-0.3, top + 1.5)
    ax.axis('off')
    save(fig, 'SuppFig12_s200_windowing_design')


# -------------------------------------------------- four-case maps
def fig_windowing_maps():
    # drop the 2-run / lag-0-only case (off-methodology, unexplained subcortex-visual edge) and
    # the 4-run non-overlap case (empty maps); keep the two 4-run designs (overlap 3 / overlap 2).
    ncase = len(CASES) - 2
    fig, ax = plt.subplots(ncase, 4, figsize=(19, 4.8 * ncase), layout='constrained')
    col_titles = ['Significant NF − Baseline\n(rank-sum statistic, FDR)',
                  'Significant correlation with score\n(Spearman, FDR)',
                  'Intersection\n(stronger in NF AND + score)',
                  'Control-network parcel zoom']
    for row, (name, npz, W, nfp_sel, dcp_sel) in enumerate(CASES[1:-1]):
        nf16, dc16, nfp, dcp, score = load_case(npz, W, nfp_sel, dcp_sel)
        ok = np.isfinite(score)
        nfdc, scor, hub, ms, cr = two_step_maps(nf16, dc16, score, ok)
        ctrl_edges = [(NET[i], NET[j]) for i, j in zip(*np.where(hub))
                      if NET[i] in CTRL_NET and NET[j] in CTRL_NET]

        # col A / B: signed significant maps
        for c, (M, raw) in enumerate([(nfdc, ms), (scor, cr)]):
            v = np.nanpercentile(np.abs(M[M != 0]), 99) if (M != 0).any() else (np.abs(raw).max() or 1)
            im = ax[row, c].imshow(M, cmap='coolwarm', vmin=-v, vmax=v)
            fig.colorbar(im, ax=ax[row, c], fraction=.046, pad=.03).ax.tick_params(labelsize=10)
        # col C: intersection (binary green)
        ax[row, 2].imshow(hub.astype(float), cmap='Greens', vmin=0, vmax=1)
        for c in (0, 1, 2):
            ax[row, c].set_xticks(range(16)); ax[row, c].set_yticks(range(16))
            ax[row, c].set_xticklabels(NET, rotation=90, fontsize=9)
            ax[row, c].set_yticklabels(NET, fontsize=9)
        print(f'  ({chr(97 + row)}) intersection: {int(hub.sum())} hub edges; within-control: '
              f'{[f"{a}->{b}" for a, b in ctrl_edges] if ctrl_edges else "none"}')
        ax[row, 0].set_ylabel(f'({chr(97 + row)})', fontsize=22, weight='bold',
                              rotation=0, ha='right', va='center', labelpad=18)
        if row == 0:
            for c in range(4):
                ax[row, c].set_title(col_titles[c], fontsize=14)

        # col D: control-network parcel zoom + within-control count (all rows)
        a3 = ax[row, 3]
        hubP = two_step_maps(nfp, dcp, score, ok)[2]
        cb = slice(CLO, CHI); hc = hubP[cb, cb]
        wb, inc = int(hubP.sum()), int(hc.sum())
        a3.imshow(hc.astype(float), cmap='Greens', vmin=0, vmax=1)
        a3.set_xticks(range(CHI - CLO)); a3.set_yticks(range(CHI - CLO))
        a3.set_xticklabels(CTRL_LABELS, rotation=90, fontsize=4)
        a3.set_yticklabels(CTRL_LABELS, fontsize=4)
        print(f'  ({chr(97 + row)}) parcel zoom: within-control = {inc} of {wb} whole-brain '
              f'hub edges ({100 * inc / max(wb, 1):.0f} %)')

    save(fig, 'SuppFig13_s200_windowing_maps')


if __name__ == '__main__':
    fig_windowing_design()
    fig_windowing_maps()
    print('\nfigures written to', OUT)
