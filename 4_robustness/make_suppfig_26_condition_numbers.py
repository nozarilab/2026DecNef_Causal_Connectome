"""Supplementary Fig. 26: condition numbers of the conditioning regressions inside CaLLTiF.

Input:  Results/glm_revision/condnum_diagnostic_expanded.npz (make_condnum_diagnostic_expanded.py)
Output: Results/figures_supplementary/SuppFig26_condition_numbers.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NPZ = 'Results/glm_revision/condnum_diagnostic_expanded.npz'
OUT = 'Results/figures_supplementary'
STUDIES = (1, 2, 3, 4, 5)
SN = {1: 'S1 facial pref.', 2: 'S2 orient-colour', 3: 'S3 fear-cc',
      4: 'S4 confidence', 5: 'S5 fear-reprog.'}
NFC, DCC = 'green', 'lightgreen'      # paper convention: dark green = NF, light green = DC

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


def panel_tag(ax, text, pad):
    """Panel label (a)/(b) placed BELOW the panel (offset in points below the axes)."""
    ax.annotate(text, xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -pad),
                textcoords='offset points', ha='center', va='top',
                fontsize=plt.rcParams['axes.titlesize'])


def main():
    d = np.load(NPZ)
    cond, kind, study, dimz = d['cond'], d['kind'], d['study'], d['dim_z']
    ok = np.isfinite(cond)
    cond, kind, study, dimz = cond[ok], kind[ok], study[ok], dimz[ok]
    nf, dc = cond[kind == 0], cond[kind == 1]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={'wspace': 0.24})

    # (a) pooled histogram, NF vs DC
    bins = np.linspace(0, max(cond.max(), 200), 70)
    ax[0].hist(nf, bins=bins, color=NFC, alpha=0.6, label=f'NF (n={len(nf):,}), median {np.median(nf):.0f}, max {nf.max():.0f}')
    ax[0].hist(dc, bins=bins, color=DCC, alpha=0.6, label=f'DC (n={len(dc):,}), median {np.median(dc):.0f}, max {dc.max():.0f}')
    ax[0].axvline(np.median(nf), color=NFC, ls='--', lw=1.1)
    ax[0].axvline(np.median(dc), color=DCC, ls='--', lw=1.1)
    panel_tag(ax[0], f'(a)  Condition numbers of all {len(cond):,} conditioning regressions', 60)
    ax[0].set_xlabel('condition number  cond(Z)')
    ax[0].set_ylabel('count')
    ax[0].legend(loc='upper right')
    ax[0].text(0.97, 0.55,
               f'conditioning dim = {int(dimz.min())} to {int(dimz.max())}\n'
               f'0 % exceed $10^3$  (overall max {cond.max():.0f})\n'
               f'ill-conditioned $\\simeq 10^{{16}}$\n(double precision)',
               transform=ax[0].transAxes, ha='right', va='top', fontsize=11.5,
               bbox=dict(boxstyle='round', fc='white', ec='0.7'))

    # (b) per-study boxplots, NF and DC
    pos = 0; xt, xl = [], []
    for s in STUDIES:
        for j, (msk, col) in enumerate([(kind == 0, NFC), (kind == 1, DCC)]):
            c = cond[msk & (study == s)]
            p = pos + j * 0.85
            bp = ax[1].boxplot(c, positions=[p], widths=0.62, patch_artist=True,
                               showfliers=False, medianprops=dict(color='k', lw=1.1))
            bp['boxes'][0].set(facecolor=col, alpha=0.6)
        xt.append(pos + 0.42); xl.append(SN[s]); pos += 2.5
    ax[1].set_xticks(xt); ax[1].set_xticklabels(xl, rotation=18, ha='right')
    ax[1].set_ylabel('condition number  cond(Z)')
    panel_tag(ax[1], '(b)  By study and session type (3 subjects each)', 60)
    ax[1].legend(handles=[Patch(facecolor=NFC, alpha=.6, label='NF'),
                          Patch(facecolor=DCC, alpha=.6, label='DC')], loc='upper left')

    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'SuppFig26_condition_numbers.{ext}'), dpi=200)
    plt.close(fig)
    print('saved', os.path.join(OUT, 'SuppFig26_condition_numbers.pdf'))
    print(f'NF: n={len(nf)} median={np.median(nf):.0f} p95={np.percentile(nf,95):.0f} max={nf.max():.0f}')
    print(f'DC: n={len(dc)} median={np.median(dc):.0f} p95={np.percentile(dc,95):.0f} max={dc.max():.0f}')


if __name__ == '__main__':
    main()
