"""Cognitive versus perceptive contrast (Figure 4) on baseline-subtracted subnetwork graphs.

Provides stat_map(): rank-sum statistic times the FDR-significance mask, comparing
cognitive (studies 1, 4, 5) and perceptive (studies 2, 3) neurofeedback on the
per-subject baseline-subtracted graphs (NF minus per-subject-mean DC). Used by
make_suppfig_11_cogperc_glm.py.
"""
import os
import numpy as np
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SWEEP = 'Results/glm_revision/alpha_sweep'
OUT = 'Results/glm_revision/figures'
ABAR = 0.05
COG, PERC = [1, 4, 5], [2, 3]
SM = [14, 15]
VIS = ['L-Sub', 'R-Sub', 'L-Vis', 'R-Vis', 'L-Lim', 'R-Lim', 'L-Cont', 'R-Cont',
       'L-DMN', 'R-DMN', 'L-Vent-Att', 'R-Vent-Att', 'L-Dors-Att', 'R-Dors-Att',
       'L-Som-Mot', 'R-Som-Mot']


def stat_map(cog, per):
    stat = np.zeros((16, 16)); P = np.ones((16, 16))
    for i in range(16):
        for j in range(16):
            if cog[:, i, j].std() or per[:, i, j].std():
                t = ranksums(cog[:, i, j], per[:, i, j])
                stat[i, j], P[i, j] = t.statistic, t.pvalue
    sig = fdrcorrection(P.ravel())[0].reshape(16, 16)
    return stat * sig, sig


def main():
    d = np.load(os.path.join(SWEEP, f'subnet_abar_{ABAR}.npz'))
    nf, nfidx = d['nf_weighted'], d['nf_idx']
    dc, dcidx = d['dc_weighted'], d['dc_idx']

    # per-subject mean DC (baseline)
    meandc = {}
    for (s, su, se, r), g in zip(dcidx, dc):
        meandc.setdefault((int(s), int(su)), []).append(g)
    meandc = {k: np.mean(v, axis=0) for k, v in meandc.items()}

    nf_minus_dc = np.array([g - meandc[(int(s), int(su))] for (s, su, se, r), g in zip(nfidx, nf)])
    nf_study = nfidx[:, 0]; dc_study = dcidx[:, 0]

    panels = {
        'A: NF - DC (paper Fig 4)': (nf_minus_dc[np.isin(nf_study, COG)], nf_minus_dc[np.isin(nf_study, PERC)]),
        'B: raw NF (no baseline subtraction)': (nf[np.isin(nf_study, COG)], nf[np.isin(nf_study, PERC)]),
        'C: DC only (baseline)': (dc[np.isin(dc_study, COG)], dc[np.isin(dc_study, PERC)]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    results = {}
    vmax = 0
    maps = {}
    for title, (cg, pr) in panels.items():
        m, sig = stat_map(cg, pr)
        maps[title] = (m, sig)
        vmax = max(vmax, np.abs(m).max())
        # somatomotor organization: count sig SM->other edges that are + vs -
        sm_edges = []
        for s in SM:
            for j in range(16):
                if sig[s, j] and j not in SM:
                    sm_edges.append(m[s, j])
                if sig[j, s] and j not in SM:
                    sm_edges.append(m[j, s])
        sm_edges = np.array(sm_edges)
        results[title] = (int(sig[SM, :].sum() + sig[:, SM].sum()),
                          int((sm_edges > 0).sum()), int((sm_edges < 0).sum()))
        print(f'{title}: somatomotor sig edges={results[title][0]}, '
              f'SM-to-other significant: {results[title][1]} positive / {results[title][2]} negative')

    for ax, (title, (m, sig)) in zip(axes, maps.items()):
        im = ax.imshow(m, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        for s in SM:
            ax.add_patch(mpatches.Rectangle((-0.5, s - 0.5), 16, 1, fill=False, ec='lime', lw=1.3))
            ax.add_patch(mpatches.Rectangle((s - 0.5, -0.5), 1, 16, fill=False, ec='lime', lw=1.3))
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(VIS, rotation=90, fontsize=6); ax.set_yticklabels(VIS, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig9_fig4_baseline_subtraction.{ext}'), bbox_inches='tight', dpi=130)
    plt.close(fig)
    print('saved', os.path.join(OUT, 'fig9_fig4_baseline_subtraction.png'))


if __name__ == '__main__':
    main()
