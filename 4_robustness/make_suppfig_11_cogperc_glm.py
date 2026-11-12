"""Supplementary Fig. 11: the cognitive versus perceptive contrast after GLM task regression.

Repeats the Figure 4 analysis (rank-sum statistic times FDR mask on baseline-subtracted
subnetwork graphs) on the original graphs and on the GLM task-regressed graphs (HRF and
binary regressors).
Output: Results/figures_supplementary/SuppFig11_cogperc_glm.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from make_glm_control_hub_9subj import process
from make_fig4_baseline import stat_map, COG, PERC, SM, VIS

OUT = 'Results/glm_revision/figures'
FINAL = 'Results/figures_supplementary'
RAW = 'Results/glm_revision/alpha_sweep/subnet_abar_0.05.npz'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
})


def baseline_subtract(nf16, dc16, nfidx, dcidx):
    """NF minus per-subject mean DC (paper Fig-4 baseline), then split cog / perc."""
    meandc = {}
    for (s, su, se, r), g in zip(dcidx, dc16):
        meandc.setdefault((int(s), int(su)), []).append(g)
    meandc = {k: np.mean(v, axis=0) for k, v in meandc.items()}
    nf_minus_dc = np.array([g - meandc[(int(s), int(su))] for (s, su, se, r), g in zip(nfidx, nf16)])
    study = nfidx[:, 0]
    return nf_minus_dc[np.isin(study, COG)], nf_minus_dc[np.isin(study, PERC)]


def sm_counts(m, sig):
    """Somatomotor significant edges: total, and SM-to-other split by sign (paper Fig-4)."""
    edges = []
    for s in SM:
        for j in range(16):
            if sig[s, j] and j not in SM:
                edges.append(m[s, j])
            if sig[j, s] and j not in SM:
                edges.append(m[j, s])
    edges = np.array(edges)
    total = int(sig[SM, :].sum() + sig[:, SM].sum())
    return total, int((edges > 0).sum()), int((edges < 0).sum())


def original_map():
    """Paper Fig-4 panel A on the ORIGINAL graphs (reference)."""
    d = np.load(RAW)
    cg, pr = baseline_subtract(d['nf_weighted'], d['dc_weighted'], d['nf_idx'], d['dc_idx'])
    return stat_map(cg, pr)


def glm_map(variant):
    """Paper Fig-4 panel A on the GLM task-regressed graphs (NF and DC)."""
    nf16, _, nfidx = process('nf', variant)
    dc16, _, dcidx = process('dc', variant)
    cg, pr = baseline_subtract(nf16, dc16, nfidx, dcidx)
    return stat_map(cg, pr)


def main():
    panels = [
        ('Original graphs', original_map()),
        ('GLM-regressed (HRF)', glm_map('hrf')),
        ('GLM-regressed (binary)', glm_map('binary')),
    ]
    vmax = max(np.abs(m).max() for _, (m, _) in panels)

    print('\n=== Cognitive vs Perceptive somatomotor edges (baseline-subtracted) ===')
    for title, (m, sig) in panels:
        tot, pos, neg = sm_counts(m, sig)
        print(f'{title.splitlines()[0]:32s}: SM sig edges={tot:3d}  |  SM-to-other: {pos} positive / {neg} negative')

    # ---- Figure 4 style: coolwarm imshow of statistic x FDR mask,
    #      symmetric colour scale, rotated network labels, colorbar on the right. Same colour
    #      scale across panels so original vs GLM comparison is direct.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), layout='constrained')
    for ax, (title, (m, sig)) in zip(axes, panels):
        im = ax.imshow(m, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(16), labels=VIS, fontsize=14)
        ax.set_yticks(np.arange(16), labels=VIS, fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title(title, fontsize=22)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02, shrink=0.62)
    cb.set_label('rank-sum statistic\n(+: stronger in cognitive,  $-$: stronger in perceptive)',
                 fontsize=14)
    cb.ax.tick_params(labelsize=14)
    os.makedirs(FINAL, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FINAL, f'SuppFig11_cogperc_glm.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(FINAL, 'SuppFig11_cogperc_glm.png'))


if __name__ == '__main__':
    main()
