"""Supplementary Figs. 22 and 25: sensitivity to the lagged-edge threshold alpha-bar.

  SuppFig22_alpha_control_hub   NF-vs-baseline modulation, score correlation and parcel hub
  SuppFig25_alpha_cogperc       cognitive vs perceptive contrast (baseline-subtracted graphs)
Input:  Results/glm_revision/alpha_sweep/ (run_alpha_sweep.py)
Output folder: Results/figures_supplementary/
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ranksums, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from make_fig3_alpha import per_graph_scores
import decnef_parcels as dp

SWEEP = 'Results/glm_revision/alpha_sweep'
FINAL = 'Results/figures_supplementary'
ABAR = [0.0025, 0.01, 0.025, 0.05]
COG, PERC = {1, 4, 5}, {2, 3}
SM = [14, 15]
CTRL = slice(38, 51)
VIS = ['L-Sub', 'R-Sub', 'L-Vis', 'R-Vis', 'L-Lim', 'R-Lim', 'L-Cont', 'R-Cont',
       'L-DMN', 'R-DMN', 'L-Vent-Att', 'R-Vent-Att', 'L-Dors-Att', 'R-Dors-Att',
       'L-Som-Mot', 'R-Som-Mot']
_PL = dp.get_parcel_labels(); _NP = dp.get_network_parcel_idx(_PL)
_REARR = [_PL[i] for net in dp.NET_LABELS for i in _NP[net]]
CTRL_LABELS = [l.replace('7Networks_', '').replace('LH_Cont_', 'L-').replace('RH_Cont_', 'R-')
               for l in _REARR[38:51]]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'pdf.fonttype': 42,
    'figure.dpi': 130, 'savefig.bbox': 'tight',
})


def load(a):
    return np.load(os.path.join(SWEEP, f'subnet_abar_{a}.npz'))


def two_step(nf, dc, nf_ok, score):
    P = nf.shape[1]
    ms = np.zeros((P, P)); mp = np.ones((P, P)); cr = np.zeros((P, P)); cp = np.ones((P, P))
    for i in range(P):
        for j in range(P):
            x, y = nf[:, i, j], dc[:, i, j]
            if x.std() > 0 or y.std() > 0:
                t = ranksums(x, y); ms[i, j], mp[i, j] = t.statistic, t.pvalue
            xs = nf_ok[:, i, j]
            if xs.std() > 0:
                tt = spearmanr(xs, score, nan_policy='omit'); cr[i, j], cp[i, j] = tt.statistic, tt.pvalue
    msig = fdrcorrection(mp.ravel())[0].reshape(P, P)
    csig = fdrcorrection(cp.ravel())[0].reshape(P, P)
    return (ms > 0) & msig, (cr > 0) & csig, ((ms > 0) & msig) & ((cr > 0) & csig), ms * msig, cr * csig


def _net_axes(ax):
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels(VIS, rotation=90, fontsize=10); ax.set_yticklabels(VIS, fontsize=10)


def save(fig, name):
    os.makedirs(FINAL, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'{name}.{ext}'), dpi=200)
    plt.close(fig)
    print('saved', os.path.join(FINAL, name + '.pdf'))


def _sig_map(cog, per):
    stat = np.zeros((16, 16)); P = np.ones((16, 16))
    for i in range(16):
        for j in range(16):
            if cog[:, i, j].std() or per[:, i, j].std():
                t = ranksums(cog[:, i, j], per[:, i, j]); stat[i, j], P[i, j] = t.statistic, t.pvalue
    sig = fdrcorrection(P.ravel())[0].reshape(16, 16)
    return stat * sig


# ---------------- Figure 4: cognitive vs perceptive (baseline-subtracted) ----------------
def fig_cogperc():
    maps = []
    for a in ABAR:
        d = load(a); nf, dc = d['nf_weighted'].astype(float), d['dc_weighted'].astype(float)
        nfidx, dcidx = d['nf_idx'], d['dc_idx']
        meandc = {}
        for (s, su, se, r), g in zip(dcidx, dc):
            meandc.setdefault((int(s), int(su)), []).append(g)
        meandc = {k: np.mean(v, 0) for k, v in meandc.items()}
        nmd = np.array([g - meandc[(int(s), int(su))] for (s, su, se, r), g in zip(nfidx, nf)])
        study = nfidx[:, 0]
        maps.append(_sig_map(nmd[np.isin(study, list(COG))], nmd[np.isin(study, list(PERC))]))
    vmax = np.max(np.abs(maps))
    fig, axes = plt.subplots(1, 4, figsize=(21, 6), layout='constrained')
    for k, a in enumerate(ABAR):
        im = axes[k].imshow(maps[k], cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[k].set_title(f'$\\bar\\alpha$ = {a}', fontsize=22); _net_axes(axes[k])
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, shrink=0.6, aspect=20)
    cb.set_label('rank-sum statistic\n(positive: stronger in cognitive)', fontsize=14)
    save(fig, 'SuppFig25_alpha_cogperc')


# ---------------- Figure 3: control hub two-step ----------------
def fig_hub():
    scores = per_graph_scores()
    fig, axes = plt.subplots(len(ABAR), 3, figsize=(16, 4.6 * len(ABAR)), layout='constrained')
    col_t = ['Significant NF $-$ Baseline\n(rank-sum, FDR; hub circled)',
             'Significant score correlation\n(Spearman, FDR)',
             'Control-network parcel hub\n(stronger in NF AND + score)']
    for k, a in enumerate(ABAR):
        d = load(a); nf16, dc16 = d['nf_weighted'], d['dc_weighted']
        nfP, dcP = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
        idx = d['nf_idx']
        sc = np.array([scores.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in idx])
        ok = np.isfinite(sc)
        _, _, hub16, nfdc16, scorr16 = two_step(nf16, dc16, nf16[ok], sc[ok])
        _, _, hubP, mdP, _ = two_step(nfP, dcP, nfP[ok], sc[ok])
        cbk = hubP[CTRL, CTRL]; md_ctrl = mdP[CTRL, CTRL]

        v1 = np.nanpercentile(np.abs(nfdc16[nfdc16 != 0]), 99) if (nfdc16 != 0).any() else 1
        im = axes[k, 0].imshow(nfdc16, cmap='coolwarm', vmin=-v1, vmax=v1)
        yi, xi = np.where(hub16)
        axes[k, 0].scatter(xi, yi, s=150, facecolors='none', edgecolors='lime', linewidths=2.4)
        fig.colorbar(im, ax=axes[k, 0], fraction=.046, pad=.04)
        v2 = np.nanpercentile(np.abs(scorr16[scorr16 != 0]), 99) if (scorr16 != 0).any() else 0.1
        im = axes[k, 1].imshow(scorr16, cmap='coolwarm', vmin=-v2, vmax=v2)
        fig.colorbar(im, ax=axes[k, 1], fraction=.046, pad=.04)
        for c in (0, 1):
            _net_axes(axes[k, c])
        nc = cbk.shape[0]
        v3 = np.nanpercentile(np.abs(md_ctrl[cbk]), 99) if cbk.any() else (np.abs(md_ctrl).max() or 1)
        im = axes[k, 2].imshow(np.where(cbk, md_ctrl, 0), cmap='Reds', vmin=0, vmax=v3)
        fig.colorbar(im, ax=axes[k, 2], fraction=.046, pad=.04)
        axes[k, 2].set_xticks(range(nc)); axes[k, 2].set_yticks(range(nc))
        axes[k, 2].set_xticklabels(CTRL_LABELS, rotation=90, fontsize=7)
        axes[k, 2].set_yticklabels(CTRL_LABELS, fontsize=7)
        axes[k, 2].set_xlabel(f'{int(cbk.sum())} within-control hub edges', fontsize=12)
        axes[k, 0].set_ylabel(f'$\\bar\\alpha$ = {a}', fontsize=22, weight='bold')
        if k == 0:
            for c in range(3):
                axes[k, c].set_title(col_t[c], fontsize=13)
        print(f'abar={a}: subnet hub={int(hub16.sum())}, within-control parcel edges={int(cbk.sum())}')
    save(fig, 'SuppFig22_alpha_control_hub')


if __name__ == '__main__':
    fig_cogperc()
    fig_hub()
    print('figures written to', FINAL)
