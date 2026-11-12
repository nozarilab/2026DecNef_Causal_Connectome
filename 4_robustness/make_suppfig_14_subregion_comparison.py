"""Supplementary Fig. 14: anatomical sub-region composition of the control hub across parcellations.

For each control sub-region (pCun, Cing, Par, PFCl, PFCv, PFCmp, Temp, OFC), counts the
within-control hub edges with an endpoint in it, for the Schaefer-100 analysis and the two
Schaefer-200 four-run designs.
Output: Results/figures_supplementary/SuppFig14_s200_subregion_comparison.{pdf,png}
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import re
import collections
import numpy as np
from scipy.stats import ranksums, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_fig3_alpha import per_graph_scores
from compare_s200_windowing import window_scores, select
import decnef_parcels as dp116
import decnef_parcels216 as dp216

OUT = 'Results/glm_revision/figures'
SUBREGS = ['pCun', 'Cing', 'Par', 'PFCl', 'PFCv', 'PFCmp', 'Temp', 'OFC']


def two_step(nf, dc, score, ok):
    P = nf.shape[1]; ms = np.zeros((P, P)); mp = np.ones((P, P)); cr = np.zeros((P, P)); cp = np.ones((P, P))
    for i in range(P):
        for j in range(P):
            a, b = nf[:, i, j], dc[:, i, j]
            if a.std() > 0 or b.std() > 0:
                t = ranksums(a, b); ms[i, j], mp[i, j] = t.statistic, t.pvalue
            w = nf[ok, i, j]
            if w.std() > 0:
                s = spearmanr(w, score[ok], nan_policy='omit'); cr[i, j], cp[i, j] = s.statistic, s.pvalue
    return ((ms > 0) & fdrcorrection(mp.ravel())[0].reshape(P, P)) & ((cr > 0) & fdrcorrection(cp.ravel())[0].reshape(P, P))


def subreg(lbl):
    m = re.search(r'(LH|RH)_Cont_([A-Za-z]+)', lbl)
    return m.group(2) if m else '?'


def ctrl_labels(dpmod, lo, hi):
    PL = dpmod.get_parcel_labels(); NP = dpmod.get_network_parcel_idx(PL)
    REARR = [PL[i] for net in dpmod.NET_LABELS for i in NP[net]]
    return [l.replace('7Networks_', '') for l in REARR[lo:hi]]


def hub_endpoints(hub, labs):
    cnt = collections.Counter()
    for i, j in zip(*np.where(hub)):
        cnt[subreg(labs[i])] += 1; cnt[subreg(labs[j])] += 1
    return cnt


def get_score(idx, W):
    sc = window_scores(W)
    return np.array([sc.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in idx])


def main():
    # --- paper 116
    d = np.load('Results/glm_revision/alpha_sweep/subnet_abar_0.05.npz'); idx = d['nf_idx']
    sc = per_graph_scores(); score = np.array([sc.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in idx]); ok = np.isfinite(score)
    labs116 = ctrl_labels(dp116, 38, 51)
    h116 = two_step(d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float), score, ok)[38:51, 38:51]
    ep116 = hub_endpoints(h116, labs116)
    pool116 = collections.Counter(subreg(l) for l in labs116)

    # --- 216 control labels & pool
    d2 = np.load('Results/glm_revision/alpha_sweep_s200/subnet_abar_0.05_s200.npz', allow_pickle=True)
    clo, chi = int(d2['control_block'][0]), int(d2['control_block'][1])
    labs216 = ctrl_labels(dp216, clo, chi); pool216 = collections.Counter(subreg(l) for l in labs216)

    d4 = np.load('Results/glm_revision/s200_4run_graphs/s200_4run_weighted_abar0.05.npz', allow_pickle=True)
    sc4 = get_score(d4['nf_idx'], 4); ok4 = np.isfinite(sc4)
    h4 = two_step(d4['nf_parcel_w'].astype(float), d4['dc_parcel_w'].astype(float), sc4, ok4)[clo:chi, clo:chi]
    ep4 = hub_endpoints(h4, labs216)

    # --- 216 four-run, stride-2 / overlap-2 (windowing design c): NF windows {0,2,4}, DC {0,2}
    nm2, dm2 = select(d4['nf_idx'], {0, 2, 4}), select(d4['dc_idx'], {0, 2})
    sc4b = get_score(d4['nf_idx'][nm2], 4); ok4b = np.isfinite(sc4b)
    h4b = two_step(d4['nf_parcel_w'][nm2].astype(float), d4['dc_parcel_w'][dm2].astype(float),
                   sc4b, ok4b)[clo:chi, clo:chi]
    ep4b = hub_endpoints(h4b, labs216)

    # --- table figure: rows = sub-regions, cols = pools + hub endpoint counts per analysis
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'pdf.fonttype': 42})
    FULL = {'pCun': 'precuneus (pCun)', 'Cing': 'post. cingulate (Cing)',
            'Par': 'lateral parietal (Par)', 'PFCl': 'lateral PFC (PFCl)',
            'PFCv': 'ventral PFC (PFCv)', 'PFCmp': 'medial-post. PFC (PFCmp)',
            'Temp': 'temporal (Temp)', 'OFC': 'orbitofrontal (OFC)'}
    rows = [s for s in SUBREGS if pool116.get(s) or pool216.get(s)]
    row_labels = [FULL.get(s, s) for s in rows]
    col_labels = ['# control parcels\n(S100 / S200)', 'Schaefer-100\n(paper, 2-run)',
                  'Schaefer-200\n4-run, overlap 3 (a)', 'Schaefer-200\n4-run, overlap 2 (b)']
    cell, colors = [], []
    for s in rows:
        c = [f'{pool116.get(s,0)} / {pool216.get(s,0)}', ep116.get(s, 0),
             ep4.get(s, 0), ep4b.get(s, 0)]
        cell.append([str(x) for x in c])
        colors.append(['white'] + ['#bfe3bf' if c[k] else 'white' for k in (1, 2, 3)])

    fig, ax = plt.subplots(figsize=(10.6, 0.62 * len(rows) + 2.6)); ax.axis('off')
    tbl = ax.table(cellText=cell, rowLabels=row_labels, colLabels=col_labels, cellColours=colors,
                   cellLoc='center', rowLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.0)
    for (r, cc), cell_obj in tbl.get_celld().items():
        if r == 0 or cc == -1:
            cell_obj.set_text_props(weight='bold')
    FINAL = 'Results/figures_supplementary'
    os.makedirs(FINAL, exist_ok=True)
    fig.canvas.draw()                                    # crop tightly to the table itself
    bb = tbl.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    bb = bb.expanded(1.03, 1.08)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FINAL, f'SuppFig14_s200_subregion_comparison.{ext}'), dpi=200, bbox_inches=bb)
    print('saved', os.path.join(FINAL, 'SuppFig14_s200_subregion_comparison.pdf'))
    print('116 hub endpoints:', dict(ep116))
    print('216 4-run overlap-3:', dict(ep4), '| 216 4-run overlap-2:', dict(ep4b))
    print('control-net pools: S100', dict(pool116), '| S200', dict(pool216))


if __name__ == '__main__':
    main()
