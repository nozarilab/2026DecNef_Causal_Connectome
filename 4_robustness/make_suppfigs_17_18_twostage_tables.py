"""Supplementary Figs. 17-18: per-edge statistics of the two-stage discovery-test analysis.

Discovery on NF session 1, test on sessions 2-3, non-overlapping graphs.
  SuppFig17_twostage_subnet_table   subnetwork candidates (top 10 carried to the test)
  SuppFig18_twostage_parcel_table   within-control parcel edges
Output folder: Results/figures_supplementary/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ranksums, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from make_fig3_alpha import per_graph_scores
from make_suppfigs_22_25_alpha import VIS
import make_suppfigs_23_24_alpha_brainmaps as BM

SWEEP = 'Results/glm_revision/alpha_sweep/subnet_abar_0.05.npz'
FINAL = 'Results/figures_supplementary'
CB = list(range(38, 51))
L = BM.ZOOM_LBL
ZPOS = BM.ZOOM_POS


def hemi(k):
    return 'L' if ZPOS[k][0] < 0.5 else 'R'


def load():
    d = np.load(SWEEP)
    nf16, dc16 = d['nf_weighted'].astype(float), d['dc_weighted'].astype(float)
    nfP, dcP = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
    nfidx, dcidx = d['nf_idx'], d['dc_idx']
    sc = np.array([per_graph_scores().get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])
    ses, npair, dpair = nfidx[:, 2], nfidx[:, 3], dcidx[:, 3]
    nf_ok, dc_ok = np.isin(npair, [0, 2, 4, 6]), np.isin(dpair, [0, 2, 4])
    disc = (ses == 0) & nf_ok; okd = np.isfinite(sc[disc])
    test = np.isin(ses, [1, 2]) & nf_ok & np.isfinite(sc)
    return dict(scd=sc[disc], okd=okd, Yt=sc[test],
                nfd16=nf16[disc], DCd16=dc16[dc_ok], NFt16=nf16[test], DCt16=dc16[dc_ok],
                nfdP=nfP[disc], DCdP=dcP[dc_ok], NFtP=nfP[test], DCtP=dcP[dc_ok])


def stats(NFd, DCd, scd, okd, NFt, DCt, Yt, edges):
    cand = []
    dpv = {}                                             # discovery p-values (descriptive only)
    for e in edges:
        m0 = ranksums(NFd[:, e[0], e[1]], DCd[:, e[0], e[1]]); ms = m0.statistic
        col = NFd[okd, e[0], e[1]]
        r0, p0 = spearmanr(col, scd[okd]) if col.std() > 0 else (0.0, 1.0)
        if ms > 0 and r0 > 0:
            cand.append((ms, r0, e))
            dpv[e] = (m0.pvalue / 2.0, p0)               # 1-sided modulation p, score p
    cand.sort(reverse=True, key=lambda x: x[0])
    dpmod = np.array([dpv[e][0] for ms, rho, e in cand])
    dpsc = np.array([dpv[e][1] for ms, rho, e in cand])
    pmod, psc, rsc, mstat = [], [], [], []
    for ms, rho, e in cand:
        m = ranksums(NFt[:, e[0], e[1]], DCt[:, e[0], e[1]])
        pmod.append(m.pvalue if m.statistic > 0 else 1.0); mstat.append(m.statistic)
        r, p = spearmanr(NFt[:, e[0], e[1]], Yt); rsc.append(r); psc.append(p)
    smod, amod = (fdrcorrection(pmod)[0], fdrcorrection(pmod)[1]) if pmod else (np.array([]), np.array([]))
    ssc, asc = (fdrcorrection(psc)[0], fdrcorrection(psc)[1]) if psc else (np.array([]), np.array([]))
    return cand, np.array(mstat), amod, smod, np.array(rsc), np.array(psc), asc, ssc, dpmod, dpsc


def render(rows, cols, colw, title, fname, figsize):
    cell = [r[0] for r in rows]; colors = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=figsize); ax.axis('off')
    tbl = ax.table(cellText=cell, colLabels=cols, cellColours=colors, colWidths=colw, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.6)
    for (r, c), obj in tbl.get_celld().items():
        if r == 0:
            obj.set_text_props(weight='bold'); obj.set_facecolor('#efefef')
            obj.set_height(obj.get_height() * 1.8)          # taller header row so 2-line labels fit
        if cell and r >= 1 and cell[r - 1][-1] == 'YES':
            obj.set_text_props(weight='bold')
        if c == 1:
            obj.set_text_props(ha='left')
    _ = title                                            # no titles; captions are in the paper
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'{fname}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


K = 10                                                       # top-K by discovery NF>baseline


def subnet_table(D):
    # full discovery ranking over ALL 16x16 subnetwork edges (candidates = stronger in NF &
    # +score on session 1); the top K by NF>baseline are carried to the held-out test, FDR over K.
    alle = [(i, j) for i in range(16) for j in range(16)]
    cand, _, _, _, _, _, _, _, dpmA, dpsA = stats(D['nfd16'], D['DCd16'], D['scd'], D['okd'],
                                                   D['NFt16'], D['DCt16'], D['Yt'], alle)
    discp = {e: (dpmA[k], dpsA[k]) for k, (ms, rho, e) in enumerate(cand)}
    topk = [e for (ms, rho, e) in cand[:K]]
    tt, _, tamod, tsmod, trsc, tpsc, tasc, tssc, _, _ = stats(D['nfd16'], D['DCd16'], D['scd'], D['okd'],
                                                              D['NFt16'], D['DCt16'], D['Yt'], topk)
    ctest = {e: dict(am=am, sm=sm, rs=rs, ps=ps, asci=asci, ss=ss)
             for (ms, rho, e), am, sm, rs, ps, asci, ss in zip(tt, tamod, tsmod, trsc, tpsc, tasc, tssc)}

    cols = ['rank', 'edge', 'discovery\nNF>base', 'disc NF>base\np-value (1-sided)', 'discovery\nscore-$\\rho$',
            'disc score\np-value', 'carried to\nheld-out test', 'test\nNF>base FDR', 'test\nscore-$\\rho$',
            f'test score\nFDR (/{K})', 'HUB']
    rows = []
    for k, (ms, rho, e) in enumerate(cand, 1):
        dpm, dps = discp[e]
        tested = e in ctest
        if tested:
            t = ctest[e]; hub = 'YES' if (t['sm'] and t['ss']) else ''
            trow = ['YES', f"{t['am']:.3f}" + (' *' if t['sm'] else ''), f"{t['rs']:+.3f}",
                    f"{t['asci']:.3f}" + (' *' if t['ss'] else ''), hub]
        else:
            trow = ['', '—', '—', '—', '']
        ishub = tested and ctest[e]['sm'] and ctest[e]['ss']
        rows.append(([str(k), f'{VIS[e[0]]} → {VIS[e[1]]}', f'{ms:+.2f}', f'{dpm:.3f}', f'{rho:+.3f}', f'{dps:.3f}'] + trow,
                     ['#c9e8c9' if ishub else '#f4f4f4' if tested else 'white'] * len(cols)))
    render(rows, cols, [0.04, 0.16, 0.085, 0.13, 0.085, 0.105, 0.10, 0.10, 0.085, 0.10, 0.05],
           'Subnetwork split-half (non-overlap runs, graph-level test): all candidate edges ranked by '
           'discovery NF>baseline\n'
           f'candidates = stronger in NF AND + score-correlated on session 1 (discovery p-values shown for '
           f'reference only, not used to filter).\nThe top {K} by NF>baseline (shaded) are carried to the '
           f'held-out test (sessions 2-3), FDR over those {K}.  Only the two control self-loops (green) '
           'survive both tests.  * = survives FDR;  — = not tested',
           'SuppFig17_twostage_subnet_table', (18.5, 0.42 * len(rows) + 1.8))
    hubs = [f'{VIS[e[0]]}->{VIS[e[1]]}' for e in ctest if ctest[e]['sm'] and ctest[e]['ss']]
    print(f'subnet: {len(cand)} candidates; top {K} tested; HUB={hubs}')


def zoom_table(D):
    edges = [(i, j) for i in CB for j in CB if i != j]
    cand, mstat, amod, smod, rsc, psc, asc, ssc, dpmod, dpsc = stats(D['nfdP'], D['DCdP'], D['scd'], D['okd'],
                                                                     D['NFtP'], D['DCtP'], D['Yt'], edges)
    keep = [k for k in range(len(cand)) if smod[k]]           # drawn edges (stronger in NF)
    cols = ['rank', 'edge', 'discovery\nNF>base', 'disc NF>base\np-value (1-sided)', 'discovery\nscore-$\\rho$',
            'disc score\np-value', 'test\nNF>base FDR', 'test\nscore-$\\rho$', 'test\nscore p-value',
            f'test score\nFDR (/{len(cand)})', 'HUB']
    rows = []
    for rank, k in enumerate(keep, 1):
        ms, rho, e = cand[k]; a, b = CB.index(e[0]), CB.index(e[1])
        hub = 'YES' if (smod[k] and ssc[k]) else ''
        rows.append(([str(rank), f'{hemi(a)}-{L[a]} → {hemi(b)}-{L[b]}', f'{ms:+.2f}', f'{dpmod[k]:.3f}',
                      f'{rho:+.3f}', f'{dpsc[k]:.3f}', f'{amod[k]:.3f} *', f'{rsc[k]:+.3f}', f'{psc[k]:.4f}',
                      f'{asc[k]:.3f}' + (' *' if ssc[k] else ''), hub],
                     ['#c9e8c9' if hub else 'white'] * len(cols)))
    render(rows, cols, [0.04, 0.14, 0.085, 0.13, 0.085, 0.105, 0.095, 0.085, 0.105, 0.10, 0.05],
           'Whole control-network zoom split-half (non-overlap runs, graph-level test)\n'
           'edges stronger in NF on held-out data, ranked by discovery NF>baseline (discovery p-values '
           'shown for reference only, not used to filter);\nFDR over the '
           f'{len(cand)} within-control candidate edges;  * = survives FDR',
           'SuppFig18_twostage_parcel_table', (18, 0.5 * len(keep) + 1.6))
    print(f'zoom: {len(cand)} candidates, {len(keep)} stronger-in-NF drawn, {int((smod & ssc).sum())} hubs')


if __name__ == '__main__':
    os.makedirs(FINAL, exist_ok=True)
    D = load()
    subnet_table(D)
    zoom_table(D)
