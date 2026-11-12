"""Helpers for the 216-parcel control-hub analysis (Supplementary Figs. 13-14).

window_scores(W): neurofeedback score matched to the runs of each graph (mean of the
within-study z-scored logit score over the W runs that formed the graph; W = 2 gives the
score used in the main analysis). select(): picks the graphs of a given windowing design.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ranksums, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import decnef_parcels216 as dp216

SCORES = 'Results/glm_revision/scores'
SES_OFFSET = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
SUBJECT_IDS = {
    1: ['03', '24', '21', '15', '23', '05', '17', '13', '18'],
    2: ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    3: ['01', '05', '08', '10', '12', '13', '14', '15', '16'],
    4: ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    5: ['01', '02', '03', '06', '08', '09', '11', '12', '15'],
}
N_SUB, N_SES, N_TRIAL, N_RUN = 9, 3, 15, 8
NET = ['L-Sub', 'R-Sub', 'L-Vis', 'R-Vis', 'L-Lim', 'R-Lim', 'L-Cont', 'R-Cont',
       'L-DMN', 'R-DMN', 'L-Vent-Att', 'R-Vent-Att', 'L-Dors-Att', 'R-Dors-Att', 'L-Som-Mot', 'R-Som-Mot']
CTRL_NET = ('L-Cont', 'R-Cont')

PL = dp216.get_parcel_labels()
M_npidx = dp216.get_network_parcel_idx(PL)
CTRL_PARCELS = list(np.asarray(M_npidx['LH_Con'])) + list(np.asarray(M_npidx['RH_Con']))
CTRL_LABELS = [PL[i].replace('7Networks_', '').replace('LH_Cont_', 'L-').replace('RH_Cont_', 'R-') for i in CTRL_PARCELS]


def window_scores(W):
    """Mean within-study z-scored logit(in-score) over the W consecutive runs of each
    window (start run r). W=2 reproduces the per-graph score of the main analysis."""
    n_win = N_RUN - W + 1
    out = {}
    for s in range(1, 6):
        inv = np.full((N_SUB, N_SES, n_win, W * N_TRIAL), np.nan)
        for si, sub in enumerate(SUBJECT_IDS[s]):
            for ses in range(N_SES):
                f = os.path.join(SCORES, f'Study{s}', f'sub-{sub}', 'pattern', f'in_score_d{ses + SES_OFFSET[s]}.csv')
                if not os.path.exists(f):
                    continue
                vals = pd.read_csv(f).values[:120].astype(float).ravel()
                for r in range(n_win):
                    seg = vals[r * N_TRIAL:(r + W) * N_TRIAL]
                    iv = logit(seg); iv[np.isinf(iv)] = np.nan
                    inv[si, ses, r, :len(iv)] = iv
        mu, sd = np.nanmean(inv), np.nanstd(inv)
        z = (inv - mu) / sd
        for si in range(N_SUB):
            for ses in range(N_SES):
                for r in range(n_win):
                    out[(s, si, ses, r)] = np.nanmean(z[si, ses, r])
    return out


def two_step(nf, dc, score, ok):
    P = nf.shape[1]
    ms = np.zeros((P, P)); mp = np.ones((P, P)); cr = np.zeros((P, P)); cp = np.ones((P, P))
    for i in range(P):
        for j in range(P):
            a, b = nf[:, i, j], dc[:, i, j]
            if a.std() > 0 or b.std() > 0:
                t = ranksums(a, b, alternative='two-sided'); ms[i, j], mp[i, j] = t.statistic, t.pvalue
            w = nf[ok, i, j]
            if w.std() > 0:
                s = spearmanr(w, score[ok], alternative='two-sided', nan_policy='omit'); cr[i, j], cp[i, j] = s.statistic, s.pvalue
    stronger = (ms > 0) & fdrcorrection(mp.ravel())[0].reshape(P, P)
    pos = (cr > 0) & fdrcorrection(cp.ravel())[0].reshape(P, P)
    return stronger & pos, ms, cr


def select(idx, pairs):
    return np.array([int(p) in pairs for *_, p in idx])


def run_case(name, npz, W, nf_pairs, dc_pairs, clo, chi):
    d = np.load(npz, allow_pickle=True)
    nf16, dc16 = d['nf_weighted'], d['dc_weighted']
    nfp, dcp = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
    nfidx, dcidx = d['nf_idx'], d['dc_idx']
    nm = select(nfidx, nf_pairs); dm = select(dcidx, dc_pairs)
    nf16, dc16, nfp, dcp, nfidx = nf16[nm], dc16[dm], nfp[nm], dcp[dm], nfidx[nm]
    sc = window_scores(W)
    score = np.array([sc.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])
    ok = np.isfinite(score)
    print(f'\n========== {name} ==========')
    print(f'NF={len(nf16)} DC={len(dc16)} graphs | score window W={W}')

    hub16, ms16, cr16 = two_step(nf16, dc16, score, ok)
    hub_edges = [(NET[i], NET[j]) for i, j in zip(*np.where(hub16))]
    ctrl_hub = [(a, b) for a, b in hub_edges if a in CTRL_NET and b in CTRL_NET]
    print(f'SUBNET hub ({int(hub16.sum())} edges): {hub_edges}')
    print(f'   within-control: {ctrl_hub if ctrl_hub else "NONE -> would STOP here, no zoom"}')

    if not ctrl_hub:
        return name, len(nf16), len(dc16), hub_edges, None, None
    # parcel zoom only if a control edge survived the network-level gate
    hubP, msP, crP = two_step(nfp, dcp, score, ok)
    cb = slice(clo, chi)
    hc = hubP[cb, cb]; mc = msP[cb, cb]; cc = crP[cb, cb]
    print(f'PARCEL: whole-brain hub={int(hubP.sum())}, within-control[{clo}:{chi}]={int(hc.sum())} '
          f'({100*hc.sum()/max(hubP.sum(),1):.0f}%)')
    parcels = set()
    for i, j in zip(*np.where(hc)):
        print(f'     {CTRL_LABELS[i]:>9s} -> {CTRL_LABELS[j]:<9s} (z={mc[i,j]:+.2f}, rho={cc[i,j]:+.3f})')
        parcels |= {CTRL_LABELS[i], CTRL_LABELS[j]}
    print('   control parcels involved:', sorted(parcels))
    return name, len(nf16), len(dc16), hub_edges, int(hubP.sum()), int(hc.sum())


def main():
    s2 = 'Results/glm_revision/alpha_sweep_s200/subnet_abar_0.05_s200.npz'          # 2-run lag0
    s4 = 'Results/glm_revision/s200_4run_graphs/s200_4run_weighted_abar0.05.npz'    # 4-run lag0+lag1
    d2 = np.load(s2, allow_pickle=True); clo, chi = int(d2['control_block'][0]), int(d2['control_block'][1])

    # check that window_scores(2) reproduces the per-graph score of the main analysis
    from make_fig3_alpha import per_graph_scores
    a = per_graph_scores(); b = window_scores(2)
    keys = set(a) & set(b)
    diff = np.nanmax([abs(a[k] - b[k]) for k in keys if np.isfinite(a[k]) and np.isfinite(b[k])])
    print(f'window_scores(2) vs per_graph_scores: max abs diff = {diff:.2e} over {len(keys)} keys')

    rows = []
    rows.append(run_case('1. 2-run, lag-0 only', s2, 2, {0,1,2,3,4,5,6}, {0,1,2,3,4}, clo, chi))
    rows.append(run_case('2. 4-run, stride-1 (overlap 3)', s4, 4, {0,1,2,3,4}, {0,1,2}, clo, chi))
    rows.append(run_case('3. 4-run, stride-2 (overlap 2)', s4, 4, {0,2,4}, {0,2}, clo, chi))
    rows.append(run_case('4. 4-run, non-overlap', s4, 4, {0,4}, {0}, clo, chi))

    print('\n\n================= SUMMARY =================')
    print(f'{"case":34s} {"NF":>4s} {"DC":>4s} {"subnet hub":>11s} {"ctrl?":>6s} {"WB hub":>7s} {"in-ctrl":>8s}')
    for nm, nnf, ndc, he, wb, ic in rows:
        ctrl = any(a in CTRL_NET and b in CTRL_NET for a, b in he)
        print(f'{nm:34s} {nnf:4d} {ndc:4d} {len(he):11d} {("yes" if ctrl else "no"):>6s} '
              f'{(str(wb) if wb is not None else "-"):>7s} {(str(ic) if ic is not None else "-"):>8s}')


if __name__ == '__main__':
    main()
