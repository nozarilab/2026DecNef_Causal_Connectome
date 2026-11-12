"""Two-step control-hub maps for the 216-parcel designs (used by make_suppfigs_12_13_s200.py).

For each way of forming the graphs: edges stronger in NF than baseline (rank-sum
statistic, FDR over the whole matrix), edges correlated with score (Spearman, FDR over the
whole matrix), their intersection, and the parcel-level zoom into the control network.
"""
import os
import numpy as np
from scipy.stats import ranksums, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_s200_windowing import window_scores, NET, CTRL_NET, CTRL_LABELS, select

S2 = 'Results/glm_revision/alpha_sweep_s200/subnet_abar_0.05_s200.npz'
S4 = 'Results/glm_revision/s200_4run_graphs/s200_4run_weighted_abar0.05.npz'
OUT = 'Results/glm_revision/figures'
CLO, CHI = 57, 87

CASES = [
    ('1. 2-run, lag-0 only',           S2, 2, {0,1,2,3,4,5,6}, {0,1,2,3,4}),
    ('2. 4-run, stride-1 (overlap 3)', S4, 4, {0,1,2,3,4},     {0,1,2}),
    ('3. 4-run, stride-2 (overlap 2)', S4, 4, {0,2,4},         {0,2}),
    ('4. 4-run, non-overlap',          S4, 4, {0,4},           {0}),
]


def two_step_maps(nf, dc, score, ok):
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
    mod_sig = fdrcorrection(mp.ravel())[0].reshape(P, P)
    cor_sig = fdrcorrection(cp.ravel())[0].reshape(P, P)
    hub = ((ms > 0) & mod_sig) & ((cr > 0) & cor_sig)          # intersection
    return ms * mod_sig, cr * cor_sig, hub, ms, cr             # signed maps + hub


def load_case(npz, W, nfp, dcp):
    d = np.load(npz, allow_pickle=True)
    nm = select(d['nf_idx'], nfp); dm = select(d['dc_idx'], dcp)
    sc = window_scores(W)
    nfidx = d['nf_idx'][nm]
    score = np.array([sc.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])
    return (d['nf_weighted'][nm], d['dc_weighted'][dm], d['nf_parcel_w'][nm].astype(float),
            d['dc_parcel_w'][dm].astype(float), score)


def main():
    fig, ax = plt.subplots(len(CASES), 4, figsize=(20, 5 * len(CASES)), layout='constrained')
    for row, (name, npz, W, nfp_sel, dcp_sel) in enumerate(CASES):
        nf16, dc16, nfp, dcp, score = load_case(npz, W, nfp_sel, dcp_sel)
        ok = np.isfinite(score)
        nfdc, scor, hub, ms, cr = two_step_maps(nf16, dc16, score, ok)
        ctrl_edges = [(NET[i], NET[j]) for i, j in zip(*np.where(hub)) if NET[i] in CTRL_NET and NET[j] in CTRL_NET]
        print(f'{name}: NF={len(nf16)} DC={len(dc16)} | subnet hub={int(hub.sum())} control={ctrl_edges}')

        # col A: significant NF - Baseline (signed statistic x FDR)
        v = np.max(np.abs(ms)) or 1
        im = ax[row, 0].imshow(nfdc, cmap='coolwarm', vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax[row, 0], fraction=.046, pad=.04).ax.tick_params(labelsize=6)
        # col B: significant correlation with score (signed statistic x FDR)
        v = np.max(np.abs(cr)) or 1
        im = ax[row, 1].imshow(scor, cmap='coolwarm', vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax[row, 1], fraction=.046, pad=.04).ax.tick_params(labelsize=6)
        # col C: intersection hub (binary)
        ax[row, 2].imshow(hub.astype(float), cmap='Greens', vmin=0, vmax=1)
        for c in (0, 1, 2):
            ax[row, c].set_xticks(range(16)); ax[row, c].set_yticks(range(16))
            ax[row, c].set_xticklabels(NET, rotation=90, fontsize=6); ax[row, c].set_yticklabels(NET, fontsize=6)
        if row == 0:
            ax[row, 0].set_title('Significant NF - Baseline (rank-sum stat x FDR)', fontsize=10)
            ax[row, 1].set_title('Significant correlation with score (FDR)', fontsize=10)
            ax[row, 2].set_title('Intersection: stronger-in-NF AND +score', fontsize=10)
            ax[row, 3].set_title('Control parcel zoom (if control survived)', fontsize=10)
        ax[row, 0].set_ylabel(f'{name}\nNF={len(nf16)} DC={len(dc16)}', fontsize=9)
        ax[row, 2].set_xlabel(f'{int(hub.sum())} hub edges; control: '
                              f'{ctrl_edges if ctrl_edges else "NONE"}', fontsize=8)

        # col D: control parcel zoom -- ONLY if a control edge survived the network gate
        a3 = ax[row, 3]
        if ctrl_edges:
            hubP, msP, crP = two_step_maps(nfp, dcp, score, ok)[2], None, None
            cb = slice(CLO, CHI); hc = hubP[cb, cb]
            wb = int(hubP.sum()); inc = int(hc.sum())
            a3.imshow(hc.astype(float), cmap='Greens', vmin=0, vmax=1)
            a3.set_xticks(range(CHI - CLO)); a3.set_yticks(range(CHI - CLO))
            a3.set_xticklabels(CTRL_LABELS, rotation=90, fontsize=4.5); a3.set_yticklabels(CTRL_LABELS, fontsize=4.5)
            a3.set_xlabel(f'within-control={inc} of {wb} whole-brain ({100*inc/max(wb,1):.0f}%)', fontsize=8)
            print(f'    parcel zoom: whole-brain={wb}, within-control={inc}')
        else:
            a3.text(0.5, 0.5, 'no control edge\nat network level\n-> no zoom', ha='center', va='center',
                    fontsize=12, color='gray', transform=a3.transAxes)
            a3.set_xticks([]); a3.set_yticks([])

    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_s200_windowing_maps.{ext}'), dpi=130, bbox_inches='tight')
    print('saved', os.path.join(OUT, 'fig_s200_windowing_maps.png'))


if __name__ == '__main__':
    main()
