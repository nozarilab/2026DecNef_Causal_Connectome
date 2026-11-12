"""Subnetwork and parcel graphs from the GLM task-regressed CaLLTiF outputs.

process(cond, variant) builds the weighted 16x16 subnetwork and 116x116 parcel graphs for
every NF and DC graph, with the same per-graph summarization as the main analysis
(contemporaneous alpha = 0.0025, relaxed lagged threshold alpha-bar = 0.05). Used by
make_suppfig_11_cogperc_glm.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import re
import glob
import numpy as np
from scipy.stats import spearmanr, ranksums
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functions as funcs
import decnef_parcels as dp
from make_fig3_alpha import per_graph_scores

ALPHA0 = 1e-2 / 4
ABAR = 0.05
RATIO = 20
SRC = 'Results/glm_revision/glm_results_9subj'
OUT = 'Results/glm_revision/figures'
CTRL = slice(38, 51)                       # control parcels in rearranged 116 order
NET = ['L-Sub', 'R-Sub', 'L-Vis', 'R-Vis', 'L-Lim', 'R-Lim', 'L-Cont', 'R-Cont',
       'L-DMN', 'R-DMN', 'L-Vent-Att', 'R-Vent-Att', 'L-Dors-Att', 'R-Dors-Att',
       'L-Som-Mot', 'R-Som-Mot']

PARCEL_LABELS = dp.get_parcel_labels()
NET_LABELS = dp.NET_LABELS
NETWORK_DIC = '{' + ','.join("'%s':[]" % k for k in NET_LABELS) + '}'
NPIDX = dp.get_network_parcel_idx(PARCEL_LABELS)
REARR_LABELS = [PARCEL_LABELS[i] for net in NET_LABELS for i in NPIDX[net]]
CTRL_LABELS = [l.replace('LH_Cont_', 'L-').replace('RH_Cont_', 'R-') for l in REARR_LABELS[38:51]]


def subnet(matrix):
    return funcs.network_map_v2_without_decoder(matrix, NET_LABELS, NETWORK_DIC, PARCEL_LABELS)


def process(cond, variant):
    """Reconstruct weighted subnet (16x16) + parcel (116x116) graphs from residual npz."""
    files = sorted(glob.glob(os.path.join(SRC, f'{cond}_{variant}_9subj_study*_subj*_ses*_pair*.npz')))
    alpha_summ = ABAR / RATIO
    sub_w, par_w, idx = [], [], []
    for f in files:
        m = re.search(r'study(\d+)_subj(\d+)_ses(\d+)_pair(\d+)', os.path.basename(f))
        study, su, se, r = (int(m.group(k)) for k in (1, 2, 3, 4))
        g = np.load(f); p, val = g['p_matrix'], g['val_matrix']
        sg = funcs.convert_to_string_graph((p <= ALPHA0).astype(int))
        _, _, weighted = funcs.summarize_across_all_lags_v2_with_pvalues_more_directed_v2(
            sg, p, val, alpha_summ)
        sub_w.append(subnet(weighted))
        par_w.append(np.asarray(funcs.rearange_matrix_v2(weighted, NPIDX), dtype=np.float32))
        idx.append((study, su, se, r))
    return np.array(sub_w), np.array(par_w, dtype=np.float32), np.array(idx)


def two_step(nf, dc, score_ok_nf, score):
    """Two-step control-hub analysis (same as make_fig3_alpha.two_step)."""
    P = nf.shape[1]
    modstat = np.zeros((P, P)); modp = np.ones((P, P))
    corr = np.zeros((P, P)); corrp = np.ones((P, P))
    for i in range(P):
        for j in range(P):
            x, y = nf[:, i, j], dc[:, i, j]
            if x.std() > 0 or y.std() > 0:
                t = ranksums(x, y); modstat[i, j], modp[i, j] = t.statistic, t.pvalue
            xs = score_ok_nf[:, i, j]
            if xs.std() > 0:
                tt = spearmanr(xs, score, nan_policy='omit')
                corr[i, j], corrp[i, j] = tt.statistic, tt.pvalue
    modsig = fdrcorrection(modp.ravel())[0].reshape(P, P)
    corrsig = fdrcorrection(corrp.ravel())[0].reshape(P, P)
    stronger = (modstat > 0) & modsig
    poscorr = (corr > 0) & corrsig
    return stronger, poscorr, (stronger & poscorr), (modstat * modsig), (corr * corrsig)


def run_variant(variant, scores, make_fig):
    nf16, nfP, nfidx = process('nf', variant)
    dc16, dcP, _ = process('dc', variant)
    sc = np.array([scores.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])
    ok = np.isfinite(sc)
    print(f'\n=== {variant.upper()} residual graphs: NF={len(nf16)} DC={len(dc16)} | scored NF graphs={int(ok.sum())} ===')

    # subnetwork two-step
    _, _, hub16, nfdc16, scorr16 = two_step(nf16, dc16, nf16[ok], sc[ok])
    sub_hub = [(NET[i], NET[j]) for i, j in zip(*np.where(hub16))]
    sub_ctrl = [(a, b) for a, b in sub_hub if a in ('L-Cont', 'R-Cont') and b in ('L-Cont', 'R-Cont')]
    print(f'  SUBNET hub={int(hub16.sum())} edges: {sub_hub}')
    print(f'    within-control: {sub_ctrl}')

    # parcel two-step over FULL 116x116, then zoom into the control block
    strP, posP, hubP, mdP, crP = two_step(nfP, dcP, nfP[ok], sc[ok])
    ctrl_hub = hubP[CTRL, CTRL]
    print(f'  PARCEL (full-matrix FDR): stronger-in-NF={int(strP.sum())} pos-score={int(posP.sum())} '
          f'HUB={int(hubP.sum())} (total 116x116)')
    print(f'    -> within control block [38:51]: HUB={int(ctrl_hub.sum())} edges')
    cb = mdP[CTRL, CTRL]; cc = crP[CTRL, CTRL]
    for i, j in zip(*np.where(ctrl_hub)):
        print(f'      {CTRL_LABELS[i]} -> {CTRL_LABELS[j]}  (NF-DC={cb[i,j]:+.3f}, rho={cc[i,j]:+.3f})')

    if not make_fig:
        return

    # ---- figure: subnet NF-DC | subnet score-corr | subnet hub | parcel control-block zoom
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.6), layout='constrained')
    for a_, Mt, sig, ttl in [(ax[0], nfdc16, (nfdc16 != 0), 'NF - DC (stronger in NF, FDR)'),
                             (ax[1], scorr16, (scorr16 != 0), 'edge-score corr (FDR)')]:
        vlim = np.nanpercentile(np.abs(Mt[Mt != 0]), 99) if (Mt != 0).any() else 1
        im = a_.imshow(Mt, cmap='coolwarm', vmin=-vlim, vmax=vlim)
        a_.set_title(ttl, fontsize=10); a_.set_xticks(range(16)); a_.set_yticks(range(16))
        a_.set_xticklabels(NET, rotation=90, fontsize=6); a_.set_yticklabels(NET, fontsize=6)
        fig.colorbar(im, ax=a_, fraction=.046, pad=.04).ax.tick_params(labelsize=6)
    ax[2].imshow(hub16.astype(float), cmap='Greens', vmin=0, vmax=1)
    ax[2].set_title(f'subnetwork control hub: {int(hub16.sum())} edges', fontsize=10)
    ax[2].set_xticks(range(16)); ax[2].set_yticks(range(16))
    ax[2].set_xticklabels(NET, rotation=90, fontsize=6); ax[2].set_yticklabels(NET, fontsize=6)

    nc = ctrl_hub.shape[0]
    vlim = np.nanpercentile(np.abs(cb[ctrl_hub]), 99) if ctrl_hub.any() else (np.abs(cb).max() or 1)
    im = ax[3].imshow(np.where(ctrl_hub, cb, 0), cmap='coolwarm', vmin=-vlim, vmax=vlim)
    ax[3].set_title(f'parcel control-network hub ({int(ctrl_hub.sum())} edges)', fontsize=10)
    ax[3].set_xticks(range(nc)); ax[3].set_yticks(range(nc))
    ax[3].set_xticklabels(CTRL_LABELS, rotation=90, fontsize=5); ax[3].set_yticklabels(CTRL_LABELS, fontsize=5)
    fig.colorbar(im, ax=ax[3], fraction=.046, pad=.04).ax.tick_params(labelsize=6)

    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_glm_control_hub_9subj.{ext}'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  saved', os.path.join(OUT, 'fig_glm_control_hub_9subj.png'))


def main():
    scores = per_graph_scores()
    run_variant('hrf', scores, make_fig=True)      # main figure on HRF residuals
    run_variant('binary', scores, make_fig=False)  # report binary numbers too


if __name__ == '__main__':
    main()
