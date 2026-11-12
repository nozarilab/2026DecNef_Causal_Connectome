"""Two-step control-hub analysis (Figure 3) at a given alpha-bar.

per_graph_scores(): per-graph neurofeedback score (logit of the trial scores of the two
runs, z-scored within study, averaged). The two-step analysis keeps edges that are both
stronger in NF than baseline (rank-sum, FDR) and positively correlated with score
(Spearman, FDR), at the subnetwork level and within the control network at the parcel level.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr, ranksums
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import decnef_parcels as dp

SWEEP = 'Results/glm_revision/alpha_sweep'
OUT = 'Results/glm_revision/figures'
ABAR = [0.0025, 0.01, 0.025, 0.05]
SES_OFFSET = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
SUBJECT_IDS = {
    1: ['03', '24', '21', '15', '23', '05', '17', '13', '18'],
    2: ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    3: ['01', '05', '08', '10', '12', '13', '14', '15', '16'],
    4: ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    5: ['01', '02', '03', '06', '08', '09', '11', '12', '15'],
}
SCORES = 'Results/glm_revision/scores'
N_SUB, N_SES, N_PAIR, N_TRIAL = 9, 3, 7, 15
CTRL = slice(38, 51)                  # control parcels in rearranged order
SUBNET_CTRL = [6, 7]                  # L-Cont, R-Cont
VIS = ['L-Sub', 'R-Sub', 'L-Vis', 'R-Vis', 'L-Lim', 'R-Lim', 'L-Cont', 'R-Cont',
       'L-DMN', 'R-DMN', 'L-Vent-Att', 'R-Vent-Att', 'L-Dors-Att', 'R-Dors-Att',
       'L-Som-Mot', 'R-Som-Mot']

PL = dp.get_parcel_labels()
NPIDX = dp.get_network_parcel_idx(PL)
REARR_LABELS = [PL[i] for net in dp.NET_LABELS for i in NPIDX[net]]
CTRL_LABELS = [l.replace('LH_Cont_', 'L-').replace('RH_Cont_', 'R-') for l in REARR_LABELS[38:51]]


def per_graph_scores():
    out = {}
    for s in range(1, 6):
        inv = np.full((N_SUB, N_SES, N_PAIR, 2 * N_TRIAL), np.nan)
        for si, sub in enumerate(SUBJECT_IDS[s]):
            for ses in range(N_SES):
                f = os.path.join(SCORES, f'Study{s}', f'sub-{sub}', 'pattern',
                                 f'in_score_d{ses + SES_OFFSET[s]}.csv')
                if not os.path.exists(f):
                    continue
                vals = pd.read_csv(f).values[:120].astype(float).ravel()
                for r in range(N_PAIR):
                    seg = np.concatenate([vals[r * N_TRIAL:(r + 1) * N_TRIAL],
                                          vals[(r + 1) * N_TRIAL:(r + 2) * N_TRIAL]])
                    iv = logit(seg); iv[np.isinf(iv)] = np.nan
                    inv[si, ses, r, :len(iv)] = iv
        mu, sd = np.nanmean(inv), np.nanstd(inv)
        z = (inv - mu) / sd
        for si in range(N_SUB):
            for ses in range(N_SES):
                for r in range(N_PAIR):
                    out[(s, si, ses, r)] = np.nanmean(z[si, ses, r])
    return out


def two_step(nf, dc, score_ok_nf, score):
    """Return (stronger_in_NF mask, pos_corr mask, hub) for an (N,P,P) stack."""
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


def main():
    scores = per_graph_scores()
    fig, axes = plt.subplots(len(ABAR), 3, figsize=(15, 4.2 * len(ABAR)), layout='constrained')
    for k, a in enumerate(ABAR):
        d = np.load(os.path.join(SWEEP, f'subnet_abar_{a}.npz'))
        nf16, dc16 = d['nf_weighted'], d['dc_weighted']
        nfP, dcP = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
        idx = d['nf_idx']
        sc = np.array([scores.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in idx])
        ok = np.isfinite(sc)

        # subnetwork two-step
        _, _, hub16, nfdc16, scorr16 = two_step(nf16, dc16, nf16[ok], sc[ok])
        # parcel two-step
        _, _, hubP, _, _ = two_step(nfP, dcP, nfP[ok], sc[ok])
        ctrl_hub = hubP[CTRL, CTRL]
        print(f'abar={a}: subnetwork hub={int(hub16.sum())} {[ (VIS[i],VIS[j]) for i,j in zip(*np.where(hub16))]}; '
              f'parcel control-hub edges={int(ctrl_hub.sum())}')

        vlim = np.nanpercentile(np.abs(scorr16[scorr16 != 0]), 99) if (scorr16 != 0).any() else 0.1
        # col 0: subnetwork significant correlations with score
        ax = axes[k, 0]
        im = ax.imshow(scorr16, cmap='coolwarm', vmin=-vlim, vmax=vlim)
        if k == 0: ax.set_title('Significant correlations with score', fontsize=10)
        ax.set_ylabel(f'ᾱ = {a}', fontsize=11)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(VIS, rotation=90, fontsize=5); ax.set_yticklabels(VIS, fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
        # col 1: subnetwork significant NF-Baseline, hub circled
        ax = axes[k, 1]
        vlim2 = np.nanpercentile(np.abs(nfdc16[nfdc16 != 0]), 99) if (nfdc16 != 0).any() else 1
        im = ax.imshow(nfdc16, cmap='coolwarm', vmin=-vlim2, vmax=vlim2)
        yi, xi = np.where(hub16)
        ax.scatter(xi, yi, s=80, facecolors='none', edgecolors='lime', linewidths=2)
        if k == 0: ax.set_title('Significant NF − Baseline (hub = lime)', fontsize=10)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(VIS, rotation=90, fontsize=5); ax.set_yticklabels(VIS, fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
        # col 2: parcel-level control hub (Fig 3c), stronger-in-NF & help-with-score
        ax = axes[k, 2]
        ax.imshow(ctrl_hub, cmap='Greens', vmin=0, vmax=1)
        if k == 0: ax.set_title('Parcel-level control hub', fontsize=10)
        ax.set_xticks(range(13)); ax.set_yticks(range(13))
        ax.set_xticklabels(CTRL_LABELS, rotation=90, fontsize=5); ax.set_yticklabels(CTRL_LABELS, fontsize=5)

    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig8_fig3_control_hub_alpha.{ext}'), bbox_inches='tight', dpi=130)
    plt.close(fig)
    print('saved', os.path.join(OUT, 'fig8_fig3_control_hub_alpha.png'))


if __name__ == '__main__':
    main()
