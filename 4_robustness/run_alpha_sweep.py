"""Subnetwork and parcel graphs at several lagged-edge thresholds alpha-bar
(Supplementary Figs. 15-25).

Reprocesses the saved CaLLTiF graphs (NF and DC) for alpha-bar in {0.0025, 0.01, 0.025, 0.05},
keeping the contemporaneous threshold fixed at 0.0025; no CaLLTiF recomputation.
alpha-bar = 0.05 reproduces the published analysis.
Output: Results/glm_revision/alpha_sweep/subnet_abar_{alpha-bar}.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import functions as funcs
import decnef_parcels as dp

ALPHA0 = 1e-2 / 4                       # contemporaneous threshold (fixed)
RATIO = 20                             # summarize_v2 hardcoded alpha_level_ratio
ABAR = [0.0025, 0.01, 0.025, 0.05]     # relaxed lagged thresholds to sweep
STUDIES = (1, 2, 3, 4, 5)
N_SUBJECTS = 9                          # first 9 subjects per study
NF_DIR = 'Results/glm_revision/raw'
DC_DIR = 'Results'
OUT = 'Results/glm_revision/alpha_sweep'

PARCEL_LABELS = dp.get_parcel_labels()
NET_LABELS = dp.NET_LABELS
NETWORK_DIC = '{' + ','.join("'%s':[]" % k for k in NET_LABELS) + '}'
NPIDX = dp.get_network_parcel_idx(PARCEL_LABELS)   # dict in NET_LABELS order (control at 38:51)


def load_link(cond, study):
    if cond == 'nf':
        f = os.path.join(NF_DIR, f'calltif_decnef_study{study}_nf_two_runs_one_graph.npz')
    else:
        f = os.path.join(DC_DIR, f'calltif_decnef_study{study}_dec_two_runs_one_graph.npz')
    d = np.load(f, allow_pickle=True)
    return d[[k for k in d.files if 'link_matrix' in k][0]]


def subnet(matrix):
    return funcs.network_map_v2_without_decoder(matrix, NET_LABELS, NETWORK_DIC, PARCEL_LABELS)


def process(abar):
    alpha_summ = abar / RATIO
    out = {'nf': {'binary': [], 'weighted': [], 'parcel_w': [], 'idx': []},
           'dc': {'binary': [], 'weighted': [], 'parcel_w': [], 'idx': []}}
    for cond in ('nf', 'dc'):
        for study in STUDIES:
            lm = load_link(cond, study)
            nsub = min(len(lm), N_SUBJECTS)        # first 9 subjects, as in the main analysis
            for subj in range(nsub):
                for ses in range(len(lm[subj])):
                    runs = lm[subj][ses]
                    for r in range(len(runs)):
                        g = runs[r]
                        if not isinstance(g, dict):
                            continue
                        p = np.array(g['p_matrix']); val = np.array(g['val_matrix'])
                        sg = funcs.convert_to_string_graph((p <= ALPHA0).astype(int))
                        binary, _, weighted = funcs.summarize_across_all_lags_v2_with_pvalues_more_directed_v2(
                            sg, p, val, alpha_summ)
                        out[cond]['binary'].append(subnet(binary))
                        out[cond]['weighted'].append(subnet(weighted))
                        # parcel-level (116x116) rearranged weighted, for Fig 3 parcel analysis
                        out[cond]['parcel_w'].append(
                            funcs.rearange_matrix_v2(weighted, NPIDX).astype(np.float32))
                        out[cond]['idx'].append((study, subj, ses, r))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for abar in ABAR:
        res = process(abar)
        save = {}
        for cond in ('nf', 'dc'):
            save[f'{cond}_binary'] = np.array(res[cond]['binary'])
            save[f'{cond}_weighted'] = np.array(res[cond]['weighted'])
            save[f'{cond}_parcel_w'] = np.array(res[cond]['parcel_w'], dtype=np.float32)
            save[f'{cond}_idx'] = np.array(res[cond]['idx'])
        fp = os.path.join(OUT, f'subnet_abar_{abar}.npz')
        np.savez(fp, **save)
        nf_b = save['nf_binary']; dc_b = save['dc_binary']
        print(f'abar={abar}: NF graphs={len(nf_b)} (mean density {nf_b.mean():.3f}), '
              f'DC graphs={len(dc_b)} (mean density {dc_b.mean():.3f}) -> {fp}', flush=True)


if __name__ == '__main__':
    main()
