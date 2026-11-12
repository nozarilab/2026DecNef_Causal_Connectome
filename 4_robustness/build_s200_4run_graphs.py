"""Weighted subnetwork (16x16) and parcel (216x216) graphs from the four-run 216-parcel CaLLTiF outputs.

Uses the published thresholds (contemporaneous alpha = 0.0025, lagged alpha-bar = 0.05)
and both lags; no CaLLTiF recomputation.
Input:  Results/glm_revision/s200_results_4run/
Output: Results/glm_revision/s200_4run_graphs/s200_4run_weighted_abar0.05.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import re
import glob
import numpy as np
import functions as funcs
import decnef_parcels216 as dp216

ALPHA0 = 1e-2 / 4                      # contemporaneous threshold (fixed, published)
ABAR = 0.05                            # relaxed-lagged threshold (single published value)
RATIO = 20
SRC = 'Results/glm_revision/s200_results_4run'
OUT = 'Results/glm_revision/s200_4run_graphs'

PL = dp216.get_parcel_labels()
NET = dp216.NET_LABELS
NPIDX = dp216.get_network_parcel_idx(PL)
CUM = np.cumsum([len(NPIDX[k]) for k in NET])
STARTS = np.concatenate([[0], CUM[:-1]])
CONTROL = (int(STARTS[NET.index('LH_Con')]), int(CUM[NET.index('RH_Con')]))   # rearranged control block


def to_subnet(parcel_w):
    S = np.zeros((16, 16))
    for u in range(16):
        for v in range(16):
            S[u, v] = parcel_w[STARTS[u]:CUM[u], STARTS[v]:CUM[v]].mean()
    return S


def process(cond):
    weighted_sub, parcel_w_all, idx = [], [], []
    files = sorted(glob.glob(os.path.join(SRC, f'{cond}_study*_subj*_ses*_pair*.npz')))
    alpha_summ = ABAR / RATIO
    for f in files:
        m = re.search(r'study(\d+)_subj(\d+)_ses(\d+)_pair(\d+)', os.path.basename(f))
        study, subj, ses, pair = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        g = np.load(f); p, val = g['p_matrix'], g['val_matrix']
        # 4-run: lag-1 p is fully estimable (0 NaN verified) -> BOTH lags drive summarize_v2.
        # nan_to_num is a harmless no-op safety net.
        p = np.nan_to_num(p, nan=1.0)
        sg = funcs.convert_to_string_graph((p <= ALPHA0).astype(int))
        _, _, weighted = funcs.summarize_across_all_lags_v2_with_pvalues_more_directed_v2(sg, p, val, alpha_summ)
        pw = np.array(funcs.rearange_matrix_v2(weighted, NPIDX), dtype=np.float32)
        parcel_w_all.append(pw)
        weighted_sub.append(to_subnet(pw))
        idx.append((study, subj, ses, pair))
    return np.array(weighted_sub), np.array(parcel_w_all, dtype=np.float32), np.array(idx)


def main():
    os.makedirs(OUT, exist_ok=True)
    save = {'control_block': np.array(CONTROL), 'net_labels': np.array(NET)}
    for cond in ('nf', 'dc'):
        ws, pw, idx = process(cond)
        save[f'{cond}_weighted'] = ws
        save[f'{cond}_parcel_w'] = pw
        save[f'{cond}_idx'] = idx
        print(f'{cond}: {len(ws)} graphs | subnet {ws.shape} | parcel {pw.shape}', flush=True)
    fp = os.path.join(OUT, 's200_4run_weighted_abar0.05.npz')
    np.savez(fp, **save)
    print('control block (rearranged 216):', CONTROL, '| saved', fp)


if __name__ == '__main__':
    main()
