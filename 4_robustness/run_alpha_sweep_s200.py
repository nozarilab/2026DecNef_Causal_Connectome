"""Weighted subnetwork and parcel graphs from the two-run 216-parcel CaLLTiF outputs (alpha-bar = 0.05).

Input:  Results/glm_revision/s200_results/
Output: Results/glm_revision/alpha_sweep_s200/subnet_abar_0.05_s200.npz
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

ALPHA0 = 1e-2 / 4
ABAR = 0.05
RATIO = 20
SRC = 'Results/glm_revision/s200_results'
OUT = 'Results/glm_revision/alpha_sweep_s200'

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
        # At 216 parcels the LAGGED (lag-1) partial-corr p-values are undefined (NaN) due to
        # high-dim under-determination (conditioning dimension vs 152 samples). Treat undefined
        # p as not-significant (1.0) so the valid CONTEMPORANEOUS (lag-0) graph drives summarize_v2.
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
    fp = os.path.join(OUT, f'subnet_abar_{ABAR}_s200.npz')
    np.savez(fp, **save)
    print('control block (rearranged 216):', CONTROL, '| saved', fp)


if __name__ == '__main__':
    main()
