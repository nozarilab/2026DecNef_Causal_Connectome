"""Parcellate Study 5, sub-15 (NF sessions 2-4, runs 1-8) at Schaefer-200 + Tian S1 and add
it to study5_s200_nf.npz, using the functions of parcelate_s200.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import numpy as np
import nibabel as nib
import parcelate_s200 as P

DERIV = os.path.join(config.ROBUSTNESS_DIR, 'bids_sub15', 'derivatives')
OUT = os.path.join(config.ROBUSTNESS_DIR, 's200_data', 'study5_s200_nf.npz')


def main():
    mask = np.asanyarray(nib.load(P.ATLAS).dataobj)
    d = np.load(OUT, allow_pickle=True)
    par = d['fmri_data_all_preprocessed_parcelated']        # (9,3,8) object
    nr = d['num_runs_all'].copy()
    subs = list(d['subs']); sessions = list(d['sessions'])  # [2,3,4]
    si = subs.index('15')
    print('sub-15 index', si, '| sessions', sessions, '| before: sub-15 runs', int(nr[si].sum()))
    for sj, ses in enumerate(sessions):
        got = 0
        for ri in range(8):
            r = P.load_run(DERIV, '15', int(ses), 'nfb', ri + 1)
            if r is None:
                print(f'  ses-{ses} run-{ri+1}: MISSING'); continue
            par[si, sj, ri] = P.process_run(r[0], r[1], mask).astype(np.float32)
            got += 1
        nr[si, sj] = got
        print(f'  sub-15 ses-{ses}: {got}/8  (example shape {par[si,sj,0].shape})')
    np.savez(OUT, fmri_data_all_preprocessed_parcelated=par, num_runs_all=nr,
             subs=d['subs'], sessions=d['sessions'], task=d['task'])
    print('MERGED. study5 NF per-subj run totals:', nr.sum(1).astype(int).tolist(),
          '| grand total', int(nr.sum()))


if __name__ == '__main__':
    main()
