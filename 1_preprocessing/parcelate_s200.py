"""Schaefer-200 + Tian S1 (216-parcel) parcellation, for the robustness analyses
(Supplementary Figs. 12-14).

Same pipeline as the 116-parcel loaders (load_preprocessed_fmri_and_parcelate_study*.py):
9P confound removal, linear detrending, parcellation. Only the atlas and the number of
parcels change. A vectorized detrend is used and checked against the original in smoke mode.

Usage:
  python 1_preprocessing/parcelate_s200.py smoke               # 1 subject / 1 session check
  python 1_preprocessing/parcelate_s200.py run <study> <cond>  # cond in {nf, dc}; study in 1..5
Output: Results/glm_revision/s200_data/study{N}_s200_{nf,dc}.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '16')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '16')
import sys
import time
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.linear_model import LinearRegression

ATLAS = os.path.join(config.EXTERNAL_DIR, 'Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii')
OUTDIR = os.path.join(config.ROBUSTNESS_DIR, 's200_data')
NUM_PARCELS = 216
CONF9 = ['global_signal', 'csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

# per-study config (base, subjects[first 9], nf task/sessions/runs, dc task/sessions/runs)
STUDIES = {
    1: dict(base=config.FMRIPREP_DIRS[1],
            subs=['03', '24', '21', '15', '23', '05', '17', '13', '18'],
            nf=('nfb', [1, 2, 3], 8), dc=('decoder', [0], 6)),
    2: dict(base=config.FMRIPREP_DIRS[2],
            subs=['01', '02', '03', '04', '05', '06', '07', '08', '09'],
            nf=('nfb', [1, 2, 3], 8), dc=('decoder', [0], 6)),
    3: dict(base=config.FMRIPREP_DIRS[3],
            subs=['01', '05', '08', '10', '12', '13', '14', '15', '16'],
            nf=('nfb', [1, 2, 3], 8), dc=('decoder', [0], 6)),
    4: dict(base=config.FMRIPREP_DIRS[4],
            subs=['01', '02', '03', '04', '05', '06', '07', '08', '09'],
            nf=('decoder', [2, 3, 4], 8), dc=('decoder', [0], 6)),
    5: dict(base=config.FMRIPREP_DIRS[5],
            subs=['01', '02', '03', '06', '08', '09', '11', '12', '15'],
            nf=('nfb', [2, 3, 4], 8), dc=('decoder', [0], 6)),
}


# ---- same as functions.py ----
def apply_mask(fmri_data, parcellation_mask):
    mask_idx = np.where(parcellation_mask > 0)
    data = fmri_data[np.array(mask_idx[1]), :]
    data_updated = data[np.std(data, axis=1) > 0, :]
    parcellation_mask_updated = parcellation_mask[parcellation_mask > 0]
    parcellation_mask_updated = parcellation_mask_updated[np.std(data, axis=1) > 0]
    return data_updated, parcellation_mask_updated


def detrend_sklearn(data_sample, parcellation_mask_updated):
    num_voxels = data_sample.shape[0]
    num_time_points = data_sample.shape[1]
    t = np.reshape(np.linspace(0, num_time_points, num_time_points), (num_time_points, 1))
    data_sample_detrend = np.zeros((num_voxels, num_time_points))
    for v in range(0, num_voxels):
        sig = np.reshape(data_sample[v, :], (num_time_points, 1))
        reg = LinearRegression().fit(t, sig)
        a = reg.coef_; b = reg.intercept_
        y = a * t + b
        sig_linear_trend_removed = sig - y
        data_sample_detrend[v, :] = np.squeeze(sig_linear_trend_removed) / np.std(np.squeeze(sig_linear_trend_removed))
    return data_sample_detrend


def parcelate(fmri_data, parcellation_mask, num_parcels):
    num_samples = fmri_data.shape[1]
    parcelated_data = np.zeros((num_samples, num_parcels))
    for p in range(0, num_parcels):
        voxel_idx = np.where(parcellation_mask == (p + 1))[0]
        parcelated_data[:, p] = np.mean(fmri_data[voxel_idx, :], axis=0)
    return parcelated_data


def regressout_confounders(fmri_data, confounders_9):
    y = fmri_data.T
    A = confounders_9.T
    x = np.linalg.pinv(A) @ y
    y_err = y - A @ x
    return y_err.T


# ---- vectorized detrend, asserted identical to detrend_sklearn ----
def detrend_vec(data_sample, parcellation_mask_updated=None):
    V, T = data_sample.shape
    t = np.linspace(0, T, T)
    X = np.column_stack([t, np.ones(T)])              # (T,2), same OLS as LinearRegression
    beta, *_ = np.linalg.lstsq(X, data_sample.T, rcond=None)   # (2,V)
    resid = data_sample - (X @ beta).T                # (V,T)
    return resid / np.std(resid, axis=1, keepdims=True)


def load_run(base, sub, ses, task, run):
    d = f'{base}/sub-{sub}/ses-{ses}/func'
    stem = f'{d}/sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsLR_den-91k_bold.dtseries.nii'
    conf = f'{d}/sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv'
    if not (os.path.exists(stem) and os.path.exists(conf)):
        return None
    try:
        fmri = np.asanyarray(nib.load(stem).dataobj).T             # (91282, T)
        c = pd.read_csv(conf, sep='\t')
        confounders_9 = np.array([c[k].values for k in CONF9])     # (9, T)
    except Exception as e:
        print(f'  SKIP damaged {os.path.basename(stem)}: {e}', flush=True)
        return None
    return fmri, confounders_9


def process_run(fmri, confounders_9, mask):
    cr = regressout_confounders(fmri, confounders_9)
    masked, mask_upd = apply_mask(cr, mask)
    det = detrend_vec(masked, mask_upd)
    return parcelate(det, mask_upd, NUM_PARCELS)                   # (T, 216)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    mask = np.asanyarray(nib.load(ATLAS).dataobj)
    print('atlas mask shape', mask.shape, 'labels', int(mask.max()))

    if mode == 'smoke':
        cfg = STUDIES[1]; task, sess, _ = cfg['nf']
        r = load_run(cfg['base'], cfg['subs'][0], sess[0], task, 1)
        assert r is not None, 'smoke run not found'
        fmri, c9 = r
        cr = regressout_confounders(fmri, c9)
        masked, mu = apply_mask(cr, mask)
        sub = masked[:400]                                          # check 400 voxels both ways
        a = detrend_sklearn(sub, mu); b = detrend_vec(sub, mu)
        print('detrend vec==sklearn:', np.allclose(a, b, atol=1e-6), 'max|diff|', np.abs(a - b).max())
        p = parcelate(detrend_vec(masked, mu), mu, NUM_PARCELS)
        print('parcelated shape', p.shape, 'finite', np.isfinite(p).all())
        return

    study = int(sys.argv[2]); cond = sys.argv[3]
    cfg = STUDIES[study]; task, sess, nruns = cfg[cond]
    os.makedirs(OUTDIR, exist_ok=True)
    nsub, nses = len(cfg['subs']), len(sess)
    par = [[[0] * nruns for _ in range(nses)] for _ in range(nsub)]
    nrun_all = np.zeros((nsub, nses))
    t0 = time.time()
    for si, sub in enumerate(cfg['subs']):
        for sj, ses in enumerate(sess):
            got = 0
            for ri in range(nruns):
                r = load_run(cfg['base'], sub, ses, task, ri + 1)
                if r is None:
                    continue
                par[si][sj][ri] = process_run(r[0], r[1], mask).astype(np.float32)
                got += 1
            nrun_all[si, sj] = got
            print(f'study{study} {cond} sub-{sub} ses-{ses}: {got}/{nruns} runs  '
                  f'[{time.time()-t0:.0f}s]', flush=True)
    out = f'{OUTDIR}/study{study}_s200_{cond}.npz'
    np.savez(out, fmri_data_all_preprocessed_parcelated=np.asanyarray(par, dtype=object),
             num_runs_all=nrun_all, subs=np.array(cfg['subs']), sessions=np.array(sess), task=task)
    print('saved', out, f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
