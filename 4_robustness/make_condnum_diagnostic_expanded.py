"""Condition numbers of the conditioning regressions inside CaLLTiF (data for Supplementary Fig. 26).

Records cond(Z) for every partial-correlation regression performed while running the
standard two-run CaLLTiF graph (last 152 samples per run, tau_min = 0, tau_max = 1,
pc_alpha = 1), for both NF and DC sessions. By default three subjects per study are used.

Output: Results/glm_revision/condnum_diagnostic_expanded.npz (cond, study, kind, dim_z, subj, ses, pair)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
sys.path.insert(0, config.TIGRAMITE_DIR)
from tigramite.independence_tests import ParCorr        # noqa: E402
import tigramite.data_processing as pp                    # noqa: E402
import calltif                                            # noqa: E402

RESULTS = config.RESULTS_DIR
NSAMP = 152
NODES = ['X%d' % i for i in range(1, 117)]
ALPHA = {1: 1e-2 / 12, 2: 1e-2 / 4, 3: 1e-2 / 4, 4: 1e-2 / 4, 5: 1e-2 / 4}

# ---- expansion scope (override via env if desired) --------------------------
N_SUBJ = int(os.environ.get('CN_SUBJ', 3))          # first N subjects per study
NF_SES = [int(x) for x in os.environ.get('CN_NF_SES', '0,1,2').split(',')]
NF_PAIRS = [int(x) for x in os.environ.get('CN_NF_PAIRS', '0,2,4').split(',')]
DC_PAIRS = [int(x) for x in os.environ.get('CN_DC_PAIRS', '0,2,4').split(',')]

NF_FILE = 'study%d_preprocessed_confounders_removed_selected_subjects.npz'
DC_FILE = 'study%d_preprocessed_confounders_removed_selected_subjects_decoder_session0.npz'
KEY = 'fmri_data_all_preprocessed_parcelated'

COND = []            # (kind, study, subj, ses, pair, dim_z, cond_number)
_CTX = {}
_orig = ParCorr._get_single_residuals


def _patched(self, array, target_var, standardize=True, return_means=False):
    if standardize and array.shape[0] > 2:
        a = array.astype(float).copy()
        a -= a.mean(axis=1, keepdims=True)
        std = a.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        a /= std
        z = a[2:, :].T                       # (T, dim_z) standardized conditioning matrix
        try:
            c = float(np.linalg.cond(z))
        except Exception:
            c = np.nan
        COND.append((_CTX['kind'], _CTX['study'], _CTX['subj'], _CTX['ses'],
                     _CTX['pair'], z.shape[1], c))
    return _orig(self, array, target_var, standardize, return_means)


ParCorr._get_single_residuals = _patched
_cache = {}


def load(kind, study):
    key = (kind, study)
    if key not in _cache:
        f = (NF_FILE if kind == 'nf' else DC_FILE) % study
        d = np.load(os.path.join(RESULTS, f), allow_pickle=True)
        _cache[key] = d[KEY]
    return _cache[key]


def run_one(kind, study, subj, ses, r):
    ts = load(kind, study)
    a0 = np.array(ts[subj][ses][r])[-NSAMP:, :].astype(np.float64)
    a1 = np.array(ts[subj][ses][r + 1])[-NSAMP:, :].astype(np.float64)
    df = pp.DataFrame({0: a0, 1: a1}, var_names=NODES, analysis_mode='multiple')
    _CTX.update(kind=(0 if kind == 'nf' else 1), study=study, subj=subj, ses=ses, pair=r)
    calltif.run_calltif(df, ParCorr(), tau_min=0, tau_max=1, pc_alpha=1, alpha_level=ALPHA[study])


def main():
    plan = []
    for s in (1, 2, 3, 4, 5):
        nsub_nf = len(load('nf', s)); nsub_dc = len(load('dc', s))
        for sub in range(min(N_SUBJ, nsub_nf, nsub_dc)):
            for ses in NF_SES:
                for r in NF_PAIRS:
                    plan.append(('nf', s, sub, ses, r))
            for r in DC_PAIRS:                       # DC has a single decoder session (ses 0)
                plan.append(('dc', s, sub, 0, r))
    n_nf = sum(1 for p in plan if p[0] == 'nf')
    n_dc = sum(1 for p in plan if p[0] == 'dc')
    print('planned %d CaLLTiF runs (NF %d, DC %d)' % (len(plan), n_nf, n_dc), flush=True)

    t_all = time.time()
    for i, (kind, s, sub, ses, r) in enumerate(plan):
        t0 = time.time()
        try:
            run_one(kind, s, sub, ses, r)
        except Exception as e:
            print(f'  [skip] {kind} study{s} sub{sub} ses{ses} pair{r}: {e}', flush=True)
            continue
        if i % 10 == 0 or i == len(plan) - 1:
            arr = np.array([c for *_, c in COND], dtype=float)
            arr = arr[np.isfinite(arr)]
            print(f'[{i+1}/{len(plan)}] {kind} study{s} sub{sub} ses{ses} pair{r} '
                  f'({time.time()-t0:.0f}s) | records={len(COND)} '
                  f'| cond median={np.median(arr):.0f} max={arr.max():.0f}', flush=True)

    kind = np.array([k for k, *_ in COND], dtype=np.int8)
    study = np.array([s for _, s, *_ in COND], dtype=np.int8)
    subj = np.array([su for _, _, su, *_ in COND], dtype=np.int8)
    ses = np.array([se for _, _, _, se, *_ in COND], dtype=np.int8)
    pair = np.array([p for _, _, _, _, p, _, _ in COND], dtype=np.int8)
    dimz = np.array([dz for *_, dz, _ in COND], dtype=np.int16)
    cond = np.array([c for *_, c in COND], dtype=np.float32)
    out = os.path.join(config.ROBUSTNESS_DIR, 'condnum_diagnostic_expanded.npz')
    np.savez(out, kind=kind, study=study, subj=subj, ses=ses, pair=pair, dim_z=dimz, cond=cond)

    fin = cond[np.isfinite(cond)]
    print(f'\n=== EXPANDED CONDITION-NUMBER DIAGNOSTIC ({time.time()-t_all:.0f}s) ===')
    print(f'total regressions: {len(fin)}  (NF {int((kind==0).sum())}, DC {int((kind==1).sum())})')
    for lab, m in [('NF', kind == 0), ('DC', kind == 1)]:
        c = cond[m]; c = c[np.isfinite(c)]
        print(f'  {lab}: median={np.median(c):.0f} p95={np.percentile(c,95):.0f} '
              f'max={c.max():.0f}  (%>1000={100*(c>1000).mean():.3f})')
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
