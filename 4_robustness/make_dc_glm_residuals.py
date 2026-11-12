"""GLM task-regressed residuals of the decoder-construction (DC) sessions.

DC counterpart of make_glm_residuals.py: loads the 116-parcel DC time series, uses one
reconstructed event template per study (build_dc_events.py), 1 session x 6 runs.
Output folder: Results/glm_revision/glm_data_9subj/
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import csv
import numpy as np
import glm_task_regression as glm

PREP = 'Results/DecNef_five_studies_preprocessed'
EV_DIR = 'DecNef_events_DC'
OUTDIR = os.environ.get('DECNEF_DC_OUT', os.path.join(config.ROBUSTNESS_DIR, 'glm_data_9subj'))
N_SUBJECTS = 9   # ALL 9 subjects (matches the main analyses) — see *_9subj naming
N_DC_RUNS = 6

# per-study DC task-event regressors (decoder design from source papers; see build_dc_events.py)
DC_EVENT_TYPES = {
    1: ['face', 'rating', 'report'],
    2: ['grating'],
    3: ['grating'],
    4: ['stimulus', 'response'],
    5: ['image'],
}


def load_dc_npz(study):
    f = os.path.join(PREP, f'study{study}_preprocessed_confounders_removed_selected_subjects_decoder_session0.npz')
    d = np.load(f, allow_pickle=True)
    return d['fmri_data_all_preprocessed_parcelated']


def dc_events(study):
    return glm.load_events(os.path.join(EV_DIR, f'Study{study}', f'study{study}_task-decoder_events.csv'))


def build_for_study(study, hrf, diag):
    data = load_dc_npz(study)
    sub_ids = glm.SUBJECT_IDS[study][:N_SUBJECTS]
    ets = DC_EVENT_TYPES[study]
    events = dc_events(study)
    resid_all = [[[0 for _ in range(N_DC_RUNS)] for _ in range(1)] for _ in range(N_SUBJECTS)]
    for si in range(N_SUBJECTS):
        for r in range(N_DC_RUNS):
            ts = np.asarray(data[si][0][r], dtype=float)
            if ts.ndim != 2 or ts.shape[1] != 116:
                continue
            X = glm.build_design_matrix(events, ts.shape[0], hrf=hrf, event_types=ets)
            resid, beta = glm.regress_out(ts, X)
            vr = 1.0 - resid.var(0) / np.where(ts.var(0) > 0, ts.var(0), 1.0)
            resid_all[si][0][r] = resid
            diag.append({'study': study, 'subject': sub_ids[si], 'run': r + 1,
                         'variant': 'hrf' if hrf else 'binary', 'n_trs': ts.shape[0],
                         'n_reg': X.shape[1], 'cond_number': float(np.linalg.cond(X)),
                         'mean_var_removed': float(vr.mean()), 'max_var_removed': float(vr.max())})
    return resid_all, sub_ids


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    diag = []
    for hrf in (False, True):
        v = 'hrf' if hrf else 'binary'
        for study in (1, 2, 3, 4, 5):
            resid_all, sub_ids = build_for_study(study, hrf, diag)
            out = os.path.join(OUTDIR, f'study{study}_dc_residuals_{v}.npz')
            np.savez(out, fmri_data_all_preprocessed_parcelated_resid=np.asanyarray(resid_all, dtype=object),
                     subject_ids=np.array(sub_ids))
            # version-portable FLAT npz (float arrays keyed run__{subj}__{ses}__{r})
            flat = {}
            for si in range(N_SUBJECTS):
                for r in range(N_DC_RUNS):
                    e = resid_all[si][0][r]
                    if isinstance(e, np.ndarray) and e.ndim == 2:
                        flat[f'run__{si}__0__{r}'] = e.astype(np.float32)
            np.savez(os.path.join(OUTDIR, f'study{study}_dc_residuals_{v}_9subj_flat.npz'), **flat)
            print(f'[{v}] study{study}: saved {out} (+ flat)')
    # raw DC time series in the same flat format (variant 'raw') -> identical-code raw vs residual
    for study in (1, 2, 3, 4, 5):
        data = load_dc_npz(study)
        flat = {}
        for si in range(N_SUBJECTS):
            for r in range(N_DC_RUNS):
                e = np.asarray(data[si][0][r], dtype=float)
                if e.ndim == 2 and e.shape[1] == 116:
                    flat[f'run__{si}__0__{r}'] = e.astype(np.float32)
        np.savez(os.path.join(OUTDIR, f'study{study}_dc_residuals_raw_9subj_flat.npz'), **flat)
    print('[raw] saved raw DC flat npz for all studies')

    dp = os.path.join(OUTDIR, 'dc_glm_diagnostics_9subj.csv')
    with open(dp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(diag[0].keys())); w.writeheader(); w.writerows(diag)
    print(f'\nWrote {dp} ({len(diag)} runs)')
    for v in ('binary', 'hrf'):
        rows = [r for r in diag if r['variant'] == v]
        cond = np.array([r['cond_number'] for r in rows]); vr = np.array([r['mean_var_removed'] for r in rows])
        print(f'  {v}: {len(rows)} runs | cond(X) median={np.median(cond):.1f} max={cond.max():.1f} '
              f'| var removed mean={vr.mean()*100:.1f}% max={max(r["max_var_removed"] for r in rows)*100:.1f}%')


if __name__ == '__main__':
    main()
