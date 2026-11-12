"""GLM task-regressed residuals of the neurofeedback (NF) sessions.

Writes, per study and regressor variant (binary, hrf), the residual time series in the
nested [subject][session][run] layout read by run_calltif_glm_9subj.py, plus a CSV of
diagnostics (variance removed, condition numbers).
Output folder: Results/glm_revision/glm_data_9subj/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import numpy as np
import glm_task_regression as glm

N_SUBJECTS = 9          # ALL 9 subjects (matches the main analyses) — see *_9subj naming
N_SESSIONS = 3          # NF sessions used in the paper
OUTDIR = os.environ.get('DECNEF_OUT', os.path.join(config.ROBUSTNESS_DIR, 'glm_data_9subj'))


def build_for_study(study, variant_hrf, diag_rows):
    data, nruns = glm.load_nf_npz(study)
    sub_ids = glm.SUBJECT_IDS[study][:N_SUBJECTS]
    offset = glm.SESSION_OFFSET[study]

    resid_all = [[[0 for _ in range(8)] for _ in range(N_SESSIONS)]
                 for _ in range(N_SUBJECTS)]
    disk_sessions = []

    for si, sub_id in enumerate(sub_ids):
        for ses_idx in range(N_SESSIONS):
            disk_ses = ses_idx + offset
            if si == 0:
                disk_sessions.append(disk_ses)
            for r in range(8):
                slot = data[si][ses_idx][r]
                ts = np.array(slot)
                if ts.ndim != 2 or ts.shape[1] != 116:
                    continue  # empty/unfilled run slot
                ts = ts.astype(float)
                run = r + 1
                try:
                    resid, X, beta, vr = glm.regress_run(
                        ts, study, sub_id, disk_ses, run, hrf=variant_hrf)
                except FileNotFoundError:
                    continue
                resid_all[si][ses_idx][r] = resid
                diag_rows.append({
                    'study': study, 'subject': sub_id, 'disk_session': disk_ses,
                    'run': run, 'variant': 'hrf' if variant_hrf else 'binary',
                    'n_trs': ts.shape[0], 'cond_number': float(np.linalg.cond(X)),
                    'mean_var_removed': float(vr.mean()),
                    'max_var_removed': float(vr.max()),
                })
    return resid_all, sub_ids, disk_sessions


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    diag_rows = []
    for variant_hrf in (False, True):
        vname = 'hrf' if variant_hrf else 'binary'
        for study in (1, 2, 3, 4, 5):
            resid_all, sub_ids, disk_sessions = build_for_study(
                study, variant_hrf, diag_rows)
            # version-portable FLAT npz (run__{subj}__{ses}__{r} float arrays)
            flat = {}
            for si in range(N_SUBJECTS):
                for ses in range(N_SESSIONS):
                    for r in range(8):
                        e = resid_all[si][ses][r]
                        if isinstance(e, np.ndarray) and e.ndim == 2:
                            flat[f'run__{si}__{ses}__{r}'] = e.astype(np.float32)
            out = os.path.join(OUTDIR, f'study{study}_nf_residuals_{vname}_9subj_flat.npz')
            np.savez(out, **flat)
            print(f'[{vname}] study{study}: saved {out} ({len(flat)} runs, {N_SUBJECTS} subj)')

    # diagnostics table
    import csv
    diag_path = os.path.join(OUTDIR, 'nf_glm_diagnostics_9subj.csv')
    with open(diag_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)
    print(f'\nWrote diagnostics: {diag_path} ({len(diag_rows)} runs)')

    # quick summary
    for v in ('binary', 'hrf'):
        rows = [r for r in diag_rows if r['variant'] == v]
        cond = np.array([r['cond_number'] for r in rows])
        vr = np.array([r['mean_var_removed'] for r in rows])
        print(f'  {v}: {len(rows)} runs | cond(X) median={np.median(cond):.1f} '
              f'max={cond.max():.1f} | var removed mean={vr.mean()*100:.1f}%')


if __name__ == '__main__':
    main()
