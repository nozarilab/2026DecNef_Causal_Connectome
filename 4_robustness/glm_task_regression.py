"""GLM task regression of the parcellated BOLD time series (Supplementary Figs. 1-5 and 11).

Regresses the task structure (one regressor per task interval) out of the preprocessed,
confound-removed, parcellated time series, so that CaLLTiF can be re-run on the residuals.

Two regressor variants:
  hrf=False : binary boxcars, one per task interval;
  hrf=True  : the same boxcars convolved with a canonical double-gamma HRF.

Inputs:
  Results/study{S}_preprocessed_confounders_removed_selected_subjects[_decoder_session0].npz
  NF events from the DecNef Collection (DECNEF_EVENTS, default DecNef_events/) and the
  DC templates from build_dc_events.py.
"""

import os
import csv
import numpy as np
from scipy.stats import gamma

TR = 2.0  # seconds, all studies (Cortese et al. 2021)

# Data locations (can be overridden with environment variables).
RESULTS_DIR = os.environ.get('DECNEF_RESULTS', 'Results')      # preprocessed NF npz
EVENTS_DIR = os.environ.get('DECNEF_EVENTS', 'DecNef_events')  # events.csv tree

# --- mapping from npz subject/session index back to the events folder ----------
# subject_id_selected as used by each NF loader (Codes_for_Loading_Data/load_*.py)
SUBJECT_IDS = {
    1: ['03', '24', '21', '15', '23', '05', '17', '13', '18', '07'],
    2: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    3: ['01', '05', '08', '10', '12', '13', '14', '15', '16'],
    4: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
    5: ['01', '02', '03', '06', '08', '09', '11', '12', '15', '16'],
}
# NF session on-disk offset: disk_session = npz_session_index + SESSION_OFFSET
# (studies 1-3 load ses+1; studies 4-5 load ses+2)
SESSION_OFFSET = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}

# event types modelled as task regressors (ITI1/ITI2 left as implicit baseline)
DEFAULT_EVENT_TYPES = ['induction', 'feedback', 'baseline']


def canonical_hrf(t, peak_delay=6.0, undershoot_delay=16.0,
                  peak_disp=1.0, undershoot_disp=1.0, ratio=1.0 / 6.0):
    """SPM-style canonical double-gamma HRF sampled at times `t` (seconds)."""
    hrf = (gamma.pdf(t, peak_delay / peak_disp, scale=peak_disp)
           - ratio * gamma.pdf(t, undershoot_delay / undershoot_disp,
                               scale=undershoot_disp))
    return hrf


def load_events(path):
    """Read an events.csv -> list of (event_name, onset_s, duration_s)."""
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append((row['event'], float(row['onset']), float(row['duration'])))
    return out


def build_design_matrix(events, n_trs, tr=TR, hrf=True,
                        event_types=DEFAULT_EVENT_TYPES, dt=0.1,
                        add_constant=True):
    """
    Build a (n_trs, n_regressors) design matrix from an events list.

    One regressor per entry in `event_types`. Each is a boxcar over that
    interval, optionally convolved with the canonical HRF, then resampled to
    TR resolution by averaging the high-resolution signal within each TR bin
    (handles sub-TR events and fractional overlaps). A constant column is
    appended last when add_constant=True.
    """
    hi_t = np.arange(0.0, n_trs * tr + dt, dt)
    cols = []
    for ev in event_types:
        box = np.zeros_like(hi_t)
        for name, onset, dur in events:
            if name == ev and dur > 0:
                box[(hi_t >= onset) & (hi_t < onset + dur)] = 1.0
        if hrf:
            h = canonical_hrf(np.arange(0.0, 32.0 + dt, dt))
            sig = np.convolve(box, h)[:box.size]
        else:
            sig = box
        # average within each TR bin
        reg = np.empty(n_trs)
        for i in range(n_trs):
            m = (hi_t >= i * tr) & (hi_t < (i + 1) * tr)
            reg[i] = sig[m].mean() if m.any() else 0.0
        cols.append(reg)
    X = np.column_stack(cols) if cols else np.empty((n_trs, 0))
    if add_constant:
        X = np.column_stack([X, np.ones(n_trs)])
    return X


def regress_out(Y, X):
    """Return residuals Y - X @ beta_hat, with beta_hat = pinv(X) @ Y.

    Y : (n_trs, n_parcels)   task residuals are taken column-wise.
    X : (n_trs, n_regressors)
    """
    beta = np.linalg.pinv(X) @ Y
    return Y - X @ beta, beta


def events_path(study, sub_id, disk_session, run):
    """Path to the events.csv for a given study/subject/disk-session/run."""
    return os.path.join(
        EVENTS_DIR, f'Study{study}', f'sub-{sub_id}', 'func',
        f'ses-{disk_session}',
        f'sub-{sub_id}_task-nfb_run-{run}_events.csv')


def regress_run(timeseries, study, sub_id, disk_session, run, hrf=True,
                event_types=DEFAULT_EVENT_TYPES):
    """Task-regress a single (n_trs, n_parcels) NF run.

    Returns (residuals, design_matrix, beta, frac_var_removed_per_parcel).
    """
    n_trs = timeseries.shape[0]
    events = load_events(events_path(study, sub_id, disk_session, run))
    X = build_design_matrix(events, n_trs, hrf=hrf, event_types=event_types)
    resid, beta = regress_out(timeseries, X)
    var_before = timeseries.var(axis=0)
    var_removed = 1.0 - resid.var(axis=0) / np.where(var_before > 0, var_before, 1.0)
    return resid, X, beta, var_removed


def load_nf_npz(study):
    """Load the NF parcellated time series npz for a study.

    Returns (data, num_runs_all) where data[subj][ses][run] is (n_trs, 116)
    or a default 0 for unfilled slots.
    """
    f = os.path.join(
        RESULTS_DIR,
        f'study{study}_preprocessed_confounders_removed_selected_subjects.npz')
    d = np.load(f, allow_pickle=True)
    return d['fmri_data_all_preprocessed_parcelated'], d['num_runs_all']
