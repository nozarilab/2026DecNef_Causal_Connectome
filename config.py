"""Paths shared by all scripts.

Run every script from the repository root, e.g.
    python 2_causal_graphs/run_calltif_study1_both_dec_nf_two_runs_one_graph.py

Edit the defaults below, or override them with the environment variables.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# BIDS datasets (with fMRIPrep outputs) of the five DecNef studies
DATA_DIR = os.environ.get('DECNEF_DATA', os.path.join(ROOT, 'Data'))

# Raw, non-BIDS data as distributed (only needed to rebuild Study 5, sub-15)
RAW_DIR = os.environ.get('DECNEF_RAW', os.path.join(DATA_DIR, 'raw'))

# Intermediate and final results
RESULTS_DIR = os.environ.get('DECNEF_RESULTS', os.path.join(ROOT, 'Results'))
ROBUSTNESS_DIR = os.path.join(RESULTS_DIR, 'glm_revision')

# Third-party code and atlas files (see README)
EXTERNAL_DIR = os.environ.get('DECNEF_EXTERNAL', os.path.join(ROOT, 'External Packages and Files'))
TIGRAMITE_DIR = os.environ.get('TIG_PATH', os.path.join(EXTERNAL_DIR, 'tigramite-master'))

# BIDS root of each study
BIDS_DIRS = {
    1: os.path.join(DATA_DIR, 'DRMD_Study1', 'Study1', 'Study1_BIDS'),
    2: os.path.join(DATA_DIR, 'DRMD_Study2', 'Study2', 'Study2_BIDS'),
    3: os.path.join(DATA_DIR, 'DRMD_Study3', 'Study3', 'Study3_BIDS'),
    4: os.path.join(DATA_DIR, 'DRMD_Study4', 'Study4_BIDS'),
    5: os.path.join(DATA_DIR, 'DRMD_Study5', 'Study5'),
}

# fMRIPrep output folder of each study
FMRIPREP_DIRS = {s: os.path.join(d, 'preprocessed') for s, d in BIDS_DIRS.items()}
FMRIPREP_DIRS[5] = os.path.join(BIDS_DIRS[5], 'preprocessed', 'fMRIPrep')
