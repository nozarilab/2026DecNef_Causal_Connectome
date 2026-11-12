# Whole-brain causal connectivity during decoded neurofeedback

Code for

> Arab F., Ghassami A., Jamalabadi H., Peters M. A. K., Nozari E.
> *Whole-brain causal connectivity during decoded neurofeedback: a meta-study.*
> *Network Neuroscience* (2026). [doi:10.1162/NETN.a.626](https://doi.org/10.1162/NETN.a.626)

The analysis learns whole-brain causal graphs with **CaLLTiF**
([Arab et al., 2025, *Network Neuroscience*](https://github.com/nozarilab/2023Arab_CaLLTiF)) from the fMRI data of
five decoded neurofeedback (DecNef) studies, and compares neurofeedback (NF) sessions with the
decoder-construction (DC) sessions used as baseline.

## Data

All fMRI data come from the **DecNef Collection**
([Cortese et al., 2021, *Scientific Data* 8:65](https://doi.org/10.1038/s41597-021-00845-7)),
available on application through the ATR repository or Synapse (see the paper).
No data are included in this repository. The code expects, under `Data/` (see `config.py`):

- the BIDS dataset of each study, with fMRIPrep outputs (fsLR 91k CIFTI) in its `preprocessed/` folder;
- the neurofeedback event files (`*_task-nfb_run-N_events.csv`), used by the GLM analyses
  (`DECNEF_EVENTS`, default `DecNef_events/Study{N}/sub-XX/func/ses-Y/`);
- the trial-by-trial neurofeedback scores (`in_score_d*.csv`), copied to
  `Results/glm_revision/scores/Study{N}/sub-XX/pattern/`.

## Setup

```
pip install -r requirements.txt
```

Third-party code and atlas files go in `External Packages and Files/` (the same layout as the CaLLTiF repository):

| File | Source |
|---|---|
| `tigramite-master/` | [tigramite](https://github.com/jakobrunge/tigramite), version 5.1. `calltif.py` passes `selected_links` to `PCMCI.run_mci`, which later versions replaced with `link_assumptions`. |
| `Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii` | [Tian et al., 2020, subcortex atlas](https://github.com/yetianmed/subcortex) (`Group-Parcellation/3T/Cortex-Subcortex/`) |
| `Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii` | same, for the Schaefer-200 robustness analysis |

Paths are set in `config.py` and can be overridden with the environment variables
`DECNEF_DATA`, `DECNEF_RAW`, `DECNEF_RESULTS`, `DECNEF_EXTERNAL` and `TIG_PATH`.
Run every script from the repository root, e.g. `python 2_causal_graphs/run_calltif_study1_both_dec_nf_two_runs_one_graph.py`.

## Pipeline

### 1. Preprocessing (`1_preprocessing/`)

| Step | Script |
|---|---|
| fMRIPrep with CIFTI outputs, per study | `fmriprep_all_subjects.sh` |
| fix the TR in NIfTI headers | `correct_TR_for_BIDS.py` |
| 9P confound removal, detrending, 116-parcel time series (Schaefer-100 + Tian S1), NF and DC sessions | `load_preprocessed_fmri_and_parcelate_study{1-5}[_decoder_session0].py` |
| 216-parcel time series (Schaefer-200 + Tian S1), for Supplementary Figs. 12-14 | `parcelate_s200.py` |
| Study 5, sub-15: rebuild BIDS and add it to the 216-parcel data | `build_bids_sub15.py`, `merge_sub15.py` |

### 2. Causal graphs (`2_causal_graphs/`)

`run_calltif_study{1-5}_both_dec_nf_two_runs_one_graph.py`: one CaLLTiF graph per pair of consecutive runs
(945 NF and 225 baseline graphs in total).

### 3. Main analysis (`3_main_analysis/`)

| Notebook | Figures |
|---|---|
| `main_analysis.ipynb` | Figures 1c-e, 2, 3b-c, 4; Supplementary Figs. 6-10 |
| `supp_modified_calltif_asymmetry.ipynb` | Supplementary Fig. 21 |

Figures 1a, 1b and 3a are schematics.

### 4. Robustness analyses (`4_robustness/`)

Run the steps of each block in order. Figures are written to `Results/figures_supplementary/`.

| Supplementary Figures | Steps |
|---|---|
| 1-5, 11: GLM task regression | `build_dc_events.py` → `make_glm_residuals.py`, `make_dc_glm_residuals.py` → `run_calltif_glm_9subj.py` → `make_glm_comparison_9subj.py` → `make_suppfigs_01_05_glm.py` (1-5), `make_suppfig_11_cogperc_glm.py` (11) |
| 22-25: lagged-edge threshold ᾱ | `run_alpha_sweep.py` → `make_suppfigs_22_25_alpha.py` (22, 25), `make_suppfigs_23_24_alpha_brainmaps.py` (23-24) |
| 15-20: non-overlapping graphs and two-stage discovery-test | after `run_alpha_sweep.py`: `make_suppfig_15_nonoverlap_heatmaps.py` (15), `make_suppfig_16_nonoverlap_brain.py` (16), `make_suppfigs_17_18_twostage_tables.py` (17-18), `make_suppfigs_19_20_twostage_brain.py` (19-20) |
| 12-14: Schaefer-200 + Tian S1 (216 parcels) | `parcelate_s200.py` → `run_calltif_s200.py`, `run_calltif_s200_4run.py` → `run_alpha_sweep_s200.py`, `build_s200_4run_graphs.py` → `make_suppfigs_12_13_s200.py` (12-13), `make_suppfig_14_subregion_comparison.py` (14, also needs `run_alpha_sweep.py`) |
| 26: condition numbers | `make_condnum_diagnostic_expanded.py` → `make_suppfig_26_condition_numbers.py` |

The CaLLTiF runs (`2_causal_graphs/` and the `run_calltif_*` scripts) are computationally heavy and were run on a
multi-core workstation; `run_calltif_glm_9subj.py`, `run_calltif_s200.py` and `run_calltif_s200_4run.py` take a
`--workers` argument and save one file per graph, so they can be resumed.

## Shared modules

| File | Content |
|---|---|
| `config.py` | paths |
| `calltif.py` | CaLLTiF (on top of tigramite PCMCI) |
| `functions.py` | parcellation, graph summarization and network measures |
| `decnef_parcels.py`, `decnef_parcels216.py` | parcel labels and the 16-subnetwork mapping |

## License

GNU General Public License v3.0 (see `LICENSE`).
