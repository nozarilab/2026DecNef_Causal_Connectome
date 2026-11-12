#!/bin/bash
# Run fMRIPrep (Docker wrapper) on one study, with CIFTI (fsLR 91k) outputs.
# Usage: bash 1_preprocessing/fmriprep_all_subjects.sh <BIDS root> <FreeSurfer license> <subject labels...>
# Example (Study 3): bash 1_preprocessing/fmriprep_all_subjects.sh Data/DRMD_Study3/Study3/Study3_BIDS license.txt 01 05 08 10 12 13 14 15 16

BIDS_DIR=$1
FS_LICENSE=$2
shift 2

for sub in "$@"
do
    fmriprep-docker "$BIDS_DIR" "$BIDS_DIR/derivatives" participant --participant_label "$sub" --fs-license-file "$FS_LICENSE" --cifti-output
done
