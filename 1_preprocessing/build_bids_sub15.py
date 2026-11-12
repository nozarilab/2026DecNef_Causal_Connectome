"""Build a BIDS dataset for Study 5, sub-15 from the raw (non-BIDS) data, so that fMRIPrep
(23.1.4, --cifti-output) can regenerate its fsLR 91k CIFTI outputs.

TR = 2.0 s, task 'nfb', no fieldmaps; NF sessions 2-4, runs 1-8.
Input:  DECNEF_RAW/Study5/sub-15
Output: Results/glm_revision/bids_sub15/
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import json
import shutil
import numpy as np
import nibabel as nib

RAW = os.path.join(config.RAW_DIR, 'Study5', 'sub-15')
BIDS = os.path.join(config.ROBUSTNESS_DIR, 'bids_sub15')
SUB = '15'
TR = 2.0
SESSIONS = [2, 3, 4]
RUNS = list(range(1, 9))     # runs 1-8 (the parcellation uses these)


def set_tr_header(path, tr):
    img = nib.load(path)
    z = img.header.get_zooms()
    if len(z) >= 4 and abs(z[3] - tr) < 1e-6:
        return False                                 # already correct -> no rewrite
    hdr = img.header.copy()
    hdr.set_zooms(tuple(z[:3]) + (tr,) + tuple(z[4:]))
    nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, hdr).to_filename(path)
    return True


def main():
    os.makedirs(BIDS, exist_ok=True)
    with open(os.path.join(BIDS, 'dataset_description.json'), 'w') as f:
        json.dump({'Name': 'DRMD_Study5_sub15_recon', 'BIDSVersion': '1.8.0',
                   'DatasetType': 'raw'}, f, indent=2)

    # anat (session-agnostic)
    adir = os.path.join(BIDS, f'sub-{SUB}', 'anat')
    os.makedirs(adir, exist_ok=True)
    src_t1 = os.path.join(RAW, 'anat', f'sub-{SUB}_T1w.nii.gz')
    dst_t1 = os.path.join(adir, f'sub-{SUB}_T1w.nii.gz')
    shutil.copy(src_t1, dst_t1)
    print('anat:', dst_t1)

    # func ses-2,3,4 runs 1-8
    n_bold = 0
    for ses in SESSIONS:
        fdir = os.path.join(BIDS, f'sub-{SUB}', f'ses-{ses}', 'func')
        os.makedirs(fdir, exist_ok=True)
        for run in RUNS:
            src = os.path.join(RAW, 'func', f'ses-{ses}', f'sub-{SUB}_task-nfb_bold_run-{run}.nii.gz')
            if not os.path.exists(src):
                print('  MISSING raw', src); continue
            stem = f'sub-{SUB}_ses-{ses}_task-nfb_run-{run}_bold'
            dst = os.path.join(fdir, stem + '.nii.gz')
            shutil.copy(src, dst)
            fixed = set_tr_header(dst, TR)
            with open(os.path.join(fdir, stem + '.json'), 'w') as f:
                json.dump({'RepetitionTime': TR, 'TaskName': 'nfb'}, f, indent=2)
            n_bold += 1
            print(f'  ses-{ses} run-{run}: {stem}.nii.gz  TR-fixed={fixed}')
    print(f'built BIDS for sub-{SUB}: {n_bold} bold runs across ses {SESSIONS}')
    print('BIDS root:', BIDS)


if __name__ == '__main__':
    main()
