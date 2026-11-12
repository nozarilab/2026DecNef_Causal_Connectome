"""Set the repetition time in each NIfTI header to the RepetitionTime of its BIDS sidecar.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from bids import grabbids
import nibabel as nb

def set_tr(img, tr):
    header = img.header.copy()
    zooms = header.get_zooms()[:3] + (tr,)
    header.set_zooms(zooms)
    return img.__class__(img.get_fdata().copy(), img.affine, header)


def sync_tr(bids_root):
    layout = grabbids.BIDSLayout(bids_root)
    for nii in layout.get(extensions=['.nii', '.nii.gz']):
        metadata = layout.get_metadata(nii.filename)
        if 'RepetitionTime' in metadata:
            img = nb.load(nii.filename)
            if img.header.get_zooms()[3:] != (metadata['RepetitionTime'],):
                fixed_img = set_tr(img, metadata['RepetitionTime'])
                fixed_img.to_filename(nii.filename)
                print(nii.filename, 'fixed')

if __name__ == '__main__':
    # usage: python 1_preprocessing/correct_TR_for_BIDS.py [study number ...]  (default: all studies)
    studies = [int(a) for a in sys.argv[1:]] or sorted(config.BIDS_DIRS)
    for study in studies:
        sync_tr(config.BIDS_DIRS[study])
