"""216-parcel (Schaefer-200 + Tian S1) labels and the 16-subnetwork mapping, read from the atlas
label table. Same ordering and mapping rules as decnef_parcels.py.
"""
import os
import numpy as np
import config

DLABEL = os.path.join(config.EXTERNAL_DIR, 'Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii')
NET_LABELS = ['LH_Sub', 'RH_Sub', 'LH_Vis', 'RH_Vis', 'LH_Lim', 'RH_Lim',
              'LH_Con', 'RH_Con', 'LH_Def', 'RH_Def', 'LH_Sal', 'RH_Sal',
              'LH_Dor', 'RH_Dor', 'LH_Som', 'RH_Som']


def get_parcel_labels(dlabel=DLABEL):
    """216 parcel labels in atlas order (cortical with '7Networks_' stripped)."""
    import nibabel as nib
    img = nib.load(dlabel)
    lt = img.header.get_axis(0).label[0]            # {int: (name, rgba)}
    labels = []
    for k in range(1, 217):                          # 1..216 (skip 0 background)
        name = lt[k][0]
        labels.append(name.replace('7Networks_', ''))
    return labels


def get_network_parcel_idx(parcel_labels=None):
    if parcel_labels is None:
        parcel_labels = get_parcel_labels()
    idx = {k: [] for k in NET_LABELS}
    for i in range(len(parcel_labels)):
        if i < 8:
            idx['RH_Sub'].append(i)
        elif i < 16:
            idx['LH_Sub'].append(i)
        else:
            key = parcel_labels[i][:6]
            if key in idx:
                idx[key].append(i)
    return idx


if __name__ == '__main__':
    pl = get_parcel_labels()
    npi = get_network_parcel_idx(pl)
    print('parcels:', len(pl), '| assigned:', sum(len(v) for v in npi.values()))
    for k in NET_LABELS:
        print(f'  {k}: {len(npi[k])}')
    print('control parcels (LH_Con+RH_Con):', len(npi['LH_Con']) + len(npi['RH_Con']))
    os.makedirs('Results/glm_revision', exist_ok=True)
    np.savez('Results/glm_revision/parcel_mapping_216.npz',
             parcel_labels=np.array(pl), net_labels=np.array(NET_LABELS),
             **{k: np.array(v) for k, v in npi.items()})
    print('saved Results/glm_revision/parcel_mapping_216.npz')
