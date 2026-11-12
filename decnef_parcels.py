"""116-parcel labels and the 16-subnetwork mapping used throughout the analysis.

Order: 8 subcortical RH, 8 subcortical LH, then the 100 Schaefer-100 7-network cortical
parcels. Subnetworks: indices 0-7 -> RH_Sub, 8-15 -> LH_Sub, cortical parcels by their
hemisphere + network prefix (e.g. 'LH_Vis', 'RH_Con').
"""
import numpy as np

NET_LABELS = ['LH_Sub', 'RH_Sub', 'LH_Vis', 'RH_Vis', 'LH_Lim', 'RH_Lim',
              'LH_Con', 'RH_Con', 'LH_Def', 'RH_Def', 'LH_Sal', 'RH_Sal',
              'LH_Dor', 'RH_Dor', 'LH_Som', 'RH_Som']


def get_parcel_labels():
    """116 parcel labels: 8 RH-sub, 8 LH-sub, then 100 Schaefer cortical (nilearn order)."""
    from nilearn import datasets
    a = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
    cort = [(l.decode() if isinstance(l, bytes) else l).replace('7Networks_', '')
            for l in a['labels']]
    cort = [l for l in cort if l != 'Background']   # nilearn >= 0.11 adds a 'Background' label
    assert len(cort) == 100, len(cort)
    sub = ['RH_Sub_%d' % i for i in range(1, 9)] + ['LH_Sub_%d' % i for i in range(1, 9)]
    return sub + cort


def get_network_parcel_idx(parcel_labels=None):
    """Map each of the 16 subnetworks -> list of parcel indices (as in 3_main_analysis/main_analysis.ipynb)."""
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
    assert len(pl) == 116, len(pl)
    total = sum(len(v) for v in npi.values())
    print('parcel_labels:', len(pl), '| assigned to subnetworks:', total)
    for k in NET_LABELS:
        print(f'  {k}: {len(npi[k])} parcels')
    # save for reuse
    np.savez('Results/glm_revision/parcel_mapping.npz',
             parcel_labels=np.array(pl),
             net_labels=np.array(NET_LABELS),
             **{k: np.array(v) for k, v in npi.items()})
    print('saved Results/glm_revision/parcel_mapping.npz')
