"""Study 4, neurofeedback (NF) sessions: load the fMRIPrep outputs (fsLR 91k CIFTI), regress out
9 confounds (6 motion parameters, white matter, CSF, global signal), detrend, and parcellate
into 116 regions (Schaefer-100 7-network + Tian S1).
Output: Results/study4_preprocessed_confounders_removed_selected_subjects.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import seaborn as sb
import glob
import scipy as sp
from scipy.io import savemat
import functions as funcs

## variables
num_parcels = 116
num_cortical_parcels = 100
num_subcortical_parcels = 16

num_sessions = 3
subject_id_selected = np.array(['01','02','03','04','05','06','07','08','09','10']) 
num_subjects = subject_id_selected.shape[0]

num_runs = 8
parcellation_mask = np.asanyarray(nib.load(os.path.join(config.EXTERNAL_DIR, 'Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii')).dataobj)

##
fmri_data_all_preprocessed_fmriprep = [[[0 for k in range(num_runs)] for i in range(num_sessions)] for j in range(num_subjects)] 
fmri_data_all_preprocessed_parcelated = [[[0 for k in range(num_runs)] for i in range(num_sessions)] for j in range(num_subjects)] 

num_runs_all = np.zeros((num_subjects,num_sessions))
parcellation_masks_all = [[[0 for k in range(num_runs)] for i in range(num_sessions)] for j in range(num_subjects)] 


for subj in range(0,num_subjects):    
    for ses in range(0,num_sessions):

        path_to_subject_session = config.FMRIPREP_DIRS[4] + '/sub-%s/ses-%d/func' %(subject_id_selected[subj], (ses+2))
        num_runs_all[subj,ses] = int(len(glob.glob1(path_to_subject_session,"*.nii")))

        for r in range(0, num_runs):

            data_path = path_to_subject_session + '/sub-%s_ses-%d_task-decoder_run-%d_space-fsLR_den-91k_bold.dtseries.nii'%(subject_id_selected[subj],ses+2,r+1)
            fmri_data  = np.asanyarray(nib.load(data_path).dataobj).T
            confounds = pd.read_csv(path_to_subject_session+'/sub-%s_ses-%d_task-decoder_run-%d_desc-confounds_timeseries.tsv'%(subject_id_selected[subj],ses+2,r+1), sep='\t')

            global_signal = confounds['global_signal']
            csf = confounds['csf']
            white_matter= confounds['white_matter']
            trans_x= confounds['trans_x']
            trans_y= confounds['trans_y']
            trans_z= confounds['trans_z']
            rot_x= confounds['rot_x']
            rot_y= confounds['rot_y']
            rot_z= confounds['rot_z']

            confounders_9 = np.array([global_signal,csf,white_matter,trans_x,trans_y,trans_z,rot_x,rot_y,rot_z])
            fmri_data_confounders_removed = funcs.regressout_confounders(fmri_data,confounders_9)            
            fmri_data_masked,parcellation_mask_updated = funcs.apply_mask(fmri_data_confounders_removed,parcellation_mask)

            parcellation_masks_all[subj][ses][r] = parcellation_mask_updated

            fmri_data_masked_detrend = funcs.detrend(fmri_data_masked, parcellation_mask_updated)
            fmri_data_masked_detrend_parcelated = funcs.parcelate(fmri_data_masked_detrend, parcellation_mask_updated, num_parcels)
            fmri_data_all_preprocessed_fmriprep[subj][ses][r] = fmri_data_masked_detrend
            fmri_data_all_preprocessed_parcelated[subj][ses][r] = fmri_data_masked_detrend_parcelated
            
            print('sub-%s, ses-%d, number of runs:%d, run-%d'%(subject_id_selected[subj],ses+2, num_runs_all[subj,ses], (r+1)), fmri_data_masked_detrend_parcelated.shape)
            
            # np.savez('Results/study4_preprocessed_confounders_removed_selected_subjects', fmri_data_all_preprocessed_parcelated = np.asanyarray(fmri_data_all_preprocessed_parcelated, dtype = object), num_runs_all = num_runs_all)          