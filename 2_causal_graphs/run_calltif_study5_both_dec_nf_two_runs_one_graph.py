"""Study 5: CaLLTiF causal graphs for the NF and DC sessions, one graph per pair of consecutive runs
(last 152 samples of each run, multiple-dataset mode, tau_min = 0, tau_max = 1).
Outputs: Results/calltif_decnef_study5_{nf,dec}_two_runs_one_graph.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import nibabel as nib
import os
import datetime
import time
import sys
import warnings
import glob
from datetime import datetime
import random 
import itertools 

# Causal Inference Methods
sys.path.insert(0, config.TIGRAMITE_DIR)

# Tigramite Package
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
import calltif

# parameters
Num_parcels = 116
nodes = ['X%d'%i for i in range(1 , Num_parcels+1 )]    # later you need to add the labels for each parcel: maybe the label of the brain structure where each parcel belongs to

tau_min = 0
tau_max = 1
pc_alpha = 1
alpha_level = 1e-2/4

subject_id_selected =  np.array(['01','02','03','06','08','09','11','12','15','16'])
num_subjects = subject_id_selected.shape[0]

num_nf_sessions = 3
num_dec_sessions = 1
num_nf_runs = 8
num_dec_runs = 6
num_samples_in_each_run = 152     #175
num_nodes = Num_parcels

parcorr = ParCorr()

# functions
def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle= True)
    for varName in Data:
        globals()[varName] = Data[varName]  

# Neurofeedback
import_npz('Results/study5_preprocessed_confounders_removed_selected_subjects.npz')

multipledataset_calltif_nf_study5_link_matrix = [[ [ 0 for i in range(num_nf_runs)] for i in range(num_nf_sessions)] for j in range(num_subjects)] 
multipledataset_calltif_nf_study5_execution_time = [[ [ 0 for i in range(num_nf_runs)] for i in range(num_nf_sessions)] for j in range(num_subjects)] 

for subj in range(0,num_subjects):    
    for ses in range(0,num_nf_sessions):  
        for r in range(0,num_nf_runs-1):
            data = dict()
            data[0] = np.array(fmri_data_all_preprocessed_parcelated[subj][ses][r])[-num_samples_in_each_run:,:].astype(np.float64) 
            data[1] = np.array(fmri_data_all_preprocessed_parcelated[subj][ses][r+1])[-num_samples_in_each_run:,:].astype(np.float64) 
                
        # run calltif
            start = time.time()	

            dataframe  = pp.DataFrame(data, var_names = nodes, analysis_mode = 'multiple')
            link_matrix = calltif.run_calltif(dataframe,parcorr, tau_min= tau_min, tau_max = tau_max,pc_alpha = pc_alpha, alpha_level = alpha_level)

            end = time.time()

            p = link_matrix['p_matrix']
            print(np.count_nonzero(np.isnan(p[:,:,0])), np.count_nonzero(np.isnan(p[:,:,1])))

            # save results
            multipledataset_calltif_nf_study5_link_matrix[subj][ses][r]= link_matrix
            multipledataset_calltif_nf_study5_execution_time[subj][ses][r] =  end - start

            print('NF','Subject %d'%(subj+1),'Session %d'%(ses+1), 'Run %d'%(r+1), 'Execution time: %0.3f s'%(end-start))

            np.savez('Results/calltif_decnef_study5_nf_two_runs_one_graph', multipledataset_calltif_nf_study5_link_matrix = multipledataset_calltif_nf_study5_link_matrix, multipledataset_calltif_nf_study5_execution_time = multipledataset_calltif_nf_study5_execution_time)


### Decoder
import_npz('Results/study5_preprocessed_confounders_removed_selected_subjects_decoder_session0.npz')

multipledataset_calltif_decoder_study5_link_matrix = [[ [ 0 for i in range(num_dec_runs)] for i in range(num_dec_sessions)] for j in range(num_subjects)] 
multipledataset_calltif_decoder_study5_execution_time = [[ [ 0 for i in range(num_dec_runs)] for i in range(num_dec_sessions)] for j in range(num_subjects)] 

for subj in range(0,num_subjects):    
    for ses in range(0,num_dec_sessions):  
        for r in range(0,num_dec_runs-1):
            data = dict()
            data[0] = np.array(fmri_data_all_preprocessed_parcelated[subj][ses][r])[-num_samples_in_each_run:,:].astype(np.float64) 
            data[1] = np.array(fmri_data_all_preprocessed_parcelated[subj][ses][r+1])[-num_samples_in_each_run:,:].astype(np.float64) 
                
            # run calltif
            start = time.time()	

            dataframe  = pp.DataFrame(data, var_names = nodes, analysis_mode = 'multiple')
            link_matrix = calltif.run_calltif(dataframe,parcorr, tau_min= tau_min, tau_max = tau_max,pc_alpha = pc_alpha, alpha_level = alpha_level)

            end = time.time()

            p = link_matrix['p_matrix']
            print(np.count_nonzero(np.isnan(p[:,:,0])), np.count_nonzero(np.isnan(p[:,:,1])))

            # save results
            multipledataset_calltif_decoder_study5_link_matrix[subj][ses][r]= link_matrix
            multipledataset_calltif_decoder_study5_execution_time[subj][ses][r] =  end - start

            print('Dec','Subject %d'%(subj+1),'Session %d'%(ses+1), 'Run %d'%(r+1), 'Execution time: %0.3f s'%(end-start))

            np.savez('Results/calltif_decnef_study5_dec_two_runs_one_graph', multipledataset_calltif_decoder_study5_link_matrix = multipledataset_calltif_decoder_study5_link_matrix ,  multipledataset_calltif_decoder_study5_execution_time = multipledataset_calltif_decoder_study5_execution_time)
