#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:48:02 2021

Pre-processing for studyforrest MEG data.
Consist of bad sensors interpolation, band-pass filter and removing artifacts using ICA.

@author: daiyuxuan
"""

import numpy as np
from os.path import join as pjoin
from os.path import isdir
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import mne
import xlrd
import shutil as sh
from mne_bids import write_raw_bids, BIDSPath
import pandas as pd

#%% 

def load_sub_raw_data(bids_root, subject_idx, run_idx):
    """
    load raw meg data. 
    """
    
    if not isinstance(subject_idx, str):
        raise ValueError('subject_dix must be str')
        
    if not isinstance(run_idx, str):
        raise ValueError('run_idx must be str')
    
    subject_data_folder = pjoin(bids_root, 'sub-{}'.format(subject_idx), 'ses-movie', 'meg')
    fname = 'sub-' + subject_idx + '_ses-movie_task-movie_run-' + run_idx + '_meg.ds'
    raw_data_path = pjoin(subject_data_folder, fname)
    raw_data = mne.io.read_raw_ctf(raw_data_path, preload=False)
    
    print('total channels number is {}'.format(len(raw_data.info['chs'])))
    print('sample frequency is {} Hz'.format(raw_data.info['sfreq']))

    return raw_data


def print_bad_channel(meg_data, subject_idx, run_idx):
    """
    print bad channel information
    """

    if not meg_data.info['bads']:
        print('No bad channels in sub-{} run-{}'.format(subject_idx, run_idx))
    else:
        print('Bad channel(s) detected in sub-{} run-{} : {}'.format(subject_idx, run_idx, meg_data.info['bads']))
            

def perform_ica(raw, n_components, save_fig, save_pth, data_info):
    """
    perform ICA and save results with 3 figures: ICA sources, full duration ICA sources, ICA components topomap
    input variables:
        data: meg raw data
        n_components: int, meg ICA components
        save: bool, whether to save ICA pics
        save_pth: folder to save ICA images
        data_info: [subject_idx, run_idx], subject_idx and run_idx should be str
    
    Return: ICA object
    """
    if save_fig:
        if not isdir(save_pth):
            os.mkdir(save_pth)
    
    # ICA
    ica = mne.preprocessing.ICA(n_components = n_components, random_state=0)
    ica.fit(raw)
    ica.detect_artifacts(raw)
    
    # plot ICA sources
    ica.plot_sources(filter_raw)
    if save_fig:
        plt.savefig(pjoin(save_pth, 'sub{}_run{}_ica_sources'.format(data_info[0], data_info[1])))
    
    # plot full duration ICA sources
    ica.plot_sources(raw, start=0, stop=raw.times[-1])
    if save_fig:
        plt.savefig(pjoin(save_pth, 'sub{}_run{}_icasources_fulldur'.format(data_info[0], data_info[1])))

    # plot ICA components topomap
    ica.plot_components()
    if save_fig:
        plt.savefig(pjoin(save_pth, 'sub{}_run{}_ica_components'.format(data_info[0], data_info[1])))
    
    return ica

def bids_dir_check(bids_root, sub_idx):
    
    sub_dir = pjoin(bids_root, f'sub-{sub_idx}')
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
        os.mkdir(pjoin(sub_dir, 'ses-movie'))
        os.mkdir(pjoin(sub_dir, 'ses-movie', 'meg'))
    else:
        print(f'sub-{sub_idx} bids dir already exist')

def artifacts2csv(artifact_dict, sub_idx, run_idx, save_path):
    
    arti_list = artifact_dict['sub{}_run{}'.format(sub_idx, run_idx)]
    components = np.arange(0,20)
    
    artifacts_label = []
    for i in components:
        if i in arti_list:
            artifacts_label.append('artifacts')
        else:
            artifacts_label.append('signal')           
   
    arti_df = pd.DataFrame({'ComponentIndex':components, 'Label':artifacts_label})
    arti_df.to_csv(pjoin(save_path, 'sub-{}_ses_movie_task-movie_run-{}_decomposition.tsv'.format(sub_idx, run_idx)), columns=['ComponentIndex', 'Label'], sep='\t', index=False)
    
#%% meg data preprocess and ICA-denoising

# 1. preproc and ICA decomposition
bids_root = '/nfs/e5/studyforrest/forrest_movie_meg'
bids_dir = '/nfs/e5/studyforrest/forrest_movie_meg/meg_preproc_data'

sub_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
run_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 9)]

for sub_idx in sub_list:
    
    # get runlist
    if sub_idx == '01':
        run_ls = run_list.append('09')
    else:
        run_ls = run_list
    
    # preproc and ICA
    for run_idx in run_ls:     
        sub_raw = load_sub_raw_data(bids_root, subject_idx=sub_idx, run_idx=run_idx)  
        # print bad channels
        print_bad_channel(sub_raw, subject_idx=sub_idx, run_idx=run_idx)       
        # 1Hz high-pass
        filter_raw = sub_raw.copy()
        filter_raw.load_data().filter(l_freq=1, h_freq=None)     
        
        # ICA
        ICA_results_dir = pjoin(bids_dir, )
        filter_ica = perform_ica(filter_raw, n_components=20, save_fig=False, save_pth=os.getcwd(), data_info=[sub_idx, run_idx])
        filter_ica.save(pjoin(ICA_results_dir, 'sub-{}_ses-movie_task-movie_run-{}_ica.fif.gz'.format(sub_idx, run_idx)))
        
        print('sub-{} run-{} done'.format(sub_idx, run_idx))
   
# ==========================================================
# 2. artifact-ICs were manually selected and saved as a excel file
# ==========================================================

# 3. read artifact-ICs
fpth = '/nfs/e2/workingshop/daiyuxuan/MEG-paper'
fname = 'A_IC_1hz_dict.xlsx'

wb = xlrd.open_workbook(pjoin(fpth, fname))
sheet = wb.sheet_by_index(0)

sub_list = sheet.col_values(0)
run_list = sheet.col_values(1)
components_list = sheet.col_values(2)
certain_list = sheet.col_values(3)

artifact_dict = {}

for idx in np.arange(len(sub_list)):
    sub = sub_list[idx]
    run = run_list[idx]
    key = sub + '_' + run
    
    value = []
    if certain_list[idx]:
        components = components_list[idx].split(',')
        for comp in components:
            if comp:
                value.append(int(comp))
    
    artifact_dict[key] = value
wb.release_resources()

# 4. filter out artifact-IC, reconstruct raw data and organize data to BIDS format
sub_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
run_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 9)]
bids_dir = '/nfs/e5/studyforrest/forrest_movie_meg/gump_meg_bids/derivatives/preproc_meg-mne_mri-fmriprep'

for sub_idx in sub_list:
    if sub_idx == '01':
        run_ls = run_list + ['09']
    else:
        run_ls = run_list
    
    # creat dir
    bids_dir_check(bids_dir, sub_idx)
    
    for run_idx in run_ls:
          
        sub_raw = load_sub_raw_data('/nfs/e5/studyforrest/forrest_movie_meg', subject_idx=sub_idx, run_idx=run_idx)
        filter_raw = sub_raw.copy()
        filter_raw.load_data().filter(l_freq=1, h_freq=None)
        ica_path = pjoin(bids_dir,f'sub-{sub_idx}','ses-movie','meg')
        filter_ica = mne.preprocessing.read_ica(pjoin(ica_path, 'sub-{}_ses-movie_task-movie_run-{}_ica.fif.gz'.format(sub_idx, run_idx)))
        
        if artifact_dict['sub{}_run{}'.format(sub_idx, run_idx)]:
            filter_ica.exclude = artifact_dict['sub{}_run{}'.format(sub_idx, run_idx)]
            recons_raw = filter_raw.copy()
            filter_ica.apply(recons_raw)
        else:
            recons_raw = filter_raw.copy()
            
        # update ica 
        filter_ica.save(pjoin(ica_path, 'sub-{}_ses-movie_task-movie_run-{}_ica.fif.gz'.format(sub_idx, run_idx)))
        
        # mark bad channels
        if sub_idx == '05' and run_idx == '05':
            recons_raw.info['bads'] = ['MRT53-4503', 'MRT54-4503']
        
        # save pre-proc data
        save_path = pjoin(bids_dir, f'sub-{sub_idx}', 'ses-movie', 'meg')
        recons_raw.save(pjoin(save_path, 'sub-{}_ses-movie_task-movie_run-{}_meg.fif'.format(sub_idx, run_idx)))
        
        # save artifacts info
        artifacts2csv(artifact_dict, sub_idx, run_idx, save_path)
        
    print('sub-{0}: MEG convertion done'.format(sub_idx))
    
