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
    raw_data = mne.io.read_raw_ctf(raw_data_path, preload='True')
    
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
            

def perform_ica(data, n_components, save_fig, save_pth, data_info):
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

#%% meg data preprocess and ICA-denoising

# 1. preproc and ICA decomposition
bids_root = '/nfs/e5/studyforrest/forrest_movie_meg'
ICA_results_dir = os.getcwd()

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
        filter_ica = perform_ica(filter_raw, n_components=20, save_fig=True, save_pth=pjoin(ICA_results_dir, 'ICA_images'), data_info=[sub_idx, run_idx])
        filter_ica.save(pjoin(ICA_results_dir, 'ICA_artifacts', 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)))
        
        print('{} {}done')
   
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

# 4. filter out artifact-IC and reconstruct raw data
sub_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
run_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 9)]
for sub_idx in sub_list:
    if sub_idx == '01':
        run_ls = run_list.append('09')
    else:
        run_ls = run_list
    for run_idx in run_ls:
          
        sub_raw = load_sub_raw_data(subject_idx=sub_idx, run_idx=run_idx)
        filter_raw = sub_raw.copy()
        filter_raw.load_data().filter(l_freq=1, h_freq=None)
        filter_ica = mne.preprocessing.read_ica(pjoin(os.getcwd(), 'ICA_artifacts', 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)))
        
        if artifact_dict['sub{}_run{}'.format(sub_idx, run_idx)]:
            filter_ica.exclude = artifact_dict['sub{}_run{}'.format(sub_idx, run_idx)]
            recons_raw = filter_raw.copy()
            filter_ica.apply(recons_raw)
        
            data_save_pth = pjoin(os.getcwd(), 'preprocessed_data')
            if not os.path.isdir(data_save_pth):
                os.mkdir(data_save_pth)
                
            recons_raw.save(pjoin(data_save_pth, 'sub{}_run{}_preprocessed_meg.fif'.format(sub_idx, run_idx)))

#%% organize preprocessed data to bids format

bids_root = pjoin(os.getcwd(), 'preprocessed_data')
task = 'movie'
ses = 'movie'
sh.rmtree(bids_root, ignore_errors=True)

sub_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
run_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 9)]
for sub_idx in sub_list:
    if sub_idx == '01':
        run_ls = run_list.append('09')
    else:
        run_ls = run_list
    for run_idx in run_ls:
        
        
        # 1. organize preprocessed data (meg_data, event)
        
        # 1. organize derivtives (ica comp, ica label)

# =============================================================================
#         path = path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/preproc_data/sub-{}/ses-movie/meg'.format(sub_idx)
#         fname = 'sub-{}_ses-movie_task-movie_run-{}_desc-preproc_meg.fif'.format(sub_idx, run_idx)
#         raw = mne.io.read_raw(pjoin(path, fname))
#         
#         # sub_ann = raw.annotations
#         # i = 0
#         # bad_seg_idx = []
#         # for item in sub_ann.__iter__():
#         #     if item['description'] == 'bad segment':
#         #         bad_seg_idx.append(i)
#         #     i = i + 1
#         # bad_annot = sub_ann.__getitem__(bad_seg_idx)
# 
#         # generate events file
#         raw.info['line_freq'] = 50
#         raw.set_annotations(None)
#         events = mne.find_events(raw, stim_channel='UPPT001', min_duration=2/raw.info['sfreq'])
#         events = mne.merge_events(events, list(np.arange(1, 92)), 1)
#         events_id = {'beginning': 255, 'watching': 1}
#         # convert data to bids format
#         bids_pth = BIDSPath(subject=sub_idx, session=ses, task=task, run=int(run), processing = 'preproc', root=bids_root)
#         write_raw_bids(raw, bids_pth, events_data=events, event_id=events_id, overwrite=True)
#         
# 
# ori_path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/ICA_artifacts'
# deri_path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/preproc_data/derivatives'
# 
# sub_list = np.arange(1,12)
# 
# for sub in sub_list:
#     if sub == 1:
#         run_list = np.arange(1,10)
#     else:
#         run_list = np.arange(1,9)
#         
#     if sub < 10:
#         sub_idx = '0' + str(sub)
#     else:
#         sub_idx = str(sub)
#     
#     os.mkdir(pjoin(deri_path, 'sub-{}'.format(sub_idx)))
#     os.mkdir(pjoin(deri_path, 'sub-{}'.format(sub_idx), 'ses-movie'))
#     sub_ica_path = pjoin(deri_path, 'sub-{}'.format(sub_idx), 'ses-movie', 'ICA')
#     os.mkdir(sub_ica_path)
#         
#     for run in run_list:
#         run_idx = '0' + str(run)
#         
#         sh.copy(pjoin(ori_path, 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)), sub_ica_path)
#         os.rename(pjoin(sub_ica_path, 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)), pjoin(sub_ica_path, 'sub-{}_ses-movie_task-movie_run-{}_ica.fif.gz'.format(sub_idx, run_idx)))
# 
# =============================================================================
