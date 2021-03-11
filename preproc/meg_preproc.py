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
import mne_bids
import mne
import xlrd
import shutil as sh
from mne_bids import write_raw_bids, BIDSPath
from scipy import stats
import re
from scipy import signal

#%%
#Loading data

def load_sub_raw_data(data_folder='/nfs/e5/studyforrest/forrest_movie_meg/', subject_idx='01', run_idx='01'):
    """
    load raw meg data. 
    
    input value: subject_idx and run_idx should be str
    """
    
    if not isinstance(subject_idx, str):
        raise ValueError('subject_dix must be str')
        
    if not isinstance(run_idx, str):
        raise ValueError('run_idx must be str')
    
    subject_data_folder = data_folder + 'sub-' + subject_idx + '/ses-movie/meg'
    fname = 'sub-' + subject_idx + '_ses-movie_task-movie_run-' + run_idx + '_meg.ds'
    raw_data_path = pjoin(subject_data_folder, fname)
    raw_data = mne.io.read_raw_ctf(raw_data_path, preload='True')
    
    print('total channels number is {}'.format(len(raw_data.info['chs'])))
    print('sample frequency is {} Hz'.format(raw_data.info['sfreq']))

    return raw_data

#sub_idx = '02'
#run_idx = '01'
#sub_raw = load_sub_raw_data(subject_idx=sub_idx, run_idx=run_idx)
#sub_raw.plot(title = 'raw data')


#%%
#marking bad channels
def detect_bad_channel(sub_data, subject_idx='01', run_idx='01'):
    if not sub_data.info['bads']:
        print('sub {} has no bad channels at run {}'.format(subject_idx, run_idx))
    else:
        print('bad channels of sub {} in run {} are {}'.format(subject_idx, run_idx, sub_data.info['bads']))
        
#%% artifacts remove      

#ICA
def arti_detec(data, n_components, save_fig, save_pth, data_info):
    """
    use ICA to detect artifacts.
    Display 3 figures: ICA sources, full duration ICA sources, ICA components topomap
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
    
    ica = mne.preprocessing.ICA(n_components = n_components, random_state=0)
    ica.fit(raw)
    ica.detect_artifacts(raw)
    ica.plot_sources(filter_raw)
    if save_fig:
        plt.savefig(pjoin(save_pth, 'sub{}_run{}_ica_sources'.format(data_info[0], data_info[1])))
    ica.plot_components()
    if save_fig:
        plt.savefig(pjoin(save_pth, 'sub{}_run{}_ica_components'.format(data_info[0], data_info[1])))
    ica.plot_sources(raw, start=0, stop=raw.times[-1])
    if save_fig:
        plt.savefig(pjoin(save_pth, 'sub{}_run{}_icasources_fulldur'.format(data_info[0], data_info[1])))
        
    return ica

#%% artifacts detecttion for all subject
sub_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
run_list = ['{0:0>2d}'.format(i) for i in np.arange(1, 9)]
for sub_idx in sub_list:
    if sub_idx == '01':
        run_ls = run_list.append('09')
    else:
        run_ls = run_list
    for run_idx in run_ls:
        
        cwd = os.getcwd()
        img_savepth = pjoin(cwd, 'ICA_images')
        
        sub_raw = load_sub_raw_data(subject_idx=sub_idx, run_idx=run_idx)
        # detect bad channels
        detect_bad_channel(sub_raw, subject_idx=sub_idx, run_idx=run_idx)
        filter_raw = sub_raw.copy()
        # 1Hz high-pass
        filter_raw.load_data().filter(l_freq=1, h_freq=None)
        filter_ica = arti_detec(filter_raw, n_components=20, save_fig=True, save_pth=img_savepth, data_info=[sub_idx, run_idx])
        filter_ica.save(pjoin(cwd, 'ICA_artifacts', 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)))
   
#%% creat artifact dictionary

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
#%% reconstruct raw data

sub_list = np.arange(1,12)
run_list = np.arange(1,9)

for sub in sub_list:
    for run in run_list:
        if sub < 10:
            sub_idx = '0' + str(sub)
        else:
            sub_idx = str(sub)
        
        run_idx = '0' + str(run)
          
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
#        ica = mne.preprocessing.ICA(n_components = 20, random_state=0)
#        ica.fit(recons_raw)
#        ica.plot_components()
#        plt.close('all')

#%% change preprocessed data to bids format
    
bids_root = pjoin(os.getcwd(), 'preprocessed_data')
task = 'movie'
ses = 'movie'
sh.rmtree(bids_root, ignore_errors=True)

sub_list = np.arange(1,12)

for sub in sub_list:
    if sub == 1:
        run_list = np.arange(1,10)
    else:
        run_list = np.arange(1,9)
        
    for run in run_list:
        if sub < 10:
            sub_idx = '0' + str(sub)
        else:
            sub_idx = str(sub)
        
        run_idx = '0' + str(run)
        
        path = path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/preproc_data/sub-{}/ses-movie/meg'.format(sub_idx)
        fname = 'sub-{}_ses-movie_task-movie_run-{}_desc-preproc_meg.fif'.format(sub_idx, run_idx)
        raw = mne.io.read_raw(pjoin(path, fname))
        
#        sub_ann = raw.annotations
#        i = 0
#        bad_seg_idx = []
#        for item in sub_ann.__iter__():
#            if item['description'] == 'bad segment':
#                bad_seg_idx.append(i)
#            i = i + 1
#        bad_annot = sub_ann.__getitem__(bad_seg_idx)

        raw.info['line_freq'] = 50
        raw.set_annotations(None)
        events = mne.find_events(raw, stim_channel='UPPT001', min_duration=2/raw.info['sfreq'])
        events = mne.merge_events(events, list(np.arange(1, 92)), 1)
        events_id = {'beginning': 255, 'watching': 1}
        bids_pth = BIDSPath(subject=sub_idx, session=ses, task=task, run=int(run), processing = 'preproc', root=bids_root)
        write_raw_bids(raw, bids_pth, events_data=events, event_id=events_id, overwrite=True)
        
#        sub_path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/preprocessed_test/sub-{}/ses-movie/meg'.format(sub_idx)
#        bad_annot.save(pjoin(sub_path, 'sub-08_ses-movie_task-movie_run-{}_annot.fif'.format(run_idx)))
        
#%% derivatives manipulation

ori_path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/ICA_artifacts'
deri_path = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/preproc_data/derivatives'

sub_list = np.arange(1,12)

for sub in sub_list:
    if sub == 1:
        run_list = np.arange(1,10)
    else:
        run_list = np.arange(1,9)
        
    if sub < 10:
        sub_idx = '0' + str(sub)
    else:
        sub_idx = str(sub)
    
    os.mkdir(pjoin(deri_path, 'sub-{}'.format(sub_idx)))
    os.mkdir(pjoin(deri_path, 'sub-{}'.format(sub_idx), 'ses-movie'))
    sub_ica_path = pjoin(deri_path, 'sub-{}'.format(sub_idx), 'ses-movie', 'ICA')
    os.mkdir(sub_ica_path)
        
    for run in run_list:
        run_idx = '0' + str(run)
        
        sh.copy(pjoin(ori_path, 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)), sub_ica_path)
        os.rename(pjoin(sub_ica_path, 'sub{}_run{}_ica.fif.gz'.format(sub_idx, run_idx)), pjoin(sub_ica_path, 'sub-{}_ses-movie_task-movie_run-{}_ica.fif.gz'.format(sub_idx, run_idx)))

#%% rename + desc-
#sub_list = np.arange(1,12)
#
#for sub in sub_list:
#    if sub == 1:
#        run_list = np.arange(1,10)
#    else:
#        run_list = np.arange(1,9)
#        
#    if sub < 10:
#        sub_idx = '0' + str(sub)
#    else:
#        sub_idx = str(sub)
#    
#    fpath = pjoin(bids_root, 'sub-{}'.format(sub_idx), 'ses-movie', 'meg')
#    file_list = os.listdir(fpath)    
#    file_list.sort()
#    os.rename(pjoin(fpath, file_list[0]), pjoin(fpath, 'sub-{}_ses-movie_desc-preproc_coordsystem.json'.format(sub_idx)))
#    file_list.pop(0)
#    
#    for fname in file_list:
#        fname_f = fname_f = fname[0:fname.find('run')+6]
#        fname_l = fname[fname.find('run')+6:-1] + fname[-1]
#        new_name = fname_f + '_desc-preproc' + fname_l
#        
#        os.rename(pjoin(fpath, fname), pjoin(fpath, new_name))
     