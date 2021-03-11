#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 22:46:16 2021

Technical validation for Forrest Gump MEG data
Including: head movement visualization, ISC maps, SNR computation
@author: daiyuxuan
"""
#%%
import numpy as np
from os.path import join as pjoin
from os.path import isdir
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import mne_bids
import mne
from mne_bids import write_raw_bids, BIDSPath
from scipy import stats
import re
from scipy import signal
import pandas as pd
from scipy import signal, fftpack

#%% define functions for head movement validation

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

def extract_hpi(raw_data):
    '''
    Extract hpi data from mne raw object.
    Return: hpi_data:[(channel_name1, value1),...] 
    '''
    picks = mne.pick_channels_regexp(raw_data.ch_names, regexp='HLC00[123][1238]...')
    hpi_data = raw_data.get_data(picks=picks)
    
    hlc_ch = []
    for i in np.arange(12):
        hlc_ch.append(raw_data.ch_names[picks[i]])
    hpi = {}
    for i in np.arange(12):
        hpi[hlc_ch[i]] = hpi_data[i]
    hpi = sorted(hpi.items(), key=lambda d: d[0], reverse=False)
    
    return hpi

def plot_sub_hmv_hist(hpi, save, save_pth, fig_info):
    """
    use hpi channels data to plot histgram of head movement
    Parameters:
        hpi: list of hpi channels data, include channel name and data
        save: bool 
        fig_info: [sub_idx, run_idx]
    """
    coord = ['nasion', 'lpa', 'rpa']
    color = ['lightskyblue', 'lightgreen', 'pink']
    hpi_de = []
    for chn, val in hpi:
        hpi_de.append((chn, 1000*(val-val[0])))

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle('Head Movement')
    for j in np.arange(3):
        mv = [np.sqrt(hpi_de[j*4][1][n]**2 + hpi_de[j*4+1][1][n]**2 + hpi_de[j*4+2][1][n]**2) for n in np.arange(len(hpi_de[j-1][1]))]
        x, bins, p = axes[j].hist(mv, alpha=0.4, density=True, bins=20, label=coord[j], color=color[j])
        height = []
        for item in p:
            height.append(item.get_height()/sum(x))
            item.set_height(item.get_height()/sum(x))
        axes[j].set_ylim(top=np.ceil(max(height)*10)/10)
        axes[j].legend()
        axes[j].set_xlabel('mm')
        axes[j].set_ylabel('normalized histgram')
        
    if save:
        plt.savefig(pjoin(save_pth, f'sub-{fig_info[0]}_run-{fig_info[1]}_deriv.jpeg'),dpi=600)
        plt.close()
        
#%% define variables
sub_list = ['{0:0>2d}'.format(sub) for sub in np.arange(1,12)]
run_list = ['{0:0>2d}'.format(run) for run in np.arange(1,9)]
coord = ['nasion', 'lpa', 'rpa']
# change path according to personal usage
data_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/output_data'
bids_root = '/nfs/e2/workingshop/daiyuxuan/MEG-paper/preproc_data'
fig_save_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/head_mv_fig'

#%% extract and load data for visualization
# extract data
# hpi_dict = {}
# for sub in sub_list:
#     hpi_dict[sub] = []
#     if sub == '01':
#         run_ls = run_list + ['09']
#     else:
#         run_ls = run_list
#     for run in run_ls:
#         sub_path = BIDSPath(subject=sub, run=int(run), task='movie', session='movie', processing='preproc', root=bids_root)
#         raw = mne_bids.read_raw_bids(sub_path)
#         raw_data = raw.copy().crop(tmin=raw.annotations.onset[0], tmax=raw.annotations.onset[-1])
#         hpi = extract_hpi(raw_data)
#         hpi_dict[sub].append(hpi)

# save hpi data
# for sub in sub_list[1:]:
#     hpi_dict[sub].append(np.nan)
# df = pd.DataFrame(hpi_dict, columns=sub_list, index=run_list+['09'])
# df.to_pickle(pjoin(data_pth, 'hpi_data.pickle'))

# load hpi data
hpi_data = pd.read_pickle(pjoin(data_pth, 'hpi_data.pickle'))

#%% head movement visualization

# head movement visualization for each subject
for sub_idx in sub_list:
    if sub_idx == '01':
        run_ls = run_list + ['09']
    else:
        run_ls = run_list
    for run_idx in run_ls:
        plot_sub_hmv_hist(hpi_data[sub_idx][run_idx], save=True, save_pth=fig_save_pth, fig_info=[sub_idx, run_idx])

# head movement visualization for all data
head_mv = []
coord = ['nasion', 'lpa', 'rpa']
for j, coil in enumerate(coord):
    coil_mv = np.array([])
    for sub in sub_list:
        for run in run_list:
            hpi = hpi_data[sub][run]
            hpi_de = []
            for i, (chn, val) in enumerate(hpi):
                hpi_de.append((chn, 1000*(val-val[0])))
            mv = [np.sqrt(hpi_de[j*4][1][n]**2 + hpi_de[j*4+1][1][n]**2 + hpi_de[j*4+2][1][n]**2) for n in np.arange(len(hpi_de[j-1][1]))]
            if np.where(np.array(mv)>10)[0].size > 0:
                print(f'lager head movement: sub-{sub} run-{run} coil:{coil}')
            coil_mv = np.append(coil_mv, np.array(mv))
    head_mv.append((coil, coil_mv))

color = ['lightskyblue', 'lightgreen', 'pink']
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle('Head Movement')
for j in np.arange(3):
    x, bins, p = axes[j].hist(mv, alpha=0.4, density=True, bins=50, label=coord[j], color=color[j])
    height = []
    for item in p:
        height.append(item.get_height()/sum(x))
        item.set_height(item.get_height()/sum(x))
    axes[j].set_ylim(top=np.ceil(max(height)*10)/10)
    axes[j].legend()
    axes[j].set_xlabel('mm')
    axes[j].set_ylabel('normalized histgram')

# plt.savefig(pjoin(save_pth, f'sub-total_run-total_deriv.jpeg'),dpi=600)

#%% define functions for SNR computation
def fALFF(data, fs, f_range='band'):
    """

    Parameters
    ----------
        data: shape = [n_samples, n_features].
              for meg data: shape = [n_channels, n_samples]
    """
    
    # remove linear trend
    data_detrend = signal.detrend(data, axis=-1)
    
    if f_range == 'band':
        # convert to frequency domain        
        freqs, psd = signal.welch(data_detrend, fs=fs)
        
        band = [[1, 4], [4, 8], [8, 13], [13, 30], [30, 100]]
        # "delta": 1-4Hz
        # "theta": 4-8Hz
        # "alpha": 8-13Hz
        # "beta": 13-30Hz
        # "gamma": 30-100Hz
            
#            band = np.linspace(0,0.7,10)
#            band = list(zip(band[:-1],band[1:]))
            
        falff = [np.sum(psd[:, (freqs>i[0]) * (freqs<i[1])], axis=-1) / np.sum(psd[:, freqs<0.5*fs], axis=-1) for i in band]
        falff = np.asarray(falff).T
    
    return falff

def load_post_megdata(sub='02', run=1):
    bids_root = '/nfs/e5/studyforrest/forrest_movie_meg/preproc_data'
    
    sub_path = BIDSPath(subject=sub, run=int(run), task='movie', session='movie', processing='preproc', root=bids_root)
    raw_sub = mne_bids.read_raw_bids(sub_path)

    ch_name_picks = mne.pick_channels_regexp(raw_sub.ch_names, regexp='M[LRZ]...-4503')
    type_picks = mne.pick_types(raw_sub.info, meg=True)
    sub_picks= np.intersect1d(ch_name_picks, type_picks)
    sub_raw_data = raw_sub.get_data(picks=sub_picks)
    events_sub = mne.events_from_annotations(raw_sub)
    sample_sub = events_sub[0][:,0]
    
    sub_data = sub_raw_data.take(sample_sub, axis=1)
    return sub_data

def load_pre_megdata(sub='02', run=1):
    bids_root = '/nfs/e5/studyforrest/forrest_movie_meg/preproc_data'
    
    sub_path = BIDSPath(subject=sub, run=int(run), task='movie', session='movie', processing='preproc', root=bids_root)
    raw_sub_ev = mne_bids.read_raw_bids(sub_path)
    raw_sub = load_sub_raw_data(subject_idx=sub, run_idx='0'+str(run))
    ch_name_picks = mne.pick_channels_regexp(raw_sub.ch_names, regexp='M[LRZ]...-4503')
    type_picks = mne.pick_types(raw_sub.info, meg=True)
    sub_picks= np.intersect1d(ch_name_picks, type_picks)
    sub_raw_data = raw_sub.get_data(picks=sub_picks)
    events_sub = mne.events_from_annotations(raw_sub_ev)
    sample_sub = events_sub[0][:,0]
    
    sub_data = sub_raw_data.take(sample_sub, axis=1)
    return sub_data

def plot_topomap(val):
    """
    input params: val: [n_chn(272), 1]
    """
    raw_sub = load_sub_raw_data(subject_idx='01', run_idx='01')
    meg_layout = mne.channels.find_layout(raw_sub.info, ch_type='meg')
    intercorr_chn = []
    ch_name_picks = mne.pick_channels_regexp(raw_sub.ch_names, regexp='M[LRZ]...-4503')
    type_picks = mne.pick_types(raw_sub.info, meg=True)
    picks= np.intersect1d(ch_name_picks, type_picks)
    for idx in picks:
        intercorr_chn.append(raw_sub.ch_names[idx][:5])
    exclude_list = [x for x in meg_layout.names if x not in intercorr_chn]
    meg_layout = mne.channels.find_layout(raw_sub.info, ch_type='meg', exclude=exclude_list)

    pos = meg_layout.pos[:,:2]
    x_min = np.min(pos[:,0])
    x_max = np.max(pos[:,0])
    y_min = np.min(pos[:,1])
    y_max = np.max(pos[:,1])
    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2
    layout_pos = np.zeros(pos.shape)
    for i, coor in enumerate(pos):
        layout_pos[i,0] = coor[0] - center_x
        layout_pos[i,1] = coor[1] - center_y
        
    plt.figure()
    mne.viz.plot_topomap(val, layout_pos, cmap='viridis', sphere=np.array([0,0,0,0.5]))
    norm = colors.Normalize(vmin=np.min(val), vmax=np.max(val))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'))
    
#%% define variables
sub_list = ['{0:0>2d}'.format(sub) for sub in np.arange(1,12)]
run_list = ['{0:0>2d}'.format(run) for run in np.arange(1,9)]
band = ['delta', 'theta', 'alpha', 'beta', 'gamma']
# change path according to personal usage
data_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/output_data'
bids_root = '/nfs/e2/workingshop/daiyuxuan/MEG-paper/preproc_data'
# fig_save_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/head_mv_fig' 
  
#%% compute SNR

# compute falff
# post_falff_data = {}
# pre_falff_data ={}
# for sub in sub_list:
#     post_falff_data[sub] = []
#     pre_falff_data[sub] = []
#     if sub == '01':
#         run_ls = run_list + ['09']
#     else:
#         run_ls = run_list
#     for run in run_ls:
#         post_raw_data = load_post_megdata(sub=sub, run=int(run))
#         post_falff = fALFF(post_raw_data, fs=600)
#         pre_raw_data = load_pre_megdata(sub=sub, run=int(run))
#         pre_falff = fALFF(pre_raw_data, fs=600)
#         post_falff_data[sub].append(post_falff)
#         pre_falff_data[sub].append(pre_falff)

# # save falff
# for sub in sub_list[1:]:
#     pre_falff_data[sub].append(np.nan)
#     post_falff_data[sub].append(np.nan)
# pre_df = pd.DataFrame(pre_falff_data, columns=sub_list, index=run_list+['09'])
# post_df = pd.DataFrame(post_falff_data, columns=sub_list, index=run_list+['09'])
# pre_df.to_pickle(pjoin(data_pth, 'pre_falff_data.pickle'))
# post_df.to_pickle(pjoin(data_pth, 'post_falff_data.pickle'))

# load falff
pre_df = pd.read_pickle(pjoin(data_pth, 'pre_falff_data.pickle'))
post_df = pd.read_pickle(pjoin(data_pth, 'post_falff_data.pickle'))

#%% boxplot of falff

for sub in sub_list:
    if sub == '01':
        run_ls = run_list + ['09']
    else:
        run_ls = run_list
    
    for run in run_ls:
        pre_falff = pre_df[sub][run]
        post_falff = post_df[sub][run]
        
        if run == '01':
            run_pre_falff = pre_falff
            run_post_falff = post_falff
        else:
            run_pre_falff = np.vstack((run_pre_falff, pre_falff))
            run_post_falff = np.vstack((run_post_falff, post_falff))
    
    if sub == '01':
        mean_pre_falff = np.mean(run_pre_falff, axis=0)
        mean_post_falff = np.mean(run_post_falff, axis=0)
    else:
        mean_pre_falff = np.vstack((mean_pre_falff, np.mean(run_pre_falff, axis=0)))
        mean_post_falff = np.vstack((mean_post_falff, np.mean(run_post_falff, axis=0)))

fig, ax1 = plt.subplots()
plt.suptitle('Pre & post fALFF')
bp1 = ax1.boxplot(mean_pre_falff, patch_artist=True, whis=0.95, labels=band)
for patch in bp1['boxes']:
    patch.set_facecolor('lightgreen')
ax1.set_ylabel('pre falff')

ax2 = ax1.twinx()
bp2 = ax2.boxplot(mean_post_falff, patch_artist=True, whis=0.95, labels=band)
for patch in bp2['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_ylabel('post falff')
ax2.set_ylim(top=0.35)
    
ax1.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Pre', 'Post'] ,loc='best')

#%% topographic brain map of falff
for i, band_n in enumerate(band):
    plot_topomap(pre_df['01']['01'][:,i])
    plt.title(f'pre {band_n}')
    plot_topomap(post_df['01']['01'][:,i])
    plt.title(f'post {band_n}')

#%% define functions
def band_split(raw, band_name='gamma'):
    """
    band_name: "delta": 1-4Hz
               "theta": 4-8Hz
               "alpha": 8-13Hz
               "beta": 13-30Hz
               "gamma": 30-100Hz
    """
    
    if band_name == 'delta':
        raw_band = raw.copy().load_data().filter(l_freq=1, h_freq=4)
    elif band_name == 'theta':
        raw_band = raw.copy().load_data().filter(l_freq=4, h_freq=8)
    elif band_name == 'alpha':
        raw_band = raw.copy().load_data().filter(l_freq=8, h_freq=13)
    elif band_name == 'beta':
        raw_band = raw.copy().load_data().filter(l_freq=13, h_freq=30)
    elif band_name == 'gamma':
        raw_band = raw.copy().load_data().filter(l_freq=30, h_freq=100)
    
    return raw_band

def compute_band_intercorr(raw_sub1, raw_sub2, events_sub1, events_sub2, band_name):

    ch_name_picks1 = mne.pick_channels_regexp(raw_sub1.ch_names, regexp='M[LRZ]...-4503')
    type_picks1 = mne.pick_types(raw_sub1.info, meg=True)
    sub1_picks= np.intersect1d(ch_name_picks1, type_picks1)
    ch_name_picks2 = mne.pick_channels_regexp(raw_sub2.ch_names, regexp='M[LRZ]...-4503')
    type_picks2 = mne.pick_types(raw_sub2.info, meg=True)
    sub2_picks= np.intersect1d(ch_name_picks2, type_picks2)
    if len(sub1_picks) == len(sub2_picks):
        if (sub1_picks == sub2_picks).all():
            picks = sub1_picks
        else:
            picks = np.intersect1d(sub1_picks, sub2_picks)
        bad_idx = []
    else:
        picks = sub1_picks if len(sub1_picks) > len(sub2_picks) else sub2_picks
        bad_picks = np.union1d(np.setdiff1d(sub1_picks, sub2_picks), np.setdiff1d(sub2_picks, sub1_picks))
    
        bad_idx = []
        for chn in bad_picks:
            bad_idx.append(np.where(picks == chn)[0][0])
            
    raw_sub1_theta = band_split(raw_sub1, band_name=band_name)
    raw_sub2_theta = band_split(raw_sub2, band_name=band_name)
    sub1_band_data = raw_sub1_theta.get_data(picks=picks)
    sub2_band_data = raw_sub2_theta.get_data(picks=picks)
    
    sub1_envlope = np.abs(signal.hilbert(sub1_band_data))
    sub2_envlope = np.abs(signal.hilbert(sub2_band_data))
    
    if len(bad_idx) != 0:
        for idx in bad_idx:
            sub1_envlope[idx,:] = 0
            sub2_envlope[idx,:] = 0
    
    #downsampling
    sample_sub1 = events_sub1[0][1:-1:25,0]
    sample_sub2 = events_sub2[0][1:-1:25,0]
    sub1_band_dsamp = sub1_envlope.take(sample_sub1, axis=1)
    sub2_band_dsamp = sub2_envlope.take(sample_sub2, axis=1)
    
                
    #def hilbert_filter(x, sfreq, order=201):
    #    
    #    co = [2*np.sin(np.pi*n/2)**2/np.pi/n for n in range(1, order+1)]
    #    co1 = [2*np.sin(np.pi*n/2)**2/np.pi/n for n in range(-order, 0)]
    #    co = co1+[0]+ co
    #    # out = signal.filtfilt(b=co, a=1, x=x, padlen=int((order-1)/2))
    #    out = signal.convolve(x, co, mode='same', method='direct')
    #    envolope = np.sqrt(out**2 + x**2)
    #    
    #    return envolope
        
    ch_r = []
    ch_p = []
    for ch_idx in np.arange(sub1_band_dsamp.shape[0]):
        x = sub1_band_dsamp[ch_idx, :]
        y = sub2_band_dsamp[ch_idx, :]
        
        r, p = stats.pearsonr(x, y)
        ch_r.append(r)
        ch_p.append(p)
    
    del raw_sub1, raw_sub2
    
    return ch_r, ch_p

def extract_events(sub_idx, run_idx):
    bids_root = '/nfs/e5/studyforrest/forrest_movie_meg/preproc_data'
    sub_path = BIDSPath(subject=sub_idx, run=int(run_idx), task='movie', session='movie', processing='preproc', root=bids_root)
    sub_raw = mne_bids.read_raw_bids(sub_path)
    events_sub = mne.events_from_annotations(sub_raw)
    
    return events_sub

#%% define variables
bids_root = '/nfs/e5/studyforrest/forrest_movie_meg/preproc_data'
sub_list = np.arange(1,12)
run_list = np.arange(1,9)
save_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/output_data'
band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

#%% compute pre-post preproc ISC
pre_corr_band_split = {}
for band_name in band_names:
    
    pre_corr_band_split[band_name] = np.zeros((len(run_list), len(sub_list), len(sub_list), 272))
            
    for run in run_list:
        if run >= 7:
            sub_ls = np.arange(2,12)
        else:
            sub_ls = sub_list
            
        for sub1 in sub_ls:
            if sub1 < 10:
                sub1_idx = '0' + str(sub1)
            else:
                sub1_idx = str(sub1)
                
            run_idx = '0'+str(run)
            events_sub1 = extract_events(sub1_idx, run_idx)
            raw_sub1 = load_sub_raw_data(subject_idx=sub1_idx, run_idx=run_idx)

            for sub2 in sub_list:
                if sub2 < 10:
                        sub2_idx = '0' + str(sub2)
                else:
                        sub2_idx = str(sub2)
                
                events_sub2 = extract_events(sub2_idx, run_idx)
                raw_sub2 = load_sub_raw_data(subject_idx=sub2_idx, run_idx=run_idx)
                ch_r, ch_p = compute_band_intercorr(raw_sub1, raw_sub2, events_sub1, events_sub2, band_name)
                pre_corr_band_split[band_name][run-1][sub1-1][sub2-1] = pre_corr_band_split[band_name][run-1][sub2-1][sub1-1] = np.array(ch_r)
                # a = np.array(ch_r)
    np.save(pjoin(save_pth, 'pre_corr_band_split_{}'.format(band_name)), pre_corr_band_split[band_name])
np.save(pjoin(save_pth, 'pre_corr_band_split'), pre_corr_band_split)

# post pre-proc
corr_band_split = {}
for band_name in band_names:
    
    corr_band_split[band_name] = np.zeros((len(run_list), len(sub_list), len(sub_list), 272))
    
    for run in run_list:
        if run >= 7:
            sub_ls = np.arange(2, 12)
        else:
            sub_ls = sub_list
        
        for sub1 in sub_ls:
            if sub1 < 10:
                sub1_idx = '0' + str(sub1)
            else:
                sub1_idx = str(sub1)
            
            sub1_path = BIDSPath(subject=sub1_idx, run=int(run), task='movie', session='movie', processing='preproc', root=bids_root)
            raw_sub1 = mne_bids.read_raw_bids(sub1_path)
            events_sub1 = mne.events_from_annotations(raw_sub1)
            
            for sub2 in np.arange(sub1+1, 12):
                
                if sub2 < 10:
                    sub2_idx = '0' + str(sub2)
                else:
                    sub2_idx = str(sub2)
                
                sub2_path = BIDSPath(subject=sub2_idx, run=int(run), task='movie', session='movie', processing='preproc', root=bids_root)
                raw_sub2 = mne_bids.read_raw_bids(sub2_path)
                events_sub2 = mne.events_from_annotations(raw_sub2)
                ch_r, ch_p = compute_band_intercorr(raw_sub1, raw_sub2, events_sub1, events_sub2, band_name)
                corr_band_split[band_name][run-1][sub1-1][sub2-1] = corr_band_split[band_name][run-1][sub2-1][sub1-1] = np.array(ch_r)
np.save(pjoin(save_pth, 'corr_band_split'), corr_band_split)

#load data
corr_band_split = np.load(pjoin(save_pth, 'corr_band_split.npy'), allow_pickle=True)
corr_band_split = corr_band_split.all()

#%% plot ISC topomap
tmp_mean = np.mean(corr_band_split['beta'], axis=1)
mean_run_corr = np.mean(tmp_mean, axis=1)
run_idx = 4
plot_topomap(mean_run_corr[run_idx-1])
plt.title('Post pre-proc Beta band')

#%% define functions for compute interlobe correlation

def compute_intercorr(sub_idx, run, bids_root):
    
    sub_path = BIDSPath(subject=sub_idx, run=int(run), task='movie', session='movie', processing='preproc', root=bids_root)
    raw_sub = mne_bids.read_raw_bids(sub_path)
    ref_path = BIDSPath(subject='02', run=int('01'), task='movie', session='movie', processing='preproc', root=bids_root)
    raw_ref = mne_bids.read_raw_bids(ref_path)
    # A reference is necessary for detecting bad channel, I simply use the original raw data for the conveinence.
    
    ch_name_picks = mne.pick_channels_regexp(raw_sub.ch_names, regexp='M[LRZ]...-4503')
    type_picks = mne.pick_types(raw_sub.info, meg=True)
    sub_picks= np.intersect1d(ch_name_picks, type_picks)
    ch_name_picks_ref = mne.pick_channels_regexp(raw_ref.ch_names, regexp='M[LRZ]...-4503')
    type_picks_ref = mne.pick_types(raw_ref.info, meg=True)
    ref_picks= np.intersect1d(ch_name_picks_ref, type_picks_ref)
    if len(sub_picks) == len(ref_picks):
        if (sub_picks == ref_picks).all():
            picks = sub_picks
        else:
            picks = np.intersect1d(sub_picks, ref_picks)
        bad_idx = []
    else:
        picks = sub_picks if len(sub_picks) > len(ref_picks) else ref_picks
        bad_picks = np.union1d(np.setdiff1d(sub_picks, ref_picks), np.setdiff1d(ref_picks, sub_picks))
    
        bad_idx = []
        for chn in bad_picks:
            bad_idx.append(np.where(picks == chn)[0][0])

    sub_data = raw_sub.get_data(picks=picks)
    sub_envlope = np.abs(signal.hilbert(sub_data))
    
    if len(bad_idx) != 0:
        for idx in bad_idx:
            sub_envlope[idx,:] = 0
    
    #downsampling
    events_sub = mne.events_from_annotations(raw_sub)
    sample_sub = events_sub[0][1:-1:25,0]
    sub_dsamp = sub_envlope.take(sample_sub, axis=1)
    
    intra_lobe_corr = np.zeros((len(picks), len(picks)))

    for i in np.arange(len(picks)):
        sub_data1 = sub_dsamp.take(i, axis=0)
        
        for j in np.arange(len(picks)):
            sub_data2 = sub_dsamp.take(j, axis=0)
        
            r, p = stats.pearsonr(sub_data1, sub_data2)
            
            intra_lobe_corr[i][j] = intra_lobe_corr[j][i] = r
    
    ch_name = []
    for idx in picks:
        ch_name.append(raw_sub.ch_names[idx])
    labels = ['LC', 'LF', 'LO', 'LP', 'LT', 'RC', 'RF', 'RO', 'RP', 'RT']
    ch_labels = dict()
    for label in labels:
        reg = re.compile('M'+label+'.*')
        tmp_chs = list(filter(reg.match, ch_name))
        ch_labels[label] = [ch_name.index(x) for x in tmp_chs]
    
    mean_lobe_corr = np.zeros((len(labels), len(labels)))
    
    for i, (roi_x, ch_idx_x) in enumerate(ch_labels.items()):
        for j, (roi_y, ch_idx_y) in enumerate(ch_labels.items()):
            tmp = intra_lobe_corr.take(ch_idx_x, axis=0)
            roi_val = tmp.take(ch_idx_y, axis=1)
            mean_lobe_corr[i][j] = np.mean(roi_val)
            
    return mean_lobe_corr

def plot_lobecorr_rdm(lobe_corr, fig, ax, label):
    
    c = ax.pcolor(lobe_corr)
    fig.colorbar(c, ax=ax)
    
    ax.set_xticks(np.arange(0.5, len(label), 1))
    ax.set_yticks(np.arange(0.5, len(label), 1))
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)
    ax.set_aspect('equal', adjustable='box')

def plot_lobecorr_box(lobe_corr, ax):
    
    same_hemi_same_lobe = []
    for i in np.arange(lobe_corr.shape[0]):
        same_hemi_same_lobe.append(lobe_corr[i, i])
        
    diff_hemi_same_lobe = []
    for i in np.arange(5):
        diff_hemi_same_lobe.append(lobe_corr[i, 5+i])
        diff_hemi_same_lobe.append(lobe_corr[5+i, i])
        
    same_hemi_diff_lobe = np.array([])
    for i in [0,5]:
        same_hemi_diff_lobe = np.append(same_hemi_diff_lobe, (lobe_corr[i:i+5, i:i+5].ravel()[np.flatnonzero(np.tril(lobe_corr[i:i+5, i:i+5], k=-1))]))
        same_hemi_diff_lobe = np.append(same_hemi_diff_lobe,(lobe_corr[i:i+5, i:i+5].ravel()[np.flatnonzero(np.triu(lobe_corr[i:i+5, i:i+5], k=1))]))
    diff_hemi_diff_lobe = np.array([])
    for i in [0,5]:
        diff_hemi_diff_lobe = np.append(diff_hemi_diff_lobe, (lobe_corr[i:i+5, 5-i:10-i].ravel()[np.flatnonzero(np.tril(lobe_corr[i:i+5, 5-i:10-i], k=-1))]))
        diff_hemi_diff_lobe = np.append(diff_hemi_diff_lobe,(lobe_corr[i:i+5, 5-i:10-i].ravel()[np.flatnonzero(np.triu(lobe_corr[i:i+5, 5-i:10-i], k=1))]))
    
    ax.boxplot(same_hemi_same_lobe, positions=[1], whis=0.95)
    ax.boxplot(same_hemi_diff_lobe, positions=[3], whis=0.95)
    ax.boxplot(diff_hemi_same_lobe, positions=[5], whis=0.95)
    ax.boxplot(diff_hemi_diff_lobe, positions=[7], whis=0.95)
    
    ax.set_xticks([1,3,5,7])
    ax.set_xticklabels(['s_hem_s_lobe', 's_hem_d_lobe', 'd_hem_s_lobe', 'd_hem_d_lobe'])
    ax.set_ylabel('corr coef')
    
    x = [1, 3, 5, 7]
    y = [np.array(same_hemi_same_lobe), same_hemi_diff_lobe, np.array(diff_hemi_same_lobe), diff_hemi_diff_lobe]
    for xs, val in zip(x, y):
        xx = np.ones(val.shape)*xs
        ax.scatter(xx, val, alpha=0.4)
        
#%% define variables
bids_root = '/nfs/e5/studyforrest/forrest_movie_meg/preproc_data'
labels = ['LC', 'LF', 'LO', 'LP', 'LT', 'RC', 'RF', 'RO', 'RP', 'RT']
sub_list = ['{0:0>2d}'.format(sub) for sub in np.arange(1,12)]
run_list = ['{0:0>2d}'.format(run) for run in np.arange(1,9)]
data_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MEG-paper/output_data'

#%% compute interlobe corr
# interlobe_corr = {}
# for sub in sub_list:
#     interlobe_corr[sub] = []
#     if sub == '01':
#         run_ls = run_list + ['09']
#     else:
#         run_ls = run_list
#     for run in run_ls:
#         mean_lobe_corr = compute_intercorr(sub, run, bids_root)
#         interlobe_corr[sub].append(mean_lobe_corr)
        
# # save ILC data
# for sub in sub_list[1:]:
#     interlobe_corr[sub].append(np.nan)
# df = pd.DataFrame(interlobe_corr, columns=sub_list, index=run_list+['09'])
# df.to_pickle(pjoin(data_pth, 'interlobe_correlation.pickle'))

# load data
interlobe_corr = pd.read_pickle(pjoin(data_pth, 'interlobe_correlation.pickle'))

#%% visualization for ILC
lobe_corr = np.zeros(interlobe_corr['01']['01'].shape)
for sub in sub_list:
    if sub == '01':
        run_ls = run_list + ['09']
    else:
        run_ls = run_list
    for run in run_ls:
        interlobe_corr[sub][run][np.isnan(interlobe_corr[sub][run])] = 0
        # bad channel will cause NaN corr, replace those with value )
        lobe_corr += interlobe_corr[sub][run]
lobe_corr /= (len(sub_list)*len(run_list)+1)

fig, axes = plt.subplots(1, 2, figsize=(10,4))
plt.suptitle('Interlobe Correlation')
plot_lobecorr_rdm(lobe_corr, fig, axes[0], labels)
plot_lobecorr_box(lobe_corr, axes[1])
