#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:22:02 2019

@author: liuxingyu
"""

import os
import numpy as np
import pandas as pd
import subprocess
import glob
import mne
from mne_bids import BIDSPath, write_raw_bids

#%% bash script to reorganize files from date-wise to sub-wise

# =============================================================================
# raw_root=/nfs/e5/studyforrest/forrest_movie_meg/raw_data
# raw_subwise=/nfs/e5/studyforrest/forrest_movie_meg/raw_data_subwise
# 
# # reorganize meg data
# for i in $(ls ${raw_root}/2019* -d) ; do cd $i; for j in $(ls S* -d); do mkdir ${raw_subwise}/sub-${j: 1: 2}-raw ; ln -s $i/$j ${raw_subwise}/sub-${j: 1: 2}-raw/ ; done ; done
# # reorganize mri data
# for i in $(ls ${raw_root}/2019* -d) ; do cd ${i}/MRI; for j in $(ls *S* -d); do mkdir ${raw_subwise}/sub-${j: 0-2: 2}-raw ; ln -s $i/MRI/$j ${raw_subwise}/sub-${j: 0-2: 2}-raw/MRI ; done ; done
# # reorganize emptyroom meg
# mkdir ${raw_subwise}/sub-emptyroom
# for i in $(ls ${raw_root}/2019* -d) ; do cd $i; for j in $(ls *Noise* -d); do ln -s $i/$j ${raw_subwise}/sub-emptyroom/ ; done ; done
# =============================================================================

#%% convert raw data to bids-format
raw_subwise = '/nfs/e5/studyforrest/forrest_movie_meg/raw_data_subwise'
bids_dir = '/nfs/e5/studyforrest/forrest_movie_meg'

#%% data of real participant
sublist = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
for subid in sublist:
    sub_raw_dir = os.path.join(raw_subwise, 'sub-{0}-raw'.format(subid))
    os.chdir(sub_raw_dir)
    
    # MRI
    if len(glob.glob(os.path.join(sub_raw_dir, '*t1*.nii.gz'))) == 0:
        # convert to dicom
        cmd_dcm2nii = 'dcm2niix -o {0} -b y {0}/MRI'.format(sub_raw_dir)
        subprocess.call(cmd_dcm2nii, shell=True)
        subprocess.call('rm {0}/*localizer*'.format(sub_raw_dir), shell=True)    
   
    # rename T1w
    t1 = glob.glob(os.path.join(sub_raw_dir, '*t1*.nii.gz'))
    t1_bids_name = 'sub-{0}_ses-movie_T1w.nii.gz'.format(subid)
    subprocess.call('mv {0} {1}/{2}'.format(t1[0], sub_raw_dir, t1_bids_name), shell=True)
    subprocess.call('mv {0} {1}/{2}'.format(t1[0].replace('nii.gz', 'json'), sub_raw_dir, 
                                            t1_bids_name.replace('nii.gz', 'json')), shell=True)     
    # deface for t1w
    anat_dir = os.path.join(bids_dir, 'sub-{0}'.format(subid), 'ses-movie', 'anat')
    if os.path.exists(anat_dir) is False: os.makedirs(anat_dir)
    subprocess.call('cp {0}/{1} {2}/{1}'.format(sub_raw_dir,t1_bids_name.replace('nii.gz', 'json'), anat_dir), shell=True)
    subprocess.call('pydeface {0}/{1} --outfile {2}/{1}'.format(sub_raw_dir, t1_bids_name, anat_dir), shell=True)   
    
    print('sub-{0}: MRI convertion done'.format(subid))
    
    # MEG
    meg_files = glob.glob('S*.ds')
    meg_files.sort()
    
    for file in meg_files:    
        runid = file[-5:-3]
        raw = mne.io.read_raw_ctf(file)
        raw.info['line_freq'] = 50
        
        # event data
        raw.set_annotations(None)
        events = mne.find_events(raw, stim_channel='UPPT001', min_duration=2/raw.info['sfreq']) 

        # repair annotations
        events_id = {'watch_{0}_{1}s'.format((i-1)*10, i*10) : int(i) for i in np.unique(events[:,-1]) if i != 255}
        events_id['beginning'] = 255
        annot_new = mne.annotations_from_events(events, raw.info['sfreq'], event_desc={v: k for k, v in events_id.items()}, 
                                                first_samp=raw.first_samp)
        raw.set_annotations(annot_new)
             
        # convert data to bids format
        bids_path = BIDSPath(subject=subid, session='movie', task='movie', 
                             run=runid, datatype='meg', root=bids_dir)
        write_raw_bids(raw, bids_path=bids_path, overwrite=True)
        
        # repair events files
        events = events.copy()
        events[:, 0] -= raw.first_samp
        events_df = pd.DataFrame(np.c_[events[:, 0]/raw.info['sfreq'], events[:, 1], annot_new.description, events[:, 2], events[:, 0]], 
                                 columns=['onset','duration','trial_type','value','sample'])
        events_df.to_csv(str(bids_path.copy().update(suffix='events', extension='.tsv').fpath), 
                         sep='\t', columns=['onset','duration','trial_type','value','sample'], index=False)
    
    print('sub-{0}: MEG convertion done'.format(subid))
    
    
#%% data of empty room
filelist = os.listdir(os.path.join(raw_subwise, 'sub-emptyroom'))
seslist = [f.split('_')[-2] for f in filelist]

for ses in seslist:
    meg_file = glob.glob(os.path.join(raw_subwise, 'sub-emptyroom','*{0}*.ds'.format(ses)))[0]

    raw = mne.io.read_raw_ctf(meg_file)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject='emptyroom', session=ses, task='noise', datatype='meg', root=bids_dir)
    
    write_raw_bids(raw, bids_path=bids_path, overwrite=True)



