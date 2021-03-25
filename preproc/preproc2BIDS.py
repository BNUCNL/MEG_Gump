#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:22:02 2019

@author: liuxingyu
"""

import os
import numpy as np
import subprocess
import glob
import mne_bids

# %%
parent_dir = '/nfs/e5/studyforrest/forrest_movie_meg'
bids_dir = os.path.join(parent_dir,'gump_meg_bids')
bids_preproc_dir = os.path.join(bids_dir, 'derivatives', 'preproc_meg-mne_mri-fmriprep')

# origianl MEG ad MRI data
raw_data = os.path.join(parent_dir, 'raw_data_bids')
# preproced  MEG data
preproc_meg = os.path.join(parent_dir, 'meg_preproc_data')
# preproced  MRI data
preproc_mri = os.path.join(parent_dir, 'mri_preproc_data')
# freesurfer data
freesurfer_data = os.path.join(preproc_mri, 'sourcedata')

# assosiated files
study_ass_files = ['dataset_description.json',
                   'participants.json',
                   'participants.tsv',
                   'README']
mri_study_ass_files = ['dataset_description.json',
                       'desc-aparcaseg_dseg.tsv',
                       'desc-aseg_dseg.tsv']
ses_ass_files = ['*_ses-movie_scans.tsv']
meg_ass_files = ['*_ses-movie_coordsystem.json',
                 '*_channels.tsv',
                 '*_events.tsv',
                 '*_meg.json']

# %% copy meg associated files to preproced meg
sublist = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
for sub in sublist:
    sub_raw_ses_dir = os.path.join(raw_data, 'sub-{0}'.format(sub), 'ses-movie')
    sub_preproc_meg_ses_dir = sub_raw_ses_dir.replace(raw_data, preproc_meg)
    for f in ses_ass_files:
        subprocess.call('cp {0}/{1} {2}'.format(sub_raw_ses_dir, f, sub_preproc_meg_ses_dir), shell=True)
    
    sub_raw_meg_dir = os.path.join(sub_raw_ses_dir, 'meg')
    sub_preproc_meg_meg_dir = os.path.join(sub_preproc_meg_ses_dir, 'meg')
    for f in meg_ass_files:
        subprocess.call('cp {0}/{1} {2}'.format(sub_raw_meg_dir, f, sub_preproc_meg_meg_dir), shell=True)


# %% move data to bids_dir
# raw data
subprocess.call('mv {0} {1}/{2}'.format(raw_data, bids_dir, 'rawdata'), shell=True)

# freesurfer
subprocess.call('mv {0} {1}/{2}'.format(freesurfer_data, bids_preproc_dir, 'sourcedata'), shell=True)

# preproced meg and mri
sublist = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
for sub in sublist:
    bids_preproc_ses_dir = os.path.join(bids_preproc_dir, 'sub-{0}'.format(sub), 'ses-movie')
    if os.path.exists(bids_preproc_ses_dir) is False: os.makedirs(bids_preproc_ses_dir)
    subprocess.call('mv {0}/sub-{1}/ses-movie/anat {2}/anat'.format(preproc_mri, sub, bids_preproc_ses_dir), shell=True)
    subprocess.call('mv {0}/sub-{1}/ses-movie/meg {2}/meg'.format(preproc_meg, sub, bids_preproc_ses_dir), shell=True)
    
    ses_ass_f = glob.glob(os.path.join(preproc_meg, 'sub-{0}'.format(sub), 'ses-movie', ses_ass_files[0]))[0]
    subprocess.call('mv {0} {1}'.format(ses_ass_f, bids_preproc_ses_dir), shell=True)

# copy study associated files
for f in study_ass_files:
    subprocess.call('cp {0}/{1} {2}'.format(os.path.join(bids_dir, 'rawdata'), f, bids_dir), shell=True)

for f in mri_study_ass_files:
    subprocess.call('cp {0}/{1} {2}'.format(preproc_mri, f, bids_preproc_dir), shell=True)
    
# %%
tree = mne_bids.print_dir_tree(bids_dir, return_str=True)
with open('/nfs/s2/userhome/liuxingyu/workingdir/temp/gump_bids', 'w') as f:
    f.write(tree)
    
