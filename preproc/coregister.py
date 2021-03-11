#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:05:30 2021

@author: liuxingyu & daiyuxuan
"""
import os
import mne
import numpy as np
import subprocess

#%% set working dir
fs_dir = '/nfs/e5/studyforrest/forrest_movie_meg/mri_bids_fmriprep_output/sourcedata/freesurfer_bem'
sublist = ['sub-{0:0>2d}'.format(i) for i in np.arange(1,12,1)]

#%% deidentification and coregister
for sub in sublist:
    
    # deidentification
    mri_dir = os.path.join(fs_dir, sub, 'mri')
    subprocess.call('mri_convert {0}/T1.mgz {0}/T1.nii.gz'.format(mri_dir), shell=True)
    subprocess.call('pydeface {0}/T1.nii.gz --outfile {0}/T1_defaced.nii.gz'.format(mri_dir), shell=True)
    subprocess.call('rm {0}/T1.mgz'.format(mri_dir), shell=True)
    subprocess.call('mri_convert {0}/T1_defaced.nii.gz {0}/T1.mgz'.format(mri_dir), shell=True)
    subprocess.call('rm {0}/T1_defaced.nii.gz {0}/T1.nii.gz'.format(mri_dir), shell=True)
    
    # create high res surface for coregister
    subprocess.call('mne make_scalp_surfaces --overwrite -s {0} -d {1} -f'.format(sub, fs_dir), shell=True)
    
    #==============================
    # coregister with mne coreg gui
    #==============================
    
    print(sub + ' done')
    


    
