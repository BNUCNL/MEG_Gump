#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:51:21 2021

@author: liuxingyu
"""
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
import mne
from mne_bids import BIDSPath, write_raw_bids


#%% convert raw data to bids-format
raw_subwise = '/nfs/e5/studyforrest/forrest_movie_meg/raw_data_subwise'
bids_dir = '/nfs/s2/userhome/liuxingyu/workingdir/temp/test'

#%% data of real participant
sublist = ['{0:0>2d}'.format(i) for i in np.arange(1, 12)]
for subid in sublist:
    sub_raw_dir = os.path.join(raw_subwise, 'sub-{0}-raw'.format(subid))
    os.chdir(sub_raw_dir)

    # MEG
    meg_files = glob.glob('S*.ds')
    meg_files.sort()
    
    for file in meg_files:    
        runid = file[-5:-3]
        raw = mne.io.read_raw_ctf(file)
        raw.info['line_freq'] = 50
        
        # event data
        events = mne.find_events(raw, stim_channel='UPPT001', min_duration=2/raw.info['sfreq']) 
        events_id = {'watch_{0}_{1}s'.format((i-1)*10, i*10) : i for i in np.unique(events[:,-1])[:-1]}
        events_id['beginning'] = 255
        
        # convert data to bids format
        bids_path = BIDSPath(subject=subid, session='movie', task='movie', 
                             run=runid, datatype='meg', root=bids_dir)
        write_raw_bids(raw, bids_path=bids_path, events_data=events, 
                       event_id=events_id, overwrite=True)
    
    print('sub-{0}: MEG convertion done'.format(subid))
    
    

