
import os
import numpy as np
import pandas as pd
import subprocess
import glob
import mne
from mne_bids import BIDSPath, write_raw_bids

#%% convert raw data to bids-format
raw_subwise = '/nfs/e5/studyforrest/forrest_movie_meg/raw_data_subwise'
bids_dir = '/nfs/e5/studyforrest/forrest_movie_meg'

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
        raw.set_annotations(None)
        events = mne.find_events(raw, stim_channel='UPPT001', min_duration=2/raw.info['sfreq']) 

        # repair annotations
        events_id = {'watch_{0}_{1}s'.format((i-1)*10, i*10) : int(i) for i in np.unique(events[:,-1])[:-1]}
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
    