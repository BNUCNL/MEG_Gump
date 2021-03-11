# fmriprep
fmriprep-docker .\mri_bids .\mri_bids_fmriprep_output participant --output-spaces anat MNI152NLin6Asym:res-2 MNI152NLin2009cAsym:res-2 fsnative fsaverage fsLR --task-id rest --use-aroma --skull-strip-t1w force -w .\mri_bids_fmriprep_work --fs-license-file .\license.txt --output-layout bids

