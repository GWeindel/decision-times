'''
This script jsut removes the date time of the recording (to avoid any possible link with participants)
and converts to fif
'''
import mne
import os
import numpy as np

participants_id = np.random.choice(26, 26, replace=False) #Random index

for idx, sub_file in enumerate(os.listdir('raw')):
    if '.bdf' in sub_file:
        raw = mne.io.read_raw_bdf('raw/'+sub_file, preload=True)
        raw.anonymize()
        raw.save(f'anon_raw/{participants_id[idx]}_raw.fif')