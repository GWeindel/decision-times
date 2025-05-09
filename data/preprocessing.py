import mne
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import parallel_config
from warnings import warn

#For use in a python console uncomment next line
#%matplotlib qt

def creating_montage(raw):
    '''
    Creating standard montage
    '''
    raw.drop_channels(['EXG3','EXG7','EXG8'])#Empty

    # Creating the bipolar montage, the 4 first are the EOG
    mne.set_bipolar_reference(raw,anode=['EXG1','EXG4'],cathode=['EXG2','A1'],ch_name=['EOGH','EOGV'],copy=False, drop_refs=False) 
    raw.set_channel_types({'EOGH':'eog','EOGV':'eog','EXG5':'misc','EXG6':'misc','Erg1':'misc',
                          'EXG1':'misc','EXG2':'misc','EXG4':'misc','EXG5':'misc'})#declare type to avoid confusion with EEG channels
    #Renaming electrodes
    dict_to_biosemi = dict(zip(raw.copy().pick_types(eeg=True).ch_names, mne.channels.make_standard_montage('biosemi32').ch_names))
    raw.rename_channels(dict_to_biosemi)
    raw.set_montage('biosemi32')
    raw.drop_channels(['EXG1','EXG2','EXG4'])# remove elec used for EOGs

def reject_breaks(raw, stim_events, tmin, tmax, break_duration = 10):
    '''
    Annotating breaks, not strictly necessrary but helps diagnosing bad electrodes with psd
    '''
    tmin -= .6 #stop annotation before planned epoching
    tmax += .1 #start annotation right after epoching window
    too_long = np.where(np.diff(stim_events[:,0], n=1) > (raw.info['sfreq']*(break_duration)))[0]
    onset_breaks = stim_events[too_long][:,0]/raw.info['sfreq'] + tmax  #start x sec after last stimulus trigger
    offset_breaks = stim_events[too_long+1][:,0]/raw.info['sfreq'] + tmin #stops break x sec before next stimulus trigger
    
    onset_breaks = np.insert(onset_breaks,0,0)#adding start of the recording to the breaks
    offset_breaks = np.insert(offset_breaks,0,stim_events[0,0]/raw.info['sfreq']+tmin)#adding start of the recording to the breaks
    onset_breaks = np.insert(onset_breaks, len(onset_breaks), stim_events[-1,0]/raw.info['sfreq']+tmax)#adding end of the recording to the breaks
    offset_breaks = np.insert(offset_breaks,len(offset_breaks), raw.times.max())#adding end of the recording to the breaks
    duration_breaks = offset_breaks - onset_breaks
    
    logging.info(f"Detected {len(duration_breaks)} breaks")
    break_annot = mne.Annotations(onset=onset_breaks,
        duration=duration_breaks,
        description=['BAD_breaks'])
    raw.set_annotations(break_annot)

def find_onset_photodiode(data, event_sample, baseline=500):
    '''
    Detects onset of stim through std deviation of photodiode signal
    '''
    index = np.where(data <0)[0][0]
    if event_sample - (index-baseline+event_sample) >0:
        warn(f'inconsistent photodiode detected onset at sample {event_sample}')
        return(event_sample) #Isolated stim event
    else:
        return(index-baseline+event_sample)

def find_and_correct_events(raw):
    '''
    Read events from trigger channel, correct based on photodiode timing
    '''
    events = mne.find_events(raw)
    all_events = np.array(np.unique(events[:,2]))
    stim_trigger = all_events[all_events<99]
    baseline = 500
    photodiode = raw.copy().pick(['Erg1']).get_data()[0]
    new_events = events.copy()
    for i in np.arange(len(events)):
        if events[i,2] in stim_trigger:
            new_events[i,0] = find_onset_photodiode(photodiode[events[i,0]-baseline:events[i,0]+150], 
                                                events[i,0])
    events = new_events
    raw.add_events(new_events, replace=True)
    raw.drop_channels(['Erg1'])# remove elec used for EOGs
    return events, stim_trigger

def run_ica(raw, events, stim_id, tmin, tmax):
    ## Epoching for ICA with filtering/resampling
    epochs_ica = mne.Epochs(raw.copy().filter(l_freq=1, h_freq=None),
                            events, event_id=[int(x) for x in list(stim_id.values())],
                            tmin=tmin, tmax=tmax, preload=True, decim=4, baseline=None)
    
    ## Defining  and fitting ICA
    n_components = len(mne.pick_types(raw.info, meg=False, eeg=True))-len(raw.info["bads"])-1 #-1 as average reference is used (rank)
    ica = mne.preprocessing.ICA(n_components = n_components, method='fastica', max_iter='auto')
    ica.fit(epochs_ica)
    
    ## Finding EOG ICs, Excluding eye related ICs and save diagnostic plot
    # two window open, if you want to change the retained IC you need to click on the
    # epoched/timecourse plot
    eog_indices, eog_scores = ica.find_bads_eog(epochs_ica, threshold=3.5)
    eog_indices = [index for index in eog_indices if index < 20] #No need to remove IC with low var
    ica.exclude = eog_indices
    ica.plot_sources(epochs_ica, picks=range(20), block=False)
    ica.plot_components(picks=range(20))

    fig, ax = plt.subplots(4, 5, figsize=(10, 10))
    ax = ax.flatten()
    ica.plot_components(picks=range(20), axes=ax, show=False)
    fig.savefig("ica_plots/%s.png"%name_subj)
    plt.close()
    return ica

## Infos
EEG_data_path = "anon_raw/"
parallel_config(n_jobs=-1)
                       
## Trigger definition
fixation_trigger = {"fixation":100}
condition_id = {"condition/accuracy":101, "condition/speed":201}#condition trigger
side_id = {"side/left":99, "side/right":199}#Expected response side (correct answer)
resp_id = {"response/left":100,  "response/right":200}#Given esponse side events

# Epoch window
tmin = -0.2 #tmin is how much data (in s) needs to be used for baseline correction
tmax = 3 #tmax is how much far in time from stim should we look for a response

for sub_file in os.listdir(EEG_data_path):
    if '.fif' in sub_file:
        name_subj = sub_file.split(".")[0].split('_')[0]
        if name_subj+"_epo.fif" not in os.listdir("preprocessed"):
            mne.set_log_file(fname=f"logs/preprocessing_{name_subj}.log")
            mne.set_log_level(verbose="INFO")
            raw = mne.io.read_raw_fif(EEG_data_path+sub_file, preload=True)
            
            ## Setting montage
            creating_montage(raw)
            
            ## Reading events 
            events, stim_trigger = find_and_correct_events(raw)
            stim_id = {'stimulus/'+str(k):k for k in stim_trigger}#building dict on those
            event_id = condition_id | stim_id | side_id | resp_id #all retained events
            stim_events = np.array([list(x) for x in events if x[2] in list(stim_id.values())])
            all_events = np.array(np.unique(events[:,2]))

            # Coloring for visual inspection
            color_dict = {k:"b" for k in list(stim_id.values())}
            color_dict.update({k:"g" for k in list(resp_id.values())})
            color_dict_i = color_dict.copy()
            color_dict_i.update({k:'gray' for k in set.difference(set(all_events),set(list(stim_id.values())), set(list(resp_id.values())))})
        
            ## Filtering and annotating breaks
            raw.filter(l_freq=None, h_freq=40, picks='eeg')
            reject_breaks(raw, stim_events, tmin, tmax)
            
            ## Inspecting bad electrodes, annotating bad segments
            raw.plot_psd(tmin=500, fmax=41, picks="eeg")
            raw.plot(events=events, scalings=5e-5, block=True, event_color=color_dict_i,
                    remove_dc=True, n_channels=64)
            
            raw.set_eeg_reference("average")
            
            # ICA 
            ica = run_ica(raw, events, stim_id, tmin, tmax)

            # reinspect to eventually reject electrode picked up by ICA, re-run ICA
            n_bads = len(raw.info["bads"])
            raw.plot(events=events, scalings=5e-5, block=True, event_color=color_dict_i,
                    remove_dc=True, n_channels=64)
            n_bads_new = len(raw.info["bads"])
            if n_bads != n_bads_new:
                ica = run_ica(raw, events, stim_id, tmin, tmax)

            ## Clean data by removing eye related ICs on original raw data and interpolate bads
            ica.apply(raw)
            raw.interpolate_bads()
            
            # Definitive epoching
            ## Extract metadata based on triggers
            # added -.25 to tmin because condition trigger is missed with window tmin
            metadata, events, new_event_id = mne.epochs.make_metadata(
                events=events, event_id=event_id, tmin=tmin-.25, tmax=tmax,
                sfreq=raw.info["sfreq"], row_events=list(stim_id.keys()), keep_first=["condition","side","stimulus","response"])
    
            keep_cols = ['event_name', 'response', 'first_condition', 'first_side','first_stimulus','first_response']
            metadata = metadata[keep_cols]
            metadata.reset_index(drop=True, inplace=True)
            metadata.columns = ['event_name', 'rt', 'condition', 'side', 'stimulus','response']
            epochs = mne.Epochs(raw, events, new_event_id, tmin, tmax, proj=False,
                    baseline=(None, 0), preload=True, decim=2,# Baseline correction and decimation to 512Hz
                    verbose=True, detrend=None, on_missing = "warn", event_repeated="drop",
                    metadata=metadata, reject_by_annotation=False)
            epochs.save("preprocessed/%s_epo.fif"%name_subj, overwrite=True)
            plt.close('all')
