import numpy as np 
import pandas as pd
from mne import compute_covariance, cov, read_epochs, filter
from scipy.signal import lfilter

# Defininf a half-sine, as the one expected by HMP
class HalfSine:
    def __init__(self, sfreq):
        width = 50
        steps = 1000/sfreq
        width_samples = width/steps
        
        event_idx = np.arange(width_samples) * steps + steps / 2
        # gives event frequency given that events are defined as half-sines
        event_frequency = 1000 / (width * 2)
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        self.template = template

# Class for evidence accumulation traces
class EvidenceAccumulation:
    def __init__(self, sfreq, drift_rate, time_steps=20000, n_simulations=100000):
        traces = np.zeros((n_simulations, time_steps))
        decision_times = np.zeros(n_simulations)
        time_steps = int(time_steps/(1000/sfreq))# Same sampling frequency as signal
        self.sfreq = sfreq
        self.traces = {}
        self.decision_times = {}
        #Initiate a bunch of traces
        for i in range(n_simulations):
            trace, decision_time = self.accumulation_trace(drift_rate, 1, time_steps)
            traces[i, :len(trace)] = trace
            decision_times[i] = decision_time
        self.traces = traces
        self.decision_times = decision_times

    def accumulation_trace(self, drift_rate, threshold, time_steps):
        noise_std = .1 
        #Random walk
        evidence = np.cumsum(np.random.normal(drift_rate, noise_std, time_steps))
        decision_time = np.where(np.abs(evidence) >= threshold)[0]
        if len(decision_time) > 0:
            decision_time = decision_time[0]
        else:
            decision_time = time_steps
        # Invert trace if threshold is neg
        if evidence[decision_time] <= -threshold:
            evidence = -evidence
        return evidence[:decision_time+1], decision_time
    
    def find_closest(self, time):
        # To mitigate randomness we average the top 10 traces that are closest to the actual time     
        top_10 = np.argsort(np.abs(self.decision_times - time))[:10]
        if np.mean(np.abs(self.decision_times[top_10] - time)) > 50/(1000/self.sfreq):
            warn(f'distance is {np.mean(np.abs(self.decision_times[top_10] - time))} samples away, increase n_simulations?')
        average_trace = np.mean(self.traces[top_10], axis = 0)
        average_trace = average_trace[:int(time)+1]
        return average_trace

def simulate_from_hmpfit(epoch_data,
                         weights,
                         times,
                         template_dict,
                         add_noise = False
                        ):
    """
    epoch_data is the EEG data from hmp.io.read_mne_data
    weights is the values of the electrodes at the single-trial times detected by HMP
    times is the single-trial times detected by HMP
    template_dict is the dictionnary mapping between event in the sequence and one of the classes above
    """
    sfreq = epoch_data.sfreq
    n_chan, _, n_events = np.shape(weights)
    surrogate = epoch_data.copy(deep=True)
    surrogate['data'] = 0 * surrogate['data']#Overwrite all the real data
    surrogate = surrogate.stack({'trial':["participant","epoch"]})
    # Add event activations at corresponding times
    for trial in times.trial:
        trial_times = times.sel(trial=trial).values
        trial_rt = int(trial_times[-1])
        # use weights of each event, sum over events,
        for event in range(n_events):
            trial_template = template_dict[event]
            # EA traces are defined from previous event
            if isinstance(trial_template, EvidenceAccumulation):
                prev_event = trial_times[event-1].astype(int)
                dist_from_prev = trial_times[event]-prev_event+1
                trial_template = trial_template.find_closest(dist_from_prev)
                length_of_event = len(trial_template)
                proposed_onset = prev_event
                proposed_offset = prev_event+length_of_event
            else:# Half-Sine's peak is symetric
                trial_template = trial_template.template
                length_of_event = len(trial_template)   
                proposed_onset = trial_times[event].astype(int) - length_of_event // 2
                proposed_offset = trial_times[event].astype(int) + length_of_event // 2 + 1
                if length_of_event%2 == 0:
                    proposed_offset -= 1
            # Ensures that pattern doesn't exceed stim/RT boundaries if detected event close to those
            onset = max(proposed_onset, 0)
            offset = min(proposed_offset, trial_rt)
            length = offset - onset
            trim_start = onset - proposed_onset
            trim_end = proposed_offset - offset
            surrogate.sel(trial=trial)['data'][:,onset:offset] += \
                np.outer(trial_template[trim_start:length_of_event - trim_end],
                         weights.sel(trial=trial, event=event).values).T
    surrogate = surrogate.unstack().transpose('participant','epoch', 'channel', 'sample')
    if add_noise:
        # Add noise, using participant covariance among electrodes and an IIR filter
        for participant in surrogate.participant:
            surrogate.sel(participant=participant)["data"] += create_noise(participant.values, \
                        sfreq, n_trials=len(surrogate.epoch))[:,:,:len(surrogate.sel(participant=participant)["data"])]
   
    surrogate = surrogate.unstack()
    return surrogate


def create_noise(participant, sfreq, n_trials):
    """Create spatially colored and temporally IIR-filtered noise.
    
    Adapted from: https://github.com/mne-tools/mne-python/blob/maint/1.9/mne/simulation/evoked.py#L171
    """

    # Use actual epoched data
    epochs = read_epochs(f'data/preprocessed/{participant}.fif', verbose=False)
    epochs = epochs.pick('eeg')
    epochs = epochs.resample(100)#Speed
    data_cov = compute_covariance(epochs, keep_sample_mean=False, tmin=-.2,tmax=0, verbose=False)#Use only baseline, less actual ERPs
    data_std = np.std(epochs.get_data()[:,:,:20])
    data_cov = cov.prepare_noise_cov(data_cov, epochs.info, verbose=False)
    _, _, colorer = cov.compute_whitener(data_cov, pca=True, return_colorer=True, verbose=False)
    n_samples = int(np.rint(3000/(1000/sfreq)))
    noise_matrix = np.zeros((n_trials, len(epochs.info["ch_names"]), n_samples))
    for trial in range(n_trials):
        noise = np.dot(colorer, np.random.standard_normal((colorer.shape[1], n_samples)))
        zi = np.zeros((len(colorer), len([0.2, -0.2, 0.04]) - 1))
        noise, _ = lfilter([1], [0.2, -0.2, 0.04], noise, axis=-1, zi=zi)
        noise = noise * (data_std / np.std(noise))
        noise_matrix[trial] = noise
    return noise_matrix