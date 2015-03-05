import mne
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def read_file(file_name):
    names = ['Fc1', 'FC2', 'C3', 'C4', 'O1', 'O2', 'Fz', 'Cz', 'M_L', 'M_R', 'VEOG', 'HEOG']
    r = mne.io.brainvision.read_raw_brainvision(file_name, elp_names= names, preload=True, reference='Fcz', eog=['HEOG', 'VEOG'], verbose=None)
    #r.plot(events=None, duration=1, start=0.0, n_channels=5)
    return r

def make_epoch(r):
    event = mne.io.brainvision.brainvision.RawBrainVision.get_brainvision_events(r) #get event array
    epoch = mne.Epochs(r, event, [4, 8], -0.5, 3, proj=False, baseline=(None, None), preload=False, decim=1, detrend=1, add_eeg_ref=True, verbose=True) #create epochs aligned on precue
    return epoch, event
    
def reject_eog(r, epoch, event, verbose=False):
    epoch_copy = deepcopy(epoch)
    eog = mne.preprocessing.find_eog_events(r, event_id=998, l_freq=1, h_freq=10, filter_length='10s', ch_name= 'VEOG', tstart=0, verbose=None )
    epoch_edges = event[(event[:, 2] == 4) | (event[:, 2] == 8)][:,0] #
    epoch_with_eog = np.digitize(eog[:,0], epoch_edges)
    epoch_copy.drop_epochs(epoch_with_eog[np.where(epoch_with_eog > 1)], reason = 'EOG')
    if verbose:
        print epoch_copy.drop_log
    return epoch_copy
    
def process_window_ff(window):
    win2 = window.transpose([0, 2, 1])
    win3 = win2.reshape(win2.shape[0] * win2.shape[1], win2.shape[2])
    m = np.mean(win3, axis=0)
    v = np.var(win3, axis=0)
    ff = m / v
    return ff

def process_window_psds(window):
    win2 = window.transpose([0, 2, 1])
    win3 = win2.reshape(win2.shape[0] * win2.shape[1], win2.shape[2])
    psds, freqs = mne.time_frequency.compute_epochs_psd(epoch_copy, fmin= 400, fmax= 1000)
    average_psds = psds.mean(0)
    average_psds = 10 * np.log10(average_psds)
    
    for i in aligned:
    out = [plt.psd(d, Fs=Fs, NFFT=n_fft, noverlap=n_overlap, pad_to=pad_to)
        for d in data]
    psd = np.array([o[0] for o in out])
    freqs = out[0][1]
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    return psd[:, mask], freqs    
    
    
def create_window(aligned, process_window):
    nwid = 500
    winstep = 125
    nchan = aligned.shape[1]
    nsamp = aligned.shape[2]
    starts = np.arange(0, nsamp - nwid, winstep)
    nwin = len(starts)
    result = np.empty([nchan, nwin])
    for i, start in enumerate(starts):
        stop = start + nwid
        window = aligned[:,:,start:stop]
        result[:,i] = process_window(window)
    return window, result
    
def graph_ff(result):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(result.T)
    ax.set_xlabel('Window #')
    ax.set_ylabel('Fano factor')
    ax.set_title('Fano factor across channels')
    plt.show()

if __name__ == "__main__":
    file_name = 'C:\\Users\\mwar136\\Desktop\\Summer Scholarship\\borrowed_EEG_files\\mne test\\AMCtrial.vhdr'
    raw = read_file(file_name)
    epoch, event = make_epoch(raw)
    clean_epoch = reject_eog(raw, epoch, event)
    aligned = clean_epoch.get_data()
    window, result = create_window(aligned, process_window_ff)
    graph_ff(result)
    