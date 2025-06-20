import numpy as np
from scipy.signal import welch

def extract_bandpower_multichannel(window_data, fs):
    #sampling frequency in Hz
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }
    all_features = []
    #number of channels in the input data
    n_channels = window_data.shape[0]

    for ch_idx in range(n_channels):
        #extract the signal for the current channel
        channel_data = window_data[ch_idx, :]
        #If the channel has almost no variation (flat signal), append zeros and skip further processing
        if np.std(channel_data) < 1e-10:
            all_features.extend([0.0] * len(bands))
            continue
        #Welch’s method to compute the Power Spectral Density (PSD)  ------> shows how the power of a signal is distributed across different frequencies
        freqs, psd = welch(channel_data, fs=fs, nperseg=min(len(channel_data), fs*2))
        
        channel_bandpowers = []
        for band, (low, high) in bands.items():
            #compute the area under the PSD curve (band power) using the trapezoidal rule and apply a log transform
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx_band):
                power = np.trapz(psd[idx_band], freqs[idx_band])
                power = np.log1p(power)
            else:
                power = 0.0
            #Store the power for this band
            channel_bandpowers.append(power)
        all_features.extend(channel_bandpowers)

    return np.array(all_features)

def extract_additional_features(window_data, fs):
    n_channels = window_data.shape[0]
    additional_features = []

    for ch_idx in range(n_channels):
        channel_data = window_data[ch_idx, :]
        if np.std(channel_data) < 1e-10:
            additional_features.extend([0.0] * 4)
            continue

        features = [
            #Seizure or movement
            np.mean(np.abs(channel_data)),
            #Activity level
            np.std(channel_data),
            #Rate of change / jitteriness	Spikes, sharp waves
            np.mean(np.diff(channel_data)**2),
            #Captures extreme activity (like spikes) without being as sensitive to outliers as max()
            np.percentile(np.abs(channel_data), 95)
        ]
        additional_features.extend(features)

    return np.array(additional_features)
