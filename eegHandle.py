

import numpy as np
import re
import os
import mne

def load_seizure_annotations_from_summary(edf_filename, summary_file_path=None):


    #Initialize an empty list to store seizures
    seizures = []
    #find the sumary file exe: from chb01_3.edf -> find chb01-summary.txt 
    if summary_file_path is None:
        case_match = re.match(r'(chb\d+)', edf_filename)
        if case_match:
            case_name = case_match.group(1)
            summary_file_path = f"{case_name}-summary.txt"
        else:
            return []

    try:
        #load the content
        with open(summary_file_path, 'r') as f:
            content = f.read()
        
        #split the file into sections that start with File Name
        file_sections = re.split(r'File Name:', content)
        #find section for the target file exe: chb01_03.edf
        target_section = next((s for s in file_sections if edf_filename in s), None)
        if target_section is None:
            return []
        
        #break the section into individual lines for line-by-line parsing
        lines = target_section.split('\n')
        current_seizure_start = None

        #search for stuff like Seizure 1 Start Time: 2996 seconds
        for line in lines:
            line = line.strip()
            start_match = re.search(r'Seizure.*?Start Time:\s*(\d+)\s*seconds?', line, re.IGNORECASE)
            if start_match:
                current_seizure_start = int(start_match.group(1))
                continue
            #search for stuff like Seizure 1 End Time: 2996 seconds
            end_match = re.search(r'Seizure.*?End Time:\s*(\d+)\s*seconds?', line, re.IGNORECASE)
            if end_match and current_seizure_start is not None:
                end_time = int(end_match.group(1))
                #append the time to the lsit
                seizures.append((current_seizure_start, end_time))
                current_seizure_start = None
        #return the seizures
        return seizures
    except:
        return []



#create a 1D numpay array with labels for seazures example:[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
def create_seizure_labels(seizures, n_windows, window_duration=5):
 
    labels = np.zeros(n_windows)
    for start_time, end_time in seizures:
        start_window = int(start_time / window_duration)
        end_window = int(end_time / window_duration)
        start_window = max(0, start_window)
        end_window = min(n_windows - 1, end_window)
        labels[start_window:end_window + 1] = 1
    return labels

def filter_eeg_channels(raw):
    """
    Filters out non-EEG channels (e.g., ECG, EMG) from an MNE Raw object.
    Returns a list of indices corresponding to valid EEG channels.
    """
    channel_names = raw.ch_names
    eeg_channels = []
    exclude_patterns = ['ecg', 'ekg', 'emg', 'eog', 'resp', 'flow', 'snore', 'vns', 'mark', 'event', 'trigger', 'stim', 'dc']

    for i, ch_name in enumerate(channel_names):
        ch_lower = ch_name.lower()
        if any(p in ch_lower for p in exclude_patterns):
            continue
        #select eeg chanels on naming
        if any(eeg_key in ch_lower for eeg_key in ['fp', 'f', 'c', 'p', 'o', 't', 'fz', 'cz', 'pz']):
            eeg_channels.append(i)
        elif len(ch_name) <= 10 and not ch_name.isdigit() and ch_name != '-':
            eeg_channels.append(i)

    if not eeg_channels:
        eeg_channels = list(range(len(channel_names)))
    return eeg_channels

def create_virtual_montage_for_bipolar_channels(channel_names):
    """
    Maps bipolar EEG channel names to approximate 3D and 2D coordinates.
    Returns dictionaries of 3D and 2D positions.
    """

    #define standard 3D positions
    standard_positions_3d = {
        'FP1': (-0.3, 0.9, 0.3), 'FP2': (0.3, 0.9, 0.3),
        'F7': (-0.7, 0.6, 0.2), 'F3': (-0.4, 0.6, 0.4), 'FZ': (0.0, 0.6, 0.5),
        'F4': (0.4, 0.6, 0.4), 'F8': (0.7, 0.6, 0.2),
        'T7': (-0.9, 0.0, 0.0), 'C3': (-0.4, 0.0, 0.5), 'CZ': (0.0, 0.0, 0.6),
        'C4': (0.4, 0.0, 0.5), 'T8': (0.9, 0.0, 0.0),
        'P7': (-0.7, -0.6, 0.2), 'P3': (-0.4, -0.6, 0.4), 'PZ': (0.0, -0.6, 0.5),
        'P4': (0.4, -0.6, 0.4), 'P8': (0.7, -0.6, 0.2),
        'O1': (-0.3, -0.9, 0.3), 'O2': (0.3, -0.9, 0.3)
    }

    #define standard 2D positions
    standard_positions_2d = {
        k: (x, y) for k, (x, y, _) in standard_positions_3d.items()
    }

    channel_positions_3d = {}
    channel_positions_2d = {}
    mapped_channels = []

    for ch in channel_names:
        #bipolar chanel case
        if '-' in ch:
            
            parts = ch.split('-')
            if len(parts) == 2:
                elec1, elec2 = parts[0].strip().upper(), parts[1].strip().upper()
             #if both electrodes are recognized in the standard electrode map
                if elec1 in standard_positions_3d and elec2 in standard_positions_3d:
                    #Calculate the midpoint between the two 3D coordinates (their average)
                    avg_3d = tuple((a + b) / 2 for a, b in zip(standard_positions_3d[elec1], standard_positions_3d[elec2]))
                    avg_2d = tuple((a + b) / 2 for a, b in zip(standard_positions_2d[elec1], standard_positions_2d[elec2]))
                    #Store the results in the dictionaries and note the channel as successfully mapped
                    channel_positions_3d[ch] = avg_3d
                    channel_positions_2d[ch] = avg_2d
                    mapped_channels.append(ch)
                elif ch.strip().upper() in standard_positions_3d:
                    ch_std = ch.strip().upper()
                    channel_positions_3d[ch] = standard_positions_3d[ch_std]
                    channel_positions_2d[ch] = standard_positions_2d[ch_std]
                    mapped_channels.append(ch)

    return channel_positions_3d, channel_positions_2d, mapped_channels
