def main_pipeline_with_3d_viz(edf_path="chb01_03.edf", summary_path=None, use_additional_features=True):
    """
    Complete pipeline with modular 3D visualizations
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import mne
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tSNE_implementation import TSNEImplementation
    from fetures import extract_bandpower_multichannel, extract_additional_features
    from eegHandle import filter_eeg_channels, create_virtual_montage_for_bipolar_channels, load_seizure_annotations_from_summary, create_seizure_labels
    from treeDvisualization import (
        plot_3d_feature_space, plot_3d_brain_and_freq,
         plot_2d_tsne_pca, plot_alpha_band_topomap_comparison
    )

    print("\n🚀 === EEG Analysis Pipeline ===")

    if not os.path.exists(edf_path):
        print(f"ERROR: EEG file not found: {edf_path}")
        return None, None, None

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    eeg_channel_indices = filter_eeg_channels(raw)
    eeg_channel_names = [raw.ch_names[i] for i in eeg_channel_indices]
    fs = int(raw.info['sfreq'])
    eeg_data, _ = raw[eeg_channel_indices, :]

    window_sec = 5
    step_sec = 5
    window_size = window_sec * fs
    step_size = step_sec * fs
    total_samples = eeg_data.shape[1]

    windows = [eeg_data[:, start:start + window_size] for start in range(0, total_samples - window_size + 1, step_size)]
    windows = np.array(windows)

    features = []
    for window in windows:
        bandpower = extract_bandpower_multichannel(window, fs)
        if use_additional_features:
            additional = extract_additional_features(window, fs)
            combined = np.concatenate([bandpower, additional])
        else:
            combined = bandpower
        features.append(combined)

    features = np.array(features)
    features = np.nan_to_num(features)
    features_scaled = StandardScaler().fit_transform(features)

    edf_filename = os.path.basename(edf_path)
    seizures = load_seizure_annotations_from_summary(edf_filename, summary_path)
    seizure_labels = create_seizure_labels(seizures, len(windows), window_sec)

    tsne = TSNEImplementation(perplexity=min(30, max(5, len(windows)//4)), learning_rate=200, n_iter=250)
    Y_tsne = tsne.fit_transform(features_scaled)

    pca = PCA(n_components=3)
    Y_pca = pca.fit_transform(features_scaled)

    channel_positions_3d, channel_positions_2d, mapped_channels = create_virtual_montage_for_bipolar_channels(eeg_channel_names)

    if mapped_channels:
        plot_3d_feature_space(Y_tsne, features_scaled, seizure_labels)
        plot_3d_brain_and_freq(eeg_data, channel_positions_3d, mapped_channels, seizure_labels, fs, window_sec)
        
        # Use the new comparison function instead of single topomap
        plot_alpha_band_topomap_comparison(eeg_data, channel_positions_2d, mapped_channels, seizure_labels, fs, window_sec)

    plot_2d_tsne_pca(Y_tsne, features_scaled, seizure_labels)
    return Y_tsne, seizure_labels, features_scaled
