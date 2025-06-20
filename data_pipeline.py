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

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mne

def main_pipeline_with_3d_viz(edf_path="chb01_03.edf", summary_path=None, use_additional_features=True):
    """
    Complete pipeline with modular 3D visualizations
    """


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
    if len(np.unique(seizure_labels)) > 1:  # Only if we have both classes
        svm_classification_analysis(features_scaled, Y_tsne, seizure_labels, window_sizes=[5, 10, 15, 20])
    return Y_tsne, seizure_labels, features_scaled


def svm_classification_analysis(features_scaled, Y_tsne, seizure_labels, test_edf_path="chb01_04.edf", test_summary_path=None, window_sizes=[5, 10, 15, 20]):
    """
    Simple SVM comparison using train on current data, test on different file
    """
    for window_sec in window_sizes:
        print(f"\n📊 Testing Window Size: {window_sec} seconds")
        print("\n🤖 === SVM Classification Analysis ===")
        print(f"Training on current data ({len(features_scaled)} samples)")
        
        # Load test data
        if not os.path.exists(test_edf_path):
            print(f"Test file not found: {test_edf_path}")
            return
        
        # Process test file (same pipeline)
        raw_test = mne.io.read_raw_edf(test_edf_path, preload=True, verbose=False)
        eeg_channel_indices_test = filter_eeg_channels(raw_test)
        fs_test = int(raw_test.info['sfreq'])
        eeg_data_test, _ = raw_test[eeg_channel_indices_test, :]
        
        
        window_size = window_sec * fs_test
        step_size = window_size
        total_samples = eeg_data_test.shape[1]
        
        windows_test = [eeg_data_test[:, start:start + window_size] 
                    for start in range(0, total_samples - window_size + 1, step_size)]
        windows_test = np.array(windows_test)
        
        # Extract test features
        features_test = []
        for window in windows_test:
            bandpower = extract_bandpower_multichannel(window, fs_test)
            additional = extract_additional_features(window, fs_test)
            combined = np.concatenate([bandpower, additional])
            features_test.append(combined)
        
        features_test = np.array(features_test)
        features_test = np.nan_to_num(features_test)
        features_test_scaled = StandardScaler().fit_transform(features_test)
        
        # Test labels
        test_edf_filename = os.path.basename(test_edf_path)
        seizures_test = load_seizure_annotations_from_summary(test_edf_filename, test_summary_path)
        seizure_labels_test = create_seizure_labels(seizures_test, len(windows_test), window_sec)
        
        print(f"Testing on {test_edf_filename} ({len(features_test_scaled)} samples)")
        print(f"Test seizure windows: {np.sum(seizure_labels_test)} ({np.mean(seizure_labels_test)*100:.1f}%)")
        
        # 1. Original Features
        svm_original = SVC(kernel='rbf', class_weight='balanced')
        svm_original.fit(features_scaled, seizure_labels)
        pred_original = svm_original.predict(features_test_scaled)
        acc_original = accuracy_score(seizure_labels_test, pred_original)
        
        # 2. PCA Features
        pca = PCA(n_components=min(len(features_scaled), 10))
        features_pca = pca.fit_transform(features_scaled)
        features_test_pca = pca.transform(features_test_scaled)
        
        svm_pca = SVC(kernel='rbf', class_weight='balanced')
        svm_pca.fit(features_pca, seizure_labels)
        pred_pca = svm_pca.predict(features_test_pca)
        acc_pca = accuracy_score(seizure_labels_test, pred_pca)
        
        # 3. t-SNE Features (use existing Y_tsne for training)
        # For test data, use k-NN to approximate t-SNE mapping
        from sklearn.neighbors import KNeighborsRegressor
        
        knn1 = KNeighborsRegressor(n_neighbors=5)
        knn2 = KNeighborsRegressor(n_neighbors=5)
        knn1.fit(features_scaled, Y_tsne[:, 0])
        knn2.fit(features_scaled, Y_tsne[:, 1])
        
        Y_test_tsne = np.column_stack([
            knn1.predict(features_test_scaled),
            knn2.predict(features_test_scaled)
        ])
        
        svm_tsne = SVC(kernel='rbf', class_weight='balanced')
        svm_tsne.fit(Y_tsne, seizure_labels)
        pred_tsne = svm_tsne.predict(Y_test_tsne)
        acc_tsne = accuracy_score(seizure_labels_test, pred_tsne)
        
        # Results
        print("\nSVM Classification Results:")
        print(f"Original Features: {acc_original*100:.1f}% accuracy")
        print(f"PCA Features: {acc_pca*100:.1f}% accuracy")
        print(f"t-SNE Features: {acc_tsne*100:.1f}% accuracy")
        
        # Best method
        accuracies = [acc_original, acc_pca, acc_tsne]
        methods = ["Original Features", "PCA", "t-SNE"]
        best_idx = np.argmax(accuracies)
        
        print(f"Best method: {methods[best_idx]} ({accuracies[best_idx]*100:.1f}%)")
        
    return accuracies, methods