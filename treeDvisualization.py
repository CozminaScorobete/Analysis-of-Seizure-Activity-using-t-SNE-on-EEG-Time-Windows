import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.signal import welch
import matplotlib.colors as mcolors


def plot_3d_brain_and_freq(eeg_data, channel_positions_3d, mapped_channels, seizure_labels, fs, window_sec):
    """
    Plots 3D brain activity and 3D frequency band analysis with proper legends and brain shape.
    """
    fig = plt.figure(figsize=(16, 7))
    seizure_indices = np.where(seizure_labels == 1)[0]
    non_seizure_indices = np.where(seizure_labels == 0)[0]
    bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

    # 1. Brain activity with proper brain shape outline
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Create brain outline (sphere wireframe)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v)) * 0.9
    y_sphere = np.outer(np.sin(u), np.sin(v)) * 0.9
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.9
    ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # Determine which period to analyze (prioritize seizure data if available)
    if len(seizure_indices) > 0:
        period_data = seizure_indices[:min(10, len(seizure_indices))]
        analysis_type = "Seizure"
        color_scheme = 'Reds'
    else:
        period_data = non_seizure_indices[:10]
        analysis_type = "Non-seizure"
        color_scheme = 'Blues'
    
    brain_activity = []
    brain_coords = []
    for ch_idx, ch_name in enumerate(mapped_channels):
        if ch_name in channel_positions_3d:
            powers = []
            for w_idx in period_data:
                start, end = w_idx * window_sec * fs, (w_idx + 1) * window_sec * fs
                if end <= eeg_data.shape[1]:
                    signal = eeg_data[ch_idx, start:end]
                    if np.std(signal) > 1e-10:
                        powers.append(np.mean(np.abs(signal)))
            if powers:
                brain_activity.append(np.mean(powers))
                brain_coords.append(channel_positions_3d[ch_name])

    if brain_activity:
        brain_activity = np.array(brain_activity)
        brain_coords = np.array(brain_coords)
        # Normalize activity
        if np.max(brain_activity) > np.min(brain_activity):
            brain_activity = (brain_activity - np.min(brain_activity)) / (np.max(brain_activity) - np.min(brain_activity))
        
        scatter = ax1.scatter(brain_coords[:, 0], brain_coords[:, 1], brain_coords[:, 2],
                            c=brain_activity, cmap=color_scheme, s=brain_activity * 500 + 100, 
                            edgecolors='black', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.1)
        cbar.set_label(f'{analysis_type} Activity Level')
    
    ax1.set_title(f"3D Brain Activity Model ({analysis_type} Periods)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 2. Frequency band comparison with manual legend (fixed)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Calculate frequency powers for both seizure and non-seizure
    band_names = list(bands.keys())
    seizure_powers = []
    non_seizure_powers = []
    
    for band, (low, high) in bands.items():
        # Seizure powers
        seizure_band_powers = []
        for ch in range(min(10, len(mapped_channels))):
            for idx in seizure_indices[:min(10, len(seizure_indices))]:
                start, end = idx * window_sec * fs, (idx + 1) * window_sec * fs
                if end <= eeg_data.shape[1]:
                    signal = eeg_data[ch, start:end]
                    if np.std(signal) > 1e-10:
                        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs))
                        mask = np.logical_and(freqs >= low, freqs <= high)
                        if np.any(mask):
                            seizure_band_powers.append(np.trapz(psd[mask], freqs[mask]))
        
        # Non-seizure powers
        non_seizure_band_powers = []
        for ch in range(min(10, len(mapped_channels))):
            for idx in non_seizure_indices[:min(10, len(non_seizure_indices))]:
                start, end = idx * window_sec * fs, (idx + 1) * window_sec * fs
                if end <= eeg_data.shape[1]:
                    signal = eeg_data[ch, start:end]
                    if np.std(signal) > 1e-10:
                        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs))
                        mask = np.logical_and(freqs >= low, freqs <= high)
                        if np.any(mask):
                            non_seizure_band_powers.append(np.trapz(psd[mask], freqs[mask]))
        
        seizure_powers.append(np.mean(seizure_band_powers) if seizure_band_powers else 0)
        non_seizure_powers.append(np.mean(non_seizure_band_powers) if non_seizure_band_powers else 0)
    
    # Normalize powers
    all_powers = seizure_powers + non_seizure_powers
    max_power = max(all_powers) if max(all_powers) > 0 else 1
    seizure_powers = np.array(seizure_powers) / max_power
    non_seizure_powers = np.array(non_seizure_powers) / max_power
    
    # Create 3D bars
    x_pos = np.arange(len(bands))
    width = 0.35
    
    # Seizure bars (red) - no label parameter for bar3d
    seizure_bars = ax2.bar3d(x_pos - width/2, np.zeros(len(bands)), np.zeros(len(bands)), 
                            width, 0.8, seizure_powers, color='red', alpha=0.7)
    
    # Non-seizure bars (blue) - no label parameter for bar3d
    non_seizure_bars = ax2.bar3d(x_pos + width/2, np.zeros(len(bands)), np.zeros(len(bands)), 
                                width, 0.8, non_seizure_powers, color='blue', alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(band_names)
    ax2.set_ylabel('Condition')
    ax2.set_zlabel('Normalized Power')
    ax2.set_title("3D Frequency Band Comparison")
    
    # Create manual legend using dummy plots (FIXED - this is the solution)
    from matplotlib.patches import Rectangle
    red_patch = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7)
    blue_patch = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7)
    ax2.legend([red_patch, blue_patch], ['Seizure', 'Non-seizure'], loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_alpha_band_topomap_comparison(eeg_data, channel_positions_2d, mapped_channels, seizure_labels, fs, window_sec):
    """
    Plots 2D topomaps for Alpha band comparing seizure vs non-seizure periods.
    """
    from matplotlib.patches import Circle
    
    seizure_indices = np.where(seizure_labels == 1)[0]
    non_seizure_indices = np.where(seizure_labels == 0)[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = [
        (seizure_indices, "Seizure", axes[0]),
        (non_seizure_indices, "Non-seizure", axes[1])
    ]
    
    for indices, title, ax in conditions:
        if len(indices) == 0:
            ax.text(0.5, 0.5, f'No {title.lower()} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Alpha Band Activity - {title}")
            ax.axis('off')
            continue
            
        alpha_powers, coords_2d, valid_channels = [], [], []
        
        for ch_idx, ch_name in enumerate(mapped_channels):
            if ch_name in channel_positions_2d:
                powers = []
                for idx in indices[:min(20, len(indices))]:  # Limit to 20 windows for performance
                    start, end = idx * window_sec * fs, (idx + 1) * window_sec * fs
                    if end <= eeg_data.shape[1]:
                        signal = eeg_data[ch_idx, start:end]
                        if np.std(signal) > 1e-10:
                            freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs))
                            mask = np.logical_and(freqs >= 8, freqs <= 13)
                            if np.any(mask):
                                powers.append(np.trapz(psd[mask], freqs[mask]))
                if powers:
                    alpha_powers.append(np.mean(powers))
                    coords_2d.append(channel_positions_2d[ch_name])
                    valid_channels.append(ch_name)

        if alpha_powers:
            alpha_powers = np.array(alpha_powers)
            coords_2d = np.array(coords_2d)
            
            # Normalize powers
            if np.std(alpha_powers) > 0:
                alpha_powers = (alpha_powers - np.mean(alpha_powers)) / np.std(alpha_powers)
            
            # Create topomap
            scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=alpha_powers,
                               s=200, cmap='RdBu_r', edgecolors='black', vmin=-2, vmax=2)
            
            # Add channel labels
            for coord, ch in zip(coords_2d, valid_channels):
                ax.annotate(ch.split('-')[0], coord, ha='center', va='center', fontsize=6)
            
            # Add brain outline
            brain_circle = Circle((0, 0), 1.0, fill=False, linewidth=2, color='black')
            ax.add_patch(brain_circle)
            
            # Add nose indicator
            ax.plot([0, 0], [1.0, 1.2], 'k-', linewidth=2)
            
            # Add ear indicators
            left_ear = Circle((-1.0, 0), 0.1, fill=False, linewidth=2, color='black')
            right_ear = Circle((1.0, 0), 0.1, fill=False, linewidth=2, color='black')
            ax.add_patch(left_ear)
            ax.add_patch(right_ear)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Normalized Alpha Power')
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_title(f"Alpha Band Activity - {title}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_2d_tsne_pca(Y_tsne, features_scaled, seizure_labels):
    """
    Plots 2D t-SNE and PCA visualizations with proper legends.
    """
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define colors and labels
    colors = ['blue', 'red']
    labels = ['Non-seizure', 'Seizure']
    
    # t-SNE plot
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = seizure_labels == i
        if np.any(mask):
            axes[0].scatter(Y_tsne[mask, 0], Y_tsne[mask, 1], 
                          c=color, alpha=0.6, s=20, label=label)
    
    axes[0].set_title("2D t-SNE")
    axes[0].set_xlabel("t-SNE Component 1")
    axes[0].set_ylabel("t-SNE Component 2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features_scaled)
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = seizure_labels == i
        if np.any(mask):
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=color, alpha=0.6, s=20, label=label)
    
    axes[1].set_title("2D PCA")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_3d_feature_space(Y_tsne, features_scaled, seizure_labels):
    """
    Plots 3D t-SNE and 3D PCA feature space side by side.
    """
    fig = plt.figure(figsize=(16, 7))

    # ----- 3D t-SNE -----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    z_variation = np.mean(features_scaled, axis=1) * 10 if features_scaled.shape[1] > 0 else np.zeros(len(Y_tsne))
    non_seizure_mask = seizure_labels == 0
    seizure_mask = seizure_labels == 1

    ax1.scatter(Y_tsne[non_seizure_mask, 0], Y_tsne[non_seizure_mask, 1], z_variation[non_seizure_mask],
                c='blue', alpha=0.6, s=30, label='Non-seizure')
    if np.any(seizure_mask):
        ax1.scatter(Y_tsne[seizure_mask, 0], Y_tsne[seizure_mask, 1], z_variation[seizure_mask],
                    c='red', alpha=0.8, s=60, label='Seizure')

    ax1.set_title('3D t-SNE Feature Space')
    ax1.set_xlabel('t-SNE Comp 1')
    ax1.set_ylabel('t-SNE Comp 2')
    ax1.set_zlabel('Feature Intensity')
    ax1.legend()

    # ----- 3D PCA -----
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if features_scaled.shape[1] >= 3:
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features_scaled)

        ax2.scatter(features_3d[non_seizure_mask, 0], features_3d[non_seizure_mask, 1], features_3d[non_seizure_mask, 2],
                    c='blue', alpha=0.6, s=30, label='Non-seizure')
        if np.any(seizure_mask):
            ax2.scatter(features_3d[seizure_mask, 0], features_3d[seizure_mask, 1], features_3d[seizure_mask, 2],
                        c='red', alpha=0.8, s=60, label='Seizure')

        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax2.set_title('3D PCA Feature Space')
        ax2.legend()

    plt.tight_layout()
    plt.show()