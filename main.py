import os
from data_pipeline import main_pipeline_with_3d_viz

if __name__ == "__main__":
    edf_file = "chb01_03.edf"
    summary_file = "chb01-summary.txt"

    print("\U0001F680\U0001F9E0 === 3D Multi-Channel CHB-MIT EEG Analysis === \U0001F9E0\U0001F680")
    print(f"Analyzing: {edf_file}")

    if not os.path.exists(edf_file):
        print(f"\n❌ ERROR: EEG file '{edf_file}' not found!")
        print("Please ensure the EDF file is in the current directory")
    else:
        if os.path.exists(summary_file):
            print(f"Using summary file: {summary_file}")
            Y, labels, features = main_pipeline_with_3d_viz(edf_file, summary_file)
        else:
            print(f"⚠️  Summary file '{summary_file}' not found - using fallback")
            Y, labels, features = main_pipeline_with_3d_viz(edf_file)

        if Y is not None:
            print(f"\n\U0001F389\U0001F680 3D Analysis Complete! \U0001F680\U0001F389")
            print(f"t-SNE embedding: {Y.shape}")
            print(f"Features: {features.shape}")
            print(f"Labels: {labels.shape}")

            print(f"\n=== \U0001F680 3D Visualization Features === \U0001F680")
            print(f"✅ 3D t-SNE embedding with time dimension")
            print(f"✅ 3D brain model with activity mapping")
            print(f"✅ 3D frequency analysis comparison")
            print(f"✅ 3D brain connectivity networks")
            print(f"✅ 3D time-evolution visualization")
            print(f"✅ Enhanced 2D topographic comparisons")

            print(f"\n=== \U0001F31F 3D Insights Available === \U0001F31F")
            print(f"\U0001F3AF Spatial seizure patterns in 3D brain space")
            print(f"⚡ Frequency-specific activity in 3D")
            print(f"\U0001F578️ Brain connectivity during seizures")
            print(f"\U0001F553 Activity evolution over time in 3D")
            print(f"\U0001F4CA Multi-dimensional feature analysis")
            print(f"\U0001F9E0 Complete 3D neurological perspective")
    