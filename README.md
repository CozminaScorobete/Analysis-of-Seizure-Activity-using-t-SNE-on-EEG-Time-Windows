# Analysis-of-Seizure-Activity-using-t-SNE-on-EEG-Time-Windows


Epileptic seizure detection from EEG signals is a complex task that that requites a specialist in order to read the EEG. However the automatic detection increased in popularity and more studies appeared on the matter. In this paper we explore the use of t-distributed Stochastic Neighbor Embedding (t-SNE) as a tool for visualizing EEG and identifying seizure patterns. A custom implementation of t-SNE was developed to project extracted EEG features into 2D and 3D spaces, aiming to improve interpretability of seizure vs. non-seizure data. We use the CHB-MIT Scalp EEG dataset and extract both spectral (band power) and statistical features from segmented time windows of varying lengths (5, 10, 15, and 20 seconds). In parallel, Support Vector Machines (SVM) are used to classify seizure activity based on the original features, PCA-reduced features, and t-SNE embeddings.

Requiremants:
import mne
and ML requiremants



In order to use the code just clone main.py and run it
