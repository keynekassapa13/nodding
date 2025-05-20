import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from fastdtw import fastdtw

# Nodding gesture template
nod_template = np.array([0, 3, 7, 10, 7, 3, 0])

def detect_nods(
    pitch: np.ndarray,
    threshold: float = 20.0,
    window_size: int = 35,
    min_pause: int = 5,
    pitch_threshold: float = 10
):
    """
    DTW-based nod detection.
    Args:
        pitch (np.ndarray): Pitch time series.
        threshold (float): DTW distance threshold for nod detection.
        window_size (int): Size of the sliding window for DTW.
        min_pause (int): Minimum frames between nods.
        pitch_threshold (float): Minimum pitch delta to consider a nod.
    Returns:
        nod_mask (np.ndarray): Boolean mask of nods.
        nod_indices (np.ndarray): Indices of detected nods.
    """
    nod_indices = []
    last_nod_frame = -min_pause

    for i in range(len(pitch) - window_size):
        window = pitch[i:i+window_size]
        pitch_range = np.max(window) - np.min(window)
        if pitch_range < pitch_threshold:
            continue
        window = (window - np.mean(window)) / (np.std(window) + 1e-6)
        template = (nod_template - np.mean(nod_template)) / (np.std(nod_template) + 1e-6)
        # DTW distance
        distance, _ = fastdtw(window, template)
        if distance < threshold and i - last_nod_frame >= min_pause:
            nod_indices.append(i + window_size // 2)
            last_nod_frame = i

    nod_mask = np.zeros_like(pitch, dtype=bool)
    nod_mask[nod_indices] = True
    return nod_mask, np.array(nod_indices)