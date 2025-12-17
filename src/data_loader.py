"""
data_loader.py

Utilities to load per-trial CSV recordings stored per-person, preprocess,
segment, normalize, and produce arrays or a tf.data.Dataset suitable for
a 1D-CNN expecting (L, 4) inputs.

Usage example (simple):
    X, y, persons, files = build_dataset_from_folder("dataset/", window_size=4000)
    # then feed X,y -> your train_test_cnn or model.fit

Usage example (tf.data):
    ds = dataset_from_arrays(X_tr, y_tr, batch_size=32, shuffle=True)

Notes:
- This loader performs safe per-window normalization (min-max by default), DC removal,
  energy-peak segmentation and optional bandpass filtering.
- Adjust window_size, min_energy_ratio and normalization method to match your data.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Callable
from scipy.signal import sosfiltfilt, butter

# -------------------------
# Helpers: label parsing
# -------------------------
def default_label_from_filename(fname: str) -> int:
    """
    Simple heuristic: if 'valid' in filename -> 1 else 0.
    Case-insensitive.
    You may supply your own label parser if your filenames differ.
    """
    base = os.path.basename(fname).lower()
    return 1 if "valid" in base else 0

# -------------------------
# Read CSV to IQ 4-channel array
# -------------------------
def load_csv_iq(path: str,
                expected_cols: Optional[List[str]] = None) -> np.ndarray:
    """
    Read a CSV file and return Nx4 array in column order:
     [I_lower, Q_lower, I_upper, Q_upper]
    If column names differ, expected_cols can be passed (list of 4 colnames).
    Will attempt to infer by substring if expected_cols is None.
    """
    df = pd.read_csv(path)
    if expected_cols is not None:
        cols = expected_cols
    else:
        # infer by substring matches (robust to slight name differences)
        colnames = df.columns.str.lower().tolist()
        def find(substrs):
            for i, c in enumerate(colnames):
                for s in substrs:
                    if s in c:
                        return df.columns[i]
            raise ValueError(f"Could not find column for {substrs} in {path}")
        cols = [
            find(["i_data_lower", "i_lower", "i_rx1", "i_rx_1", "i1", "i_ch1"]),
            find(["q_data_lower", "q_lower", "q_rx1", "q_rx_1", "q1", "q_ch1"]),
            find(["i_data_upper", "i_upper", "i_rx2", "i_rx_2", "i2", "i_ch2"]),
            find(["q_data_upper", "q_upper", "q_rx2", "q_rx_2", "q2", "q_ch2"]),
        ]
    arr = df[cols].values.astype(np.float32)
    # shape (N,4)
    return arr

# -------------------------
# Preprocessing utilities
# -------------------------
def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract mean per-channel (axis=0)."""
    return x - np.mean(x, axis=0, keepdims=True)

def bandpass_iq(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    Apply zero-phase bandpass (filtfilt) on each channel.
    x: (N, C)
    fs: sampling frequency (Hz)
    low_hz / high_hz: band limits (Hz)
    """
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    sos = butter(order, [low, high], btype="band", output="sos")
    # apply along axis 0 for each column
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = sosfiltfilt(sos, x[:, c])
    return out

# -------------------------
# Segmentation: energy-peak window
# -------------------------
from typing import Optional

def energy_peak_segment(X: np.ndarray,
                        L: int,
                        frame: int = 50,
                        min_energy_ratio: float = 0.05) -> Optional[np.ndarray]:
    """
    Return a length-L segment centered at the energy peak.
    - X: (N,4) raw/time-series
    - L: target length (samples)
    - frame: smoothing window for energy (samples)
    - min_energy_ratio: segment energy must be >= ratio * total_energy else return None
    """
    if X.shape[0] < 8:
        return None
    mag = np.sqrt(np.sum(X**2, axis=1))  # per-sample magnitude across channels
    # squared energy smoothed
    kernel = np.ones(frame, dtype=float) / float(frame)
    energy = np.convolve(mag**2, kernel, mode="same")
    total_energy = energy.sum()
    if total_energy <= 0:
        return None
    peak = int(np.argmax(energy))
    start = peak - L // 2
    end = start + L
    # clamp
    if start < 0:
        start = 0
        end = L
    if end > len(X):
        end = len(X)
        start = max(0, end - L)
    seg = X[start:end]
    if seg.shape[0] != L:
        return None
    # energy check
    if seg.sum() < min_energy_ratio * total_energy:
        return None
    return seg

# -------------------------
# Normalization (per-window)
# -------------------------
def normalize_window(seg: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize a window segment per-channel.
    method: "minmax" -> scales to [-1,1] per window
            "zscore" -> (x - mean) / std
            "None"   -> return unchanged
    """
    if method is None or method.lower() in ("none", "false"):
        return seg
    seg = seg.astype(np.float32)
    if method == "minmax":
        mn = seg.min(axis=0, keepdims=True)
        mx = seg.max(axis=0, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        out = 2.0 * (seg - mn) / rng - 1.0
        return out
    elif method == "zscore":
        mu = seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (seg - mu) / sd
    else:
        raise ValueError("Unknown normalization method: " + str(method))

# -------------------------
# Main dataset builder
# -------------------------
from typing import Tuple

def build_dataset_from_folder(root_folder: str,
                              window_size: int = 4000,
                              filename_glob: str = "*.csv",
                              label_parser: Callable[[str], int] = default_label_from_filename,
                              expected_cols: Optional[List[str]] = None,
                              fs: Optional[float] = None,
                              bp_limits: Optional[Tuple[float, float]] = None,
                              normalize: str = "minmax",
                              min_energy_ratio: float = 0.05,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Walk root_folder/<person>/*.csv and build dataset.
    Returns: (X, y, persons, file_paths)
      - X: (N, L, 4) float32
      - y: (N,) int (0/1)
      - persons: (N,) str (person id)
      - file_paths: list of source files (strings)
    Parameters:
      - fs, bp_limits: if you want to apply bandpass, set sampling freq and (low,high).
      - expected_cols: pass if CSV column names are nonstandard.
    """
    X_list = []
    y_list = []
    persons = []
    file_paths = []

    persons_dirs = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
    if verbose:
        print(f"[build_dataset] found {len(persons_dirs)} person folders in {root_folder}")

    for p in persons_dirs:
        pdir = os.path.join(root_folder, p)
        files = sorted(glob.glob(os.path.join(pdir, filename_glob)))
        if len(files) == 0:
            continue
        for fpath in files:
            try:
                arr = load_csv_iq(fpath, expected_cols=expected_cols)  # (N,4)
            except Exception as e:
                if verbose:
                    print(f"[WARN] skip {fpath}: load error: {e}")
                continue
            # optional preprocessing: DC remove
            arr = remove_dc(arr)
            # optional bandpass
            if fs is not None and bp_limits is not None:
                try:
                    arr = bandpass_iq(arr, fs, bp_limits[0], bp_limits[1], order=6)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] bandpass failed for {fpath}: {e}")
            # segmentation
            seg = energy_peak_segment(arr, window_size, frame=50, min_energy_ratio=min_energy_ratio)
            if seg is None:
                # skip low-quality / empty trials
                continue
            # normalization
            seg = normalize_window(seg, method=normalize)
            # append
            X_list.append(seg.astype(np.float32))
            y_list.append(int(label_parser(fpath)))
            persons.append(p)
            file_paths.append(fpath)

    if len(X_list) == 0:
        raise RuntimeError("No valid segments found. Check window_size and segmentation parameters.")

    X = np.stack(X_list, axis=0)  # (N, L, 4)
    y = np.array(y_list, dtype=np.int32)
    persons = np.array(persons, dtype=object)
    if verbose:
        print(f"[build_dataset] built dataset: N={len(X)}, window_size={window_size}, channels=4")
    return X, y, persons, file_paths

# -------------------------
# tf.data wrapper (optional)
# -------------------------
import tensorflow as tf

def dataset_from_arrays(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                        shuffle: bool = True, augment_fn: Optional[Callable] = None,
                        buffer_size: int = 2048) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from numpy arrays.
    augment_fn: callable that takes (x,y) and returns (x_aug,y) -- used only for training.
                It must work on numpy arrays (we use tf.numpy_function) or be a TF graph.
    """
    N = X.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    ds = ds.batch(batch_size)

    if augment_fn is not None:
        # wrap python augmentation function
        def _aug(batch_x, batch_y):
            # numpy_function requires explicit output types/shapes
            aug_x = tf.numpy_function(
                func=lambda bx, by: augment_fn(bx, by),
                inp=[batch_x, batch_y],
                Tout=tf.float32
            )
            aug_x.set_shape([None, X.shape[1], X.shape[2]])
            return aug_x, batch_y
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Utility: class weight compute
# -------------------------
def compute_class_weights(y: np.ndarray) -> dict:
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(enumerate(cw))

# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    DATA_ROOT = "./dataset"
    WINDOW = 4000
    try:
        X, y, persons, paths = build_dataset_from_folder(DATA_ROOT, window_size=WINDOW,
                                                        fs=None, bp_limits=None,
                                                        normalize="minmax", verbose=True)
        print("X.shape:", X.shape, "y.shape:", y.shape)
    except Exception as e:
        print("Error building dataset:", e)
