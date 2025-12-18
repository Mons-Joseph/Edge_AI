"""
augmentations.py

Implements the three augmentation operations required by the thesis:
- time-domain stretch/compress (time-axis warping)
- Doppler-frequency scaling (operates in spectrogram domain)
- Gaussian noise (AWGN)

Also provides helper functions to generate augmented datasets (precomputed variants)
for experiments where we expand the training set. Designed to be deterministic when a
numpy random seed is set externally.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import zoom
import math
import random
from plots import plot_time_augmentation, plot_freq_augmentation
# -------------------------
# Time-domain: stretch/compress
# -------------------------
time_augmetation_plotted = False
def time_stretch_time_domain(segment: np.ndarray, gamma_t: float) -> np.ndarray:
    """
    Stretch/compress time-domain IQ segment and resample back to original length.
    - segment: (L, C)
    - gamma_t: 0.8..1.2 typically
    Returns (L, C)
    """
    global time_augmetation_plotted

    L, C = segment.shape
    new_len = max(1, int(round(L * gamma_t)))
    stretched = signal.resample(segment, new_len, axis=0)
    out = signal.resample(stretched, L, axis=0)

    if(not time_augmetation_plotted):
        plot_time_augmentation(
            segment=segment,
            gamma_t=gamma_t,
            stretched=stretched
        )
        time_augmetation_plotted = True


    return out.astype(np.float32)

# -------------------------
# AWGN
# -------------------------
def add_awgn(segment: np.ndarray, snr_db: float) -> np.ndarray:
    x = segment.astype(np.float32)
    sig_power = np.mean(x**2)
    if sig_power <= 1e-12:
        return x
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(scale=math.sqrt(noise_power), size=x.shape).astype(np.float32)
    return x + noise

# -------------------------
# Spectrogram helpers
# -------------------------
def segment_to_spectrogram(segment: np.ndarray, fs: float = 100.0,
                           nperseg: int = 256, noverlap: int = 128) -> np.ndarray:
    mag = np.sqrt(np.sum(segment**2, axis=1))
    f, t, Sxx = signal.spectrogram(mag, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
    return Sxx.T.astype(np.float32)  # (T, F)

def doppler_frequency_rescale_spectrogram(spec: np.ndarray, gamma_f: float,
                                          preserve_center_bins: int = 3) -> np.ndarray:
    T, F = spec.shape
    bins = np.arange(F) - (F - 1) / 2.0
    src_positions = bins / gamma_f
    out = np.zeros_like(spec)
    preserve_mask = np.abs(bins) <= preserve_center_bins
    for t in range(T):
        col = spec[t, :]
        new_col = np.interp(bins, src_positions, col, left=0.0, right=0.0)
        if preserve_mask.any():
            new_col[preserve_mask] = col[preserve_mask]
        out[t, :] = new_col
    return out

def spectrogram_to_pseudo_time(spec: np.ndarray, L_target: int, fs: float = 100.0,
                               nperseg: int = 256, noverlap: int = 128) -> np.ndarray:
    # Create complex STFT with random phase and inverse
    Sxx = spec.T
    phase = np.exp(1j * 2 * np.pi * np.random.rand(*Sxx.shape)).astype(np.complex64)
    S_complex = Sxx * phase
    _, x = signal.istft(S_complex, fs=fs, nperseg=nperseg, noverlap=noverlap)
    x = np.real(x)
    if len(x) < L_target:
        pad = np.zeros((L_target - len(x),), dtype=np.float32)
        x = np.concatenate([x, pad])
    else:
        x = x[:L_target]
    return np.tile(x.reshape(-1,1), (1,4)).astype(np.float32)

# -------------------------
# Dataset augmentation generator (precompute variants)
# -------------------------
plotted_freq_augmentation = False
def generate_augmented_dataset(X: np.ndarray, y: np.ndarray,
                               n_per_sample: int = 2,
                               time_range: tuple = (0.9, 1.1),
                               freq_range: tuple = (0.9, 1.1),
                               snr_range: tuple = (25, 35),
                               fs: float = 100.0,
                               use_time: bool = True,
                               use_freq: bool = True,
                               use_noise: bool = True,
                               seed: int = 42) -> tuple:
    """
    For each sample in X, create `n_per_sample` augmented variants using the enabled methods.
    Returns X_aug (original + variants), y_aug.
    """
    global plotted_freq_augmentation

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    N, L, C = X.shape
    total = N * (1 + n_per_sample)
    X_aug = np.zeros((total, L, C), dtype=np.float32)
    y_aug = np.zeros((total,), dtype=y.dtype)

    idx = 0
    for i in range(N):
        X_aug[idx] = X[i]
        y_aug[idx] = y[i]
        idx += 1
        for k in range(n_per_sample):
            x = X[i].copy()
            # apply time
            if use_time:
                gamma_t = float(rng.uniform(*time_range))
                x = time_stretch_time_domain(x, gamma_t)
            # apply noise
            if use_noise:
                snr = float(rng.uniform(*snr_range))
                x = add_awgn(x, snr)
            # apply freq via spectrogram warping and back projection
            if use_freq:
                gamma_f = float(rng.uniform(*freq_range))
                spec = segment_to_spectrogram(x, fs=fs)
                spec2 = doppler_frequency_rescale_spectrogram(spec, gamma_f)
                if(not plotted_freq_augmentation):
                    plot_freq_augmentation(gamma_f= gamma_f,spec_orig= spec,spec_aug= spec2,)
                    plotted_freq_augmentation = True
                x = spectrogram_to_pseudo_time(spec2, L_target=L, fs=fs)
            X_aug[idx] = x
            y_aug[idx] = y[i]
            idx += 1
    return X_aug, y_aug


if __name__ == '__main__':
    print('augmentations module loaded')
