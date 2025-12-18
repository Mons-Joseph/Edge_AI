import os
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# -------------------------
# Utility: ensure output directory exists and save
# -------------------------
def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------------
# Figure 1: Raw radar signal (Valid vs Invalid)
# -------------------------
def plot_raw_signal(
    valid_segment: np.ndarray,
    invalid_segment: np.ndarray,
    out_path: str = "figures/fig1_raw_signal.png",
    sample_length: Optional[int] = None,
    title: str = "Raw Radar Signal: Valid vs Invalid Gesture",
) -> None:
    """
    Plot time-domain magnitude of one valid and one invalid segment.

    Parameters
    ----------
    valid_segment : (L, C) numpy array
    invalid_segment : (L, C) numpy array
    out_path : file path to save PNG (300 dpi)
    sample_length : if provided, truncate/pad to this length for visualization
    """
    # Basic checks
    if valid_segment is None or invalid_segment is None:
        raise ValueError("Both valid_segment and invalid_segment must be provided")

    def _prepare(x, L_target):
        if L_target is None:
            return x
        L = x.shape[0]
        if L == L_target:
            return x
        # simple center-crop or pad
        if L > L_target:
            start = (L - L_target) // 2
            return x[start : start + L_target]
        else:
            pad = np.zeros((L_target - L, x.shape[1]), dtype=x.dtype)
            return np.vstack([x, pad])

    valid = _prepare(valid_segment, sample_length)
    invalid = _prepare(invalid_segment, sample_length)

    mag_valid = np.sqrt(np.sum(valid**2, axis=1))
    mag_invalid = np.sqrt(np.sum(invalid**2, axis=1))

    _ensure_dir_for_file(out_path)
    plt.figure(figsize=(10, 3.5))
    plt.plot(mag_valid, label="Valid")
    plt.plot(mag_invalid, label="Invalid")
    plt.xlabel("Time (samples)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------
# Figure 2: Energy-based segmentation
# -------------------------
def plot_energy_segmentation(
    X_full: np.ndarray,
    L: int,
    frame: int = 50,
    out_path: str = "figures/fig2_energy_segmentation.png",
    title: str = "Energy-based Peak Segmentation",
) -> None:
    """
    Plot smoothed energy envelope and the extracted L-length window centered on the energy peak.

    Parameters
    ----------
    X_full : (N, C) full recording array
    L : target segment length (samples)
    frame : smoothing window for energy calculation (samples)
    """
    if X_full is None:
        raise ValueError("X_full must be provided")

    mag = np.sqrt(np.sum(X_full**2, axis=1))
    energy = np.convolve(mag**2, np.ones(frame), mode="same")
    peak = int(np.argmax(energy))
    start = max(0, peak - L // 2)
    end = start + L
    if end > len(X_full):
        end = len(X_full)
        start = max(0, end - L)

    _ensure_dir_for_file(out_path)
    plt.figure(figsize=(10, 3.5))
    plt.plot(energy, label="Smoothed energy")
    plt.axvline(peak, linestyle="--", label="Energy peak")
    plt.axvspan(start, end, color="gray", alpha=0.25, label=f"Window L={L}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------
# Figure 3: Time-domain augmentation (original vs stretched)
# -------------------------
def plot_time_augmentation(
    segment: np.ndarray,
    gamma_t: float = 1.2,
    stretched: np.ndarray = None,
) -> None:
    """
    Visualize original segment and time-stretched version.
    Requires augmentations.time_stretch_time_domain to be importable.
    """

    mag_orig = np.sqrt(np.sum(segment**2, axis=1))
    mag_aug = np.sqrt(np.sum(stretched**2, axis=1))

    # If shapes differ, resample magnitudes to same x-axis for plotting
    n = max(len(mag_orig), len(mag_aug))
    x_orig = np.linspace(0, 1, len(mag_orig))
    x_aug = np.linspace(0, 1, len(mag_aug))
    x_common = np.linspace(0, 1, n)

    import numpy as _np
    mag_orig_rs = np.interp(x_common, x_orig, mag_orig)
    mag_aug_rs = np.interp(x_common, x_aug, mag_aug)

    out_path = "figures/fig3_time_augmentation.png"

    _ensure_dir_for_file(out_path)
    plt.figure(figsize=(10, 3.5))
    plt.plot(mag_orig_rs, label="Original", alpha=0.9, linewidth=2)
    plt.plot(mag_aug_rs, label=f"Time-stretched (γ={gamma_t:.2f})",alpha=0.7, linestyle="--", linewidth=2)
    plt.xlabel("Normalized time")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("Time-domain Augmentation: Original vs Stretched")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------
# Figure 4: Frequency-domain augmentation (spectrogram before/after)
# -------------------------
def plot_freq_augmentation(
    gamma_f: float = 1.2,
    spec_orig: np.ndarray = None,
    spec_aug: np.ndarray = None,
) -> None:
    """
    Plot original spectrogram and Doppler-scaled spectrogram side-by-side (saved as two files).
    Requires segment_to_spectrogram and doppler_frequency_rescale_spectrogram.
    """
    out_orig = "figures/fig4a_spectrogram_original.png"
    out_aug = "figures/fig4b_spectrogram_doppler_scaled.png"
    # Save original
    _ensure_dir_for_file(out_orig)
    plt.figure(figsize=(8, 4))
    plt.imshow(spec_orig.T, aspect="auto", origin="lower")
    plt.title("Original Spectrogram")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig(out_orig, dpi=300)
    plt.close()

    # Save augmented
    _ensure_dir_for_file(out_aug)
    plt.figure(figsize=(8, 4))
    plt.imshow(spec_aug.T, aspect="auto", origin="lower")
    plt.title(f"{"Doppler-scaled Spectrogram"} (γ={gamma_f:.2f})")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig(out_aug, dpi=300)
    plt.close()


# -------------------------
# Figure 5 & 6: Confusion matrix
# -------------------------
def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: str = "figures/fig5_confusion_baseline.png",
    labels: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot normalized confusion matrix and save PNG.

    Parameters
    ----------
    y_true : ground-truth labels (0/1)
    y_pred : predicted labels (0/1)
    out_path : filename for saved PNG
    labels : optional label names, default ["Invalid", "Valid"]
    """
    if labels is None:
        labels = ["Invalid", "Valid"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    title = "Confusion Matrix",

    _ensure_dir_for_file(out_path)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Annotate cells
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({int(cm[i,j])})",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

