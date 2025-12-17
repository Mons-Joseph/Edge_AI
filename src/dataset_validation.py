# step1_data_check_and_load.py
import os, glob, hashlib, pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Import the loader (paste or import from data_loader.py if you saved it)
# from data_loader import build_dataset_from_folder, load_csv_iq, remove_dc, normalize_window
# If you haven't saved data_loader.py, paste its functions above and import here.
# For brevity, this script assumes build_dataset_from_folder is available in the PATH.

DATA_ROOT = "./dataset"     # edit if needed
SAMPLE_CHECK_COUNT = 3      # how many files per person to inspect
WINDOW = 4000               # initial window size to test segmentation
EXPECTED_MIN_COLS = 4       # at least I_low, Q_low, I_up, Q_up

def sha1_of_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1048576)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def find_person_folders(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def basic_tree_report(root):
    people = find_person_folders(root)
    print(f"Found {len(people)} person folders.\n")
    counts = {}
    label_counter = Counter()
    for p in people:
        files = sorted(glob.glob(os.path.join(root, p, "*.csv")))
        counts[p] = len(files)
        # label inference from filename
        for f in files:
            fname = os.path.basename(f).lower()
            label = "valid" if "valid" in fname else "invalid" if "invalid" in fname else "unknown"
            label_counter[label] += 1
    pprint.pprint(counts)
    print("\nLabel counts (inferred from filenames):")
    pprint.pprint(label_counter)
    return people, counts

def check_duplicates_and_sizes(root, people, max_check=500):
    print("\nChecking SHA1 duplicates and file sizes...")
    hash_map = defaultdict(list)
    sizes = []
    checked = 0
    for p in people:
        files = sorted(glob.glob(os.path.join(root, p, "*.csv")))
        for f in files:
            sizes.append(os.path.getsize(f))
            h = sha1_of_file(f)
            hash_map[h].append(f)
            checked += 1
            if checked >= max_check:
                break
        if checked >= max_check:
            break
    dup_groups = [v for v in hash_map.values() if len(v) > 1]
    if dup_groups:
        print("WARNING: Duplicate file contents detected (identical SHA1). Examples:")
        for g in dup_groups[:5]:
            pprint.pprint(g)
    else:
        print("No identical-file duplicates found (by SHA1) in first", checked, "files.")
    print("File size stats (bytes): min, median, mean, max ->",
          np.min(sizes), np.median(sizes), np.mean(sizes), np.max(sizes))

def sample_csv_inspect(root, people, samples_per_person=1):
    print("\nSampling a few CSV files and printing column + value stats...")
    issues = []
    checked = 0
    for p in people:
        files = sorted(glob.glob(os.path.join(root, p, "*.csv")))
        for f in files[:samples_per_person]:
            checked += 1
            try:
                df = pd.read_csv(f)
            except Exception as e:
                issues.append((f, "read_error", str(e)))
                continue
            cols = list(df.columns)
            ncols = len(cols)
            if ncols < EXPECTED_MIN_COLS:
                issues.append((f, "bad_columns", cols))
            # simple checks
            has_nan = df.isna().any().any()
            if has_nan:
                issues.append((f, "has_nan", None))
            # lengths
            L = len(df)
            if L < 10:
                issues.append((f, "short_length", L))
            # print summary for first few
            print(f"\nFile: {f}")
            print(" Columns:", cols)
            print(" Length:", L)
            print(" Head values:")
            print(df.head(3).to_string(index=False))
            # simple numeric stats for the first 4 numeric cols
            numeric = df.select_dtypes(include=[float, int])
            if not numeric.empty:
                stats = numeric.iloc[:, :min(4, numeric.shape[1])].describe().loc[["min","mean","std","max"]]
                print(" Stats (first 4 numeric cols):")
                print(stats.to_string())
        if checked >= SAMPLE_CHECK_COUNT * len(people):
            break
    if issues:
        print("\nPotential issues found (examples):")
        pprint.pprint(issues[:10])
    else:
        print("\nNo immediate CSV-level issues detected in sampled files.")

def run_loader_and_report(root, window_size=WINDOW):
    print("\nRunning dataset builder (loader) to produce segments. This will apply segmentation and normalization.")
    try:
        from data_loader import build_dataset_from_folder, normalize_window  # expects you saved the loader
    except Exception as e:
        print("Error importing data_loader. Make sure data_loader.py is in the same folder or PYTHONPATH.")
        print(e)
        return None, None, None, None

    X, y, persons, paths = build_dataset_from_folder(root, window_size=window_size,
                                                    fs=None, bp_limits=None,
                                                    normalize="minmax", verbose=True)
    print("\nLoader produced:")
    print(" X.shape:", X.shape)
    print(" y.shape:", y.shape)
    print(" Unique persons:", len(np.unique(persons)))
    print(" Label distribution:", Counter(y))
    # sample one segment and plot waveform + spectrogram
    import matplotlib.pyplot as plt
    import scipy.signal as signal
    if X.shape[0] > 0:
        i = 0
        seg = X[i]  # (L,4)
        t = np.arange(seg.shape[0])
        plt.figure(figsize=(12,4))
        for ch in range(seg.shape[1]):
            plt.plot(t, seg[:,ch] + ch*2.5, label=f"ch{ch}")
        plt.title(f"Segment waveform (sample 0) label={y[0]} person={persons[0]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("sample_segment_waveform.png")
        print("Saved sample_segment_waveform.png")

        # spectrogram of magnitude (sum across channels)
        mag = np.sqrt(np.sum(seg**2, axis=1))
        f, tt, S = signal.spectrogram(mag, fs=100.0, nperseg=256, noverlap=128)  # fs is illustrative
        plt.figure(figsize=(6,4))
        plt.pcolormesh(tt, f, 10*np.log10(S+1e-12), shading='gouraud')
        plt.ylabel('freq [Hz]')
        plt.xlabel('time [sec]')
        plt.title("Spectrogram (magnitude) sample 0")
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.savefig("sample_segment_spectrogram.png")
        print("Saved sample_segment_spectrogram.png")
    return X, y, persons, paths

if __name__ == "__main__":
    people, counts = basic_tree_report(DATA_ROOT)
    check_duplicates_and_sizes(DATA_ROOT, people, max_check=500)
    sample_csv_inspect(DATA_ROOT, people, samples_per_person=1)
    X, y, persons, paths = run_loader_and_report(DATA_ROOT, window_size=WINDOW)
    if X is None:
        print("Loader failed — fix data_loader import or paste loader functions here.")
    else:
        print("\nStep 1 complete. If the plots look sane, proceed to augmentation coding (step 2).")
