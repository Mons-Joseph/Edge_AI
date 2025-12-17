"""
augmentation_experiments.py

Run the 4 augmentation experiments required by the thesis:
- baseline (no augmentation)
- time-only
- freq-only
- noise-only
- combined (time+freq+noise)

This script reuses data_loader.build_dataset_from_folder and cnn_baseline.make_cnn_1d
and trains the same 1D CNN on an expanded training set per experiment.

Augmentation is applied by precomputing `n_per_sample` variants per training sample
(using augmentations.generate_augmented_dataset) so we can use class_weight in model.fit.

Usage:
    python augmentation_experiments.py --data_root ./dataset --window_size 4000 --epochs 20

"""

import os
import argparse
import json
import numpy as np
import random
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# import project modules
from data_loader import build_dataset_from_folder
from cnn_baseline import make_cnn_1d, set_seed
from augmentation_helper import generate_augmented_dataset

import tensorflow as tf

# -------------------------
# Experiment runner
# -------------------------

def run_one_experiment(X, y, persons, tr_idx, te_idx, experiment_name, args,
                       n_per_sample=2, seed=42,
                       use_time=False, use_freq=False, use_noise=False):
    set_seed(seed)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # Build augmented training set (precomputed variants)
    if experiment_name == 'baseline':
        Xtr_aug, ytr_aug = Xtr, ytr
    else:
        Xtr_aug, ytr_aug = generate_augmented_dataset(Xtr, ytr, n_per_sample=n_per_sample,
                                                      use_time=use_time, use_freq=use_freq, use_noise=use_noise,
                                                      seed=seed)

    # compute class weights
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(ytr_aug), y=ytr_aug)
    class_weight_dict = dict(enumerate(cw))

    # model
    model = make_cnn_1d(X.shape[1], X.shape[2])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(f"best_{experiment_name}.h5", monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(Xtr_aug, ytr_aug, validation_data=(Xte, yte),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, class_weight=class_weight_dict, verbose=2)

    # evaluate
    y_prob = model.predict(Xte).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(yte, y_prob)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(yte, y_pred)

    out = {'experiment': experiment_name,
           'n_train': int(len(Xtr_aug)), 'n_test': int(len(Xte)),
           'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'auc': float(auc),
           'confusion_matrix': cm.tolist(), 'history': history.history}

    # save per-experiment json
    outpath = f"results_{experiment_name}.json"
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {outpath}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./dataset')
    p.add_argument('--window_size', type=int, default=4000)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_per_sample', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    X, y, persons, paths = build_dataset_from_folder(args.data_root, window_size=args.window_size,
                                                    normalize='minmax', verbose=True)
    print('Loaded dataset:', X.shape, y.shape)

    # single person-wise split (use same split across experiments)
    gss = GroupShuffleSplit(test_size=0.2, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=persons))

    experiments = [
        ('baseline', False, False, False),
        ('time_only', True, False, False),
        ('freq_only', False, True, False),
        ('noise_only', False, False, True),
        ('combined', True, True, True)
    ]

    all_results = []
    for name, ut, uf, un in experiments:
        print('\n==============================')
        print('Running experiment:', name)
        res = run_one_experiment(X, y, persons, tr_idx, te_idx, name, args,
                                 n_per_sample=args.n_per_sample, seed=args.seed,
                                 use_time=ut, use_freq=uf, use_noise=un)
        all_results.append(res)

    # aggregate summary CSV
    import pandas as pd
    rows = []
    for r in all_results:
        rows.append({'experiment': r['experiment'], 'n_train': r['n_train'], 'n_test': r['n_test'],
                     'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1'], 'auc': r['auc']})
    df = pd.DataFrame(rows)
    df.to_csv('augmentation_experiments_summary.csv', index=False)
    print('\nSaved augmentation_experiments_summary.csv')


if __name__ == '__main__':
    main()
