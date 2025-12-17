"""
cnn_baseline.py

Thesis-ready 1D-CNN baseline training script.

Features:
- Loads per-trial CSV dataset using data_loader.build_dataset_from_folder
- Person-wise GroupShuffleSplit (80/20) to produce train/test
- Per-window normalization done by the loader (minmax default)
- 1D CNN with BatchNorm + Dropout
- EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Class-weighting for imbalance
- Reproducible seed control
- Saves model and a results CSV (metrics)

Usage (example):
    python cnn_baseline.py --data_root ./dataset --window_size 4000 --epochs 30

Notes:
- This script expects `data_loader.py` to be present in the same folder or importable.
- Augmentation is NOT applied here (baseline run only).
"""

import os
import argparse
import random
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight

# -------------------------------
# Reproducibility helper
# -------------------------------
def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Note: full determinism not guaranteed across TF versions/hardware

# -------------------------------
# CNN model (1D) - baseline
# -------------------------------
def make_cnn_1d(L: int, channels: int = 4):
    inp = layers.Input(shape=(L, channels))
    x = layers.Conv1D(16, kernel_size=7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(32, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# -------------------------------
# Training + evaluation
# -------------------------------

def train_and_evaluate(X, y, persons, args):
    # person-wise split
    gss = GroupShuffleSplit(test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=persons))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    persons_tr, persons_te = persons[tr_idx], persons[te_idx]

    # class weights
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    class_weight_dict = dict(enumerate(cw))

    # model
    model = make_cnn_1d(X.shape[1], X.shape[2])
    model.summary()

    # callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(args.model_out, monitor='val_loss', save_best_only=True)
    ]

    # fit
    history = model.fit(Xtr, ytr,
                        validation_data=(Xte, yte),
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight_dict,
                        verbose=2)

    # evaluate
    y_prob = model.predict(Xte).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(yte, y_prob)
    except Exception:
        auc = float('nan')

    results = {
        'n_train': int(len(Xtr)),
        'n_test': int(len(Xte)),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist()
    }

    # save results
    out_csv = args.results_csv
    df = pd.DataFrame([{
        'seed': args.seed,
        'n_train': results['n_train'],
        'n_test': results['n_test'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'auc': results['auc']
    }])
    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)

    # save model metrics and history
    with open(args.history_json, 'w') as f:
        json.dump(history.history, f, indent=2)
    print('\nTraining complete. Results:')
    print(json.dumps(results, indent=2))
    return model, results

# -------------------------------
# CLI and main
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='1D-CNN baseline trainer for radar gestures')
    p.add_argument('--data_root', type=str, default='./dataset', help='root folder of dataset')
    p.add_argument('--window_size', type=int, default=4000, help='segment window size (samples)')
    p.add_argument('--fs', type=float, default=None, help='sampling frequency (Hz) if bandpass used (optional)')
    p.add_argument('--bp_low', type=float, default=None, help='bandpass low Hz (optional)')
    p.add_argument('--bp_high', type=float, default=None, help='bandpass high Hz (optional)')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model_out', type=str, default='best_model.h5')
    p.add_argument('--results_csv', type=str, default='baseline_results.csv')
    p.add_argument('--history_json', type=str, default='history.json')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # import loader
    try:
        from data_loader import build_dataset_from_folder
    except Exception as e:
        print('ERROR: cannot import data_loader. Make sure data_loader.py is in the same folder.')
        raise

    bp_limits = None
    if args.fs and args.bp_low and args.bp_high:
        bp_limits = (args.bp_low, args.bp_high)

    X, y, persons, paths = build_dataset_from_folder(args.data_root, window_size=args.window_size,
                                                    fs=args.fs, bp_limits=bp_limits,
                                                    normalize='minmax', verbose=True)

    # quick sanity
    print('Dataset shapes:', X.shape, y.shape)

    # train + eval
    model, results = train_and_evaluate(X, y, persons, args)

if __name__ == '__main__':
    main()
