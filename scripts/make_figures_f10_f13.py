#!/usr/bin/env python3
"""
make_figures_f10_f13.py — generates 4 figures in unified grayscale style for thesis §3.3/§3.4:

    F10  confusion matrix XGBoost baseline   (§3.3, normalized by true class)
    F11  confusion matrix XGBoost temporal   (§3.4, normalized by true class)
    F12  ROC curves per-class, temporal      (§3.4, one-vs-rest)
    F13  Precision-Recall curves per-class   (§3.4, one-vs-rest)

Style: scripts/plot_style.py (Greys cmap, Times New Roman, DPI 300, FIG_SQUARE).
Test split is reproduced from train_xgb.py logic: 80/20 stratified, random_state=42.

Run on usrv01 (parquets live in /data/cicids_zeek/):

    cd ~/HSE-Thesis-26/scripts
    python3 make_figures_f10_f13.py \
        --data-baseline /data/cicids_zeek/ml_dataset.parquet \
        --data-temporal /data/cicids_zeek/ml_dataset_temporal.parquet \
        --baseline-model ../models/baseline/xgb_baseline.json \
        --temporal-model ../models/temporal/xgb_baseline.json \
        --out ../figures

Outputs:
    figures/F10_confusion_baseline.png
    figures/F11_confusion_temporal.png
    figures/F12_roc_temporal.png
    figures/F13_pr_temporal.png
    figures/roc_pr_metrics.json   (AUC + AP per class)

Wall-clock estimate: 5-15 min (predict + predict_proba on ~6M row test split).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make plot_style importable when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, FIG_SQUARE, PALETTE  # noqa: E402
apply_style()

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier  # noqa: E402


NUM_BASELINE = [
    'duration', 'orig_bytes', 'resp_bytes',
    'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes',
]
CAT_FEATURES = ['proto', 'service', 'conn_state', 'history']
PORT_FEATURES = ['id.resp_p']
TEMPORAL = [
    'conn_rate', 'unique_dst_ports', 'mean_duration',
    'std_duration', 'bytes_per_sec',
]
SEED = 42
TEST_SIZE = 0.2


def load_split(data_path: Path, use_temporal: bool):
    """Reproduce train_xgb.py preprocessing + train/test split, return X_test, y_test, classes."""
    print(f'  Loading {data_path}...', flush=True)
    t0 = time.time()
    df = pd.read_parquet(data_path)
    print(f'  Loaded {len(df):,} rows in {time.time()-t0:.1f}s', flush=True)

    num = NUM_BASELINE.copy()
    if use_temporal:
        present = [f for f in TEMPORAL if f in df.columns]
        if not present:
            raise RuntimeError(
                f'Temporal features requested but none of {TEMPORAL} found in parquet'
            )
        num += present
        print(f'  Using temporal features: {present}', flush=True)
    features = num + CAT_FEATURES + PORT_FEATURES

    for col in num:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        df[col] = np.log1p(df[col])
    df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)
    for col in CAT_FEATURES:
        df[col] = LabelEncoder().fit_transform(df[col].fillna('-').astype(str))

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    x = df[features].values
    _, x_te, _, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f'  Test set: {len(x_te):,} rows, {len(le.classes_)} classes', flush=True)
    return x_te, y_te, list(le.classes_), features


def short_label(cls: str) -> str:
    """Shorten verbose class names for axis labels."""
    return (
        cls.replace('DoS-', '')
           .replace('DDOS-', '')
           .replace('-Bruteforce', '-BF')
           .replace('-BruteForce', '-BF')
    )


def make_confusion(y_test, y_pred, classes, out_path: Path, caption: str):
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    n = len(classes)

    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    im = ax.imshow(cm, cmap='Greys', vmin=0.0, vmax=1.0, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [short_label(c) for c in classes]
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel('Predicted class', fontsize=9)
    ax.set_ylabel('True class', fontsize=9)
    ax.grid(False)
    ax.tick_params(top=False, right=False)

    # Annotate every cell with the value; switch to white when background is dark.
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            txt = '0' if val < 0.005 else (f'{val:.2f}' if val < 1.0 else '1.0')
            color = 'white' if val >= 0.55 else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Recall (row-normalized)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}  ({caption})', flush=True)


def make_roc(y_test, y_proba, classes, out_path: Path):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    auc_per_class = {}
    for i in range(n):
        y_bin = (y_test == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_per_class[classes[i]] = float(roc_auc)
        color = PALETTE[i % len(PALETTE)]
        ls = '-' if i < 4 else '--'  # dash for the lighter half so they read in print
        ax.plot(fpr, tpr, color=color, linewidth=1.1, linestyle=ls,
                label=f'{short_label(classes[i])} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], color='#999999', linestyle=':', linewidth=0.7, label='Chance')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=6.5, frameon=False, ncol=1)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}', flush=True)
    return auc_per_class


def make_pr(y_test, y_proba, classes, out_path: Path):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    ap_per_class = {}
    for i in range(n):
        y_bin = (y_test == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        ap = average_precision_score(y_bin, y_proba[:, i])
        ap_per_class[classes[i]] = float(ap)
        color = PALETTE[i % len(PALETTE)]
        ls = '-' if i < 4 else '--'
        ax.plot(rec, prec, color=color, linewidth=1.1, linestyle=ls,
                label=f'{short_label(classes[i])} (AP={ap:.4f})')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize=6.5, frameon=False, ncol=1)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}', flush=True)
    return ap_per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-baseline', default='/data/cicids_zeek/ml_dataset.parquet',
                   help='Path to baseline parquet (no temporal columns)')
    p.add_argument('--data-temporal', default='/data/cicids_zeek/ml_dataset_temporal.parquet',
                   help='Path to temporal parquet (with conn_rate / unique_dst_ports / ...)')
    p.add_argument('--baseline-model', default='../models/baseline/xgb_baseline.json',
                   help='XGBoost baseline model JSON')
    p.add_argument('--temporal-model', default='../models/temporal/xgb_baseline.json',
                   help='XGBoost temporal model JSON')
    p.add_argument('--out', default='../figures',
                   help='Output directory for PNGs and metrics JSON')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── F10 — baseline confusion ──────────────────────────────────────────
    print('[F10] baseline confusion matrix')
    x_te, y_te, classes_b, _ = load_split(Path(args.data_baseline), use_temporal=False)
    clf_b = XGBClassifier()
    clf_b.load_model(args.baseline_model)
    print('  predict()...', flush=True)
    y_pred_b = clf_b.predict(x_te)
    make_confusion(y_te, y_pred_b, classes_b,
                   out / 'F10_confusion_baseline.png',
                   'XGBoost baseline, normalized by true class')

    # ── F11/F12/F13 — temporal ─────────────────────────────────────────────
    print('\n[F11/F12/F13] temporal — confusion, ROC, PR')
    x_te, y_te, classes_t, _ = load_split(Path(args.data_temporal), use_temporal=True)
    clf_t = XGBClassifier()
    clf_t.load_model(args.temporal_model)

    print('  predict()...', flush=True)
    y_pred_t = clf_t.predict(x_te)
    make_confusion(y_te, y_pred_t, classes_t,
                   out / 'F11_confusion_temporal.png',
                   'XGBoost temporal, normalized by true class')

    print('  predict_proba()...', flush=True)
    t0 = time.time()
    y_proba_t = clf_t.predict_proba(x_te)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    auc_pc = make_roc(y_te, y_proba_t, classes_t, out / 'F12_roc_temporal.png')
    ap_pc = make_pr(y_te, y_proba_t, classes_t, out / 'F13_pr_temporal.png')

    # Save AUC + AP per class as JSON for interpretation paragraphs
    metrics = {
        'model': 'XGBoost temporal',
        'split': {'test_size': TEST_SIZE, 'random_state': SEED, 'stratified': True},
        'auc_per_class': auc_pc,
        'ap_per_class': ap_pc,
    }
    with open(out / 'roc_pr_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\n  Saved {out / "roc_pr_metrics.json"}', flush=True)

    print('\nAll 4 figures generated. Send the .png files + roc_pr_metrics.json back '
          'for interpretation paragraphs.', flush=True)


if __name__ == '__main__':
    main()
