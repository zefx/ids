#!/usr/bin/env python3
"""
train_catboost.py — CatBoost on Zeek conn.log features (CSE-CIC-IDS2018)

Key difference: categorical features passed as raw strings — no LabelEncoder needed.
CatBoost handles them natively via ordered target statistics.

Save format: catboost_baseline.cbm  (native — faster load, portable, no Python needed)
"""

import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',       default='/data/cicids_zeek/ml_dataset.parquet')
parser.add_argument('--out',        default='/data/cicids_zeek/models/baseline')
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--depth',      type=int, default=6)   # 8 → OOM on 30.8M rows
parser.add_argument('--seed',       type=int, default=42)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'train_catboost.log'),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Features ──────────────────────────────────────────────────────────────────
NUM_FEATURES  = ['duration', 'orig_bytes', 'resp_bytes',
                 'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
CAT_FEATURES  = ['proto', 'service', 'conn_state', 'history']  # strings, no encoding
PORT_FEATURES = ['id.resp_p']
FEATURES      = NUM_FEATURES + CAT_FEATURES + PORT_FEATURES

# ── Load & preprocess ─────────────────────────────────────────────────────────
log.info(f'Loading {args.data}')
df = pd.read_parquet(args.data, columns=FEATURES + ['label'])
log.info(f'Loaded {len(df):,} rows')

for col in NUM_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    df[col] = np.log1p(df[col])

df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)

# CatBoost: keep categoricals as strings
for col in CAT_FEATURES:
    df[col] = df[col].fillna('-').astype(str)

# ── Sample 15M rows to avoid CatBoost OOM on full 30.8M ──────────────────────
SAMPLE_SIZE = 15_000_000
if len(df) > SAMPLE_SIZE:
    log.info(f'Sampling {SAMPLE_SIZE:,} rows from {len(df):,}...')
    df = df.sample(n=SAMPLE_SIZE, random_state=args.seed).reset_index(drop=True)
    log.info(f'After sampling: {len(df):,} rows')

le_label = LabelEncoder()
y = le_label.fit_transform(df['label'])
X = df[FEATURES].copy()

log.info(f'Classes: {list(le_label.classes_)}')
for i, cnt in zip(*np.unique(y, return_counts=True)):
    log.info(f'  {le_label.classes_[i]:<30} {cnt:>10,} ({cnt/len(y)*100:.2f}%)')

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.seed, stratify=y)
log.info(f'Train: {len(X_train):,}  Test: {len(X_test):,}')

# ── Train ─────────────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    log.error('catboost not installed. Run: pip install catboost --break-system-packages')
    sys.exit(1)

cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]
log.info(f'Cat feature indices: {cat_idx}')

train_pool = Pool(data=X_train, label=y_train,
                  cat_features=cat_idx, feature_names=FEATURES)
test_pool  = Pool(data=X_test,  label=y_test,
                  cat_features=cat_idx, feature_names=FEATURES)

log.info(f'Training CatBoost (iterations={args.iterations}, depth={args.depth})...')
model = CatBoostClassifier(
    iterations=args.iterations,
    depth=args.depth,
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Weighted',
    auto_class_weights='Balanced',
    border_count=32,        # default=254 → OOM; 32 sufficient for our features
    bootstrap_type='Bernoulli',  # required for subsample parameter
    subsample=0.8,
    thread_count=-1,
    random_seed=args.seed,
    verbose=50,
    early_stopping_rounds=30,
    task_type='CPU',
)
t0 = time.time()
model.fit(train_pool, eval_set=test_pool, use_best_model=True)
elapsed = time.time() - t0
log.info(f'Done in {elapsed:.1f}s  (best iteration: {model.best_iteration_})')

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(test_pool).flatten()
log.info(f'\n{classification_report(y_test, y_pred, target_names=le_label.classes_, digits=4)}')

f1_w = f1_score(y_test, y_pred, average='weighted')
f1_m = f1_score(y_test, y_pred, average='macro')
log.info(f'Weighted F1: {f1_w:.6f}  Macro F1: {f1_m:.6f}')

rd = classification_report(y_test, y_pred, target_names=le_label.classes_, output_dict=True)

# ── Save metrics JSON ─────────────────────────────────────────────────────────
metrics = {
    'model': 'CatBoost',
    'best_iteration': int(model.best_iteration_),
    'depth': args.depth,
    'train_time_s': round(elapsed, 1),
    'weighted_f1': round(f1_w, 6),
    'macro_f1': round(f1_m, 6),
    'per_class': {
        cls: {'precision': round(rd[cls]['precision'], 4),
              'recall':    round(rd[cls]['recall'], 4),
              'f1':        round(rd[cls]['f1-score'], 4),
              'support':   int(rd[cls]['support'])}
        for cls in le_label.classes_
    }
}
with open(OUT / 'metrics_catboost.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log.info(f'Metrics → {OUT}/metrics_catboost.json')

# ── Save model (native .cbm — no Python required to load) ────────────────────
model.save_model(str(OUT / 'catboost_baseline.cbm'))
joblib.dump(le_label, OUT / 'label_encoder_catboost.joblib')
log.info(f'Model   → {OUT}/catboost_baseline.cbm')
log.info(f'Encoder → {OUT}/label_encoder_catboost.joblib')

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, normalize='true')
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay(cm, display_labels=le_label.classes_).plot(
    ax=ax, xticks_rotation=45, cmap='Blues', values_format='.2f')
ax.set_title(f'CatBoost — Normalized Confusion Matrix  (Weighted F1={f1_w:.4f})')
plt.tight_layout()
plt.savefig(OUT / 'confusion_matrix_catboost.png', dpi=150)
plt.close()
log.info(f'Confusion matrix → {OUT}/confusion_matrix_catboost.png')

# ── Feature importance ────────────────────────────────────────────────────────
importances = model.get_feature_importance()
feat_names  = model.feature_names_
idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([feat_names[i] for i in idx], importances[idx], color='seagreen')
ax.set_xlabel('CatBoost importance (PredictionValuesChange)')
ax.set_title('CatBoost — Feature Importances')
plt.tight_layout()
plt.savefig(OUT / 'feature_importance_catboost.png', dpi=150)
plt.close()
log.info(f'Feature importance → {OUT}/feature_importance_catboost.png')

log.info('=== CatBoost done ===')
