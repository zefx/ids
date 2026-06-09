#!/usr/bin/env python3
"""
train_xgb.py — XGBoost on Zeek conn.log features (CSE-CIC-IDS2018)

Save format: xgb_baseline.json  (native XGBoost — faster load, version-independent)
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
parser.add_argument('--data',         default='/data/cicids_zeek/ml_dataset.parquet')
parser.add_argument('--out',          default='/data/cicids_zeek/models/baseline')
parser.add_argument('--n_estimators', type=int, default=200)
parser.add_argument('--max_depth',    type=int, default=8)
parser.add_argument('--seed',         type=int, default=42)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'train_xgb.log'),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Features ──────────────────────────────────────────────────────────────────
NUM_FEATURES      = ['duration', 'orig_bytes', 'resp_bytes',
                     'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
CAT_FEATURES      = ['proto', 'service', 'conn_state', 'history']
PORT_FEATURES     = ['id.resp_p']
TEMPORAL_FEATURES = ['conn_rate', 'unique_dst_ports', 'mean_duration',
                     'std_duration', 'bytes_per_sec']
FEATURES          = NUM_FEATURES + CAT_FEATURES + PORT_FEATURES

# ── Load & preprocess ─────────────────────────────────────────────────────────
log.info(f'Loading {args.data}')
df = pd.read_parquet(args.data)
log.info(f'Loaded {len(df):,} rows')

# Auto-detect temporal features
temporal_present = [f for f in TEMPORAL_FEATURES if f in df.columns]
if temporal_present:
    log.info(f'Temporal features detected: {temporal_present}')
    NUM_FEATURES = NUM_FEATURES + temporal_present
    FEATURES     = NUM_FEATURES + CAT_FEATURES + PORT_FEATURES

for col in NUM_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    df[col] = np.log1p(df[col])

df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)

for col in CAT_FEATURES:
    df[col] = LabelEncoder().fit_transform(df[col].fillna('-').astype(str))

le_label = LabelEncoder()
y = le_label.fit_transform(df['label'])
X = df[FEATURES].values

log.info(f'Classes: {list(le_label.classes_)}')
for i, cnt in zip(*np.unique(y, return_counts=True)):
    log.info(f'  {le_label.classes_[i]:<30} {cnt:>10,} ({cnt/len(y)*100:.2f}%)')

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.seed, stratify=y)
log.info(f'Train: {len(X_train):,}  Test: {len(X_test):,}')

# ── Class weights → sample_weight (XGBoost doesn't support class_weight natively) ──
classes, counts = np.unique(y_train, return_counts=True)
weight_map = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
sample_weights = np.array([weight_map[c] for c in y_train])
log.info('Sample weights computed (inverse frequency)')

# ── Train ─────────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
except ImportError:
    log.error('xgboost not installed. Run: pip install xgboost --break-system-packages')
    sys.exit(1)

log.info(f'Training XGBoost (n_estimators={args.n_estimators}, max_depth={args.max_depth})...')
model = XGBClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=args.seed,
    verbosity=1,
    early_stopping_rounds=20,
)
t0 = time.time()
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=50,
)
elapsed = time.time() - t0
log.info(f'Done in {elapsed:.1f}s  (best iteration: {model.best_iteration})')

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
log.info(f'\n{classification_report(y_test, y_pred, target_names=le_label.classes_, digits=4)}')

f1_w = f1_score(y_test, y_pred, average='weighted')
f1_m = f1_score(y_test, y_pred, average='macro')
log.info(f'Weighted F1: {f1_w:.6f}  Macro F1: {f1_m:.6f}')

rd = classification_report(y_test, y_pred, target_names=le_label.classes_, output_dict=True)

# ── Save metrics JSON ─────────────────────────────────────────────────────────
metrics = {
    'model': 'XGBoost',
    'n_estimators': args.n_estimators,
    'best_iteration': int(model.best_iteration),
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
with open(OUT / 'metrics_xgb.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log.info(f'Metrics → {OUT}/metrics_xgb.json')

# ── Save model (native JSON — portable, version-independent) ──────────────────
model.save_model(OUT / 'xgb_baseline.json')
joblib.dump(le_label, OUT / 'label_encoder_xgb.joblib')
log.info(f'Model   → {OUT}/xgb_baseline.json')
log.info(f'Encoder → {OUT}/label_encoder_xgb.joblib')

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, normalize='true')
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay(cm, display_labels=le_label.classes_).plot(
    ax=ax, xticks_rotation=45, cmap='Blues', values_format='.2f')
ax.set_title(f'XGBoost — Normalized Confusion Matrix  (Weighted F1={f1_w:.4f})')
plt.tight_layout()
plt.savefig(OUT / 'confusion_matrix_xgb.png', dpi=150)
plt.close()
log.info(f'Confusion matrix → {OUT}/confusion_matrix_xgb.png')

# ── Feature importance ────────────────────────────────────────────────────────
importances = model.feature_importances_
idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([FEATURES[i] for i in idx], importances[idx], color='darkorange')
ax.set_xlabel('XGBoost feature importance (gain)')
ax.set_title('XGBoost — Feature Importances')
plt.tight_layout()
plt.savefig(OUT / 'feature_importance_xgb.png', dpi=150)
plt.close()
log.info(f'Feature importance → {OUT}/feature_importance_xgb.png')

log.info('=== XGBoost done ===')
