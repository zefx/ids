#!/usr/bin/env python3
"""
make_dataset.py — Step 3: Convert labeled.csv → ml_dataset.parquet

Pipeline:
  1. process_cicids.sh  → raw Zeek conn.log
  2. label_all.sh       → combined/labeled.csv  (raw strings: proto='tcp', conn_state='SF', ...)
  3. make_dataset.py    → ml_dataset.parquet     (RAW values, NO log1p — same as build_temporal_dataset.py)
  4. train_*.py         → models (apply log1p + LabelEncoder in-place during training)
  5. save_cat_encoders.py → label_encoder_cats_*.joblib (same LabelEncoder → same mapping)
  6. consumer.py        → real-time predictions (applies log1p once — matches training)

IMPORTANT: Numeric features are stored as RAW values (no log1p).
log1p is applied only in train_*.py and consumer.py — once, consistently.
This matches build_temporal_dataset.py behavior.

IMPORTANT: Categorical features (proto, service, conn_state, history) are stored
as RAW STRINGS in parquet. DO NOT encode them here. Training scripts apply
LabelEncoder themselves and consumer.py needs the same mapping.

Usage:
    python3 make_dataset.py
    python3 make_dataset.py --csv /data/cicids_zeek/combined/labeled.csv
    python3 make_dataset.py --csv /data/cicids_zeek/combined/labeled.csv \\
                            --out /data/cicids_zeek/ml_dataset.parquet
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = '/data/cicids_zeek/combined/labeled.csv'
DEFAULT_OUT = '/data/cicids_zeek/ml_dataset.parquet'

NUM_FEATURES  = ['duration', 'orig_bytes', 'resp_bytes',
                  'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
CAT_FEATURES  = ['proto', 'service', 'conn_state', 'history']
PORT_FEATURES = ['id.resp_p']
META_FEATURES = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h']
KEEP_COLS     = META_FEATURES + NUM_FEATURES + CAT_FEATURES + PORT_FEATURES + ['label']

# No label merging — DoS-GoldenEye and DoS-Slowloris kept as separate classes
# to match build_temporal_dataset.py (9 classes, consistent train/temporal comparison).
# Removed Apr 5 2026: was merging GoldenEye+Slowloris → 'DoS-Slow' (wrong decision).
LABEL_MERGE = {}

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Convert labeled.csv to ml_dataset.parquet')
parser.add_argument('--csv', default=DEFAULT_CSV, help='Input labeled.csv')
parser.add_argument('--out', default=DEFAULT_OUT, help='Output parquet path')
# log1p is NOT applied here — train_*.py applies it at load time.
# This matches build_temporal_dataset.py (both store raw values).
# Fixed Apr 5 2026: was applying log1p by default → double log1p bug
# (make_dataset saved log1p(raw), train_*.py applied log1p again → log1p(log1p(raw))).
# Consumer applies log1p once → mismatched with model trained on log1p(log1p(raw)).
args = parser.parse_args()
args.no_log1p = True  # always store raw — log1p is responsibility of train_*.py

csv_path = Path(args.csv)
out_path = Path(args.out)

if not csv_path.exists():
    sys.exit(f'ERROR: {csv_path} not found. Run label_all.sh first.')

# ── Load ──────────────────────────────────────────────────────────────────────
print(f'Loading {csv_path} ...')
df = pd.read_csv(csv_path, low_memory=False)
print(f'  Rows: {len(df):,}  Cols: {len(df.columns)}')

# Keep only columns we need (others may not exist in all days)
available = [c for c in KEEP_COLS if c in df.columns]
missing   = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    print(f'  WARNING: columns not found in CSV (will be skipped): {missing}')
df = df[available].copy()

# ── Merge label variants ───────────────────────────────────────────────────────
df['label'] = df['label'].replace(LABEL_MERGE)
print('\nClass distribution:')
for lbl, cnt in df['label'].value_counts().items():
    print(f'  {lbl:<40} {cnt:>10,}  ({cnt/len(df)*100:.2f}%)')

# ── Numeric preprocessing ─────────────────────────────────────────────────────
print('\nPreprocessing numeric features ...')
for col in [c for c in NUM_FEATURES if c in df.columns]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    if not args.no_log1p:
        df[col] = np.log1p(df[col])

# Port: integer, bounded 0-65535
if 'id.resp_p' in df.columns:
    df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)

# ── Categorical: keep as strings, fill nulls ──────────────────────────────────
# DO NOT encode here. Training scripts apply LabelEncoder themselves.
# save_cat_encoders.py will read these strings and create matching encoders.
print('Categorical features (stored as raw strings):')
for col in [c for c in CAT_FEATURES if c in df.columns]:
    df[col] = df[col].fillna('-').astype(str)
    n_unique = df[col].nunique()
    sample   = sorted(df[col].unique())[:6]
    print(f'  {col:<12}: {n_unique} unique → {sample}...')

# Verify proto looks like Zeek strings, not integers
if 'proto' in df.columns:
    sample_proto = df['proto'].dropna().iloc[0]
    try:
        int(sample_proto)
        print(f'\n  ERROR: proto value is "{sample_proto}" — looks like an integer!')
        print('  This means labeled.csv was already encoded. Use raw conn.log → label_all.sh.')
        sys.exit(1)
    except ValueError:
        print(f'\n  proto check: "{sample_proto}" ✓  (raw Zeek string)')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path.parent.mkdir(parents=True, exist_ok=True)
print(f'\nSaving to {out_path} ...')
df.to_parquet(out_path, index=False, compression='snappy')

size_mb = out_path.stat().st_size / 1024**2
print(f'Done. {len(df):,} rows → {out_path}  ({size_mb:.1f} MB)')
print()
print('Next steps:')
print('  python3 save_cat_encoders.py   # fit LabelEncoders from this parquet')
print('  python3 train_baseline.py      # train RF + XGB')
print('  python3 train_xgb.py           # train XGB standalone')
