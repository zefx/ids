"""
save_cat_encoders.py — Step 3b: Fit categorical encoders from ml_dataset.parquet

Must be run AFTER make_dataset.py (which stores raw Zeek strings in parquet).
The LabelEncoder fitted here uses the same fit logic as train_*.py scripts,
so the integer mapping is guaranteed identical → consumer.py predictions are correct.

Saves label_encoder_cats_{rf,xgb,lgbm}.joblib to BOTH:
  /data/cicids_zeek/models/baseline/   ← for baseline consumer
  /data/cicids_zeek/models/temporal/   ← for temporal consumer
Categorical features are identical in both datasets (same Zeek source),
so one fit → two destinations.

Each file is a dict {col: LabelEncoder} for proto, service, conn_state, history.

Usage:
    python3 save_cat_encoders.py
    python3 save_cat_encoders.py --data /data/cicids_zeek/ml_dataset.parquet
    python3 save_cat_encoders.py --data /data/cicids_zeek/ml_dataset.parquet \\
                                 --outdir /data/cicids_zeek/models/baseline \\
                                 --outdir-temporal /data/cicids_zeek/models/temporal
"""

import argparse
import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

CAT_FEATURES = ['proto', 'service', 'conn_state', 'history']
MODEL_TYPES  = ['rf', 'xgb', 'lgbm']

parser = argparse.ArgumentParser()
parser.add_argument('--data',             default='/data/cicids_zeek/ml_dataset.parquet')
parser.add_argument('--outdir',           default='/data/cicids_zeek/models/baseline')
parser.add_argument('--outdir-temporal',  default='/data/cicids_zeek/models/temporal')
args = parser.parse_args()

data_path = Path(args.data)
if not data_path.exists():
    sys.exit(f'ERROR: {data_path} not found. Run make_dataset.py first.')

print(f'Reading {data_path} ...')
df = pd.read_parquet(data_path, columns=CAT_FEATURES)
print(f'  Rows: {len(df):,}')

# Sanity check: proto should be raw Zeek strings ('tcp', 'udp'), not integers
sample_proto = df['proto'].dropna().iloc[0] if 'proto' in df.columns else None
if sample_proto is not None:
    try:
        int(str(sample_proto))
        print(f'  ERROR: proto value is "{sample_proto}" — looks like an integer!')
        print('  Parquet must be regenerated from labeled.csv: python3 make_dataset.py')
        sys.exit(1)
    except ValueError:
        print(f'  proto sample: "{sample_proto}" ✓  (raw Zeek string)')

# Fit LabelEncoder per categorical column — same logic as train_*.py:
#   df[col] = LabelEncoder().fit_transform(df[col].astype(str).fillna('-'))
encoders = {}
for col in CAT_FEATURES:
    if col not in df.columns:
        print(f'  WARNING: {col} not in parquet — skipping')
        continue
    vals = df[col].fillna('-').astype(str)
    le = LabelEncoder()
    le.fit(vals)
    encoders[col] = le
    print(f'  {col:<12}: {len(le.classes_)} classes → {list(le.classes_[:8])}...')

# Save to all output directories
outdirs = [Path(args.outdir), Path(args.outdir_temporal)]
for out in outdirs:
    out.mkdir(parents=True, exist_ok=True)
    for model_type in MODEL_TYPES:
        path = out / f'label_encoder_cats_{model_type}.joblib'
        joblib.dump(encoders, path)
        print(f'Saved: {path}')
