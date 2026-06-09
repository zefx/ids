#!/usr/bin/env python3
"""
train_autoencoder.py — Autoencoder anomaly detector for hybrid IDS

Trains ONLY on BENIGN traffic from CICIDS2018 Zeek conn.log features.
The autoencoder learns the distribution of normal traffic, and any flow
with high reconstruction error is flagged as anomalous.

This complements the supervised XGBoost model:
  - XGBoost: classifies KNOWN attack types from CICIDS2018
  - Autoencoder: detects ANY anomalous traffic (zero-day capable)

Architecture:
  Input(12) → Dense(8, ReLU) → Dense(4, ReLU)
  → Dense(8, ReLU) → Output(12, Linear)

  Compact bottleneck (4 dims) forces the network to learn a compressed
  representation of normal traffic. Attack traffic that doesn't fit
  this representation produces high reconstruction error.

Threshold selection:
  - Compute reconstruction error on BENIGN validation set
  - Set threshold at 99th percentile (1% false positive rate on benign)
  - Also save 95th and 99.9th percentiles for tuning

Output:
  - autoencoder_model.pth (PyTorch state dict)
  - autoencoder_scaler.joblib (StandardScaler for feature normalization)
  - autoencoder_threshold.json (reconstruction error thresholds)

Usage:
  python3 train_autoencoder.py --data /data/cicids_zeek/ml_dataset_temporal.parquet
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/data/cicids_zeek/ml_dataset.parquet')
parser.add_argument('--out',  default='/data/cicids_zeek/models/baseline')
parser.add_argument('--epochs',     type=int,   default=50)
parser.add_argument('--batch-size', type=int,   default=4096)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--seed',       type=int,   default=42)
parser.add_argument('--fpr',        type=float, default=0.01,
                    help='Target false positive rate on BENIGN (default: 1%%)')
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'train_autoencoder.log'),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Feature definitions (must match spark_consumer.py) ───────────────────────
NUM_FEATURES      = ['duration', 'orig_bytes', 'resp_bytes',
                     'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
CAT_FEATURES      = ['proto', 'service', 'conn_state', 'history']
PORT_FEATURES     = ['id.resp_p']
TEMPORAL_FEATURES = ['conn_rate', 'unique_dst_ports', 'mean_duration',
                     'std_duration', 'bytes_per_sec']

# ── Load & preprocess ────────────────────────────────────────────────────────
log.info(f'Loading {args.data}')
df = pd.read_parquet(args.data)
log.info(f'Loaded {len(df):,} rows')

# Baseline mode: no temporal features
all_num = NUM_FEATURES

# Apply same preprocessing as training pipeline
for col in all_num:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    df[col] = np.log1p(df[col])

df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)

# Encode categoricals (same order as XGBoost training)
cat_encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].fillna('-').astype(str))
    cat_encoders[col] = le

# Feature order: [num + temporal + cat + port] (same as spark_consumer)
FEATURES = all_num + CAT_FEATURES + PORT_FEATURES
n_features = len(FEATURES)
log.info(f'Features ({n_features}): {FEATURES}')

X_all = df[FEATURES].values.astype(np.float32)
labels = df['label'].values

# ── Split: train on BENIGN only, validate on BENIGN + attacks ────────────────
benign_mask = labels == 'BENIGN'
attack_mask = ~benign_mask

log.info(f'BENIGN: {benign_mask.sum():,} | Attacks: {attack_mask.sum():,}')

X_benign = X_all[benign_mask]
X_attack = X_all[attack_mask]

# Split benign into train (80%) and val (20%)
X_train, X_val = train_test_split(X_benign, test_size=0.2,
                                   random_state=args.seed)

log.info(f'Train (benign): {len(X_train):,} | Val (benign): {len(X_val):,}')
log.info(f'Attack samples for evaluation: {len(X_attack):,}')

# ── Normalize features (important for autoencoder) ───────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_attack_scaled = scaler.transform(X_attack)

joblib.dump(scaler, OUT / 'autoencoder_scaler.joblib')
log.info('Scaler saved')

# ── Build autoencoder ────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {device}')


class Autoencoder(nn.Module):
    """Symmetric autoencoder with compact bottleneck."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            # No activation — reconstruction of scaled features
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = Autoencoder(n_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss(reduction='none')  # per-feature for analysis

log.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

# ── Train ────────────────────────────────────────────────────────────────────
train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

log.info(f'Training for {args.epochs} epochs, batch_size={args.batch_size}')
t0 = time.time()

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for (batch_x,) in train_loader:
        batch_x = batch_x.to(device)
        recon = model(batch_x)
        loss = criterion(recon, batch_x).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    if (epoch + 1) % 5 == 0 or epoch == 0:
        log.info(f'Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.6f}')

train_time = time.time() - t0
log.info(f'Training completed in {train_time:.1f}s')

# ── Save model ───────────────────────────────────────────────────────────────
model_path = OUT / 'autoencoder_model.pth'
torch.save(model.state_dict(), model_path)
log.info(f'Model saved to {model_path}')

# Also save as ONNX for potential non-PyTorch inference
try:
    dummy = torch.randn(1, n_features, device=device)
    onnx_path = OUT / 'autoencoder_model.onnx'
    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=['features'], output_names=['reconstruction'],
                      dynamic_axes={'features': {0: 'batch'}, 'reconstruction': {0: 'batch'}})
    log.info(f'ONNX model saved to {onnx_path}')
except Exception as e:
    log.warning(f'ONNX export failed (non-critical): {e}')

# ── Compute thresholds ───────────────────────────────────────────────────────
model.eval()

def compute_reconstruction_error(X_scaled: np.ndarray) -> np.ndarray:
    """Compute per-sample mean squared reconstruction error."""
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        # Process in chunks to avoid OOM
        errors = []
        chunk_size = 50_000
        for i in range(0, len(X_t), chunk_size):
            chunk = X_t[i:i+chunk_size]
            recon = model(chunk)
            err = (chunk - recon).pow(2).mean(dim=1).cpu().numpy()
            errors.append(err)
    return np.concatenate(errors)


log.info('Computing reconstruction errors...')
errors_val    = compute_reconstruction_error(X_val_scaled)
errors_attack = compute_reconstruction_error(X_attack_scaled)

# Percentiles on BENIGN validation set
p95  = float(np.percentile(errors_val, 95))
p99  = float(np.percentile(errors_val, 99))
p999 = float(np.percentile(errors_val, 99.9))

# Target threshold: at target FPR
target_percentile = (1.0 - args.fpr) * 100
threshold = float(np.percentile(errors_val, target_percentile))

thresholds = {
    'threshold': threshold,
    'target_fpr': args.fpr,
    'p95': p95,
    'p99': p99,
    'p999': p999,
    'val_mean': float(errors_val.mean()),
    'val_std': float(errors_val.std()),
    'attack_mean': float(errors_attack.mean()),
    'attack_std': float(errors_attack.std()),
    'n_features': n_features,
    'features': FEATURES,
}

with open(OUT / 'autoencoder_threshold.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

log.info(f'Thresholds: p95={p95:.6f}  p99={p99:.6f}  p999={p999:.6f}')
log.info(f'Selected threshold (FPR={args.fpr}): {threshold:.6f}')

# ── Evaluate detection performance ───────────────────────────────────────────
# Binary: anomaly if error > threshold
val_anomaly   = (errors_val > threshold).sum()
atk_anomaly   = (errors_attack > threshold).sum()
val_fpr       = val_anomaly / len(errors_val) * 100
atk_tpr       = atk_anomaly / len(errors_attack) * 100

log.info(f'── Detection Performance (threshold={threshold:.6f}) ──')
log.info(f'  BENIGN val: {val_anomaly:,}/{len(errors_val):,} flagged ({val_fpr:.2f}% FPR)')
log.info(f'  Attacks:    {atk_anomaly:,}/{len(errors_attack):,} detected ({atk_tpr:.2f}% TPR)')

# Per-attack-class breakdown
attack_labels = labels[attack_mask]
unique_attacks = sorted(set(attack_labels))
log.info(f'  Per-class detection rates:')
for atk in unique_attacks:
    atk_mask_cls = attack_labels == atk
    atk_errors = errors_attack[atk_mask_cls]
    detected = (atk_errors > threshold).sum()
    total = len(atk_errors)
    rate = detected / total * 100 if total > 0 else 0
    log.info(f'    {atk:<30} {detected:>8,}/{total:>8,} ({rate:>6.2f}%)')

# ── Summary ──────────────────────────────────────────────────────────────────
log.info(f'──────────────────────────────────────────────────')
log.info(f'Files saved to {OUT}:')
log.info(f'  autoencoder_model.pth       — PyTorch weights')
log.info(f'  autoencoder_scaler.joblib   — StandardScaler')
log.info(f'  autoencoder_threshold.json  — detection thresholds')
log.info(f'  train_autoencoder.log       — full training log')
log.info(f'Train time: {train_time:.1f}s | Features: {n_features}')
log.info(f'FPR: {val_fpr:.2f}% | TPR: {atk_tpr:.2f}%')
