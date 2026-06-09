#!/usr/bin/env python3
"""
build_temporal_dataset.py — temporal aggregation features (O(1) running statistics)

Для каждого flow вычисляет агрегаты по src_ip за скользящее окно WINDOW_SEC секунд:
  - conn_rate         : соединений в секунду
  - unique_dst_ports  : уникальных dst портов (Counter-based, O(1))
  - mean_duration     : средняя длительность flow
  - std_duration      : std длительности (Bot beaconing сигнал)
  - bytes_per_sec     : (orig_bytes + resp_bytes) / WINDOW_SEC

Алгоритм: running statistics — O(1) на каждую строку независимо от размера окна.
Предыдущая версия была O(n) и зависала на DoS трафике (60k записей в окне).

Читает:    /data/cicids_zeek/<day>/labeled.csv  (дни в хронологическом порядке)
Исключает: /data/cicids_zeek/combined/          (содержит те же данные, дубли)
Сохраняет: /data/cicids_zeek/ml_dataset_temporal.parquet  (12 + 5 признаков)

Оригинальный ml_dataset.parquet НЕ изменяется.
"""

import argparse
import logging
import sys
import time
from collections import defaultdict, deque, Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/data/cicids_zeek')
parser.add_argument('--out',      default='/data/cicids_zeek/ml_dataset_temporal.parquet')
parser.add_argument('--window',   type=float, default=60.0)
parser.add_argument('--log',      default='/data/cicids_zeek/build_temporal.log')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(args.log),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

WINDOW_SEC = args.window
DATA_DIR   = Path(args.data_dir)

# ── Feature columns ───────────────────────────────────────────────────────────
NUM_FEATURES  = ['duration', 'orig_bytes', 'resp_bytes',
                 'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
CAT_FEATURES  = ['proto', 'service', 'conn_state', 'history']
PORT_FEATURES = ['id.resp_p']
BASE_FEATURES = NUM_FEATURES + CAT_FEATURES + PORT_FEATURES
TEMPORAL_FEATURES = ['conn_rate', 'unique_dst_ports', 'mean_duration',
                     'std_duration', 'bytes_per_sec']

NEEDED_COLS = BASE_FEATURES + ['ts', 'id.orig_h', 'label']

# ── Running window state per src_ip (O(1) updates) ───────────────────────────
class IPWindow:
    """Sliding window with O(1) insert/evict via running statistics."""
    __slots__ = ('entries', 'count', 'sum_dur', 'sum_dur_sq',
                 'sum_bytes', 'port_counter')

    def __init__(self):
        self.entries     = deque()   # (ts, duration, dst_port, bytes)
        self.count       = 0
        self.sum_dur     = 0.0
        self.sum_dur_sq  = 0.0
        self.sum_bytes   = 0.0
        self.port_counter = Counter()

    def evict(self, cutoff: float):
        """Remove entries older than cutoff. Amortized O(1)."""
        while self.entries and self.entries[0][0] < cutoff:
            _, dur, port, byt = self.entries.popleft()
            self.count       -= 1
            self.sum_dur     -= dur
            self.sum_dur_sq  -= dur * dur
            self.sum_bytes   -= byt
            self.port_counter[port] -= 1
            if self.port_counter[port] == 0:
                del self.port_counter[port]

    def add(self, ts: float, dur: float, port: int, byt: float):
        """Add new entry. O(1)."""
        self.entries.append((ts, dur, port, byt))
        self.count      += 1
        self.sum_dur    += dur
        self.sum_dur_sq += dur * dur
        self.sum_bytes  += byt
        self.port_counter[port] += 1

    def stats(self, window_sec: float):
        """Return (conn_rate, unique_ports, mean_dur, std_dur, bytes_per_sec). O(1)."""
        n = self.count
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        mean_dur = self.sum_dur / n
        var_dur  = max(0.0, self.sum_dur_sq / n - mean_dur * mean_dur)
        return (
            n / window_sec,
            float(len(self.port_counter)),
            mean_dur,
            var_dur ** 0.5,
            self.sum_bytes / window_sec,
        )

# ── Find day directories (exclude combined/) ──────────────────────────────────
day_dirs = sorted([d for d in DATA_DIR.iterdir()
                   if d.is_dir()
                   and d.name != 'combined'
                   and d.name != 'models'
                   and (d / 'labeled.csv').exists()])

if not day_dirs:
    log.error(f'No labeled.csv files found in {DATA_DIR}')
    sys.exit(1)

log.info(f'Found {len(day_dirs)} day directories:')
for d in day_dirs:
    log.info(f'  {d.name}')

# ── Load all labeled.csv ──────────────────────────────────────────────────────
log.info('Loading labeled.csv files...')
dfs = []
for day_dir in day_dirs:
    path = day_dir / 'labeled.csv'
    try:
        df_day = pd.read_csv(path, low_memory=False,
                             usecols=lambda c: c in NEEDED_COLS)
        log.info(f'  {day_dir.name}: {len(df_day):,} rows')
        dfs.append(df_day)
    except Exception as e:
        log.error(f'  Failed {path}: {e}')

if not dfs:
    log.error('No data loaded.')
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
log.info(f'Total rows: {len(df):,}')

# ── Validate required columns ─────────────────────────────────────────────────
missing = {'ts', 'id.orig_h', 'label'} - set(df.columns)
if missing:
    log.error(f'Required columns missing: {missing}')
    log.error(f'Available columns: {list(df.columns)}')
    sys.exit(1)

# ── Sort by timestamp ─────────────────────────────────────────────────────────
log.info('Sorting by ts...')
df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
df = df.dropna(subset=['ts']).sort_values('ts').reset_index(drop=True)
log.info(f'After sort: {len(df):,} rows')
log.info(f'Time range: {df["ts"].min():.0f} → {df["ts"].max():.0f}')

# ── Preprocess base features ──────────────────────────────────────────────────
log.info('Preprocessing base features...')
for col in NUM_FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)

if 'id.resp_p' in df.columns:
    df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)

for col in CAT_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna('-').astype(str)

# ── Extract arrays for fast loop ──────────────────────────────────────────────
n_rows    = len(df)
ts_arr    = df['ts'].to_numpy(dtype=np.float64)
src_arr   = df['id.orig_h'].to_numpy()
port_arr  = df['id.resp_p'].to_numpy(dtype=np.int32) if 'id.resp_p' in df.columns else np.zeros(n_rows, dtype=np.int32)
dur_arr   = df['duration'].to_numpy(dtype=np.float64) if 'duration' in df.columns else np.zeros(n_rows)
ob_arr    = df['orig_bytes'].to_numpy(dtype=np.float64) if 'orig_bytes' in df.columns else np.zeros(n_rows)
rb_arr    = df['resp_bytes'].to_numpy(dtype=np.float64) if 'resp_bytes' in df.columns else np.zeros(n_rows)

# Output arrays
out_conn_rate    = np.empty(n_rows, dtype=np.float32)
out_unique_ports = np.empty(n_rows, dtype=np.float32)
out_mean_dur     = np.empty(n_rows, dtype=np.float32)
out_std_dur      = np.empty(n_rows, dtype=np.float32)
out_bytes_ps     = np.empty(n_rows, dtype=np.float32)

# ── Main loop (O(1) per row) ──────────────────────────────────────────────────
log.info(f'Computing temporal features (window={WINDOW_SEC}s, O(1) per row)...')
windows: dict = defaultdict(IPWindow)

t0        = time.time()
log_every = max(1, n_rows // 20)

for i in range(n_rows):
    if i % log_every == 0 and i > 0:
        elapsed = time.time() - t0
        eta     = elapsed / i * (n_rows - i)
        log.info(f'  {i:>10,} / {n_rows:,}  ({i/n_rows*100:.0f}%)'
                 f'  elapsed={elapsed:.0f}s  eta={eta:.0f}s')

    ts_i  = ts_arr[i]
    src_i = src_arr[i]
    win   = windows[src_i]

    # evict stale entries
    win.evict(ts_i - WINDOW_SEC)

    # compute stats BEFORE adding current flow (no leakage)
    cr, up, md, sd, bp = win.stats(WINDOW_SEC)
    out_conn_rate[i]    = cr
    out_unique_ports[i] = up
    out_mean_dur[i]     = md
    out_std_dur[i]      = sd
    out_bytes_ps[i]     = bp

    # add current flow
    win.add(ts_i, float(dur_arr[i]), int(port_arr[i]),
            float(ob_arr[i]) + float(rb_arr[i]))

elapsed_total = time.time() - t0
log.info(f'Temporal features computed in {elapsed_total:.1f}s  '
         f'({n_rows/elapsed_total:.0f} rows/s)')

# ── Attach temporal features (raw values — log1p applied by training scripts) ─
df['conn_rate']        = out_conn_rate
df['unique_dst_ports'] = out_unique_ports
df['mean_duration']    = out_mean_dur
df['std_duration']     = out_std_dur
df['bytes_per_sec']    = out_bytes_ps

# NOTE: NO log1p here — base numeric features also saved raw.
# Training scripts (train_rf.py etc.) apply log1p to all numeric features.

# ── Save ──────────────────────────────────────────────────────────────────────
ALL_FEATURES = BASE_FEATURES + TEMPORAL_FEATURES
out_cols = [c for c in ALL_FEATURES + ['label'] if c in df.columns]
df_out   = df[out_cols].reset_index(drop=True)

log.info(f'Final: {len(df_out):,} rows × {len(out_cols)} columns')
log.info(f'Columns: {list(df_out.columns)}')

for cls, cnt in df_out['label'].value_counts().items():
    log.info(f'  {cls:<30} {cnt:>10,} ({cnt/len(df_out)*100:.2f}%)')

log.info(f'Saving → {args.out}')
df_out.to_parquet(args.out, index=False, compression='snappy')
log.info(f'File size: {Path(args.out).stat().st_size / 1e9:.2f} GB')
log.info('=== done ===')
