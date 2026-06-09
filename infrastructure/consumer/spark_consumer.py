#!/usr/bin/env python3
"""
spark_consumer.py — Hybrid IDS: Spark Streaming + XGBoost + Rules

Two-layer detection cascade on Zeek conn.log flows from Kafka:
  Layer 1: XGBoost classifier (9 attack classes)
           - Baseline mode: 12 per-flow features
           - Temporal mode: 17 features (12 + 5 sliding-window aggregates)
  Layer 2: Rule engine (SYN flood, SSH/FTP brute force, port scan, Slowloris)

Temporal features (--enable-temporal): conn_rate, unique_dst_ports,
mean_duration, std_duration, bytes_per_sec — computed per src_ip in a
60s sliding window with O(1) running statistics. Improves macro F1 from
0.86 to 0.96 (+10pp), particularly for DoS-Hulk, SlowHTTPTest, FTP-BF.

Usage:
  spark-submit --master local[*] \
    spark_consumer.py --broker 127.0.0.1:9092 \
    --enable-rules --enable-temporal
"""

import argparse, json, logging, os, sys, time
from collections import defaultdict, deque, Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Hybrid IDS consumer')
parser.add_argument('--broker',      default=os.getenv('KAFKA_BROKER', '127.0.0.1:9092'))
parser.add_argument('--topic',       default='zeek-conn')
parser.add_argument('--model-dir',   default=os.getenv('MODEL_DIR', '/models/baseline'))
parser.add_argument('--max-offsets',  type=int, default=10000)
parser.add_argument('--trigger',      default='1 second')
parser.add_argument('--stats-interval', type=int, default=10)
parser.add_argument('--enable-rules',       action='store_true')
parser.add_argument('--rule-window',  type=float, default=60.0)
parser.add_argument('--enable-temporal',    action='store_true',
                    help='Use temporal model (17 features) instead of baseline (12)')
parser.add_argument('--temporal-window', type=float, default=60.0,
                    help='Sliding window for temporal feature aggregation (seconds)')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger('ids')
logging.getLogger('py4j').setLevel(logging.WARNING)

RED    = '\033[91m'
GREEN  = '\033[92m'
ORANGE = '\033[38;5;208m'
CYAN   = '\033[96m'
RESET  = '\033[0m'

MODEL_DIR = Path(args.model_dir)

# ── Feature definitions ──────────────────────────────────────────────────────
NUM_FEATURES = ['duration', 'orig_bytes', 'resp_bytes',
                'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
TEMPORAL_FEATURES = ['conn_rate', 'unique_dst_ports', 'mean_duration',
                     'std_duration', 'bytes_per_sec']
CAT_FEATURES = ['proto', 'service', 'conn_state', 'history']

# ── Temporal sliding window (O(1) per flow) ─────────────────────────────────
class IPWindow:
    """Per-IP sliding window with running statistics for temporal features."""
    __slots__ = ('entries', 'count', 'sum_dur', 'sum_dur_sq',
                 'sum_bytes', 'port_counter', 'last_wall')

    def __init__(self):
        self.entries      = deque()   # (ts, duration, dst_port, bytes)
        self.count        = 0
        self.sum_dur      = 0.0
        self.sum_dur_sq   = 0.0
        self.sum_bytes    = 0.0
        self.port_counter = Counter()
        self.last_wall    = 0.0       # last wall-clock time this IP was seen

    def clear(self):
        """Reset window completely (stale wall-clock guard)."""
        self.entries.clear()
        self.count = 0
        self.sum_dur = 0.0
        self.sum_dur_sq = 0.0
        self.sum_bytes = 0.0
        self.port_counter.clear()

    def evict(self, cutoff: float):
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
        self.entries.append((ts, dur, port, byt))
        self.count      += 1
        self.sum_dur    += dur
        self.sum_dur_sq += dur * dur
        self.sum_bytes  += byt
        self.port_counter[port] += 1

    def stats(self, window_sec: float):
        """Return (conn_rate, unique_ports, mean_dur, std_dur, bytes_per_sec)."""
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

# ── Load XGBoost model ──────────────────────────────────────────────────────
import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODEL_DIR / 'xgb_baseline.json'))
label_enc = joblib.load(MODEL_DIR / 'label_encoder_xgb.joblib')
n_model_features = xgb_model.n_features_in_
log.info(f'XGBoost loaded | classes={list(label_enc.classes_)} | features={n_model_features}')

# Sanity check: model features must match mode
expected = 17 if args.enable_temporal else 12
if n_model_features != expected:
    log.error(f'Feature mismatch: model expects {n_model_features}, '
              f'but --enable-temporal={"on" if args.enable_temporal else "off"} → {expected}. '
              f'Check --model-dir and --enable-temporal flags.')
    sys.exit(1)

# ── Categorical encoders (precompute mappings for speed) ────────────────────
cat_encoders = joblib.load(MODEL_DIR / 'label_encoder_cats_xgb.joblib')
cat_mappings = {}
for col, enc in cat_encoders.items():
    cat_mappings[col] = {v: int(enc.transform([v])[0]) for v in enc.classes_}

# ── Temporal feature windows (optional) ─────────────────────────────────────
temporal_windows: dict = defaultdict(IPWindow) if args.enable_temporal else None
TEMPORAL_WINDOW_SEC = args.temporal_window
if args.enable_temporal:
    log.info(f'Temporal features enabled (window={TEMPORAL_WINDOW_SEC}s, 17 features)')

# ── Rule engine (optional) ──────────────────────────────────────────────────
rule_engine = None
if args.enable_rules:
    from rule_engine import RuleEngine, RuleConfig
    rule_engine = RuleEngine(RuleConfig(window_sec=args.rule_window))
    log.info('Rule engine enabled')

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_float(s):
    """Convert Zeek field to float. Handles '-', None, '(empty)' → 0."""
    if s is None or s == '' or s == '-' or s == '(empty)':
        return 0.0
    try:
        return max(0.0, float(s))
    except (ValueError, TypeError):
        return 0.0

def safe_int(s):
    if s is None or s == '' or s == '-':
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0

def ts_str(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

# ── Feature extraction (vectorised) ─────────────────────────────────────────
def extract_features(df, temporal_arr=None):
    """Convert raw Zeek DataFrame to feature matrix.

    Args:
        df:           Raw Zeek DataFrame
        temporal_arr: Optional (N, 5) array of temporal features (raw, pre-log1p)
    Returns:
        Feature matrix: (N, 12) baseline or (N, 17) with temporal
    """
    n = len(df)
    arrays = []

    # 7 numeric features: log1p transform (vectorised, no Python-level loops)
    for col in NUM_FEATURES:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(lower=0.0).values
        else:
            vals = np.zeros(n, dtype=np.float64)
        arrays.append(np.log1p(vals).astype(np.float32))

    # 5 temporal features: log1p transform (same as training)
    if temporal_arr is not None:
        for j in range(5):
            arrays.append(np.log1p(temporal_arr[:, j]).astype(np.float32))

    # 4 categorical features: label encoding (dict map — C-level in pandas)
    for col in CAT_FEATURES:
        series = df[col].fillna('-').astype(str) if col in df.columns else pd.Series(['-'] * n)
        mapping = cat_mappings.get(col, {})
        arrays.append(series.map(mapping).fillna(0).values.astype(np.float32))

    # 1 port feature
    if 'id.resp_p' in df.columns:
        arrays.append(pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).values.astype(np.float32))
    else:
        arrays.append(np.zeros(n, dtype=np.float32))

    return np.column_stack(arrays)  # (N, 12) or (N, 17)

# ── Alert tracking ──────────────────────────────────────────────────────────
active_attacks = {}   # (src_ip, label) → {first_seen, last_seen, count, ...}
src_ip_spans   = {}   # src_ip → {first_seen, last_seen, labels, total_flows}
ATTACK_TIMEOUT = 30.0

def _ts(t):
    """Wall-clock timestamp for alert output (local time, matches Spark logs)."""
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

def expire_attacks(now):
    """End attacks that have not been seen for ATTACK_TIMEOUT seconds."""
    expired = [k for k, v in active_attacks.items() if now - v['last_seen'] > ATTACK_TIMEOUT]
    for k in expired:
        a = active_attacks.pop(k)
        dur = a['last_seen'] - a['first_seen']
        src = a.get('source', 'ML')
        avg_conf = a['sum_conf'] / a['count'] if a['count'] > 0 else 0
        print(f'{ORANGE}{_ts(now)} [END] [{src}] {k[1]:<22} src={k[0]:<16} '
              f'duration={dur:.0f}s flows={a["count"]:,} '
              f'conf=avg:{avg_conf:.2f}/max:{a["max_conf"]:.2f}{RESET}', flush=True)

    # Expire src_ip spans: if no active attacks remain for this IP, print summary
    expired_ips = {k[0] for k in expired}
    for eip in expired_ips:
        if not any(ak[0] == eip for ak in active_attacks):
            span = src_ip_spans.pop(eip, None)
            if span and span['total_flows'] > 0:
                wall = now - span['first_seen']
                labels = ', '.join(sorted(span['labels']))
                print(f'{CYAN}{_ts(now)} [SPAN] src={eip:<16} '
                      f'wall_clock={wall:.0f}s total_flows={span["total_flows"]:,} '
                      f'labels=[{labels}]{RESET}', flush=True)

def log_attack(label, src_ip, dst_ip, dst_port, conf, now, source='ML'):
    """Track and log detected attacks with deduplication.

    Uses wall-clock `now` (time.time()) for tracking, not Zeek flow ts,
    so that expire_attacks() timeout works correctly.
    """
    if label == 'BENIGN':
        return

    key = (src_ip, label)

    # Per-IP span tracking (across all labels)
    if src_ip not in src_ip_spans:
        src_ip_spans[src_ip] = {
            'first_seen': now, 'last_seen': now,
            'labels': {label}, 'total_flows': 1,
        }
    else:
        sp = src_ip_spans[src_ip]
        sp['last_seen'] = now
        sp['labels'].add(label)
        sp['total_flows'] += 1

    if key not in active_attacks:
        active_attacks[key] = {
            'first_seen': now, 'last_seen': now, 'count': 1,
            'dst_ips': {dst_ip}, 'source': source,
            'max_conf': conf, 'sum_conf': conf,
        }
        print(f'{RED}{_ts(now)} [NEW] [{source}] {label:<22} src={src_ip:<16} → {dst_ip}:{dst_port} '
              f'conf={conf:.2f}{RESET}', flush=True)
    else:
        a = active_attacks[key]
        a['last_seen'] = now
        a['count'] += 1
        a['dst_ips'].add(dst_ip)
        a['sum_conf'] += conf
        if conf > a['max_conf']:
            a['max_conf'] = conf

# ── Stats ────────────────────────────────────────────────────────────────────
stats_start    = time.time()
total_flows    = 0
total_batches  = 0
attack_counts  = defaultdict(int)
last_stats     = time.time()

def print_stats():
    elapsed = time.time() - stats_start
    fps = total_flows / elapsed if elapsed > 0 else 0
    log.info(f'── STATS ──────────────────────────────────────────')
    layers = 'XGBoost'
    if temporal_windows: layers += ' (temporal)'
    if rule_engine: layers += ' + Rules'
    log.info(f'  Layers:     {layers}')
    log.info(f'  Throughput: {fps:,.0f} flows/sec')
    log.info(f'  Total:      {total_flows:,} flows in {elapsed:.0f}s')
    for lbl, cnt in sorted(attack_counts.items(), key=lambda x: -x[1]):
        if lbl == 'BENIGN':
            print(f'{GREEN}    {lbl:<28} {cnt:>8,}{RESET}', flush=True)
        else:
            print(f'{RED}  * {lbl:<28} {cnt:>8,}{RESET}', flush=True)
    if active_attacks:
        log.info(f'  Active attacks: {len(active_attacks)}')
    log.info(f'───────────────────────────────────────────────────')

# ── Batch processing (called by Spark for each micro-batch) ──────────────────
def process_batch(batch_df, batch_id):
    global total_flows, total_batches, last_stats

    pdf = batch_df.toPandas()
    if pdf.empty:
        expire_attacks(time.time())
        if time.time() - last_stats >= args.stats_interval and total_flows > 0:
            print_stats()
            last_stats = time.time()
        return

    # Parse Kafka JSON messages
    raw = pdf['value'].values
    rows = []
    for v in raw:
        try:
            rows.append(json.loads(v))
        except Exception:
            pass
    if not rows:
        return

    df = pd.DataFrame(rows)
    n = len(df)
    now = time.time()

    # Pre-extract raw columns (used by temporal, rules, and alerts)
    sip = df['id.orig_h'].values if 'id.orig_h' in df.columns else np.full(n, '?')
    dip = df['id.resp_h'].values if 'id.resp_h' in df.columns else np.full(n, '?')
    dp  = df['id.resp_p'].values if 'id.resp_p' in df.columns else np.full(n, 0)
    ts_arr = df['ts'].values     if 'ts'        in df.columns else np.full(n, None)
    cs  = df['conn_state'].values if 'conn_state' in df.columns else np.full(n, '-')
    dur = df['duration'].values   if 'duration'   in df.columns else np.full(n, 0)
    ob  = df['orig_bytes'].values if 'orig_bytes' in df.columns else np.full(n, 0)
    rb  = df['resp_bytes'].values if 'resp_bytes' in df.columns else np.full(n, 0)

    # ── Temporal feature aggregation (before XGBoost) ──────────────────
    temporal_arr = None
    if temporal_windows is not None:
        t0 = time.perf_counter()
        temporal_arr = np.empty((n, 5), dtype=np.float32)
        for i in range(n):
            src_i = str(sip[i] or '?')
            ts_i  = safe_float(ts_arr[i]) or now
            win   = temporal_windows[src_i]
            # Wall-clock guard: if IP hasn't been seen for > window, clear stale data
            if win.last_wall > 0 and now - win.last_wall > TEMPORAL_WINDOW_SEC:
                win.clear()
            win.last_wall = now
            win.evict(ts_i - TEMPORAL_WINDOW_SEC)
            # Stats BEFORE adding current flow (matches training pipeline)
            temporal_arr[i] = win.stats(TEMPORAL_WINDOW_SEC)
            win.add(ts_i, safe_float(dur[i]), safe_int(dp[i]),
                    safe_float(ob[i]) + safe_float(rb[i]))
        temp_ms = (time.perf_counter() - t0) * 1000
    else:
        temp_ms = 0.0

    # ── Layer 1: XGBoost prediction ──────────────────────────────────────
    t0 = time.perf_counter()
    features = extract_features(df, temporal_arr)
    feat_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    proba = xgb_model.predict_proba(features)
    pred_idx = np.argmax(proba, axis=1)
    labels = label_enc.classes_[pred_idx].tolist()
    confs  = proba[np.arange(n), pred_idx].tolist()
    pred_ms = (time.perf_counter() - t0) * 1000

    # Source tracking: ML by default, Rule when overridden
    sources = ['ML'] * n

    # ── Layer 2: Rule engine (override BENIGN → specific attack) ─────────
    t0 = time.perf_counter()
    rule_hits = 0
    if rule_engine:
        for i in range(n):
            if labels[i] == 'BENIGN':
                result = rule_engine.evaluate(
                    str(sip[i] or '?'), safe_int(dp[i]), str(cs[i] or '-'),
                    safe_float(dur[i]), safe_float(ob[i]), safe_float(ts_arr[i]) or now)
                if result:
                    labels[i] = result
                    confs[i] = 0.95
                    sources[i] = 'Rule'
                    rule_hits += 1
    rule_ms = (time.perf_counter() - t0) * 1000

    # ── Log alerts ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for i in range(n):
        attack_counts[labels[i]] += 1
        log_attack(labels[i], str(sip[i] or '?'), str(dip[i] or '?'),
                   safe_int(dp[i]), confs[i], now, sources[i])
    expire_attacks(now)
    alert_ms = (time.perf_counter() - t0) * 1000

    total_flows  += n
    total_batches += 1
    total_ms = temp_ms + feat_ms + pred_ms + rule_ms + alert_ms

    extra = ''
    if temp_ms > 0: extra += f' temp={temp_ms:.0f}ms'
    if rule_hits:   extra += f' rules={rule_hits}'
    log.info(f'Batch #{batch_id}: {n:,} flows | {total_ms:.0f}ms '
             f'({n / (total_ms / 1000):,.0f} fl/s){extra}')

    # Periodic cleanup
    if rule_engine and total_batches % 60 == 0:
        rule_engine.cleanup(now)
    if temporal_windows and total_batches % 60 == 0:
        stale = [ip for ip, w in temporal_windows.items()
                 if now - w.last_wall > TEMPORAL_WINDOW_SEC]
        for ip in stale:
            del temporal_windows[ip]

    if time.time() - last_stats >= args.stats_interval:
        print_stats()
        last_stats = time.time()


# ── Main: Spark session + streaming query ────────────────────────────────────
def main():
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .appName('zeek-ids')
             .master('local[*]')
             .config('spark.jars.packages',
                     'org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1')
             .config('spark.sql.shuffle.partitions', '4')
             .config('spark.driver.memory', '2g')
             .config('spark.ui.showConsoleProgress', 'false')
             .config('spark.sql.streaming.forceDeleteTempCheckpointLocation', 'true')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')

    log.info(f'Spark {spark.version} started')

    stream = (spark.readStream
              .format('kafka')
              .option('kafka.bootstrap.servers', args.broker)
              .option('subscribe', args.topic)
              .option('startingOffsets', 'latest')
              .option('maxOffsetsPerTrigger', str(args.max_offsets))
              .option('failOnDataLoss', 'false')
              .load()
              .selectExpr('CAST(value AS STRING) as value'))

    query = (stream.writeStream
             .foreachBatch(process_batch)
             .trigger(processingTime=args.trigger)
             .option('checkpointLocation', os.getenv('CHECKPOINT_DIR', '/tmp/spark-ids-checkpoint'))
             .start())

    log.info('Listening for Zeek flows...')
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
    finally:
        print_stats()
        spark.stop()


if __name__ == '__main__':
    main()
