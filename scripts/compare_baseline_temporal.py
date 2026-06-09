#!/usr/bin/env python3
"""
compare_baseline_temporal.py — сравнение baseline (12 features) vs temporal (17 features)

Usage:
    python3 compare_baseline_temporal.py
    python3 compare_baseline_temporal.py \
        --baseline /data/cicids_zeek/models \
        --temporal /data/cicids_zeek/models_temporal \
        --out      /data/cicids_zeek/models_temporal

Outputs:
    comparison_baseline_vs_temporal.csv
    comparison_baseline_vs_temporal.png   (per-class F1, grouped by model)
    comparison_delta.png                  (delta F1: temporal - baseline per class)
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='/data/cicids_zeek/models/baseline')
parser.add_argument('--temporal', default='/data/cicids_zeek/models/temporal')
parser.add_argument('--out',      default='/data/cicids_zeek/models/temporal')
args = parser.parse_args()

BASE_DIR = Path(args.baseline)
TEMP_DIR = Path(args.temporal)
OUT_DIR  = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ('RF',   'metrics_rf.json'),
    ('XGB',  'metrics_xgb.json'),
    ('LGBM', 'metrics_lgbm.json'),
]

# ── Load metrics ──────────────────────────────────────────────────────────────
def load_dir(directory, label):
    result = {}
    for name, fname in MODELS:
        path = directory / fname
        if path.exists():
            with open(path) as f:
                result[name] = json.load(f)
            print(f'[OK] {label}/{fname}')
        else:
            print(f'[--] {label}/{fname} not found')
    return result

baseline = load_dir(BASE_DIR, 'baseline')
temporal = load_dir(TEMP_DIR, 'temporal')

if not baseline or not temporal:
    print('\nRun training scripts first:')
    print('  Baseline: python3 train_rf.py && python3 train_xgb.py && python3 train_lgbm.py')
    print('  Temporal: python3 train_rf.py --data ml_dataset_temporal.parquet --out models_temporal')
    sys.exit(1)

# ── Overall summary ───────────────────────────────────────────────────────────
print('\n' + '='*80)
print(f'{"Model":<8} {"Variant":<10} {"Wt.F1":>7} {"MacroF1":>8} {"ΔWt.F1":>8} {"ΔMacroF1":>10} {"Time(s)":>8}')
print('-'*80)

for name, _ in MODELS:
    if name not in baseline or name not in temporal:
        continue
    b = baseline[name]
    t = temporal[name]
    d_wf1 = t['weighted_f1'] - b['weighted_f1']
    d_mf1 = t['macro_f1']    - b['macro_f1']
    print(f'{name:<8} {"baseline":<10} {b["weighted_f1"]:>7.4f} {b["macro_f1"]:>8.4f}'
          f' {"":>8} {"":>10} {b["train_time_s"]:>8.0f}')
    sign_wf1 = '+' if d_wf1 >= 0 else ''
    sign_mf1 = '+' if d_mf1 >= 0 else ''
    print(f'{name:<8} {"temporal":<10} {t["weighted_f1"]:>7.4f} {t["macro_f1"]:>8.4f}'
          f' {sign_wf1}{d_wf1:>7.4f} {sign_mf1}{d_mf1:>9.4f} {t["train_time_s"]:>8.0f}')
    print('-'*80)

print('='*80)

# ── Per-class F1 comparison ───────────────────────────────────────────────────
all_classes = sorted({
    cls
    for m in list(baseline.values()) + list(temporal.values())
    for cls in m.get('per_class', {})
})

print(f'\nPer-class F1 — baseline vs temporal (Δ = temporal - baseline):')
header = f'{"Class":<25}' + ''.join(
    f'{"B-"+n:>8} {"T-"+n:>8} {"Δ"+n:>7}' for n in [m for m, _ in MODELS if m in baseline and m in temporal]
)
print(header)
print('-' * len(header))

csv_rows = []
for cls in all_classes:
    line = f'{cls:<25}'
    csv_row = [cls]
    for name, _ in MODELS:
        if name not in baseline or name not in temporal:
            continue
        b_f1 = baseline[name].get('per_class', {}).get(cls, {}).get('f1', float('nan'))
        t_f1 = temporal[name].get('per_class', {}).get(cls, {}).get('f1', float('nan'))
        delta = t_f1 - b_f1 if not (np.isnan(b_f1) or np.isnan(t_f1)) else float('nan')
        b_str = f'{b_f1:>8.4f}' if not np.isnan(b_f1) else f'{"—":>8}'
        t_str = f'{t_f1:>8.4f}' if not np.isnan(t_f1) else f'{"—":>8}'
        d_str = f'{delta:>+7.4f}' if not np.isnan(delta) else f'{"—":>7}'
        line += b_str + t_str + d_str
        csv_row += [b_f1, t_f1, delta]
    print(line)
    csv_rows.append(csv_row)

# ── Save CSV ──────────────────────────────────────────────────────────────────
active_models = [n for n, _ in MODELS if n in baseline and n in temporal]
csv_path = OUT_DIR / 'comparison_baseline_vs_temporal.csv'
with open(csv_path, 'w') as f:
    header_cols = ['class'] + [f'{pfx}_{n}' for n in active_models
                                for pfx in ('baseline', 'temporal', 'delta')]
    f.write(','.join(header_cols) + '\n')
    for row in csv_rows:
        vals = [row[0]] + [
            f'{v:.4f}' if isinstance(v, float) and not np.isnan(v) else ''
            for v in row[1:]
        ]
        f.write(','.join(vals) + '\n')
    # overall metrics
    for metric_key, label in [('weighted_f1', 'WEIGHTED_F1'), ('macro_f1', 'MACRO_F1')]:
        parts = [label]
        for n in active_models:
            b_v = baseline[n][metric_key]
            t_v = temporal[n][metric_key]
            parts += [f'{b_v:.4f}', f'{t_v:.4f}', f'{t_v-b_v:+.4f}']
        f.write(','.join(parts) + '\n')
print(f'\nCSV → {csv_path}')

# ── Chart 1: per-class F1 baseline vs temporal grouped by model ───────────────
x      = np.arange(len(all_classes))
n_m    = len(active_models)
width  = 0.35
colors_base = ['#4C72B0', '#DD8452', '#55A868']
colors_temp = ['#1a3d6e', '#8b4a1a', '#1f5c30']

fig, axes = plt.subplots(1, n_m, figsize=(7 * n_m, 6), sharey=True)
if n_m == 1:
    axes = [axes]

for ax, (name, color_b, color_t) in zip(axes, zip(active_models, colors_base, colors_temp)):
    b_f1s = [baseline[name].get('per_class', {}).get(c, {}).get('f1', 0) for c in all_classes]
    t_f1s = [temporal[name].get('per_class', {}).get(c, {}).get('f1', 0) for c in all_classes]
    ax.bar(x - width/2, b_f1s, width, label='Baseline (12 feat)', color=color_b, alpha=0.8)
    ax.bar(x + width/2, t_f1s, width, label='Temporal (17 feat)', color=color_t, alpha=0.8)
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('F1 Score')

fig.suptitle('Per-Class F1: Baseline vs Temporal Features\n(CSE-CIC-IDS2018, Zeek conn.log)',
             fontsize=13)
plt.tight_layout()
chart1 = OUT_DIR / 'comparison_baseline_vs_temporal.png'
plt.savefig(chart1, dpi=150)
plt.close()
print(f'Chart → {chart1}')

# ── Chart 2: delta F1 (temporal - baseline) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x      = np.arange(len(all_classes))
width  = 0.25

for i, (name, color) in enumerate(zip(active_models, colors_base)):
    deltas = []
    for cls in all_classes:
        b = baseline[name].get('per_class', {}).get(cls, {}).get('f1', float('nan'))
        t = temporal[name].get('per_class', {}).get(cls, {}).get('f1', float('nan'))
        deltas.append(t - b if not (np.isnan(b) or np.isnan(t)) else 0)
    offset = (i - n_m / 2 + 0.5) * width
    bars = ax.bar(x + offset, deltas, width, label=name, color=color, alpha=0.85)
    # label bars > 0.01
    for bar, d in zip(bars, deltas):
        if abs(d) > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{d:+.2f}', ha='center', va='bottom', fontsize=7)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Attack Class')
ax.set_ylabel('ΔF1 (temporal − baseline)')
ax.set_title('F1 Improvement: Temporal Features vs Baseline\n(CSE-CIC-IDS2018, Zeek conn.log)')
ax.set_xticks(x)
ax.set_xticklabels(all_classes, rotation=35, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
chart2 = OUT_DIR / 'comparison_delta.png'
plt.savefig(chart2, dpi=150)
plt.close()
print(f'Delta chart → {chart2}')
print('\n=== done ===')
