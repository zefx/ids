#!/usr/bin/env python3
"""
compare_models.py — aggregate metrics from all trained models and print comparison.

Run after all four train_*.py scripts complete:
    python compare_models.py [--models_dir /data/cicids_zeek/models]

Outputs:
    comparison_table.csv
    comparison_f1_barchart.png
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', default='/data/cicids_zeek/models/baseline')
args = parser.parse_args()

DIR = Path(args.models_dir)

MODELS = [
    ('RandomForest', 'metrics_rf.json'),
    ('XGBoost',      'metrics_xgb.json'),
    ('LightGBM',     'metrics_lgbm.json'),
    ('CatBoost',     'metrics_catboost.json'),
]

# ── Load ──────────────────────────────────────────────────────────────────────
loaded = {}
for name, fname in MODELS:
    path = DIR / fname
    if path.exists():
        with open(path) as f:
            loaded[name] = json.load(f)
        print(f'[OK] {fname}')
    else:
        print(f'[--] {fname} not found — run train_{name.lower()}.py first')

if not loaded:
    print('No metrics found.')
    sys.exit(1)

names = list(loaded.keys())

# ── Overall summary ───────────────────────────────────────────────────────────
print('\n' + '='*72)
print(f'{"Model":<14} {"Wt.F1":>7} {"MacroF1":>8} {"Time(s)":>9} {"Iters/Trees":>12}')
print('-'*72)
for n, m in loaded.items():
    iters = m.get('best_iteration', m.get('n_estimators', '—'))
    print(f'{n:<14} {m["weighted_f1"]:>7.4f} {m["macro_f1"]:>8.4f}'
          f' {m["train_time_s"]:>9.0f} {str(iters):>12}')
print('='*72)

# ── Per-class F1 ──────────────────────────────────────────────────────────────
all_classes = []
for m in loaded.values():
    for cls in m.get('per_class', {}):
        if cls not in all_classes:
            all_classes.append(cls)
all_classes = sorted(all_classes)

print(f'\nPer-class F1:')
header = f'{"Class":<30}' + ''.join(f'{n:>12}' for n in names)
print(header)
print('-' * len(header))
rows = []
for cls in all_classes:
    line = f'{cls:<30}'
    row = [cls]
    for n in names:
        val = loaded[n].get('per_class', {}).get(cls, {}).get('f1', float('nan'))
        line += f'{val:>12.4f}' if not np.isnan(val) else f'{"—":>12}'
        row.append(val)
    print(line)
    rows.append(row)

# ── Per-class Precision / Recall for hard classes ─────────────────────────────
HARD = ['Bot', 'DoS-SlowHTTPTest', 'DoS-SlowLoris']
hard_found = [c for c in HARD if c in all_classes]
if hard_found:
    print(f'\nHard classes — Precision / Recall / F1:')
    print(f'{"Class":<22} {"Model":<14} {"Prec":>7} {"Rec":>7} {"F1":>7}')
    print('-'*60)
    for cls in hard_found:
        for n in names:
            m = loaded[n].get('per_class', {}).get(cls, {})
            if m:
                print(f'{cls:<22} {n:<14}'
                      f' {m["precision"]:>7.4f} {m["recall"]:>7.4f} {m["f1"]:>7.4f}')
        print()

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = DIR / 'comparison_table.csv'
with open(csv_path, 'w') as f:
    f.write('class,' + ','.join(names) + '\n')
    for row in rows:
        vals = ','.join(f'{v:.4f}' if isinstance(v, float) and not np.isnan(v) else ''
                        for v in row[1:])
        f.write(f'{row[0]},{vals}\n')
    f.write('WEIGHTED_F1,' + ','.join(f'{loaded[n]["weighted_f1"]:.4f}' for n in names) + '\n')
    f.write('MACRO_F1,'    + ','.join(f'{loaded[n]["macro_f1"]:.4f}'    for n in names) + '\n')
    f.write('TRAIN_TIME_S,'+ ','.join(f'{loaded[n]["train_time_s"]:.0f}' for n in names) + '\n')
print(f'\nCSV → {csv_path}')

# ── Bar chart ─────────────────────────────────────────────────────────────────
x      = np.arange(len(all_classes))
width  = 0.8 / len(names)
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(14, 6))
for i, n in enumerate(names):
    f1s = [loaded[n].get('per_class', {}).get(c, {}).get('f1', 0) for c in all_classes]
    ax.bar(x + (i - len(names)/2 + 0.5) * width, f1s, width,
           label=n, color=colors[i % len(colors)], alpha=0.85)

ax.set_xlabel('Attack Class')
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 — Model Comparison\n(CSE-CIC-IDS2018, Zeek conn.log, 80/20 stratified split)')
ax.set_xticks(x)
ax.set_xticklabels(all_classes, rotation=35, ha='right', fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

chart_path = DIR / 'comparison_f1_barchart.png'
plt.savefig(chart_path, dpi=150)
plt.close()
print(f'Chart → {chart_path}')
