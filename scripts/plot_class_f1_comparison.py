#!/usr/bin/env python3
"""
plot_class_f1_comparison.py — Figure 3.X: per-class F1 baseline vs temporal.

Horizontal grouped bar chart showing F1-score for each of 9 classes,
comparing 12-feature baseline XGBoost against 17-feature temporal XGBoost.

Output: figures/figure_class_f1_comparison.png  (300 DPI grayscale)

Numbers source: models/{baseline,temporal}/metrics_xgb.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
from plot_style import apply_style, COLORS, FIG_WIDE  # noqa: E402

apply_style()

baseline = json.loads((REPO / 'models/baseline/metrics_xgb.json').read_text())
temporal = json.loads((REPO / 'models/temporal/metrics_xgb.json').read_text())

# Sort classes by support descending (BENIGN largest, on top)
classes_sorted = sorted(
    baseline['per_class'].keys(),
    key=lambda c: -baseline['per_class'][c]['support']
)

f1_baseline = [baseline['per_class'][c]['f1'] for c in classes_sorted]
f1_temporal = [temporal['per_class'][c]['f1'] for c in classes_sorted]

fig, ax = plt.subplots(figsize=FIG_WIDE)

y = np.arange(len(classes_sorted))
height = 0.38

# Baseline — light gray, no hatch
ax.barh(y - height/2, f1_baseline, height,
        label='Baseline (12 features)',
        color=COLORS['neutral'], edgecolor='black', linewidth=0.5)

# Temporal — black with diagonal hatch (works in b&w print)
ax.barh(y + height/2, f1_temporal, height,
        label='Temporal (17 features)',
        color=COLORS['primary'], edgecolor='black', linewidth=0.5,
        hatch='///')

ax.set_yticks(y)
ax.set_yticklabels(classes_sorted)
ax.set_xlabel('F1-score')
ax.set_xlim(0, 1.15)
ax.invert_yaxis()
ax.grid(True, axis='x')
ax.grid(False, axis='y')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)

# Annotate temporal F1 values to the right
for i, val in enumerate(f1_temporal):
    ax.text(val + 0.01, i + height/2, f'{val:.3f}',
            va='center', ha='left', fontsize=8, color='#333333')

plt.savefig(REPO / 'figures' / 'figure_class_f1_comparison.png')
plt.close()

print(f'Saved: {REPO / "figures" / "figure_class_f1_comparison.png"}')
