#!/usr/bin/env python3
"""
plot_class_distribution.py — Figure 2.X (Class distribution, log-scale).

Visualises the extreme class imbalance in CSE-CIC-IDS2018 labelled dataset
(89% BENIGN vs 0.02% Slowloris — 4 orders of magnitude span).

Output: figures/figure_class_distribution.png  (300 DPI grayscale)

Numbers source: models/baseline/metrics_xgb.json (test_support * 5).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
from plot_style import apply_style, COLORS, FIG_WIDE  # noqa: E402

apply_style()

# Class distribution (counts, %) — recovered from metrics_xgb.json test_support * 5
DATA = [
    ('BENIGN',           27_459_940, 89.13),
    ('DoS-Hulk',          1_803_685,  5.85),
    ('DDOS-HOIC',         1_074_410,  3.49),
    ('FTP-BruteForce',      190_300,  0.62),
    ('DoS-SlowHTTPTest',    105_550,  0.34),
    ('SSH-Bruteforce',       92_650,  0.30),
    ('Bot',                  48_005,  0.16),
    ('DoS-GoldenEye',        28_660,  0.09),
    ('DoS-Slowloris',         6_980,  0.02),
]

DATA.sort(key=lambda x: -x[1])

classes = [d[0] for d in DATA]
counts  = np.array([d[1] for d in DATA])
pcts    = [d[2] for d in DATA]

fig, ax = plt.subplots(figsize=FIG_WIDE)

y = np.arange(len(classes))
bars = ax.barh(y, counts, color=COLORS['attack'], edgecolor='black', linewidth=0.5)

# BENIGN — special: lighter shade (majority class, contrast with attacks)
bars[0].set_color(COLORS['benign'])
bars[0].set_edgecolor('black')

ax.set_yticks(y)
ax.set_yticklabels(classes)
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel('Number of flows (log scale)')
ax.grid(True, axis='x', which='both')
ax.grid(False, axis='y')


def fmt_count(n: int) -> str:
    """Thousand separator: regular space (Russian/European convention)."""
    return f'{n:,}'.replace(',', ' ')


for i, (c, p) in enumerate(zip(counts, pcts)):
    label = f'{fmt_count(int(c))}  ({p:.2f}%)'
    ax.text(c * 1.15, i, label, va='center', ha='left',
            fontsize=8, color='#333333')

ax.set_xlim(1e3, 5e8)

plt.savefig(REPO / 'figures' / 'figure_class_distribution.png')
plt.close()

print(f'Saved: {REPO / "figures" / "figure_class_distribution.png"}')
