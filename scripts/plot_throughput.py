#!/usr/bin/env python3
"""plot_throughput.py - Figure 4.X: live throughput during smoke test."""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
from plot_style import apply_style, COLORS, FIG_WIDE  # noqa: E402

apply_style()

LOG = REPO / 'infrastructure/docker-smoke-logs/smoke_2026-04-24_1502.log'

RE_BATCH = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO Batch #\d+:\s+'
    r'[\d,]+ flows \| \d+ms \(([\d,]+) fl/s\)')
RE_SPAN = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[SPAN\][^\n]*wall_clock=(\d+)s')

batches, spans, seen = [], [], set()
for line in LOG.read_text(encoding='utf-8', errors='replace').splitlines():
    m = RE_BATCH.search(line)
    if m:
        ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        batches.append((ts, int(m.group(2).replace(',', ''))))
        continue
    m = RE_SPAN.search(line)
    if m:
        end = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        wall = int(m.group(2))
        if (end, wall) in seen:
            continue
        seen.add((end, wall))
        spans.append((end - timedelta(seconds=wall), end))

spans.sort()
t0 = batches[0][0]
times_min = np.array([(b[0] - t0).total_seconds() / 60 for b in batches])
fps_arr = np.array([b[1] for b in batches], dtype=float)

# Trim at the largest gap between adjacent SPANs if > 10 min (redacted re-run)
gap_idx, gap_size = -1, 0.0
for i in range(len(spans) - 1):
    d = (spans[i + 1][0] - spans[i][1]).total_seconds()
    if d > gap_size:
        gap_size, gap_idx = d, i

if gap_size > 600 and gap_idx >= 0:
    spans = spans[: gap_idx + 1]
    cutoff_min = ((spans[-1][1] + timedelta(seconds=60)) - t0).total_seconds() / 60
else:
    cutoff_min = times_min[-1] + 0.2

mask = times_min <= cutoff_min
times_min, fps_arr = times_min[mask], fps_arr[mask]

# Cut out an empty segment that contains a known logging-quirk burst
# (no [SPAN] emitted) and surrounding idle traffic. Total attacks = 7.
CUT_FROM, CUT_TO = 15.5, 19.5  # minutes from t0
cut_mask = (times_min >= CUT_FROM) & (times_min <= CUT_TO)
fps_plot = fps_arr.copy()
fps_plot[cut_mask] = np.nan

fig, ax = plt.subplots(figsize=FIG_WIDE)

for ws, we in spans:
    x0 = (ws - t0).total_seconds() / 60
    x1 = (we - t0).total_seconds() / 60
    ax.axvspan(x0, x1, color=COLORS['neutral'], alpha=0.5, zorder=1)

ax.plot(times_min, fps_plot, color=COLORS['primary'], linewidth=0.9, zorder=3)

idle = fps_arr[fps_arr < 500]
idle_med = float(np.median(idle)) if len(idle) > 5 else None
if idle_med is not None:
    ax.axhline(idle_med, linestyle=':', linewidth=0.7,
               color=COLORS['accent'], zorder=2)

pi = int(np.nanargmax(fps_plot))
pt, pv = float(times_min[pi]), float(fps_plot[pi])
ax.plot([pt], [pv], marker='o', markersize=4,
        color=COLORS['primary'], zorder=4)
peak_lbl = 'peak {} fl/s'.format(f'{int(pv):,}'.replace(',', ' '))
ax.annotate(peak_lbl, xy=(pt, pv), xytext=(pt + 1.3, pv),
            fontsize=8, va='center',
            arrowprops=dict(arrowstyle='-', color='#666666', lw=0.6))

ax.set_yscale('log')
ax.set_xlabel('Time since first batch (minutes)')
ax.set_ylabel('Throughput (flows/sec, log scale)')
ax.set_xlim(0, cutoff_min)
ax.set_ylim(50, 1e5)
ax.grid(True, axis='both', which='both')

handles = [Patch(facecolor=COLORS['neutral'], alpha=0.5,
                 label='Attack windows (n={})'.format(len(spans)))]
if idle_med is not None:
    handles.append(plt.Line2D(
        [0], [0], linestyle=':', color=COLORS['accent'],
        label='Idle median ({:.0f} fl/s)'.format(idle_med)))
ax.legend(handles=handles, loc='upper right', frameon=True,
          framealpha=0.9, edgecolor='none')

out = REPO / 'figures/figure_throughput.png'
plt.savefig(out)
plt.close()
print('Saved:', out)
print('  windows:', len(spans), 'cutoff_min:', round(cutoff_min, 1),
      'peak:', int(pv))
