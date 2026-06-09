"""
plot_style.py — единый стиль для всех графиков в ВКР.

    from plot_style import apply_style, COLORS, PALETTE
    apply_style()

Стиль ориентирован на единство с thesis_fin.docx (Times New Roman 10pt body)
и стандарты IEEE/Springer ML-публикаций. Палитра — grayscale (8 уровней серого)
для строгих ч/б научных figures, совместимых с b&w-печатью без потерь.

Параметры:
    DPI:    300 (печать)
    Шрифт:  Times New Roman → Liberation Serif → DejaVu Serif (fallback chain)
    Размер: 5×3.5 inch по умолчанию (single-column thesis figure)
    Цвета:  grayscale 8 levels (#000 → #EEE) + hatch patterns для различения
    Сетка:  --, alpha=0.5, серая
    Spines: top/right убраны
    Title:  не используется (caption идёт под рисунком в HSE формате)
"""

import matplotlib as mpl
from cycler import cycler

# ── Grayscale palette (для строгих ч/б научных публикаций) ──────────────────
# 8 уровней серого — от чёрного до светло-серого. Подбор контраста для b&w-печати.
PALETTE = [
    '#000000',  # 0  black
    '#404040',  # 1  very dark gray
    '#606060',  # 2  dark gray
    '#808080',  # 3  medium gray
    '#A0A0A0',  # 4  mid-light gray
    '#C0C0C0',  # 5  light gray
    '#D8D8D8',  # 6  very light gray
    '#EEEEEE',  # 7  near-white
]

COLORS = {
    'primary':   '#000000',  # main series
    'secondary': '#808080',  # second series
    'positive':  '#404040',  # для приростов (тёмный)
    'negative':  '#A0A0A0',  # для убыли (светлее)
    'neutral':   '#C0C0C0',  # baseline / reference
    'accent':    '#606060',  # подчёркивание
    'benign':    '#A0A0A0',  # benign — светлее (большой объём, не акцент)
    'attack':    '#404040',  # attack — темнее (акцент)
}

# Hatching patterns для различения bars в grayscale печати без цвета
HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '|||', '+++', '---']

# Standard figure dimensions for single-column thesis layout
FIG_NORMAL  = (5.0, 3.5)   # default — bars, lines
FIG_WIDE    = (6.5, 3.5)   # wide horizontal bars, time series
FIG_SQUARE  = (5.0, 4.5)   # confusion matrix
FIG_TALL    = (5.0, 5.5)   # vertical lists with many categories


def apply_style():
    """Apply unified scientific-publication style. Call once at script start."""
    mpl.rcParams.update({
        # Font: Times New Roman → Liberation Serif → DejaVu Serif (fallback)
        # Liberation Serif is visually identical to TNR and ships with Ubuntu.
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'font.size':         10,
        'axes.titlesize':    10,
        'axes.labelsize':    10,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   9,
        'figure.titlesize':  10,

        # Math text — match body font
        'mathtext.fontset':  'stix',

        # Lines and markers
        'lines.linewidth':   1.2,
        'lines.markersize':  5,
        'lines.markeredgewidth': 0.6,

        # Axes — top/right spines removed (modern scientific style)
        'axes.linewidth':    0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.edgecolor':    '#333333',
        'axes.labelcolor':   '#000000',
        'axes.titlepad':     8,
        'axes.labelpad':     4,

        # Ticks — outward, short
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  3.0,
        'xtick.major.width': 0.8,
        'ytick.major.size':  3.0,
        'ytick.major.width': 0.8,
        'xtick.color':       '#333333',
        'ytick.color':       '#333333',

        # Grid — subtle, behind data
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.color':        '#cccccc',
        'grid.linestyle':    '--',
        'grid.linewidth':    0.5,
        'grid.alpha':        0.5,

        # Legend — frameless, compact
        'legend.frameon':    False,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.2,
        'legend.borderaxespad': 0.4,

        # Figure
        'figure.figsize':    FIG_NORMAL,
        'figure.dpi':        100,
        'figure.facecolor':  'white',
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',
        'savefig.format':    'png',

        # Color cycle — Wong palette
        'axes.prop_cycle':   cycler(color=PALETTE),

        # Heatmap defaults — grayscale (Greys cmap)
        'image.cmap':        'Greys',
    })


def style_axes(ax, *, grid_axis='y'):
    """
    Дополнительная косметика для отдельной оси.

    grid_axis: 'x', 'y', 'both' или None — по какой оси показывать сетку.
               Для bar charts обычно 'y' (для horizontal bars — 'x').
    """
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)


def fmt_int(n: int) -> str:
    """
    Format integer with thousand separator as space (Russian/European convention).
    Example: 27_459_940 → '27 459 940'.
    """
    return f'{n:,}'.replace(',', ' ')
