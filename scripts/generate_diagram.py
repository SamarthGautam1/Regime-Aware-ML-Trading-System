"""
Generate docs/architecture.png — system architecture flowchart.

Run from project root:
    python scripts/generate_diagram.py

All X positions are derived mathematically from CX = CANVAS_W / 2.
Nothing is eyeballed.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DOCS_DIR     = os.path.join(PROJECT_ROOT, "docs")
OUT_PATH     = os.path.join(DOCS_DIR, "architecture.png")

os.makedirs(DOCS_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG           = "#1a1a2e"
ARROW_MAIN   = "#7f8fa6"
ARROW_BRANCH = "#e55039"
TEXT_SUB     = "#b2bec3"
SPINE_COLOR  = "#2d3436"
SEP_COLOR    = "#2a2a4e"

C = {
    "external":  "#1565C0",
    "data":      "#00796B",
    "strategy":  "#E64A19",
    "features":  "#6A1B9A",
    "ml":        "#283593",
    "signal":    "#0277BD",
    "backtest":  "#2E7D32",
    "metrics":   "#1B5E20",
    "dashboard": "#00695C",
    "paper":     "#B71C1C",
}

# ── Canvas ────────────────────────────────────────────────────────────────────
CANVAS_W = 13.0
CANVAS_H = 14.0

fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, CANVAS_W)
ax.set_ylim(0.2, CANVAS_H)
ax.axis("off")

# ── All X positions derived from the canvas centre ───────────────────────────
CX          = CANVAS_W / 2          # 6.5  — centre of every main-flow box

SPLIT_OFF   = 3.0                   # symmetric offset for strategy split
CX_L        = CX - SPLIT_OFF        # 3.5  — EMA Strategy
CX_R        = CX + SPLIT_OFF        # 9.5  — Mean Reversion

BRANCH_OFF  = 5.0                   # offset of paper-trading column from centre
CX_P        = CX + BRANCH_OFF       # 11.5 — Paper Trading / Alpaca Orders

LABEL_X     = 0.35                  # fixed X for every section label

# ── Box dimensions ────────────────────────────────────────────────────────────
BW_M = 3.8    # main-flow boxes
BW_S = 2.8    # strategy split boxes (EMA, MR)
BW_P = 2.6    # paper-trading column boxes
BH   = 0.64   # universal box height
HH   = BH / 2

# Derived edges used for arrows
MAIN_LEFT  = CX   - BW_M / 2    # 4.6
MAIN_RIGHT = CX   + BW_M / 2    # 8.4
EMA_RIGHT  = CX_L + BW_S / 2    # 4.9
MR_LEFT    = CX_R - BW_S / 2    # 8.1 (≈ MAIN_RIGHT — good visual symmetry)
PAPER_LEFT = CX_P - BW_P / 2    # 10.2

# ── Row Y positions (top → bottom, uniform 1.4-unit spacing) ─────────────────
Y = {
    "alpaca_in": 12.9,
    "data":      11.5,
    "strategy":  10.0,   # EMA and MR share this row
    "features":   8.5,
    "ml":         7.1,
    "signal":     5.7,
    "backtest":   4.3,   # Paper Trading shares this row
    "metrics":    2.9,   # Alpaca Orders shares this row
    "dashboard":  1.5,
}

# ── Nodes ─────────────────────────────────────────────────────────────────────
NODES = [
    # key           cx     cy                 label                  sublabel                       color           bw
    ("alpaca_in",   CX,    Y["alpaca_in"],    "Alpaca API",           "Historical market data",      C["external"],  BW_M),
    ("data",        CX,    Y["data"],         "Data Fetcher",          "data/fetch_data.py",          C["data"],      BW_M),
    ("ema",         CX_L,  Y["strategy"],     "EMA Strategy",          "Regime detection · signals",  C["strategy"],  BW_S),
    ("mr",          CX_R,  Y["strategy"],     "Mean Reversion",        "Bollinger Bands · regime",    C["strategy"],  BW_S),
    ("features",    CX,    Y["features"],     "Feature Engineering",   "7 features · no leakage",     C["features"],  BW_M),
    ("ml",          CX,    Y["ml"],           "ML Model",              "Logistic Regression pipeline",C["ml"],        BW_M),
    ("signal",      CX,    Y["signal"],       "Signal Generator",      "Long / Flat positions",       C["signal"],    BW_M),
    ("backtest",    CX,    Y["backtest"],     "Backtest Engine",       "5-strategy simulation",       C["backtest"],  BW_M),
    ("metrics",     CX,    Y["metrics"],      "Performance Metrics",   "Sharpe · CAGR · Drawdown",    C["metrics"],   BW_M),
    ("dashboard",   CX,    Y["dashboard"],    "Streamlit Dashboard",   "Interactive results",         C["dashboard"], BW_M),
    ("paper",       CX_P,  Y["backtest"],     "Paper Trading",         "Alpaca live execution",       C["paper"],     BW_P),
    ("orders",      CX_P,  Y["metrics"],      "Alpaca Orders",         "Live market orders",          C["external"],  BW_P),
]

# ── draw_box ──────────────────────────────────────────────────────────────────
def draw_box(cx, cy, label, sublabel, color, bw):
    x0 = cx - bw / 2
    y0 = cy - BH / 2

    # Glow shadow
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 - 0.06, y0 - 0.06), bw + 0.12, BH + 0.12,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="none",
        alpha=0.22, zorder=2,
    ))
    # Main box
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), bw, BH,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="white",
        linewidth=0.8, alpha=0.93, zorder=3,
    ))
    # Label
    ax.text(cx, cy + 0.10, label,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground=color)])
    # Sublabel
    ax.text(cx, cy - 0.13, sublabel,
            ha="center", va="center",
            fontsize=8.2, color=TEXT_SUB, alpha=0.85, zorder=5)


for _, cx, cy, label, sub, color, bw in NODES:
    draw_box(cx, cy, label, sub, color, bw)

# ── arrow ─────────────────────────────────────────────────────────────────────
def arrow(x1, y1, x2, y2, color=ARROW_MAIN):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.8, mutation_scale=15,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=4,
    )

# ── Main vertical flow (all start/end on CX) ─────────────────────────────────
arrow(CX, Y["alpaca_in"] - HH,  CX, Y["data"]     + HH)
arrow(CX, Y["features"]  - HH,  CX, Y["ml"]       + HH)
arrow(CX, Y["ml"]        - HH,  CX, Y["signal"]   + HH)
arrow(CX, Y["signal"]    - HH,  CX, Y["backtest"] + HH)
arrow(CX, Y["backtest"]  - HH,  CX, Y["metrics"]  + HH)
arrow(CX, Y["metrics"]   - HH,  CX, Y["dashboard"]+ HH)

# ── Data fetcher fan-out (symmetric about CX) ─────────────────────────────────
arrow(CX, Y["data"]     - HH,  CX_L, Y["strategy"] + HH)
arrow(CX, Y["data"]     - HH,  CX_R, Y["strategy"] + HH)

# ── Strategy merge into features (symmetric about CX) ────────────────────────
arrow(CX_L, Y["strategy"] - HH,  CX, Y["features"] + HH)
arrow(CX_R, Y["strategy"] - HH,  CX, Y["features"] + HH)

# ── Branch: backtest → paper (horizontal, right side) ────────────────────────
arrow(MAIN_RIGHT, Y["backtest"], PAPER_LEFT, Y["backtest"], color=ARROW_BRANCH)

# ── Branch: paper → orders (vertical, CX_P column) ───────────────────────────
arrow(CX_P, Y["backtest"] - HH,  CX_P, Y["metrics"] + HH, color=ARROW_BRANCH)

# ── Section labels (all at LABEL_X, vertically centred on each row) ──────────
SECTION_LABELS = [
    (Y["alpaca_in"], "EXTERNAL"),
    (Y["data"],      "DATA"),
    (Y["strategy"],  "STRATEGY"),
    (Y["features"],  "FEATURES"),
    (Y["ml"],        "ML"),
    (Y["signal"],    "SIGNAL"),
    (Y["backtest"],  "BACKTEST"),
    (Y["metrics"],   "METRICS"),
    (Y["dashboard"], "DASHBOARD"),
]

for y, label in SECTION_LABELS:
    ax.text(LABEL_X, y, label,
            ha="left", va="center",
            fontsize=6.8, color="#636e72",
            style="italic", fontweight="bold",
            alpha=0.65, zorder=5)

# Left spine line (runs the full height of the label column)
ax.plot([0.20, 0.20],
        [Y["dashboard"] - HH, Y["alpaca_in"] + HH],
        color=SPINE_COLOR, lw=0.8, zorder=1)

# ── Live trading label — centred above the horizontal branch arrow ────────────
# Midpoint of the horizontal arrow: x is between MAIN_RIGHT and PAPER_LEFT,
# y is the arrow's constant height (Y["backtest"]).
# The label is placed 25 points above that midpoint in screen space so it
# never overlaps the arrow line.  A solid background patch hides any bleed.
_arrow_mid_x = (MAIN_RIGHT + PAPER_LEFT) / 2
_arrow_y     = Y["backtest"]

ax.annotate(
    "LIVE TRADING",
    xy=(_arrow_mid_x, _arrow_y),
    xycoords="data",
    xytext=(0, 25),                    # 25 points above the arrow midpoint
    textcoords="offset points",
    ha="center", va="bottom",
    fontsize=6.8, color="#e55039",
    style="italic",
    alpha=0.85,
    zorder=6,
    bbox=dict(
        boxstyle="square,pad=2",
        facecolor=BG,                  # canvas background — hides the arrow line
        edgecolor="none",
        alpha=1.0,
    ),
    annotation_clip=False,
)

# Dashed separator between main column and paper-trading column
SEP_X = (MAIN_RIGHT + PAPER_LEFT) / 2   # midpoint between the two columns
ax.plot([SEP_X, SEP_X],
        [Y["dashboard"] - HH, Y["backtest"] + HH],
        color=SEP_COLOR, lw=1.0, linestyle="--", alpha=0.5, zorder=1)

# ── Title (centred on CX) ─────────────────────────────────────────────────────
ax.text(CX, 13.65,
        "Regime-Aware ML Trading System",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="white", zorder=5)

# Title underline
ax.plot([1.5, CANVAS_W - 1.5], [13.1, 13.1],
        color="#4a4a7a", lw=0.8, zorder=1)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.4)
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
