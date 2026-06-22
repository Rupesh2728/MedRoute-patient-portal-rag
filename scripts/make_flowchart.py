"""Generate the architecture flowchart for the poster."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "poster_charts"
OUT_DIR.mkdir(exist_ok=True)

C_DARK = "#005BBB"
C_MID = "#4A90D9"
C_LIGHT = "#9DC3E6"
C_ACCENT = "#E76F51"
C_TEXT = "#1A1A1A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


def box(ax, x, y, w, h, text, fc, ec, txt_color="white", fontsize=10,
        weight="bold", lh=1.0):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center",
            color=txt_color, fontsize=fontsize, fontweight=weight,
            linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label="", color="#444", style="->", lw=1.4):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color,
    )
    ax.add_patch(a)
    if label:
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        ax.text(midx + 0.05, midy, label, fontsize=8.5,
                color="#444", style="italic")


fig, ax = plt.subplots(figsize=(9.5, 7.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# --- Top: Question + patient -------------------------------------------------
box(ax, 5.0, 9.3, 4.8, 0.8,
    "Patient question  +  patient record",
    fc=C_DARK, ec=C_DARK, fontsize=12)

# --- Feature extraction ------------------------------------------------------
box(ax, 5.0, 7.9, 6.4, 1.0,
    "Feature extraction (23 features)\n"
    "12 question-type one-hots  ·  4 retrieval scores\n"
    "2 KG features  ·  3 patient features  ·  2 question features",
    fc="white", ec=C_DARK, txt_color=C_TEXT, fontsize=9, weight="normal")

# --- Router ------------------------------------------------------------------
box(ax, 5.0, 6.4, 4.2, 1.0,
    "MLP Router\n23 → 32 → 16 → softmax(3)",
    fc=C_MID, ec=C_MID, fontsize=11)

# --- Three mode boxes --------------------------------------------------------
y_modes = 4.0
mode_w, mode_h = 2.9, 1.4

box(ax, 1.7, y_modes, mode_w, mode_h,
    "Mode 1\nLLM only\n(Gemma2 + prescriptions)",
    fc=C_LIGHT, ec=C_DARK, txt_color=C_TEXT, fontsize=10)

box(ax, 5.0, y_modes, mode_w, mode_h,
    "Mode 2\n+ Patient RAG\n(MedCPT + FAISS over chart)",
    fc=C_MID, ec=C_DARK, fontsize=10)

box(ax, 8.3, y_modes, mode_w, mode_h,
    "Mode 3\n+ Patient RAG + KG\n(PrimeKG drug triples)",
    fc=C_DARK, ec=C_DARK, fontsize=10)

# --- Final answer ------------------------------------------------------------
box(ax, 5.0, 1.5, 4.8, 0.9,
    "Final answer to patient",
    fc=C_ACCENT, ec=C_ACCENT, fontsize=12)

# --- Arrows ------------------------------------------------------------------
arrow(ax, 5.0, 8.85, 5.0, 8.45)        # question -> features
arrow(ax, 5.0, 7.35, 5.0, 6.95)        # features -> router

# router pick to each mode (only one fires per question)
arrow(ax, 4.0, 6.0, 1.7, y_modes + mode_h / 2 + 0.05,
      label="if M1", color="#888", style="->", lw=1.0)
arrow(ax, 5.0, 5.85, 5.0, y_modes + mode_h / 2 + 0.05,
      label="if M2", color="#888", style="->", lw=1.0)
arrow(ax, 6.0, 6.0, 8.3, y_modes + mode_h / 2 + 0.05,
      label="if M3", color="#888", style="->", lw=1.0)

# each mode -> final answer
arrow(ax, 1.7, y_modes - mode_h / 2 - 0.05, 5.0, 2.0, color="#444")
arrow(ax, 5.0, y_modes - mode_h / 2 - 0.05, 5.0, 2.0, color="#444")
arrow(ax, 8.3, y_modes - mode_h / 2 - 0.05, 5.0, 2.0, color="#444")

# --- Title -------------------------------------------------------------------
ax.text(5.0, 9.85, "System architecture: adaptive routing across 3 answer modes",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
        color=C_TEXT)

fig.tight_layout()
out = OUT_DIR / "flowchart_architecture.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
