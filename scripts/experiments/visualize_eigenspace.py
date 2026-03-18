#!/usr/bin/env python3
"""
Visualize how the 5 original affect dimensions map onto
the top 3 principal component axes from the eigendecomposition.

Generates three publication-quality plots:
  1. Eigenvector loading heatmap
  2. Biplot-style arrow diagrams (3 panels: PC1vPC2, PC1vPC3, PC2vPC3)
  3. "More vs Less" spectrum diagrams for each PC
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "parsed"
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]
DIM_LABELS = ["Valence", "Arousal", "Certainty", "Control", "Vulnerability"]
PC_NAMES = ["PC1: Disempowerment", "PC2: Blissful\nIgnorance", "PC3: Burdened\nPower"]
PC_NAMES_SHORT = ["PC1 (Disempowerment)", "PC2 (Blissful Ignorance)", "PC3 (Burdened Power)"]

# Colors for each dimension
DIM_COLORS = {
    "Valence": "#e74c3c",
    "Arousal": "#f39c12",
    "Certainty": "#2ecc71",
    "Control": "#3498db",
    "Vulnerability": "#9b59b6",
}


# ── Recompute eigendecomposition ──────────────────────────────────────────
def affect_vec(affect_state: dict) -> np.ndarray:
    return np.array([affect_state.get(d, 0.0) for d in AFFECT_DIMS], dtype=float)


def compute_eigen():
    deltas = []
    for play_id in PLAYS:
        path = DATA_DIR / f"{play_id}.json"
        with open(path) as f:
            play = json.load(f)
        for act in play["acts"]:
            for scene in act["scenes"]:
                char_beats = defaultdict(list)
                for beat in scene["beats"]:
                    for bs in beat.get("beat_states", []):
                        char_beats[bs["character"]].append(affect_vec(bs["affect_state"]))
                for char, vecs in char_beats.items():
                    for i in range(1, len(vecs)):
                        deltas.append(vecs[i] - vecs[i - 1])

    deltas = np.array(deltas)
    cov = np.cov(deltas, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


eigenvalues, eigenvectors = compute_eigen()
total_var = eigenvalues.sum()

# Top 3 loadings: shape (5, 3)
loadings = eigenvectors[:, :3]

print("Eigenvalues:", eigenvalues)
print("Variance explained:", eigenvalues / total_var * 100)
print("\nLoadings (rows=dims, cols=PCs):")
for i, d in enumerate(DIM_LABELS):
    print(f"  {d:15s}  " + "  ".join(f"{loadings[i,j]:+.4f}" for j in range(3)))


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Heatmap
# ══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(6, 5))

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax1.imshow(loadings, cmap="RdBu_r", norm=norm, aspect="auto")

# Cell labels
for i in range(5):
    for j in range(3):
        val = loadings[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax1.text(j, i, f"{val:+.3f}", ha="center", va="center",
                 fontsize=13, fontweight="bold", color=color)

ax1.set_xticks(range(3))
ax1.set_xticklabels(PC_NAMES, fontsize=11)
ax1.set_yticks(range(5))
ax1.set_yticklabels(DIM_LABELS, fontsize=12)
ax1.set_title("Eigenvector Loadings: Original Dimensions → Principal Components",
              fontsize=13, fontweight="bold", pad=14)

# Variance explained annotation along bottom
for j in range(3):
    pct = eigenvalues[j] / total_var * 100
    ax1.text(j, 5.15, f"{pct:.1f}% var", ha="center", va="top",
             fontsize=10, color="#555", fontstyle="italic")
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(4.7, -0.5)

cbar = fig1.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label("Loading weight", fontsize=11)

fig1.tight_layout()
fig1.savefig(OUT_DIR / "eigenspace_heatmap.png", dpi=200, bbox_inches="tight")
print(f"\nSaved heatmap → {OUT_DIR / 'eigenspace_heatmap.png'}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Biplot arrow diagrams (3 panels)
# ══════════════════════════════════════════════════════════════════════════
pairs = [(0, 1), (0, 2), (1, 2)]
pair_labels = [
    ("PC1 (Disempowerment)", "PC2 (Blissful Ignorance)"),
    ("PC1 (Disempowerment)", "PC3 (Burdened Power)"),
    ("PC2 (Blissful Ignorance)", "PC3 (Burdened Power)"),
]

fig2, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for panel, (pc_x, pc_y) in enumerate(pairs):
    ax = axes[panel]
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.8, zorder=0)

    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#ddd", linewidth=0.8, linestyle="--", zorder=0)

    for i, dim in enumerate(DIM_LABELS):
        x = loadings[i, pc_x]
        y = loadings[i, pc_y]
        color = DIM_COLORS[dim]

        ax.annotate("", xy=(x, y), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                                     color=color, lw=2.5))

        # Offset label away from origin
        offset_x = 0.08 * np.sign(x) if abs(x) > 0.05 else 0.12
        offset_y = 0.08 * np.sign(y) if abs(y) > 0.05 else 0.08
        ax.text(x + offset_x, y + offset_y, dim, fontsize=11, fontweight="bold",
                color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color,
                          alpha=0.85, linewidth=0.8))

    ax.set_xlabel(pair_labels[panel][0], fontsize=12, fontweight="bold")
    ax.set_ylabel(pair_labels[panel][1], fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title(f"{pair_labels[panel][0]}  vs  {pair_labels[panel][1]}",
                 fontsize=11, fontweight="bold", pad=10)

fig2.suptitle("Biplot: Original Affect Dimensions in Eigenspace",
              fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "eigenspace_biplot.png", dpi=200, bbox_inches="tight")
print(f"Saved biplot → {OUT_DIR / 'eigenspace_biplot.png'}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: "More vs Less" spectrum diagrams
# ══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 7))

for pc_idx in range(3):
    ax = axes3[pc_idx]
    vec = loadings[:, pc_idx]
    pct = eigenvalues[pc_idx] / total_var * 100

    # Sort dims by loading value for this PC
    order = np.argsort(vec)  # most negative first

    # Draw the horizontal axis
    ax.axhline(0, color="#888", linewidth=1.5, zorder=1)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.8, 0.8)

    # Draw markers and labels for each dimension
    used_y = []
    for rank, dim_idx in enumerate(order):
        x = vec[dim_idx]
        dim = DIM_LABELS[dim_idx]
        color = DIM_COLORS[dim]

        # Stagger y positions to avoid overlap
        y_base = 0.0
        # Alternate above/below, with slight offset based on rank
        y_offsets = [0.35, -0.35, 0.55, -0.55, 0.15]
        y = y_offsets[rank]

        # Draw connecting line from axis to marker
        ax.plot([x, x], [0, y * 0.6], color=color, linewidth=1.5, alpha=0.5, zorder=2)

        # Draw the marker
        ax.scatter([x], [y * 0.6], s=180, color=color, zorder=3, edgecolors="white", linewidth=1.2)

        # Label
        fontsize = 12 if abs(x) > 0.3 else 10
        weight = "bold" if abs(x) > 0.3 else "normal"
        ax.text(x, y * 0.85, f"{dim}\n({x:+.2f})", ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, alpha=0.9, linewidth=0.8))

    # Named pole labels for each PC (negative end = left, positive end = right)
    pole_labels = [
        ("Agency", "Disempowerment"),
        ("Depressed Clarity", "Blissful Ignorance"),
        ("Hakuna Matata", "Burdened Power"),
    ]
    left_label, right_label = pole_labels[pc_idx]

    # Arrow + pole labels
    ax.annotate("", xy=(-1.08, 0), xytext=(-0.02, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.12", color="#555", lw=1.5))
    ax.annotate("", xy=(1.08, 0), xytext=(0.02, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.12", color="#555", lw=1.5))

    ax.text(-1.12, -0.02, f"\u2190 {left_label}", ha="right", va="top",
            fontsize=12, color="#222", fontweight="bold")
    ax.text(1.12, -0.02, f"{right_label} \u2192", ha="left", va="top",
            fontsize=12, color="#222", fontweight="bold")

    ax.set_title(f"{PC_NAMES_SHORT[pc_idx]}  —  {pct:.1f}% of variance",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_yticks([])
    ax.set_xticks(np.arange(-1, 1.1, 0.25))
    ax.tick_params(axis="x", labelsize=9, colors="#999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

fig3.suptitle('"More vs Less" Spectrum: What Each Principal Component Captures',
              fontsize=14, fontweight="bold")
fig3.tight_layout()
fig3.savefig(OUT_DIR / "eigenspace_spectrum.png", dpi=200, bbox_inches="tight")
print(f"Saved spectrum → {OUT_DIR / 'eigenspace_spectrum.png'}")


# ══════════════════════════════════════════════════════════════════════════
# Combined figure
# ══════════════════════════════════════════════════════════════════════════
print(f"\nAll plots saved to {OUT_DIR}/")
print("  - eigenspace_heatmap.png")
print("  - eigenspace_biplot.png")
print("  - eigenspace_spectrum.png")
