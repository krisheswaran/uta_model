#!/usr/bin/env python3
"""
Affect Transition Eigendecomposition
=====================================
Determines whether the factor graph should model 5 correlated affect
dimensions or 2-3 independent ones by analyzing the covariance structure
of affect *transitions* (deltas between consecutive beats for a character
within a scene).

Steps:
  1. Extract affect transition deltas across all characters/plays.
  2. Compute 5x5 covariance matrix of deltas.
  3. Eigendecomposition → eigenvalues, eigenvectors, variance explained.
  4. Label the principal axes.
  5. Condition number comparison: full vs reduced diagonal model.
  6. Rotation examples for well-known characters.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "parsed"
PLAYS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]

# Characters of interest for rotation examples
SPOTLIGHT_CHARS = {
    "hamlet": ["HAMLET", "OPHELIA", "KING"],
    "cherry_orchard": ["LOPAKHIN", "LUBOV", "TROFIMOV"],
    "importance_of_being_earnest": ["LADY BRACKNELL", "JACK", "ALGERNON"],
}


# ── helpers ────────────────────────────────────────────────────────────────
def affect_vec(affect_state: dict) -> np.ndarray:
    """Extract the 5-d affect vector from a beat_state's affect_state dict."""
    return np.array([affect_state.get(d, 0.0) for d in AFFECT_DIMS], dtype=float)


def load_play(play_id: str) -> dict:
    path = DATA_DIR / f"{play_id}.json"
    with open(path) as f:
        return json.load(f)


# ── 1. collect deltas ─────────────────────────────────────────────────────
def collect_deltas_and_states():
    """Return (deltas, char_states) where:
      deltas: list of 5-d arrays (one per consecutive beat transition)
      char_states: {play_id: {character: [5-d arrays]}} for mean-state analysis
    """
    deltas = []
    char_states = defaultdict(lambda: defaultdict(list))
    total_beats = 0
    play_summaries = {}

    for play_id in PLAYS:
        play = load_play(play_id)
        play_deltas = 0
        play_beats = 0

        for act in play["acts"]:
            for scene in act["scenes"]:
                # Group beat_states by character, ordered by beat index
                char_beats = defaultdict(list)
                for beat in scene["beats"]:
                    play_beats += 1
                    for bs in beat.get("beat_states", []):
                        char = bs["character"]
                        vec = affect_vec(bs["affect_state"])
                        char_beats[char].append(vec)
                        char_states[play_id][char].append(vec)

                # Compute deltas for consecutive beats
                for char, vecs in char_beats.items():
                    for i in range(1, len(vecs)):
                        delta = vecs[i] - vecs[i - 1]
                        deltas.append(delta)
                        play_deltas += 1

        total_beats += play_beats
        play_summaries[play_id] = {"beats": play_beats, "deltas": play_deltas}

    return np.array(deltas), char_states, play_summaries, total_beats


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  AFFECT TRANSITION EIGENDECOMPOSITION")
    print("=" * 72)

    # ── Step 1: Collect data ──
    deltas, char_states, play_summaries, total_beats = collect_deltas_and_states()
    N = len(deltas)

    print(f"\n{'─'*72}")
    print("1. DATA COLLECTION")
    print(f"{'─'*72}")
    print(f"   Total beats across corpus:      {total_beats}")
    print(f"   Total affect transition deltas:  {N}")
    for pid, info in play_summaries.items():
        print(f"   {pid:40s}  beats={info['beats']:4d}  deltas={info['deltas']:4d}")

    # ── Step 2: Covariance matrix ──
    mean_delta = deltas.mean(axis=0)
    cov = np.cov(deltas, rowvar=False)

    print(f"\n{'─'*72}")
    print("2. COVARIANCE MATRIX OF AFFECT DELTAS")
    print(f"{'─'*72}")
    print(f"\n   Mean delta (should be near zero):")
    for i, d in enumerate(AFFECT_DIMS):
        print(f"     {d:15s}  {mean_delta[i]:+.4f}")
    print(f"\n   Covariance matrix:")
    header = "              " + "".join(f"{d:>14s}" for d in AFFECT_DIMS)
    print(header)
    for i, d in enumerate(AFFECT_DIMS):
        row = f"   {d:12s}" + "".join(f"{cov[i,j]:14.4f}" for j in range(5))
        print(row)

    # Also show correlation matrix for interpretability
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    print(f"\n   Correlation matrix:")
    print(header)
    for i, d in enumerate(AFFECT_DIMS):
        row = f"   {d:12s}" + "".join(f"{corr[i,j]:14.3f}" for j in range(5))
        print(row)

    # ── Step 3: Eigendecomposition ──
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var

    print(f"\n{'─'*72}")
    print("3. EIGENDECOMPOSITION")
    print(f"{'─'*72}")
    print(f"\n   Total variance: {total_var:.4f}")
    print(f"\n   {'Axis':<8s} {'Eigenvalue':>12s} {'% Var':>10s} {'Cumul %':>10s}")
    print(f"   {'─'*40}")
    for i in range(5):
        pct = eigenvalues[i] / total_var * 100
        print(f"   PC{i+1:<4d}  {eigenvalues[i]:12.4f} {pct:9.1f}% {cum_var[i]*100:9.1f}%")

    # How many for 90%?
    n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
    n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    print(f"\n   Axes for 90% variance: {n_90}")
    print(f"   Axes for 95% variance: {n_95}")

    # ── Step 4: Eigenvector loadings + labels ──
    print(f"\n{'─'*72}")
    print("4. EIGENVECTOR LOADINGS AND AXIS LABELS")
    print(f"{'─'*72}")

    for i in range(min(5, len(eigenvalues))):
        vec = eigenvectors[:, i]
        pct = eigenvalues[i] / total_var * 100
        print(f"\n   PC{i+1}  (eigenvalue={eigenvalues[i]:.4f}, {pct:.1f}% variance)")
        # Sort by absolute loading
        sorted_dims = sorted(range(5), key=lambda j: abs(vec[j]), reverse=True)
        for j in sorted_dims:
            bar = "█" * int(abs(vec[j]) * 30)
            sign = "+" if vec[j] > 0 else "-"
            print(f"     {AFFECT_DIMS[j]:15s}  {sign}{abs(vec[j]):.3f}  {bar}")

    # Label top axes
    print(f"\n   ── AXIS LABELS (based on loadings) ──")
    for i in range(min(3, len(eigenvalues))):
        vec = eigenvectors[:, i]
        loadings = {AFFECT_DIMS[j]: vec[j] for j in range(5)}
        sorted_l = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)

        # Build label
        pos_dims = [(d, v) for d, v in sorted_l if v > 0.3]
        neg_dims = [(d, v) for d, v in sorted_l if v < -0.3]

        pos_str = " + ".join(f"{d}({v:+.2f})" for d, v in pos_dims)
        neg_str = " + ".join(f"{d}({v:+.2f})" for d, v in neg_dims)

        if pos_str and neg_str:
            label = f"[{pos_str}]  vs  [{neg_str}]"
        elif pos_str:
            label = f"[{pos_str}]"
        else:
            label = f"[{neg_str}]"

        # Suggest semantic name
        top_abs = sorted_l[0][0]
        second_abs = sorted_l[1][0] if len(sorted_l) > 1 else ""

        print(f"\n   PC{i+1}: {label}")

    # ── Step 5: Condition number comparison ──
    print(f"\n{'─'*72}")
    print("5. CONDITION NUMBER ANALYSIS")
    print(f"{'─'*72}")

    cond_full = np.linalg.cond(cov)
    print(f"   Condition number of full 5×5 covariance:  {cond_full:.2f}")

    # Rotated covariance is diag(eigenvalues) — the off-diagonals are zero by construction
    # But if we keep only top-k, the reduced covariance is diag(eigenvalues[:k])
    for k in [2, 3]:
        if k <= len(eigenvalues):
            reduced_eigs = eigenvalues[:k]
            cond_reduced = max(reduced_eigs) / min(reduced_eigs)
            print(f"   Condition number of top-{k} diagonal model: {cond_reduced:.2f}")
            improvement = cond_full / cond_reduced
            print(f"     → {improvement:.1f}x better conditioned than full model")

    # Full diagonal in rotated basis (all 5 but diagonal)
    cond_diag_full = max(eigenvalues) / min(eigenvalues[eigenvalues > 1e-10])
    print(f"   Condition number of full diagonal (rotated): {cond_diag_full:.2f}")

    # ── Step 6: Rotation examples ──
    print(f"\n{'─'*72}")
    print("6. CHARACTER ROTATION EXAMPLES")
    print(f"{'─'*72}")

    # Build rotation matrix (columns are eigenvectors)
    W = eigenvectors  # shape (5, 5), columns are PCs

    pc_labels = [f"PC{i+1}" for i in range(5)]

    for play_id in PLAYS:
        spotlight = SPOTLIGHT_CHARS.get(play_id, [])
        for char in spotlight:
            states = char_states[play_id].get(char, [])
            if not states:
                # Try case variations
                for key in char_states[play_id]:
                    if key.upper() == char.upper():
                        states = char_states[play_id][key]
                        break
            if not states:
                print(f"\n   {char} ({play_id}): NO DATA FOUND")
                continue

            states_arr = np.array(states)
            mean_state = states_arr.mean(axis=0)
            std_state = states_arr.std(axis=0)

            # Rotate into eigenspace
            mean_rotated = W.T @ mean_state
            std_rotated = np.sqrt((W.T ** 2) @ (std_state ** 2))

            print(f"\n   {char} ({play_id})  [n={len(states)} beat-states]")
            print(f"     Original 5D space:")
            for j, d in enumerate(AFFECT_DIMS):
                bar = "█" * int(abs(mean_state[j]) * 20)
                sign = "+" if mean_state[j] >= 0 else "-"
                print(f"       {d:15s}  mean={mean_state[j]:+.3f}  std={std_state[j]:.3f}  {sign}{bar}")

            print(f"     Rotated eigenspace:")
            for j in range(5):
                bar = "█" * int(abs(mean_rotated[j]) * 20)
                sign = "+" if mean_rotated[j] >= 0 else "-"
                print(f"       {pc_labels[j]:15s}  mean={mean_rotated[j]:+.3f}  std={std_rotated[j]:.3f}  {sign}{bar}")

    # ── Summary ──
    print(f"\n{'═'*72}")
    print("  SUMMARY & RECOMMENDATIONS")
    print(f"{'═'*72}")
    print(f"""
   Data: {N} affect transition deltas from {len(PLAYS)} plays, {total_beats} beats.

   Dimensionality:
     - {n_90} principal components explain 90% of transition variance.
     - {n_95} principal components explain 95% of transition variance.
     - Top eigenvalue ratio: PC1 explains {eigenvalues[0]/total_var*100:.1f}% alone.

   Conditioning:
     - Full 5×5 covariance condition number: {cond_full:.1f}
     - Top-{min(3,len(eigenvalues))} diagonal model: {max(eigenvalues[:3])/min(eigenvalues[:3]):.1f}
     - Improvement: {cond_full/(max(eigenvalues[:3])/min(eigenvalues[:3])):.1f}x

   Recommendation:
     {"→ Model in REDUCED eigenspace (2-3 independent axes)" if n_90 <= 3 else "→ Model all 5 dimensions (variance is distributed)"}
     {"  The 5 affect dims are highly redundant; 2-3 rotated axes suffice." if n_90 <= 3 else "  The 5 affect dims each carry meaningful independent variance."}
""")


if __name__ == "__main__":
    main()
