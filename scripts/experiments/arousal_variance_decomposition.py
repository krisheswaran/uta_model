"""
Variance decomposition for affect dimensions across plays.

Decomposes affect variance into:
  - Between-character (trait-like)
  - Between-scene (context-like)
  - Character x Scene interaction
  - Within-cell residual (moment-to-moment state)

Also computes ICCs and compares across all 5 affect dimensions.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "parsed"
PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]


def load_observations(play_ids: list[str]) -> list[dict]:
    """Extract (play, scene, character, affect_state) from all beat_states."""
    rows = []
    for play_id in play_ids:
        path = DATA_DIR / f"{play_id}.json"
        with open(path) as f:
            play = json.load(f)
        for act in play["acts"]:
            for scene in act["scenes"]:
                scene_id = f"{play_id}_a{scene['act']}_s{scene['scene']}"
                for beat in scene["beats"]:
                    for bs in beat["beat_states"]:
                        aff = bs.get("affect_state", {})
                        if aff is None:
                            continue
                        # Only include if we have numeric values for all dims
                        vals = {}
                        skip = False
                        for dim in AFFECT_DIMS:
                            v = aff.get(dim)
                            if v is None or not isinstance(v, (int, float)):
                                skip = True
                                break
                            vals[dim] = float(v)
                        if skip:
                            continue
                        rows.append({
                            "play_id": play_id,
                            "scene_id": scene_id,
                            "character": bs["character"],
                            **vals,
                        })
    return rows


def variance_decomposition(values: np.ndarray, characters: np.ndarray,
                           scenes: np.ndarray) -> dict:
    """
    Two-way unbalanced ANOVA-style variance decomposition.

    Uses Type-I sum of squares with OLS for a proper decomposition:
      SS_total = SS_character + SS_scene + SS_interaction + SS_residual

    Returns eta-squared (fraction of total SS) for each component.
    """
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)

    if ss_total == 0:
        return {"character": 0, "scene": 0, "interaction": 0, "residual": 0,
                "ss_total": 0}

    # Character means
    char_labels = np.unique(characters)
    scene_labels = np.unique(scenes)

    # --- SS between characters ---
    char_means = {}
    for c in char_labels:
        mask = characters == c
        char_means[c] = values[mask].mean()
    ss_character = sum(
        np.sum(characters == c) * (char_means[c] - grand_mean) ** 2
        for c in char_labels
    )

    # --- SS between scenes ---
    scene_means = {}
    for s in scene_labels:
        mask = scenes == s
        scene_means[s] = values[mask].mean()
    ss_scene = sum(
        np.sum(scenes == s) * (scene_means[s] - grand_mean) ** 2
        for s in scene_labels
    )

    # --- Cell means and interaction + residual ---
    # Build cell (character x scene) structure
    ss_cells = 0.0
    ss_residual = 0.0
    n_cells = 0
    for c in char_labels:
        for s in scene_labels:
            mask = (characters == c) & (scenes == s)
            cell_vals = values[mask]
            if len(cell_vals) == 0:
                continue
            n_cells += 1
            cell_mean = cell_vals.mean()
            ss_cells += len(cell_vals) * (cell_mean - grand_mean) ** 2
            ss_residual += np.sum((cell_vals - cell_mean) ** 2)

    # SS_interaction = SS_cells - SS_character - SS_scene
    ss_interaction = max(0, ss_cells - ss_character - ss_scene)

    # Sanity: ss_total ≈ ss_character + ss_scene + ss_interaction + ss_residual
    # (may not be exact due to unbalanced design, but close)

    return {
        "character": ss_character / ss_total,
        "scene": ss_scene / ss_total,
        "interaction": ss_interaction / ss_total,
        "residual": ss_residual / ss_total,
        "ss_total": ss_total,
        "n_obs": len(values),
        "n_characters": len(char_labels),
        "n_scenes": len(scene_labels),
        "n_cells": n_cells,
    }


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """
    ICC(1) — one-way random effects model.
    Fraction of total variance attributable to group membership.
    """
    labels = np.unique(groups)
    grand_mean = values.mean()
    n = len(values)
    k = len(labels)

    # Between-group MS and within-group MS
    group_ns = []
    group_means = []
    ss_within = 0.0
    for g in labels:
        mask = groups == g
        gv = values[mask]
        group_ns.append(len(gv))
        group_means.append(gv.mean())
        ss_within += np.sum((gv - gv.mean()) ** 2)

    ss_between = sum(
        gn * (gm - grand_mean) ** 2
        for gn, gm in zip(group_ns, group_means)
    )

    df_between = k - 1
    df_within = n - k

    if df_between == 0 or df_within == 0:
        return 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Average group size (harmonic mean for unbalanced)
    n0 = (n - sum(gn ** 2 for gn in group_ns) / n) / (k - 1)

    if (ms_between + (n0 - 1) * ms_within) == 0:
        return 0.0

    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    return max(0.0, icc)  # floor at 0


def main():
    rows = load_observations(PLAY_IDS)
    print(f"Total beat-state observations: {len(rows)}")
    print(f"Plays: {PLAY_IDS}\n")

    characters = np.array([r["character"] for r in rows])
    scenes = np.array([r["scene_id"] for r in rows])

    # ── 1. Variance decomposition for all 5 dimensions ──────────────
    print("=" * 72)
    print("VARIANCE DECOMPOSITION (eta-squared) — ALL AFFECT DIMENSIONS")
    print("=" * 72)
    print(f"{'Dimension':<15} {'Character':>10} {'Scene':>10} "
          f"{'Interact.':>10} {'Residual':>10}")
    print("-" * 72)

    decomp_results = {}
    for dim in AFFECT_DIMS:
        vals = np.array([r[dim] for r in rows])
        d = variance_decomposition(vals, characters, scenes)
        decomp_results[dim] = d
        print(f"{dim:<15} {d['character']:>10.3f} {d['scene']:>10.3f} "
              f"{d['interaction']:>10.3f} {d['residual']:>10.3f}")

    print()

    # ── 2. ICCs for all dimensions ──────────────────────────────────
    print("=" * 72)
    print("INTRACLASS CORRELATIONS (ICC)")
    print("=" * 72)
    print(f"{'Dimension':<15} {'ICC(char)':>10} {'ICC(scene)':>10} "
          f"{'Most like':>15}")
    print("-" * 72)

    icc_results = {}
    for dim in AFFECT_DIMS:
        vals = np.array([r[dim] for r in rows])
        icc_char = icc_oneway(vals, characters)
        icc_scene = icc_oneway(vals, scenes)
        label = "TRAIT" if icc_char > icc_scene else "CONTEXT"
        icc_results[dim] = {"character": icc_char, "scene": icc_scene}
        print(f"{dim:<15} {icc_char:>10.3f} {icc_scene:>10.3f} "
              f"{label:>15}")

    print()

    # ── 3. Rank dimensions by trait-likeness (character ICC) ────────
    print("=" * 72)
    print("RANKING BY TRAIT-LIKENESS (character ICC, descending)")
    print("=" * 72)
    ranked = sorted(AFFECT_DIMS, key=lambda d: icc_results[d]["character"],
                    reverse=True)
    for i, dim in enumerate(ranked, 1):
        marker = " <-- most trait-like" if i == 1 else ""
        print(f"  {i}. {dim:<15} ICC(char) = "
              f"{icc_results[dim]['character']:.3f}{marker}")

    print()

    # ── 4. Arousal deep-dive ────────────────────────────────────────
    print("=" * 72)
    print("AROUSAL DEEP-DIVE")
    print("=" * 72)

    arousal_vals = np.array([r["arousal"] for r in rows])
    print(f"  Grand mean arousal:    {arousal_vals.mean():.3f}")
    print(f"  Grand SD arousal:      {arousal_vals.std():.3f}")
    print(f"  ICC (character):       {icc_results['arousal']['character']:.3f}")
    print(f"  ICC (scene):           {icc_results['arousal']['scene']:.3f}")
    print()

    # Within-character-within-scene SD
    cell_sds = []
    cell_counts = defaultdict(list)
    for r in rows:
        cell_counts[(r["character"], r["scene_id"])].append(r["arousal"])

    for (char, scene), vals in cell_counts.items():
        if len(vals) >= 2:
            cell_sds.append(np.std(vals, ddof=1))

    mean_within_sd = np.mean(cell_sds) if cell_sds else 0
    median_within_sd = np.median(cell_sds) if cell_sds else 0
    print(f"  Within-character-within-scene arousal SD:")
    print(f"    Mean:   {mean_within_sd:.3f}")
    print(f"    Median: {median_within_sd:.3f}")
    print(f"    (across {len(cell_sds)} character-scene cells with >= 2 beats)")
    print()

    # ── 5. Per-play breakdown ───────────────────────────────────────
    print("=" * 72)
    print("PER-PLAY AROUSAL VARIANCE DECOMPOSITION")
    print("=" * 72)
    for play_id in PLAY_IDS:
        play_rows = [r for r in rows if r["play_id"] == play_id]
        if not play_rows:
            continue
        pv = np.array([r["arousal"] for r in play_rows])
        pc = np.array([r["character"] for r in play_rows])
        ps = np.array([r["scene_id"] for r in play_rows])
        d = variance_decomposition(pv, pc, ps)
        icc_c = icc_oneway(pv, pc)
        icc_s = icc_oneway(pv, ps)
        print(f"\n  {play_id} (n={len(play_rows)}, "
              f"{d['n_characters']} chars, {d['n_scenes']} scenes)")
        print(f"    Character:   {d['character']:.3f}   "
              f"Scene: {d['scene']:.3f}   "
              f"Interact: {d['interaction']:.3f}   "
              f"Residual: {d['residual']:.3f}")
        print(f"    ICC(char): {icc_c:.3f}   ICC(scene): {icc_s:.3f}")

    print()

    # ── 6. Interpretation ───────────────────────────────────────────
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    ar = decomp_results["arousal"]
    components = [
        ("Between-character (trait)", ar["character"]),
        ("Between-scene (context)", ar["scene"]),
        ("Character x Scene interaction", ar["interaction"]),
        ("Within-cell residual (moment-to-moment)", ar["residual"]),
    ]
    dominant = max(components, key=lambda x: x[1])

    print(f"\n  Arousal variance is dominated by: {dominant[0]} "
          f"({dominant[1]:.1%} of total variance)")
    print()

    if ar["residual"] > 0.5:
        print("  --> Arousal is primarily a MOMENT-TO-MOMENT STATE.")
        print("      It fluctuates substantially within the same character")
        print("      in the same scene, rather than being fixed by character")
        print("      identity or scene context.")
    elif ar["character"] > ar["scene"] and ar["character"] > ar["residual"]:
        print("  --> Arousal is primarily a CHARACTER TRAIT.")
        print("      Knowing who is speaking tells you more about arousal")
        print("      than knowing which scene we are in.")
    elif ar["scene"] > ar["character"] and ar["scene"] > ar["residual"]:
        print("  --> Arousal is primarily a SCENE-LEVEL CONTEXT variable.")
        print("      The dramatic situation dominates over character identity.")
    else:
        print("  --> Arousal variance is spread across multiple components.")
        print("      No single factor dominates.")

    # Compare arousal to other dims
    arousal_char_rank = ranked.index("arousal") + 1
    print(f"\n  Arousal ranks #{arousal_char_rank}/{len(AFFECT_DIMS)} "
          f"in trait-likeness (character ICC) among affect dimensions.")
    if arousal_char_rank == 1:
        print("  It is the MOST trait-like affect dimension.")
    elif arousal_char_rank == len(AFFECT_DIMS):
        print("  It is the LEAST trait-like affect dimension.")


if __name__ == "__main__":
    main()
