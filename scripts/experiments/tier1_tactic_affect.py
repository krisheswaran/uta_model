#!/usr/bin/env python3
"""
Tier 1 Experiments: Tactic Discriminant Analysis (H3) & Affect Distribution Profiling (H1)

Reads parsed play data for cherry_orchard, hamlet, importance_of_being_earnest.
Produces structured text output suitable for an experiment log.

Experiment 1 (H3): Tactic Discriminant Analysis
  - For each canonical tactic with >= 15 occurrences across the corpus,
    compute mean affect/social vectors per play.
  - Test within-tactic vs between-tactic variance (F-ratio).
  - Flag tactics where between-play variance is high (potential genre effect).

Experiment 2 (H1): Affect Distribution Profiling
  - Descriptive stats for 5 affect dimensions by play and character.
  - Between-character vs within-character variance for each dimension.
  - Logistic regression / random forest predicting canonical_tactic from affect vector:
    compare all 5 dims vs valence+arousal+vulnerability only.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "parsed"
PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]
SOCIAL_DIMS = ["status", "warmth"]
ALL_DIMS = AFFECT_DIMS + SOCIAL_DIMS
MIN_TACTIC_COUNT = 15

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_beat_states():
    """Return list of dicts with flat fields for every beat_state across corpus."""
    rows = []
    for pid in PLAY_IDS:
        fpath = DATA_DIR / f"{pid}.json"
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping", file=sys.stderr)
            continue
        play = json.loads(fpath.read_text())
        for act in play["acts"]:
            for scene in act["scenes"]:
                for beat in scene["beats"]:
                    for bs in beat["beat_states"]:
                        ct = bs.get("canonical_tactic")
                        if ct is None:
                            continue
                        aff = bs.get("affect_state", {})
                        soc = bs.get("social_state", {})
                        row = {
                            "play_id": pid,
                            "character": bs["character"],
                            "canonical_tactic": ct,
                            "beat_id": bs["beat_id"],
                        }
                        for d in AFFECT_DIMS:
                            row[d] = float(aff.get(d, 0.0))
                        for d in SOCIAL_DIMS:
                            row[d] = float(soc.get(d, 0.0))
                        rows.append(row)
    return rows


def print_header(title):
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_subheader(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Experiment 1: Tactic Discriminant Analysis (H3)
# ---------------------------------------------------------------------------

def experiment1(rows):
    print_header("EXPERIMENT 1: Tactic Discriminant Analysis (H3)")
    print(f"Total beat_states with canonical_tactic: {len(rows)}")

    # Group by tactic
    by_tactic = defaultdict(list)
    for r in rows:
        by_tactic[r["canonical_tactic"]].append(r)

    # Filter to tactics with >= MIN_TACTIC_COUNT
    qualifying = {t: rs for t, rs in by_tactic.items() if len(rs) >= MIN_TACTIC_COUNT}
    print(f"Tactics with >= {MIN_TACTIC_COUNT} occurrences: {len(qualifying)} / {len(by_tactic)}")
    print()

    # --- 1a. Per-tactic affect/social profile ---
    print_subheader("1a. Per-tactic mean affect/social profile (play-weighted)")
    tactic_profiles = {}
    for tactic in sorted(qualifying.keys()):
        rs = qualifying[tactic]
        # Group by play, compute per-play means, then average across plays
        by_play = defaultdict(list)
        for r in rs:
            by_play[r["play_id"]].append(r)
        play_means = []
        for pid, prs in by_play.items():
            vec = [np.mean([r[d] for r in prs]) for d in ALL_DIMS]
            play_means.append(vec)
        play_means = np.array(play_means)
        overall_mean = play_means.mean(axis=0)
        tactic_profiles[tactic] = overall_mean

        plays_present = list(by_play.keys())
        counts = {pid: len(by_play[pid]) for pid in PLAY_IDS if pid in by_play}
        print(f"  {tactic:25s}  n={len(rs):3d}  plays={len(plays_present)}  "
              f"val={overall_mean[0]:+.2f}  aro={overall_mean[1]:+.2f}  "
              f"cer={overall_mean[2]:+.2f}  ctl={overall_mean[3]:+.2f}  "
              f"vul={overall_mean[4]:.2f}  sta={overall_mean[5]:+.2f}  "
              f"wrm={overall_mean[6]:+.2f}  {counts}")

    # --- 1b. Within-tactic vs between-tactic variance (MANOVA-lite via F-ratios) ---
    print_subheader("1b. Between-tactic vs within-tactic F-ratios per dimension")
    # One-way ANOVA per dimension across qualifying tactics
    for dim_idx, dim_name in enumerate(ALL_DIMS):
        groups = []
        for tactic in sorted(qualifying.keys()):
            vals = [r[dim_name] for r in qualifying[tactic]]
            groups.append(vals)
        f_stat, p_val = stats.f_oneway(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {dim_name:15s}  F={f_stat:8.2f}  p={p_val:.2e}  {sig}")

    # --- 1c. Genre effect: between-play variance within each tactic ---
    print_subheader("1c. Between-play variance within each tactic (genre effect flag)")
    print(f"  {'TACTIC':25s}  {'DIM':12s}  {'F':>8s}  {'p':>10s}  FLAG")
    genre_flags = []
    for tactic in sorted(qualifying.keys()):
        rs = qualifying[tactic]
        by_play = defaultdict(list)
        for r in rs:
            by_play[r["play_id"]].append(r)
        # Need at least 2 plays with >= 2 obs each
        play_groups = {pid: prs for pid, prs in by_play.items() if len(prs) >= 2}
        if len(play_groups) < 2:
            continue
        for dim_name in ALL_DIMS:
            groups = [np.array([r[dim_name] for r in prs]) for prs in play_groups.values()]
            if any(len(g) < 2 for g in groups):
                continue
            f_stat, p_val = stats.f_oneway(*groups)
            flag = "GENRE_EFFECT" if p_val < 0.01 else ""
            if flag:
                genre_flags.append((tactic, dim_name, f_stat, p_val))
                print(f"  {tactic:25s}  {dim_name:12s}  {f_stat:8.2f}  {p_val:10.2e}  {flag}")

    if not genre_flags:
        print("  (no strong genre effects detected at p<0.01)")

    # --- 1d. Tactic separation summary ---
    print_subheader("1d. Tactic cluster separation (pairwise Euclidean in 7D)")
    profiles = np.array([tactic_profiles[t] for t in sorted(qualifying.keys())])
    tactic_names = sorted(qualifying.keys())
    n_tactics = len(tactic_names)
    dists = np.zeros((n_tactics, n_tactics))
    for i in range(n_tactics):
        for j in range(n_tactics):
            dists[i, j] = np.linalg.norm(profiles[i] - profiles[j])

    # Find most separated and most overlapping pairs
    upper_tri = []
    for i in range(n_tactics):
        for j in range(i + 1, n_tactics):
            upper_tri.append((dists[i, j], tactic_names[i], tactic_names[j]))
    upper_tri.sort(reverse=True)

    print("\n  Most separated pairs (top 10):")
    for dist, t1, t2 in upper_tri[:10]:
        print(f"    {t1:25s} <-> {t2:25s}  d={dist:.3f}")

    print("\n  Most overlapping pairs (bottom 10):")
    for dist, t1, t2 in upper_tri[-10:]:
        print(f"    {t1:25s} <-> {t2:25s}  d={dist:.3f}")

    # Mean nearest-neighbor distance for each tactic
    print("\n  Per-tactic isolation (mean distance to nearest neighbor):")
    for i, tn in enumerate(tactic_names):
        row = dists[i].copy()
        row[i] = np.inf
        nn_dist = row.min()
        nn_name = tactic_names[np.argmin(row)]
        print(f"    {tn:25s}  nn_dist={nn_dist:.3f}  nearest={nn_name}")


# ---------------------------------------------------------------------------
# Experiment 2: Affect Distribution Profiling (H1)
# ---------------------------------------------------------------------------

def experiment2(rows):
    print_header("EXPERIMENT 2: Affect Distribution Profiling (H1)")
    print(f"Total beat_states: {len(rows)}")

    # --- 2a. Descriptive stats by play ---
    print_subheader("2a. Affect descriptive stats by play")
    by_play = defaultdict(list)
    for r in rows:
        by_play[r["play_id"]].append(r)

    for pid in PLAY_IDS:
        prs = by_play.get(pid, [])
        if not prs:
            continue
        print(f"\n  {pid} (n={len(prs)})")
        print(f"    {'DIM':15s}  {'mean':>6s}  {'std':>6s}  {'Q25':>6s}  {'Q50':>6s}  {'Q75':>6s}")
        for dim in AFFECT_DIMS:
            vals = np.array([r[dim] for r in prs])
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            print(f"    {dim:15s}  {vals.mean():+6.3f}  {vals.std():6.3f}  {q25:+6.3f}  {q50:+6.3f}  {q75:+6.3f}")

    # --- 2b. Descriptive stats by character (top characters only) ---
    print_subheader("2b. Affect stats for top characters (>= 10 beat_states)")
    by_char = defaultdict(list)
    for r in rows:
        key = f"{r['play_id']}:{r['character']}"
        by_char[key].append(r)

    qualifying_chars = {k: v for k, v in by_char.items() if len(v) >= 10}
    print(f"  Characters with >= 10 beat_states: {len(qualifying_chars)}")

    for key in sorted(qualifying_chars.keys()):
        crs = qualifying_chars[key]
        pid, char = key.split(":", 1)
        print(f"\n  {char} ({pid}, n={len(crs)})")
        print(f"    {'DIM':15s}  {'mean':>6s}  {'std':>6s}")
        for dim in AFFECT_DIMS:
            vals = np.array([r[dim] for r in crs])
            print(f"    {dim:15s}  {vals.mean():+6.3f}  {vals.std():6.3f}")

    # --- 2c. Between-character vs within-character variance ---
    print_subheader("2c. Between-character vs within-character F-ratios (ICC-like)")
    # One-way ANOVA: groups = characters (with >= 10 obs)
    for dim in AFFECT_DIMS:
        groups = []
        for key in sorted(qualifying_chars.keys()):
            vals = [r[dim] for r in qualifying_chars[key]]
            groups.append(vals)
        f_stat, p_val = stats.f_oneway(*groups)
        # Compute eta-squared
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {dim:15s}  F={f_stat:8.2f}  eta2={eta_sq:.3f}  p={p_val:.2e}  {sig}")

    # --- 2d. Tactic prediction: all 5 dims vs valence+arousal+vulnerability ---
    print_subheader("2d. Tactic prediction from affect vector")

    # Filter to tactics with enough samples for CV
    tactic_counts = defaultdict(int)
    for r in rows:
        tactic_counts[r["canonical_tactic"]] += 1
    valid_tactics = {t for t, c in tactic_counts.items() if c >= 10}
    pred_rows = [r for r in rows if r["canonical_tactic"] in valid_tactics]
    print(f"  Using {len(pred_rows)} beat_states across {len(valid_tactics)} tactics (each >= 10 obs)")

    le = LabelEncoder()
    y = le.fit_transform([r["canonical_tactic"] for r in pred_rows])

    # Feature sets
    X_full = np.array([[r[d] for d in AFFECT_DIMS] for r in pred_rows])
    X_reduced = np.array([[r["valence"], r["arousal"], r["vulnerability"]] for r in pred_rows])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest
    print("\n  Random Forest (5-fold stratified CV):")
    rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_reduced = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    scores_full = cross_val_score(rf_full, X_full, y, cv=cv, scoring="accuracy")
    scores_reduced = cross_val_score(rf_reduced, X_reduced, y, cv=cv, scoring="accuracy")
    print(f"    All 5 dims:              acc = {scores_full.mean():.3f} +/- {scores_full.std():.3f}")
    print(f"    Valence+Arousal+Vuln:    acc = {scores_reduced.mean():.3f} +/- {scores_reduced.std():.3f}")
    delta = scores_full.mean() - scores_reduced.mean()
    print(f"    Delta (certainty+control contribution): {delta:+.3f}")

    # Feature importance from full RF
    rf_full.fit(X_full, y)
    importances = rf_full.feature_importances_
    print("\n  Feature importance (Random Forest, all 5 dims):")
    for dim, imp in sorted(zip(AFFECT_DIMS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 100)
        print(f"    {dim:15s}  {imp:.3f}  {bar}")

    # Logistic Regression
    print("\n  Logistic Regression (5-fold stratified CV, multinomial):")
    lr_full = LogisticRegression(max_iter=1000, random_state=42)
    lr_reduced = LogisticRegression(max_iter=1000, random_state=42)

    lr_scores_full = cross_val_score(lr_full, X_full, y, cv=cv, scoring="accuracy")
    lr_scores_reduced = cross_val_score(lr_reduced, X_reduced, y, cv=cv, scoring="accuracy")
    print(f"    All 5 dims:              acc = {lr_scores_full.mean():.3f} +/- {lr_scores_full.std():.3f}")
    print(f"    Valence+Arousal+Vuln:    acc = {lr_scores_reduced.mean():.3f} +/- {lr_scores_reduced.std():.3f}")
    delta_lr = lr_scores_full.mean() - lr_scores_reduced.mean()
    print(f"    Delta (certainty+control contribution): {delta_lr:+.3f}")

    # Also include social dims to see full picture
    X_all7 = np.array([[r[d] for d in ALL_DIMS] for r in pred_rows])
    rf_all7 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    scores_all7 = cross_val_score(rf_all7, X_all7, y, cv=cv, scoring="accuracy")
    print(f"\n  RF with all 7 dims (affect+social): acc = {scores_all7.mean():.3f} +/- {scores_all7.std():.3f}")

    rf_all7.fit(X_all7, y)
    importances7 = rf_all7.feature_importances_
    print("  Feature importance (all 7 dims):")
    for dim, imp in sorted(zip(ALL_DIMS, importances7), key=lambda x: -x[1]):
        bar = "#" * int(imp * 100)
        print(f"    {dim:15s}  {imp:.3f}  {bar}")

    # --- 2e. Chance baseline ---
    print_subheader("2e. Baseline comparison")
    from collections import Counter
    tactic_dist = Counter(y)
    majority_class_acc = max(tactic_dist.values()) / len(y)
    uniform_chance = 1.0 / len(valid_tactics)
    print(f"  Majority class baseline: {majority_class_acc:.3f}")
    print(f"  Uniform random baseline: {uniform_chance:.3f}")
    print(f"  RF (5 affect dims):      {scores_full.mean():.3f}")
    print(f"  RF (3 core dims):        {scores_reduced.mean():.3f}")
    print(f"  RF (7 all dims):         {scores_all7.mean():.3f}")
    print(f"  LR (5 affect dims):      {lr_scores_full.mean():.3f}")
    print(f"  LR (3 core dims):        {lr_scores_reduced.mean():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  TIER 1 EXPERIMENTS: Tactic & Affect Analysis")
    print("  Data: cherry_orchard, hamlet, importance_of_being_earnest")
    print("=" * 72)

    rows = load_beat_states()
    if not rows:
        print("ERROR: No data loaded. Check data paths.", file=sys.stderr)
        sys.exit(1)

    # Corpus summary
    plays = set(r["play_id"] for r in rows)
    tactics = set(r["canonical_tactic"] for r in rows)
    chars = set(f"{r['play_id']}:{r['character']}" for r in rows)
    print(f"\nCorpus: {len(plays)} plays, {len(chars)} character-play pairs, "
          f"{len(tactics)} distinct tactics, {len(rows)} beat_states")

    experiment1(rows)
    experiment2(rows)

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
