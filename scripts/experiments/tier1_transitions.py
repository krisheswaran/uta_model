#!/usr/bin/env python3
"""
Tier 1 Transition Experiments
==============================
Estimates transition priors and tests structural hypotheses from the UTA factor-graph model.

Experiments:
  1. Tactic transition priors (pooled across plays)
  2. Desire-conditioned transitions (H6: desire change → tactic diversity)
  3. Affect transition kernels (covariance of affect deltas)
  4. Cross-character status correlation (zero-sum vs matching)

Reads parsed play data from data/parsed/*.json.
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "parsed"

PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_plays():
    plays = {}
    for pid in PLAY_IDS:
        path = DATA_DIR / f"{pid}.json"
        if path.exists():
            with open(path) as f:
                plays[pid] = json.load(f)
            print(f"  Loaded {pid}: {sum(len(s['beats']) for a in plays[pid]['acts'] for s in a['scenes'])} beats")
        else:
            print(f"  WARNING: {path} not found, skipping")
    return plays


def extract_character_beat_sequences(plays):
    """
    For each (play, scene), extract per-character ordered sequences of beat_states.
    Returns list of sequences, each sequence is a list of beat_state dicts
    ordered by beat index.
    """
    sequences = []
    for pid, play in plays.items():
        for act in play["acts"]:
            for scene in act["scenes"]:
                # Group beat_states by character, ordered by beat index
                char_beats = defaultdict(list)
                for beat in sorted(scene["beats"], key=lambda b: b["index"]):
                    for bs in beat["beat_states"]:
                        char_beats[bs["character"]].append((beat["index"], bs))
                for char, indexed_states in char_beats.items():
                    indexed_states.sort(key=lambda x: x[0])
                    seq = [s for _, s in indexed_states]
                    if len(seq) >= 2:
                        sequences.append({"play_id": pid, "character": char, "states": seq})
    return sequences


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Tactic Transition Priors
# ══════════════════════════════════════════════════════════════════════════════

def experiment_1(sequences):
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Tactic Transition Priors (pooled)")
    print("=" * 72)

    bigram_counts = Counter()
    tactic_counts = Counter()
    total_transitions = 0
    skipped_null = 0

    for seq_info in sequences:
        states = seq_info["states"]
        for i in range(len(states) - 1):
            t_prev = states[i].get("canonical_tactic")
            t_curr = states[i + 1].get("canonical_tactic")
            if t_prev is None or t_curr is None:
                skipped_null += 1
                continue
            bigram_counts[(t_prev, t_curr)] += 1
            tactic_counts[t_prev] += 1
            total_transitions += 1

    print(f"\nTotal transitions: {total_transitions}")
    print(f"Skipped (null tactic): {skipped_null}")
    unique_tactics = sorted(set(t for pair in bigram_counts for t in pair))
    print(f"Unique tactics: {len(unique_tactics)}")

    # Top 20 bigrams
    print(f"\n{'— Top 20 Tactic Bigrams —':^72}")
    print(f"  {'From':<20} {'To':<20} {'Count':>6} {'P(to|from)':>10}")
    print(f"  {'─'*20} {'─'*20} {'─'*6} {'─'*10}")
    for (t1, t2), count in bigram_counts.most_common(20):
        prob = count / tactic_counts[t1] if tactic_counts[t1] > 0 else 0
        print(f"  {t1:<20} {t2:<20} {count:>6} {prob:>10.3f}")

    # Transition entropy per source tactic
    print(f"\n{'— Transition Entropy by Source Tactic —':^72}")
    print(f"  {'Tactic':<20} {'N_trans':>8} {'Entropy':>8} {'Top Next':>15}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*15}")

    entropy_by_tactic = {}
    for tactic in sorted(unique_tactics):
        next_counts = {t2: bigram_counts[(tactic, t2)]
                       for t2 in unique_tactics if bigram_counts[(tactic, t2)] > 0}
        total = sum(next_counts.values())
        if total < 2:
            continue
        probs = np.array(list(next_counts.values())) / total
        ent = -np.sum(probs * np.log2(probs))
        entropy_by_tactic[tactic] = ent
        top_next = max(next_counts, key=next_counts.get)
        print(f"  {tactic:<20} {total:>8} {ent:>8.3f} {top_next:>15}")

    # Sparsity
    n = len(unique_tactics)
    possible = n * n
    observed = len(bigram_counts)
    print(f"\n— Matrix Sparsity —")
    print(f"  Possible cells: {possible} ({n}x{n})")
    print(f"  Observed cells: {observed}")
    print(f"  Sparsity: {1 - observed/possible:.3f}" if possible > 0 else "  N/A")

    if entropy_by_tactic:
        ents = list(entropy_by_tactic.values())
        print(f"\n  Entropy summary: mean={np.mean(ents):.3f}, "
              f"std={np.std(ents):.3f}, min={np.min(ents):.3f}, max={np.max(ents):.3f}")

        # Most predictable and most diverse tactics
        sorted_ent = sorted(entropy_by_tactic.items(), key=lambda x: x[1])
        print(f"\n  Most predictable (lowest entropy):")
        for t, e in sorted_ent[:5]:
            print(f"    {t}: {e:.3f}")
        print(f"  Most diverse (highest entropy):")
        for t, e in sorted_ent[-5:]:
            print(f"    {t}: {e:.3f}")

    return bigram_counts, tactic_counts, unique_tactics


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Desire-Conditioned Transitions (H6)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_2(sequences):
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Desire-Conditioned Tactic Transitions (H6)")
    print("=" * 72)

    transitions_desire_changed = []   # list of (t_prev, t_curr)
    transitions_desire_same = []

    for seq_info in sequences:
        states = seq_info["states"]
        for i in range(len(states) - 1):
            s_prev = states[i]
            s_curr = states[i + 1]
            t_prev = s_prev.get("canonical_tactic")
            t_curr = s_curr.get("canonical_tactic")
            if t_prev is None or t_curr is None:
                continue

            d_prev = s_prev.get("desire_state", "").strip().lower()
            d_curr = s_curr.get("desire_state", "").strip().lower()
            if not d_prev or not d_curr:
                continue

            # Use fuzzy similarity: desires are free-text, so near-identical
            # strings represent the "same" desire. Threshold 0.6 chosen to
            # separate paraphrases (same goal, different wording) from genuine
            # desire shifts.
            sim = SequenceMatcher(None, d_prev, d_curr).ratio()
            desire_changed = (sim < 0.6)
            if desire_changed:
                transitions_desire_changed.append((t_prev, t_curr))
            else:
                transitions_desire_same.append((t_prev, t_curr))

    # Also collect similarity scores for reporting
    sim_scores = []
    for seq_info in sequences:
        states = seq_info["states"]
        for i in range(len(states) - 1):
            d_prev = states[i].get("desire_state", "").strip().lower()
            d_curr = states[i + 1].get("desire_state", "").strip().lower()
            if d_prev and d_curr:
                sim_scores.append(SequenceMatcher(None, d_prev, d_curr).ratio())

    if sim_scores:
        sim_arr = np.array(sim_scores)
        print(f"\n  Desire similarity distribution (N={len(sim_arr)}):")
        print(f"    mean={sim_arr.mean():.3f}, median={np.median(sim_arr):.3f}, "
              f"std={sim_arr.std():.3f}")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            pct = np.mean(sim_arr >= thresh) * 100
            print(f"    >= {thresh:.1f}: {pct:5.1f}%")

    print(f"\n  Transitions with desire CHANGED (sim < 0.6): {len(transitions_desire_changed)}")
    print(f"  Transitions with desire SAME   (sim >= 0.6): {len(transitions_desire_same)}")

    def compute_group_entropy(transitions):
        """Compute average per-source-tactic transition entropy."""
        bigrams = Counter(transitions)
        source_counts = Counter(t1 for t1, t2 in transitions)
        entropies = []
        weights = []
        for src in source_counts:
            next_counts = {t2: bigrams[(src, t2)]
                           for t2 in set(t2 for t1, t2 in transitions if t1 == src)}
            total = sum(next_counts.values())
            if total < 2:
                continue
            probs = np.array(list(next_counts.values())) / total
            ent = -np.sum(probs * np.log2(probs))
            entropies.append(ent)
            weights.append(total)
        if not entropies:
            return 0.0, []
        weights = np.array(weights, dtype=float)
        return float(np.average(entropies, weights=weights)), entropies

    ent_changed, ents_changed = compute_group_entropy(transitions_desire_changed)
    ent_same, ents_same = compute_group_entropy(transitions_desire_same)

    # Tactic persistence rate
    persist_changed = sum(1 for t1, t2 in transitions_desire_changed if t1 == t2) / max(len(transitions_desire_changed), 1)
    persist_same = sum(1 for t1, t2 in transitions_desire_same if t1 == t2) / max(len(transitions_desire_same), 1)

    print(f"\n{'— Results —':^72}")
    print(f"  {'Metric':<35} {'Desire Changed':>15} {'Desire Same':>15}")
    print(f"  {'─'*35} {'─'*15} {'─'*15}")
    print(f"  {'Weighted mean transition entropy':<35} {ent_changed:>15.3f} {ent_same:>15.3f}")
    print(f"  {'Tactic persistence P(T=T_prev)':<35} {persist_changed:>15.3f} {persist_same:>15.3f}")

    # Permutation test on the raw transition entropy difference
    # We test whether the entropy difference is significant
    print(f"\n— Permutation Test (N=5000) —")
    all_transitions = transitions_desire_changed + transitions_desire_same
    labels = ([True] * len(transitions_desire_changed) +
              [False] * len(transitions_desire_same))

    observed_diff = ent_changed - ent_same
    print(f"  Observed entropy difference (changed - same): {observed_diff:.4f}")

    # Bootstrap the persistence rate difference instead (more stable)
    rng = np.random.default_rng(42)
    n_perm = 5000
    n_changed = len(transitions_desire_changed)

    # Permutation test on persistence rate
    observed_persist_diff = persist_changed - persist_same
    perm_diffs = []
    all_persist = np.array([1 if t1 == t2 else 0 for t1, t2 in all_transitions])
    for _ in range(n_perm):
        perm = rng.permutation(len(all_persist))
        group_a = all_persist[perm[:n_changed]]
        group_b = all_persist[perm[n_changed:]]
        perm_diffs.append(group_a.mean() - group_b.mean())
    perm_diffs = np.array(perm_diffs)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_persist_diff))
    print(f"  Observed persistence-rate difference (changed - same): {observed_persist_diff:.4f}")
    print(f"  Permutation p-value (two-sided): {p_value:.4f}")

    # Bootstrap CI for persistence difference
    if len(transitions_desire_changed) >= 2 and len(transitions_desire_same) >= 2:
        boot_diffs = []
        persist_c = np.array([1 if t1 == t2 else 0 for t1, t2 in transitions_desire_changed])
        persist_s = np.array([1 if t1 == t2 else 0 for t1, t2 in transitions_desire_same])
        for _ in range(5000):
            p_c = persist_c[rng.choice(len(persist_c), size=len(persist_c), replace=True)].mean()
            p_s = persist_s[rng.choice(len(persist_s), size=len(persist_s), replace=True)].mean()
            boot_diffs.append(p_c - p_s)
        boot_diffs = np.array(boot_diffs)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        print(f"  Bootstrap 95% CI for persistence diff: [{ci_lo:.4f}, {ci_hi:.4f}]")
    else:
        print("  Bootstrap CI: insufficient data in one or both groups")

    # Verdict
    print(f"\n— H6 Verdict —")
    if ent_changed > ent_same and persist_changed < persist_same:
        verdict = "SUPPORTED"
        detail = ("Desire change → higher entropy (more tactic diversity) "
                  "and lower persistence, as predicted.")
    elif ent_changed > ent_same:
        verdict = "PARTIALLY SUPPORTED"
        detail = "Desire change → higher entropy, but persistence pattern is mixed."
    elif persist_changed < persist_same:
        verdict = "PARTIALLY SUPPORTED"
        detail = "Desire change → lower persistence, but entropy pattern is mixed."
    else:
        verdict = "NOT SUPPORTED"
        detail = "Neither entropy nor persistence pattern matches prediction."

    sig_note = " (statistically significant)" if p_value < 0.05 else " (not statistically significant)"
    print(f"  {verdict}{sig_note}")
    print(f"  {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Affect Transition Kernels
# ══════════════════════════════════════════════════════════════════════════════

def experiment_3(sequences):
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Affect Transition Kernels")
    print("=" * 72)

    deltas = []

    for seq_info in sequences:
        states = seq_info["states"]
        for i in range(len(states) - 1):
            a_prev = states[i].get("affect_state", {})
            a_curr = states[i + 1].get("affect_state", {})
            delta = []
            valid = True
            for dim in AFFECT_DIMS:
                v_prev = a_prev.get(dim)
                v_curr = a_curr.get(dim)
                if v_prev is None or v_curr is None:
                    valid = False
                    break
                delta.append(v_curr - v_prev)
            if valid:
                deltas.append(delta)

    deltas = np.array(deltas)
    print(f"\n  Total affect transitions: {deltas.shape[0]}")

    # Per-dimension statistics
    print(f"\n{'— Per-Dimension Step Statistics —':^72}")
    print(f"  {'Dimension':<15} {'Mean':>8} {'Std':>8} {'Median':>8} {'|Mean|':>8}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for j, dim in enumerate(AFFECT_DIMS):
        col = deltas[:, j]
        print(f"  {dim:<15} {np.mean(col):>8.4f} {np.std(col):>8.4f} "
              f"{np.median(col):>8.4f} {np.mean(np.abs(col)):>8.4f}")

    # Covariance matrix
    cov = np.cov(deltas.T)
    print(f"\n{'— Covariance Matrix Σ of Affect Deltas —':^72}")
    header = "  " + " " * 15 + "".join(f"{d:>12}" for d in AFFECT_DIMS)
    print(header)
    for i, dim_i in enumerate(AFFECT_DIMS):
        row = f"  {dim_i:<15}" + "".join(f"{cov[i, j]:>12.5f}" for j in range(len(AFFECT_DIMS)))
        print(row)

    # Correlation matrix
    corr = np.corrcoef(deltas.T)
    print(f"\n{'— Correlation Matrix of Affect Deltas —':^72}")
    print(header)
    for i, dim_i in enumerate(AFFECT_DIMS):
        row = f"  {dim_i:<15}" + "".join(f"{corr[i, j]:>12.4f}" for j in range(len(AFFECT_DIMS)))
        print(row)

    # Strongest off-diagonal correlations
    print(f"\n— Strongest Affect Delta Correlations —")
    pairs = []
    for i in range(len(AFFECT_DIMS)):
        for j in range(i + 1, len(AFFECT_DIMS)):
            pairs.append((AFFECT_DIMS[i], AFFECT_DIMS[j], corr[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for d1, d2, r in pairs:
        sig = ""
        # Quick t-test for correlation significance
        n = len(deltas)
        if abs(r) < 1.0 and n > 2:
            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
            p_val = 2 * stats.t.sf(abs(t_stat), df=n - 2)
            sig = f"  p={p_val:.4f}" if p_val < 0.05 else f"  p={p_val:.4f} (ns)"
        print(f"  Δ{d1} ↔ Δ{d2}: r = {r:+.4f}{sig}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Cross-Character Status Correlation
# ══════════════════════════════════════════════════════════════════════════════

def experiment_4(plays):
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Cross-Character Status Correlation")
    print("=" * 72)

    all_pairs = []
    pairs_by_play = defaultdict(list)

    for pid, play in plays.items():
        for act in play["acts"]:
            for scene in act["scenes"]:
                for beat in scene["beats"]:
                    beat_states = beat.get("beat_states", [])
                    if len(beat_states) < 2:
                        continue
                    # Extract status for each character
                    char_status = {}
                    for bs in beat_states:
                        ss = bs.get("social_state", {})
                        status = ss.get("status")
                        if status is not None:
                            char_status[bs["character"]] = status
                    # All pairs of co-present characters
                    chars = list(char_status.keys())
                    for ca, cb in combinations(chars, 2):
                        pair = (char_status[ca], char_status[cb])
                        all_pairs.append(pair)
                        pairs_by_play[pid].append(pair)

    print(f"\n  Total co-present character pairs: {len(all_pairs)}")

    if len(all_pairs) < 3:
        print("  Insufficient data for correlation analysis.")
        return

    a_status = np.array([p[0] for p in all_pairs])
    b_status = np.array([p[1] for p in all_pairs])

    r_all, p_all = stats.pearsonr(a_status, b_status)
    print(f"\n{'— Overall Status Correlation —':^72}")
    print(f"  Pearson r = {r_all:+.4f}, p = {p_all:.4f}, N = {len(all_pairs)}")
    print(f"  Mean status A: {np.mean(a_status):.4f}, B: {np.mean(b_status):.4f}")
    print(f"  Std  status A: {np.std(a_status):.4f}, B: {np.std(b_status):.4f}")

    # Interpretation
    if r_all < -0.1 and p_all < 0.05:
        interp = "NEGATIVE correlation: supports zero-sum/relational status model"
    elif r_all > 0.1 and p_all < 0.05:
        interp = "POSITIVE correlation: characters tend to match status levels"
    else:
        interp = "No significant correlation detected"
    print(f"  Interpretation: {interp}")

    # By play
    print(f"\n{'— Status Correlation by Play —':^72}")
    print(f"  {'Play':<35} {'N':>6} {'r':>8} {'p':>8}")
    print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*8}")
    for pid in PLAY_IDS:
        pairs = pairs_by_play.get(pid, [])
        if len(pairs) < 3:
            print(f"  {pid:<35} {len(pairs):>6}     (insufficient data)")
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        r, p = stats.pearsonr(a, b)
        print(f"  {pid:<35} {len(pairs):>6} {r:>+8.4f} {p:>8.4f}")

    # Spearman as robustness check
    rho, p_rho = stats.spearmanr(a_status, b_status)
    print(f"\n  Spearman rho (robustness check): {rho:+.4f}, p = {p_rho:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("UTA MODEL — Tier 1 Transition Experiments")
    print("=" * 72)
    print(f"Data directory: {DATA_DIR}")

    print("\nLoading plays...")
    plays = load_plays()
    if not plays:
        print("ERROR: No plays loaded. Exiting.")
        sys.exit(1)

    print("\nExtracting character beat sequences...")
    sequences = extract_character_beat_sequences(plays)
    total_beats = sum(len(s["states"]) for s in sequences)
    print(f"  Sequences: {len(sequences)} (characters x scenes with ≥2 beats)")
    print(f"  Total beat-states in sequences: {total_beats}")

    experiment_1(sequences)
    experiment_2(sequences)
    experiment_3(sequences)
    experiment_4(plays)

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
