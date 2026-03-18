#!/usr/bin/env python3
"""
H6 Desire-Embedding Experiment
================================
Re-tests H6 (desire-state changes predict tactic transitions) using semantic
embedding similarity (all-MiniLM-L6-v2) instead of string matching.

The previous tier1_transitions.py experiment found the right direction
(desire change -> more tactic diversity) but was not significant (p=0.11)
because SequenceMatcher classified only 28/609 transitions as "same desire."

This script:
  1. Embeds all desire_state strings with sentence-transformers
  2. Computes cosine similarity for consecutive beat pairs per character
  3. Tests H6 at multiple thresholds (0.5..0.9) and as continuous correlation
  4. Reports distribution statistics and significance
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "parsed"
VOCAB_PATH = PROJECT_ROOT / "data" / "vocab" / "tactic_vocabulary.json"

PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_tactic_vocab():
    """Load tactic vocabulary for mapping tactic_state -> canonical."""
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    member_to_canonical = {}
    for entry in vocab["tactics"]:
        cid = entry["canonical_id"]
        for m in entry["members"]:
            member_to_canonical[m.lower().strip()] = cid
    return member_to_canonical


def resolve_tactic(bs, member_to_canonical):
    """Get canonical tactic from beat_state, falling back to vocab lookup."""
    ct = bs.get("canonical_tactic")
    if ct:
        return ct
    ts = bs.get("tactic_state", "")
    if ts:
        return member_to_canonical.get(ts.lower().strip())
    return None


def load_plays():
    plays = {}
    for pid in PLAY_IDS:
        path = DATA_DIR / f"{pid}.json"
        if path.exists():
            with open(path) as f:
                plays[pid] = json.load(f)
            n_beats = sum(len(s['beats']) for a in plays[pid]['acts'] for s in a['scenes'])
            print(f"  Loaded {pid}: {n_beats} beats")
        else:
            print(f"  WARNING: {path} not found, skipping")
    return plays


def extract_character_beat_sequences(plays):
    """Per (play, scene), extract per-character ordered beat_state sequences."""
    sequences = []
    for pid, play in plays.items():
        for act in play["acts"]:
            for scene in act["scenes"]:
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


# ── Embedding ────────────────────────────────────────────────────────────────

def collect_unique_desires(sequences):
    """Collect all unique desire_state strings."""
    desires = set()
    for seq_info in sequences:
        for s in seq_info["states"]:
            d = s.get("desire_state", "").strip()
            if d:
                desires.add(d)
    return sorted(desires)


def embed_desires(desires, model):
    """Embed desire strings, returns dict: desire_string -> embedding vector."""
    print(f"  Embedding {len(desires)} unique desire strings...")
    embeddings = model.encode(desires, show_progress_bar=False, normalize_embeddings=True)
    return {d: embeddings[i] for i, d in enumerate(desires)}


# ── Core analysis ────────────────────────────────────────────────────────────

def collect_transitions(sequences, desire_embs, member_to_canonical):
    """
    For each consecutive beat pair (same character, same scene),
    collect (tactic_prev, tactic_curr, desire_cosine_similarity).
    """
    transitions = []
    for seq_info in sequences:
        states = seq_info["states"]
        for i in range(len(states) - 1):
            s_prev = states[i]
            s_curr = states[i + 1]

            t_prev = resolve_tactic(s_prev, member_to_canonical)
            t_curr = resolve_tactic(s_curr, member_to_canonical)
            if t_prev is None or t_curr is None:
                continue

            d_prev = s_prev.get("desire_state", "").strip()
            d_curr = s_curr.get("desire_state", "").strip()
            if not d_prev or not d_curr or d_prev not in desire_embs or d_curr not in desire_embs:
                continue

            # Cosine similarity (embeddings are already L2-normalized)
            sim = float(np.dot(desire_embs[d_prev], desire_embs[d_curr]))
            transitions.append((t_prev, t_curr, sim))
    return transitions


def compute_group_stats(transitions_subset):
    """Compute weighted entropy and persistence rate for a group of transitions."""
    if not transitions_subset:
        return 0.0, 0.0, 0

    bigrams = Counter((t1, t2) for t1, t2, _ in transitions_subset)
    source_counts = Counter(t1 for t1, t2, _ in transitions_subset)

    entropies = []
    weights = []
    for src in source_counts:
        next_counts = {t2: bigrams[(src, t2)]
                       for t2 in set(t2 for (t1, t2) in bigrams if t1 == src)}
        total = sum(next_counts.values())
        if total < 2:
            continue
        probs = np.array(list(next_counts.values())) / total
        ent = -np.sum(probs * np.log2(probs))
        entropies.append(ent)
        weights.append(total)

    if not entropies:
        w_ent = 0.0
    else:
        w_ent = float(np.average(entropies, weights=weights))

    n = len(transitions_subset)
    persist = sum(1 for t1, t2, _ in transitions_subset if t1 == t2) / n
    return w_ent, persist, n


def permutation_test_persistence(transitions, threshold, n_perm=10000, rng=None):
    """
    Permutation test: is persistence rate different between
    same-desire (sim >= threshold) vs changed-desire (sim < threshold)?
    """
    if rng is None:
        rng = np.random.default_rng(42)

    same = [t for t in transitions if t[2] >= threshold]
    changed = [t for t in transitions if t[2] < threshold]

    if len(same) < 2 or len(changed) < 2:
        return None, None, len(same), len(changed)

    persist_same = np.mean([1 if t1 == t2 else 0 for t1, t2, _ in same])
    persist_changed = np.mean([1 if t1 == t2 else 0 for t1, t2, _ in changed])
    observed_diff = persist_same - persist_changed

    # Pool all persistence labels
    all_persist = np.array([1 if t1 == t2 else 0 for t1, t2, _ in transitions])
    n_same = len(same)

    perm_diffs = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(all_persist))
        g_same = all_persist[perm[:n_same]]
        g_changed = all_persist[perm[n_same:]]
        perm_diffs[k] = g_same.mean() - g_changed.mean()

    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))
    return observed_diff, p_value, len(same), len(changed)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("H6 DESIRE-EMBEDDING EXPERIMENT")
    print("Re-testing with semantic similarity instead of string matching")
    print("=" * 72)

    # Load data
    print("\nLoading plays...")
    plays = load_plays()
    if not plays:
        print("ERROR: No plays loaded.")
        sys.exit(1)

    member_to_canonical = load_tactic_vocab()
    print(f"  Tactic vocab: {len(member_to_canonical)} member -> canonical mappings")

    print("\nExtracting character beat sequences...")
    sequences = extract_character_beat_sequences(plays)
    print(f"  {len(sequences)} sequences (characters x scenes with >= 2 beats)")

    # Embed desires
    print("\nLoading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    unique_desires = collect_unique_desires(sequences)
    desire_embs = embed_desires(unique_desires, model)

    # Collect transitions with desire similarity
    print("\nCollecting transitions...")
    transitions = collect_transitions(sequences, desire_embs, member_to_canonical)
    print(f"  Total transitions with valid tactic + desire: {len(transitions)}")

    sims = np.array([s for _, _, s in transitions])
    persists = np.array([1 if t1 == t2 else 0 for t1, t2, _ in transitions])

    # ── 1. Desire similarity distribution ────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 1: Desire Similarity Distribution (Semantic Embeddings)")
    print("=" * 72)
    print(f"\n  N transitions: {len(sims)}")
    print(f"  Mean:   {sims.mean():.4f}")
    print(f"  Median: {np.median(sims):.4f}")
    print(f"  Std:    {sims.std():.4f}")
    print(f"  Min:    {sims.min():.4f}")
    print(f"  Max:    {sims.max():.4f}")
    q25, q50, q75 = np.percentile(sims, [25, 50, 75])
    print(f"  Q25:    {q25:.4f}")
    print(f"  Q50:    {q50:.4f}")
    print(f"  Q75:    {q75:.4f}")

    print(f"\n  Cumulative distribution:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        n_above = np.sum(sims >= thresh)
        pct = n_above / len(sims) * 100
        print(f"    sim >= {thresh:.2f}: {n_above:>5} ({pct:5.1f}%)")

    # ASCII histogram
    print(f"\n  Histogram (bin width = 0.05):")
    bins = np.arange(0.0, 1.05, 0.05)
    counts, _ = np.histogram(sims, bins=bins)
    max_count = max(counts)
    bar_width = 40
    for i in range(len(counts)):
        lo, hi = bins[i], bins[i + 1]
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"    [{lo:.2f}-{hi:.2f}) {counts[i]:>4} {bar}")

    # ── 2. Threshold-based analysis ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 2: Threshold-Based H6 Analysis")
    print("=" * 72)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    rng = np.random.default_rng(42)

    print(f"\n  {'Thresh':>6}  {'N_same':>7} {'N_changed':>9}  "
          f"{'Ent_same':>8} {'Ent_chg':>8}  "
          f"{'Pers_same':>9} {'Pers_chg':>9}  "
          f"{'Diff':>7} {'p-val':>7} {'Sig':>4}")
    print(f"  {'─'*6}  {'─'*7} {'─'*9}  {'─'*8} {'─'*8}  {'─'*9} {'─'*9}  {'─'*7} {'─'*7} {'─'*4}")

    results = []
    for thresh in thresholds:
        same = [t for t in transitions if t[2] >= thresh]
        changed = [t for t in transitions if t[2] < thresh]

        ent_same, pers_same, n_same = compute_group_stats(same)
        ent_changed, pers_changed, n_changed = compute_group_stats(changed)

        obs_diff, p_val, _, _ = permutation_test_persistence(
            transitions, thresh, n_perm=10000, rng=np.random.default_rng(42))

        sig = ""
        if p_val is not None:
            if p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"

        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        diff_str = f"{obs_diff:+.4f}" if obs_diff is not None else "N/A"

        print(f"  {thresh:>6.1f}  {n_same:>7} {n_changed:>9}  "
              f"{ent_same:>8.3f} {ent_changed:>8.3f}  "
              f"{pers_same:>9.3f} {pers_changed:>9.3f}  "
              f"{diff_str:>7} {p_str:>7} {sig:>4}")

        results.append({
            "threshold": thresh,
            "n_same": n_same, "n_changed": n_changed,
            "ent_same": ent_same, "ent_changed": ent_changed,
            "pers_same": pers_same, "pers_changed": pers_changed,
            "diff": obs_diff, "p_value": p_val, "sig": sig,
        })

    # ── 3. Continuous correlation analysis ───────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 3: Continuous Correlation (no threshold needed)")
    print("=" * 72)

    # Point-biserial: desire_similarity vs tactic_persistence (0/1)
    r_pb, p_pb = stats.pointbiserialr(persists, sims)
    print(f"\n  Point-biserial correlation:")
    print(f"    r = {r_pb:+.4f}, p = {p_pb:.6f}, N = {len(sims)}")

    # Spearman: desire_similarity rank vs tactic_persistence
    r_sp, p_sp = stats.spearmanr(sims, persists)
    print(f"\n  Spearman rank correlation:")
    print(f"    rho = {r_sp:+.4f}, p = {p_sp:.6f}")

    # Logistic-style analysis: bin by similarity quintile and show persistence
    print(f"\n  Persistence rate by desire-similarity quintile:")
    quintile_edges = np.percentile(sims, [0, 20, 40, 60, 80, 100])
    print(f"    {'Quintile':>10} {'Sim Range':>18} {'N':>5} {'Persist':>8}")
    print(f"    {'─'*10} {'─'*18} {'─'*5} {'─'*8}")
    for q in range(5):
        lo = quintile_edges[q]
        hi = quintile_edges[q + 1]
        if q < 4:
            mask = (sims >= lo) & (sims < hi)
        else:
            mask = (sims >= lo) & (sims <= hi)
        n_q = mask.sum()
        if n_q > 0:
            pers_q = persists[mask].mean()
            print(f"    Q{q+1:>8} [{lo:.3f}, {hi:.3f}{')'  if q < 4 else ']':>2} {n_q:>5} {pers_q:>8.3f}")

    # Bootstrap CI for correlation
    boot_rs = []
    for _ in range(5000):
        idx = rng.choice(len(sims), size=len(sims), replace=True)
        r_b, _ = stats.pointbiserialr(persists[idx], sims[idx])
        boot_rs.append(r_b)
    boot_rs = np.array(boot_rs)
    ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI for point-biserial r: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # ── 4. Comparison with string matching baseline ──────────────────────
    print("\n" + "=" * 72)
    print("SECTION 4: Comparison with String-Matching Baseline")
    print("=" * 72)

    from difflib import SequenceMatcher
    str_sims = []
    for t1, t2, sem_sim in transitions:
        # We need the original desire strings -- reconstruct from transitions
        pass  # We'll compute this separately

    # Re-collect with both similarity measures
    str_same_count = 0
    sem_same_counts = {t: 0 for t in thresholds}
    total_count = len(transitions)

    # Re-extract with string similarity
    str_transitions = []
    for seq_info in sequences:
        states = seq_info["states"]
        for i in range(len(states) - 1):
            s_prev = states[i]
            s_curr = states[i + 1]
            t_prev = resolve_tactic(s_prev, member_to_canonical)
            t_curr = resolve_tactic(s_curr, member_to_canonical)
            if t_prev is None or t_curr is None:
                continue
            d_prev = s_prev.get("desire_state", "").strip()
            d_curr = s_curr.get("desire_state", "").strip()
            if not d_prev or not d_curr:
                continue
            str_sim = SequenceMatcher(None, d_prev.lower(), d_curr.lower()).ratio()
            str_transitions.append(str_sim)

    str_sims_arr = np.array(str_transitions)
    sem_sims_arr = sims

    print(f"\n  String matching (SequenceMatcher):")
    print(f"    Mean similarity: {str_sims_arr.mean():.4f}")
    print(f"    >= 0.6 ('same'): {np.sum(str_sims_arr >= 0.6)} / {len(str_sims_arr)}")

    print(f"\n  Semantic embedding (all-MiniLM-L6-v2):")
    print(f"    Mean similarity: {sem_sims_arr.mean():.4f}")
    for t in thresholds:
        print(f"    >= {t}: {np.sum(sem_sims_arr >= t)} / {len(sem_sims_arr)}")

    # Correlation between the two similarity measures
    # They should be related but semantic should be more generous
    if len(str_sims_arr) == len(sem_sims_arr):
        r_methods, p_methods = stats.pearsonr(str_sims_arr, sem_sims_arr)
        print(f"\n  Correlation between string and semantic similarity:")
        print(f"    Pearson r = {r_methods:.4f}, p = {p_methods:.6f}")

    # ── 5. Summary and verdict ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 5: Summary and Verdict")
    print("=" * 72)

    # Find best threshold
    sig_results = [r for r in results if r["p_value"] is not None and r["p_value"] < 0.05]

    print(f"\n  Thresholds achieving significance (p < 0.05):")
    if sig_results:
        for r in sig_results:
            print(f"    threshold={r['threshold']:.1f}: "
                  f"persist_same={r['pers_same']:.3f}, "
                  f"persist_changed={r['pers_changed']:.3f}, "
                  f"diff={r['diff']:+.4f}, p={r['p_value']:.4f}")
    else:
        print(f"    None")

    # Best threshold (lowest p)
    valid_results = [r for r in results if r["p_value"] is not None]
    if valid_results:
        best = min(valid_results, key=lambda r: r["p_value"])
        print(f"\n  Best threshold: {best['threshold']:.1f} "
              f"(p={best['p_value']:.4f}, diff={best['diff']:+.4f})")

    print(f"\n  Continuous correlation:")
    print(f"    Point-biserial r = {r_pb:+.4f}, p = {p_pb:.6f}")
    print(f"    95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Overall verdict
    print(f"\n  --- H6 VERDICT ---")
    continuous_sig = p_pb < 0.05
    any_threshold_sig = len(sig_results) > 0

    if continuous_sig and any_threshold_sig:
        direction_ok = r_pb > 0  # Higher similarity -> more persistence
        if direction_ok:
            print(f"  SUPPORTED (significant)")
            print(f"  Higher desire similarity predicts higher tactic persistence.")
            print(f"  When the desire does not change, actors tend to persist with the same tactic.")
            print(f"  This justifies desire-conditioning in the transition factor.")
        else:
            print(f"  SIGNIFICANT BUT REVERSED")
            print(f"  Higher desire similarity predicts LOWER persistence (unexpected).")
    elif continuous_sig:
        print(f"  SUPPORTED (continuous test significant, threshold tests marginal)")
        print(f"  Point-biserial r = {r_pb:+.4f}, p = {p_pb:.6f}")
    elif any_threshold_sig:
        print(f"  PARTIALLY SUPPORTED (some thresholds significant, continuous marginal)")
    else:
        # Check if direction is at least right
        direction_ok = r_pb > 0
        if direction_ok and p_pb < 0.10:
            print(f"  WEAKLY SUPPORTED (right direction, marginal significance)")
            print(f"  r = {r_pb:+.4f}, p = {p_pb:.6f}")
        else:
            print(f"  NOT SUPPORTED")
            print(f"  Semantic embeddings did not improve significance over string matching.")

    # Effect size interpretation
    print(f"\n  Effect size interpretation:")
    abs_r = abs(r_pb)
    if abs_r < 0.1:
        print(f"    |r| = {abs_r:.4f} -> negligible effect")
    elif abs_r < 0.3:
        print(f"    |r| = {abs_r:.4f} -> small effect")
    elif abs_r < 0.5:
        print(f"    |r| = {abs_r:.4f} -> medium effect")
    else:
        print(f"    |r| = {abs_r:.4f} -> large effect")

    print(f"\n  Practical implication for factor graph:")
    if continuous_sig and r_pb > 0:
        print(f"    YES - desire similarity should modulate tactic transition priors.")
        print(f"    When desire_sim is high, increase self-transition probability.")
        print(f"    Suggested: P(T_t | T_{{t-1}}, D_sim) = softmax(alpha * T_{{t-1}} + beta * D_sim)")
    else:
        print(f"    The effect is {'in the right direction but weak' if r_pb > 0 else 'not in the predicted direction'}.")
        print(f"    Desire-conditioning may still be useful but sample size may be")
        print(f"    insufficient to detect a small effect at N={len(sims)}.")

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
