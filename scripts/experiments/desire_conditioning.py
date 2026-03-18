#!/usr/bin/env python3
"""
Desire-Conditioning Deep Dive
==============================
Building on H6 (desire similarity modulates tactic persistence, r=0.106, p=0.005),
this experiment investigates HOW desire conditions tactic transitions to inform
the design of the desire-conditioning factor in the factor graph.

Four analyses:
  1. Desire-conditioned transition matrices (tercile split)
  2. Desire cluster analysis (k-means on embeddings)
  3. Desire→tactic predictive model (logistic regression / random forest)
  4. Desire shift direction → tactic change mapping
"""

import json
import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "parsed"
VOCAB_PATH = PROJECT_ROOT / "data" / "vocab" / "tactic_vocabulary.json"

PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]


# ── Data loading (reused from h6_desire_embeddings.py) ───────────────────────

def load_tactic_vocab():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    member_to_canonical = {}
    for entry in vocab["tactics"]:
        cid = entry["canonical_id"]
        for m in entry["members"]:
            member_to_canonical[m.lower().strip()] = cid
    return member_to_canonical


def resolve_tactic(bs, member_to_canonical):
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
    return plays


def extract_character_beat_sequences(plays):
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


def collect_unique_desires(sequences):
    desires = set()
    for seq_info in sequences:
        for s in seq_info["states"]:
            d = s.get("desire_state", "").strip()
            if d:
                desires.add(d)
    return sorted(desires)


def embed_desires(desires, model):
    print(f"  Embedding {len(desires)} unique desire strings...")
    embeddings = model.encode(desires, show_progress_bar=False, normalize_embeddings=True)
    return {d: embeddings[i] for i, d in enumerate(desires)}


def collect_transitions_full(sequences, desire_embs, member_to_canonical):
    """
    Collect full transition records including desire strings, embeddings,
    character, and play info.
    """
    transitions = []
    for seq_info in sequences:
        states = seq_info["states"]
        character = seq_info["character"]
        play_id = seq_info["play_id"]
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

            sim = float(np.dot(desire_embs[d_prev], desire_embs[d_curr]))
            transitions.append({
                "t_prev": t_prev,
                "t_curr": t_curr,
                "d_prev": d_prev,
                "d_curr": d_curr,
                "d_emb_prev": desire_embs[d_prev],
                "d_emb_curr": desire_embs[d_curr],
                "sim": sim,
                "character": character,
                "play_id": play_id,
            })
    return transitions


def collect_all_beat_states(sequences, desire_embs, member_to_canonical):
    """Collect all individual beat states with resolved tactics and desire embeddings."""
    records = []
    for seq_info in sequences:
        for s in seq_info["states"]:
            tactic = resolve_tactic(s, member_to_canonical)
            if tactic is None:
                continue
            d = s.get("desire_state", "").strip()
            if not d or d not in desire_embs:
                continue
            records.append({
                "tactic": tactic,
                "desire": d,
                "desire_emb": desire_embs[d],
                "character": seq_info["character"],
                "play_id": seq_info["play_id"],
            })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Desire-conditioned transition matrices
# ══════════════════════════════════════════════════════════════════════════════

def analysis_1_tercile_transition_matrices(transitions):
    print("\n" + "=" * 78)
    print("ANALYSIS 1: DESIRE-CONDITIONED TRANSITION MATRICES (TERCILE SPLIT)")
    print("=" * 78)

    sims = np.array([t["sim"] for t in transitions])
    t1, t2 = np.percentile(sims, [33.33, 66.67])
    print(f"\n  Tercile boundaries: low < {t1:.3f}, medium [{t1:.3f}, {t2:.3f}), high >= {t2:.3f}")

    groups = {"low": [], "medium": [], "high": []}
    for t in transitions:
        if t["sim"] < t1:
            groups["low"].append(t)
        elif t["sim"] < t2:
            groups["medium"].append(t)
        else:
            groups["high"].append(t)

    for g, trs in groups.items():
        print(f"  {g}: {len(trs)} transitions")

    # Build transition matrices per tercile
    all_tactics = sorted(set(t["t_prev"] for t in transitions) | set(t["t_curr"] for t in transitions))
    tactic_idx = {t: i for i, t in enumerate(all_tactics)}
    n_tactics = len(all_tactics)

    matrices = {}
    for g_name, trs in groups.items():
        mat = np.zeros((n_tactics, n_tactics))
        for t in trs:
            mat[tactic_idx[t["t_prev"]], tactic_idx[t["t_curr"]]] += 1
        # Normalize rows
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat_norm = mat / row_sums
        matrices[g_name] = {"counts": mat, "probs": mat_norm}

    # Find top 10 most-changed transitions between low and high tercile
    print(f"\n  --- Top 10 Most-Changed Transitions (Low vs High Tercile) ---")
    print(f"  These transitions are most sensitive to desire stability:\n")

    diffs = []
    for i, t_from in enumerate(all_tactics):
        for j, t_to in enumerate(all_tactics):
            p_low = matrices["low"]["probs"][i, j]
            p_high = matrices["high"]["probs"][i, j]
            c_low = matrices["low"]["counts"][i, j]
            c_high = matrices["high"]["counts"][i, j]
            # Only consider transitions with some data in at least one tercile
            row_total_low = matrices["low"]["counts"][i, :].sum()
            row_total_high = matrices["high"]["counts"][i, :].sum()
            if row_total_low >= 3 or row_total_high >= 3:
                diffs.append({
                    "from": t_from, "to": t_to,
                    "p_low": p_low, "p_high": p_high,
                    "delta": p_high - p_low,
                    "abs_delta": abs(p_high - p_low),
                    "c_low": int(c_low), "c_high": int(c_high),
                })

    diffs.sort(key=lambda x: x["abs_delta"], reverse=True)

    print(f"  {'From':<14} {'To':<14} {'P(low)':<8} {'P(high)':<8} {'Delta':>8} {'N_low':>6} {'N_high':>6} {'Interpretation'}")
    print(f"  {'─'*14} {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*30}")
    for d in diffs[:10]:
        if d["from"] == d["to"]:
            interp = "PERSISTENCE " + ("INCREASES" if d["delta"] > 0 else "DECREASES") + " w/ desire stability"
        elif d["delta"] > 0:
            interp = "More likely when desire STABLE"
        else:
            interp = "More likely when desire SHIFTS"
        print(f"  {d['from']:<14} {d['to']:<14} {d['p_low']:<8.3f} {d['p_high']:<8.3f} "
              f"{d['delta']:>+8.3f} {d['c_low']:>6} {d['c_high']:>6} {interp}")

    # Transitions that ONLY happen during desire shifts (present in low, absent in high)
    print(f"\n  --- Transitions That ONLY Occur During Desire Shifts ---")
    print(f"  (Present in low tercile, absent in high tercile, count >= 2)\n")

    shift_only = [d for d in diffs if d["c_high"] == 0 and d["c_low"] >= 2]
    shift_only.sort(key=lambda x: x["c_low"], reverse=True)

    if shift_only:
        print(f"  {'From':<14} {'To':<14} {'Count (low)':>12}")
        print(f"  {'─'*14} {'─'*14} {'─'*12}")
        for d in shift_only[:15]:
            print(f"  {d['from']:<14} {d['to']:<14} {d['c_low']:>12}")
    else:
        print(f"  None found (all transitions with count >= 2 appear in both terciles)")

    # Transitions that ONLY happen during desire stability
    print(f"\n  --- Transitions That ONLY Occur During Desire Stability ---")
    print(f"  (Present in high tercile, absent in low tercile, count >= 2)\n")

    stable_only = [d for d in diffs if d["c_low"] == 0 and d["c_high"] >= 2]
    stable_only.sort(key=lambda x: x["c_high"], reverse=True)

    if stable_only:
        print(f"  {'From':<14} {'To':<14} {'Count (high)':>12}")
        print(f"  {'─'*14} {'─'*14} {'─'*12}")
        for d in stable_only[:15]:
            print(f"  {d['from']:<14} {d['to']:<14} {d['c_high']:>12}")
    else:
        print(f"  None found")

    # Overall persistence rates per tercile
    print(f"\n  --- Persistence (self-transition) Rates by Tercile ---")
    for g_name in ["low", "medium", "high"]:
        trs = groups[g_name]
        n = len(trs)
        persist = sum(1 for t in trs if t["t_prev"] == t["t_curr"]) / n if n > 0 else 0
        print(f"  {g_name:>8}: {persist:.3f} (N={n})")

    return groups, matrices, all_tactics


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Desire cluster analysis
# ══════════════════════════════════════════════════════════════════════════════

def analysis_2_desire_clusters(beat_states, desire_embs):
    print("\n" + "=" * 78)
    print("ANALYSIS 2: DESIRE CLUSTER ANALYSIS")
    print("=" * 78)

    # Get unique desires and their embeddings
    unique_desires = sorted(set(r["desire"] for r in beat_states))
    desire_list = unique_desires
    emb_matrix = np.array([desire_embs[d] for d in desire_list])

    print(f"\n  Unique desires: {len(desire_list)}")
    print(f"  Embedding dim:  {emb_matrix.shape[1]}")

    # Find optimal k by silhouette score
    print(f"\n  --- Silhouette Scores for k=3..10 ---")
    k_range = range(3, 11)
    sil_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(emb_matrix)
        sil = silhouette_score(emb_matrix, labels)
        sil_scores[k] = sil
        print(f"    k={k:>2}: silhouette = {sil:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n  Best k = {best_k} (silhouette = {sil_scores[best_k]:.4f})")

    # Fit with best k
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    desire_labels = km.fit_predict(emb_matrix)
    desire_to_cluster = {d: int(desire_labels[i]) for i, d in enumerate(desire_list)}

    # For each cluster: representative desires and tactic distribution
    print(f"\n  --- Desire Clusters (k={best_k}) ---")

    for c in range(best_k):
        cluster_desires = [d for d, cl in desire_to_cluster.items() if cl == c]
        # Find desires closest to centroid
        cluster_embs = np.array([desire_embs[d] for d in cluster_desires])
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        closest_idx = np.argsort(dists)[:3]

        print(f"\n  Cluster {c} ({len(cluster_desires)} desires):")
        print(f"    Representative desires:")
        for idx in closest_idx:
            d = cluster_desires[idx]
            # Truncate long desires
            display = d[:100] + "..." if len(d) > 100 else d
            print(f"      - \"{display}\"")

        # Tactic distribution for this cluster
        cluster_tactics = []
        for r in beat_states:
            if desire_to_cluster.get(r["desire"]) == c:
                cluster_tactics.append(r["tactic"])

        tactic_counts = Counter(cluster_tactics)
        total = sum(tactic_counts.values())
        print(f"    Tactic distribution (N={total}):")
        for tactic, count in tactic_counts.most_common(5):
            pct = count / total * 100
            print(f"      {tactic:<14} {count:>4} ({pct:5.1f}%)")

    # Chi-square test: is tactic distribution different across clusters?
    print(f"\n  --- Chi-Square Test: Tactic Distribution ~ Desire Cluster ---")

    # Build contingency table (cluster x tactic) for tactics with enough data
    all_tactics = sorted(set(r["tactic"] for r in beat_states))
    tactic_counts_global = Counter(r["tactic"] for r in beat_states)
    # Only include tactics that appear >= 5 times
    frequent_tactics = [t for t, c in tactic_counts_global.items() if c >= 5]

    contingency = np.zeros((best_k, len(frequent_tactics)))
    for r in beat_states:
        cl = desire_to_cluster.get(r["desire"])
        if cl is not None and r["tactic"] in frequent_tactics:
            contingency[cl, frequent_tactics.index(r["tactic"])] += 1

    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (contingency.sum() * (min(contingency.shape) - 1)))
    print(f"    chi2 = {chi2:.2f}, dof = {dof}, p = {p_chi:.6f}")
    print(f"    Cramer's V = {cramers_v:.4f}")
    if p_chi < 0.05:
        print(f"    SIGNIFICANT: Desire cluster predicts tactic distribution")
    else:
        print(f"    Not significant at p < 0.05")

    # Find the most distinctive tactic per cluster (highest residual)
    print(f"\n  --- Most Distinctive Tactic per Cluster (Standardized Residuals) ---")
    residuals = (contingency - expected) / np.sqrt(expected + 1e-10)
    for c in range(best_k):
        top_idx = np.argmax(residuals[c])
        print(f"    Cluster {c}: {frequent_tactics[top_idx]} (residual = {residuals[c, top_idx]:+.2f})")

    return desire_to_cluster, best_k


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Desire→tactic predictive model
# ══════════════════════════════════════════════════════════════════════════════

def analysis_3_predictive_models(beat_states):
    print("\n" + "=" * 78)
    print("ANALYSIS 3: DESIRE -> TACTIC PREDICTIVE MODEL")
    print("=" * 78)

    # Filter to tactics with enough samples for CV
    tactic_counts = Counter(r["tactic"] for r in beat_states)
    # Need at least 5 samples for 5-fold CV
    valid_tactics = {t for t, c in tactic_counts.items() if c >= 5}
    filtered = [r for r in beat_states if r["tactic"] in valid_tactics]

    print(f"\n  Total beat states: {len(beat_states)}")
    print(f"  After filtering (tactics with N>=5): {len(filtered)}")
    print(f"  Unique tactics: {len(valid_tactics)}")

    # Prepare features
    X_desire = np.array([r["desire_emb"] for r in filtered])
    le_tactic = LabelEncoder()
    y = le_tactic.fit_transform([r["tactic"] for r in filtered])

    le_char = LabelEncoder()
    char_labels = le_char.fit_transform([r["character"] for r in filtered])
    n_chars = len(le_char.classes_)
    X_char = np.zeros((len(filtered), n_chars))
    for i, cl in enumerate(char_labels):
        X_char[i, cl] = 1.0

    X_combined = np.hstack([X_desire, X_char])

    # Majority class baseline
    majority_class = Counter(y).most_common(1)[0][0]
    majority_acc = Counter(y).most_common(1)[0][1] / len(y)

    print(f"\n  Majority class: {le_tactic.inverse_transform([majority_class])[0]} "
          f"({majority_acc:.3f} accuracy)")

    # Cross-validated models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Character-only (LR)": (LogisticRegression(max_iter=1000, random_state=42, C=1.0), X_char),
        "Desire-only (LR)": (LogisticRegression(max_iter=1000, random_state=42, C=1.0), X_desire),
        "Combined (LR)": (LogisticRegression(max_iter=1000, random_state=42, C=1.0), X_combined),
        "Desire-only (RF)": (RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10), X_desire),
        "Combined (RF)": (RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10), X_combined),
    }

    print(f"\n  --- 5-Fold Cross-Validated Accuracy ---")
    print(f"  {'Model':<25} {'Mean Acc':>10} {'Std':>8} {'vs Majority':>12}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*12}")

    results = {}
    print(f"  {'Majority baseline':<25} {majority_acc:>10.3f} {'':>8} {'':>12}")

    for name, (model, X) in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()
        lift = mean_acc - majority_acc
        results[name] = {"mean": mean_acc, "std": std_acc, "lift": lift}
        print(f"  {name:<25} {mean_acc:>10.3f} {std_acc:>8.3f} {lift:>+12.3f}")

    # Compare desire-only vs character-only
    print(f"\n  --- Key Comparisons ---")
    desire_lr = results["Desire-only (LR)"]["mean"]
    char_lr = results["Character-only (LR)"]["mean"]
    combined_lr = results["Combined (LR)"]["mean"]

    print(f"  Desire adds {combined_lr - char_lr:+.3f} accuracy over character-only (LR)")
    print(f"  Character adds {combined_lr - desire_lr:+.3f} accuracy over desire-only (LR)")

    if combined_lr > char_lr + 0.01:
        print(f"  -> Desire provides meaningful predictive info beyond character identity")
    elif combined_lr > char_lr:
        print(f"  -> Desire provides marginal predictive info beyond character identity")
    else:
        print(f"  -> Desire does NOT add predictive power beyond character identity")

    # Top predicted tactics by the combined model
    print(f"\n  --- Per-Tactic F1 (Combined LR, full data) ---")
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_combined, y)
    y_pred = clf.predict(X_combined)
    report = classification_report(y, y_pred, target_names=le_tactic.classes_,
                                    output_dict=True, zero_division=0)
    tactic_f1 = [(t, report[t]["f1-score"], report[t]["support"])
                 for t in le_tactic.classes_ if t in report]
    tactic_f1.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Tactic':<14} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*14} {'─'*6} {'─'*8}")
    for tactic, f1, support in tactic_f1[:10]:
        print(f"  {tactic:<14} {f1:>6.3f} {support:>8}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Desire shift direction → tactic change
# ══════════════════════════════════════════════════════════════════════════════

def analysis_4_desire_shift_direction(transitions):
    print("\n" + "=" * 78)
    print("ANALYSIS 4: DESIRE SHIFT DIRECTION -> TACTIC CHANGE")
    print("=" * 78)

    # Filter to transitions where tactic changes AND desire changes (sim < median)
    sims = np.array([t["sim"] for t in transitions])
    median_sim = np.median(sims)

    change_transitions = [t for t in transitions
                          if t["t_prev"] != t["t_curr"] and t["sim"] < median_sim]
    print(f"\n  Transitions with tactic change AND desire shift (sim < {median_sim:.3f}): {len(change_transitions)}")

    if len(change_transitions) < 20:
        print("  Too few transitions for meaningful analysis. Skipping.")
        return

    # Compute delta vectors
    deltas = np.array([t["d_emb_curr"] - t["d_emb_prev"] for t in change_transitions])
    tactic_changes = [(t["t_prev"], t["t_curr"]) for t in change_transitions]

    # Cluster delta vectors
    print(f"\n  --- Clustering Desire Delta Vectors ---")
    k_range = range(3, 8)
    sil_scores = {}
    for k in k_range:
        if k >= len(change_transitions):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(deltas)
        sil = silhouette_score(deltas, labels)
        sil_scores[k] = sil
        print(f"    k={k}: silhouette = {sil:.4f}")

    if not sil_scores:
        print("  Could not cluster (too few samples).")
        return

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  Best k = {best_k} (silhouette = {sil_scores[best_k]:.4f})")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    delta_labels = km.fit_predict(deltas)

    # Per delta cluster: what tactic transitions happen?
    print(f"\n  --- Tactic Transitions per Desire-Delta Cluster ---")
    for c in range(best_k):
        mask = delta_labels == c
        cluster_changes = [tactic_changes[i] for i in range(len(tactic_changes)) if mask[i]]
        cluster_trans = [change_transitions[i] for i in range(len(change_transitions)) if mask[i]]

        n_cluster = len(cluster_changes)
        bigram_counts = Counter(cluster_changes)

        # Representative desire shifts
        cluster_deltas = deltas[mask]
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(cluster_deltas - centroid, axis=1)
        closest_idx = np.argsort(dists)[:2]

        print(f"\n  Delta Cluster {c} ({n_cluster} transitions):")
        print(f"    Representative desire shifts:")
        for idx in closest_idx:
            global_idx = np.where(mask)[0][idx]
            t = change_transitions[global_idx]
            d_from = t["d_prev"][:80] + "..." if len(t["d_prev"]) > 80 else t["d_prev"]
            d_to = t["d_curr"][:80] + "..." if len(t["d_curr"]) > 80 else t["d_curr"]
            print(f"      \"{d_from}\"")
            print(f"        -> \"{d_to}\"")

        print(f"    Top tactic transitions:")
        for (t_from, t_to), count in bigram_counts.most_common(5):
            pct = count / n_cluster * 100
            print(f"      {t_from:<14} -> {t_to:<14} {count:>3} ({pct:5.1f}%)")

        # What's the dominant target tactic?
        target_counts = Counter(tc for _, tc in cluster_changes)
        dominant_target = target_counts.most_common(1)[0]
        print(f"    Dominant target tactic: {dominant_target[0]} "
              f"({dominant_target[1]}/{n_cluster}, {dominant_target[1]/n_cluster*100:.1f}%)")

    # Test: does delta direction predict target tactic?
    print(f"\n  --- Predictive Test: Delta Vector -> Target Tactic ---")
    target_tactics = [tc for _, tc in tactic_changes]
    target_counts = Counter(target_tactics)
    valid_targets = {t for t, c in target_counts.items() if c >= 3}

    if len(valid_targets) < 2:
        print("  Too few valid target tactics for prediction.")
        return

    mask_valid = [tc in valid_targets for _, tc in tactic_changes]
    X_delta = deltas[mask_valid]
    y_target_raw = [tc for (_, tc), valid in zip(tactic_changes, mask_valid) if valid]

    le = LabelEncoder()
    y_target = le.fit_transform(y_target_raw)

    majority_acc = Counter(y_target).most_common(1)[0][1] / len(y_target)

    if len(y_target) >= 10 and len(set(y_target)) >= 2:
        n_splits = min(5, min(Counter(y_target).values()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            try:
                scores = cross_val_score(
                    LogisticRegression(max_iter=1000, random_state=42, C=0.1),
                    X_delta, y_target, cv=cv, scoring="accuracy"
                )
                print(f"    Majority baseline: {majority_acc:.3f}")
                print(f"    Delta-vector LR:   {scores.mean():.3f} (+/- {scores.std():.3f})")
                if scores.mean() > majority_acc + 0.02:
                    print(f"    -> Direction of desire shift DOES predict which new tactic is adopted")
                else:
                    print(f"    -> Direction of desire shift does NOT strongly predict target tactic")
            except Exception as e:
                print(f"    CV failed: {e}")
        else:
            print(f"    Too few samples per class for CV (min class has {min(Counter(y_target).values())})")
    else:
        print(f"    Not enough data for CV (N={len(y_target)}, classes={len(set(y_target))})")

    # Cosine analysis: are delta vectors for same target tactic more similar?
    print(f"\n  --- Within-Target vs Between-Target Delta Similarity ---")
    if len(valid_targets) >= 2 and len(X_delta) >= 10:
        within_sims = []
        between_sims = []
        for i in range(len(X_delta)):
            for j in range(i + 1, len(X_delta)):
                cos_sim = float(np.dot(X_delta[i], X_delta[j]) /
                               (np.linalg.norm(X_delta[i]) * np.linalg.norm(X_delta[j]) + 1e-10))
                if y_target[i] == y_target[j]:
                    within_sims.append(cos_sim)
                else:
                    between_sims.append(cos_sim)

        if within_sims and between_sims:
            within_mean = np.mean(within_sims)
            between_mean = np.mean(between_sims)
            t_stat, p_val = stats.mannwhitneyu(within_sims, between_sims, alternative="greater")
            print(f"    Within-target mean sim:  {within_mean:.4f} (N={len(within_sims)})")
            print(f"    Between-target mean sim: {between_mean:.4f} (N={len(between_sims)})")
            print(f"    Mann-Whitney U test (within > between): U={t_stat:.0f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"    SIGNIFICANT: Same target tactic -> more similar desire deltas")
            else:
                print(f"    Not significant: desire delta direction does not cluster by target tactic")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 78)
    print("SUMMARY: IMPLICATIONS FOR FACTOR GRAPH DESIGN")
    print("=" * 78)
    print("""
  This experiment explored HOW desire conditions tactic transitions, building
  on the H6 result that desire similarity modulates persistence (r=0.106, p=0.005).

  Key questions for factor graph design:

  1. TRANSITION MATRICES: Should the desire-conditioning factor use a single
     continuous modulation, or qualitatively different transition matrices?
     -> Check Analysis 1 for whether low/high similarity produce structurally
        different transition patterns.

  2. DESIRE CLUSTERS: Can desires be reduced to a small set of latent types
     that differentially drive tactics?
     -> Check Analysis 2 for cluster quality and tactic associations.

  3. PREDICTIVE POWER: Is desire embedding a useful feature, or is character
     identity sufficient?
     -> Check Analysis 3 for accuracy comparisons.

  4. DELTA DIRECTION: Should the factor graph model the direction of desire
     change, not just its magnitude?
     -> Check Analysis 4 for whether delta vectors predict target tactics.
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("DESIRE CONDITIONING DEEP DIVE")
    print("Building on H6: desire similarity -> tactic persistence (r=0.106, p=0.005)")
    print("=" * 78)

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
    print("\nLoading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    unique_desires = collect_unique_desires(sequences)
    desire_embs = embed_desires(unique_desires, model)

    # Collect all data
    print("\nCollecting transitions...")
    transitions = collect_transitions_full(sequences, desire_embs, member_to_canonical)
    print(f"  Total transitions: {len(transitions)}")

    print("\nCollecting all beat states...")
    beat_states = collect_all_beat_states(sequences, desire_embs, member_to_canonical)
    print(f"  Total beat states: {len(beat_states)}")

    # Run analyses
    analysis_1_tercile_transition_matrices(transitions)
    analysis_2_desire_clusters(beat_states, desire_embs)
    analysis_3_predictive_models(beat_states)
    analysis_4_desire_shift_direction(transitions)

    print_summary()

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
