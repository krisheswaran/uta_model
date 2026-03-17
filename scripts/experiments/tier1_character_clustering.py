#!/usr/bin/env python3
"""
Tier 1 Character Clustering Experiments
========================================
H2: Character tactic distributions cluster by dramatic function, not by play.
Cross-genre discriminant validity and affect trajectory analysis.

Experiments:
  1. Character clustering by tactic distribution (H2)
  2. Cross-genre discriminant validity (PCA on combined embeddings)
  3. Character clustering by affect trajectory (windowed per-act)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PARSED_DIR = DATA_DIR / "parsed"
VOCAB_PATH = DATA_DIR / "vocab" / "tactic_vocabulary.json"

PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
PLAY_SHORT = {
    "cherry_orchard": "CO",
    "hamlet": "HAM",
    "importance_of_being_earnest": "IBE",
}

# Minimum number of beat_states to include a character (skip walk-ons)
MIN_BEATS = 5

# Expected clusters for H2 validation
EXPECTED_CLUSTERS = {
    "Wit-driven deflectors": {
        "chars": [("importance_of_being_earnest", "ALGERNON"),
                  ("hamlet", "HAMLET"),
                  ("cherry_orchard", "TROFIMOV")],
        "expected_tactics": ["MOCK", "DEFLECT", "CHALLENGE", "PROVOKE"],
    },
    "Romantic idealists/naifs": {
        "chars": [("cherry_orchard", "ANYA"),
                  ("importance_of_being_earnest", "CECILY"),
                  ("hamlet", "OPHELIA")],
        "expected_tactics": ["AFFIRM", "EMBRACE", "PLEAD"],
    },
    "Servants": {
        "chars": [("importance_of_being_earnest", "LANE"),
                  ("importance_of_being_earnest", "MERRIMAN"),
                  ("cherry_orchard", "FIERS"),
                  ("cherry_orchard", "YASHA")],
        "expected_tactics": [],
    },
    "Social dominators": {
        "chars": [("importance_of_being_earnest", "LADY BRACKNELL"),
                  ("hamlet", "KING"),
                  ("hamlet", "POLONIUS")],
        "expected_tactics": [],
    },
}

# Expected arc shapes for Experiment 3
EXPECTED_ARCS = {
    "descending": [("cherry_orchard", "LUBOV"), ("hamlet", "OPHELIA")],
    "ascending": [("importance_of_being_earnest", "JACK"),
                  ("importance_of_being_earnest", "CECILY"),
                  ("cherry_orchard", "ANYA")],
    "volatile": [("hamlet", "HAMLET"),
                 ("cherry_orchard", "LOPAKHIN"),
                 ("importance_of_being_earnest", "ALGERNON")],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_plays():
    plays = {}
    for pid in PLAY_IDS:
        with open(PARSED_DIR / f"{pid}.json") as f:
            plays[pid] = json.load(f)
    return plays


def load_tactic_vocab():
    with open(VOCAB_PATH) as f:
        v = json.load(f)
    return [t["canonical_id"] for t in v["tactics"]]


def extract_beat_states(play_data):
    """Yield (act_num, beat_state_dict) for every beat_state in the play."""
    for act in play_data["acts"]:
        act_num = act["number"]
        for scene in act["scenes"]:
            for beat in scene["beats"]:
                for bs in beat["beat_states"]:
                    yield act_num, bs


def resolve_tactic(bs, tactic_set):
    """Get canonical tactic from a beat_state, falling back to tactic_state matching.

    Some plays (e.g. IBE) have raw tactic_state but no canonical_tactic mapping.
    We try canonical_tactic first, then match tactic_state.upper() against vocab.
    """
    ct = bs.get("canonical_tactic")
    if ct and ct in tactic_set:
        return ct
    raw = bs.get("tactic_state", "").strip().upper()
    if raw in tactic_set:
        return raw
    return None


# ---------------------------------------------------------------------------
# Experiment 1: Character clustering by tactic distribution
# ---------------------------------------------------------------------------

def compute_tactic_distributions(plays, tactic_list):
    """
    Returns:
      char_labels: list of (play_id, character) tuples
      tactic_matrix: (n_chars, 66) normalized probability vectors
      char_tactic_counts: dict of (play_id, char) -> {tactic: count}
    """
    tactic_idx = {t: i for i, t in enumerate(tactic_list)}
    tactic_set = set(tactic_list)
    char_counts = defaultdict(lambda: np.zeros(len(tactic_list)))
    char_total_beats = defaultdict(int)

    for pid, play in plays.items():
        for act_num, bs in extract_beat_states(play):
            char = bs["character"]
            tactic = resolve_tactic(bs, tactic_set)
            char_total_beats[(pid, char)] += 1
            if tactic:
                char_counts[(pid, char)][tactic_idx[tactic]] += 1

    # Filter to characters with enough data
    char_labels = []
    rows = []
    char_tactic_counts = {}
    for key, counts in sorted(char_counts.items()):
        if char_total_beats[key] < MIN_BEATS:
            continue
        total = counts.sum()
        if total == 0:
            continue
        char_labels.append(key)
        rows.append(counts / total)  # normalized probability vector
        char_tactic_counts[key] = {tactic_list[i]: int(counts[i]) for i in range(len(tactic_list)) if counts[i] > 0}

    return char_labels, np.array(rows), char_tactic_counts


def run_experiment_1(plays, tactic_list):
    print("=" * 80)
    print("EXPERIMENT 1: Character Clustering by Tactic Distribution (H2)")
    print("=" * 80)

    char_labels, tactic_matrix, char_tactic_counts = compute_tactic_distributions(plays, tactic_list)
    n = len(char_labels)
    print(f"\nCharacters included (>={MIN_BEATS} beat_states): {n}")
    for pid, char in char_labels:
        top = sorted(char_tactic_counts[(pid, char)].items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{t}({c})" for t, c in top)
        print(f"  {PLAY_SHORT[pid]:>3s} | {char:<25s} | top: {top_str}")

    # Pairwise cosine similarity
    cos_dists = pdist(tactic_matrix, metric="cosine")
    cos_sim_matrix = 1 - squareform(cos_dists)

    # Hierarchical clustering (Ward linkage)
    Z = linkage(cos_dists, method="average")

    # Print dendrogram structure as text (merge order)
    print("\n--- Hierarchical Clustering (average linkage on cosine distance) ---")
    print(f"{'Step':>4s}  {'Cluster A':<30s}  {'Cluster B':<30s}  {'Distance':>8s}")
    cluster_names = {i: f"{PLAY_SHORT[char_labels[i][0]]}/{char_labels[i][1]}" for i in range(n)}
    for step, (a, b, dist, count) in enumerate(Z):
        a, b = int(a), int(b)
        name_a = cluster_names.get(a, f"cluster_{a}")
        name_b = cluster_names.get(b, f"cluster_{b}")
        new_id = n + step
        cluster_names[new_id] = f"({name_a} + {name_b})"
        print(f"{step:4d}  {name_a:<30s}  {name_b:<30s}  {dist:8.4f}")

    # Check expected clusters
    print("\n--- Expected Cluster Validation ---")
    for cluster_name, info in EXPECTED_CLUSTERS.items():
        chars = info["chars"]
        # Find indices
        indices = []
        for pid, char in chars:
            matches = [i for i, (p, c) in enumerate(char_labels) if p == pid and c == char]
            if matches:
                indices.append(matches[0])
            else:
                print(f"  WARNING: {char} ({pid}) not found in data")

        if len(indices) < 2:
            print(f"\n  {cluster_name}: insufficient characters found")
            continue

        # Within-cluster similarities
        within_sims = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_sims.append(cos_sim_matrix[indices[i], indices[j]])

        # Random baseline: all pairwise similarities
        all_sims = 1 - cos_dists
        mean_all = np.mean(all_sims)
        std_all = np.std(all_sims)

        mean_within = np.mean(within_sims)
        z_score = (mean_within - mean_all) / std_all if std_all > 0 else 0

        print(f"\n  {cluster_name}:")
        for idx in indices:
            pid, char = char_labels[idx]
            print(f"    - {PLAY_SHORT[pid]}/{char}")
        print(f"    Within-cluster mean cosine sim: {mean_within:.4f}")
        print(f"    Corpus baseline mean sim:       {mean_all:.4f} (std={std_all:.4f})")
        print(f"    Z-score vs baseline:            {z_score:+.2f}")

        # Show expected tactics enrichment
        if info["expected_tactics"]:
            tactic_idx = {t: i for i, t in enumerate(tactic_list)}
            print(f"    Expected enriched tactics: {info['expected_tactics']}")
            for idx in indices:
                pid, char = char_labels[idx]
                enriched_vals = {t: tactic_matrix[idx, tactic_idx[t]] for t in info["expected_tactics"]}
                enriched_str = ", ".join(f"{t}={v:.3f}" for t, v in enriched_vals.items())
                print(f"      {PLAY_SHORT[pid]}/{char}: {enriched_str}")

    # Sub-experiment: HAMLET per-act distributions
    print("\n--- Sub-experiment: HAMLET per-act tactic distributions ---")
    hamlet_play = plays["hamlet"]
    tactic_idx_map = {t: i for i, t in enumerate(tactic_list)}
    tactic_set_h = set(tactic_list)
    act_counts = defaultdict(lambda: np.zeros(len(tactic_list)))
    act_totals = defaultdict(int)

    for act_num, bs in extract_beat_states(hamlet_play):
        if bs["character"] == "HAMLET":
            tactic = resolve_tactic(bs, tactic_set_h)
            act_totals[act_num] += 1
            if tactic:
                act_counts[act_num][tactic_idx_map[tactic]] += 1

    for act_num in sorted(act_counts.keys()):
        counts = act_counts[act_num]
        total = counts.sum()
        if total == 0:
            print(f"  Act {act_num}: no tactic data")
            continue
        dist = counts / total

        # Find closest character (excluding HAMLET itself)
        best_sim = -1
        best_char = None
        for i, (pid, char) in enumerate(char_labels):
            if pid == "hamlet" and char == "HAMLET":
                continue
            sim = 1 - cosine(dist, tactic_matrix[i])
            if sim > best_sim:
                best_sim = sim
                best_char = f"{PLAY_SHORT[pid]}/{char}"

        # Find closest expected cluster
        best_cluster_sim = -1
        best_cluster = None
        for cluster_name, info in EXPECTED_CLUSTERS.items():
            indices = []
            for cpid, cchar in info["chars"]:
                matches = [j for j, (p, c) in enumerate(char_labels) if p == cpid and c == cchar]
                indices.extend(matches)
            if not indices:
                continue
            centroid = tactic_matrix[indices].mean(axis=0)
            sim = 1 - cosine(dist, centroid)
            if sim > best_cluster_sim:
                best_cluster_sim = sim
                best_cluster = cluster_name

        top_tactics = sorted(
            [(tactic_list[i], dist[i]) for i in range(len(tactic_list)) if dist[i] > 0],
            key=lambda x: -x[1]
        )[:5]
        top_str = ", ".join(f"{t}({v:.2f})" for t, v in top_tactics)
        print(f"  Act {act_num} (n={int(total)}): top={top_str}")
        print(f"    Closest character: {best_char} (sim={best_sim:.4f})")
        print(f"    Closest cluster:   {best_cluster} (sim={best_cluster_sim:.4f})")

    return char_labels, tactic_matrix


# ---------------------------------------------------------------------------
# Experiment 2: Cross-genre discriminant validity
# ---------------------------------------------------------------------------

def run_experiment_2(plays, tactic_list):
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Cross-Genre Discriminant Validity")
    print("=" * 80)

    tactic_idx = {t: i for i, t in enumerate(tactic_list)}
    tactic_set_2 = set(tactic_list)
    n_tactics = len(tactic_list)

    char_data = {}  # (pid, char) -> {"tactic": 66-vec, "affect": 5-vec, "social": 2-vec}

    for pid, play in plays.items():
        char_tactic_counts = defaultdict(lambda: np.zeros(n_tactics))
        char_affect_sums = defaultdict(lambda: np.zeros(5))
        char_social_sums = defaultdict(lambda: np.zeros(2))
        char_n = defaultdict(int)

        for act_num, bs in extract_beat_states(play):
            char = bs["character"]
            char_n[(pid, char)] += 1

            tactic = resolve_tactic(bs, tactic_set_2)
            if tactic:
                char_tactic_counts[(pid, char)][tactic_idx[tactic]] += 1

            aff = bs["affect_state"]
            char_affect_sums[(pid, char)] += np.array([
                aff["valence"], aff["arousal"], aff["certainty"],
                aff["control"], aff["vulnerability"]
            ])
            soc = bs["social_state"]
            char_social_sums[(pid, char)] += np.array([soc["status"], soc["warmth"]])

        for key in char_n:
            if char_n[key] < MIN_BEATS:
                continue
            tc = char_tactic_counts[key]
            total = tc.sum()
            if total == 0:
                continue
            char_data[key] = {
                "tactic": tc / total,
                "affect": char_affect_sums[key] / char_n[key],
                "social": char_social_sums[key] / char_n[key],
            }

    # Build combined 73D embedding
    labels = sorted(char_data.keys())
    embeddings = []
    for key in labels:
        d = char_data[key]
        embeddings.append(np.concatenate([d["tactic"], d["affect"], d["social"]]))
    X = np.array(embeddings)

    print(f"\nCharacters in embedding: {len(labels)}")
    print(f"Embedding dimensions: {X.shape[1]} (66 tactic + 5 affect + 2 social)")

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print(f"\nPCA variance explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    # Check play-level clustering vs mixing
    play_groups = defaultdict(list)
    for i, (pid, char) in enumerate(labels):
        play_groups[pid].append(i)

    print("\n--- Play-level cluster analysis in PCA space ---")
    # Compute within-play vs between-play distances
    all_dists = pdist(X_pca)
    dist_matrix = squareform(all_dists)

    within_dists = []
    between_dists = []
    for pid, indices in play_groups.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_dists.append(dist_matrix[indices[i], indices[j]])
    for pid1 in play_groups:
        for pid2 in play_groups:
            if pid1 >= pid2:
                continue
            for i in play_groups[pid1]:
                for j in play_groups[pid2]:
                    between_dists.append(dist_matrix[i, j])

    within_mean = np.mean(within_dists)
    between_mean = np.mean(between_dists)
    ratio = within_mean / between_mean if between_mean > 0 else float("inf")

    print(f"  Within-play mean distance:  {within_mean:.4f}")
    print(f"  Between-play mean distance: {between_mean:.4f}")
    print(f"  Ratio (within/between):     {ratio:.4f}")
    print(f"  Interpretation: {'GOOD - overlap/mixing' if ratio > 0.7 else 'CONCERNING - plays cluster separately'}")

    # Identify bridging characters
    print("\n--- Bridging Characters (closest to another play's characters) ---")
    for pid in PLAY_IDS:
        other_indices = [i for i, (p, c) in enumerate(labels) if p != pid]
        my_indices = [i for i, (p, c) in enumerate(labels) if p == pid]
        for mi in my_indices:
            best_dist = float("inf")
            best_other = None
            for oi in other_indices:
                d = dist_matrix[mi, oi]
                if d < best_dist:
                    best_dist = d
                    best_other = oi
            if best_other is not None:
                pid_me, char_me = labels[mi]
                pid_them, char_them = labels[best_other]
                # Only print if this is notably close
                if best_dist < np.percentile(all_dists, 25):
                    print(f"  {PLAY_SHORT[pid_me]}/{char_me} <-> {PLAY_SHORT[pid_them]}/{char_them} (dist={best_dist:.4f})")

    # Print PCA coordinates by play
    print("\n--- PCA coordinates (PC1, PC2) by play ---")
    for pid in PLAY_IDS:
        print(f"\n  {PLAY_SHORT[pid]}:")
        for i, (p, c) in enumerate(labels):
            if p == pid:
                print(f"    {c:<25s}  ({X_pca[i, 0]:+.3f}, {X_pca[i, 1]:+.3f}, {X_pca[i, 2]:+.3f})")


# ---------------------------------------------------------------------------
# Experiment 3: Character clustering by affect trajectory
# ---------------------------------------------------------------------------

def run_experiment_3(plays, tactic_list):
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Character Clustering by Affect Trajectory (Windowed)")
    print("=" * 80)

    # For each character, compute per-act: mean valence, mean arousal, mean vulnerability, trend slope
    char_act_data = defaultdict(lambda: defaultdict(list))  # (pid, char) -> act -> list of (val, aro, vul)

    for pid, play in plays.items():
        for act_num, bs in extract_beat_states(play):
            char = bs["character"]
            aff = bs["affect_state"]
            char_act_data[(pid, char)][act_num].append(
                (aff["valence"], aff["arousal"], aff["vulnerability"])
            )

    # Build feature vectors: for each act, (mean_val, mean_aro, mean_vul, val_slope)
    # Concatenate across acts (pad with zeros for missing acts)
    max_acts = max(
        max(acts.keys()) for acts in char_act_data.values()
    )

    char_labels = []
    feature_rows = []
    char_summaries = {}

    for key in sorted(char_act_data.keys()):
        acts = char_act_data[key]
        total_beats = sum(len(v) for v in acts.values())
        if total_beats < MIN_BEATS:
            continue

        # Compute per-act summaries
        act_means = {}
        valence_series = []
        for a in range(1, max_acts + 1):
            if a in acts and len(acts[a]) > 0:
                arr = np.array(acts[a])
                mean_val = arr[:, 0].mean()
                mean_aro = arr[:, 1].mean()
                mean_vul = arr[:, 2].mean()
                act_means[a] = (mean_val, mean_aro, mean_vul)
                valence_series.append((a, mean_val))
            else:
                act_means[a] = (0, 0, 0)

        # Compute valence trend slope
        if len(valence_series) >= 2:
            xs = np.array([v[0] for v in valence_series], dtype=float)
            ys = np.array([v[1] for v in valence_series], dtype=float)
            slope = np.polyfit(xs, ys, 1)[0]
        else:
            slope = 0.0

        # Compute arousal variance (proxy for volatility)
        if len(valence_series) >= 2:
            all_vals = []
            for a in acts:
                all_vals.extend([v[0] for v in acts[a]])
            val_variance = np.var(all_vals)
            all_aro = []
            for a in acts:
                all_aro.extend([v[1] for v in acts[a]])
            aro_variance = np.var(all_aro)
        else:
            val_variance = 0.0
            aro_variance = 0.0

        # Feature vector: per-act means + overall slope + variances
        feats = []
        for a in range(1, max_acts + 1):
            feats.extend(act_means[a])
        feats.extend([slope, val_variance, aro_variance])

        char_labels.append(key)
        feature_rows.append(feats)
        char_summaries[key] = {
            "slope": slope,
            "val_variance": val_variance,
            "aro_variance": aro_variance,
            "act_valences": {a: act_means[a][0] for a in range(1, max_acts + 1) if a in acts},
        }

    X = np.array(feature_rows)
    print(f"\nCharacters included: {len(char_labels)}")
    print(f"Feature dimensions: {X.shape[1]}")

    # Cluster
    dists = pdist(X, metric="euclidean")
    Z = linkage(dists, method="ward")

    # Cut into k=5 clusters
    k = 5
    cluster_assignments = fcluster(Z, t=k, criterion="maxclust")

    print(f"\n--- Ward Clustering (k={k}) ---")
    cluster_members = defaultdict(list)
    for i, cl in enumerate(cluster_assignments):
        cluster_members[cl].append(i)

    for cl in sorted(cluster_members.keys()):
        members = cluster_members[cl]
        print(f"\n  Cluster {cl}:")
        for idx in members:
            pid, char = char_labels[idx]
            s = char_summaries[(pid, char)]
            val_str = " -> ".join(f"{v:.2f}" for a, v in sorted(s["act_valences"].items()))
            print(f"    {PLAY_SHORT[pid]}/{char:<22s}  slope={s['slope']:+.3f}  var_v={s['val_variance']:.3f}  arc: [{val_str}]")

    # Check expected arc shapes
    print("\n--- Expected Arc Shape Validation ---")
    for arc_type, expected_chars in EXPECTED_ARCS.items():
        print(f"\n  {arc_type.upper()} arc:")
        for pid, char in expected_chars:
            key = (pid, char)
            if key in char_summaries:
                s = char_summaries[key]
                idx = char_labels.index(key)
                cl = cluster_assignments[idx]
                val_str = " -> ".join(f"{v:.2f}" for a, v in sorted(s["act_valences"].items()))
                print(f"    {PLAY_SHORT[pid]}/{char:<22s}  cluster={cl}  slope={s['slope']:+.3f}  var_v={s['val_variance']:.3f}  arc: [{val_str}]")
            else:
                print(f"    {PLAY_SHORT[pid]}/{char}: not found (insufficient data)")

    # Report whether expected arc groups co-cluster
    print("\n  Co-clustering check:")
    for arc_type, expected_chars in EXPECTED_ARCS.items():
        clusters_found = set()
        for pid, char in expected_chars:
            key = (pid, char)
            if key in char_summaries:
                idx = char_labels.index(key)
                clusters_found.add(cluster_assignments[idx])
        if len(clusters_found) == 1:
            print(f"    {arc_type}: ALL in cluster {clusters_found.pop()} (strong agreement)")
        else:
            print(f"    {arc_type}: split across clusters {clusters_found}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    plays = load_plays()
    tactic_list = load_tactic_vocab()
    print(f"Loaded {len(plays)} plays, {len(tactic_list)} canonical tactics")

    for pid, play in plays.items():
        n_beats = sum(
            1 for act in play["acts"]
            for scene in act["scenes"]
            for beat in scene["beats"]
        )
        n_bs = sum(
            len(beat["beat_states"])
            for act in play["acts"]
            for scene in act["scenes"]
            for beat in scene["beats"]
        )
        print(f"  {pid}: {len(play['characters'])} characters, {n_beats} beats, {n_bs} beat_states")

    run_experiment_1(plays, tactic_list)
    run_experiment_2(plays, tactic_list)
    run_experiment_3(plays, tactic_list)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
