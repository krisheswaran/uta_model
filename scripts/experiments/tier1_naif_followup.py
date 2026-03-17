#!/usr/bin/env python3
"""
Tier 1 Naif Follow-up: Affect-Space Clustering
================================================
Follow-up on the finding that romantic idealists/naifs (Anya, Cecily, Ophelia)
did NOT cluster in tactic space (z=-0.30). Hypothesis: they cluster in affect
space — their similarity is emotional/relational, not behavioral.

Experiments:
  1. Affect-space clustering (5D: valence, arousal, certainty, control, vulnerability)
  2. Combined affect+social clustering (7D: + status, warmth)
  3. Vulnerability/control profile check
  4. Affect vs tactic dissociation across all 4 expected clusters
  5. What tactics do the naifs actually use?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine, pdist, squareform

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

MIN_BEATS = 5

EXPECTED_CLUSTERS = {
    "Romantic idealists/naifs": {
        "chars": [("cherry_orchard", "ANYA"),
                  ("importance_of_being_earnest", "CECILY"),
                  ("hamlet", "OPHELIA")],
    },
    "Wit-driven deflectors": {
        "chars": [("importance_of_being_earnest", "ALGERNON"),
                  ("hamlet", "HAMLET"),
                  ("cherry_orchard", "TROFIMOV")],
    },
    "Servants": {
        "chars": [("importance_of_being_earnest", "LANE"),
                  ("importance_of_being_earnest", "MERRIMAN"),
                  ("cherry_orchard", "FIERS"),
                  ("cherry_orchard", "YASHA")],
    },
    "Social dominators": {
        "chars": [("importance_of_being_earnest", "LADY BRACKNELL"),
                  ("hamlet", "KING"),
                  ("hamlet", "POLONIUS")],
    },
}

NAIF_CHARS = EXPECTED_CLUSTERS["Romantic idealists/naifs"]["chars"]


# ---------------------------------------------------------------------------
# Data loading (reused from tier1_character_clustering.py)
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
    for act in play_data["acts"]:
        act_num = act["number"]
        for scene in act["scenes"]:
            for beat in scene["beats"]:
                for bs in beat["beat_states"]:
                    yield act_num, bs


def resolve_tactic(bs, tactic_set):
    """Get canonical tactic from a beat_state, falling back to tactic_state matching."""
    ct = bs.get("canonical_tactic")
    if ct and ct in tactic_set:
        return ct
    raw = bs.get("tactic_state", "").strip().upper()
    if raw in tactic_set:
        return raw
    return None


# ---------------------------------------------------------------------------
# Build character vectors
# ---------------------------------------------------------------------------

def build_character_data(plays, tactic_list):
    """
    For each character, compute:
      - mean affect vector (5D)
      - mean social vector (2D)
      - tactic distribution (normalized)
      - raw tactic counts
      - total beat count
    """
    tactic_idx = {t: i for i, t in enumerate(tactic_list)}
    tactic_set = set(tactic_list)
    n_tactics = len(tactic_list)

    char_affect_sums = defaultdict(lambda: np.zeros(5))
    char_social_sums = defaultdict(lambda: np.zeros(2))
    char_tactic_counts = defaultdict(lambda: np.zeros(n_tactics))
    char_n = defaultdict(int)

    for pid, play in plays.items():
        for act_num, bs in extract_beat_states(play):
            char = bs["character"]
            key = (pid, char)
            char_n[key] += 1

            aff = bs["affect_state"]
            char_affect_sums[key] += np.array([
                aff["valence"], aff["arousal"], aff["certainty"],
                aff["control"], aff["vulnerability"]
            ])
            soc = bs["social_state"]
            char_social_sums[key] += np.array([soc["status"], soc["warmth"]])

            tactic = resolve_tactic(bs, tactic_set)
            if tactic:
                char_tactic_counts[key][tactic_idx[tactic]] += 1

    # Build result dict for characters with enough data
    result = {}
    for key in sorted(char_n.keys()):
        if char_n[key] < MIN_BEATS:
            continue
        tc = char_tactic_counts[key]
        tc_total = tc.sum()
        if tc_total == 0:
            continue
        result[key] = {
            "affect_mean": char_affect_sums[key] / char_n[key],
            "social_mean": char_social_sums[key] / char_n[key],
            "tactic_dist": tc / tc_total,
            "tactic_counts": tc,
            "n_beats": char_n[key],
        }

    return result


def label_str(pid, char):
    return f"{PLAY_SHORT[pid]}/{char}"


# ---------------------------------------------------------------------------
# Cluster similarity helpers
# ---------------------------------------------------------------------------

def compute_cluster_similarity(char_keys, sim_matrix, all_keys):
    """Compute mean pairwise cosine similarity within a cluster."""
    indices = []
    for key in char_keys:
        if key in all_keys:
            indices.append(all_keys.index(key))
    if len(indices) < 2:
        return None, indices
    sims = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sims.append(sim_matrix[indices[i], indices[j]])
    return np.mean(sims), indices


def z_score_vs_baseline(within_mean, all_pairwise_sims):
    baseline_mean = np.mean(all_pairwise_sims)
    baseline_std = np.std(all_pairwise_sims)
    if baseline_std == 0:
        return 0.0, baseline_mean, baseline_std
    return (within_mean - baseline_mean) / baseline_std, baseline_mean, baseline_std


# ---------------------------------------------------------------------------
# Experiment 1: Affect-space clustering
# ---------------------------------------------------------------------------

def run_experiment_1(char_data):
    print("=" * 80)
    print("EXPERIMENT 1: Affect-Space Clustering (5D)")
    print("  Do naifs (Anya, Cecily, Ophelia) cluster by affect?")
    print("=" * 80)

    keys = list(char_data.keys())
    affect_matrix = np.array([char_data[k]["affect_mean"] for k in keys])

    # Pairwise cosine similarity
    cos_dists = pdist(affect_matrix, metric="cosine")
    cos_sim_matrix = 1 - squareform(cos_dists)
    all_pairwise_sims = 1 - cos_dists

    # Check naif cluster
    naif_sim, naif_indices = compute_cluster_similarity(NAIF_CHARS, cos_sim_matrix, keys)

    print(f"\nCharacters included: {len(keys)}")
    print(f"\n--- Naif Affect Vectors (5D) ---")
    dims = ["valence", "arousal", "certainty", "control", "vulnerability"]
    for nc in NAIF_CHARS:
        if nc in char_data:
            v = char_data[nc]["affect_mean"]
            dim_str = "  ".join(f"{d}={v[i]:+.3f}" for i, d in enumerate(dims))
            print(f"  {label_str(*nc):<20s}  {dim_str}")

    if naif_sim is not None:
        z, base_mean, base_std = z_score_vs_baseline(naif_sim, all_pairwise_sims)
        print(f"\n--- Naif Within-Cluster Cosine Similarity (Affect) ---")
        print(f"  Within-cluster mean sim: {naif_sim:.4f}")
        print(f"  Corpus baseline:         {base_mean:.4f} (std={base_std:.4f})")
        print(f"  Z-score:                 {z:+.2f}")
        print(f"  Verdict:                 {'CLUSTERS' if z > 1.0 else 'MARGINAL' if z > 0.5 else 'DOES NOT CLUSTER'}")

        # Show pairwise
        print(f"\n  Pairwise affect similarities:")
        for i in range(len(naif_indices)):
            for j in range(i + 1, len(naif_indices)):
                ki, kj = keys[naif_indices[i]], keys[naif_indices[j]]
                sim = cos_sim_matrix[naif_indices[i], naif_indices[j]]
                print(f"    {label_str(*ki):<18s} <-> {label_str(*kj):<18s}  sim={sim:.4f}")

    # Who are the naifs' nearest neighbors in affect space?
    print(f"\n--- Nearest Neighbors in Affect Space ---")
    for nc in NAIF_CHARS:
        if nc not in char_data:
            continue
        idx = keys.index(nc)
        sims = [(keys[j], cos_sim_matrix[idx, j]) for j in range(len(keys)) if j != idx]
        sims.sort(key=lambda x: -x[1])
        top5 = sims[:5]
        print(f"  {label_str(*nc)}:")
        for k, s in top5:
            print(f"    {label_str(*k):<25s}  sim={s:.4f}")

    return keys, cos_sim_matrix, all_pairwise_sims


# ---------------------------------------------------------------------------
# Experiment 2: Combined affect+social clustering (7D)
# ---------------------------------------------------------------------------

def run_experiment_2(char_data):
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Combined Affect+Social Clustering (7D)")
    print("  Does adding status/warmth improve naif clustering?")
    print("=" * 80)

    keys = list(char_data.keys())
    combined_matrix = np.array([
        np.concatenate([char_data[k]["affect_mean"], char_data[k]["social_mean"]])
        for k in keys
    ])

    cos_dists = pdist(combined_matrix, metric="cosine")
    cos_sim_matrix = 1 - squareform(cos_dists)
    all_pairwise_sims = 1 - cos_dists

    naif_sim, naif_indices = compute_cluster_similarity(NAIF_CHARS, cos_sim_matrix, keys)

    dims = ["valence", "arousal", "certainty", "control", "vulnerability", "status", "warmth"]
    print(f"\n--- Naif 7D Vectors ---")
    for nc in NAIF_CHARS:
        if nc in char_data:
            v = np.concatenate([char_data[nc]["affect_mean"], char_data[nc]["social_mean"]])
            dim_str = "  ".join(f"{d}={v[i]:+.3f}" for i, d in enumerate(dims))
            print(f"  {label_str(*nc):<20s}  {dim_str}")

    if naif_sim is not None:
        z, base_mean, base_std = z_score_vs_baseline(naif_sim, all_pairwise_sims)
        print(f"\n--- Naif Within-Cluster Cosine Similarity (Affect+Social) ---")
        print(f"  Within-cluster mean sim: {naif_sim:.4f}")
        print(f"  Corpus baseline:         {base_mean:.4f} (std={base_std:.4f})")
        print(f"  Z-score:                 {z:+.2f}")
        print(f"  Verdict:                 {'CLUSTERS' if z > 1.0 else 'MARGINAL' if z > 0.5 else 'DOES NOT CLUSTER'}")

        # Pairwise
        print(f"\n  Pairwise affect+social similarities:")
        for i in range(len(naif_indices)):
            for j in range(i + 1, len(naif_indices)):
                ki, kj = keys[naif_indices[i]], keys[naif_indices[j]]
                sim = cos_sim_matrix[naif_indices[i], naif_indices[j]]
                print(f"    {label_str(*ki):<18s} <-> {label_str(*kj):<18s}  sim={sim:.4f}")

    return keys, cos_sim_matrix, all_pairwise_sims


# ---------------------------------------------------------------------------
# Experiment 3: Vulnerability/control profile
# ---------------------------------------------------------------------------

def run_experiment_3(char_data):
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Vulnerability/Control/Status Profile")
    print("  Naifs should share: high vulnerability, low control, low status")
    print("=" * 80)

    keys = list(char_data.keys())
    dims = ["valence", "arousal", "certainty", "control", "vulnerability"]

    # Extract vulnerability, control, status for all characters
    all_vuln = [char_data[k]["affect_mean"][4] for k in keys]
    all_ctrl = [char_data[k]["affect_mean"][3] for k in keys]
    all_status = [char_data[k]["social_mean"][0] for k in keys]

    print(f"\n--- Corpus-Wide Distributions ---")
    print(f"  Vulnerability: mean={np.mean(all_vuln):.3f}, std={np.std(all_vuln):.3f}, range=[{np.min(all_vuln):.3f}, {np.max(all_vuln):.3f}]")
    print(f"  Control:       mean={np.mean(all_ctrl):.3f}, std={np.std(all_ctrl):.3f}, range=[{np.min(all_ctrl):.3f}, {np.max(all_ctrl):.3f}]")
    print(f"  Status:        mean={np.mean(all_status):.3f}, std={np.std(all_status):.3f}, range=[{np.min(all_status):.3f}, {np.max(all_status):.3f}]")

    print(f"\n--- Naif Profiles ---")
    print(f"  {'Character':<25s}  {'Vuln':>7s}  {'Control':>7s}  {'Status':>7s}  {'Warmth':>7s}  Vuln%ile  Ctrl%ile  Stat%ile")
    for nc in NAIF_CHARS:
        if nc not in char_data:
            continue
        v = char_data[nc]["affect_mean"]
        s = char_data[nc]["social_mean"]
        vuln = v[4]
        ctrl = v[3]
        status = s[0]
        warmth = s[1]
        vuln_pct = 100 * sum(1 for x in all_vuln if x <= vuln) / len(all_vuln)
        ctrl_pct = 100 * sum(1 for x in all_ctrl if x <= ctrl) / len(all_ctrl)
        stat_pct = 100 * sum(1 for x in all_status if x <= status) / len(all_status)
        print(f"  {label_str(*nc):<25s}  {vuln:+.3f}  {ctrl:+.3f}  {status:+.3f}  {warmth:+.3f}    {vuln_pct:5.1f}%    {ctrl_pct:5.1f}%    {stat_pct:5.1f}%")

    # Compare to other clusters
    print(f"\n--- Cluster Mean Profiles (for comparison) ---")
    print(f"  {'Cluster':<30s}  {'Vuln':>7s}  {'Control':>7s}  {'Status':>7s}  {'Warmth':>7s}")
    for cluster_name, info in EXPECTED_CLUSTERS.items():
        vulns, ctrls, statuses, warmths = [], [], [], []
        for c in info["chars"]:
            if c in char_data:
                vulns.append(char_data[c]["affect_mean"][4])
                ctrls.append(char_data[c]["affect_mean"][3])
                statuses.append(char_data[c]["social_mean"][0])
                warmths.append(char_data[c]["social_mean"][1])
        if vulns:
            print(f"  {cluster_name:<30s}  {np.mean(vulns):+.3f}  {np.mean(ctrls):+.3f}  {np.mean(statuses):+.3f}  {np.mean(warmths):+.3f}")


# ---------------------------------------------------------------------------
# Experiment 4: Affect vs Tactic dissociation
# ---------------------------------------------------------------------------

def run_experiment_4(char_data):
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Affect vs Tactic Dissociation")
    print("  Which clusters are tactic-driven vs affect-driven?")
    print("=" * 80)

    keys = list(char_data.keys())

    # Build affect similarity matrix
    affect_matrix = np.array([char_data[k]["affect_mean"] for k in keys])
    affect_dists = pdist(affect_matrix, metric="cosine")
    affect_sim_matrix = 1 - squareform(affect_dists)
    affect_all_sims = 1 - affect_dists

    # Build tactic similarity matrix
    tactic_matrix = np.array([char_data[k]["tactic_dist"] for k in keys])
    tactic_dists = pdist(tactic_matrix, metric="cosine")
    tactic_sim_matrix = 1 - squareform(tactic_dists)
    tactic_all_sims = 1 - tactic_dists

    # Combined affect+social
    combined_matrix = np.array([
        np.concatenate([char_data[k]["affect_mean"], char_data[k]["social_mean"]])
        for k in keys
    ])
    combined_dists = pdist(combined_matrix, metric="cosine")
    combined_sim_matrix = 1 - squareform(combined_dists)
    combined_all_sims = 1 - combined_dists

    print(f"\n{'Cluster':<30s}  {'Tactic z':>9s}  {'Affect z':>9s}  {'Aff+Soc z':>9s}  {'Driven by':>12s}")
    print("-" * 95)

    for cluster_name, info in EXPECTED_CLUSTERS.items():
        chars = info["chars"]

        # Tactic similarity
        t_sim, t_idx = compute_cluster_similarity(chars, tactic_sim_matrix, keys)
        if t_sim is not None:
            t_z, _, _ = z_score_vs_baseline(t_sim, tactic_all_sims)
        else:
            t_z = float("nan")

        # Affect similarity
        a_sim, a_idx = compute_cluster_similarity(chars, affect_sim_matrix, keys)
        if a_sim is not None:
            a_z, _, _ = z_score_vs_baseline(a_sim, affect_all_sims)
        else:
            a_z = float("nan")

        # Combined affect+social similarity
        c_sim, c_idx = compute_cluster_similarity(chars, combined_sim_matrix, keys)
        if c_sim is not None:
            c_z, _, _ = z_score_vs_baseline(c_sim, combined_all_sims)
        else:
            c_z = float("nan")

        # Determine what drives the cluster
        if np.isnan(t_z) or np.isnan(a_z):
            driven = "insufficient"
        elif t_z > 1.0 and a_z > 1.0:
            driven = "BOTH"
        elif t_z > a_z + 0.3:
            driven = "TACTIC"
        elif a_z > t_z + 0.3:
            driven = "AFFECT"
        elif max(t_z, a_z) < 0.5:
            driven = "NEITHER"
        else:
            driven = "mixed"

        print(f"  {cluster_name:<28s}  {t_z:+8.2f}   {a_z:+8.2f}   {c_z:+8.2f}   {driven:>12s}")

    # Detailed pairwise for naifs
    print(f"\n--- Naif Pairwise Detail ---")
    print(f"  {'Pair':<40s}  {'Tactic sim':>10s}  {'Affect sim':>10s}  {'Aff+Soc sim':>11s}")
    naif_indices = [keys.index(c) for c in NAIF_CHARS if c in char_data]
    for i in range(len(naif_indices)):
        for j in range(i + 1, len(naif_indices)):
            ki, kj = keys[naif_indices[i]], keys[naif_indices[j]]
            t_s = tactic_sim_matrix[naif_indices[i], naif_indices[j]]
            a_s = affect_sim_matrix[naif_indices[i], naif_indices[j]]
            c_s = combined_sim_matrix[naif_indices[i], naif_indices[j]]
            pair = f"{label_str(*ki)} <-> {label_str(*kj)}"
            print(f"  {pair:<40s}  {t_s:10.4f}  {a_s:10.4f}  {c_s:11.4f}")


# ---------------------------------------------------------------------------
# Experiment 5: What tactics do the naifs actually use?
# ---------------------------------------------------------------------------

def run_experiment_5(char_data, tactic_list):
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: What Tactics Do the Naifs Actually Use?")
    print("  Expected: AFFIRM, EMBRACE, PLEAD")
    print("=" * 80)

    expected_tactics = {"AFFIRM", "EMBRACE", "PLEAD"}

    for nc in NAIF_CHARS:
        if nc not in char_data:
            print(f"\n  {label_str(*nc)}: insufficient data")
            continue

        counts = char_data[nc]["tactic_counts"]
        n_total = int(counts.sum())

        # Top 10 tactics
        top_indices = np.argsort(counts)[::-1]
        top_tactics = [(tactic_list[i], int(counts[i]), counts[i] / n_total)
                       for i in top_indices if counts[i] > 0][:10]

        print(f"\n  {label_str(*nc)} (n={char_data[nc]['n_beats']} beats, {n_total} resolved tactics):")
        for rank, (t, c, pct) in enumerate(top_tactics, 1):
            marker = " <-- expected" if t in expected_tactics else ""
            print(f"    {rank:2d}. {t:<20s}  {c:3d}  ({pct:5.1%}){marker}")

        # Check expected tactics coverage
        expected_found = {t: int(counts[tactic_list.index(t)]) if t in tactic_list else 0
                          for t in expected_tactics}
        total_expected = sum(expected_found.values())
        print(f"    Expected tactics (AFFIRM/EMBRACE/PLEAD): {total_expected}/{n_total} = {total_expected/n_total:.1%}")

    # Cross-character tactic overlap
    print(f"\n--- Naif Tactic Overlap ---")
    naif_keys = [nc for nc in NAIF_CHARS if nc in char_data]
    if len(naif_keys) >= 2:
        # Find tactics used by all naifs
        tactic_sets = []
        for nc in naif_keys:
            counts = char_data[nc]["tactic_counts"]
            used = set(tactic_list[i] for i in range(len(tactic_list)) if counts[i] > 0)
            tactic_sets.append(used)

        shared = tactic_sets[0]
        for ts in tactic_sets[1:]:
            shared = shared & ts

        print(f"  Tactics used by ALL naifs ({len(shared)}):")
        # Sort by total count across naifs
        shared_with_counts = []
        for t in shared:
            total = int(sum(char_data[nc]["tactic_counts"][tactic_list.index(t)] for nc in naif_keys))
            shared_with_counts.append((t, total))
        shared_with_counts.sort(key=lambda x: -x[1])
        for t, total in shared_with_counts:
            per_char = "  ".join(
                f"{label_str(*nc)}={int(char_data[nc]['tactic_counts'][tactic_list.index(t)])}"
                for nc in naif_keys
            )
            print(f"    {t:<20s}  total={total:3d}  ({per_char})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    plays = load_plays()
    tactic_list = load_tactic_vocab()
    print(f"Loaded {len(plays)} plays, {len(tactic_list)} canonical tactics\n")

    char_data = build_character_data(plays, tactic_list)
    print(f"Characters with >={MIN_BEATS} beats and resolved tactics: {len(char_data)}")

    # Check naifs are present
    for nc in NAIF_CHARS:
        if nc in char_data:
            print(f"  {label_str(*nc)}: {char_data[nc]['n_beats']} beats")
        else:
            print(f"  {label_str(*nc)}: NOT FOUND (below threshold)")

    run_experiment_1(char_data)
    run_experiment_2(char_data)
    run_experiment_3(char_data)
    run_experiment_4(char_data)
    run_experiment_5(char_data, tactic_list)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    keys = list(char_data.keys())

    # Affect z-score for naifs
    affect_matrix = np.array([char_data[k]["affect_mean"] for k in keys])
    affect_dists = pdist(affect_matrix, metric="cosine")
    affect_sims = 1 - affect_dists
    affect_sim_matrix = 1 - squareform(affect_dists)
    naif_aff_sim, _ = compute_cluster_similarity(NAIF_CHARS, affect_sim_matrix, keys)
    if naif_aff_sim is not None:
        naif_aff_z, _, _ = z_score_vs_baseline(naif_aff_sim, affect_sims)
    else:
        naif_aff_z = float("nan")

    # Tactic z-score for naifs
    tactic_matrix = np.array([char_data[k]["tactic_dist"] for k in keys])
    tactic_dists = pdist(tactic_matrix, metric="cosine")
    tactic_sims = 1 - tactic_dists
    tactic_sim_matrix = 1 - squareform(tactic_dists)
    naif_tac_sim, _ = compute_cluster_similarity(NAIF_CHARS, tactic_sim_matrix, keys)
    if naif_tac_sim is not None:
        naif_tac_z, _, _ = z_score_vs_baseline(naif_tac_sim, tactic_sims)
    else:
        naif_tac_z = float("nan")

    # Combined z-score
    combined_matrix = np.array([
        np.concatenate([char_data[k]["affect_mean"], char_data[k]["social_mean"]])
        for k in keys
    ])
    combined_dists = pdist(combined_matrix, metric="cosine")
    combined_sims = 1 - combined_dists
    combined_sim_matrix = 1 - squareform(combined_dists)
    naif_comb_sim, _ = compute_cluster_similarity(NAIF_CHARS, combined_sim_matrix, keys)
    if naif_comb_sim is not None:
        naif_comb_z, _, _ = z_score_vs_baseline(naif_comb_sim, combined_sims)
    else:
        naif_comb_z = float("nan")

    print(f"\n  Naif clustering z-scores:")
    print(f"    Tactic space (from prior experiment):  z={naif_tac_z:+.2f}")
    print(f"    Affect space (5D):                     z={naif_aff_z:+.2f}")
    print(f"    Affect+Social space (7D):              z={naif_comb_z:+.2f}")

    if naif_aff_z > naif_tac_z + 0.5:
        print(f"\n  HYPOTHESIS SUPPORTED: Naifs cluster more strongly in affect space than tactic space.")
        print(f"  Dissociation: affect z - tactic z = {naif_aff_z - naif_tac_z:+.2f}")
    elif naif_aff_z > 1.0:
        print(f"\n  HYPOTHESIS PARTIALLY SUPPORTED: Naifs cluster in affect space (z={naif_aff_z:+.2f}),")
        print(f"  but the dissociation from tactic space is weak.")
    else:
        print(f"\n  HYPOTHESIS NOT SUPPORTED: Naifs do not cluster in affect space either.")
        print(f"  The 'naif' category may not be a natural cluster in this data.")

    print()


if __name__ == "__main__":
    main()
