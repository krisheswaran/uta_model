"""
Experiment: Does the superobjective predict anything useful?

Tests whether psi_arc is a real factor in the factor graph or just a label.
Five analyses:
  1. Superobjective embedding similarity vs tactic distribution similarity
  2. Superobjective embedding similarity vs affect profile similarity
  3. Superobjective clusters — do they make dramaturgical sense?
  4. Within-character superobjective consistency (bible vs beat-level reminders)
  5. Superobjective as tactic predictor — information gain over baseline entropy
"""

import json
import sys
import os
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine as cosine_dist, pdist, squareform

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "parsed"
VOCAB = ROOT / "data" / "vocab" / "tactic_vocabulary.json"
PLAYS = ["hamlet", "cherry_orchard", "importance_of_being_earnest"]

# ── Load data ────────────────────────────────────────────────────────────

def load_plays():
    plays = {}
    for pid in PLAYS:
        with open(DATA / f"{pid}.json") as f:
            plays[pid] = json.load(f)
    return plays

def load_tactic_vocab():
    with open(VOCAB) as f:
        vocab = json.load(f)
    # Build canonical_id list and member→canonical mapping
    canonical_ids = [t["canonical_id"] for t in vocab["tactics"]]
    member_to_canonical = {}
    for t in vocab["tactics"]:
        for m in t["members"]:
            member_to_canonical[m.upper()] = t["canonical_id"]
        member_to_canonical[t["canonical_id"]] = t["canonical_id"]
    return canonical_ids, member_to_canonical

def extract_characters(plays, canonical_ids, member_to_canonical):
    """
    Returns list of dicts with:
      character, play_id, superobjective, tactic_vec (66D), affect_vec (5D),
      beat_superobjective_reminders (list of str), beat_tactics (list of str)
    """
    tactic_index = {t: i for i, t in enumerate(canonical_ids)}
    characters = []

    for pid, play in plays.items():
        bibles = {b["character"]: b for b in play.get("character_bibles", [])}

        # Collect beat-level data per character
        char_beats = defaultdict(list)  # char -> list of beat_states
        for act in play["acts"]:
            for scene in act["scenes"]:
                for beat in scene["beats"]:
                    for bs in beat.get("beat_states", []):
                        char_beats[bs["character"]].append(bs)

        for char_name, bible in bibles.items():
            so = bible.get("superobjective")
            if not so:
                continue

            # Build tactic vector from character bible's tactic_distribution
            tactic_dist = bible.get("tactic_distribution", {})
            tactic_vec = np.zeros(len(canonical_ids))
            for tactic_str, count in tactic_dist.items():
                key = tactic_str.upper()
                cid = member_to_canonical.get(key)
                if cid and cid in tactic_index:
                    tactic_vec[tactic_index[cid]] += count
            # Normalize
            total = tactic_vec.sum()
            if total > 0:
                tactic_vec = tactic_vec / total

            # Build affect vector from beat states
            beats = char_beats.get(char_name, [])
            affect_dims = ["valence", "arousal", "certainty", "control", "vulnerability"]
            affect_vals = []
            so_reminders = []
            beat_tactics = []
            for bs in beats:
                aff = bs.get("affect_state", {})
                if aff and all(d in aff for d in affect_dims):
                    affect_vals.append([aff[d] for d in affect_dims])
                rem = bs.get("superobjective_reminder")
                if rem:
                    so_reminders.append(rem)
                # Resolve canonical tactic
                ct = bs.get("canonical_tactic")
                if not ct:
                    ts = bs.get("tactic_state", "")
                    if ts:
                        ct = member_to_canonical.get(ts.upper())
                if ct:
                    beat_tactics.append(ct)

            affect_vec = np.array(affect_vals).mean(axis=0) if affect_vals else np.zeros(5)

            characters.append({
                "character": char_name,
                "play_id": pid,
                "superobjective": so,
                "tactic_vec": tactic_vec,
                "affect_vec": affect_vec,
                "so_reminders": so_reminders,
                "beat_tactics": beat_tactics,
                "n_beats": len(beats),
            })

    return characters


# ── Test 1: Superobjective similarity vs tactic distribution similarity ──

def test_so_vs_tactic_similarity(characters, so_embeddings):
    """Mantel-style correlation between SO distance matrix and tactic distance matrix."""
    n = len(characters)
    if n < 3:
        return {"error": "Too few characters"}

    # Pairwise SO cosine distances
    so_dists = pdist(so_embeddings, metric="cosine")

    # Pairwise tactic cosine distances
    tactic_vecs = np.array([c["tactic_vec"] for c in characters])
    # Some characters may have all-zero tactic vecs; use small epsilon
    tactic_vecs_safe = tactic_vecs + 1e-10
    tactic_dists = pdist(tactic_vecs_safe, metric="cosine")

    # Remove NaN pairs
    mask = ~(np.isnan(so_dists) | np.isnan(tactic_dists))
    r, p = pearsonr(so_dists[mask], tactic_dists[mask])
    rho, p_s = spearmanr(so_dists[mask], tactic_dists[mask])

    return {
        "n_characters": n,
        "n_pairs": int(mask.sum()),
        "pearson_r": round(r, 4),
        "pearson_p": round(p, 6),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_s, 6),
        "interpretation": (
            "Positive r means similar superobjectives -> similar tactic distributions. "
            "p < 0.05 would indicate a real relationship."
        ),
    }


# ── Test 2: Superobjective similarity vs affect profile similarity ───────

def test_so_vs_affect_similarity(characters, so_embeddings):
    n = len(characters)
    if n < 3:
        return {"error": "Too few characters"}

    so_dists = pdist(so_embeddings, metric="cosine")
    affect_vecs = np.array([c["affect_vec"] for c in characters])
    affect_dists = pdist(affect_vecs, metric="cosine")

    mask = ~(np.isnan(so_dists) | np.isnan(affect_dists))
    r, p = pearsonr(so_dists[mask], affect_dists[mask])
    rho, p_s = spearmanr(so_dists[mask], affect_dists[mask])

    return {
        "n_characters": n,
        "n_pairs": int(mask.sum()),
        "pearson_r": round(r, 4),
        "pearson_p": round(p, 6),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_s, 6),
        "interpretation": (
            "Positive r means similar superobjectives -> similar affect profiles."
        ),
    }


# ── Test 3: Superobjective clusters ─────────────────────────────────────

def test_so_clusters(characters, so_embeddings):
    n = len(characters)
    if n < 6:
        return {"error": "Too few characters for clustering"}

    # Try k=3..8, pick best silhouette
    best_k, best_sil = 3, -1
    for k in range(3, min(9, n)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(so_embeddings)
        sil = silhouette_score(so_embeddings, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(so_embeddings)

    # Also cluster by tactic vectors for comparison
    tactic_vecs = np.array([c["tactic_vec"] for c in characters]) + 1e-10
    km_t = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    tactic_labels = km_t.fit_predict(tactic_vecs)

    # Report clusters
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        c = characters[i]
        clusters[int(lab)].append(f"{c['character']} ({c['play_id']})")

    tactic_clusters = defaultdict(list)
    for i, lab in enumerate(tactic_labels):
        c = characters[i]
        tactic_clusters[int(lab)].append(f"{c['character']} ({c['play_id']})")

    # Agreement between SO clusters and tactic clusters (adjusted Rand)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels, tactic_labels)

    return {
        "best_k": best_k,
        "silhouette_score": round(best_sil, 4),
        "so_clusters": dict(clusters),
        "tactic_clusters": dict(tactic_clusters),
        "adjusted_rand_index_so_vs_tactic": round(ari, 4),
        "interpretation": (
            "ARI near 0 = SO clusters differ from tactic clusters (SO captures something different). "
            "ARI near 1 = they agree (SO is redundant with tactics)."
        ),
    }


# ── Test 4: Within-character superobjective consistency ──────────────────

def test_so_consistency(characters, model):
    """Compare global superobjective to beat-level superobjective_reminders."""
    results = []
    for c in characters:
        if not c["so_reminders"]:
            continue
        global_emb = model.encode([c["superobjective"]])[0]
        reminder_embs = model.encode(c["so_reminders"])
        sims = [1 - cosine_dist(global_emb, r) for r in reminder_embs]
        results.append({
            "character": c["character"],
            "play_id": c["play_id"],
            "n_beats": len(c["so_reminders"]),
            "mean_similarity": round(float(np.mean(sims)), 4),
            "std_similarity": round(float(np.std(sims)), 4),
            "min_similarity": round(float(np.min(sims)), 4),
            "max_similarity": round(float(np.max(sims)), 4),
        })

    if not results:
        return {"error": "No characters with SO reminders"}

    all_means = [r["mean_similarity"] for r in results]
    # Sort by mean similarity ascending to highlight outliers
    results.sort(key=lambda r: r["mean_similarity"])

    return {
        "n_characters_with_reminders": len(results),
        "global_mean_similarity": round(float(np.mean(all_means)), 4),
        "global_std": round(float(np.std(all_means)), 4),
        "global_min": round(float(np.min(all_means)), 4),
        "global_max": round(float(np.max(all_means)), 4),
        "least_consistent_5": results[:5],
        "most_consistent_5": results[-5:],
        "interpretation": (
            "Similarity > 0.7 means SO reminders track the global SO well (stable prior). "
            "< 0.5 means the SO drifts across the play (dynamic, not a fixed prior)."
        ),
    }


# ── Test 5: SO as tactic predictor — information gain ────────────────────

def test_so_information_gain(characters, so_embeddings, canonical_ids):
    """
    Measure how much knowing the SO cluster reduces tactic entropy.
    H(tactic) - H(tactic | SO_cluster) = information gain.
    """
    n = len(characters)
    if n < 6:
        return {"error": "Too few characters"}

    # Assign SO clusters (use k from test 3, or default k=5)
    k = min(5, n - 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(so_embeddings)

    tactic_index = {t: i for i, t in enumerate(canonical_ids)}

    def entropy(counts):
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in counts if c > 0])
        return -np.sum(probs * np.log2(probs))

    # Global tactic distribution (pool all characters' beat tactics)
    global_counts = Counter()
    for c in characters:
        global_counts.update(c["beat_tactics"])
    H_global = entropy(list(global_counts.values()))

    # Conditional entropy: H(tactic | cluster) = sum_k P(cluster=k) * H(tactic | cluster=k)
    cluster_chars = defaultdict(list)
    for i, lab in enumerate(cluster_labels):
        cluster_chars[lab].append(characters[i])

    total_beats = sum(len(c["beat_tactics"]) for c in characters)
    H_conditional = 0.0
    cluster_details = {}
    for lab, chars in cluster_chars.items():
        cluster_counts = Counter()
        for c in chars:
            cluster_counts.update(c["beat_tactics"])
        cluster_beats = sum(len(c["beat_tactics"]) for c in chars)
        weight = cluster_beats / total_beats if total_beats > 0 else 0
        h = entropy(list(cluster_counts.values()))
        H_conditional += weight * h
        top_tactics = cluster_counts.most_common(5)
        cluster_details[int(lab)] = {
            "n_characters": len(chars),
            "characters": [c["character"] for c in chars],
            "n_beats": cluster_beats,
            "entropy": round(h, 4),
            "top_tactics": top_tactics,
        }

    info_gain = H_global - H_conditional
    # Normalized information gain (proportion of entropy explained)
    nig = info_gain / H_global if H_global > 0 else 0

    # Per-character baseline entropy vs cluster-conditional
    char_entropies = []
    for c in characters:
        if c["beat_tactics"]:
            counts = Counter(c["beat_tactics"])
            char_entropies.append(entropy(list(counts.values())))

    return {
        "n_clusters": k,
        "total_beats": total_beats,
        "H_tactic_global": round(H_global, 4),
        "H_tactic_given_cluster": round(H_conditional, 4),
        "information_gain_bits": round(info_gain, 4),
        "normalized_info_gain": round(nig, 4),
        "mean_per_character_entropy": round(float(np.mean(char_entropies)), 4) if char_entropies else None,
        "cluster_details": cluster_details,
        "interpretation": (
            "NIG > 0.05 means SO clusters meaningfully constrain tactic choice. "
            "NIG > 0.10 is a strong signal that psi_arc is a real factor."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EXPERIMENT: Does the superobjective predict behavior?")
    print("=" * 72)

    # Load
    print("\nLoading data...")
    plays = load_plays()
    canonical_ids, member_to_canonical = load_tactic_vocab()
    characters = extract_characters(plays, canonical_ids, member_to_canonical)
    print(f"  Loaded {len(characters)} characters with superobjectives across {len(PLAYS)} plays")

    # Embed superobjectives
    print("\nEmbedding superobjectives with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    so_texts = [c["superobjective"] for c in characters]
    so_embeddings = model.encode(so_texts, show_progress_bar=False)
    print(f"  Embedded {len(so_texts)} superobjectives -> shape {so_embeddings.shape}")

    # ── Test 1 ──
    print("\n" + "─" * 72)
    print("TEST 1: Superobjective similarity vs tactic distribution similarity")
    print("─" * 72)
    r1 = test_so_vs_tactic_similarity(characters, so_embeddings)
    for k, v in r1.items():
        print(f"  {k}: {v}")

    # ── Test 2 ──
    print("\n" + "─" * 72)
    print("TEST 2: Superobjective similarity vs affect profile similarity")
    print("─" * 72)
    r2 = test_so_vs_affect_similarity(characters, so_embeddings)
    for k, v in r2.items():
        print(f"  {k}: {v}")

    # ── Test 3 ──
    print("\n" + "─" * 72)
    print("TEST 3: Superobjective clusters vs tactic clusters")
    print("─" * 72)
    r3 = test_so_clusters(characters, so_embeddings)
    for k, v in r3.items():
        if k in ("so_clusters", "tactic_clusters"):
            print(f"  {k}:")
            for cluster_id, members in v.items():
                print(f"    Cluster {cluster_id}: {', '.join(members)}")
        else:
            print(f"  {k}: {v}")

    # ── Test 4 ──
    print("\n" + "─" * 72)
    print("TEST 4: Within-character superobjective consistency")
    print("─" * 72)
    r4 = test_so_consistency(characters, model)
    for k, v in r4.items():
        if k in ("least_consistent_5", "most_consistent_5"):
            print(f"  {k}:")
            for entry in v:
                print(f"    {entry['character']} ({entry['play_id']}): "
                      f"mean={entry['mean_similarity']:.3f}, "
                      f"std={entry['std_similarity']:.3f}, "
                      f"range=[{entry['min_similarity']:.3f}, {entry['max_similarity']:.3f}], "
                      f"n={entry['n_beats']}")
        else:
            print(f"  {k}: {v}")

    # ── Test 5 ──
    print("\n" + "─" * 72)
    print("TEST 5: Superobjective as tactic predictor (information gain)")
    print("─" * 72)
    r5 = test_so_information_gain(characters, so_embeddings, canonical_ids)
    for k, v in r5.items():
        if k == "cluster_details":
            print(f"  {k}:")
            for cid, detail in v.items():
                print(f"    Cluster {cid}: {detail['n_characters']} chars, "
                      f"{detail['n_beats']} beats, H={detail['entropy']:.3f}, "
                      f"top={detail['top_tactics'][:3]}")
                print(f"      members: {', '.join(detail['characters'])}")
        else:
            print(f"  {k}: {v}")

    # ── Verdict ──
    print("\n" + "=" * 72)
    print("VERDICT: Is psi_arc a real factor?")
    print("=" * 72)

    signals = []
    # Check tactic correlation
    if "pearson_r" in r1 and r1["pearson_p"] < 0.05:
        signals.append(f"Tactic correlation: r={r1['pearson_r']}, p={r1['pearson_p']} (significant)")
    else:
        signals.append(f"Tactic correlation: r={r1.get('pearson_r','?')}, p={r1.get('pearson_p','?')} (NOT significant)")

    # Check affect correlation
    if "pearson_r" in r2 and r2["pearson_p"] < 0.05:
        signals.append(f"Affect correlation: r={r2['pearson_r']}, p={r2['pearson_p']} (significant)")
    else:
        signals.append(f"Affect correlation: r={r2.get('pearson_r','?')}, p={r2.get('pearson_p','?')} (NOT significant)")

    # Check cluster divergence
    if "adjusted_rand_index_so_vs_tactic" in r3:
        ari = r3["adjusted_rand_index_so_vs_tactic"]
        signals.append(f"SO vs tactic cluster ARI: {ari} ({'redundant' if ari > 0.3 else 'different grouping'})")

    # Check consistency
    if "global_mean_similarity" in r4:
        sim = r4["global_mean_similarity"]
        signals.append(f"SO consistency: mean cosine sim={sim} ({'stable prior' if sim > 0.6 else 'dynamic/drifting'})")

    # Check info gain
    if "normalized_info_gain" in r5:
        nig = r5["normalized_info_gain"]
        if nig > 0.10:
            signals.append(f"Info gain: NIG={nig} (STRONG — SO constrains tactics)")
        elif nig > 0.05:
            signals.append(f"Info gain: NIG={nig} (moderate — SO partially constrains tactics)")
        else:
            signals.append(f"Info gain: NIG={nig} (weak — SO barely constrains tactics)")

    for s in signals:
        print(f"  * {s}")

    # Overall
    n_positive = sum(1 for s in signals if "significant" in s or "STRONG" in s or "stable" in s or "different" in s)
    if n_positive >= 3:
        print("\n  >> psi_arc is a REAL factor: superobjective carries predictive information")
    elif n_positive >= 2:
        print("\n  >> psi_arc is a WEAK factor: superobjective has some predictive value")
    else:
        print("\n  >> psi_arc may be JUST A LABEL: limited predictive value found")


if __name__ == "__main__":
    main()
