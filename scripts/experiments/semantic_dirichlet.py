#!/usr/bin/env python3
"""
Experiment: Uniform vs Semantically-Informed Dirichlet Smoothing
for the Tactic Transition Matrix.

Hypothesis: Uniform Dirichlet smoothing produces near-uniform rows (43/66),
causing the factor graph smoother to override 94.8% of LLM tactic assignments.
Semantically-informed smoothing (weighting alpha by embedding distance) should
produce more peaked rows that respect LLM observations.

Usage:
    conda run -n uta_model python scripts/experiments/semantic_dirichlet.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import FACTORS_DIR, PARSED_DIR, VOCAB_DIR
from factor_graph.learning import (
    FactorLearner,
    _load_tactic_vocab,
    _resolve_tactic,
)


# ---------------------------------------------------------------------------
# Part 1: Load data and compute transition counts
# ---------------------------------------------------------------------------

def load_transition_counts() -> tuple[np.ndarray, list[str], dict[str, str], set[str]]:
    """Load raw transition counts from all parsed plays.
    Returns (counts_matrix, canonical_ids, member_to_canonical, canonical_set).
    """
    canonical_ids, member_to_canonical = _load_tactic_vocab()
    canonical_set = set(canonical_ids)
    N = len(canonical_ids)
    tactic_to_idx = {t: i for i, t in enumerate(canonical_ids)}

    counts = np.zeros((N, N), dtype=float)
    n_transitions = 0
    n_skipped = 0

    # Observed transition list for leave-one-out
    observed_transitions = []

    plays_dir = PARSED_DIR
    play_files = list(plays_dir.glob("*.json"))
    print(f"Found {len(play_files)} parsed plays")

    for play_path in play_files:
        with open(play_path) as f:
            play = json.load(f)
        play_id = play["id"]

        # Extract character beat sequences
        for act in play["acts"]:
            for scene in act["scenes"]:
                char_beats: dict[str, list[tuple[int, dict]]] = defaultdict(list)
                for beat in scene["beats"]:
                    for bs in beat["beat_states"]:
                        resolved = _resolve_tactic(bs, member_to_canonical, canonical_set)
                        bs_copy = dict(bs, _resolved_tactic=resolved)
                        char_beats[bs["character"]].append((beat["index"], bs_copy))

                for char, indexed_states in char_beats.items():
                    indexed_states.sort(key=lambda x: x[0])
                    states = [s for _, s in indexed_states]
                    for i in range(len(states) - 1):
                        t_prev = states[i]["_resolved_tactic"]
                        t_curr = states[i + 1]["_resolved_tactic"]
                        if t_prev is None or t_curr is None:
                            n_skipped += 1
                            continue
                        if t_prev not in tactic_to_idx or t_curr not in tactic_to_idx:
                            n_skipped += 1
                            continue
                        idx_prev = tactic_to_idx[t_prev]
                        idx_curr = tactic_to_idx[t_curr]
                        counts[idx_prev, idx_curr] += 1
                        n_transitions += 1
                        observed_transitions.append((idx_prev, idx_curr))

    print(f"Transitions: {n_transitions}, skipped: {n_skipped}")
    print(f"Rows with zero counts: {np.sum(counts.sum(axis=1) == 0)}")
    print(f"Rows with >= 1 count: {np.sum(counts.sum(axis=1) >= 1)}")

    return counts, canonical_ids, observed_transitions, tactic_to_idx


def compute_tactic_embeddings(canonical_ids: list[str]) -> np.ndarray:
    """Embed tactic descriptions using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    vocab_path = VOCAB_DIR / "tactic_vocabulary.json"
    with open(vocab_path) as f:
        vocab = json.load(f)

    # Build id -> description mapping
    id_to_desc = {}
    for entry in vocab["tactics"]:
        id_to_desc[entry["canonical_id"]] = entry["description"]

    # Embed descriptions
    texts = [id_to_desc.get(tid, tid.lower()) for tid in canonical_ids]
    print(f"Embedding {len(texts)} tactic descriptions...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    N = embeddings.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = cosine_dist(embeddings[i], embeddings[j])
    return D


def row_entropy(P: np.ndarray) -> np.ndarray:
    """Compute per-row entropy (in bits) of a probability matrix."""
    N = P.shape[0]
    ent = np.zeros(N)
    for i in range(N):
        probs = P[i][P[i] > 0]
        if len(probs) > 0:
            ent[i] = -np.sum(probs * np.log2(probs))
    return ent


def build_uniform_dirichlet(counts: np.ndarray, alpha_base: float = 0.1) -> np.ndarray:
    """Build transition matrix with entropy-adaptive uniform Dirichlet smoothing
    (current method from learning.py)."""
    N = counts.shape[0]

    # Compute per-row entropy of raw counts
    row_entropies = np.zeros(N)
    for i in range(N):
        row_sum = counts[i].sum()
        if row_sum < 1:
            continue
        probs = counts[i] / row_sum
        probs = probs[probs > 0]
        row_entropies[i] = -np.sum(probs * np.log2(probs))

    max_entropy = row_entropies.max() if row_entropies.max() > 0 else 1.0

    transition_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        alpha_i = alpha_base * (row_entropies[i] / max_entropy) if max_entropy > 0 else alpha_base
        alpha_i = max(alpha_i, 1e-6)
        smoothed = counts[i] + alpha_i
        transition_matrix[i] = smoothed / smoothed.sum()

    return transition_matrix


def build_semantic_dirichlet(
    counts: np.ndarray,
    distance_matrix: np.ndarray,
    alpha_base: float = 0.1,
    tau: float = 0.3,
) -> np.ndarray:
    """Build transition matrix with semantically-informed Dirichlet smoothing.

    alpha_ij = alpha_base * exp(-D[i,j] / tau)

    Semantically similar tactics get higher pseudo-counts, dissimilar ones get less.
    """
    N = counts.shape[0]

    # Compute per-row entropy of raw counts (same adaptive scaling)
    raw_entropies = np.zeros(N)
    for i in range(N):
        row_sum = counts[i].sum()
        if row_sum < 1:
            continue
        probs = counts[i] / row_sum
        probs = probs[probs > 0]
        raw_entropies[i] = -np.sum(probs * np.log2(probs))

    max_entropy = raw_entropies.max() if raw_entropies.max() > 0 else 1.0

    # Semantic alpha matrix: alpha_ij = alpha_base * scale_i * exp(-D[i,j] / tau)
    alpha_matrix = alpha_base * np.exp(-distance_matrix / tau)

    transition_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        # Entropy-adaptive scaling (same as uniform)
        scale_i = (raw_entropies[i] / max_entropy) if max_entropy > 0 else 1.0
        scale_i = max(scale_i, 1e-6)
        alpha_row = alpha_matrix[i] * scale_i
        smoothed = counts[i] + alpha_row
        transition_matrix[i] = smoothed / smoothed.sum()

    return transition_matrix


def count_near_uniform_rows(P: np.ndarray, threshold: float = 0.95) -> int:
    """Count rows that are near-uniform (max prob < 1/N * (1 + threshold_ratio))."""
    N = P.shape[1]
    max_probs = P.max(axis=1)
    uniform_prob = 1.0 / N
    # A row is "near-uniform" if its max prob is close to uniform
    # Use entropy criterion: entropy > 95% of max entropy
    max_entropy = np.log2(N)
    ent = row_entropy(P)
    return int(np.sum(ent > threshold * max_entropy))


def leave_one_out_test(
    counts: np.ndarray,
    distance_matrix: np.ndarray,
    observed_transitions: list[tuple[int, int]],
    tau_values: list[float],
) -> dict:
    """Leave-one-out prediction test.

    For each observed transition, hold it out, rebuild the matrix,
    and check P(T_t | T_{t-1}).
    """
    N = counts.shape[0]
    n_sample = min(len(observed_transitions), 500)  # subsample for speed

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(observed_transitions), size=n_sample, replace=False)
    sample_transitions = [observed_transitions[i] for i in sample_indices]

    results = {}

    # Uniform baseline
    print(f"\n  LOO: Uniform Dirichlet ({n_sample} samples)...")
    log_likelihoods_uniform = []
    for idx_prev, idx_curr in sample_transitions:
        loo_counts = counts.copy()
        loo_counts[idx_prev, idx_curr] -= 1
        P = build_uniform_dirichlet(loo_counts)
        ll = np.log(max(P[idx_prev, idx_curr], 1e-300))
        log_likelihoods_uniform.append(ll)
    results["uniform"] = {
        "mean_ll": float(np.mean(log_likelihoods_uniform)),
        "std_ll": float(np.std(log_likelihoods_uniform)),
    }
    print(f"    Mean log-likelihood: {results['uniform']['mean_ll']:.4f}")

    # Semantic for each tau
    for tau in tau_values:
        print(f"  LOO: Semantic tau={tau} ({n_sample} samples)...")
        log_likelihoods = []
        for idx_prev, idx_curr in sample_transitions:
            loo_counts = counts.copy()
            loo_counts[idx_prev, idx_curr] -= 1
            P = build_semantic_dirichlet(loo_counts, distance_matrix, tau=tau)
            ll = np.log(max(P[idx_prev, idx_curr], 1e-300))
            log_likelihoods.append(ll)
        results[f"semantic_tau={tau}"] = {
            "mean_ll": float(np.mean(log_likelihoods)),
            "std_ll": float(np.std(log_likelihoods)),
        }
        print(f"    Mean log-likelihood: {results[f'semantic_tau={tau}']['mean_ll']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("EXPERIMENT: Uniform vs Semantic Dirichlet Smoothing")
    print("=" * 72)

    t0 = time.time()

    # 1. Load transition counts
    print("\n--- 1. Loading transition counts ---")
    counts, canonical_ids, observed_transitions, tactic_to_idx = load_transition_counts()
    N = len(canonical_ids)
    print(f"  Vocabulary size: {N}")
    print(f"  Total observed transitions: {len(observed_transitions)}")

    # 2. Compute tactic embeddings
    print("\n--- 2. Computing tactic embeddings ---")
    embeddings = compute_tactic_embeddings(canonical_ids)
    print(f"  Embedding shape: {embeddings.shape}")

    # 3. Compute distance matrix
    print("\n--- 3. Computing pairwise cosine distance ---")
    D = compute_distance_matrix(embeddings)
    print(f"  Distance matrix shape: {D.shape}")
    print(f"  Mean distance: {D[np.triu_indices(N, k=1)].mean():.4f}")
    print(f"  Min non-zero distance: {D[D > 0].min():.4f}")
    print(f"  Max distance: {D.max():.4f}")

    # 4. Build uniform Dirichlet matrix
    print("\n--- 4. Building Uniform Dirichlet matrix ---")
    P_uniform = build_uniform_dirichlet(counts)
    ent_uniform = row_entropy(P_uniform)
    max_ent = np.log2(N)
    near_uniform_count = count_near_uniform_rows(P_uniform)

    print(f"  Mean row entropy: {ent_uniform.mean():.4f} bits (max={max_ent:.2f})")
    print(f"  Median row entropy: {np.median(ent_uniform):.4f} bits")
    print(f"  Near-uniform rows (>95% max entropy): {near_uniform_count}/{N}")
    print(f"  Mean max prob: {P_uniform.max(axis=1).mean():.6f}")
    print(f"  Mean self-transition: {np.diag(P_uniform).mean():.6f}")

    # 5. Build semantic Dirichlet matrices for various tau
    tau_values = [0.1, 0.2, 0.3, 0.5, 0.7]
    print("\n--- 5. Building Semantic Dirichlet matrices ---")
    print(f"  Testing tau values: {tau_values}")

    semantic_results = {}
    for tau in tau_values:
        P_sem = build_semantic_dirichlet(counts, D, tau=tau)
        ent_sem = row_entropy(P_sem)
        near_uni = count_near_uniform_rows(P_sem)

        # Probability mass on semantically similar tactics (top-10 nearest by embedding)
        # for rows with zero observed counts
        zero_rows = np.where(counts.sum(axis=1) == 0)[0]
        sim_mass_zero = []
        for i in zero_rows:
            top10 = np.argsort(D[i])[:10]  # 10 nearest (includes self)
            sim_mass_zero.append(P_sem[i, top10].sum())

        # For rows WITH observed counts
        nonzero_rows = np.where(counts.sum(axis=1) > 0)[0]
        sim_mass_nonzero = []
        for i in nonzero_rows:
            top10 = np.argsort(D[i])[:10]
            sim_mass_nonzero.append(P_sem[i, top10].sum())

        semantic_results[tau] = {
            "mean_entropy": float(ent_sem.mean()),
            "median_entropy": float(np.median(ent_sem)),
            "near_uniform_rows": near_uni,
            "mean_max_prob": float(P_sem.max(axis=1).mean()),
            "mean_self_transition": float(np.diag(P_sem).mean()),
            "mean_sim_mass_zero": float(np.mean(sim_mass_zero)) if sim_mass_zero else 0.0,
            "mean_sim_mass_nonzero": float(np.mean(sim_mass_nonzero)) if sim_mass_nonzero else 0.0,
        }

        print(f"\n  tau={tau}:")
        print(f"    Mean row entropy: {ent_sem.mean():.4f} bits "
              f"(delta from uniform: {ent_sem.mean() - ent_uniform.mean():+.4f})")
        print(f"    Near-uniform rows: {near_uni}/{N}")
        print(f"    Mean max prob: {P_sem.max(axis=1).mean():.6f}")
        print(f"    Mean self-transition: {np.diag(P_sem).mean():.6f}")
        if sim_mass_zero:
            print(f"    Sim mass on top-10 nearest (zero-count rows): {np.mean(sim_mass_zero):.4f}")
        if sim_mass_nonzero:
            print(f"    Sim mass on top-10 nearest (nonzero rows): {np.mean(sim_mass_nonzero):.4f}")

    # 6. Leave-one-out test
    print("\n--- 6. Leave-one-out prediction test ---")
    loo_results = leave_one_out_test(counts, D, observed_transitions, tau_values)

    # 7. Summary comparison
    print("\n" + "=" * 72)
    print("SUMMARY: COMPARISON TABLE")
    print("=" * 72)
    print(f"{'Method':<25} {'Mean Entropy':>12} {'Near-Uniform':>14} "
          f"{'Mean Max P':>12} {'LOO Mean LL':>12}")
    print("-" * 75)
    print(f"{'Uniform Dirichlet':<25} {ent_uniform.mean():>12.4f} "
          f"{near_uniform_count:>14d}/{N} "
          f"{P_uniform.max(axis=1).mean():>12.6f} "
          f"{loo_results['uniform']['mean_ll']:>12.4f}")

    best_tau = None
    best_ll = -float('inf')
    for tau in tau_values:
        sr = semantic_results[tau]
        ll = loo_results[f"semantic_tau={tau}"]["mean_ll"]
        label = f"Semantic tau={tau}"
        print(f"{label:<25} {sr['mean_entropy']:>12.4f} "
              f"{sr['near_uniform_rows']:>14d}/{N} "
              f"{sr['mean_max_prob']:>12.6f} "
              f"{ll:>12.4f}")
        if ll > best_ll:
            best_ll = ll
            best_tau = tau

    print(f"\nBest tau by LOO log-likelihood: {best_tau} (LL={best_ll:.4f})")
    uniform_ll = loo_results["uniform"]["mean_ll"]
    print(f"Improvement over uniform: {best_ll - uniform_ll:+.4f} nats")

    # Entropy reduction
    best_entropy = semantic_results[best_tau]["mean_entropy"]
    print(f"Entropy reduction: {ent_uniform.mean() - best_entropy:.4f} bits "
          f"({(ent_uniform.mean() - best_entropy) / ent_uniform.mean() * 100:.1f}%)")

    # Near-uniform reduction
    best_near_uni = semantic_results[best_tau]["near_uniform_rows"]
    print(f"Near-uniform rows: {near_uniform_count} -> {best_near_uni} "
          f"(reduction of {near_uniform_count - best_near_uni})")

    elapsed = time.time() - t0
    print(f"\nExperiment completed in {elapsed:.1f}s")

    # Return best tau for Part 2
    return best_tau, best_ll


if __name__ == "__main__":
    best_tau, best_ll = main()
    print(f"\n>>> BEST TAU: {best_tau}")
