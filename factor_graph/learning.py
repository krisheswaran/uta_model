#!/usr/bin/env python3
"""
Factor Graph Parameter Learning — Pass 1.5a

Reads Pass 1 outputs (parsed plays with BeatStates and CharacterBibles) and
computes all factor potentials, saving them to data/factors/.

Usage:
    python -m factor_graph.learning --plays cherry_orchard hamlet importance_of_being_earnest
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score

from config import FACTORS_DIR, PARSED_DIR, VOCAB_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]

UTT_FEATURES = [
    "word_count", "question_density", "exclamation_density",
    "imperative_density", "mean_sentence_length", "lexical_diversity",
    "first_person_rate", "second_person_rate", "sentiment_polarity",
]

IMPERATIVE_STARTERS = {
    w.lower()
    for w in [
        "Go", "Come", "Tell", "Let", "Give", "Stop", "Look", "Listen",
        "Wait", "Leave", "Stand", "Sit", "Take", "Bring", "Send", "Run",
        "Stay", "Hold", "Speak", "Say", "See", "Hear", "Think", "Try",
        "Help", "Keep", "Put", "Show", "Watch", "Mark", "Note", "Pray",
        "Hark", "Fear", "Know", "Believe", "Remember", "Forgive", "Answer",
        "Rise", "Enter", "Follow", "Read", "Behold", "Haste", "Bid",
        "Bear", "Yield", "Swear", "Confess", "Return", "Obey", "Away",
        "Hush", "Pardon", "Retire", "Withdraw", "Proceed", "Consider",
        "Imagine", "Suppose", "Do", "Be", "Have", "Make", "Get", "Set",
    ]
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}


# ---------------------------------------------------------------------------
# Tactic vocabulary helpers
# ---------------------------------------------------------------------------

def _load_tactic_vocab() -> tuple[list[str], dict[str, str]]:
    """Return (list of canonical_ids sorted, member->canonical_id mapping)."""
    vocab_path = VOCAB_DIR / "tactic_vocabulary.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    canonical_ids: list[str] = sorted(e["canonical_id"] for e in vocab["tactics"])
    member_to_canonical: dict[str, str] = {}
    canonical_set = set(canonical_ids)
    for entry in vocab["tactics"]:
        cid = entry["canonical_id"]
        for m in entry["members"]:
            member_to_canonical[m.lower().strip()] = cid
    return canonical_ids, member_to_canonical


def _resolve_tactic(
    bs: dict,
    member_to_canonical: dict[str, str],
    canonical_set: set[str],
) -> Optional[str]:
    """Return canonical tactic ID, falling back to tactic_state lookup."""
    ct = bs.get("canonical_tactic")
    if ct and ct in canonical_set:
        return ct
    ts = bs.get("tactic_state")
    if ts:
        key = ts.lower().strip()
        upper = key.upper()
        if upper in canonical_set:
            return upper
        return member_to_canonical.get(key)
    return None


# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]


def _get_vader():
    """Lazy-init VADER analyzer."""
    if not hasattr(_get_vader, "_instance"):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _get_vader._instance = SentimentIntensityAnalyzer()
    return _get_vader._instance


def compute_utterance_features(texts: list[str]) -> Optional[dict[str, float]]:
    """Compute 9 text features from a list of utterance texts for one character in one beat."""
    all_text = " ".join(texts)
    words = re.findall(r"[A-Za-z']+", all_text)
    word_count = len(words)
    if word_count == 0:
        return None

    sentences: list[str] = []
    for t in texts:
        sentences.extend(_split_sentences(t))
    n_sent = max(len(sentences), 1)

    q_count = sum(1 for s in sentences if s.rstrip().endswith("?"))
    exc_count = sum(1 for s in sentences if s.rstrip().endswith("!"))

    imp_count = 0
    for s in sentences:
        first_word = re.match(r"[A-Za-z']+", s.strip())
        if first_word and first_word.group().lower() in IMPERATIVE_STARTERS:
            imp_count += 1

    sent_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
    mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0

    words_lower = [w.lower() for w in words]
    lex_div = len(set(words_lower)) / len(words_lower)

    fp_count = sum(1 for w in words_lower if w in FIRST_PERSON)
    sp_count = sum(1 for w in words_lower if w in SECOND_PERSON)

    try:
        compound = _get_vader().polarity_scores(all_text)["compound"]
    except Exception:
        compound = 0.0

    return {
        "word_count": word_count,
        "question_density": q_count / n_sent,
        "exclamation_density": exc_count / n_sent,
        "imperative_density": imp_count / n_sent,
        "mean_sentence_length": mean_sent_len,
        "lexical_diversity": lex_div,
        "first_person_rate": fp_count / word_count,
        "second_person_rate": sp_count / word_count,
        "sentiment_polarity": compound,
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_play(play_id: str) -> dict:
    path = PARSED_DIR / f"{play_id}.json"
    with open(path) as f:
        return json.load(f)


def _iter_scenes(play: dict):
    """Yield (act_num, scene_num, scene_dict) for every scene."""
    for act in play["acts"]:
        for scene in act["scenes"]:
            yield act["number"], scene["scene"], scene


def _extract_character_beat_sequences(
    play: dict,
    member_to_canonical: dict[str, str],
    canonical_set: set[str],
) -> list[dict]:
    """
    Return list of dicts with keys: play_id, character, act, scene, states.
    Each states list is an ordered list of beat_state dicts augmented with
    'resolved_tactic' (canonical ID or None).
    """
    sequences = []
    play_id = play["id"]
    for act in play["acts"]:
        for scene in act["scenes"]:
            char_beats: dict[str, list[tuple[int, dict]]] = defaultdict(list)
            for beat in scene["beats"]:
                for bs in beat["beat_states"]:
                    bs_copy = dict(bs)
                    bs_copy["_resolved_tactic"] = _resolve_tactic(
                        bs, member_to_canonical, canonical_set
                    )
                    # Also attach the utterances for this character in this beat
                    utt_by_speaker: dict[str, list[str]] = defaultdict(list)
                    for u in beat.get("utterances", []):
                        utt_by_speaker[u["speaker"]].append(u["text"])
                    bs_copy["_utterance_texts"] = utt_by_speaker.get(bs["character"], [])
                    char_beats[bs["character"]].append((beat["index"], bs_copy))
            for char, indexed_states in char_beats.items():
                indexed_states.sort(key=lambda x: x[0])
                states = [s for _, s in indexed_states]
                if states:
                    sequences.append({
                        "play_id": play_id,
                        "character": char,
                        "act": act["number"],
                        "scene": scene["scene"],
                        "states": states,
                    })
    return sequences


# ---------------------------------------------------------------------------
# FactorLearner
# ---------------------------------------------------------------------------

class FactorLearner:
    """Learns all factor graph parameters from Pass 1 corpus outputs."""

    def __init__(self):
        self.canonical_ids, self.member_to_canonical = _load_tactic_vocab()
        self.canonical_set = set(self.canonical_ids)
        self.n_tactics = len(self.canonical_ids)
        self.tactic_to_idx = {t: i for i, t in enumerate(self.canonical_ids)}
        # populated by _load_corpus
        self.plays: dict[str, dict] = {}
        self.sequences: list[dict] = []

    # ------------------------------------------------------------------ #
    # Corpus loading
    # ------------------------------------------------------------------ #

    def _load_corpus(self, play_ids: list[str]):
        print(f"\n--- Loading corpus: {play_ids}")
        self.plays = {}
        self.sequences = []
        for pid in play_ids:
            path = PARSED_DIR / f"{pid}.json"
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping")
                continue
            play = _load_play(pid)
            self.plays[pid] = play
            n_beats = sum(
                len(s["beats"]) for a in play["acts"] for s in a["scenes"]
            )
            print(f"  Loaded {pid}: {n_beats} beats")
            seqs = _extract_character_beat_sequences(
                play, self.member_to_canonical, self.canonical_set
            )
            self.sequences.extend(seqs)
        total_states = sum(len(s["states"]) for s in self.sequences)
        print(f"  Total sequences: {len(self.sequences)} "
              f"({total_states} beat-states)")

    # ------------------------------------------------------------------ #
    # 2.1  psi_T: Tactic transition (base)
    # ------------------------------------------------------------------ #

    def learn_tactic_transition_base(self) -> dict:
        """Compute pooled 66x66 tactic transition matrix with entropy-adaptive
        Dirichlet smoothing. Saves tactic_transition_base.json."""
        print("\n=== 2.1 psi_T: Tactic transition (base) ===")

        N = self.n_tactics
        counts = np.zeros((N, N), dtype=float)
        n_transitions = 0
        n_skipped = 0

        for seq in self.sequences:
            states = seq["states"]
            for i in range(len(states) - 1):
                t_prev = states[i]["_resolved_tactic"]
                t_curr = states[i + 1]["_resolved_tactic"]
                if t_prev is None or t_curr is None:
                    n_skipped += 1
                    continue
                if t_prev not in self.tactic_to_idx or t_curr not in self.tactic_to_idx:
                    n_skipped += 1
                    continue
                counts[self.tactic_to_idx[t_prev], self.tactic_to_idx[t_curr]] += 1
                n_transitions += 1

        print(f"  Transitions: {n_transitions}, skipped: {n_skipped}")

        # Compute per-row entropy
        row_entropies = np.zeros(N)
        for i in range(N):
            row_sum = counts[i].sum()
            if row_sum < 1:
                continue
            probs = counts[i] / row_sum
            probs = probs[probs > 0]
            row_entropies[i] = -np.sum(probs * np.log2(probs))

        max_entropy = row_entropies.max() if row_entropies.max() > 0 else 1.0
        alpha_base = 0.1

        # Per-row Dirichlet smoothing: alpha_i = alpha_base * (entropy_i / max_entropy)
        transition_matrix = np.zeros((N, N), dtype=float)
        for i in range(N):
            alpha_i = alpha_base * (row_entropies[i] / max_entropy) if max_entropy > 0 else alpha_base
            alpha_i = max(alpha_i, 1e-6)  # avoid zero alpha
            smoothed = counts[i] + alpha_i
            transition_matrix[i] = smoothed / smoothed.sum()

        # Save as nested dict: {from_tactic: {to_tactic: prob}}
        result = {}
        for i, t_from in enumerate(self.canonical_ids):
            row_dict = {}
            for j, t_to in enumerate(self.canonical_ids):
                p = float(transition_matrix[i, j])
                if p > 1e-8:
                    row_dict[t_to] = round(p, 8)
            result[t_from] = row_dict

        out_path = FACTORS_DIR / "tactic_transition_base.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path} ({N}x{N} matrix, "
              f"alpha_base={alpha_base}, entropy-adaptive)")

        return result

    # ------------------------------------------------------------------ #
    # 2.1b  Desire-conditioned transitions
    # ------------------------------------------------------------------ #

    def learn_desire_conditioned_transitions(self) -> dict:
        """Embed desires, cluster k=7, compute per-cluster transition matrices,
        fit persistence modulation beta. Saves:
          - desire_cluster_centroids.npy
          - tactic_transition_by_desire.json
          - persistence_modulation_beta.json
        """
        print("\n=== 2.1b Desire-conditioned transitions ===")

        from sentence_transformers import SentenceTransformer

        # Collect all unique desire strings
        all_desires: set[str] = set()
        for seq in self.sequences:
            for s in seq["states"]:
                d = s.get("desire_state", "").strip()
                if d:
                    all_desires.add(d)

        desire_list = sorted(all_desires)
        print(f"  Unique desire strings: {len(desire_list)}")

        # Embed
        print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Embedding {len(desire_list)} desires...")
        embeddings = model.encode(desire_list, show_progress_bar=False,
                                  normalize_embeddings=True)
        desire_to_emb = {d: embeddings[i] for i, d in enumerate(desire_list)}

        # K-means k=7
        print("  Running k-means (k=7)...")
        km = KMeans(n_clusters=7, n_init=20, random_state=42)
        labels = km.fit_predict(embeddings)
        desire_to_cluster = {d: int(labels[i]) for i, d in enumerate(desire_list)}

        centroids = km.cluster_centers_
        centroid_path = FACTORS_DIR / "desire_cluster_centroids.npy"
        np.save(centroid_path, centroids)
        print(f"  Saved {centroid_path} (shape {centroids.shape})")

        # Per-cluster transition matrices
        N = self.n_tactics
        alpha_base = 0.1
        cluster_counts = {c: np.zeros((N, N), dtype=float) for c in range(7)}
        n_trans = 0

        for seq in self.sequences:
            states = seq["states"]
            for i in range(len(states) - 1):
                t_prev = states[i]["_resolved_tactic"]
                t_curr = states[i + 1]["_resolved_tactic"]
                if t_prev is None or t_curr is None:
                    continue
                if t_prev not in self.tactic_to_idx or t_curr not in self.tactic_to_idx:
                    continue
                # Use the desire at the current beat to determine cluster
                d_curr = states[i + 1].get("desire_state", "").strip()
                if not d_curr or d_curr not in desire_to_cluster:
                    continue
                cluster = desire_to_cluster[d_curr]
                cluster_counts[cluster][
                    self.tactic_to_idx[t_prev], self.tactic_to_idx[t_curr]
                ] += 1
                n_trans += 1

        print(f"  Desire-conditioned transitions: {n_trans}")

        # Smooth each cluster matrix
        result: dict[str, dict[str, dict[str, float]]] = {}
        for c in range(7):
            mat = cluster_counts[c]
            row_entropies = np.zeros(N)
            for i in range(N):
                row_sum = mat[i].sum()
                if row_sum < 1:
                    continue
                probs = mat[i] / row_sum
                probs = probs[probs > 0]
                row_entropies[i] = -np.sum(probs * np.log2(probs))
            max_ent = row_entropies.max() if row_entropies.max() > 0 else 1.0

            smoothed_mat = np.zeros((N, N))
            for i in range(N):
                alpha_i = alpha_base * (row_entropies[i] / max_ent) if max_ent > 0 else alpha_base
                alpha_i = max(alpha_i, 1e-6)
                smoothed = mat[i] + alpha_i
                smoothed_mat[i] = smoothed / smoothed.sum()

            cluster_dict: dict[str, dict[str, float]] = {}
            for i, t_from in enumerate(self.canonical_ids):
                row_dict = {}
                for j, t_to in enumerate(self.canonical_ids):
                    p = float(smoothed_mat[i, j])
                    if p > 1e-8:
                        row_dict[t_to] = round(p, 8)
                cluster_dict[t_from] = row_dict
            result[str(c)] = cluster_dict

        out_path = FACTORS_DIR / "tactic_transition_by_desire.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path} (7 cluster matrices)")

        # Persistence modulation beta: logistic regression of
        # tactic_persisted ~ desire_similarity
        print("  Fitting persistence modulation beta...")
        X_sims = []
        y_persist = []
        for seq in self.sequences:
            states = seq["states"]
            for i in range(len(states) - 1):
                t_prev = states[i]["_resolved_tactic"]
                t_curr = states[i + 1]["_resolved_tactic"]
                if t_prev is None or t_curr is None:
                    continue
                d_prev = states[i].get("desire_state", "").strip()
                d_curr = states[i + 1].get("desire_state", "").strip()
                if not d_prev or not d_curr:
                    continue
                if d_prev not in desire_to_emb or d_curr not in desire_to_emb:
                    continue
                sim = float(np.dot(desire_to_emb[d_prev], desire_to_emb[d_curr]))
                X_sims.append([sim])
                y_persist.append(1 if t_prev == t_curr else 0)

        beta_result = {"beta": 0.0, "intercept": 0.0, "n_samples": len(X_sims)}
        if len(X_sims) >= 10 and len(set(y_persist)) == 2:
            X_arr = np.array(X_sims)
            y_arr = np.array(y_persist)
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_arr, y_arr)
            beta_result["beta"] = float(lr.coef_[0, 0])
            beta_result["intercept"] = float(lr.intercept_[0])
            print(f"  beta = {beta_result['beta']:.4f}, "
                  f"intercept = {beta_result['intercept']:.4f} "
                  f"(N={len(X_sims)})")
        else:
            print(f"  WARNING: insufficient data for logistic regression "
                  f"(N={len(X_sims)})")

        beta_path = FACTORS_DIR / "persistence_modulation_beta.json"
        with open(beta_path, "w") as f:
            json.dump(beta_result, f, indent=2)
        print(f"  Saved {beta_path}")

        return result

    # ------------------------------------------------------------------ #
    # 2.2  psi_A: Affect transition
    # ------------------------------------------------------------------ #

    def learn_affect_transition(self) -> dict:
        """Compute affect delta covariance, eigendecompose, fit Student-t df.
        Saves:
          - affect_eigenvectors.npy (3x5 rotation matrix)
          - affect_eigenvalues.npy
          - affect_transition_variance.npy (3 diagonal values in eigenspace)
          - affect_transition_df.json (df per axis)
        """
        print("\n=== 2.2 psi_A: Affect transition ===")

        deltas = []
        for seq in self.sequences:
            states = seq["states"]
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
                    delta.append(float(v_curr) - float(v_prev))
                if valid and len(delta) == 5:
                    deltas.append(delta)

        deltas_arr = np.array(deltas)
        print(f"  Affect transitions: {deltas_arr.shape[0]}")

        if deltas_arr.shape[0] < 5:
            print("  WARNING: too few affect transitions, skipping")
            return {}

        # 5x5 covariance matrix
        cov = np.cov(deltas_arr.T)

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Top 3 eigenvectors as rotation matrix R (3x5)
        R = eigenvectors[:, :3].T  # shape (3, 5)

        np.save(FACTORS_DIR / "affect_eigenvectors.npy", R)
        np.save(FACTORS_DIR / "affect_eigenvalues.npy", eigenvalues)
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Top-3 explain {eigenvalues[:3].sum() / eigenvalues.sum() * 100:.1f}% of variance")

        # Project deltas into eigenspace
        projected = deltas_arr @ R.T  # shape (N, 3)

        # Transition variance in eigenspace (diagonal)
        transition_var = np.var(projected, axis=0)
        np.save(FACTORS_DIR / "affect_transition_variance.npy", transition_var)
        print(f"  Transition variance in eigenspace: {transition_var}")

        # Fit Student-t df per axis
        df_per_axis = {}
        for axis_idx in range(3):
            col = projected[:, axis_idx]
            try:
                df_fit, loc_fit, scale_fit = stats.t.fit(col)
                df_per_axis[f"axis_{axis_idx}"] = {
                    "df": float(df_fit),
                    "loc": float(loc_fit),
                    "scale": float(scale_fit),
                }
            except Exception as e:
                print(f"  WARNING: Student-t fit failed for axis {axis_idx}: {e}")
                df_per_axis[f"axis_{axis_idx}"] = {
                    "df": 30.0,  # fallback to near-Gaussian
                    "loc": float(np.mean(col)),
                    "scale": float(np.std(col)),
                }

        df_path = FACTORS_DIR / "affect_transition_df.json"
        with open(df_path, "w") as f:
            json.dump(df_per_axis, f, indent=2)

        print(f"  Saved affect_eigenvectors.npy, affect_eigenvalues.npy, "
              f"affect_transition_variance.npy, affect_transition_df.json")
        for k, v in df_per_axis.items():
            print(f"    {k}: df={v['df']:.2f}, loc={v['loc']:.4f}, scale={v['scale']:.4f}")

        return df_per_axis

    # ------------------------------------------------------------------ #
    # 2.3  A_emit: Arousal regression
    # ------------------------------------------------------------------ #

    def learn_arousal_regression(self) -> dict:
        """Train Ridge regression: arousal ~ text_features + character_mean_arousal.
        Saves:
          - arousal_regressor.pkl (joblib)
          - arousal_residual_variance.json
        """
        print("\n=== 2.3 A_emit: Arousal regression ===")

        # Collect per-character mean arousal
        char_arousal_sums: dict[str, list[float]] = defaultdict(list)
        for seq in self.sequences:
            for s in seq["states"]:
                a = s.get("affect_state", {}).get("arousal")
                if a is not None:
                    key = f"{seq['play_id']}_{s['character']}"
                    char_arousal_sums[key].append(float(a))

        char_mean_arousal = {
            k: np.mean(v) for k, v in char_arousal_sums.items()
        }

        # Build training data
        X_rows = []
        y_rows = []

        for seq in self.sequences:
            for s in seq["states"]:
                texts = s.get("_utterance_texts", [])
                if not texts:
                    continue
                feats = compute_utterance_features(texts)
                if feats is None:
                    continue
                arousal = s.get("affect_state", {}).get("arousal")
                if arousal is None:
                    continue

                char_key = f"{seq['play_id']}_{s['character']}"
                mean_a = char_mean_arousal.get(char_key, 0.0)

                feature_vec = [feats[f] for f in UTT_FEATURES] + [mean_a]
                X_rows.append(feature_vec)
                y_rows.append(float(arousal))

        if len(X_rows) < 10:
            print(f"  WARNING: too few samples ({len(X_rows)}), skipping")
            return {}

        X = np.array(X_rows)
        y = np.array(y_rows)
        print(f"  Training samples: {len(X)}")

        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        y_pred = ridge.predict(X)
        residuals = y - y_pred
        residual_var = float(np.var(residuals))

        # Save model
        model_path = FACTORS_DIR / "arousal_regressor.pkl"
        joblib.dump(ridge, model_path)

        result = {
            "residual_variance": residual_var,
            "n_samples": len(X),
            "feature_names": UTT_FEATURES + ["character_mean_arousal"],
            "coefficients": {
                name: float(c) for name, c in
                zip(UTT_FEATURES + ["character_mean_arousal"], ridge.coef_)
            },
            "intercept": float(ridge.intercept_),
            "r_squared": float(1 - residual_var / np.var(y)) if np.var(y) > 0 else 0.0,
        }

        var_path = FACTORS_DIR / "arousal_residual_variance.json"
        with open(var_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  R^2 = {result['r_squared']:.4f}, residual variance = {residual_var:.6f}")
        print(f"  Saved {model_path}, {var_path}")

        return result

    # ------------------------------------------------------------------ #
    # 2.4  psi_D: Desire transition
    # ------------------------------------------------------------------ #

    def learn_desire_transition(self) -> dict:
        """Compute 7x7 desire cluster transition matrix.
        Requires desire_cluster_centroids.npy (from 2.1b).
        Saves desire_transition_matrix.json.
        """
        print("\n=== 2.4 psi_D: Desire transition ===")

        centroid_path = FACTORS_DIR / "desire_cluster_centroids.npy"
        if not centroid_path.exists():
            print("  ERROR: desire_cluster_centroids.npy not found. "
                  "Run learn_desire_conditioned_transitions first.")
            return {}

        from sentence_transformers import SentenceTransformer

        centroids = np.load(centroid_path)
        k = centroids.shape[0]  # should be 7

        # Collect unique desires and embed
        all_desires: set[str] = set()
        for seq in self.sequences:
            for s in seq["states"]:
                d = s.get("desire_state", "").strip()
                if d:
                    all_desires.add(d)
        desire_list = sorted(all_desires)

        print(f"  Loading sentence-transformers model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(desire_list, show_progress_bar=False,
                                  normalize_embeddings=True)
        desire_to_emb = {d: embeddings[i] for i, d in enumerate(desire_list)}

        # Assign each desire to nearest centroid
        def assign_cluster(emb: np.ndarray) -> int:
            dists = np.linalg.norm(centroids - emb, axis=1)
            return int(np.argmin(dists))

        desire_to_cluster = {d: assign_cluster(e) for d, e in desire_to_emb.items()}

        # Compute transition counts
        counts = np.zeros((k, k), dtype=float)
        n_trans = 0
        for seq in self.sequences:
            states = seq["states"]
            for i in range(len(states) - 1):
                d_prev = states[i].get("desire_state", "").strip()
                d_curr = states[i + 1].get("desire_state", "").strip()
                if not d_prev or not d_curr:
                    continue
                if d_prev not in desire_to_cluster or d_curr not in desire_to_cluster:
                    continue
                c_prev = desire_to_cluster[d_prev]
                c_curr = desire_to_cluster[d_curr]
                counts[c_prev, c_curr] += 1
                n_trans += 1

        print(f"  Desire transitions: {n_trans}")

        # Dirichlet smoothing (uniform alpha=0.1)
        alpha = 0.1
        smoothed = counts + alpha
        transition_matrix = smoothed / smoothed.sum(axis=1, keepdims=True)

        # Save as nested dict
        result = {}
        for i in range(k):
            row = {}
            for j in range(k):
                row[str(j)] = round(float(transition_matrix[i, j]), 8)
            result[str(i)] = row

        out_path = FACTORS_DIR / "desire_transition_matrix.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path} ({k}x{k} matrix, alpha={alpha})")

        return result

    # ------------------------------------------------------------------ #
    # 2.5  psi_arc: Superobjective prior
    # ------------------------------------------------------------------ #

    def learn_superobjective_prior(self) -> dict:
        """Embed superobjectives, cluster by silhouette (k=3-5), compute per-cluster
        tactic distribution. Saves:
          - superobjective_tactic_prior.json
          - superobjective_cluster_centroids.npy
        """
        print("\n=== 2.5 psi_arc: Superobjective prior ===")

        from sentence_transformers import SentenceTransformer

        # Collect (character, play_id, superobjective) triples
        char_so: list[dict] = []
        for pid, play in self.plays.items():
            for cb in play.get("character_bibles", []):
                so = cb.get("superobjective", "").strip()
                if so:
                    char_so.append({
                        "character": cb["character"],
                        "play_id": pid,
                        "superobjective": so,
                    })

        if len(char_so) < 3:
            print(f"  WARNING: only {len(char_so)} superobjectives found, skipping")
            return {}

        print(f"  Superobjectives: {len(char_so)}")

        # Embed
        print("  Loading sentence-transformers model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        so_texts = [c["superobjective"] for c in char_so]
        so_embeddings = model.encode(so_texts, show_progress_bar=False,
                                     normalize_embeddings=True)

        # Find best k by silhouette (k=3..min(5, n-1))
        max_k = min(5, len(char_so) - 1)
        if max_k < 3:
            max_k = min(len(char_so) - 1, 3)

        best_k = 3
        best_sil = -1.0
        for k in range(3, max_k + 1):
            if k >= len(char_so):
                break
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            labels = km.fit_predict(so_embeddings)
            sil = silhouette_score(so_embeddings, labels)
            print(f"  k={k}: silhouette={sil:.4f}")
            if sil > best_sil:
                best_sil = sil
                best_k = k

        print(f"  Best k={best_k} (silhouette={best_sil:.4f})")

        km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
        so_labels = km.fit_predict(so_embeddings)

        # Save centroids
        centroid_path = FACTORS_DIR / "superobjective_cluster_centroids.npy"
        np.save(centroid_path, km.cluster_centers_)

        # Assign each character to a cluster
        char_to_cluster: dict[str, int] = {}
        for i, cs in enumerate(char_so):
            key = f"{cs['play_id']}_{cs['character']}"
            char_to_cluster[key] = int(so_labels[i])

        # Per SO cluster, compute empirical tactic distribution
        cluster_tactic_counts: dict[int, Counter] = defaultdict(Counter)
        for seq in self.sequences:
            key = f"{seq['play_id']}_{seq['character']}"
            cluster = char_to_cluster.get(key)
            if cluster is None:
                continue
            for s in seq["states"]:
                tactic = s["_resolved_tactic"]
                if tactic:
                    cluster_tactic_counts[cluster][tactic] += 1

        result: dict[str, dict[str, float]] = {}
        for c in range(best_k):
            counts = cluster_tactic_counts[c]
            total = sum(counts.values())
            if total == 0:
                result[str(c)] = {}
                continue
            dist = {t: round(cnt / total, 6) for t, cnt in counts.most_common()}
            result[str(c)] = dist

            # Print top 5 for this cluster
            top5 = counts.most_common(5)
            members = [cs["character"] for i, cs in enumerate(char_so) if so_labels[i] == c]
            print(f"  Cluster {c} (chars: {', '.join(members[:5])}):")
            for t, cnt in top5:
                print(f"    {t:<16} {cnt:>4} ({cnt / total * 100:.1f}%)")

        out_path = FACTORS_DIR / "superobjective_tactic_prior.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path}, {centroid_path}")

        return result

    # ------------------------------------------------------------------ #
    # 2.6  psi_emit: Tactic emission profiles
    # ------------------------------------------------------------------ #

    def learn_tactic_emission_profiles(self) -> dict:
        """For each canonical tactic, compute mean and std of each text feature.
        Saves tactic_emission_profiles.json.
        """
        print("\n=== 2.6 psi_emit: Tactic emission profiles ===")

        # Collect features per tactic
        tactic_features: dict[str, list[dict[str, float]]] = defaultdict(list)

        for seq in self.sequences:
            for s in seq["states"]:
                tactic = s["_resolved_tactic"]
                if tactic is None:
                    continue
                texts = s.get("_utterance_texts", [])
                if not texts:
                    continue
                feats = compute_utterance_features(texts)
                if feats is None:
                    continue
                tactic_features[tactic].append(feats)

        total_beats = sum(len(v) for v in tactic_features.values())
        print(f"  Beats with tactic + text features: {total_beats}")
        print(f"  Unique tactics with data: {len(tactic_features)}")

        result = {}
        for tactic in sorted(tactic_features.keys()):
            feat_dicts = tactic_features[tactic]
            n = len(feat_dicts)
            profile = {"n": n, "mean": {}, "std": {}}
            for f in UTT_FEATURES:
                vals = [fd[f] for fd in feat_dicts]
                profile["mean"][f] = round(float(np.mean(vals)), 6)
                profile["std"][f] = round(float(np.std(vals)), 6) if n > 1 else 0.0
            result[tactic] = profile

        out_path = FACTORS_DIR / "tactic_emission_profiles.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path} ({len(result)} tactic profiles)")

        return result

    # ------------------------------------------------------------------ #
    # 2.7  psi_social: Status coupling
    # ------------------------------------------------------------------ #

    def learn_status_coupling(self) -> dict:
        """Compute Pearson r between co-present characters' status values,
        derive gamma = |r| / (1 - r^2). Saves status_coupling_gamma.json.
        """
        print("\n=== 2.7 psi_social: Status coupling ===")

        all_pairs: list[tuple[float, float]] = []

        for pid, play in self.plays.items():
            for act in play["acts"]:
                for scene in act["scenes"]:
                    for beat in scene["beats"]:
                        beat_states = beat.get("beat_states", [])
                        if len(beat_states) < 2:
                            continue
                        char_status: dict[str, float] = {}
                        for bs in beat_states:
                            ss = bs.get("social_state", {})
                            status = ss.get("status")
                            if status is not None:
                                char_status[bs["character"]] = float(status)
                        chars = list(char_status.keys())
                        for ca, cb in combinations(chars, 2):
                            all_pairs.append((char_status[ca], char_status[cb]))

        if len(all_pairs) < 3:
            print(f"  WARNING: only {len(all_pairs)} status pairs, skipping")
            result = {"gamma": 0.0, "pearson_r": 0.0, "p_value": 1.0,
                      "n_pairs": len(all_pairs)}
        else:
            a_status = np.array([p[0] for p in all_pairs])
            b_status = np.array([p[1] for p in all_pairs])
            r, p_val = stats.pearsonr(a_status, b_status)
            # gamma = |r| / (1 - r^2), clamp denominator
            denom = max(1 - r ** 2, 1e-6)
            gamma = abs(r) / denom
            result = {
                "gamma": round(float(gamma), 6),
                "pearson_r": round(float(r), 6),
                "p_value": round(float(p_val), 6),
                "n_pairs": len(all_pairs),
            }
            print(f"  Status pairs: {len(all_pairs)}")
            print(f"  Pearson r = {r:+.4f}, p = {p_val:.4f}")
            print(f"  gamma = {gamma:.4f}")

        out_path = FACTORS_DIR / "status_coupling_gamma.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path}")

        return result

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def learn_from_corpus(self, play_ids: list[str]):
        """Run all sub-computations and save parameters to data/factors/."""
        print("=" * 72)
        print("FACTOR GRAPH PARAMETER LEARNING (Pass 1.5a)")
        print("=" * 72)

        self._load_corpus(play_ids)
        if not self.plays:
            print("ERROR: No plays loaded. Exiting.")
            return

        # Order matters: 2.1b must run before 2.4 (desire centroids needed)
        self.learn_tactic_transition_base()               # 2.1
        self.learn_desire_conditioned_transitions()       # 2.1b
        self.learn_affect_transition()                    # 2.2
        self.learn_arousal_regression()                   # 2.3
        self.learn_desire_transition()                    # 2.4
        self.learn_superobjective_prior()                 # 2.5
        self.learn_tactic_emission_profiles()             # 2.6
        self.learn_status_coupling()                      # 2.7

        print("\n" + "=" * 72)
        print("ALL FACTOR PARAMETERS SAVED TO:", FACTORS_DIR)
        print("=" * 72)
        print("\nOutput files:")
        for p in sorted(FACTORS_DIR.iterdir()):
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:<45} {size_kb:>8.1f} KB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Learn factor graph parameters from Pass 1 outputs"
    )
    parser.add_argument(
        "--plays", nargs="+", required=True,
        help="Play IDs to learn from (e.g. cherry_orchard hamlet importance_of_being_earnest)",
    )
    args = parser.parse_args()

    learner = FactorLearner()
    learner.learn_from_corpus(args.plays)


if __name__ == "__main__":
    main()
