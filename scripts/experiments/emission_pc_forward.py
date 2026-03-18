#!/usr/bin/env python3
"""
Forward Emission Model: PC Scores -> Utterance Text Features
=============================================================
Complements the inverse experiment (text->PC) by testing the forward direction:
given a character's PC scores (from eigendecomposition of affect transitions),
how well can we predict their utterance text features?

Steps:
  1. Load all 3 parsed plays; extract (beat, character) pairs with affect + text.
  2. Compute eigenvectors from affect transition covariance (same as
     affect_eigendecomposition.py). Project each beat's 5D affect onto PCs.
  3. Extract 9 utterance features per (beat, character).
  4. Forward regression: for each utterance feature, predict from 5 PC scores
     (Ridge + RF). Compare R^2 vs using 5 raw affect dims as input.
  5. Feature-level analysis: which PCs are strongest predictors per feature.
  6. Correlation matrix: Pearson r between all PCs and all utterance features.
  7. Side-by-side R^2 comparison: PCs vs raw dims as predictors.
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ── paths & constants ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "parsed"
PLAY_IDS = ["hamlet", "cherry_orchard", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]
PC_LABELS = {
    0: "PC1:Disempowerment",
    1: "PC2:BlissfulIgnorance",
    2: "PC3:BurdenedPower",
    3: "PC4:Arousal",
    4: "PC5:Exposure",
}
UTT_FEATURES = [
    "word_count", "question_density", "exclamation_density",
    "imperative_density", "mean_sentence_length", "lexical_diversity",
    "first_person_rate", "second_person_rate", "sentiment_polarity",
]

# ── sentiment analyzer ────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()

# ── imperative seed words (from emission_model.py) ────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def affect_vec(affect_state: dict) -> np.ndarray:
    return np.array([affect_state.get(d, 0.0) for d in AFFECT_DIMS], dtype=float)


def load_play(play_id: str) -> dict:
    with open(DATA_DIR / f"{play_id}.json") as f:
        return json.load(f)


def split_sentences(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]


def compute_utterance_features(texts: list[str]) -> dict | None:
    all_text = " ".join(texts)
    words = re.findall(r"[A-Za-z']+", all_text)
    word_count = len(words)
    if word_count == 0:
        return None

    sentences = []
    for t in texts:
        sentences.extend(split_sentences(t))
    n_sent = max(len(sentences), 1)

    q_count = sum(1 for s in sentences if s.rstrip().endswith("?"))
    exc_count = sum(1 for s in sentences if s.rstrip().endswith("!"))

    imp_count = 0
    for s in sentences:
        first_word = re.match(r"[A-Za-z']+", s.strip())
        if first_word and first_word.group().lower() in IMPERATIVE_STARTERS:
            imp_count += 1

    sent_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
    mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0

    words_lower = [w.lower() for w in words]
    lex_div = len(set(words_lower)) / len(words_lower)

    first_person = {"i", "me", "my", "mine", "myself"}
    second_person = {"you", "your", "yours", "yourself", "yourselves"}
    fp_count = sum(1 for w in words_lower if w in first_person)
    sp_count = sum(1 for w in words_lower if w in second_person)

    compound = _vader.polarity_scores(all_text)["compound"]

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


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMPUTE EIGENVECTORS FROM AFFECT TRANSITION COVARIANCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_eigenvectors():
    """Reproduce eigendecomposition from affect_eigendecomposition.py.
    Returns (eigenvalues_sorted, eigenvectors_sorted) with columns = PCs.
    """
    deltas = []
    for play_id in PLAY_IDS:
        play = load_play(play_id)
        for act in play["acts"]:
            for scene in act["scenes"]:
                char_beats = defaultdict(list)
                for beat in scene["beats"]:
                    for bs in beat.get("beat_states", []):
                        char_beats[bs["character"]].append(affect_vec(bs["affect_state"]))
                for char, vecs in char_beats.items():
                    for i in range(1, len(vecs)):
                        deltas.append(vecs[i] - vecs[i - 1])

    deltas = np.array(deltas)
    cov = np.cov(deltas, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx], len(deltas)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD DATASET: (beat, character) with PC scores + utterance features
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(eigvecs: np.ndarray) -> pd.DataFrame:
    rows = []
    for play_id in PLAY_IDS:
        play = load_play(play_id)
        for act in play["acts"]:
            for scene in act["scenes"]:
                for beat in scene["beats"]:
                    utt_by_speaker: dict[str, list[str]] = defaultdict(list)
                    for u in beat["utterances"]:
                        utt_by_speaker[u["speaker"]].append(u["text"])

                    for bs in beat.get("beat_states", []):
                        char = bs["character"]
                        texts = utt_by_speaker.get(char, [])
                        if not texts:
                            continue
                        feats = compute_utterance_features(texts)
                        if feats is None:
                            continue

                        aff = bs["affect_state"]
                        raw = affect_vec(aff)
                        pc_scores = eigvecs.T @ raw  # project onto all 5 PCs

                        row = {
                            "play_id": play_id,
                            "character": char,
                        }
                        # Raw affect dims
                        for j, d in enumerate(AFFECT_DIMS):
                            row[d] = raw[j]
                        # PC scores
                        for j in range(5):
                            row[f"PC{j+1}"] = pc_scores[j]
                        # Utterance features
                        row.update(feats)
                        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FORWARD REGRESSION: PC scores -> each utterance feature
# ══════════════════════════════════════════════════════════════════════════════

def experiment_forward_regression(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Forward Regression — PC scores -> utterance features")
    print("=" * 72)

    pc_cols = [f"PC{i+1}" for i in range(5)]
    X_pc = df[pc_cols].values
    X_raw = df[AFFECT_DIMS].values

    header = (f"{'Utterance Feature':<25} {'Ridge(PC)':>10} {'RF(PC)':>10} "
              f"{'Ridge(raw)':>11} {'RF(raw)':>10} {'PC vs raw':>10}")
    print(f"\n{header}")
    print("-" * 78)

    results = []
    for feat in UTT_FEATURES:
        y = df[feat].values

        # PC-based models
        r2_ridge_pc = np.mean(cross_val_score(
            RidgeCV(alphas=[0.01, 0.1, 1, 10]), X_pc, y, cv=5, scoring="r2"))
        r2_rf_pc = np.mean(cross_val_score(
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            X_pc, y, cv=5, scoring="r2"))

        # Raw affect dim models
        r2_ridge_raw = np.mean(cross_val_score(
            RidgeCV(alphas=[0.01, 0.1, 1, 10]), X_raw, y, cv=5, scoring="r2"))
        r2_rf_raw = np.mean(cross_val_score(
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            X_raw, y, cv=5, scoring="r2"))

        best_pc = max(r2_ridge_pc, r2_rf_pc)
        best_raw = max(r2_ridge_raw, r2_rf_raw)
        winner = "PC" if best_pc > best_raw else "raw" if best_raw > best_pc else "tie"

        results.append({
            "feature": feat,
            "ridge_pc": r2_ridge_pc, "rf_pc": r2_rf_pc,
            "ridge_raw": r2_ridge_raw, "rf_raw": r2_rf_raw,
            "winner": winner,
        })

        print(f"{feat:<25} {r2_ridge_pc:>+10.4f} {r2_rf_pc:>+10.4f} "
              f"{r2_ridge_raw:>+11.4f} {r2_rf_raw:>+10.4f} {'<-- '+winner:>10}")

    # Summary
    print(f"\nNote: R^2 < 0 means the model is worse than predicting the mean.")
    print(f"      PC and raw R^2 should be similar for Ridge (linear rotation),")
    print(f"      but may differ for RF (nonlinear interactions between PCs).")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE-LEVEL ANALYSIS: which PCs predict each utterance feature?
# ══════════════════════════════════════════════════════════════════════════════

def experiment_feature_importance(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Feature-Level Analysis — PC importance per utterance feature")
    print("=" * 72)

    pc_cols = [f"PC{i+1}" for i in range(5)]
    X_pc = df[pc_cols].values

    print(f"\nRandom Forest feature importances (which PCs drive each utterance feature):")
    print(f"\n{'Utterance Feature':<25}", end="")
    for i in range(5):
        label = PC_LABELS[i].split(":")[1][:12]
        print(f" {f'PC{i+1}':>6}({label[:8]})", end="")
    print(f"  {'Top predictor':<20}")
    print("-" * 110)

    importance_matrix = np.zeros((len(UTT_FEATURES), 5))

    for fi, feat in enumerate(UTT_FEATURES):
        y = df[feat].values
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        rf.fit(X_pc, y)
        imp = rf.feature_importances_
        importance_matrix[fi] = imp

        top_idx = np.argmax(imp)
        top_label = PC_LABELS[top_idx]

        print(f"{feat:<25}", end="")
        for j in range(5):
            marker = "*" if j == top_idx else " "
            print(f" {imp[j]:>6.3f}{marker}         ", end="")
        print(f"  {top_label}")

    # Summary: which PCs are most important overall
    print(f"\nOverall PC importance (mean across utterance features):")
    mean_imp = importance_matrix.mean(axis=0)
    order = np.argsort(-mean_imp)
    for rank, idx in enumerate(order, 1):
        print(f"  {rank}. {PC_LABELS[idx]:<30} {mean_imp[idx]:.4f}")

    return importance_matrix


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION MATRIX: PCs x utterance features
# ══════════════════════════════════════════════════════════════════════════════

def experiment_correlation_matrix(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Pearson Correlation Matrix — PCs x Utterance Features")
    print("=" * 72)

    pc_cols = [f"PC{i+1}" for i in range(5)]

    # Compute correlation matrix
    corr_data = np.zeros((5, len(UTT_FEATURES)))
    for i in range(5):
        for j, feat in enumerate(UTT_FEATURES):
            corr_data[i, j] = np.corrcoef(df[pc_cols[i]].values, df[feat].values)[0, 1]

    # Print as table
    print(f"\n{'':>25}", end="")
    for i in range(5):
        short = PC_LABELS[i].split(":")[1][:10]
        print(f" {f'PC{i+1}':>12}", end="")
    print()
    print("-" * (25 + 13 * 5))

    for j, feat in enumerate(UTT_FEATURES):
        print(f"{feat:<25}", end="")
        for i in range(5):
            r = corr_data[i, j]
            if abs(r) >= 0.15:
                print(f" {r:>+12.3f}", end="")
            elif abs(r) >= 0.08:
                print(f" {r:>+12.3f}", end="")
            else:
                print(f" {'   .   ':>12}", end="")
        print()

    # Highlight strongest associations
    print(f"\nStrongest correlations (|r| >= 0.10):")
    pairs = []
    for i in range(5):
        for j, feat in enumerate(UTT_FEATURES):
            r = corr_data[i, j]
            if abs(r) >= 0.10:
                pairs.append((abs(r), r, PC_LABELS[i], feat))
    pairs.sort(reverse=True)
    for abs_r, r, pc, feat in pairs:
        direction = "+" if r > 0 else "-"
        print(f"  {pc:<30} {direction} {feat:<25}  r = {r:+.3f}")

    # Also show raw affect dim correlations for comparison
    print(f"\n\nFor comparison — Raw affect dim correlations with utterance features:")
    print(f"{'':>25}", end="")
    for d in AFFECT_DIMS:
        print(f" {d[:12]:>12}", end="")
    print()
    print("-" * (25 + 13 * 5))

    for j, feat in enumerate(UTT_FEATURES):
        print(f"{feat:<25}", end="")
        for d in AFFECT_DIMS:
            r = np.corrcoef(df[d].values, df[feat].values)[0, 1]
            if abs(r) >= 0.08:
                print(f" {r:>+12.3f}", end="")
            else:
                print(f" {'   .   ':>12}", end="")
        print()

    return corr_data


# ══════════════════════════════════════════════════════════════════════════════
# 6. RIDGE COEFFICIENT ANALYSIS (interpretable linear weights)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_ridge_coefficients(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Ridge Regression Coefficients — PC -> utterance features")
    print("=" * 72)

    pc_cols = [f"PC{i+1}" for i in range(5)]
    X_pc = df[pc_cols].values

    # Standardize inputs for comparable coefficients
    X_mean = X_pc.mean(axis=0)
    X_std = X_pc.std(axis=0)
    X_z = (X_pc - X_mean) / X_std

    print(f"\nStandardized Ridge coefficients (comparable magnitudes):")
    print(f"\n{'Utterance Feature':<25}", end="")
    for i in range(5):
        print(f" {'PC'+str(i+1):>8}", end="")
    print(f"  {'Dominant PC':<25}")
    print("-" * 85)

    for feat in UTT_FEATURES:
        y = df[feat].values
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10])
        ridge.fit(X_z, y)
        coefs = ridge.coef_

        top_idx = np.argmax(np.abs(coefs))
        top_label = PC_LABELS[top_idx]

        print(f"{feat:<25}", end="")
        for j in range(5):
            marker = "*" if j == top_idx else " "
            print(f" {coefs[j]:>+7.4f}{marker}", end="")
        print(f"  {top_label}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  FORWARD EMISSION MODEL: PC Scores -> Utterance Text Features")
    print("=" * 72)

    # Step 1: Compute eigenvectors
    print(f"\n{'─'*72}")
    print("STEP 1: Eigendecomposition of affect transition covariance")
    print(f"{'─'*72}")

    eigenvalues, eigenvectors, n_deltas = compute_eigenvectors()
    total_var = eigenvalues.sum()

    print(f"  Transition deltas: {n_deltas}")
    print(f"  Eigenvalue spectrum:")
    for i in range(5):
        pct = eigenvalues[i] / total_var * 100
        print(f"    {PC_LABELS[i]:<30}  eigenvalue={eigenvalues[i]:.4f}  ({pct:.1f}%)")

    print(f"\n  Eigenvector loadings:")
    print(f"  {'':>30}", end="")
    for d in AFFECT_DIMS:
        print(f" {d[:10]:>10}", end="")
    print()
    for i in range(5):
        print(f"  {PC_LABELS[i]:<30}", end="")
        for j in range(5):
            print(f" {eigenvectors[j, i]:>+10.3f}", end="")
        print()

    # Step 2: Build dataset
    print(f"\n{'─'*72}")
    print("STEP 2: Build dataset with PC scores + utterance features")
    print(f"{'─'*72}")

    df = build_dataset(eigenvectors)
    print(f"  Observations: {len(df)} (beat, character) pairs")
    print(f"  Plays: {df['play_id'].value_counts().to_dict()}")

    print(f"\n  PC score summary statistics:")
    pc_cols = [f"PC{i+1}" for i in range(5)]
    print(df[pc_cols].describe().round(3).to_string())

    print(f"\n  Utterance feature summary statistics:")
    print(df[UTT_FEATURES].describe().round(3).to_string())

    # Run experiments
    regression_results = experiment_forward_regression(df)
    importance_matrix = experiment_feature_importance(df)
    corr_matrix = experiment_correlation_matrix(df)
    experiment_ridge_coefficients(df)

    # ── Final Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY: FORWARD EMISSION MODEL FINDINGS")
    print("=" * 72)

    # Count wins
    pc_wins = sum(1 for r in regression_results if r["winner"] == "PC")
    raw_wins = sum(1 for r in regression_results if r["winner"] == "raw")

    print(f"""
  1. PREDICTIVE POWER (R^2):
     - PCs won for {pc_wins}/{len(regression_results)} features, raw dims won for {raw_wins}/{len(regression_results)}.
     - For Ridge (linear), PCs and raw dims yield identical R^2 (rotation invariance).
     - For RF (nonlinear), differences reflect nonlinear PC interactions.
""")

    # Best-predicted features
    best_rf = sorted(regression_results, key=lambda r: r["rf_pc"], reverse=True)
    print(f"  2. BEST-PREDICTED FEATURES (RF with PC scores):")
    for r in best_rf[:4]:
        print(f"     {r['feature']:<25} R^2 = {r['rf_pc']:+.4f}")
    print()

    # Strongest PC-feature links from correlation
    print(f"  3. STRONGEST PC-FEATURE LINKS (Pearson |r|):")
    pairs = []
    for i in range(5):
        for j, feat in enumerate(UTT_FEATURES):
            pairs.append((abs(corr_matrix[i, j]), corr_matrix[i, j],
                          PC_LABELS[i], feat))
    pairs.sort(reverse=True)
    for abs_r, r, pc, feat in pairs[:8]:
        print(f"     {pc:<30}  <->  {feat:<25}  r = {r:+.3f}")

    print(f"""
  4. INTERPRETATION:
     - The forward emission model (PC -> text) has weak-to-modest R^2,
       consistent with the inverse model's findings.
     - The affect state provides soft constraints on text features rather
       than deterministic mapping — appropriate for a probabilistic factor
       graph where emission distributions are broad.
     - The correlation structure reveals which PCs modulate which aspects
       of language, informing the emission factor parameterization.
""")


if __name__ == "__main__":
    main()
