#!/usr/bin/env python3
"""
PC Recovery from Utterance Features
=====================================
Tests how well each of the 5 principal component axes (from the affect
transition eigendecomposition) can be recovered from utterance text features.

This directly informs the eigenspace caveat: if the top-3 PCs (which we
propose to keep) are *less* recoverable from text than PC4-5, then the
emission model must rely on transition dynamics for the primary axes.

Steps:
  1. Compute eigenvectors from affect transition covariance (replicating
     affect_eigendecomposition.py).
  2. Project each (beat, character) affect vector onto all 5 eigenvectors
     to get PC1-PC5 scores.
  3. Extract the same 9 utterance features used in the emission model.
  4. Per-PC regression (Ridge + Random Forest).
  5. Per-PC classification (binarized at median).
  6. Feature importance per PC.
  7. Compare PC recoverability vs raw dimension recoverability.
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "parsed"
PLAY_IDS = ["hamlet", "cherry_orchard", "importance_of_being_earnest"]
AFFECT_DIMS = ["valence", "arousal", "certainty", "control", "vulnerability"]

# ── sentiment analyzer ─────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()

# ── imperative seed words ──────────────────────────────────────────────────
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

UTT_FEATURES = [
    "word_count", "question_density", "exclamation_density",
    "imperative_density", "mean_sentence_length", "lexical_diversity",
    "first_person_rate", "second_person_rate", "sentiment_polarity",
]


# ══════════════════════════════════════════════════════════════════════════
# UTTERANCE FEATURE EXTRACTION (reused from emission_model.py)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# 1. COMPUTE EIGENVECTORS FROM AFFECT TRANSITION COVARIANCE
# ══════════════════════════════════════════════════════════════════════════

def affect_vec(affect_state: dict) -> np.ndarray:
    return np.array([affect_state.get(d, 0.0) for d in AFFECT_DIMS], dtype=float)


def compute_eigenvectors():
    """Replicate eigendecomposition to get eigenvectors and eigenvalues."""
    deltas = []
    for play_id in PLAY_IDS:
        path = DATA_DIR / f"{play_id}.json"
        with open(path) as f:
            play = json.load(f)
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
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors, cov


# ══════════════════════════════════════════════════════════════════════════
# 2. BUILD DATASET WITH PC SCORES AND UTTERANCE FEATURES
# ══════════════════════════════════════════════════════════════════════════

def build_dataset(eigenvectors: np.ndarray) -> pd.DataFrame:
    """Build (beat, character) dataset with PC scores and utterance features."""
    rows = []
    for play_id in PLAY_IDS:
        path = DATA_DIR / f"{play_id}.json"
        with open(path) as f:
            play = json.load(f)
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
                        aff_vec = affect_vec(aff)
                        # Project onto eigenvectors
                        pc_scores = eigenvectors.T @ aff_vec

                        row = {
                            "play_id": play_id,
                            "character": char,
                        }
                        # Raw affect dimensions
                        for d in AFFECT_DIMS:
                            row[d] = aff.get(d, 0.0)
                        # PC scores
                        for i in range(5):
                            row[f"PC{i+1}"] = pc_scores[i]
                        # Utterance features
                        row.update(feats)
                        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 3. PER-PC REGRESSION
# ══════════════════════════════════════════════════════════════════════════

def experiment_regression(df: pd.DataFrame, targets: list[str], target_label: str):
    """Train Ridge + RF regressors predicting each target from utterance features."""
    print(f"\n{'='*72}")
    print(f"REGRESSION: Utterance features -> {target_label}")
    print(f"{'='*72}")

    X = df[UTT_FEATURES].values
    results = []

    print(f"\n  {'Target':<20} {'R2 (Ridge)':<14} {'R2 (RF)':<14}")
    print(f"  {'-'*48}")

    for target in targets:
        y = df[target].values

        ridge_scores = cross_val_score(
            RidgeCV(alphas=[0.01, 0.1, 1, 10]),
            X, y, cv=5, scoring="r2"
        )
        rf_scores = cross_val_score(
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            X, y, cv=5, scoring="r2"
        )

        r2_ridge = np.mean(ridge_scores)
        r2_rf = np.mean(rf_scores)
        results.append({
            "target": target,
            "r2_ridge": r2_ridge,
            "r2_rf": r2_rf,
            "r2_best": max(r2_ridge, r2_rf),
        })
        print(f"  {target:<20} {r2_ridge:>+.4f}        {r2_rf:>+.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 4. PER-PC CLASSIFICATION (binarized at median)
# ══════════════════════════════════════════════════════════════════════════

def experiment_classification(df: pd.DataFrame, targets: list[str], target_label: str):
    """Binarize each target at its median, train RF classifier."""
    print(f"\n{'='*72}")
    print(f"CLASSIFICATION (median split): Utterance features -> {target_label}")
    print(f"{'='*72}")

    X = df[UTT_FEATURES].values
    results = []

    print(f"\n  {'Target':<20} {'Baseline':<10} {'Accuracy':<10} {'Lift':<8}")
    print(f"  {'-'*48}")

    for target in targets:
        y_raw = df[target].values
        median = np.median(y_raw)
        y_bin = (y_raw >= median).astype(int)
        baseline = max(y_bin.mean(), 1 - y_bin.mean())

        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        y_pred = cross_val_predict(clf, X, y_bin, cv=5)
        acc = accuracy_score(y_bin, y_pred)
        lift = acc / baseline

        results.append({
            "target": target,
            "baseline": baseline,
            "accuracy": acc,
            "lift": lift,
        })
        print(f"  {target:<20} {baseline:<10.3f} {acc:<10.3f} {lift:<8.2f}x")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE PER PC
# ══════════════════════════════════════════════════════════════════════════

def experiment_feature_importance(df: pd.DataFrame, targets: list[str], target_label: str):
    """Fit RF regressor per target, extract and report feature importances."""
    print(f"\n{'='*72}")
    print(f"FEATURE IMPORTANCE: Which utterance features predict each {target_label}?")
    print(f"{'='*72}")

    X = df[UTT_FEATURES].values
    all_importances = {}

    for target in targets:
        y = df[target].values
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        rf.fit(X, y)
        all_importances[target] = rf.feature_importances_

    # Print top-3 features per target
    print(f"\n  Top-3 predictive features per {target_label}:")
    print(f"  {'-'*60}")
    for target in targets:
        imp = all_importances[target]
        order = np.argsort(-imp)
        top3 = [(UTT_FEATURES[i], imp[i]) for i in order[:3]]
        desc = ", ".join(f"{name} ({val:.3f})" for name, val in top3)
        print(f"  {target:<12} {desc}")

    # Full importance matrix
    print(f"\n  Full importance matrix:")
    header = f"  {'Feature':<22}" + "".join(f"{t:>10}" for t in targets)
    print(header)
    print(f"  {'-'*(22 + 10*len(targets))}")
    for i, feat in enumerate(UTT_FEATURES):
        row = f"  {feat:<22}"
        for target in targets:
            row += f"{all_importances[target][i]:>10.3f}"
        print(row)

    return all_importances


# ══════════════════════════════════════════════════════════════════════════
# 6. COMPARATIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def comparative_summary(
    pc_reg_results, pc_cls_results,
    raw_reg_results, raw_cls_results,
    eigenvalues, eigenvectors
):
    total_var = eigenvalues.sum()

    print(f"\n{'='*72}")
    print(f"COMPARATIVE SUMMARY: PC vs Raw Dimension Recoverability")
    print(f"{'='*72}")

    # PC recovery ranked
    print(f"\n  PCs ranked by text recoverability (best R2):")
    print(f"  {'PC':<12} {'%Var':<8} {'R2 (best)':<12} {'Clf Lift':<10} {'Verdict'}")
    print(f"  {'-'*60}")
    pc_sorted = sorted(pc_reg_results, key=lambda x: -x["r2_best"])
    for rr in pc_sorted:
        target = rr["target"]
        idx = int(target[2:]) - 1
        pct_var = eigenvalues[idx] / total_var * 100
        cls = next(c for c in pc_cls_results if c["target"] == target)
        verdict = "RECOVERABLE" if cls["lift"] >= 1.10 else "WEAK" if cls["lift"] >= 1.03 else "NOT RECOVERABLE"
        print(f"  {target:<12} {pct_var:<8.1f} {rr['r2_best']:<12.4f} {cls['lift']:<10.2f}x {verdict}")

    # Raw dimension recovery ranked
    print(f"\n  Raw dimensions ranked by text recoverability (best R2):")
    print(f"  {'Dim':<16} {'R2 (best)':<12} {'Clf Lift':<10}")
    print(f"  {'-'*40}")
    raw_sorted = sorted(raw_reg_results, key=lambda x: -x["r2_best"])
    for rr in raw_sorted:
        cls = next(c for c in raw_cls_results if c["target"] == rr["target"])
        print(f"  {rr['target']:<16} {rr['r2_best']:<12.4f} {cls['lift']:<10.2f}x")

    # Direct comparison: mean R2 for PCs vs raw
    pc_mean_r2 = np.mean([r["r2_best"] for r in pc_reg_results])
    raw_mean_r2 = np.mean([r["r2_best"] for r in raw_reg_results])
    pc_mean_lift = np.mean([c["lift"] for c in pc_cls_results])
    raw_mean_lift = np.mean([c["lift"] for c in raw_cls_results])

    print(f"\n  Aggregate comparison:")
    print(f"  {'Metric':<30} {'PCs':<12} {'Raw dims':<12}")
    print(f"  {'-'*54}")
    print(f"  {'Mean best R2':<30} {pc_mean_r2:<12.4f} {raw_mean_r2:<12.4f}")
    print(f"  {'Mean classification lift':<30} {pc_mean_lift:<12.2f}x {raw_mean_lift:<12.2f}x")

    # Key question: are top-3 PCs more or less recoverable than PC4-5?
    top3_r2 = np.mean([r["r2_best"] for r in pc_reg_results if r["target"] in ["PC1", "PC2", "PC3"]])
    bot2_r2 = np.mean([r["r2_best"] for r in pc_reg_results if r["target"] in ["PC4", "PC5"]])
    top3_lift = np.mean([c["lift"] for c in pc_cls_results if c["target"] in ["PC1", "PC2", "PC3"]])
    bot2_lift = np.mean([c["lift"] for c in pc_cls_results if c["target"] in ["PC4", "PC5"]])

    print(f"\n  Key question: Top-3 PCs (keep) vs PC4-5 (drop):")
    print(f"  {'Metric':<30} {'PC1-3 (keep)':<15} {'PC4-5 (drop)':<15}")
    print(f"  {'-'*60}")
    print(f"  {'Mean best R2':<30} {top3_r2:<15.4f} {bot2_r2:<15.4f}")
    print(f"  {'Mean classification lift':<30} {top3_lift:<15.2f}x {bot2_lift:<15.2f}x")

    # Eigenvector loadings for reference
    print(f"\n  Eigenvector loadings (for interpretation):")
    print(f"  {'PC':<8}" + "".join(f"{d:>14}" for d in AFFECT_DIMS))
    print(f"  {'-'*(8 + 14*5)}")
    for i in range(5):
        row = f"  PC{i+1:<5}"
        for j in range(5):
            row += f"{eigenvectors[j, i]:>+14.3f}"
        print(row)

    # Final interpretation
    print(f"\n{'='*72}")
    print(f"IMPLICATIONS FOR THE EIGENSPACE MODEL")
    print(f"{'='*72}")

    if top3_r2 > bot2_r2:
        print(f"""
  The top-3 PCs (91.6% of transition variance) are MORE recoverable
  from text than PC4-5. This is favorable: the emission model can
  provide direct evidence for the primary eigenaxes, and transition
  dynamics complement rather than substitute for text-based inference.
""")
    else:
        print(f"""
  The top-3 PCs (91.6% of transition variance) are LESS recoverable
  from text than PC4-5. This means the emission model provides better
  evidence for the minor axes, while the dominant axes must be inferred
  primarily from transition dynamics. The factor graph's transition
  model carries the load for the dimensions that matter most.
""")

    if pc_mean_r2 > raw_mean_r2:
        print(f"  PCs are more recoverable than raw dimensions (R2: {pc_mean_r2:.4f} vs {raw_mean_r2:.4f}).")
        print(f"  The eigenrotation concentrates text-predictable variance.")
    else:
        print(f"  Raw dimensions are more recoverable than PCs (R2: {raw_mean_r2:.4f} vs {pc_mean_r2:.4f}).")
        print(f"  The eigenrotation disperses text-predictable variance; raw dims")
        print(f"  may be better emission targets even if PCs are better transition targets.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  PC RECOVERY FROM UTTERANCE TEXT FEATURES")
    print("  Which eigenaxes can the emission model see?")
    print("=" * 72)

    # Step 1: Compute eigenvectors
    print(f"\n{'─'*72}")
    print("1. EIGENDECOMPOSITION (recomputed)")
    print(f"{'─'*72}")
    eigenvalues, eigenvectors, cov = compute_eigenvectors()
    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var

    for i in range(5):
        pct = eigenvalues[i] / total_var * 100
        print(f"  PC{i+1}: eigenvalue={eigenvalues[i]:.4f}  ({pct:.1f}% var, cumul {cum_var[i]*100:.1f}%)")

    # Step 2: Build dataset
    print(f"\n{'─'*72}")
    print("2. DATASET")
    print(f"{'─'*72}")
    df = build_dataset(eigenvectors)
    print(f"  Observations (beat, character): {len(df)}")
    print(f"  Plays: {df['play_id'].value_counts().to_dict()}")

    # Quick stats on PC scores
    pc_cols = [f"PC{i+1}" for i in range(5)]
    print(f"\n  PC score summary:")
    print(df[pc_cols].describe().round(3).to_string())

    print(f"\n  Utterance feature summary:")
    print(df[UTT_FEATURES].describe().round(3).to_string())

    # Step 3: Per-PC regression
    pc_reg = experiment_regression(df, pc_cols, "PC scores")

    # Step 4: Per-PC classification
    pc_cls = experiment_classification(df, pc_cols, "PC high/low")

    # Step 5: Feature importance per PC
    pc_imp = experiment_feature_importance(df, pc_cols, "PC")

    # Step 6+7: Raw dimension regression + classification for comparison
    raw_reg = experiment_regression(df, AFFECT_DIMS, "raw affect dimensions")
    raw_cls = experiment_classification(df, AFFECT_DIMS, "raw affect high/low")
    raw_imp = experiment_feature_importance(df, AFFECT_DIMS, "raw dimension")

    # Step 6: Comparative summary
    comparative_summary(pc_reg, pc_cls, raw_reg, raw_cls, eigenvalues, eigenvectors)


if __name__ == "__main__":
    main()
