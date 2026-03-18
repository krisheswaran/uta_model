#!/usr/bin/env python3
"""
Emission Model Experiment: Does BeatState predict observable utterance features?

Tests whether the 6-axis hidden state (affect + social + tactic) can predict
surface-level text features, establishing the viability of an emission model
P(utterance_features | BeatState) for the factor graph.

Sections:
  1. Extract utterance features per (beat, character)
  2. BeatState -> utterance feature prediction (R^2)
  3. Tactic -> utterance feature associations
  4. Invertibility: utterance features -> tactic / affect quadrant
  5. Feature importance ranking
"""

import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mutual_info_score,
    r2_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "parsed"
VOCAB_PATH = ROOT / "data" / "vocab" / "tactic_vocabulary.json"
PLAY_IDS = ["hamlet", "cherry_orchard", "importance_of_being_earnest"]

# ── load tactic vocabulary for Earnest fallback ────────────────────────────────
with open(VOCAB_PATH) as f:
    tactic_vocab = json.load(f)

# Build member -> canonical_id mapping
_member_to_canonical: dict[str, str] = {}
for entry in tactic_vocab["tactics"]:
    cid = entry["canonical_id"]
    for m in entry["members"]:
        _member_to_canonical[m.lower().strip()] = cid


def resolve_tactic(beat_state: dict) -> str | None:
    """Return canonical tactic, falling back to tactic_state lookup for Earnest."""
    ct = beat_state.get("canonical_tactic")
    if ct:
        return ct
    ts = beat_state.get("tactic_state")
    if ts:
        key = ts.lower().strip()
        if key.upper() in {e["canonical_id"] for e in tactic_vocab["tactics"]}:
            return key.upper()
        return _member_to_canonical.get(key)
    return None


# ── sentiment analyzer ─────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()

# ── imperative seed words ──────────────────────────────────────────────────────
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
# 1. EXTRACT UTTERANCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> list[str]:
    """Split text into sentences (approximate)."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]


def compute_utterance_features(texts: list[str]) -> dict:
    """Compute text features from a list of utterance texts for one character in one beat."""
    all_text = " ".join(texts)
    words = re.findall(r"[A-Za-z']+", all_text)
    word_count = len(words)
    if word_count == 0:
        return None  # skip empty

    sentences = []
    for t in texts:
        sentences.extend(split_sentences(t))
    n_sent = max(len(sentences), 1)

    # Question / exclamation density
    q_count = sum(1 for s in sentences if s.rstrip().endswith("?"))
    exc_count = sum(1 for s in sentences if s.rstrip().endswith("!"))

    # Imperative density
    imp_count = 0
    for s in sentences:
        first_word = re.match(r"[A-Za-z']+", s.strip())
        if first_word and first_word.group().lower() in IMPERATIVE_STARTERS:
            imp_count += 1

    # Mean sentence length
    sent_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
    mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0

    # Lexical diversity (type-token ratio)
    words_lower = [w.lower() for w in words]
    lex_div = len(set(words_lower)) / len(words_lower)

    # First/second person rates
    first_person = {"i", "me", "my", "mine", "myself"}
    second_person = {"you", "your", "yours", "yourself", "yourselves"}
    fp_count = sum(1 for w in words_lower if w in first_person)
    sp_count = sum(1 for w in words_lower if w in second_person)

    # VADER sentiment
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


# ── build dataset ──────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    rows = []
    for play_id in PLAY_IDS:
        path = DATA_DIR / f"{play_id}.json"
        with open(path) as f:
            play = json.load(f)
        for act in play["acts"]:
            for scene in act["scenes"]:
                for beat in scene["beats"]:
                    # group utterances by speaker
                    utt_by_speaker: dict[str, list[str]] = defaultdict(list)
                    for u in beat["utterances"]:
                        utt_by_speaker[u["speaker"]].append(u["text"])

                    for bs in beat["beat_states"]:
                        char = bs["character"]
                        texts = utt_by_speaker.get(char, [])
                        if not texts:
                            continue
                        feats = compute_utterance_features(texts)
                        if feats is None:
                            continue

                        tactic = resolve_tactic(bs)
                        aff = bs["affect_state"]
                        soc = bs["social_state"]

                        row = {
                            "play_id": play_id,
                            "beat_id": bs["beat_id"],
                            "character": char,
                            "canonical_tactic": tactic,
                            # affect dimensions
                            "valence": aff["valence"],
                            "arousal": aff["arousal"],
                            "certainty": aff["certainty"],
                            "control": aff["control"],
                            "vulnerability": aff["vulnerability"],
                            # social dimensions
                            "status": soc["status"],
                            "warmth": soc["warmth"],
                        }
                        row.update(feats)
                        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE / STATE COLUMN DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

UTT_FEATURES = [
    "word_count", "question_density", "exclamation_density",
    "imperative_density", "mean_sentence_length", "lexical_diversity",
    "first_person_rate", "second_person_rate", "sentiment_polarity",
]

STATE_DIMS = [
    "valence", "arousal", "certainty", "control", "vulnerability",
    "status", "warmth",
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. BeatState -> utterance feature prediction
# ══════════════════════════════════════════════════════════════════════════════

def experiment_state_to_features(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: BeatState dimensions -> utterance feature prediction")
    print("=" * 72)

    # Build predictor matrix: 7 continuous dims + tactic one-hot
    df_valid = df.dropna(subset=["canonical_tactic"]).copy()
    tactic_dummies = pd.get_dummies(df_valid["canonical_tactic"], prefix="tac")
    X_full = pd.concat([df_valid[STATE_DIMS].reset_index(drop=True),
                        tactic_dummies.reset_index(drop=True)], axis=1)
    X_cont = df_valid[STATE_DIMS].values  # continuous only

    print(f"\nDataset: {len(df_valid)} (beat, character) observations")
    print(f"Tactics: {df_valid['canonical_tactic'].nunique()} unique")
    print(f"Full predictor dims: {X_full.shape[1]} (7 continuous + tactic one-hot)")

    results = []
    print(f"\n{'Feature':<25} {'R2 (Ridge, cont)':<20} {'R2 (RF, full)':>15}")
    print("-" * 60)

    for feat in UTT_FEATURES:
        y = df_valid[feat].values

        # Ridge with continuous state dims only
        ridge_scores = cross_val_score(
            RidgeCV(alphas=[0.01, 0.1, 1, 10]),
            X_cont, y, cv=5, scoring="r2"
        )

        # Random forest with full features (continuous + tactic one-hot)
        rf_scores = cross_val_score(
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            X_full.values, y, cv=5, scoring="r2"
        )

        r2_ridge = np.mean(ridge_scores)
        r2_rf = np.mean(rf_scores)
        results.append((feat, r2_ridge, r2_rf))
        print(f"{feat:<25} {r2_ridge:>+.3f}             {r2_rf:>+.3f}")

    # Per-dimension correlations
    print("\n\nPer-dimension Pearson correlations (|r| > 0.10 shown):")
    print(f"{'State dim':<16}", end="")
    for f in UTT_FEATURES:
        print(f"{f[:12]:<14}", end="")
    print()
    print("-" * (16 + 14 * len(UTT_FEATURES)))

    for dim in STATE_DIMS:
        print(f"{dim:<16}", end="")
        for feat in UTT_FEATURES:
            r = np.corrcoef(df_valid[dim].values, df_valid[feat].values)[0, 1]
            marker = f"{r:+.2f}" if abs(r) > 0.10 else "   . "
            print(f"{marker:<14}", end="")
        print()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Tactic -> utterance feature associations
# ══════════════════════════════════════════════════════════════════════════════

def experiment_tactic_profiles(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Tactic -> utterance feature profiles")
    print("=" * 72)

    df_valid = df.dropna(subset=["canonical_tactic"]).copy()
    global_means = df_valid[UTT_FEATURES].mean()
    global_stds = df_valid[UTT_FEATURES].std()

    # Only tactics with >= 5 observations
    tactic_counts = df_valid["canonical_tactic"].value_counts()
    good_tactics = tactic_counts[tactic_counts >= 5].index.tolist()

    print(f"\nTactics with >= 5 observations: {len(good_tactics)}")
    print(f"Total observations with tactic: {len(df_valid)}")

    profiles = []
    for tac in good_tactics:
        mask = df_valid["canonical_tactic"] == tac
        means = df_valid.loc[mask, UTT_FEATURES].mean()
        z_scores = (means - global_means) / global_stds
        profiles.append((tac, tactic_counts[tac], z_scores))

    # Print distinctive associations (|z| > 0.5)
    print(f"\nDistinctive associations (|z-score| > 0.5 from global mean):")
    print(f"{'Tactic':<16} {'n':>4}  Distinctive features")
    print("-" * 72)
    for tac, n, z in sorted(profiles, key=lambda x: -x[1]):
        distinctive = [(f, z[f]) for f in UTT_FEATURES if abs(z[f]) > 0.5]
        if distinctive:
            desc = ", ".join(f"{f}={v:+.2f}z" for f, v in
                           sorted(distinctive, key=lambda x: -abs(x[1])))
            print(f"{tac:<16} {n:>4}  {desc}")

    # Top associations table
    print(f"\n\nFull z-score matrix (tactics with n >= 8):")
    big_tactics = [t for t, n, _ in profiles if n >= 8]
    if big_tactics:
        print(f"{'Tactic':<16}", end="")
        for f in UTT_FEATURES:
            print(f"{f[:11]:<13}", end="")
        print()
        print("-" * (16 + 13 * len(UTT_FEATURES)))
        for tac, n, z in profiles:
            if tac not in big_tactics:
                continue
            print(f"{tac:<16}", end="")
            for f in UTT_FEATURES:
                val = z[f]
                marker = f"{val:+.2f}" if abs(val) > 0.3 else "  .  "
                print(f"{marker:<13}", end="")
            print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Invertibility: utterance features -> hidden state
# ══════════════════════════════════════════════════════════════════════════════

def experiment_invertibility(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Invertibility — utterance features -> hidden state")
    print("=" * 72)

    df_valid = df.dropna(subset=["canonical_tactic"]).copy()
    X = df_valid[UTT_FEATURES].values

    # ── 4a. Predict canonical tactic ──────────────────────────────────────────
    # Only keep tactics with >= 5 observations for classification
    tactic_counts = df_valid["canonical_tactic"].value_counts()
    keep = tactic_counts[tactic_counts >= 5].index
    mask = df_valid["canonical_tactic"].isin(keep)
    X_tac = df_valid.loc[mask, UTT_FEATURES].values
    y_tac = df_valid.loc[mask, "canonical_tactic"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_tac)

    n_classes = len(le.classes_)
    baseline_acc = max(np.bincount(y_enc)) / len(y_enc)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    y_pred = cross_val_predict(clf, X_tac, y_enc, cv=5)
    tac_acc = accuracy_score(y_enc, y_pred)

    print(f"\n4a. Tactic classification from utterance features")
    print(f"    Classes: {n_classes} tactics (>= 5 obs each)")
    print(f"    Baseline (majority): {baseline_acc:.3f}")
    print(f"    RF accuracy (5-fold): {tac_acc:.3f}")
    print(f"    Lift over baseline:   {tac_acc / baseline_acc:.2f}x")

    # Top-3 accuracy
    clf.fit(X_tac, y_enc)  # for probability estimates
    y_proba = cross_val_predict(clf, X_tac, y_enc, cv=5, method="predict_proba")
    top3_correct = 0
    for i, true_label in enumerate(y_enc):
        top3 = np.argsort(y_proba[i])[-3:]
        if true_label in top3:
            top3_correct += 1
    top3_acc = top3_correct / len(y_enc)
    print(f"    Top-3 accuracy:       {top3_acc:.3f}")

    # ── 4b. Predict affect quadrant ───────────────────────────────────────────
    # high/low valence x high/low arousal -> 4 quadrants
    med_v = df_valid["valence"].median()
    med_a = df_valid["arousal"].median()
    quadrant_labels = []
    for _, row in df_valid.iterrows():
        v = "hi_val" if row["valence"] >= med_v else "lo_val"
        a = "hi_aro" if row["arousal"] >= med_a else "lo_aro"
        quadrant_labels.append(f"{v}_{a}")
    y_quad = np.array(quadrant_labels)

    le_q = LabelEncoder()
    y_q_enc = le_q.fit_transform(y_quad)
    baseline_q = max(np.bincount(y_q_enc)) / len(y_q_enc)

    clf_q = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    y_q_pred = cross_val_predict(clf_q, X, y_q_enc, cv=5)
    quad_acc = accuracy_score(y_q_enc, y_q_pred)

    print(f"\n4b. Affect quadrant classification (valence x arousal)")
    print(f"    Classes: {le_q.classes_}")
    print(f"    Baseline (majority): {baseline_q:.3f}")
    print(f"    RF accuracy (5-fold): {quad_acc:.3f}")
    print(f"    Lift over baseline:   {quad_acc / baseline_q:.2f}x")

    # ── 4c. Predict individual affect dimensions (binarized) ──────────────────
    print(f"\n4c. Individual state dimension prediction (binarized at median)")
    print(f"    {'Dimension':<16} {'Baseline':>10} {'Accuracy':>10} {'Lift':>8}")
    print("    " + "-" * 46)
    for dim in STATE_DIMS:
        med = df_valid[dim].median()
        y_bin = (df_valid[dim].values >= med).astype(int)
        bl = max(y_bin.mean(), 1 - y_bin.mean())
        clf_d = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        y_d_pred = cross_val_predict(clf_d, X, y_bin, cv=5)
        acc = accuracy_score(y_bin, y_d_pred)
        print(f"    {dim:<16} {bl:>10.3f} {acc:>10.3f} {acc/bl:>8.2f}x")

    return tac_acc, baseline_acc, quad_acc, baseline_q


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature importance
# ══════════════════════════════════════════════════════════════════════════════

def experiment_feature_importance(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: Feature importance for hidden state prediction")
    print("=" * 72)

    df_valid = df.dropna(subset=["canonical_tactic"]).copy()
    X = df_valid[UTT_FEATURES].values

    # Feature importance for tactic classification
    tactic_counts = df_valid["canonical_tactic"].value_counts()
    keep = tactic_counts[tactic_counts >= 5].index
    mask = df_valid["canonical_tactic"].isin(keep)
    X_tac = df_valid.loc[mask, UTT_FEATURES].values
    y_tac = LabelEncoder().fit_transform(df_valid.loc[mask, "canonical_tactic"].values)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_tac, y_tac)
    imp_tactic = clf.feature_importances_

    # Feature importance for affect quadrant
    med_v = df_valid["valence"].median()
    med_a = df_valid["arousal"].median()
    y_quad = LabelEncoder().fit_transform([
        f"{'hi' if r['valence'] >= med_v else 'lo'}_{'hi' if r['arousal'] >= med_a else 'lo'}"
        for _, r in df_valid.iterrows()
    ])
    clf_q = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_q.fit(X, y_quad)
    imp_quad = clf_q.feature_importances_

    # Feature importance for each continuous state dimension (regression)
    imp_by_dim = {}
    for dim in STATE_DIMS:
        y_d = df_valid[dim].values
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X, y_d)
        imp_by_dim[dim] = rf.feature_importances_

    print(f"\nRandom Forest feature importances (higher = more predictive)")
    print(f"\n{'Feature':<22} {'Tactic':>8} {'Aff.Quad':>8}", end="")
    for dim in STATE_DIMS:
        print(f" {dim[:7]:>8}", end="")
    print()
    print("-" * (22 + 8 + 8 + 8 * len(STATE_DIMS)))

    for i, feat in enumerate(UTT_FEATURES):
        print(f"{feat:<22} {imp_tactic[i]:>8.3f} {imp_quad[i]:>8.3f}", end="")
        for dim in STATE_DIMS:
            print(f" {imp_by_dim[dim][i]:>8.3f}", end="")
        print()

    # Overall ranking (mean importance across all tasks)
    all_imps = np.column_stack([imp_tactic, imp_quad] + [imp_by_dim[d] for d in STATE_DIMS])
    mean_imp = all_imps.mean(axis=1)
    order = np.argsort(-mean_imp)

    print(f"\nOverall feature ranking (mean importance across all prediction tasks):")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank}. {UTT_FEATURES[idx]:<25} {mean_imp[idx]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("EMISSION MODEL EXPERIMENT")
    print("Does BeatState predict observable utterance features?")
    print("=" * 72)

    # Build dataset
    df = build_dataset()
    print(f"\nDataset built: {len(df)} (beat, character) observations")
    print(f"Plays: {df['play_id'].value_counts().to_dict()}")
    print(f"Unique tactics: {df['canonical_tactic'].nunique()} "
          f"({df['canonical_tactic'].isna().sum()} missing)")

    # Summary stats
    print(f"\nUtterance feature summary:")
    print(df[UTT_FEATURES].describe().round(3).to_string())

    # Run experiments
    experiment_state_to_features(df)
    experiment_tactic_profiles(df)
    experiment_invertibility(df)
    experiment_feature_importance(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY & IMPLICATIONS FOR FACTOR GRAPH")
    print("=" * 72)
    print("""
Key questions answered:
  1. Forward model P(features|state): Which state dimensions create
     observable text signatures? (Experiment 2)
  2. Emission patterns: What text features are associated with each
     tactic? (Experiment 3)
  3. Invertibility: Can we recover hidden state from text features
     alone? (Experiment 4)
  4. Feature selection: Which text features carry the most signal
     about hidden state? (Experiment 5)

Implications:
  - High R^2 in Exp 2 => the emission model is informative
  - High invertibility in Exp 4 => observations constrain the posterior
  - Low invertibility => factor graph relies on transition priors
  - Distinctive tactic profiles in Exp 3 => the emission distributions
    have separation between states
""")


if __name__ == "__main__":
    main()
