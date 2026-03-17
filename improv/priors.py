"""
Statistical Priors — Phase B

Loads learned priors from corpus analysis for use during improvisation:
  - Canonical tactic vocabulary
  - Per-character tactic prior P(tactic | character)
  - Tactic transition matrix P(next_tactic | current_tactic)
  - Relational profile (directed social tendencies)

Also generates graduated dramaturgical feedback based on deviations
from these priors. Priors act as a CRITIC, not a constraint — the LLM
generates freely, and the priors evaluate consistency after the fact.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    BeatState, CharacterBible, Play, RelationalProfile,
    StatisticalPrior, TacticVocabulary,
)


# ────────────────────────────────────────────────────────────────────────────
# Load priors from disk
# ────────────────────────────────────────────────────────────────────────────

def load_prior_for_character(
    play_id: str,
    character: str,
) -> StatisticalPrior:
    """Load a full StatisticalPrior for a character from persisted data.

    Combines: tactic vocabulary, character tactic prior, transition matrix,
    and relational profile.
    """
    from analysis.vocabulary import load_vocabulary
    from analysis.relationship_builder import load_profiles

    vocab = load_vocabulary()
    profiles = load_profiles(play_id)
    profile = next((p for p in profiles if p.character == character), None)

    # Build character-specific tactic prior and transition matrix from play data
    from config import PARSED_DIR
    play = Play.model_validate_json(
        (PARSED_DIR / f"{play_id}.json").read_text()
    )

    char_prior, transition_matrix = _compute_tactic_stats(play, character, vocab)

    return StatisticalPrior(
        tactic_vocabulary=vocab,
        character_tactic_prior=char_prior,
        tactic_transition_matrix=transition_matrix,
        relational_profile=profile,
    )


def _compute_tactic_stats(
    play: Play,
    character: str,
    vocab: TacticVocabulary,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute tactic prior and transition matrix for a character.

    Returns:
        (character_tactic_prior, tactic_transition_matrix)
        Both use canonical tactic IDs as keys.
    """
    # Collect the character's canonical tactic sequence (ordered by beat)
    tactic_sequence: list[str] = []
    for act in play.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                for bs in beat.beat_states:
                    if bs.character == character and bs.canonical_tactic:
                        tactic_sequence.append(bs.canonical_tactic)

    # Character tactic prior: P(tactic | character)
    tactic_counts = Counter(tactic_sequence)
    total = sum(tactic_counts.values())
    char_prior: dict[str, float] = {}
    if total > 0:
        for tactic, count in tactic_counts.items():
            char_prior[tactic] = round(count / total, 4)

    # Tactic transition matrix: P(next | current) — from ALL characters in the play
    # (using the full corpus gives better statistics than one character alone)
    all_sequences: list[list[str]] = []
    char_seqs: dict[str, list[str]] = defaultdict(list)
    for act in play.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                for bs in beat.beat_states:
                    if bs.canonical_tactic:
                        char_seqs[bs.character].append(bs.canonical_tactic)
    all_sequences = list(char_seqs.values())

    transition_counts: dict[str, Counter] = defaultdict(Counter)
    for seq in all_sequences:
        for i in range(len(seq) - 1):
            transition_counts[seq[i]][seq[i + 1]] += 1

    transition_matrix: dict[str, dict[str, float]] = {}
    for current, next_counts in transition_counts.items():
        total_from = sum(next_counts.values())
        if total_from > 0:
            transition_matrix[current] = {
                next_t: round(count / total_from, 4)
                for next_t, count in next_counts.items()
            }

    return char_prior, transition_matrix


# ────────────────────────────────────────────────────────────────────────────
# Deviation analysis
# ────────────────────────────────────────────────────────────────────────────

def compute_tactic_deviation(
    generated_tactic: str | None,
    prior: StatisticalPrior,
    previous_tactic: str | None = None,
) -> dict:
    """Analyze how much a generated tactic deviates from the character's priors.

    Returns a dict with:
        canonical_id: the canonical tactic ID (or None if unmapped)
        prior_probability: P(tactic | character)
        transition_probability: P(tactic | previous_tactic) if applicable
        deviation_tier: 1 (on target), 2 (mild), or 3 (sharp)
        top_tactic: the character's most common tactic
        top_tactic_pct: percentage of time they use it
    """
    vocab = prior.tactic_vocabulary
    canonical = vocab.lookup(generated_tactic) if generated_tactic else None

    # Character's top tactic
    top_tactic = None
    top_pct = 0.0
    if prior.character_tactic_prior:
        top_tactic = max(prior.character_tactic_prior, key=prior.character_tactic_prior.get)
        top_pct = prior.character_tactic_prior[top_tactic]

    result = {
        "canonical_id": canonical,
        "prior_probability": 0.0,
        "transition_probability": None,
        "deviation_tier": 1,
        "top_tactic": top_tactic,
        "top_tactic_pct": round(top_pct * 100, 1),
    }

    if canonical is None:
        # Unmapped tactic — treat as mild deviation (we don't know enough)
        result["deviation_tier"] = 2
        return result

    # Prior probability
    prior_p = prior.character_tactic_prior.get(canonical, 0.0)
    result["prior_probability"] = prior_p

    # Transition probability
    if previous_tactic:
        prev_canonical = vocab.lookup(previous_tactic) or previous_tactic
        trans_dist = prior.tactic_transition_matrix.get(prev_canonical, {})
        trans_p = trans_dist.get(canonical, 0.0)
        result["transition_probability"] = trans_p

    # Determine tier
    if prior_p >= 0.05:
        result["deviation_tier"] = 1  # on target
    elif prior_p >= 0.02:
        result["deviation_tier"] = 2  # mild deviation
    else:
        result["deviation_tier"] = 3  # sharp deviation

    # Transition-based escalation: a low-probability transition bumps up the tier
    if result["transition_probability"] is not None and result["transition_probability"] < 0.05:
        result["deviation_tier"] = max(result["deviation_tier"], 2)

    return result


def compute_affect_deviation(
    beat_state: BeatState,
    bible: CharacterBible,
    play: Play,
) -> dict:
    """Analyze how a beat state's affect deviates from the character's typical range.

    Returns dict with per-dimension analysis and an overall tier.
    """
    # Collect the character's historical affect values
    valences, arousals, vulnerabilities = [], [], []
    for act in play.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                for bs in beat.beat_states:
                    if bs.character == beat_state.character:
                        valences.append(bs.affect_state.valence)
                        arousals.append(bs.affect_state.arousal)
                        vulnerabilities.append(bs.affect_state.vulnerability)

    if len(valences) < 5:
        return {"deviation_tier": 1, "details": {}}

    from statistics import mean, stdev

    def _analyze_dim(name: str, current: float, history: list[float]) -> dict:
        mu = mean(history)
        sd = stdev(history) if len(history) > 1 else 0.3
        z = abs(current - mu) / sd if sd > 0 else 0.0
        return {
            "current": round(current, 2),
            "mean": round(mu, 2),
            "std": round(sd, 2),
            "z_score": round(z, 2),
        }

    details = {
        "valence": _analyze_dim("valence", beat_state.affect_state.valence, valences),
        "arousal": _analyze_dim("arousal", beat_state.affect_state.arousal, arousals),
        "vulnerability": _analyze_dim("vulnerability", beat_state.affect_state.vulnerability, vulnerabilities),
    }

    # Overall tier based on max z-score
    max_z = max(d["z_score"] for d in details.values())
    if max_z < 1.5:
        tier = 1
    elif max_z < 2.5:
        tier = 2
    else:
        tier = 3

    return {"deviation_tier": tier, "details": details}


# ────────────────────────────────────────────────────────────────────────────
# Graduated dramaturgical feedback
# ────────────────────────────────────────────────────────────────────────────

def generate_dramaturgical_feedback(
    beat_state: BeatState,
    bible: CharacterBible,
    prior: StatisticalPrior,
    previous_tactic: str | None = None,
    play: Play | None = None,
) -> list[str]:
    """Generate director's-note style feedback based on statistical deviations.

    Feedback is graduated in intensity:
        Tier 1 (on target): encouragement + minor polish
        Tier 2 (mild deviation): name the pattern, suggest reconsideration
        Tier 3 (sharp deviation): demand dramatic justification

    Templates reference the character's specific patterns by name.

    Returns: list of feedback strings to append to scorer feedback.
    """
    character = bible.character
    feedback: list[str] = []

    # ── Tactic deviation ──
    tactic_dev = compute_tactic_deviation(
        beat_state.tactic_state, prior, previous_tactic
    )
    tier = tactic_dev["deviation_tier"]
    canonical = tactic_dev["canonical_id"]
    top_tactic = tactic_dev["top_tactic"]
    top_pct = tactic_dev["top_tactic_pct"]
    top_verb = None
    if top_tactic:
        ct = next((t for t in prior.tactic_vocabulary.tactics
                    if t.canonical_id == top_tactic), None)
        top_verb = ct.canonical_verb if ct else top_tactic.lower()

    gen_verb = beat_state.tactic_state or "the chosen tactic"

    if tier == 1 and top_verb:
        # Encouragement — even on-target lines benefit from a polish note
        if canonical == top_tactic:
            feedback.append(
                f"Good instinct — the {top_verb} feels characteristic of {character}. "
                f"See if you can let the subtext breathe a little more through "
                f"the rhythm of the line."
            )
        else:
            feedback.append(
                f"The tactic lands. Consider whether the speech pattern "
                f"could lean harder into {character}'s characteristic "
                f"{', '.join(bible.rhetorical_patterns[:2]) or 'voice'}."
            )
    elif tier == 2 and top_verb:
        feedback.append(
            f"{character} typically operates through {top_verb} in moments like this "
            f"— the shift to {gen_verb} is noticeable. "
            f"If this is a genuine departure, let the audience feel the cost of that shift."
        )
    elif tier == 3 and top_verb:
        feedback.append(
            f"This is a significant break from {character}'s established pattern. "
            f"{character} has used {top_verb} in {top_pct:.0f}% of comparable moments "
            f"— jumping to {gen_verb} needs a very strong provocation. "
            f"What in this specific moment forces {character} out of their comfort zone? "
            f"If you can't point to it, return to {top_verb}."
        )

    # ── Affect deviation (if we have the play data) ──
    if play is not None:
        affect_dev = compute_affect_deviation(beat_state, bible, play)
        affect_tier = affect_dev["deviation_tier"]
        details = affect_dev.get("details", {})

        if affect_tier == 2:
            # Find the most deviant dimension
            worst = max(details.items(), key=lambda x: x[1]["z_score"])
            dim_name, dim_data = worst
            feedback.append(
                f"The {dim_name} here ({dim_data['current']:.1f}) runs "
                f"{'higher' if dim_data['current'] > dim_data['mean'] else 'lower'} "
                f"than {character} usually allows "
                f"(typical range: {dim_data['mean']:.1f} ± {dim_data['std']:.1f}). "
                f"If the armor is cracking, something specific should be cracking it."
            )
        elif affect_tier == 3:
            worst = max(details.items(), key=lambda x: x[1]["z_score"])
            dim_name, dim_data = worst
            feedback.append(
                f"The affect state has shifted dramatically from {character}'s baseline. "
                f"{dim_name.capitalize()} at {dim_data['current']:.1f} is "
                f"{dim_data['z_score']:.1f} standard deviations from typical "
                f"({dim_data['mean']:.1f}). "
                f"This kind of exposure would need to be earned beat by beat, "
                f"not arrived at in one jump."
            )

    return feedback
