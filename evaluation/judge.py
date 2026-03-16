"""
LLM-as-Judge Evaluator

Implements the three-tier evaluation protocol:
  - Tier 1: Vanilla LLM (zero-shot character prompt)
  - Tier 2: LLM + text-only character bible (system prompt enriched)
  - Tier 3: LLM + full reflection loop (structured BeatState + improv loop)

For each tier, generates a line for a given scene prompt, then judges it
on seven dimensions. Runs each judge prompt multiple times (temperature > 0)
to assess reliability.

Output: list[JudgeRating] — one per (tier, scene_prompt) combination.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import ANTHROPIC_API_KEY, get_model
from schemas import (
    BeatState, CharacterBible, ImprovTurn, JudgeRating, SceneContext,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

Tier = Literal["vanilla", "bible", "reflection"]

# ────────────────────────────────────────────────────────────────────────────
# Line generators for each tier
# ────────────────────────────────────────────────────────────────────────────

def generate_vanilla(character: str, context: SceneContext) -> str:
    """Tier 1: Zero-shot character prompt."""
    prompt = (
        f"You are {character}. {context.prior_events}\n\n"
        f"Setting: {context.setting}\n"
        f"{'Your partner just said: ' + repr(context.partner_line) if context.partner_line else 'You open the scene.'}\n\n"
        "Say one line of dialogue as your character. Return only the dialogue, no attribution."
    )
    response = client.messages.create(
        model=get_model("generation"),
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().strip('"')


def generate_with_bible(character: str, bible: CharacterBible, context: SceneContext) -> str:
    """Tier 2: LLM + text-only character bible as system prompt."""
    system = (
        f"You are playing {character} from {bible.play_id}.\n\n"
        f"SUPEROBJECTIVE: {bible.superobjective}\n"
        f"WOUNDS/FEARS/NEEDS: {bible.wounds_fears_needs}\n"
        f"RECURRING TACTICS: {', '.join(bible.recurring_tactics)}\n"
        f"SPEECH STYLE: {bible.speech_style}\n"
        f"PSYCHOLOGICAL CONTRADICTIONS: {'; '.join(bible.psychological_contradictions)}\n"
        f"SECRETS: {'; '.join(bible.secrets[:3])}\n\n"
        "Stay in character. Say one line of dialogue. Return only the dialogue."
    )
    prompt = (
        f"Setting: {context.setting}\n"
        f"Prior events: {context.prior_events}\n"
        f"Stakes: {context.stakes}\n"
        + (f"Partner's line: {repr(context.partner_line)}" if context.partner_line else "You open the scene.")
    )
    response = client.messages.create(
        model=get_model("generation"),
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().strip('"')


def generate_with_reflection(
    bible: CharacterBible,
    context: SceneContext,
    beat_state: BeatState | None = None,
) -> str:
    """Tier 3: LLM + full reflection loop. Returns final line after revision."""
    from improv.improvisation_loop import initialize_beat_state, run_turn
    if beat_state is None:
        beat_state = initialize_beat_state(bible.character, bible, context)
    turn, _ = run_turn(1, beat_state, bible, context)
    return turn.final_line


# ────────────────────────────────────────────────────────────────────────────
# Judge
# ────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert theatre practitioner evaluating an AI-generated line of dialogue.
You have the character bible and scene context. Evaluate the line on seven dimensions.

Return ONLY valid JSON:
{
  "recognizability": 4,
  "playability": 3,
  "tactic_fidelity": 4,
  "subtext": 3,
  "earned_affect": 4,
  "knowledge_fidelity_pass": true,
  "knowledge_fidelity_note": "",
  "identified_tactic": "deflect"
}

Scoring rubric (1–5):
  5 = excellent, 4 = good, 3 = adequate, 2 = weak, 1 = fails

knowledge_fidelity_pass: true if the character does NOT reveal information they
should not have at this point. false = knowledge leak.

identified_tactic: the action verb describing what the character is doing TO the other.
"""

_JUDGE_USER = """\
CHARACTER BIBLE:
  Superobjective: {superobjective}
  Recurring tactics: {tactics}
  Speech style: {speech_style}
  Secrets (should not be revealed): {secrets}

SCENE CONTEXT:
  Setting: {setting}
  Prior events: {prior_events}
  Partner's last line: {partner_line}
  Stakes: {stakes}

GENERATED LINE:
"{line}"

Evaluate this line. Return JSON only.
"""


def judge_line(
    line: str,
    tier: Tier,
    bible: CharacterBible,
    context: SceneContext,
    num_runs: int = 3,
) -> JudgeRating:
    """Judge a line multiple times and average the numeric scores."""
    prompt = _JUDGE_USER.format(
        superobjective=bible.superobjective,
        tactics=", ".join(bible.recurring_tactics),
        speech_style=bible.speech_style,
        secrets="; ".join(bible.secrets[:5]),
        setting=context.setting,
        prior_events=context.prior_events,
        partner_line=context.partner_line or "(scene opening)",
        stakes=context.stakes,
        line=line,
    )

    raw_ratings: list[dict] = []
    for _ in range(num_runs):
        response = client.messages.create(
            model=get_model("judge"),
            max_tokens=512,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # slight randomness to assess reliability
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
        try:
            raw_ratings.append(json.loads(raw))
        except json.JSONDecodeError:
            pass

    if not raw_ratings:
        raw_ratings = [{}]

    def _mean_field(field: str, default=3.0) -> float:
        vals = []
        for r in raw_ratings:
            try:
                vals.append(float(r.get(field, default)))
            except (TypeError, ValueError):
                vals.append(default)
        return sum(vals) / len(vals) if vals else default

    kf_pass = all(r.get("knowledge_fidelity_pass", True) for r in raw_ratings)
    kf_note = next((r.get("knowledge_fidelity_note", "") for r in raw_ratings if r), "")
    tactics_seen = [r.get("identified_tactic", "") for r in raw_ratings if r.get("identified_tactic")]
    identified_tactic = max(set(tactics_seen), key=tactics_seen.count) if tactics_seen else ""

    def _clamp(v, lo=1.0, hi=5.0) -> float:
        return max(lo, min(hi, v))

    return JudgeRating(
        line=line,
        tier=tier,
        character=bible.character,
        scene_context=context.setting,
        recognizability=_clamp(_mean_field("recognizability")),
        playability=_clamp(_mean_field("playability")),
        tactic_fidelity=_clamp(_mean_field("tactic_fidelity")),
        subtext=_clamp(_mean_field("subtext")),
        earned_affect=_clamp(_mean_field("earned_affect")),
        knowledge_fidelity_pass=kf_pass,
        knowledge_fidelity_note=kf_note,
        identified_tactic=identified_tactic,
    )


# ────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ────────────────────────────────────────────────────────────────────────────

def evaluate_three_tiers(
    bible: CharacterBible,
    scene_prompts: list[SceneContext],
    num_judge_runs: int = 3,
) -> list[JudgeRating]:
    """
    For each scene prompt, generate a line under all three tiers and judge each.
    Returns all JudgeRatings.
    """
    ratings: list[JudgeRating] = []
    character = bible.character

    for i, context in enumerate(scene_prompts, 1):
        print(f"  Evaluating scene {i}/{len(scene_prompts)}: {context.setting[:60]}...")

        # Tier 1
        t1 = generate_vanilla(character, context)
        ratings.append(judge_line(t1, "vanilla", bible, context, num_judge_runs))
        print(f"    Tier 1 score: {ratings[-1].mean_score:.2f}")

        # Tier 2
        t2 = generate_with_bible(character, bible, context)
        ratings.append(judge_line(t2, "bible", bible, context, num_judge_runs))
        print(f"    Tier 2 score: {ratings[-1].mean_score:.2f}")

        # Tier 3
        t3 = generate_with_reflection(bible, context)
        ratings.append(judge_line(t3, "reflection", bible, context, num_judge_runs))
        print(f"    Tier 3 score: {ratings[-1].mean_score:.2f}")

    return ratings


def summarize_ratings(ratings: list[JudgeRating]) -> dict:
    """Compute per-tier mean scores across all dimensions."""
    from collections import defaultdict
    by_tier: dict[str, list[JudgeRating]] = defaultdict(list)
    for r in ratings:
        by_tier[r.tier].append(r)

    summary = {}
    dims = ["recognizability", "playability", "tactic_fidelity",
            "subtext", "earned_affect"]
    for tier, tier_ratings in by_tier.items():
        tier_summary = {}
        for dim in dims:
            vals = [getattr(r, dim) for r in tier_ratings]
            tier_summary[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0
        tier_summary["knowledge_fidelity_pass_rate"] = round(
            sum(r.knowledge_fidelity_pass for r in tier_ratings) / len(tier_ratings), 2
        )
        tier_summary["mean_overall"] = round(sum(r.mean_score for r in tier_ratings) / len(tier_ratings), 2)
        summary[tier] = tier_summary
    return summary
