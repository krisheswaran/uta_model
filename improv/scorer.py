"""
Six-Axis Scorer — Improvisation System

Evaluates a candidate line against the character model along six axes:
  1. Voice fidelity
  2. Tactic fidelity
  3. Knowledge fidelity
  4. Relationship fidelity
  5. Subtext richness
  6. Emotional transition plausibility

Returns a ScoredLine with per-axis scores (1–5) and targeted state-based feedback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import ANTHROPIC_API_KEY, get_model
from schemas import BeatState, CandidateLine, CharacterBible, SceneContext, ScoredLine

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = """\
You are a master acting coach and dramaturg evaluating improvised dialogue.
You will be given:
  - A character bible describing the character's psychology
  - The current dramatic state (objective, tactic, affect, epistemic state, etc.)
  - The scene context
  - A candidate line the character is about to say

Rate the candidate line on six axes (each 1–5 where 5 = excellent) and provide
targeted, state-based feedback for any axis scoring below 3.

Feedback must be SPECIFIC and refer to the state — e.g.:
  "This line is too direct; the tactic should be 'deflect', not a direct admission."
  "The character should not know about X yet — this is a knowledge leak."
  "Status is too low here; Lopakhin is trying to dominate this exchange."

Return ONLY valid JSON:
{
  "voice_fidelity": 4,
  "tactic_fidelity": 3,
  "knowledge_fidelity": 5,
  "relationship_fidelity": 4,
  "subtext_richness": 2,
  "emotional_transition_plausibility": 4,
  "feedback": [
    "Subtext note: the character should suggest rather than state their want directly.",
    "..."
  ]
}
"""

_USER_TEMPLATE = """\
CHARACTER BIBLE:
  Superobjective: {superobjective}
  Recurring tactics: {recurring_tactics}
  Speech style: {speech_style}
  Defense mechanisms: {defense_mechanisms}
  Psychological contradictions: {contradictions}
  Known secrets (must not reveal): {secrets}

CURRENT BEAT STATE:
  Scene want: {desire_state}
  Obstacle: {obstacle}
  Tactic: {tactic_state}
  Affect: valence={valence:.1f}, arousal={arousal:.1f}, vulnerability={vulnerability:.1f}
  Relationship to partner: warmth={warmth:.1f}, status={status:.1f}
  Defense active: {defense_state}
  Contradiction active: {contradiction}

SCENE CONTEXT:
  Setting: {setting}
  Partner's last line: {partner_line}
  Stakes: {stakes}

CANDIDATE LINE:
"{candidate_text}"

Rate this line and provide targeted feedback. Return JSON only.
"""


def score_candidate(
    candidate: CandidateLine,
    beat_state: BeatState,
    character_bible: CharacterBible,
    context: SceneContext,
) -> ScoredLine:
    """Score a candidate line against the character model on six axes."""
    prompt = _USER_TEMPLATE.format(
        superobjective=character_bible.superobjective,
        recurring_tactics=", ".join(character_bible.recurring_tactics),
        speech_style=character_bible.speech_style,
        defense_mechanisms=", ".join(character_bible.preferred_defense_mechanisms),
        contradictions="; ".join(character_bible.psychological_contradictions),
        secrets="; ".join(character_bible.secrets[:5]),
        desire_state=beat_state.desire_state,
        obstacle=beat_state.obstacle,
        tactic_state=beat_state.tactic_state,
        valence=beat_state.affect_state.valence,
        arousal=beat_state.affect_state.arousal,
        vulnerability=beat_state.affect_state.vulnerability,
        warmth=beat_state.social_state.warmth,
        status=beat_state.social_state.status,
        defense_state=beat_state.defense_state,
        contradiction=beat_state.psychological_contradiction,
        setting=context.setting,
        partner_line=context.partner_line or "(scene opening)",
        stakes=context.stakes,
        candidate_text=candidate.text,
    )

    response = client.messages.create(
        model=get_model("critic"),
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    def _clamp(val, default=3.0) -> float:
        try:
            return max(1.0, min(5.0, float(val)))
        except (TypeError, ValueError):
            return default

    return ScoredLine(
        candidate=candidate,
        voice_fidelity=_clamp(data.get("voice_fidelity", 3)),
        tactic_fidelity=_clamp(data.get("tactic_fidelity", 3)),
        knowledge_fidelity=_clamp(data.get("knowledge_fidelity", 3)),
        relationship_fidelity=_clamp(data.get("relationship_fidelity", 3)),
        subtext_richness=_clamp(data.get("subtext_richness", 3)),
        emotional_transition_plausibility=_clamp(
            data.get("emotional_transition_plausibility", 3)
        ),
        feedback=data.get("feedback", []),
    )
