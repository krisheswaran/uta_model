"""
State Updater — Improvisation System

After each generated utterance, advances the BeatState:
  - updates the scene want if the line shifted it
  - adjusts affect state (small incremental change based on what was said)
  - updates relationship temperature (warmth/status shift)
  - tracks beat shifts (if a new beat has begun)

This keeps the improvisation session stateful across turns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import CRITIC_MODEL, ANTHROPIC_API_KEY
from schemas import AffectState, BeatState, SocialState

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = """\
You are tracking the hidden dramatic state of a character during an improvisation.
Given the previous state and the line just delivered, return an updated state.

Changes should be incremental — dramatic state shifts gradually unless a major
revelation or confrontation occurs.

Return ONLY valid JSON with the updated fields:
{
  "desire_state": "...",
  "tactic_state": "single action verb",
  "defense_state": "...",
  "psychological_contradiction": "...",
  "affect_state": {
    "valence": 0.0,
    "arousal": 0.0,
    "certainty": 0.0,
    "control": 0.0,
    "vulnerability": 0.0,
    "rationale": "..."
  },
  "social_state": {
    "status": 0.0,
    "warmth": 0.0,
    "rationale": "..."
  },
  "beat_shifted": false,
  "beat_shift_reason": ""
}
"""

_USER_TEMPLATE = """\
CHARACTER: {character}
PREVIOUS STATE:
  want: {desire_state}
  tactic: {tactic_state}
  affect: valence={valence:.1f} arousal={arousal:.1f} vulnerability={vulnerability:.1f}
  social: status={status:.1f} warmth={warmth:.1f}
  defense: {defense_state}

LINE JUST DELIVERED:
  "{line}"

PARTNER RESPONSE (if any):
  "{partner_line}"

Return updated state JSON.
"""


def update_beat_state(
    previous_state: BeatState,
    line_delivered: str,
    partner_line: str = "",
) -> BeatState:
    """Return a new BeatState reflecting what changed after this turn."""
    prompt = _USER_TEMPLATE.format(
        character=previous_state.character,
        desire_state=previous_state.desire_state,
        tactic_state=previous_state.tactic_state,
        valence=previous_state.affect_state.valence,
        arousal=previous_state.affect_state.arousal,
        vulnerability=previous_state.affect_state.vulnerability,
        status=previous_state.social_state.status,
        warmth=previous_state.social_state.warmth,
        defense_state=previous_state.defense_state,
        line=line_delivered,
        partner_line=partner_line or "(no response yet)",
    )

    response = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=768,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return previous_state  # keep previous on parse failure

    def _clamp(val, lo=-1.0, hi=1.0):
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return 0.0

    affect_data = data.get("affect_state", {})
    social_data = data.get("social_state", {})

    new_state = previous_state.model_copy(deep=True)
    new_state.desire_state = data.get("desire_state", previous_state.desire_state)
    new_state.tactic_state = data.get("tactic_state", previous_state.tactic_state)
    new_state.defense_state = data.get("defense_state", previous_state.defense_state)
    new_state.psychological_contradiction = data.get(
        "psychological_contradiction", previous_state.psychological_contradiction
    )
    new_state.affect_state = AffectState(
        valence=_clamp(affect_data.get("valence", previous_state.affect_state.valence)),
        arousal=_clamp(affect_data.get("arousal", previous_state.affect_state.arousal)),
        certainty=_clamp(affect_data.get("certainty", previous_state.affect_state.certainty)),
        control=_clamp(affect_data.get("control", previous_state.affect_state.control)),
        vulnerability=_clamp(affect_data.get("vulnerability", previous_state.affect_state.vulnerability), lo=0.0),
        rationale=affect_data.get("rationale", ""),
    )
    new_state.social_state = SocialState(
        status=_clamp(social_data.get("status", previous_state.social_state.status)),
        warmth=_clamp(social_data.get("warmth", previous_state.social_state.warmth)),
        rationale=social_data.get("rationale", ""),
    )
    return new_state
