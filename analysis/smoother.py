"""
Global Arc Refinement (Smoother) — Analysis Pipeline Step 4

After per-beat extraction, reads the full character trajectory across the play
and flags inconsistencies, smooths emotional transitions, and enforces arc coherence.

Runs a configurable number of passes (default: 2). Each pass reads all BeatStates
for one character and asks the LLM to flag and correct inconsistencies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import ANTHROPIC_API_KEY, SMOOTH_PASSES, get_model
from schemas import AffectState, BeatState, EpistemicState, Play, SocialState

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = """\
You are an expert dramaturg reviewing a character analysis across an entire play.
Your task is to ensure the character's arc is coherent and psychologically consistent.

Look for:
1. Tactic jumps that are unmotivated (e.g. sudden shift with no triggering event)
2. Knowledge inconsistencies (character reveals something they shouldn't know yet)
3. Emotional transitions that are too abrupt or too flat
4. Superobjective drift (if the character's deep want shifts without cause)
5. Missing contradictions (a known pattern not reflected in a specific beat)

Return a JSON array of corrections. Each correction has:
  "beat_id": the beat to correct,
  "field": which field to update (e.g. "tactic_state", "desire_state"),
  "old_value": what it currently says,
  "new_value": the corrected value,
  "rationale": why this correction improves coherence

If no corrections are needed, return an empty array: []
"""

_USER_TEMPLATE = """\
PLAY: {play_title}
CHARACTER: {character}

CHARACTER ARC (all beats where this character appears):
{arc_block}

Review this arc for coherence. Return a JSON array of corrections or [].
"""


def _format_arc(beat_states: list[BeatState]) -> str:
    lines = []
    for bs in beat_states:
        lines.append(
            f"Beat {bs.beat_id}:\n"
            f"  desire: {bs.desire_state}\n"
            f"  tactic: {bs.tactic_state}\n"
            f"  obstacle: {bs.obstacle}\n"
            f"  affect valence={bs.affect_state.valence:.1f} arousal={bs.affect_state.arousal:.1f}\n"
            f"  defense: {bs.defense_state}\n"
            f"  contradiction: {bs.psychological_contradiction}\n"
            f"  epistemic secrets: {bs.epistemic_state.hidden_secrets}\n"
        )
    return "\n".join(lines)


_NESTED_FIELDS = {
    "affect_state": AffectState,
    "social_state": SocialState,
    "epistemic_state": EpistemicState,
}


def _apply_corrections(beat_states: list[BeatState], corrections: list[dict]) -> None:
    beat_map = {bs.beat_id: bs for bs in beat_states}
    for correction in corrections:
        beat_id = correction.get("beat_id", "")
        field = correction.get("field", "")
        new_value = correction.get("new_value", "")
        if beat_id not in beat_map or not field:
            continue
        bs = beat_map[beat_id]
        if not hasattr(bs, field):
            continue
        try:
            if field in _NESTED_FIELDS and isinstance(new_value, dict):
                new_value = _NESTED_FIELDS[field](**new_value)
            elif field in _NESTED_FIELDS:
                # LLM returned a non-dict for a nested field — skip
                continue
            setattr(bs, field, new_value)
        except Exception:
            pass


def smooth_character_arc(
    character: str, beat_states: list[BeatState], play_title: str
) -> list[BeatState]:
    """Run smoothing passes on a single character's arc."""
    for pass_num in range(SMOOTH_PASSES):
        prompt = _USER_TEMPLATE.format(
            play_title=play_title,
            character=character,
            arc_block=_format_arc(beat_states),
        )
        response = client.messages.create(
            model=get_model("smoothing"),
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
        try:
            corrections = json.loads(raw)
            if isinstance(corrections, list) and corrections:
                print(f"    [smoother] {character} pass {pass_num + 1}: {len(corrections)} correction(s)")
                _apply_corrections(beat_states, corrections)
            else:
                print(f"    [smoother] {character} pass {pass_num + 1}: no corrections needed")
        except json.JSONDecodeError:
            print(f"    [smoother] {character} pass {pass_num + 1}: could not parse corrections")

    return beat_states


def smooth_play(play: Play) -> Play:
    """Run global arc smoothing for each character across the whole play."""
    print(f"[smoother] Smoothing character arcs in {play.title} ({SMOOTH_PASSES} passes each)...")

    # Collect all beat states per character
    all_beat_states: dict[str, list[BeatState]] = {}
    for act in play.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                for bs in beat.beat_states:
                    all_beat_states.setdefault(bs.character, []).append(bs)

    for character, beat_states in all_beat_states.items():
        print(f"  Smoothing {character} ({len(beat_states)} beats)...")
        smooth_character_arc(character, beat_states, play.title)

    return play
