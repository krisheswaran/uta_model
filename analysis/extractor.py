"""
Dramatic State Extraction — Analysis Pipeline Step 3

For each beat, extracts a structured BeatState per character using an LLM
prompted with acting-theory vocabulary. Each BeatState captures:
  - desire_state (scene want)
  - obstacle
  - tactic_state (action verb)
  - affect_state (valence, arousal, certainty, control, vulnerability)
  - social_state (status, warmth)
  - epistemic_state (known facts, secrets, false beliefs)
  - defense_state
  - psychological_contradiction
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import ANALYSIS_MODEL, ANTHROPIC_API_KEY
from schemas import (
    AffectState, Beat, BeatState, EpistemicState, Play, SocialState,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = """\
You are an expert dramaturg and acting coach trained in Stanislavski, Meisner, \
and Uta Hagen techniques. You analyze dramatic texts to extract the hidden \
psychological state of each character at each beat.

For each character present in a beat, produce a structured analysis as JSON.
Be specific and concrete. Use active action verbs for tactics \
(e.g. "seduce", "deflect", "shame", "plead", "provoke", "conceal", "dominate", \
"test", "reassure", "expose", "mock", "flatter", "appease", "challenge").

For affect dimensions use floats between -1.0 and 1.0 \
(valence, arousal, certainty, control) or 0.0 to 1.0 (vulnerability).
"""

_USER_TEMPLATE = """\
PLAY: {play_title}
ACT {act}, SCENE {scene}, BEAT {beat_index}

SCENE CONTEXT (prior beats summary):
{prior_context}

THIS BEAT'S UTTERANCES:
{utterance_block}

CHARACTERS IN THIS BEAT: {characters}

For EACH character listed, return a JSON object with this exact structure:
{{
  "CHARACTER_NAME": {{
    "desire_state": "what they want right now in this beat",
    "superobjective_reminder": "how this connects to their deepest want",
    "obstacle": "what blocks them",
    "tactic_state": "single action verb — what they are doing TO the other",
    "defense_state": "active defense mechanism if any, else empty string",
    "psychological_contradiction": "any active internal contradiction, else empty",
    "affect_state": {{
      "valence": 0.0,
      "arousal": 0.0,
      "certainty": 0.0,
      "control": 0.0,
      "vulnerability": 0.0,
      "rationale": "brief explanation"
    }},
    "social_state": {{
      "status": 0.0,
      "warmth": 0.0,
      "rationale": "brief explanation"
    }},
    "epistemic_state": {{
      "known_facts": ["fact1", "fact2"],
      "hidden_secrets": ["secret1"],
      "false_beliefs": ["false belief if any"],
      "rationale": "brief explanation"
    }},
    "confidence": 1.0,
    "alternative_hypothesis": "alternative reading if ambiguous, else empty"
  }}
}}

Return ONLY valid JSON. Include ALL characters listed.
"""


def _format_utterances(utterances) -> str:
    return "\n".join(f"  {u.speaker}: {u.text}" for u in utterances)


def _format_prior_context(beat: Beat, all_beats: list[Beat]) -> str:
    prior = [b for b in all_beats if b.act == beat.act and b.scene == beat.scene and b.index < beat.index]
    if not prior:
        return "(start of scene)"
    summaries = []
    for b in prior[-3:]:  # last 3 beats
        if b.beat_summary:
            summaries.append(f"Beat {b.index}: {b.beat_summary}")
        else:
            first = b.utterances[0] if b.utterances else None
            if first:
                summaries.append(f"Beat {b.index}: {first.speaker} — \"{first.text[:80]}...\"")
    return "\n".join(summaries) if summaries else "(no prior beats)"


def extract_beat_states(beat: Beat, play_title: str, all_beats: list[Beat]) -> list[BeatState]:
    """Extract BeatState for each character in this beat."""
    if not beat.utterances or not beat.characters_present:
        return []

    prior_context = _format_prior_context(beat, all_beats)
    prompt = _USER_TEMPLATE.format(
        play_title=play_title,
        act=beat.act,
        scene=beat.scene,
        beat_index=beat.index,
        prior_context=prior_context,
        utterance_block=_format_utterances(beat.utterances),
        characters=", ".join(beat.characters_present),
    )

    response = client.messages.create(
        model=ANALYSIS_MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    beat_states: list[BeatState] = []
    for character in beat.characters_present:
        char_data = data.get(character, data.get(character.upper(), {}))
        if not char_data:
            continue
        try:
            affect = AffectState(**char_data.get("affect_state", {}))
            social = SocialState(**char_data.get("social_state", {}))
            epistemic = EpistemicState(**char_data.get("epistemic_state", {}))
            bs = BeatState(
                beat_id=beat.id,
                character=character,
                desire_state=char_data.get("desire_state", ""),
                superobjective_reminder=char_data.get("superobjective_reminder", ""),
                obstacle=char_data.get("obstacle", ""),
                tactic_state=char_data.get("tactic_state", ""),
                affect_state=affect,
                social_state=social,
                epistemic_state=epistemic,
                defense_state=char_data.get("defense_state", ""),
                psychological_contradiction=char_data.get("psychological_contradiction", ""),
                confidence=float(char_data.get("confidence", 1.0)),
                alternative_hypothesis=char_data.get("alternative_hypothesis", ""),
            )
            beat_states.append(bs)
        except Exception:
            continue

    return beat_states


def extract_all_beats(play: Play) -> Play:
    """Extract BeatStates for every beat in the play."""
    all_beats = [b for act in play.acts for scene in act.scenes for b in scene.beats]
    total = len(all_beats)
    print(f"[extractor] Extracting states for {total} beats in {play.title}...")

    for i, beat in enumerate(all_beats, 1):
        print(f"  [{i}/{total}] {beat.id} ({len(beat.characters_present)} characters)")
        beat.beat_states = extract_beat_states(beat, play.title, all_beats)

    return play
