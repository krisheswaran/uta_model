"""
Beat Segmentation — Dramatic Analysis Pipeline

For each scene, uses a theory-grounded LLM prompt to identify beat boundaries.
A beat ends when a character's objective or tactic shifts, or when a significant
piece of information enters the scene.

Results are cached to BEATS_DIR/{play_id}_beats.json so the LLM annotation
acts as ground truth from that point on (no re-segmentation unless cache is deleted).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import BEATS_DIR, ANTHROPIC_API_KEY, get_model
from schemas import Beat, Play, Scene, Utterance

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = """\
You are a professional dramaturg with deep expertise in acting theory, \
particularly the Stanislavski / Meisner / Hagen tradition.

A BEAT is a unit of dramatic action defined by a consistent objective and tactic \
shared among the active characters. A NEW BEAT begins when:
  1. A character's scene-want (immediate objective) shifts
  2. A character changes their tactic — the specific action they are doing TO the other \
person (e.g. shifting from "seduce" to "deflect", or from "persuade" to "plead")
  3. A significant new piece of information enters that changes the stakes or dynamic
  4. The emotional register of the scene sharply changes

Respond ONLY with a JSON array of beat boundary utterance indices. \
Each element is the 0-based index of the FIRST utterance of a new beat \
(beat 1 always starts at index 0).

Example: [0, 5, 11, 18]
"""

_USER_TEMPLATE = """\
PLAY: {play_title}
ACT {act}, SCENE {scene}

UTTERANCES (0-indexed):
{utterance_block}

Identify the beat boundaries. Return only a JSON array of 0-based indices \
where new beats begin (always include 0).
"""


def _format_utterances(utterances: list[Utterance]) -> str:
    lines = []
    for i, u in enumerate(utterances):
        lines.append(f"[{i}] {u.speaker}: {u.text[:200]}")
    return "\n".join(lines)


def _call_llm(prompt: str) -> list[int]:
    response = client.messages.create(
        model=get_model("segmentation"),
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Parse JSON array from response
    try:
        boundaries = json.loads(raw)
        if not isinstance(boundaries, list):
            raise ValueError
        return sorted({int(b) for b in boundaries})
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract array from text
        import re
        m = re.search(r"\[[\d\s,]+\]", raw)
        if m:
            return sorted({int(b) for b in json.loads(m.group())})
        return [0]


def segment_scene(scene: Scene, play_title: str) -> list[Beat]:
    """
    Re-segment a scene's provisional single beat into multiple beats.
    Returns updated list of Beat objects.
    """
    # Collect all utterances from the provisional single beat
    all_utterances: list[Utterance] = []
    for beat in scene.beats:
        all_utterances.extend(beat.utterances)

    if len(all_utterances) < 3:
        # Too short to segment — keep as one beat
        return scene.beats

    prompt = _USER_TEMPLATE.format(
        play_title=play_title,
        act=scene.act,
        scene=scene.scene,
        utterance_block=_format_utterances(all_utterances),
    )
    boundaries = _call_llm(prompt)

    # Ensure 0 is always first
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries

    # Build beats from boundaries
    beats: list[Beat] = []
    for b_idx, start in enumerate(boundaries, start=1):
        end = boundaries[b_idx] if b_idx < len(boundaries) else len(all_utterances)
        beat_utterances = all_utterances[start:end]
        if not beat_utterances:
            continue
        beat_id = f"{scene.id}_b{b_idx}"
        beat = Beat(
            id=beat_id,
            play_id=scene.play_id,
            act=scene.act,
            scene=scene.scene,
            index=b_idx,
            utterances=beat_utterances,
            characters_present=sorted({u.speaker for u in beat_utterances}),
        )
        beats.append(beat)

    return beats if beats else scene.beats


def segment_play(play: Play, use_cache: bool = True) -> Play:
    """
    Run beat segmentation over all scenes in a play.
    Caches results to BEATS_DIR so re-runs are deterministic.
    """
    cache_path = BEATS_DIR / f"{play.id}_beats.json"

    if use_cache and cache_path.exists():
        print(f"[segmenter] Loading cached beats from {cache_path}")
        cached = json.loads(cache_path.read_text())
        # Rebuild beats from cache
        _apply_beat_cache(play, cached)
        return play

    print(f"[segmenter] Segmenting {play.title} ({sum(len(a.scenes) for a in play.acts)} scenes)...")
    all_beats_cache: dict[str, list[int]] = {}  # scene_id → boundary indices

    for act in play.acts:
        for scene in act.scenes:
            utterances_flat = [u for b in scene.beats for u in b.utterances]
            if len(utterances_flat) < 3:
                continue
            prompt = _USER_TEMPLATE.format(
                play_title=play.title,
                act=scene.act,
                scene=scene.scene,
                utterance_block=_format_utterances(utterances_flat),
            )
            boundaries = _call_llm(prompt)
            if not boundaries or boundaries[0] != 0:
                boundaries = [0] + boundaries
            all_beats_cache[scene.id] = boundaries
            new_beats = _boundaries_to_beats(scene, utterances_flat, boundaries)
            scene.beats = new_beats
            print(f"  {scene.id}: {len(new_beats)} beats")

    cache_path.write_text(json.dumps(all_beats_cache, indent=2))
    print(f"[segmenter] Cached beats to {cache_path}")
    return play


def _boundaries_to_beats(scene: Scene, utterances: list[Utterance], boundaries: list[int]) -> list[Beat]:
    beats: list[Beat] = []
    for b_idx, start in enumerate(boundaries, start=1):
        end = boundaries[b_idx] if b_idx < len(boundaries) else len(utterances)
        beat_utterances = utterances[start:end]
        if not beat_utterances:
            continue
        beat = Beat(
            id=f"{scene.id}_b{b_idx}",
            play_id=scene.play_id,
            act=scene.act,
            scene=scene.scene,
            index=b_idx,
            utterances=beat_utterances,
            characters_present=sorted({u.speaker for u in beat_utterances}),
        )
        beats.append(beat)
    return beats if beats else scene.beats


def _apply_beat_cache(play: Play, cached: dict[str, list[int]]) -> None:
    for act in play.acts:
        for scene in act.scenes:
            if scene.id not in cached:
                continue
            utterances_flat = [u for b in scene.beats for u in b.utterances]
            boundaries = cached[scene.id]
            scene.beats = _boundaries_to_beats(scene, utterances_flat, boundaries)
