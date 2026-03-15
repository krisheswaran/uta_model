"""
Parser for plain-text Project Gutenberg play scripts.

Handles the Constance Garnett formatting convention:
  - Character names appear on their own line in ALL CAPS (possibly followed by a period)
  - Stage directions appear in square brackets or parentheses, or as italicized text
  - Acts and scenes are marked with "ACT I", "SCENE I" etc.
  - Lines of dialogue immediately follow the speaker name

Usage:
    from ingest.gutenberg import parse_gutenberg_play
    play = parse_gutenberg_play("cherry_orchard", raw_text, primary_character="LOPAKHIN")
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Act, Beat, Play, Scene, Utterance


# Patterns -----------------------------------------------------------------

_ACT_RE = re.compile(
    r"^\s*ACT\s+([IVXivx\d]+)\.?\s*$", re.IGNORECASE
)
_SCENE_RE = re.compile(
    r"^\s*SCENE\s+([IVXivx\d]+)\.?\s*$", re.IGNORECASE
)
# Speaker: all-caps word(s), optionally ending in period or colon
_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Z\s\-\']{1,30}[A-Z])[\.\:]?\s*$"
)
_STAGE_RE = re.compile(
    r"^\s*[\[\(](.+?)[\]\)]\s*$"
)

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
           "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    if s in _ROMAN:
        return _ROMAN[s]
    try:
        return int(s)
    except ValueError:
        return 1


def _make_id(*parts) -> str:
    return "_".join(str(p).lower().replace(" ", "_") for p in parts)


def parse_gutenberg_play(
    play_id: str,
    raw_text: str,
    title: str = "",
    author: str = "",
    primary_character: str = "",
) -> Play:
    """
    Parse a Gutenberg plain-text play into a structured Play object.

    Returns a Play with Acts → Scenes → Beats (one beat per scene initially;
    beat segmentation is refined in pass1/segmenter.py).
    Each scene contains a single provisional beat holding all utterances.
    """
    lines = raw_text.splitlines()

    play = Play(
        id=play_id,
        title=title,
        author=author,
    )

    current_act_num = 1
    current_scene_num = 1
    current_speaker: str | None = None
    current_lines: list[str] = []
    utterance_index = 0

    # Provisional storage
    acts: dict[int, dict[int, list[Utterance]]] = {}

    def flush_speaker() -> None:
        nonlocal utterance_index
        if current_speaker and current_lines:
            text = " ".join(current_lines).strip()
            if not text:
                return
            uid = _make_id(play_id, current_act_num, current_scene_num, f"u{utterance_index}")
            utt = Utterance(
                id=uid,
                play_id=play_id,
                act=current_act_num,
                scene=current_scene_num,
                index=utterance_index,
                speaker=current_speaker,
                text=text,
            )
            acts.setdefault(current_act_num, {}).setdefault(current_scene_num, []).append(utt)
            utterance_index += 1

    for raw_line in lines:
        line = raw_line.strip()

        act_m = _ACT_RE.match(line)
        if act_m:
            flush_speaker()
            current_speaker = None
            current_lines = []
            current_act_num = _roman_to_int(act_m.group(1))
            current_scene_num = 1
            continue

        scene_m = _SCENE_RE.match(line)
        if scene_m:
            flush_speaker()
            current_speaker = None
            current_lines = []
            current_scene_num = _roman_to_int(scene_m.group(1))
            continue

        stage_m = _STAGE_RE.match(line)
        if stage_m:
            # stage direction — attach to last utterance if any, otherwise skip
            flush_speaker()
            current_speaker = None
            current_lines = []
            continue

        speaker_m = _SPEAKER_RE.match(line)
        if speaker_m:
            flush_speaker()
            current_speaker = speaker_m.group(1).strip()
            current_lines = []
            continue

        if current_speaker and line:
            current_lines.append(line)

    flush_speaker()

    # Build Play structure from collected utterances
    all_characters: set[str] = set()
    for act_num in sorted(acts):
        act_obj = Act(
            id=_make_id(play_id, act_num),
            play_id=play_id,
            number=act_num,
        )
        for scene_num in sorted(acts[act_num]):
            utterances = acts[act_num][scene_num]
            for u in utterances:
                all_characters.add(u.speaker)

            scene_id = _make_id(play_id, act_num, scene_num)
            # Provisional: one beat per scene
            beat = Beat(
                id=_make_id(play_id, act_num, scene_num, "b1"),
                play_id=play_id,
                act=act_num,
                scene=scene_num,
                index=1,
                utterances=utterances,
                characters_present=sorted({u.speaker for u in utterances}),
            )
            scene_obj = Scene(
                id=scene_id,
                play_id=play_id,
                act=act_num,
                scene=scene_num,
                beats=[beat],
            )
            act_obj.scenes.append(scene_obj)
        play.acts.append(act_obj)

    play.characters = sorted(all_characters)
    return play
