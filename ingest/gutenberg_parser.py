"""
Parser for plain-text Project Gutenberg play scripts.

Handles the Constance Garnett formatting convention:
  - Character names followed immediately by a period and dialogue on the same
    line: ``LOPAKHIN. The train's arrived, thank God.``
  - Stage directions in square brackets (inline or on their own line)
  - Acts marked with "ACT I" (Roman) or "ACT ONE" (written-out)
  - Scene headings marked with "SCENE I" / "SCENE ONE" etc. (optional)

Multi-play files (e.g. Gutenberg #7986 — Chekhov Second Series) are handled
via the optional ``text_anchor`` parameter, which tells the parser to skip
all text before the first line that exactly matches the anchor string.

Usage:
    from ingest.gutenberg_parser import parse_gutenberg_play
    play = parse_gutenberg_play(
        "cherry_orchard", raw_text,
        title="The Cherry Orchard", author="Anton Chekhov",
        text_anchor="THE CHERRY ORCHARD",
    )
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Act, Beat, Play, Scene, Utterance


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_WRITTEN_NUMS = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8,
}

_ACT_RE = re.compile(
    r"^\s*ACT\s+([IVXivx\d]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT)\.?\s*$",
    re.IGNORECASE,
)
_SCENE_RE = re.compile(
    r"^\s*SCENE\s+([IVXivx\d]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT)\.?\s*$",
    re.IGNORECASE,
)
_STAGE_RE = re.compile(r"^\s*[\[\(].+?[\]\)]\s*$")

# Primary format: "SPEAKER. rest of dialogue on same line"
# Requires at least two all-caps words (or one word of ≥2 caps chars) before
# the period so we don't match single-letter abbreviations.
_INLINE_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Z\s\-\']{0,40}[A-Z])\.\s+(.+)$"
)

# Fallback: speaker name alone on a line (some Gutenberg sources use this)
_STANDALONE_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Z\s\-\']{1,40}[A-Z])[\.\:]?\s*$"
)

_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    if s in _ROMAN:
        return _ROMAN[s]
    if s in _WRITTEN_NUMS:
        return _WRITTEN_NUMS[s]
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
    text_anchor: str = "",
) -> Play:
    """
    Parse a Gutenberg plain-text play into a structured Play object.

    Args:
        play_id: Unique identifier for the play.
        raw_text: Full text of the Gutenberg file.
        title: Human-readable title.
        author: Author name.
        primary_character: Name of the primary character (unused in parsing,
            stored for downstream use).
        text_anchor: If provided, all text before the first line that exactly
            matches this string (case-insensitive, after stripping) is
            discarded.  Use this when a single Gutenberg file contains
            multiple plays.

    Returns a Play with Acts → Scenes → Beats (one beat per scene initially;
    beat segmentation is refined in analysis/segmenter.py).
    """
    lines = raw_text.splitlines()

    # -- Anchor: skip to the target play within a multi-play file -----------
    if text_anchor:
        anchor_upper = text_anchor.strip().upper()
        # Use rstrip-only so indented CONTENTS entries (e.g. "     THE CHERRY ORCHARD")
        # don't match the actual title line ("THE CHERRY ORCHARD" at column 0).
        anchor_index = next(
            (i for i, ln in enumerate(lines) if ln.rstrip().upper() == anchor_upper),
            None,
        )
        if anchor_index is not None:
            lines = lines[anchor_index:]
        # If anchor not found, fall through and parse the whole file

    play = Play(id=play_id, title=title, author=author)

    current_act_num = 1
    current_scene_num = 1
    current_speaker: str | None = None
    current_lines: list[str] = []
    utterance_index = 0
    seen_first_act = False

    acts: dict[int, dict[int, list[Utterance]]] = {}

    def flush_speaker() -> None:
        nonlocal utterance_index
        if not (current_speaker and current_lines and seen_first_act):
            return
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

        # -- Gutenberg end-of-text marker -----------------------------------
        if line.startswith("*** END OF"):
            break

        # -- Act heading ----------------------------------------------------
        act_m = _ACT_RE.match(line)
        if act_m:
            flush_speaker()
            current_speaker = None
            current_lines = []
            current_act_num = _roman_to_int(act_m.group(1))
            current_scene_num = 1
            seen_first_act = True
            continue

        # -- Scene heading --------------------------------------------------
        scene_m = _SCENE_RE.match(line)
        if scene_m:
            flush_speaker()
            current_speaker = None
            current_lines = []
            current_scene_num = _roman_to_int(scene_m.group(1))
            continue

        # -- Stage direction (whole line) -----------------------------------
        if _STAGE_RE.match(line):
            flush_speaker()
            current_speaker = None
            current_lines = []
            continue

        if not seen_first_act:
            continue

        # -- Inline speaker: "SPEAKER. dialogue text" -----------------------
        inline_m = _INLINE_SPEAKER_RE.match(line)
        if inline_m:
            flush_speaker()
            current_speaker = inline_m.group(1).strip()
            current_lines = [inline_m.group(2).strip()]
            continue

        # -- Standalone speaker (fallback for other Gutenberg sources) ------
        standalone_m = _STANDALONE_SPEAKER_RE.match(line)
        if standalone_m:
            flush_speaker()
            current_speaker = standalone_m.group(1).strip()
            current_lines = []
            continue

        # -- Continuation line ----------------------------------------------
        if current_speaker and line:
            current_lines.append(line)

    flush_speaker()

    # -- Build Play structure -----------------------------------------------
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
                id=_make_id(play_id, act_num, scene_num),
                play_id=play_id,
                act=act_num,
                scene=scene_num,
                beats=[beat],
            )
            act_obj.scenes.append(scene_obj)
        play.acts.append(act_obj)

    play.characters = sorted(all_characters)
    return play
