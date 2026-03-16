"""Test bible builder logic (deduplication, filtering, incremental mode).

No API calls — tests the helper functions and orchestration logic only.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Act, Beat, BeatState, CharacterBible, Play, Scene, Utterance
from analysis.bible_builder import _count_character_beat_states, _existing_bible_characters


def _make_play_with_beat_states(characters_and_counts: dict[str, int]) -> Play:
    """Create a Play with the specified number of BeatStates per character."""
    beats = []
    for character, count in characters_and_counts.items():
        for i in range(count):
            beats.append(Beat(
                id=f"b{i}_{character}",
                play_id="test",
                act=1,
                scene=1,
                index=i,
                beat_states=[BeatState(beat_id=f"b{i}_{character}", character=character)],
                characters_present=[character],
            ))
    scene = Scene(id="test_1_1", play_id="test", act=1, scene=1, beats=beats)
    act = Act(id="test_1", play_id="test", number=1, scenes=[scene])
    return Play(
        id="test",
        title="Test Play",
        author="Test Author",
        acts=[act],
        characters=list(characters_and_counts.keys()),
    )


class TestCountBeatStates:
    def test_counts_per_character(self):
        play = _make_play_with_beat_states({"HAMLET": 10, "CLAUDIUS": 5, "GHOST": 2})
        counts = _count_character_beat_states(play)
        assert counts["HAMLET"] == 10
        assert counts["CLAUDIUS"] == 5
        assert counts["GHOST"] == 2

    def test_empty_play(self):
        play = Play(id="empty", title="Empty", author="Nobody")
        counts = _count_character_beat_states(play)
        assert counts == {}


class TestExistingBibleCharacters:
    def test_no_bibles(self):
        play = _make_play_with_beat_states({"A": 1})
        assert _existing_bible_characters(play) == set()

    def test_with_bibles(self):
        play = _make_play_with_beat_states({"A": 1, "B": 1})
        play.character_bibles = [
            CharacterBible(play_id="test", character="A"),
        ]
        existing = _existing_bible_characters(play)
        assert "A" in existing
        assert "B" not in existing

    def test_case_insensitive(self):
        play = _make_play_with_beat_states({"hamlet": 1})
        play.character_bibles = [
            CharacterBible(play_id="test", character="HAMLET"),
        ]
        existing = _existing_bible_characters(play)
        assert "HAMLET" in existing


class TestMinBeatStatesFilter:
    def test_filter_by_count(self):
        play = _make_play_with_beat_states({
            "MAJOR": 20, "MEDIUM": 5, "MINOR": 2, "EXTRA": 1
        })
        counts = _count_character_beat_states(play)
        filtered = [c for c in play.characters if counts.get(c, 0) >= 5]
        assert "MAJOR" in filtered
        assert "MEDIUM" in filtered
        assert "MINOR" not in filtered
        assert "EXTRA" not in filtered


class TestDeduplication:
    def test_existing_bible_skipped(self):
        """Characters with existing bibles should not be rebuilt."""
        play = _make_play_with_beat_states({"A": 10, "B": 10})
        play.character_bibles = [
            CharacterBible(play_id="test", character="A"),
        ]
        existing = _existing_bible_characters(play)
        target = [c for c in play.characters if c not in existing]
        assert target == ["B"]
