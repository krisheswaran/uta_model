"""Test that existing parsed play data loads correctly.

These tests verify that the actual data files on disk are valid and
can be deserialized without errors. They catch issues like the
malformed affect_state strings that were fixed during Phase B.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PARSED_DIR
from schemas import Play


def _parsed_play_paths():
    """Return all parsed play JSON paths that exist on disk."""
    if not PARSED_DIR.exists():
        return []
    return list(PARSED_DIR.glob("*.json"))


@pytest.mark.skipif(not _parsed_play_paths(), reason="No parsed play data on disk")
class TestParsedPlayData:
    @pytest.fixture(params=[p.stem for p in _parsed_play_paths()], ids=[p.stem for p in _parsed_play_paths()])
    def play(self, request):
        path = PARSED_DIR / f"{request.param}.json"
        return Play.model_validate_json(path.read_text())

    def test_play_loads(self, play):
        """The play file deserializes without Pydantic validation errors."""
        assert play.id
        assert play.title
        assert len(play.acts) > 0

    def test_has_characters(self, play):
        assert len(play.characters) > 0

    def test_has_beats(self, play):
        total_beats = sum(len(s.beats) for a in play.acts for s in a.scenes)
        assert total_beats > 0

    def test_beat_states_have_valid_affect(self, play):
        """Every beat state's affect_state should be a proper AffectState object."""
        from schemas import AffectState
        for act in play.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    for bs in beat.beat_states:
                        assert isinstance(bs.affect_state, AffectState), (
                            f"Beat {bs.beat_id} char {bs.character}: "
                            f"affect_state is {type(bs.affect_state)}, not AffectState"
                        )

    def test_has_at_least_one_bible(self, play):
        assert len(play.character_bibles) >= 1

    def test_canonical_tactic_field_exists(self, play):
        """BeatStates should have the canonical_tactic field (may be None)."""
        for act in play.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    for bs in beat.beat_states:
                        # Field exists (None or str)
                        assert hasattr(bs, "canonical_tactic")


@pytest.mark.skipif(
    not (PARSED_DIR / "cherry_orchard.json").exists(),
    reason="Cherry Orchard data not on disk",
)
class TestCherryOrchardData:
    @pytest.fixture
    def play(self):
        return Play.model_validate_json(
            (PARSED_DIR / "cherry_orchard.json").read_text()
        )

    def test_has_lopakhin_bible(self, play):
        bible = play.get_character_bible("LOPAKHIN")
        assert bible is not None
        assert len(bible.tactic_distribution) > 0

    def test_has_scene_bibles(self, play):
        assert len(play.scene_bibles) > 0

    def test_lopakhin_tactic_distribution(self, play):
        bible = play.get_character_bible("LOPAKHIN")
        assert len(bible.tactic_distribution) > 0


@pytest.mark.skipif(
    not (PARSED_DIR / "hamlet.json").exists(),
    reason="Hamlet data not on disk",
)
class TestHamletData:
    @pytest.fixture
    def play(self):
        return Play.model_validate_json(
            (PARSED_DIR / "hamlet.json").read_text()
        )

    def test_has_hamlet_bible(self, play):
        bible = play.get_character_bible("HAMLET")
        assert bible is not None

    def test_no_string_affect_states(self, play):
        """Regression: Hamlet had 2 BeatStates with string affect_state."""
        for act in play.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    for bs in beat.beat_states:
                        assert not isinstance(bs.affect_state, str)
