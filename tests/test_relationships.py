"""Test relationship builder — pairwise edges and relational profiles."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    Act, Beat, BeatState, Play, RelationalProfile,
    RelationshipEdge, Scene, SocialState,
)
from analysis.relationship_builder import (
    build_pairwise_edges, build_relational_profiles,
    _collect_pairwise_social_data,
)


def _make_play_with_social_data() -> Play:
    """Create a play with known social_state values for testing."""
    beats = []
    # Beat 1: A warm to B, B cold to A
    beats.append(Beat(
        id="b1", play_id="test", act=1, scene=1, index=1,
        beat_states=[
            BeatState(beat_id="b1", character="A",
                      social_state=SocialState(status=0.5, warmth=0.8)),
            BeatState(beat_id="b1", character="B",
                      social_state=SocialState(status=-0.3, warmth=-0.5)),
        ],
        characters_present=["A", "B"],
    ))
    # Beat 2: A still warm, B still cold
    beats.append(Beat(
        id="b2", play_id="test", act=1, scene=1, index=2,
        beat_states=[
            BeatState(beat_id="b2", character="A",
                      social_state=SocialState(status=0.4, warmth=0.6)),
            BeatState(beat_id="b2", character="B",
                      social_state=SocialState(status=-0.2, warmth=-0.4)),
        ],
        characters_present=["A", "B"],
    ))
    # Beat 3: A warm, B warms slightly
    beats.append(Beat(
        id="b3", play_id="test", act=1, scene=1, index=3,
        beat_states=[
            BeatState(beat_id="b3", character="A",
                      social_state=SocialState(status=0.3, warmth=0.7)),
            BeatState(beat_id="b3", character="B",
                      social_state=SocialState(status=-0.1, warmth=0.1)),
        ],
        characters_present=["A", "B"],
    ))
    # Beat 4: A with C (different partner)
    beats.append(Beat(
        id="b4", play_id="test", act=1, scene=1, index=4,
        beat_states=[
            BeatState(beat_id="b4", character="A",
                      social_state=SocialState(status=-0.2, warmth=0.2)),
            BeatState(beat_id="b4", character="C",
                      social_state=SocialState(status=0.6, warmth=0.5)),
        ],
        characters_present=["A", "C"],
    ))

    scene = Scene(id="test_1_1", play_id="test", act=1, scene=1, beats=beats)
    act = Act(id="test_1", play_id="test", number=1, scenes=[scene])
    return Play(
        id="test", title="Test Play", author="Test",
        acts=[act], characters=["A", "B", "C"],
    )


class TestCollectPairwiseData:
    def test_directed_pairs(self):
        play = _make_play_with_social_data()
        data = _collect_pairwise_social_data(play)
        # (A, B) and (B, A) should be separate entries
        assert ("A", "B") in data
        assert ("B", "A") in data

    def test_records_correct_values(self):
        play = _make_play_with_social_data()
        data = _collect_pairwise_social_data(play)
        # A's warmth toward B in beat 1 should be 0.8
        a_to_b = data[("A", "B")]
        warmths = [r[1] for r in a_to_b]
        assert warmths[0] == pytest.approx(0.8)

    def test_different_partners_separate(self):
        play = _make_play_with_social_data()
        data = _collect_pairwise_social_data(play)
        # A->B has 3 records (beats 1-3), A->C has 1 record (beat 4)
        assert len(data[("A", "B")]) == 3
        assert len(data[("A", "C")]) == 1


class TestBuildPairwiseEdges:
    def test_min_beats_filter(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=3)
        chars = {(e.character_a, e.character_b) for e in edges}
        assert ("A", "B") in chars
        assert ("B", "A") in chars
        # A->C only has 1 co-occurrence, should be filtered out
        assert ("A", "C") not in chars

    def test_temperature_by_beat(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=1)
        a_to_b = next(e for e in edges if e.character_a == "A" and e.character_b == "B")
        assert len(a_to_b.temperature_by_beat) == 3
        assert a_to_b.temperature_by_beat["b1"] == pytest.approx(0.8)

    def test_directed(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=1)
        a_to_b = next(e for e in edges if e.character_a == "A" and e.character_b == "B")
        b_to_a = next(e for e in edges if e.character_a == "B" and e.character_b == "A")
        # A is warm to B, B is cold to A
        assert list(a_to_b.temperature_by_beat.values())[0] > 0
        assert list(b_to_a.temperature_by_beat.values())[0] < 0


class TestBuildRelationalProfiles:
    def test_profiles_created(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=1)
        profiles = build_relational_profiles(play, edges)
        char_names = {p.character for p in profiles}
        assert "A" in char_names
        assert "B" in char_names

    def test_default_warmth_aggregated(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=1)
        profiles = build_relational_profiles(play, edges)
        a_profile = next(p for p in profiles if p.character == "A")
        # A is warm to B (0.8, 0.6, 0.7 avg ~0.7) and to C (0.2)
        # Default warmth should be average of per-partner means
        assert a_profile.default_warmth > 0  # A is generally warm

    def test_partner_deviations(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=1)
        profiles = build_relational_profiles(play, edges)
        a_profile = next(p for p in profiles if p.character == "A")
        # A should have deviations for B and C
        assert "B" in a_profile.partner_deviations
        assert "C" in a_profile.partner_deviations
        # B deviation should be positive (warmer than default)
        assert a_profile.partner_deviations["B"]["warmth_delta"] > 0

    def test_variance_captures_spread(self):
        play = _make_play_with_social_data()
        edges = build_pairwise_edges(play, min_beats=1)
        profiles = build_relational_profiles(play, edges)
        a_profile = next(p for p in profiles if p.character == "A")
        # A treats B very differently from C, so variance should be non-zero
        assert a_profile.warmth_variance > 0


class TestRealPlayData:
    """Test with actual play data on disk."""

    @pytest.fixture
    def cherry_play(self):
        from config import PARSED_DIR
        path = PARSED_DIR / "cherry_orchard.json"
        if not path.exists():
            pytest.skip("Cherry Orchard data not on disk")
        return Play.model_validate_json(path.read_text())

    def test_edges_populated(self, cherry_play):
        assert len(cherry_play.relationship_edges) > 0

    def test_lopakhin_has_multiple_partners(self, cherry_play):
        """Lopakhin should have relationship deviations for multiple partners."""
        from analysis.relationship_builder import load_profiles
        try:
            profiles = load_profiles("cherry_orchard")
        except FileNotFoundError:
            pytest.skip("Profiles not built yet")
        lopakhin = next((p for p in profiles if p.character == "LOPAKHIN"), None)
        if lopakhin is None:
            pytest.skip("No Lopakhin profile")
        assert len(lopakhin.partner_deviations) >= 5
        assert "LUBOV" in lopakhin.partner_deviations

    def test_yasha_is_cold(self, cherry_play):
        """Yasha should have negative default warmth (the coldest character)."""
        from analysis.relationship_builder import load_profiles
        try:
            profiles = load_profiles("cherry_orchard")
        except FileNotFoundError:
            pytest.skip("Profiles not built yet")
        yasha = next((p for p in profiles if p.character == "YASHA"), None)
        if yasha is None:
            pytest.skip("No Yasha profile")
        assert yasha.default_warmth < 0
