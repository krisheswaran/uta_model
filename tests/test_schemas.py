"""Test schema construction, validation, and serialization."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    AffectState, BeatState, BeatStateEstimate, CanonicalTactic,
    CharacterBible, EpistemicState, ImprovTurn, Play, RelationalProfile,
    RevisionTrace, SceneContext, ScoredLine, SocialState,
    StatisticalPrior, TacticVocabulary, CandidateLine,
)


class TestBeatState:
    def test_minimal_construction(self):
        bs = BeatState(beat_id="b1", character="HAMLET")
        assert bs.tactic_state == ""
        assert bs.canonical_tactic is None
        assert bs.confidence == 1.0

    def test_canonical_tactic_optional(self):
        bs = BeatState(beat_id="b1", character="HAMLET", canonical_tactic="DEFLECT")
        assert bs.canonical_tactic == "DEFLECT"

    def test_affect_clamping(self):
        with pytest.raises(Exception):
            AffectState(valence=2.0)  # out of range

    def test_roundtrip_json(self, sample_beat_state):
        data = sample_beat_state.model_dump_json()
        restored = BeatState.model_validate_json(data)
        assert restored.beat_id == sample_beat_state.beat_id
        assert restored.tactic_state == "probe"
        assert restored.affect_state.valence == pytest.approx(0.2)


class TestCharacterBible:
    def test_minimal_construction(self):
        cb = CharacterBible(play_id="test", character="X")
        assert cb.tactic_distribution == {}
        assert cb.few_shot_lines == []

    def test_roundtrip_json(self, sample_character_bible):
        data = sample_character_bible.model_dump_json()
        restored = CharacterBible.model_validate_json(data)
        assert restored.character == "TEST_CHAR"
        assert restored.tactic_distribution["probe"] == 10


class TestRevisionTrace:
    def test_construction(self):
        rt = RevisionTrace(
            round=0,
            candidate_text="Hello there.",
            scores={"voice_fidelity": 4.0, "tactic_fidelity": 3.5},
            feedback=["Consider more subtext"],
        )
        assert rt.round == 0
        assert len(rt.feedback) == 1

    def test_empty_defaults(self):
        rt = RevisionTrace(round=0, candidate_text="Hi")
        assert rt.scores == {}
        assert rt.feedback == []


class TestImprovTurnWithTrace:
    def test_revision_trace_default_empty(self, sample_beat_state, sample_scene_context):
        scored = ScoredLine(
            candidate=CandidateLine(text="test"),
            voice_fidelity=3.0, tactic_fidelity=3.0, knowledge_fidelity=3.0,
            relationship_fidelity=3.0, subtext_richness=3.0,
            emotional_transition_plausibility=3.0,
        )
        turn = ImprovTurn(
            turn_index=1,
            context=sample_scene_context,
            initial_beat_state=sample_beat_state,
            final_line="Test line",
            revisions=0,
            scored_line=scored,
            updated_beat_state=sample_beat_state,
        )
        assert turn.revision_trace == []

    def test_revision_trace_populated(self, sample_beat_state, sample_scene_context):
        scored = ScoredLine(
            candidate=CandidateLine(text="test"),
            voice_fidelity=3.0, tactic_fidelity=3.0, knowledge_fidelity=3.0,
            relationship_fidelity=3.0, subtext_richness=3.0,
            emotional_transition_plausibility=3.0,
        )
        trace = [
            RevisionTrace(round=0, candidate_text="Draft 1",
                          scores={"voice_fidelity": 2.5}, feedback=["Try harder"]),
            RevisionTrace(round=1, candidate_text="Draft 2",
                          scores={"voice_fidelity": 3.5}, feedback=[]),
        ]
        turn = ImprovTurn(
            turn_index=1,
            context=sample_scene_context,
            initial_beat_state=sample_beat_state,
            final_line="Draft 2",
            revisions=1,
            scored_line=scored,
            updated_beat_state=sample_beat_state,
            revision_trace=trace,
        )
        assert len(turn.revision_trace) == 2
        assert turn.revision_trace[0].candidate_text == "Draft 1"


class TestPhaseBSchemas:
    def test_canonical_tactic(self):
        ct = CanonicalTactic(
            canonical_id="DEFLECT", canonical_verb="deflect",
            description="redirect attention", members=["deflect", "redirect"],
        )
        assert len(ct.members) == 2

    def test_tactic_vocabulary_lookup(self, sample_vocabulary):
        assert sample_vocabulary.lookup("deflect") == "DEFLECT"
        assert sample_vocabulary.lookup("redirect") == "DEFLECT"
        assert sample_vocabulary.lookup("taunt") == "MOCK"
        assert sample_vocabulary.lookup("unknown_tactic") is None

    def test_tactic_vocabulary_lookup_case_insensitive(self, sample_vocabulary):
        assert sample_vocabulary.lookup("DEFLECT") == "DEFLECT"
        assert sample_vocabulary.lookup("Mock") == "MOCK"

    def test_beat_state_estimate(self):
        est = BeatStateEstimate(beat_id="b1", character="X")
        assert est.estimates == []
        assert est.consensus_confidence == 1.0

    def test_relational_profile(self):
        rp = RelationalProfile(
            character="LOPAKHIN", play_id="cherry_orchard",
            default_status_claim=-0.3, default_warmth=0.5,
            status_variance=0.2, warmth_variance=0.4,
            partner_deviations={
                "RANYEVSKAYA": {"status_delta": -0.2, "warmth_delta": 0.3},
            },
        )
        assert rp.partner_deviations["RANYEVSKAYA"]["warmth_delta"] == 0.3

    def test_statistical_prior(self, sample_vocabulary):
        sp = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.4, "MOCK": 0.3, "PROBE": 0.3},
            tactic_transition_matrix={"DEFLECT": {"DEFLECT": 0.5, "MOCK": 0.3, "PROBE": 0.2}},
        )
        assert sp.character_tactic_prior["DEFLECT"] == pytest.approx(0.4)
        assert sp.tactic_vocabulary.lookup("redirect") == "DEFLECT"
