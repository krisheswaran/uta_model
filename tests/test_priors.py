"""Test statistical priors and dramaturgical feedback."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    AffectState, BeatState, CharacterBible, SocialState, StatisticalPrior,
)


class TestComputeTacticDeviation:
    def test_on_target(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.40, "MOCK": 0.30, "PROBE": 0.30},
        )
        dev = compute_tactic_deviation("deflect", prior)
        assert dev["canonical_id"] == "DEFLECT"
        assert dev["deviation_tier"] == 1
        assert dev["prior_probability"] == pytest.approx(0.40)

    def test_mild_deviation(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.50, "MOCK": 0.30, "PROBE": 0.02},
        )
        # PROBE at 2% should be tier 2 (mild)
        dev = compute_tactic_deviation("probe", prior)
        assert dev["deviation_tier"] == 2

    def test_sharp_deviation(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.50, "MOCK": 0.30, "PROBE": 0.01},
        )
        # PROBE at 1% should be tier 3 (sharp)
        dev = compute_tactic_deviation("probe", prior)
        assert dev["deviation_tier"] == 3

    def test_unmapped_tactic(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.50},
        )
        dev = compute_tactic_deviation("zygomorphize", prior)
        assert dev["canonical_id"] is None
        assert dev["deviation_tier"] == 2  # unmapped = mild

    def test_none_tactic(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.50},
        )
        dev = compute_tactic_deviation(None, prior)
        assert dev["canonical_id"] is None

    def test_transition_escalation(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.10, "MOCK": 0.10, "PROBE": 0.10},
            tactic_transition_matrix={
                "DEFLECT": {"DEFLECT": 0.80, "MOCK": 0.01, "PROBE": 0.19},
            },
        )
        # MOCK after DEFLECT: prior_p=0.10 (tier 1) but transition_p=0.01 (escalates to tier 2)
        dev = compute_tactic_deviation("mock", prior, previous_tactic="deflect")
        assert dev["transition_probability"] == pytest.approx(0.01)
        assert dev["deviation_tier"] >= 2

    def test_top_tactic_reported(self, sample_vocabulary):
        from improv.priors import compute_tactic_deviation
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.50, "MOCK": 0.30, "PROBE": 0.20},
        )
        dev = compute_tactic_deviation("mock", prior)
        assert dev["top_tactic"] == "DEFLECT"
        assert dev["top_tactic_pct"] == pytest.approx(50.0)


class TestDramaturgicalFeedback:
    @pytest.fixture
    def prior_with_vocab(self, sample_vocabulary):
        return StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.50, "MOCK": 0.30, "PROBE": 0.20},
            tactic_transition_matrix={},
        )

    def test_tier1_returns_encouragement(self, prior_with_vocab, sample_character_bible):
        from improv.priors import generate_dramaturgical_feedback
        bs = BeatState(beat_id="t", character="TEST_CHAR", tactic_state="deflect")
        feedback = generate_dramaturgical_feedback(bs, sample_character_bible, prior_with_vocab)
        assert len(feedback) >= 1
        # Should be encouraging, not critical
        assert any("instinct" in f.lower() or "lands" in f.lower() for f in feedback)

    def test_tier3_demands_justification(self, sample_vocabulary, sample_character_bible):
        from improv.priors import generate_dramaturgical_feedback
        prior = StatisticalPrior(
            tactic_vocabulary=sample_vocabulary,
            character_tactic_prior={"DEFLECT": 0.80, "MOCK": 0.15, "PROBE": 0.005},
        )
        bs = BeatState(beat_id="t", character="TEST_CHAR", tactic_state="probe")
        feedback = generate_dramaturgical_feedback(bs, sample_character_bible, prior)
        assert len(feedback) >= 1
        # Should be forceful
        assert any("significant break" in f.lower() or "comfort zone" in f.lower()
                    for f in feedback)

    def test_no_prior_returns_empty(self, sample_character_bible):
        from improv.priors import generate_dramaturgical_feedback
        bs = BeatState(beat_id="t", character="TEST_CHAR", tactic_state="deflect")
        # Empty prior with no tactic data
        prior = StatisticalPrior()
        feedback = generate_dramaturgical_feedback(bs, sample_character_bible, prior)
        # Should not crash, may return empty or minimal feedback
        assert isinstance(feedback, list)


class TestLoadPriorFromDisk:
    """Integration test: load prior from actual data files."""

    def test_load_hamlet_prior(self):
        from config import PARSED_DIR
        if not (PARSED_DIR / "hamlet.json").exists():
            pytest.skip("Hamlet data not on disk")
        try:
            from improv.priors import load_prior_for_character
            prior = load_prior_for_character("hamlet", "HAMLET")
        except FileNotFoundError:
            pytest.skip("Vocabulary or profiles not built")

        assert len(prior.character_tactic_prior) > 0
        assert len(prior.tactic_transition_matrix) > 0
        assert prior.relational_profile is not None
        assert prior.relational_profile.character == "HAMLET"
        # Hamlet's top tactic should be MOCK or DEFLECT
        top = max(prior.character_tactic_prior, key=prior.character_tactic_prior.get)
        assert top in ("MOCK", "DEFLECT", "PROVOKE", "PROBE")
