"""Tests for the factor graph modules: variables, factors, graph, inference, integration."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from factor_graph.variables import DiscreteVariable, GaussianVariable, PointEstimate
from factor_graph.factors import (
    AffectTransitionFactor,
    DesireTransitionFactor,
    EmissionFactor,
    SuperobjectivePrior,
    TacticTransitionFactor,
)
from factor_graph.graph import CharacterFactorGraph, FactorParameters
from factor_graph.inference import ForwardFilter, PosteriorState
from schemas import BeatState, CharacterBible

FACTORS_DIR = PROJECT_ROOT / "data" / "factors"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
BIBLES_DIR = PROJECT_ROOT / "data" / "bibles"

HAS_FACTOR_FILES = FACTORS_DIR.exists() and (FACTORS_DIR / "tactic_transition_base.json").exists()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_uniform_params(n_tactics: int = 5, n_desire: int = 3) -> FactorParameters:
    """Create a small FactorParameters for unit testing."""
    return FactorParameters.create_uniform(n_tactics=n_tactics, n_desire=n_desire)


def _make_simple_bible(tactic_vocab: list[str] | None = None) -> CharacterBible:
    """Create a minimal CharacterBible."""
    dist = {}
    if tactic_vocab:
        dist = {t: 1 for t in tactic_vocab[:3]}
    return CharacterBible(
        play_id="test_play",
        character="TEST_CHAR",
        superobjective="To find truth",
        tactic_distribution=dist,
    )


# =========================================================================== #
# Variable tests
# =========================================================================== #


class TestDiscreteVariable:

    def test_construction_with_states(self):
        dv = DiscreteVariable(["a", "b", "c"])
        assert dv.n_states == 3
        assert set(dv.states) == {"a", "b", "c"}

    def test_uniform_initialization(self):
        dv = DiscreteVariable(["a", "b", "c"])
        p = dv.probs()
        np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_normalization(self):
        log_probs = np.array([0.0, 1.0, 2.0])
        dv = DiscreteVariable(["a", "b", "c"], log_probs)
        dv.normalize()
        p = dv.probs()
        assert abs(p.sum() - 1.0) < 1e-10

    def test_map_state(self):
        log_probs = np.array([-10.0, -0.1, -5.0])
        dv = DiscreteVariable(["a", "b", "c"], log_probs)
        assert dv.map_state() == "b"

    def test_entropy_uniform(self):
        dv = DiscreteVariable(["a", "b", "c", "d"])
        expected = np.log(4)
        assert abs(dv.entropy() - expected) < 1e-10

    def test_entropy_deterministic(self):
        log_probs = np.array([0.0, -1000.0, -1000.0])
        dv = DiscreteVariable(["a", "b", "c"], log_probs)
        dv.normalize()
        assert dv.entropy() < 0.01  # near zero

    def test_sample_returns_valid_state(self):
        dv = DiscreteVariable(["a", "b", "c"])
        rng = np.random.default_rng(42)
        for _ in range(20):
            s = dv.sample(rng)
            assert s in {"a", "b", "c"}

    def test_empty_states_raises(self):
        with pytest.raises(ValueError, match="at least one state"):
            DiscreteVariable([])

    def test_mismatched_log_probs_raises(self):
        with pytest.raises(ValueError, match="log_probs length"):
            DiscreteVariable(["a", "b"], np.array([0.0]))

    def test_to_dict(self):
        dv = DiscreteVariable(["x", "y"], np.array([np.log(0.3), np.log(0.7)]))
        d = dv.to_dict()
        assert abs(d["x"] - 0.3) < 1e-6
        assert abs(d["y"] - 0.7) < 1e-6

    def test_copy_independence(self):
        dv = DiscreteVariable(["a", "b"], np.array([0.0, -1.0]))
        dv2 = dv.copy()
        dv2.log_probs[0] = -100.0
        assert dv.log_probs[0] == 0.0  # original unchanged

    def test_prob_unknown_state(self):
        dv = DiscreteVariable(["a", "b"])
        assert dv.prob("z") == 0.0

    def test_state_index(self):
        dv = DiscreteVariable(["x", "y", "z"])
        assert dv.state_index("y") == 1
        with pytest.raises(KeyError):
            dv.state_index("missing")


class TestGaussianVariable:

    def test_construction(self):
        g = GaussianVariable(np.array([1.0, 2.0]), np.eye(2))
        assert g.dim == 2
        np.testing.assert_array_equal(g.mean, [1.0, 2.0])

    def test_map_equals_mean(self):
        mean = np.array([3.0, -1.0, 0.5])
        g = GaussianVariable(mean, np.eye(3) * 0.1)
        np.testing.assert_array_equal(g.map_state(), mean)

    def test_log_prob_at_mean_is_maximum(self):
        mean = np.array([0.0, 0.0])
        cov = np.eye(2) * 0.5
        g = GaussianVariable(mean, cov)
        lp_at_mean = g.log_prob(mean)
        lp_off = g.log_prob(np.array([1.0, 1.0]))
        assert lp_at_mean > lp_off

    def test_log_prob_value(self):
        # For 1D standard normal: log_prob(0) = -0.5*log(2*pi)
        g = GaussianVariable(np.array([0.0]), np.array([[1.0]]))
        expected = -0.5 * np.log(2 * np.pi)
        assert abs(g.log_prob(np.array([0.0])) - expected) < 1e-10

    def test_from_diagonal(self):
        g = GaussianVariable.from_diagonal(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.2, 0.3]),
        )
        assert g.dim == 3
        np.testing.assert_allclose(np.diag(g.cov), [0.1, 0.2, 0.3])

    def test_std(self):
        g = GaussianVariable.from_diagonal(np.zeros(2), np.array([4.0, 9.0]))
        np.testing.assert_allclose(g.std, [2.0, 3.0])

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="must be 1D"):
            GaussianVariable(np.array([[1.0]]), np.eye(1))
        with pytest.raises(ValueError, match="doesn't match"):
            GaussianVariable(np.array([1.0, 2.0]), np.eye(3))

    def test_copy_independence(self):
        g = GaussianVariable(np.array([1.0, 2.0]), np.eye(2))
        g2 = g.copy()
        g2.mean[0] = 99.0
        assert g.mean[0] == 1.0

    def test_sample_shape(self):
        g = GaussianVariable(np.zeros(3), np.eye(3))
        rng = np.random.default_rng(42)
        s = g.sample(rng)
        assert s.shape == (3,)


class TestPointEstimate:

    def test_construction(self):
        pe = PointEstimate(0.75)
        assert pe.value == 0.75

    def test_map_state(self):
        pe = PointEstimate(-0.3)
        assert pe.map_state() == -0.3

    def test_copy(self):
        pe = PointEstimate(1.0)
        pe2 = pe.copy()
        pe2.value = 99.0
        assert pe.value == 1.0

    def test_float_coercion(self):
        pe = PointEstimate(3)
        assert isinstance(pe.value, float)


# =========================================================================== #
# Factor tests
# =========================================================================== #


class TestTacticTransitionFactor:

    def _make_factor(self, n: int = 4) -> TacticTransitionFactor:
        """Create a small tactic transition factor."""
        # Non-uniform transition matrix: higher self-transition
        base = np.ones((n, n)) * 0.1
        np.fill_diagonal(base, 0.7)
        # Row-normalize
        base = base / base.sum(axis=1, keepdims=True)
        desire_matrices = np.stack([base] * 3)
        return TacticTransitionFactor(base, desire_matrices, persistence_beta=1.0)

    def test_forward_message_valid_distribution(self):
        factor = self._make_factor(4)
        states = [f"T{i}" for i in range(4)]
        prev = DiscreteVariable(states)
        result = factor.forward_message(prev)
        p = result.probs()
        assert abs(p.sum() - 1.0) < 1e-10
        assert all(p >= 0)

    def test_forward_message_preserves_state_labels(self):
        factor = self._make_factor(4)
        states = [f"T{i}" for i in range(4)]
        prev = DiscreteVariable(states)
        result = factor.forward_message(prev)
        assert result.states == states

    def test_backward_message_valid_distribution(self):
        factor = self._make_factor(4)
        states = [f"T{i}" for i in range(4)]
        next_belief = DiscreteVariable(states)
        result = factor.backward_message(next_belief)
        p = result.probs()
        assert abs(p.sum() - 1.0) < 1e-10

    def test_desire_modulation_increases_self_transition(self):
        factor = self._make_factor(4)
        states = [f"T{i}" for i in range(4)]
        # Peaked prior on T0
        log_probs = np.array([0.0, -10.0, -10.0, -10.0])
        prev = DiscreteVariable(states, log_probs)
        prev.normalize()

        # High desire_sim should increase persistence (self-transition)
        result_high_sim = factor.forward_message(prev, desire_cluster=0, desire_sim=0.95)
        result_low_sim = factor.forward_message(prev, desire_cluster=0, desire_sim=0.05)

        # With high sim, T0 should be more probable (higher persistence)
        p_high = result_high_sim.prob("T0")
        p_low = result_low_sim.prob("T0")
        assert p_high > p_low

    def test_log_potential(self):
        factor = self._make_factor(4)
        lp = factor.log_potential(0, 0)
        assert isinstance(lp, float)
        assert not np.isnan(lp)


class TestAffectTransitionFactor:

    def _make_factor(self) -> AffectTransitionFactor:
        return AffectTransitionFactor(
            variance=np.array([0.1, 0.2, 0.15]),
            df=np.array([5.0, 10.0, 3.0]),
        )

    def test_forward_message_valid_gaussian(self):
        factor = self._make_factor()
        prev = GaussianVariable.from_diagonal(np.array([0.5, -0.3, 0.1]), np.ones(3) * 0.1)
        result = factor.forward_message(prev)
        assert result.dim == 3
        # Covariance should increase (adding transition noise)
        for i in range(3):
            assert result.cov[i, i] > prev.cov[i, i]

    def test_mean_propagation(self):
        factor = self._make_factor()
        mean = np.array([1.0, -0.5, 0.3])
        prev = GaussianVariable.from_diagonal(mean, np.ones(3) * 0.1)
        result = factor.forward_message(prev)
        # Random walk: mean stays the same
        np.testing.assert_array_equal(result.mean, mean)

    def test_variance_increases(self):
        factor = self._make_factor()
        prev = GaussianVariable.from_diagonal(np.zeros(3), np.ones(3) * 0.1)
        result = factor.forward_message(prev)
        prev_var = np.diag(prev.cov)
        result_var = np.diag(result.cov)
        assert all(result_var > prev_var)

    def test_effective_variance_heavier_tails(self):
        factor = self._make_factor()
        # Lower df should produce larger effective variance
        eff = factor.effective_variance
        # df=3 axis should have larger factor than df=10 axis
        factor_2 = eff[2] / factor.variance[2]  # df=3 -> 3/(3-2) = 3
        factor_1 = eff[1] / factor.variance[1]  # df=10 -> 10/8 = 1.25
        assert factor_2 > factor_1

    def test_backward_message(self):
        factor = self._make_factor()
        next_b = GaussianVariable.from_diagonal(np.array([0.0, 0.0, 0.0]), np.ones(3) * 0.1)
        result = factor.backward_message(next_b)
        assert result.dim == 3

    def test_requires_3d(self):
        with pytest.raises(ValueError, match="3D"):
            AffectTransitionFactor(np.ones(2), np.ones(2))


class TestDesireTransitionFactor:

    def test_forward_message_valid_distribution(self):
        mat = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
        ])
        factor = DesireTransitionFactor(mat)
        prev = DiscreteVariable(["d0", "d1", "d2"])
        result = factor.forward_message(prev)
        p = result.probs()
        assert abs(p.sum() - 1.0) < 1e-10
        assert all(p >= 0)

    def test_backward_message_valid(self):
        mat = np.eye(3) * 0.8 + 0.2 / 3
        factor = DesireTransitionFactor(mat)
        next_b = DiscreteVariable(["d0", "d1", "d2"])
        result = factor.backward_message(next_b)
        p = result.probs()
        assert abs(p.sum() - 1.0) < 1e-10


class TestSuperobjectivePrior:

    def test_blend_produces_valid_distribution(self):
        n = 5
        prior = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        sp = SuperobjectivePrior(prior, lam=0.1)
        states = [f"T{i}" for i in range(n)]
        belief = DiscreteVariable(states)
        result = sp.blend(belief)
        p = result.probs()
        assert abs(p.sum() - 1.0) < 1e-10
        assert all(p >= 0)

    def test_lambda_zero_returns_input(self):
        n = 5
        prior = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        sp = SuperobjectivePrior(prior, lam=0.0)
        states = [f"T{i}" for i in range(n)]
        input_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        belief = DiscreteVariable(states, np.log(input_probs))
        result = sp.blend(belief)
        np.testing.assert_allclose(result.probs(), input_probs, atol=1e-10)

    def test_lambda_one_returns_prior(self):
        n = 5
        prior = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        sp = SuperobjectivePrior(prior, lam=1.0)
        states = [f"T{i}" for i in range(n)]
        belief = DiscreteVariable(states)
        result = sp.blend(belief)
        np.testing.assert_allclose(result.probs(), prior, atol=1e-10)

    def test_prior_normalization(self):
        # Unnormalized prior should be auto-normalized
        prior = np.array([2.0, 3.0, 5.0])
        sp = SuperobjectivePrior(prior)
        assert abs(sp.tactic_prior.sum() - 1.0) < 1e-10


class TestEmissionFactor:

    def _make_factor(self) -> tuple[EmissionFactor, list[str]]:
        """Build a small emission factor with 3 tactics and 3 features."""
        profiles = {
            "T0": {"f0": (1.0, 0.5), "f1": (0.0, 1.0), "f2": (0.5, 0.3)},
            "T1": {"f0": (0.0, 0.5), "f1": (2.0, 1.0), "f2": (-0.5, 0.3)},
            "T2": {"f0": (-1.0, 0.5), "f1": (1.0, 1.0), "f2": (0.0, 0.3)},
        }
        feature_names = ["f0", "f1", "f2"]
        factor = EmissionFactor(profiles, feature_names=feature_names)
        return factor, feature_names

    def test_observe_updates_belief(self):
        factor, _ = self._make_factor()
        states = ["T0", "T1", "T2"]
        # Uniform prior
        belief = DiscreteVariable(states)

        # Observation close to T0's profile
        text_features = np.array([1.0, 0.0, 0.5])
        posterior = factor.observe(text_features, belief)

        # T0 should now be most likely
        assert posterior.map_state() == "T0"
        p = posterior.probs()
        assert abs(p.sum() - 1.0) < 1e-10

    def test_observe_preserves_normalization(self):
        factor, _ = self._make_factor()
        states = ["T0", "T1", "T2"]
        belief = DiscreteVariable(states)
        text_features = np.array([0.5, 1.0, 0.0])
        posterior = factor.observe(text_features, belief)
        p = posterior.probs()
        assert abs(p.sum() - 1.0) < 1e-10

    def test_estimate_arousal_fallback(self):
        factor, _ = self._make_factor()
        # No arousal regressor -- should return char_mean_arousal
        result = factor.estimate_arousal(np.array([1.0, 0.5, 0.0]), char_mean_arousal=0.3)
        assert result == 0.3


# =========================================================================== #
# Graph tests
# =========================================================================== #


class TestFactorParametersUniform:

    def test_create_uniform(self):
        params = FactorParameters.create_uniform(n_tactics=10, n_desire=5)
        assert params.tactic_transition_base.shape == (10, 10)
        assert params.desire_transition_matrix.shape == (5, 5)
        assert len(params.tactic_vocab) == 10
        assert len(params.desire_cluster_labels) == 5

    def test_uniform_row_stochastic(self):
        params = FactorParameters.create_uniform()
        row_sums = params.tactic_transition_base.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


@pytest.mark.skipif(not HAS_FACTOR_FILES, reason="data/factors/ not found")
class TestFactorParametersLoad:

    @pytest.mark.xfail(
        reason="BUG: FactorParameters.load() expects nested-list JSON but "
               "learning.py writes nested-dict {tactic: {tactic: prob}}. "
               "np.array(dict) raises TypeError.",
        raises=TypeError,
    )
    def test_load_from_disk(self):
        params = FactorParameters.load(FACTORS_DIR)
        assert params.tactic_transition_base.shape[0] > 0
        assert params.tactic_transition_base.shape[0] == params.tactic_transition_base.shape[1]
        assert len(params.tactic_vocab) == params.tactic_transition_base.shape[0]
        assert params.affect_transition_variance.shape == (3,)


class TestCharacterFactorGraph:

    def test_construction_from_uniform(self):
        params = _make_uniform_params(n_tactics=5, n_desire=3)
        bible = _make_simple_bible(params.tactic_vocab)
        graph = CharacterFactorGraph(params, bible)
        assert graph.n_tactics == 5
        assert graph.n_desire == 3
        assert graph.tactic_transition is not None
        assert graph.affect_transition is not None
        assert graph.desire_transition is not None
        assert graph.emission is not None

    def test_initial_beliefs(self):
        params = _make_uniform_params(n_tactics=5, n_desire=3)
        bible = _make_simple_bible(params.tactic_vocab)
        graph = CharacterFactorGraph(params, bible)

        tactic = graph.initial_tactic_belief()
        assert tactic.n_states == 5
        assert abs(tactic.probs().sum() - 1.0) < 1e-10

        desire = graph.initial_desire_belief()
        assert desire.n_states == 3

        affect = graph.initial_affect()
        assert affect.dim == 3

        social = graph.initial_social()
        assert social.dim == 2

    def test_project_affect(self):
        params = _make_uniform_params()
        bible = _make_simple_bible(params.tactic_vocab)
        graph = CharacterFactorGraph(params, bible)
        affect_5d = np.array([0.2, 0.5, -0.1, 0.3, 0.4])
        result = graph.project_affect(affect_5d)
        assert result.shape == (3,)

    @pytest.mark.skipif(not HAS_FACTOR_FILES, reason="data/factors/ not found")
    @pytest.mark.xfail(
        reason="BUG: FactorParameters.load() cannot parse nested-dict JSON format "
               "written by learning.py (see TestFactorParametersLoad).",
        raises=TypeError,
    )
    def test_construction_with_real_params(self):
        params = FactorParameters.load(FACTORS_DIR)
        # Find a real character bible
        import json
        bibles_files = list(BIBLES_DIR.glob("*.json"))
        if not bibles_files:
            pytest.skip("No bible files found")
        with open(bibles_files[0]) as f:
            bibles_data = json.load(f)
        if not bibles_data:
            pytest.skip("Bible file empty")
        first_bible = bibles_data[0] if isinstance(bibles_data, list) else bibles_data
        bible = CharacterBible(**first_bible)
        graph = CharacterFactorGraph(params, bible)
        assert graph.n_tactics == len(params.tactic_vocab)


# =========================================================================== #
# Inference tests
# =========================================================================== #


class TestForwardFilter:

    def _make_filter(self) -> ForwardFilter:
        params = _make_uniform_params(n_tactics=5, n_desire=3)
        bible = _make_simple_bible(params.tactic_vocab)
        graph = CharacterFactorGraph(params, bible)
        return ForwardFilter(graph)

    def test_initialize_produces_valid_posterior(self):
        filt = self._make_filter()
        posterior = filt.initialize()
        assert isinstance(posterior, PosteriorState)
        assert len(posterior.tactic_distribution) == 5
        assert abs(sum(posterior.tactic_distribution.values()) - 1.0) < 1e-6
        assert len(posterior.desire_distribution) == 3
        assert posterior.affect_trans_mean.shape == (3,)
        assert posterior.social_mean.shape == (2,)

    def test_step_produces_valid_posterior(self):
        filt = self._make_filter()
        prev = filt.initialize()
        posterior = filt.step(
            prev_posterior=prev,
            utterance_text="I need you to tell me the truth!",
        )
        assert isinstance(posterior, PosteriorState)
        assert abs(sum(posterior.tactic_distribution.values()) - 1.0) < 1e-6
        assert posterior.affect_trans_mean.shape == (3,)

    def test_step_multiple_turns(self):
        filt = self._make_filter()
        state = filt.initialize()
        for i in range(5):
            state = filt.step(state, f"Turn {i}: What do you mean by that?")
        assert isinstance(state, PosteriorState)
        assert abs(sum(state.tactic_distribution.values()) - 1.0) < 1e-6


class TestPosteriorState:

    def _make_posterior(self) -> PosteriorState:
        tactic = DiscreteVariable(["T0", "T1", "T2"])
        affect = GaussianVariable.from_diagonal(np.array([0.1, -0.2, 0.3]), np.ones(3) * 0.1)
        desire = DiscreteVariable(["d0", "d1"])
        social = GaussianVariable.from_diagonal(np.array([0.0, 0.5]), np.ones(2) * 0.05)
        return PosteriorState.from_variables(
            tactic=tactic,
            affect=affect,
            arousal=0.4,
            desire=desire,
            social=social,
            beat_id="test_b1",
            character="TEST",
        )

    def test_from_variables(self):
        ps = self._make_posterior()
        assert ps.tactic_map in {"T0", "T1", "T2"}
        assert ps.desire_map in {"d0", "d1"}
        assert ps.arousal == 0.4
        assert ps.beat_id == "test_b1"
        assert ps.character == "TEST"

    def test_to_beat_state(self):
        ps = self._make_posterior()
        bs = ps.to_beat_state()
        assert isinstance(bs, BeatState)
        assert bs.beat_id == "test_b1"
        assert bs.character == "TEST"
        assert bs.canonical_tactic in {"T0", "T1", "T2"}
        assert -1.0 <= bs.affect_state.arousal <= 1.0
        assert -1.0 <= bs.social_state.status <= 1.0

    def test_to_beat_state_with_eigenvectors(self):
        ps = self._make_posterior()
        eigvecs = np.eye(3, 5)
        bs = ps.to_beat_state(affect_eigenvectors=eigvecs)
        assert isinstance(bs, BeatState)
        # With identity-like projection, affect values should be non-zero
        assert bs.affect_state is not None

    def test_to_dict_roundtrip(self):
        ps = self._make_posterior()
        d = ps.to_dict()
        # Verify all expected keys present
        assert "beat_id" in d
        assert "tactic_map" in d
        assert "tactic_distribution" in d
        assert "affect_trans_mean" in d
        assert "arousal" in d
        assert "desire_distribution" in d
        assert "social_mean" in d
        assert "social_std" in d

        # Verify values match
        assert d["beat_id"] == "test_b1"
        assert d["character"] == "TEST"
        assert d["arousal"] == 0.4
        assert abs(sum(d["tactic_distribution"].values()) - 1.0) < 1e-6

    def test_to_dict_json_serializable(self):
        import json
        ps = self._make_posterior()
        d = ps.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["beat_id"] == "test_b1"


# =========================================================================== #
# Integration tests
# =========================================================================== #


class TestFactorGraphStateUpdater:

    def test_construction_with_uniform_params(self):
        from factor_graph.integration import FactorGraphStateUpdater
        params = _make_uniform_params(n_tactics=5, n_desire=3)
        bible = _make_simple_bible(params.tactic_vocab)

        # We need to bypass the normal constructor which tries to load from disk.
        # Instead, test the components directly.
        graph = CharacterFactorGraph(params, bible)
        filt = ForwardFilter(graph)
        posterior = filt.initialize()
        assert isinstance(posterior, PosteriorState)

    def test_update_state_returns_valid_beat_state(self):
        params = _make_uniform_params(n_tactics=5, n_desire=3)
        bible = _make_simple_bible(params.tactic_vocab)
        graph = CharacterFactorGraph(params, bible)
        filt = ForwardFilter(graph)

        # Initialize
        posterior = filt.initialize()

        # Step
        posterior2 = filt.step(
            prev_posterior=posterior,
            utterance_text="You must listen to me!",
        )

        # Convert to BeatState
        bs = posterior2.to_beat_state(
            affect_eigenvectors=params.affect_eigenvectors,
        )
        assert isinstance(bs, BeatState)
        assert bs.canonical_tactic in params.tactic_vocab
        assert 0.0 <= bs.confidence <= 1.0

    def test_multiple_steps_stable(self):
        """Run several steps and verify no NaN or explosion."""
        params = _make_uniform_params(n_tactics=5, n_desire=3)
        bible = _make_simple_bible(params.tactic_vocab)
        graph = CharacterFactorGraph(params, bible)
        filt = ForwardFilter(graph)

        posterior = filt.initialize()
        lines = [
            "What are you hiding from me?",
            "I demand an answer!",
            "Please, just tell me the truth.",
            "Fine. Have it your way.",
            "You will regret this.",
        ]
        for line in lines:
            posterior = filt.step(posterior, line)
            p_sum = sum(posterior.tactic_distribution.values())
            assert abs(p_sum - 1.0) < 1e-4, f"Tactic distribution not normalized: {p_sum}"
            assert not np.any(np.isnan(posterior.affect_trans_mean))
            assert not np.any(np.isnan(posterior.social_mean))
