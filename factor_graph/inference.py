"""
Forward and forward-backward inference algorithms for the factor graph.

Two inference modes:
  - ForwardFilter: one-step causal filtering (Pass 2 improv, <100ms per beat)
  - ForwardBackwardSmoother: full smoothing over a sequence (Pass 1.5)

Both operate on the factored state representation:
  - Exact discrete inference over T×D (66×7 = 462 joint states)
  - Kalman-style Gaussian updates for A_trans and S (conditioned on discrete)
  - Point estimate for A_emit (arousal from text features)

The LLM-extracted BeatState observations are treated as noisy measurements
with a simple confusion model: P(observed_tactic | true_tactic) = 0.7 for
match, (1-0.7)/65 spread across others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from factor_graph.graph import CharacterFactorGraph
from factor_graph.variables import DiscreteVariable, GaussianVariable, PointEstimate
from schemas import AffectState, BeatState, SocialState


# --------------------------------------------------------------------------- #
# Observation model constants
# --------------------------------------------------------------------------- #

# Confusion model for LLM tactic extraction
TACTIC_OBSERVATION_ACCURACY = 0.7  # P(observed | true) when they match
# P(observed | true) when they don't match: spread uniformly
# TACTIC_OBSERVATION_NOISE = (1 - 0.7) / 65 ≈ 0.00462

# Observation noise for affect dimensions (std dev of LLM extraction noise)
AFFECT_OBSERVATION_STD = 0.15

# Observation noise for social dimensions
SOCIAL_OBSERVATION_STD = 0.2


# --------------------------------------------------------------------------- #
# PosteriorState — output of inference at one beat
# --------------------------------------------------------------------------- #

@dataclass
class PosteriorState:
    """The output of inference at one beat.

    Preserves full distributions for analysis and visualization,
    while providing MAP point estimates for generation prompts.

    Fields
    ------
    tactic_distribution : dict[str, float]
        Full posterior over canonical tactics.
    tactic_map : str
        Most probable tactic (MAP estimate).
    affect_trans_mean : NDArray (3,)
        Mean of affect in eigenspace.
    affect_trans_std : NDArray (3,)
        Standard deviation of affect in eigenspace.
    arousal : float
        Point estimate from text features.
    desire_distribution : dict[str, float]
        Posterior over desire clusters.
    desire_map : str
        Most probable desire cluster.
    social_mean : NDArray (2,)
        Mean of (status, warmth).
    social_std : NDArray (2,)
        Standard deviation of (status, warmth).
    beat_id : str
        The beat this posterior corresponds to.
    character : str
        The character this posterior corresponds to.
    """

    tactic_distribution: dict[str, float] = field(default_factory=dict)
    tactic_map: str = ""
    affect_trans_mean: NDArray = field(default_factory=lambda: np.zeros(3))
    affect_trans_std: NDArray = field(default_factory=lambda: np.zeros(3))
    arousal: float = 0.0
    desire_distribution: dict[str, float] = field(default_factory=dict)
    desire_map: str = ""
    social_mean: NDArray = field(default_factory=lambda: np.zeros(2))
    social_std: NDArray = field(default_factory=lambda: np.zeros(2))
    beat_id: str = ""
    character: str = ""

    @classmethod
    def from_variables(
        cls,
        tactic: DiscreteVariable,
        affect: GaussianVariable,
        arousal: float,
        desire: DiscreteVariable,
        social: GaussianVariable,
        beat_id: str = "",
        character: str = "",
    ) -> PosteriorState:
        """Construct from variable nodes."""
        return cls(
            tactic_distribution=tactic.to_dict(),
            tactic_map=tactic.map_state(),
            affect_trans_mean=affect.mean.copy(),
            affect_trans_std=affect.std.copy(),
            arousal=arousal,
            desire_distribution=desire.to_dict(),
            desire_map=desire.map_state(),
            social_mean=social.mean.copy(),
            social_std=social.std.copy(),
            beat_id=beat_id,
            character=character,
        )

    def to_beat_state(
        self,
        method: str = "map",
        affect_eigenvectors: NDArray | None = None,
    ) -> BeatState:
        """Convert posterior to BeatState for generation prompts.

        Parameters
        ----------
        method : str
            "map" for maximum a posteriori, "sample" for sampling.
        affect_eigenvectors : (3, 5) NDArray | None
            Rotation matrix to project eigenspace back to 5D affect.
            If None, fills affect with zeros (eigenspace only).
        """
        # Tactic
        if method == "sample":
            tactic_var = DiscreteVariable(
                list(self.tactic_distribution.keys()),
                np.log(np.maximum(
                    list(self.tactic_distribution.values()), 1e-300
                )),
            )
            tactic = tactic_var.sample()
        else:
            tactic = self.tactic_map

        # Affect: project back from eigenspace to 5D
        if affect_eigenvectors is not None:
            # Pseudo-inverse of (3,5) → (5,3) to go back
            R_pinv = np.linalg.pinv(affect_eigenvectors)
            affect_5d = R_pinv @ self.affect_trans_mean
            # Clamp to valid ranges
            valence = float(np.clip(affect_5d[0], -1, 1))
            arousal_val = float(np.clip(self.arousal, -1, 1))
            certainty = float(np.clip(affect_5d[2], -1, 1))
            control = float(np.clip(affect_5d[3], -1, 1))
            vulnerability = float(np.clip(affect_5d[4], 0, 1))
        else:
            valence = 0.0
            arousal_val = float(np.clip(self.arousal, -1, 1))
            certainty = 0.0
            control = 0.0
            vulnerability = 0.0

        affect = AffectState(
            valence=valence,
            arousal=arousal_val,
            certainty=certainty,
            control=control,
            vulnerability=vulnerability,
        )

        social = SocialState(
            status=float(np.clip(self.social_mean[0], -1, 1)),
            warmth=float(np.clip(self.social_mean[1], -1, 1)),
        )

        return BeatState(
            beat_id=self.beat_id,
            character=self.character,
            canonical_tactic=tactic,
            tactic_state=tactic.lower().replace("_", " ") if tactic else "",
            affect_state=affect,
            social_state=social,
            confidence=float(max(self.tactic_distribution.values()))
            if self.tactic_distribution else 1.0,
        )

    def to_dict(self) -> dict:
        """Serialize for JSON output (data/smoothed/)."""
        return {
            "beat_id": self.beat_id,
            "character": self.character,
            "tactic_distribution": self.tactic_distribution,
            "tactic_map": self.tactic_map,
            "affect_trans_mean": self.affect_trans_mean.tolist(),
            "affect_trans_std": self.affect_trans_std.tolist(),
            "arousal": self.arousal,
            "desire_distribution": self.desire_distribution,
            "desire_map": self.desire_map,
            "social_mean": self.social_mean.tolist(),
            "social_std": self.social_std.tolist(),
        }


# --------------------------------------------------------------------------- #
# Helper: BeatState observation model
# --------------------------------------------------------------------------- #

def _observe_tactic(
    belief: DiscreteVariable,
    observed_tactic: str | None,
    accuracy: float = TACTIC_OBSERVATION_ACCURACY,
) -> DiscreteVariable:
    """Update tactic belief with a noisy LLM observation.

    Confusion model: P(observed | true) = accuracy if match,
    (1-accuracy)/(n-1) if mismatch.

    If observed_tactic is None or not in the vocabulary, returns
    the belief unchanged (missing data).
    """
    if observed_tactic is None:
        return belief.copy()

    try:
        obs_idx = belief.state_index(observed_tactic)
    except KeyError:
        # Unknown tactic — treat as missing observation
        return belief.copy()

    n = belief.n_states
    noise = (1.0 - accuracy) / max(n - 1, 1)

    # log P(obs | true=i) for each true state i
    log_likelihood = np.full(n, np.log(max(noise, 1e-300)))
    log_likelihood[obs_idx] = np.log(accuracy)

    new_log_probs = belief.log_probs + log_likelihood
    result = DiscreteVariable(belief.states, new_log_probs)
    result.normalize()
    return result


def _observe_affect(
    belief: GaussianVariable,
    observed: NDArray,
    obs_std: float = AFFECT_OBSERVATION_STD,
) -> GaussianVariable:
    """Kalman-style update of affect belief with a noisy observation.

    observation model: y = x + ε, ε ~ N(0, σ²I)
    """
    d = belief.dim
    obs_cov = np.eye(d) * (obs_std ** 2)

    # Kalman gain: K = Σ_prior @ (Σ_prior + Σ_obs)^{-1}
    S = belief.cov + obs_cov
    K = belief.cov @ np.linalg.inv(S)

    innovation = observed - belief.mean
    new_mean = belief.mean + K @ innovation
    new_cov = (np.eye(d) - K) @ belief.cov

    return GaussianVariable(new_mean, new_cov)


def _observe_social(
    belief: GaussianVariable,
    observed: NDArray,
    obs_std: float = SOCIAL_OBSERVATION_STD,
) -> GaussianVariable:
    """Kalman-style update of social belief with a noisy observation."""
    return _observe_affect(belief, observed, obs_std)


def _extract_text_features_from_utterances(utterances: list[str]) -> NDArray | None:
    """Extract the 9 text features from a list of utterance strings.

    Features (matching EmissionFactor.DEFAULT_FEATURES):
      word_count, question_density, exclamation_density,
      imperative_density, mean_sentence_length, lexical_diversity,
      first_person_rate, second_person_rate, sentiment_polarity

    Returns None if utterances is empty.
    """
    if not utterances:
        return None

    combined = " ".join(utterances)
    words = combined.split()
    word_count = len(words)

    if word_count == 0:
        return None

    # Sentence splitting (simple heuristic)
    sentences = [s.strip() for s in combined.replace("!", ".").replace("?", ".").split(".")
                 if s.strip()]
    n_sentences = max(len(sentences), 1)

    # Feature computation
    question_density = combined.count("?") / n_sentences
    exclamation_density = combined.count("!") / n_sentences

    # Imperative heuristic: sentences starting with a verb-like word
    imperative_starters = {
        "come", "go", "tell", "give", "let", "take", "make", "do",
        "stop", "look", "listen", "speak", "leave", "stay", "wait",
        "be", "have", "get", "say", "think", "know", "see", "hear",
    }
    imperative_count = sum(
        1 for s in sentences
        if s.split() and s.split()[0].lower().rstrip(",;:") in imperative_starters
    )
    imperative_density = imperative_count / n_sentences

    mean_sentence_length = word_count / n_sentences

    # Lexical diversity: type-token ratio
    unique_words = set(w.lower().strip(".,!?;:'\"()") for w in words)
    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0

    # Pronoun rates
    words_lower = [w.lower().strip(".,!?;:'\"()") for w in words]
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    second_person = {"you", "your", "yours", "yourself", "yourselves"}
    first_person_rate = sum(1 for w in words_lower if w in first_person) / word_count
    second_person_rate = sum(1 for w in words_lower if w in second_person) / word_count

    # Sentiment polarity (very simple: positive minus negative word count)
    # This is a crude approximation; the learned emission profiles will
    # compensate for systematic bias
    positive = {
        "good", "love", "happy", "joy", "wonderful", "beautiful", "great",
        "excellent", "dear", "sweet", "kind", "hope", "please", "thank",
    }
    negative = {
        "bad", "hate", "sad", "terrible", "awful", "horrible", "wrong",
        "never", "nothing", "miserable", "fear", "death", "die", "kill",
    }
    pos_count = sum(1 for w in words_lower if w in positive)
    neg_count = sum(1 for w in words_lower if w in negative)
    sentiment_polarity = (pos_count - neg_count) / word_count

    return np.array([
        word_count, question_density, exclamation_density,
        imperative_density, mean_sentence_length, lexical_diversity,
        first_person_rate, second_person_rate, sentiment_polarity,
    ], dtype=np.float64)


def _affect_from_beat_state(beat_state: BeatState) -> NDArray:
    """Extract 5D affect vector from a BeatState."""
    a = beat_state.affect_state
    return np.array([a.valence, a.arousal, a.certainty, a.control, a.vulnerability],
                    dtype=np.float64)


def _social_from_beat_state(beat_state: BeatState) -> NDArray:
    """Extract 2D social vector from a BeatState."""
    s = beat_state.social_state
    return np.array([s.status, s.warmth], dtype=np.float64)


# --------------------------------------------------------------------------- #
# ForwardFilter — one-step causal inference (Pass 2)
# --------------------------------------------------------------------------- #

class ForwardFilter:
    """Runs one-step forward inference for Pass 2 improv.

    Each call to step() takes the previous posterior and current observations,
    and produces the updated posterior. Designed to run in <100ms per beat.

    Parameters
    ----------
    graph : CharacterFactorGraph
        The assembled factor graph for this character.
    """

    def __init__(self, graph: CharacterFactorGraph):
        self.graph = graph
        self._prev_desire_embedding: NDArray | None = None

    def initialize(self, partner: str | None = None) -> PosteriorState:
        """Create the initial posterior (t=0) before any observations.

        Uses character-level priors from the bible and relational profile.
        """
        tactic = self.graph.initial_tactic_belief()

        # Apply superobjective prior to initial tactic
        if self.graph.superobjective_prior is not None:
            tactic = self.graph.superobjective_prior.blend(tactic)

        affect = self.graph.initial_affect()
        desire = self.graph.initial_desire_belief()
        social = self.graph.initial_social(partner)

        return PosteriorState.from_variables(
            tactic=tactic,
            affect=affect,
            arousal=0.0,
            desire=desire,
            social=social,
            character=self.graph.character_bible.character,
        )

    def step(
        self,
        prev_posterior: PosteriorState,
        utterance_text: str | list[str],
        desire_state: str | None = None,
        characters_present: list[str] | None = None,
        desire_cluster: int | None = None,
        desire_sim: float | None = None,
        partner_status: float | None = None,
    ) -> PosteriorState:
        """Run one forward inference step.

        Parameters
        ----------
        prev_posterior : PosteriorState
            Posterior from the previous beat.
        utterance_text : str or list[str]
            The utterance(s) just delivered.
        desire_state : str | None
            Current desire state text (for desire cluster assignment).
        characters_present : list[str] | None
            Characters in the scene (for social coupling).
        desire_cluster : int | None
            Pre-computed desire cluster index. If None, uses uniform.
        desire_sim : float | None
            Cosine similarity between prev and current desire embeddings.
        partner_status : float | None
            Partner's current status claim (for coupling).

        Returns
        -------
        PosteriorState for the current beat.
        """
        # Reconstruct variable nodes from previous posterior
        prev_tactic = DiscreteVariable(
            list(prev_posterior.tactic_distribution.keys()),
            np.log(np.maximum(
                list(prev_posterior.tactic_distribution.values()), 1e-300
            )),
        )
        prev_affect = GaussianVariable.from_diagonal(
            prev_posterior.affect_trans_mean,
            prev_posterior.affect_trans_std ** 2,
        )
        prev_desire = DiscreteVariable(
            list(prev_posterior.desire_distribution.keys()),
            np.log(np.maximum(
                list(prev_posterior.desire_distribution.values()), 1e-300
            )),
        )

        # --- 1. PREDICT: transition factors ---

        # Tactic prediction
        tactic_pred = self.graph.tactic_transition.forward_message(
            prev_tactic, desire_cluster=desire_cluster, desire_sim=desire_sim
        )

        # Affect prediction (Student-t kernel, Gaussian approximation)
        affect_pred = self.graph.affect_transition.forward_message(prev_affect)

        # Desire prediction
        desire_pred = self.graph.desire_transition.forward_message(prev_desire)

        # --- 2. CONDITION: superobjective prior ---
        if self.graph.superobjective_prior is not None:
            tactic_pred = self.graph.superobjective_prior.blend(tactic_pred)

        # --- 3. OBSERVE: text features ---
        if isinstance(utterance_text, str):
            utterances = [utterance_text]
        else:
            utterances = list(utterance_text)

        text_features = _extract_text_features_from_utterances(utterances)

        if text_features is not None:
            # Update tactic belief via emission model
            tactic_post = self.graph.emission.observe(text_features, tactic_pred)

            # Estimate arousal from text features
            char_mean_arousal = prev_posterior.arousal  # Use prev as fallback
            arousal = self.graph.emission.estimate_arousal(
                text_features, char_mean_arousal
            )
        else:
            tactic_post = tactic_pred
            arousal = prev_posterior.arousal

        # --- 4. COUPLE: cross-character status coupling ---
        # Social state starts from the relational prior
        primary_partner = None
        if characters_present:
            others = [c for c in characters_present
                      if c.upper() != self.graph.character_bible.character.upper()]
            if others:
                primary_partner = others[0]

        social_post = self.graph.initial_social(primary_partner)

        if partner_status is not None:
            own_status = float(social_post.mean[0])
            adj_own, _ = self.graph.status_coupling.coupled_update(
                own_status, partner_status
            )
            social_post.mean[0] = adj_own

        return PosteriorState.from_variables(
            tactic=tactic_post,
            affect=affect_pred,
            arousal=arousal,
            desire=desire_pred,
            social=social_post,
            character=self.graph.character_bible.character,
        )


# --------------------------------------------------------------------------- #
# ForwardBackwardSmoother — full smoothing (Pass 1.5)
# --------------------------------------------------------------------------- #

class ForwardBackwardSmoother:
    """Runs full forward-backward over a sequence for Pass 1.5 smoothing.

    Takes the LLM-extracted BeatStates as noisy observations and produces
    smoothed posteriors that reconcile them with learned transition dynamics.

    The smoother operates on the factored state:
      - Exact forward-backward for discrete variables (tactic × desire)
      - Kalman smoother for continuous variables (affect, social)

    Parameters
    ----------
    graph : CharacterFactorGraph
        The assembled factor graph for this character.
    """

    def __init__(self, graph: CharacterFactorGraph):
        self.graph = graph

    def smooth(
        self,
        beat_states: list[BeatState],
        utterances: list[list[str]],
        desire_clusters: list[int | None] | None = None,
        desire_sims: list[float | None] | None = None,
    ) -> list[PosteriorState]:
        """Run full forward-backward smoothing over a sequence.

        Parameters
        ----------
        beat_states : list[BeatState]
            LLM-extracted BeatStates (observations), one per beat.
        utterances : list[list[str]]
            List of utterance texts per beat.
        desire_clusters : list[int | None] | None
            Pre-computed desire cluster assignments per beat.
        desire_sims : list[float | None] | None
            Desire similarity between consecutive beats.

        Returns
        -------
        list[PosteriorState]
            Smoothed posteriors for each beat.
        """
        T = len(beat_states)
        if T == 0:
            return []

        if len(utterances) != T:
            raise ValueError(
                f"beat_states length {T} != utterances length {len(utterances)}"
            )

        if desire_clusters is None:
            desire_clusters = [None] * T
        if desire_sims is None:
            desire_sims = [None] * T

        # ================================================================== #
        # FORWARD PASS
        # ================================================================== #
        forward_tactics: list[DiscreteVariable] = []
        forward_desires: list[DiscreteVariable] = []
        forward_affects: list[GaussianVariable] = []
        forward_socials: list[GaussianVariable] = []
        forward_arousals: list[float] = []

        for t in range(T):
            bs = beat_states[t]

            if t == 0:
                # Initialize from character priors
                tactic_pred = self.graph.initial_tactic_belief()
                desire_pred = self.graph.initial_desire_belief()
                affect_pred = self.graph.initial_affect()
            else:
                # Predict from previous forward belief
                dc = desire_clusters[t] if desire_clusters else None
                ds = desire_sims[t] if desire_sims else None

                tactic_pred = self.graph.tactic_transition.forward_message(
                    forward_tactics[t - 1], desire_cluster=dc, desire_sim=ds
                )
                desire_pred = self.graph.desire_transition.forward_message(
                    forward_desires[t - 1]
                )
                affect_pred = self.graph.affect_transition.forward_message(
                    forward_affects[t - 1]
                )

            # Condition: superobjective prior
            if self.graph.superobjective_prior is not None:
                tactic_pred = self.graph.superobjective_prior.blend(tactic_pred)

            # Observe: LLM-extracted BeatState (tactic as noisy observation)
            # Use canonical_tactic if available; fall back to uppercased tactic_state
            observed_tactic = bs.canonical_tactic
            if not observed_tactic and bs.tactic_state:
                observed_tactic = bs.tactic_state.upper()
            tactic_post = _observe_tactic(tactic_pred, observed_tactic)

            # Observe: text features from utterances
            text_features = _extract_text_features_from_utterances(utterances[t])
            if text_features is not None:
                tactic_post = self.graph.emission.observe(text_features, tactic_post)
                char_mean = bs.affect_state.arousal if bs.affect_state else 0.0
                arousal = self.graph.emission.estimate_arousal(
                    text_features, char_mean
                )
            else:
                arousal = bs.affect_state.arousal if bs.affect_state else 0.0

            # Observe: affect (project to eigenspace, then Kalman update)
            affect_5d = _affect_from_beat_state(bs)
            affect_3d = self.graph.project_affect(affect_5d)
            affect_post = _observe_affect(affect_pred, affect_3d)

            # Observe: social state
            social_prior = self.graph.initial_social()
            social_obs = _social_from_beat_state(bs)
            social_post = _observe_social(social_prior, social_obs)

            # Desire: observe if canonical_tactic gives signal about desire
            # (For now, just use the forward prediction — desire observation
            # model requires embeddings which aren't available here)
            desire_post = desire_pred

            forward_tactics.append(tactic_post)
            forward_desires.append(desire_post)
            forward_affects.append(affect_post)
            forward_socials.append(social_post)
            forward_arousals.append(arousal)

        # ================================================================== #
        # BACKWARD PASS
        # ================================================================== #
        # Initialize backward messages as uniform/non-informative
        backward_tactics: list[DiscreteVariable] = [
            DiscreteVariable(self.graph.tactic_vocab)
        ] * T
        backward_desires: list[DiscreteVariable] = [
            DiscreteVariable(self.graph.desire_labels)
        ] * T
        backward_affects: list[GaussianVariable] = [
            GaussianVariable.from_diagonal(np.zeros(3), np.ones(3) * 100.0)
        ] * T

        # Backward pass: t = T-1 down to 0
        for t in range(T - 2, -1, -1):
            # The backward message at t is computed from the smoothed belief at t+1
            # passed backward through the transition factor.

            dc = desire_clusters[t + 1] if desire_clusters else None
            ds = desire_sims[t + 1] if desire_sims else None

            # Tactic backward: combine forward at t+1 with backward at t+1,
            # then pass backward through transition
            # Smoothed at t+1 ∝ forward[t+1] * backward[t+1]
            smoothed_t1_log = (forward_tactics[t + 1].log_probs
                               + backward_tactics[t + 1].log_probs)
            smoothed_t1 = DiscreteVariable(
                self.graph.tactic_vocab, smoothed_t1_log
            )
            smoothed_t1.normalize()

            backward_tactics[t] = self.graph.tactic_transition.backward_message(
                smoothed_t1, desire_cluster=dc, desire_sim=ds
            )

            # Desire backward
            smoothed_d1_log = (forward_desires[t + 1].log_probs
                               + backward_desires[t + 1].log_probs)
            smoothed_d1 = DiscreteVariable(
                self.graph.desire_labels, smoothed_d1_log
            )
            smoothed_d1.normalize()
            backward_desires[t] = self.graph.desire_transition.backward_message(
                smoothed_d1
            )

            # Affect backward (Gaussian)
            backward_affects[t] = self.graph.affect_transition.backward_message(
                forward_affects[t + 1]
            )

        # ================================================================== #
        # COMBINE: smoothed posterior = forward × backward
        # ================================================================== #
        posteriors: list[PosteriorState] = []

        for t in range(T):
            # Tactic: combine forward and backward in log-space
            smoothed_tactic_log = (forward_tactics[t].log_probs
                                   + backward_tactics[t].log_probs)
            smoothed_tactic = DiscreteVariable(
                self.graph.tactic_vocab, smoothed_tactic_log
            )
            smoothed_tactic.normalize()

            # Desire: combine forward and backward
            smoothed_desire_log = (forward_desires[t].log_probs
                                   + backward_desires[t].log_probs)
            smoothed_desire = DiscreteVariable(
                self.graph.desire_labels, smoothed_desire_log
            )
            smoothed_desire.normalize()

            # Affect: Gaussian product (forward × backward)
            fwd = forward_affects[t]
            bwd = backward_affects[t]
            # Product of two Gaussians: N(μ1, Σ1) × N(μ2, Σ2)
            # Σ_post = (Σ1^{-1} + Σ2^{-1})^{-1}
            # μ_post = Σ_post (Σ1^{-1} μ1 + Σ2^{-1} μ2)
            fwd_prec = np.linalg.inv(fwd.cov)
            bwd_prec = np.linalg.inv(bwd.cov)
            post_cov = np.linalg.inv(fwd_prec + bwd_prec)
            post_mean = post_cov @ (fwd_prec @ fwd.mean + bwd_prec @ bwd.mean)
            smoothed_affect = GaussianVariable(post_mean, post_cov)

            # Social: use forward (no temporal dynamics modeled for social)
            smoothed_social = forward_socials[t]

            # Arousal: point estimate (no smoothing — near-IID)
            arousal = forward_arousals[t]

            posterior = PosteriorState.from_variables(
                tactic=smoothed_tactic,
                affect=smoothed_affect,
                arousal=arousal,
                desire=smoothed_desire,
                social=smoothed_social,
                beat_id=beat_states[t].beat_id,
                character=beat_states[t].character,
            )
            posteriors.append(posterior)

        return posteriors
