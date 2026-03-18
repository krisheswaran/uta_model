"""
Factor node types for the factor graph.

Each factor computes log-potentials (unnormalized log-probabilities) and
produces forward/backward messages for inference. Factors encode the
learned transition dynamics, emission likelihoods, and priors from the
implementation plan (FACTOR_GRAPH_IMPLEMENTATION.md §2).

Factor types:
  - TacticTransitionFactor (ψ_T): P(T(t) | T(t-1), desire_cluster, desire_sim)
  - AffectTransitionFactor (ψ_A): P(A_trans(t) | A_trans(t-1)), Student-t kernel
  - DesireTransitionFactor (ψ_D): P(D(t) | D(t-1))
  - SuperobjectivePrior (ψ_arc): soft additive tactic bias from superobjective
  - SocialPrior (ψ_S): Gaussian prior on (status, warmth) from relational profile
  - EmissionFactor (ψ_emit): P(text_features | tactic, arousal)
  - StatusCouplingFactor (ψ_social): cross-character status coupling
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from factor_graph.variables import DiscreteVariable, GaussianVariable


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _log_matrix_vector_product(log_matrix: NDArray, log_vec: NDArray) -> NDArray:
    """Compute log(M @ exp(log_vec)) stably using log-sum-exp.

    Parameters
    ----------
    log_matrix : (n, m)
        Log of transition matrix rows: log_matrix[i, j] = log P(j | i).
    log_vec : (m,)
        Log-probabilities of the source states.

    Returns
    -------
    (n,) log-probabilities of the target states.
    """
    # result[j] = log sum_i exp(log_matrix[i, j] + log_vec[i])
    # = log sum_i exp(log_joint[i, j])
    # We want P(T_t = j) = sum_i P(T_t = j | T_{t-1} = i) * P(T_{t-1} = i)
    # log_joint[i, j] = log P(j|i) + log P(i)
    log_joint = log_matrix + log_vec[:, np.newaxis]  # (m, n) — rows=source, cols=target
    # log-sum-exp over source dimension (axis 0)
    max_per_col = np.max(log_joint, axis=0)
    result = max_per_col + np.log(np.sum(np.exp(log_joint - max_per_col[np.newaxis, :]), axis=0))
    return result


def _safe_log(x: NDArray | float, floor: float = 1e-300) -> NDArray:
    """Log with floor to avoid -inf."""
    return np.log(np.maximum(x, floor))


# --------------------------------------------------------------------------- #
# TacticTransitionFactor
# --------------------------------------------------------------------------- #

class TacticTransitionFactor:
    """ψ_T: P(T(t) | T(t-1), desire_cluster, desire_similarity).

    Combines:
      1. Base 66×66 transition matrix
      2. Desire-cluster-specific transition matrices (7 × 66×66)
      3. Desire-similarity persistence modulation on diagonal

    Parameters
    ----------
    base_matrix : (66, 66)
        Smoothed base transition probabilities. Row i = P(T_t | T_{t-1}=i).
    desire_matrices : (7, 66, 66)
        Per-desire-cluster transition matrices.
    persistence_beta : float
        Learned β for desire-similarity persistence modulation.
        Modulates self-transition: P[i,i] *= 1 + β*(sim - 0.5).
    """

    def __init__(
        self,
        base_matrix: NDArray,
        desire_matrices: NDArray,
        persistence_beta: float,
    ):
        self.base_matrix = np.asarray(base_matrix, dtype=np.float64)
        self.desire_matrices = np.asarray(desire_matrices, dtype=np.float64)
        self.persistence_beta = float(persistence_beta)

        n = self.base_matrix.shape[0]
        self.n_tactics = n
        self.log_base = _safe_log(self.base_matrix)
        self.log_desire = _safe_log(self.desire_matrices)

    def _get_log_transition(
        self, desire_cluster: int | None, desire_sim: float | None
    ) -> NDArray:
        """Build the effective log-transition matrix for a given context.

        If desire_cluster is provided, uses the cluster-specific matrix;
        otherwise falls back to the base matrix. Desire similarity modulates
        the diagonal (persistence).
        """
        if desire_cluster is not None and 0 <= desire_cluster < len(self.desire_matrices):
            log_mat = self.log_desire[desire_cluster].copy()
        else:
            log_mat = self.log_base.copy()

        # Apply persistence modulation if desire similarity is available
        if desire_sim is not None and self.persistence_beta != 0.0:
            mod = 1.0 + self.persistence_beta * (desire_sim - 0.5)
            mod = max(mod, 0.1)  # Floor to avoid zeroing out diagonal
            diag_idx = np.arange(self.n_tactics)
            # Multiply diagonal in log-space: add log(mod)
            log_mat[diag_idx, diag_idx] += np.log(mod)
            # Re-normalize each row in log-space
            for i in range(self.n_tactics):
                max_val = np.max(log_mat[i])
                log_mat[i] -= max_val + np.log(np.sum(np.exp(log_mat[i] - max_val)))

        return log_mat

    def log_potential(
        self,
        prev_tactic_idx: int,
        curr_tactic_idx: int,
        desire_cluster: int | None = None,
        desire_sim: float | None = None,
    ) -> float:
        """Log-potential for a specific (prev, curr) tactic pair."""
        log_mat = self._get_log_transition(desire_cluster, desire_sim)
        return float(log_mat[prev_tactic_idx, curr_tactic_idx])

    def forward_message(
        self,
        prev_belief: DiscreteVariable,
        desire_cluster: int | None = None,
        desire_sim: float | None = None,
    ) -> DiscreteVariable:
        """Compute predicted belief over T(t) given belief over T(t-1).

        P(T_t = j) = sum_i P(T_t = j | T_{t-1} = i) * P(T_{t-1} = i)
        """
        log_mat = self._get_log_transition(desire_cluster, desire_sim)
        log_result = _log_matrix_vector_product(log_mat, prev_belief.log_probs)
        result = DiscreteVariable(prev_belief.states, log_result)
        result.normalize()
        return result

    def backward_message(
        self,
        next_belief: DiscreteVariable,
        desire_cluster: int | None = None,
        desire_sim: float | None = None,
    ) -> DiscreteVariable:
        """Compute backward message: contribution of future to T(t-1).

        β(T_{t-1} = i) = sum_j P(T_t = j | T_{t-1} = i) * β_next(T_t = j)
        """
        log_mat = self._get_log_transition(desire_cluster, desire_sim)
        # For each source state i: sum over target states j
        # log_result[i] = logsumexp_j(log_mat[i, j] + next_belief.log_probs[j])
        log_joint = log_mat + next_belief.log_probs[np.newaxis, :]  # (n, n)
        max_per_row = np.max(log_joint, axis=1)
        log_result = max_per_row + np.log(
            np.sum(np.exp(log_joint - max_per_row[:, np.newaxis]), axis=1)
        )
        result = DiscreteVariable(next_belief.states, log_result)
        result.normalize()
        return result


# --------------------------------------------------------------------------- #
# AffectTransitionFactor
# --------------------------------------------------------------------------- #

class AffectTransitionFactor:
    """ψ_A: P(A_trans(t) | A_trans(t-1)) — Student-t transition kernel in eigenspace.

    Uses a diagonal Student-t distribution in the 3D affect eigenspace,
    providing heavier tails than Gaussian to accommodate dramatic jumps
    (revelations, betrayals).

    For message passing, we use a Gaussian approximation (moment-matching)
    since exact Student-t message passing is intractable. The heavier tails
    manifest as larger effective variance.

    Parameters
    ----------
    variance : (3,)
        Diagonal variances for the transition kernel in eigenspace.
    df : (3,)
        Degrees of freedom for each eigenspace dimension.
        Lower df = heavier tails. df > 30 ≈ Gaussian.
    """

    def __init__(self, variance: NDArray, df: NDArray):
        self.variance = np.asarray(variance, dtype=np.float64)
        self.df = np.asarray(df, dtype=np.float64)
        if len(self.variance) != 3 or len(self.df) != 3:
            raise ValueError("Affect eigenspace is 3D: need 3 variances and 3 dfs")

    @property
    def effective_variance(self) -> NDArray:
        """Effective variance of the Student-t: σ² * df/(df-2) for df > 2.

        This is the variance used for the Gaussian approximation in
        message passing. Heavier tails (lower df) inflate the variance.
        """
        result = self.variance.copy()
        for i in range(3):
            if self.df[i] > 2:
                result[i] *= self.df[i] / (self.df[i] - 2)
            else:
                # Very heavy tails — use a large but finite variance
                result[i] *= 100.0
        return result

    def forward_message(self, prev: GaussianVariable) -> GaussianVariable:
        """Predict A_trans(t) from A_trans(t-1).

        The predicted distribution is Gaussian with:
          mean = prev.mean (random walk)
          cov = prev.cov + diag(effective_variance)  (accumulated uncertainty)
        """
        transition_cov = np.diag(self.effective_variance)
        new_cov = prev.cov + transition_cov
        return GaussianVariable(prev.mean.copy(), new_cov)

    def backward_message(self, next_belief: GaussianVariable) -> GaussianVariable:
        """Backward message: what the future tells us about A_trans(t).

        Given that A_trans(t+1) ~ N(A_trans(t), Σ_trans), and we know
        the smoothed belief about t+1, we can infer about t:
          mean_backward = next.mean  (random walk reversal)
          cov_backward = next.cov + diag(effective_variance)
        """
        transition_cov = np.diag(self.effective_variance)
        new_cov = next_belief.cov + transition_cov
        return GaussianVariable(next_belief.mean.copy(), new_cov)

    def log_transition_density(self, prev: NDArray, curr: NDArray) -> float:
        """Log-density of the Student-t transition from prev to curr.

        Computed as a product of independent univariate Student-t densities.
        """
        from scipy.stats import t as student_t

        delta = curr - prev
        log_p = 0.0
        for i in range(3):
            scale = np.sqrt(self.variance[i])
            log_p += student_t.logpdf(delta[i], df=self.df[i], loc=0, scale=scale)
        return float(log_p)


# --------------------------------------------------------------------------- #
# DesireTransitionFactor
# --------------------------------------------------------------------------- #

class DesireTransitionFactor:
    """ψ_D: P(D(t) | D(t-1)) — 7×7 transition matrix over desire clusters.

    Desire transitions are sticky within a scene (high diagonal) with
    occasional shifts at dramatic turning points.

    Parameters
    ----------
    transition_matrix : (7, 7)
        Row-stochastic: transition_matrix[i, j] = P(D_t = j | D_{t-1} = i).
    """

    def __init__(self, transition_matrix: NDArray):
        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        n = self.transition_matrix.shape[0]
        self.n_clusters = n
        self.log_matrix = _safe_log(self.transition_matrix)

    def forward_message(self, prev: DiscreteVariable) -> DiscreteVariable:
        """Predict D(t) from D(t-1)."""
        log_result = _log_matrix_vector_product(self.log_matrix, prev.log_probs)
        result = DiscreteVariable(prev.states, log_result)
        result.normalize()
        return result

    def backward_message(self, next_belief: DiscreteVariable) -> DiscreteVariable:
        """Backward message for desire variable."""
        log_joint = self.log_matrix + next_belief.log_probs[np.newaxis, :]
        max_per_row = np.max(log_joint, axis=1)
        log_result = max_per_row + np.log(
            np.sum(np.exp(log_joint - max_per_row[:, np.newaxis]), axis=1)
        )
        result = DiscreteVariable(next_belief.states, log_result)
        result.normalize()
        return result


# --------------------------------------------------------------------------- #
# SuperobjectivePrior
# --------------------------------------------------------------------------- #

class SuperobjectivePrior:
    """ψ_arc: soft additive prior on tactic distribution from superobjective.

    Blends the transition-based tactic prediction with a character-level
    tactic prior derived from the superobjective cluster. Uses additive
    mixture with small λ — self-regulating: nearly invisible when the
    transition model is confident, fills the gap when uncertain.

    Parameters
    ----------
    tactic_prior : (66,)
        Probability vector from superobjective cluster (sums to 1).
    lam : float
        Blending weight. Default 0.06 (matching 6.25% information gain
        from superobjective predictiveness experiment).
    """

    def __init__(self, tactic_prior: NDArray, lam: float = 0.06):
        self.tactic_prior = np.asarray(tactic_prior, dtype=np.float64)
        self.lam = float(lam)
        # Ensure prior is normalized
        s = np.sum(self.tactic_prior)
        if s > 0:
            self.tactic_prior = self.tactic_prior / s
        else:
            self.tactic_prior = np.ones_like(self.tactic_prior) / len(self.tactic_prior)
        self.log_prior = _safe_log(self.tactic_prior)

    def blend(self, tactic_belief: DiscreteVariable) -> DiscreteVariable:
        """Apply additive mixture: (1-λ)P_transition + λP_arc.

        Operates in probability space (additive mixture), then converts
        back to log-space.
        """
        p_trans = tactic_belief.probs()
        p_blended = (1.0 - self.lam) * p_trans + self.lam * self.tactic_prior
        result = DiscreteVariable(tactic_belief.states, _safe_log(p_blended))
        result.normalize()
        return result


# --------------------------------------------------------------------------- #
# SocialPrior
# --------------------------------------------------------------------------- #

class SocialPrior:
    """ψ_S: Gaussian prior on (status, warmth) from relational profile.

    Not a transition model — social state doesn't have strong temporal
    dynamics within a scene. This is a per-beat prior conditioned on
    who is in the scene.

    Parameters
    ----------
    mean : (2,)
        [status_mean, warmth_mean] from relational profile.
    std : (2,)
        [status_std, warmth_std] from relational profile.
    """

    def __init__(self, mean: NDArray, std: NDArray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if len(self.mean) != 2 or len(self.std) != 2:
            raise ValueError("Social state is 2D: (status, warmth)")
        # Ensure positive standard deviations
        self.std = np.maximum(self.std, 1e-6)

    def log_potential(self, social_state: NDArray) -> float:
        """Log-potential for a social state observation.

        Gaussian log-density (up to constant):
          -0.5 * sum((x - μ)² / σ²)
        """
        social_state = np.asarray(social_state, dtype=np.float64)
        z = (social_state - self.mean) / self.std
        return float(-0.5 * np.sum(z ** 2))

    def as_gaussian(self) -> GaussianVariable:
        """Return as a GaussianVariable for use in Kalman-style updates."""
        return GaussianVariable.from_diagonal(self.mean, self.std ** 2)


# --------------------------------------------------------------------------- #
# EmissionFactor
# --------------------------------------------------------------------------- #

class EmissionFactor:
    """ψ_emit: P(text_features | tactic, arousal).

    Two sub-models:
      1. Tactic-specific emission profiles: per-tactic Gaussian over 9 text features
      2. Arousal regression: point estimate of arousal from text features

    Parameters
    ----------
    tactic_profiles : dict[str, dict[str, tuple[float, float]]]
        {tactic_id: {feature_name: (mean, std)}} — per-tactic emission params.
    arousal_regressor : object | None
        Fitted regression model with .predict(X) method, or None if not available.
        If None, arousal is estimated as the character mean.
    feature_names : list[str]
        Ordered list of text feature names matching the profile dictionaries.
    """

    # Default text features from the emission model experiment
    DEFAULT_FEATURES = [
        "word_count", "question_density", "exclamation_density",
        "imperative_density", "mean_sentence_length", "lexical_diversity",
        "first_person_rate", "second_person_rate", "sentiment_polarity",
    ]

    # Minimum std floor to prevent emission factor from dominating.
    # With std=1e-6, a single feature mismatch of 1.0 produces
    # log-likelihood ~ -0.5 * (1e6)^2 = -5e11, completely overwhelming
    # the observation model (5 nats) and transition model.
    # A floor of 0.5 ensures each feature contributes at most ~2 nats.
    MIN_STD_FLOOR = 0.5

    # Emission tempering: scale all emission log-likelihoods by this factor.
    # This controls the relative influence of text features vs the LLM
    # observation model (accuracy=0.7, log-odds=5.02 nats).
    # A temperature of 0.1 means emission contributes at most ~10% of
    # the observation signal, acting as soft evidence rather than override.
    DEFAULT_EMISSION_TEMPERATURE = 0.1

    def __init__(
        self,
        tactic_profiles: dict[str, dict[str, tuple[float, float]]],
        arousal_regressor: object | None = None,
        feature_names: list[str] | None = None,
        emission_temperature: float | None = None,
    ):
        self.tactic_profiles = tactic_profiles
        self.arousal_regressor = arousal_regressor
        self.feature_names = feature_names or self.DEFAULT_FEATURES
        self.emission_temperature = (
            emission_temperature if emission_temperature is not None
            else self.DEFAULT_EMISSION_TEMPERATURE
        )

        # Pre-compute log-likelihood parameters for each tactic
        # as arrays for vectorized computation
        self._tactic_ids: list[str] = sorted(tactic_profiles.keys())
        self._tactic_means: dict[str, NDArray] = {}
        self._tactic_stds: dict[str, NDArray] = {}
        for tid in self._tactic_ids:
            profile = tactic_profiles[tid]
            means = []
            stds = []
            for fname in self.feature_names:
                if fname in profile:
                    m, s = profile[fname]
                    means.append(m)
                    stds.append(max(s, self.MIN_STD_FLOOR))  # Robust floor
                else:
                    # Missing feature: use non-informative (large std)
                    means.append(0.0)
                    stds.append(10.0)
            self._tactic_means[tid] = np.array(means, dtype=np.float64)
            self._tactic_stds[tid] = np.array(stds, dtype=np.float64)

    def _log_emission(self, text_features: NDArray, tactic_id: str) -> float:
        """Tempered log-likelihood of text features given a tactic.

        Assumes independent Gaussian per feature:
          log P(features | tactic) = sum_j -0.5 * ((f_j - μ_j) / σ_j)² - log(σ_j)

        The result is scaled by emission_temperature to prevent emission from
        dominating the observation model. Without tempering, emission
        log-likelihoods can reach magnitudes of 10^10 (from near-zero stds),
        completely overriding the LLM observation signal (~5 nats).
        """
        if tactic_id not in self._tactic_means:
            # Unknown tactic: return 0 (non-informative)
            return 0.0
        mu = self._tactic_means[tactic_id]
        sigma = self._tactic_stds[tactic_id]
        z = (text_features - mu) / sigma
        # Drop the constant -0.5*log(2π) terms since they cancel in normalization
        raw_ll = float(-0.5 * np.sum(z ** 2) - np.sum(np.log(sigma)))
        return raw_ll * self.emission_temperature

    def observe(
        self, text_features: NDArray, tactic_belief: DiscreteVariable
    ) -> DiscreteVariable:
        """Update tactic belief given observed text features.

        Bayesian update: posterior ∝ prior × likelihood.
        In log-space: log_posterior = log_prior + log_likelihood.
        """
        text_features = np.asarray(text_features, dtype=np.float64)
        new_log_probs = tactic_belief.log_probs.copy()

        for i, state in enumerate(tactic_belief.states):
            ll = self._log_emission(text_features, state)
            new_log_probs[i] += ll

        result = DiscreteVariable(tactic_belief.states, new_log_probs)
        result.normalize()
        return result

    def estimate_arousal(
        self, text_features: NDArray, char_mean_arousal: float = 0.0
    ) -> float:
        """Point estimate of arousal from text features.

        Uses the fitted arousal regressor if available; otherwise
        returns the character mean arousal.
        """
        if self.arousal_regressor is not None:
            features = np.asarray(text_features, dtype=np.float64).reshape(1, -1)
            try:
                pred = float(self.arousal_regressor.predict(features)[0])
                # Clamp to valid range
                return max(-1.0, min(1.0, pred))
            except Exception:
                pass
        return float(char_mean_arousal)


# --------------------------------------------------------------------------- #
# StatusCouplingFactor
# --------------------------------------------------------------------------- #

class StatusCouplingFactor:
    """ψ_social: cross-character status coupling.

    Penalizes both characters claiming high status simultaneously,
    modeling the empirical inverse correlation (r=-0.20) in status claims.

    The pairwise potential:
      P(s_a, s_b) ∝ exp(-γ * (s_a + s_b)²)
    penalizes the sum being large (both high status).

    Parameters
    ----------
    gamma : float
        Coupling strength. Estimated from empirical correlation:
        γ ≈ |r| / (1 - r²) ≈ 0.21.
    """

    def __init__(self, gamma: float):
        self.gamma = float(gamma)

    def log_potential(self, status_a: float, status_b: float) -> float:
        """Log-potential for a pair of status values."""
        return -self.gamma * (status_a + status_b) ** 2

    def coupled_update(
        self, status_a: float, status_b: float
    ) -> tuple[float, float]:
        """Soft coupling update: nudge each status away from the other.

        Uses gradient of the log-potential to compute a small correction:
          ∂/∂s_a log P = -2γ(s_a + s_b)
        The update applies a fraction of this gradient.

        Returns adjusted (status_a, status_b).
        """
        step_size = 0.1  # Conservative step
        grad_sum = -2.0 * self.gamma * (status_a + status_b)
        adj_a = status_a + step_size * grad_sum
        adj_b = status_b + step_size * grad_sum
        # Clamp to valid range [-1, 1]
        adj_a = max(-1.0, min(1.0, adj_a))
        adj_b = max(-1.0, min(1.0, adj_b))
        return adj_a, adj_b
