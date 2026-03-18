"""
Variable node types for the factor graph.

Three variable types model the latent dramatic state:
  - DiscreteVariable: categorical distribution (tactics, desire clusters)
  - GaussianVariable: multivariate Gaussian (affect eigenspace, social state)
  - PointEstimate: scalar with no uncertainty (arousal from text features)

All probability computations use log-space for numerical stability.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DiscreteVariable:
    """Categorical distribution over a finite set of states.

    Internally stores log-probabilities for numerical stability.
    All mutation methods (normalize, update, blend) operate in log-space
    and only exponentiate when needed for output.

    Parameters
    ----------
    states : list[str]
        State labels (e.g., canonical tactic IDs or desire cluster names).
    log_probs : NDArray | None
        Log-probability for each state. If None, initializes to uniform.
    """

    __slots__ = ("states", "log_probs", "_state_to_idx")

    def __init__(self, states: list[str], log_probs: NDArray | None = None):
        self.states = list(states)
        n = len(self.states)
        if n == 0:
            raise ValueError("DiscreteVariable requires at least one state")

        if log_probs is not None:
            if len(log_probs) != n:
                raise ValueError(
                    f"log_probs length {len(log_probs)} != states length {n}"
                )
            self.log_probs = np.array(log_probs, dtype=np.float64)
        else:
            # Uniform: log(1/n) for each state
            self.log_probs = np.full(n, -np.log(n), dtype=np.float64)

        # Index lookup for fast state-to-index mapping
        self._state_to_idx = {s: i for i, s in enumerate(self.states)}

    @property
    def n_states(self) -> int:
        return len(self.states)

    def normalize(self) -> None:
        """Normalize log_probs so they represent a valid log-probability distribution.

        Uses the log-sum-exp trick: log_probs -= logsumexp(log_probs).
        """
        max_lp = np.max(self.log_probs)
        if np.isneginf(max_lp):
            # All -inf: reset to uniform
            self.log_probs[:] = -np.log(self.n_states)
            return
        shifted = self.log_probs - max_lp
        log_z = max_lp + np.log(np.sum(np.exp(shifted)))
        self.log_probs -= log_z

    def probs(self) -> NDArray:
        """Return normalized probabilities as a numpy array."""
        max_lp = np.max(self.log_probs)
        if np.isneginf(max_lp):
            return np.full(self.n_states, 1.0 / self.n_states)
        shifted = self.log_probs - max_lp
        p = np.exp(shifted)
        return p / np.sum(p)

    def prob(self, state: str) -> float:
        """Return the probability of a single state."""
        idx = self._state_to_idx.get(state)
        if idx is None:
            return 0.0
        return float(self.probs()[idx])

    def map_state(self) -> str:
        """Return the most probable state (Maximum A Posteriori)."""
        return self.states[int(np.argmax(self.log_probs))]

    def map_prob(self) -> float:
        """Return the probability of the MAP state."""
        return float(self.probs()[np.argmax(self.log_probs)])

    def entropy(self) -> float:
        """Shannon entropy of the distribution (in nats)."""
        p = self.probs()
        # Mask zeros to avoid log(0)
        mask = p > 0
        return float(-np.sum(p[mask] * np.log(p[mask])))

    def sample(self, rng: np.random.Generator | None = None) -> str:
        """Draw a single sample from the distribution."""
        if rng is None:
            rng = np.random.default_rng()
        p = self.probs()
        idx = rng.choice(self.n_states, p=p)
        return self.states[idx]

    def state_index(self, state: str) -> int:
        """Return the index of a state label, or raise KeyError."""
        idx = self._state_to_idx.get(state)
        if idx is None:
            raise KeyError(f"Unknown state: {state!r}")
        return idx

    def to_dict(self) -> dict[str, float]:
        """Return {state_label: probability} dict."""
        p = self.probs()
        return {s: float(p[i]) for i, s in enumerate(self.states)}

    def copy(self) -> DiscreteVariable:
        """Return an independent copy."""
        return DiscreteVariable(self.states, self.log_probs.copy())

    def __repr__(self) -> str:
        top = self.map_state()
        top_p = self.map_prob()
        h = self.entropy()
        return (
            f"DiscreteVariable(n={self.n_states}, "
            f"MAP={top}@{top_p:.3f}, H={h:.2f})"
        )


class GaussianVariable:
    """Multivariate Gaussian distribution.

    Used for continuous latent variables: affect in eigenspace (3D)
    and social state (2D).

    Parameters
    ----------
    mean : NDArray
        Mean vector (d-dimensional).
    cov : NDArray
        Covariance matrix (d x d). Must be symmetric positive semi-definite.
    """

    __slots__ = ("mean", "cov")

    def __init__(self, mean: NDArray, cov: NDArray):
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.cov = np.asarray(cov, dtype=np.float64).copy()
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got shape {self.mean.shape}")
        d = len(self.mean)
        if self.cov.shape != (d, d):
            raise ValueError(
                f"cov shape {self.cov.shape} doesn't match mean length {d}"
            )

    @classmethod
    def from_diagonal(cls, mean: NDArray, variance: NDArray) -> GaussianVariable:
        """Create from mean and diagonal variance vector."""
        mean = np.asarray(mean, dtype=np.float64)
        variance = np.asarray(variance, dtype=np.float64)
        return cls(mean, np.diag(variance))

    @property
    def dim(self) -> int:
        return len(self.mean)

    @property
    def std(self) -> NDArray:
        """Marginal standard deviations (square root of diagonal)."""
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))

    def map_state(self) -> NDArray:
        """MAP estimate = mean for a Gaussian."""
        return self.mean.copy()

    def sample(self, rng: np.random.Generator | None = None) -> NDArray:
        """Draw a single sample."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(self.mean, self.cov)

    def log_prob(self, x: NDArray) -> float:
        """Log-probability density at point x."""
        x = np.asarray(x, dtype=np.float64)
        d = self.dim
        diff = x - self.mean
        # Use pseudo-inverse for numerical stability
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            # Degenerate — fall back to large penalty
            return -1e10
        cov_inv = np.linalg.inv(self.cov)
        mahal = float(diff @ cov_inv @ diff)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)

    def copy(self) -> GaussianVariable:
        return GaussianVariable(self.mean.copy(), self.cov.copy())

    def __repr__(self) -> str:
        return (
            f"GaussianVariable(dim={self.dim}, "
            f"mean={self.mean}, std={self.std})"
        )


class PointEstimate:
    """A scalar value with no uncertainty model.

    Used for A_emit (arousal), which is estimated directly from text features
    via regression rather than inferred through transition dynamics.

    Parameters
    ----------
    value : float
        The point estimate value.
    """

    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = float(value)

    def map_state(self) -> float:
        return self.value

    def copy(self) -> PointEstimate:
        return PointEstimate(self.value)

    def __repr__(self) -> str:
        return f"PointEstimate({self.value:.4f})"
