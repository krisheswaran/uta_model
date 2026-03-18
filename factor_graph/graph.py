"""
FactorGraph container: loads learned parameters and builds per-character graphs.

FactorParameters loads all learned factor potentials from data/factors/.
CharacterFactorGraph assembles the factors for a specific character
using their CharacterBible and the shared learned parameters.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from factor_graph.factors import (
    AffectTransitionFactor,
    DesireTransitionFactor,
    EmissionFactor,
    StatusCouplingFactor,
    SuperobjectivePrior,
    SocialPrior,
    TacticTransitionFactor,
)
from factor_graph.variables import DiscreteVariable, GaussianVariable, PointEstimate
from schemas import CharacterBible


# --------------------------------------------------------------------------- #
# FactorParameters — all learned model parameters
# --------------------------------------------------------------------------- #

@dataclass
class FactorParameters:
    """Container for all learned factor parameters, loaded from data/factors/.

    Each field corresponds to an artifact produced by the parameter learning
    pipeline (Pass 1.5a). Fields are numpy arrays or dicts as appropriate.

    This class is the single source of truth for model parameters at
    inference time — both smoothing (Pass 1.5c) and forward filtering
    (Pass 2) load from the same FactorParameters instance.
    """

    # ψ_T: Tactic transition
    tactic_transition_base: NDArray                # (66, 66) smoothed transition matrix
    tactic_transition_by_desire: NDArray           # (7, 66, 66) per-desire-cluster matrices
    desire_cluster_centroids: NDArray              # (7, 384) k-means centroids
    persistence_beta: float                        # desire-similarity persistence modulation

    # ψ_A: Affect transition (Student-t kernel in eigenspace)
    affect_eigenvectors: NDArray                   # (3, 5) rotation matrix R
    affect_transition_variance: NDArray            # (3,) diagonal variances
    affect_transition_df: NDArray                  # (3,) degrees of freedom

    # ψ_D: Desire transition
    desire_transition_matrix: NDArray              # (7, 7) transition matrix

    # ψ_arc: Superobjective prior
    superobjective_tactic_prior: dict[int, NDArray]  # cluster_id → 66-dim prob vector
    superobjective_centroids: NDArray              # (k, 384) SO cluster centroids

    # ψ_emit: Emission model
    tactic_emission_profiles: dict[str, dict[str, tuple[float, float]]]
    # {tactic_id: {feature_name: (mean, std)}}
    arousal_regressor: Any                         # fitted sklearn regressor or None

    # ψ_social: Cross-character coupling
    status_coupling_gamma: float                   # coupling strength

    # Vocabulary
    tactic_vocab: list[str]                        # ordered list of 66 canonical tactic IDs
    desire_cluster_labels: list[str]               # labels for k=7 desire clusters

    @classmethod
    def load(cls, factors_dir: Path) -> FactorParameters:
        """Load all parameter files from disk.

        Expected files in factors_dir:
          - tactic_transition_base.json       (66×66 matrix as nested list)
          - tactic_transition_by_desire.json   (7×66×66)
          - desire_cluster_centroids.npy       (7×384)
          - persistence_modulation_beta.json   (scalar)
          - affect_eigenvectors.npy            (3×5)
          - affect_transition_variance.npy     (3,)
          - affect_transition_df.npy           (3,) — optional, defaults to df=5
          - desire_transition_matrix.json      (7×7)
          - superobjective_tactic_prior.json   ({cluster_id: [66 probs]})
          - superobjective_cluster_centroids.npy
          - tactic_emission_profiles.json      ({tactic: {feature: [mean, std]}})
          - arousal_regressor.pkl              — optional
          - status_coupling_gamma.json         (scalar)
          - tactic_vocab.json                  (list of 66 IDs)
          - desire_cluster_labels.json         (list of 7 labels) — optional
        """
        factors_dir = Path(factors_dir)

        def _load_json(name: str, required: bool = True) -> Any:
            path = factors_dir / name
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"Required parameter file not found: {path}")
                return None
            with open(path) as f:
                return json.load(f)

        def _load_npy(name: str, required: bool = True) -> NDArray | None:
            path = factors_dir / name
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"Required parameter file not found: {path}")
                return None
            return np.load(path)

        def _load_pkl(name: str) -> Any:
            path = factors_dir / name
            if not path.exists():
                return None
            # Try joblib first (sklearn models are often saved with joblib),
            # then fall back to pickle
            try:
                import joblib
                return joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f)

        # Load tactic vocabulary first — needed for sizing
        tactic_vocab_data = _load_json("tactic_vocab.json", required=False)
        if tactic_vocab_data is None:
            # Fall back to loading from the main vocab directory
            vocab_path = factors_dir.parent / "vocab" / "tactic_vocabulary.json"
            if vocab_path.exists():
                with open(vocab_path) as f:
                    vocab_data = json.load(f)
                tactic_vocab = [t["canonical_id"] for t in vocab_data.get("tactics", [])]
            else:
                raise FileNotFoundError(
                    "No tactic vocabulary found. Need tactic_vocab.json in factors/ "
                    "or tactic_vocabulary.json in vocab/"
                )
        else:
            tactic_vocab = tactic_vocab_data

        n_tactics = len(tactic_vocab)

        def _dict_matrix_to_array(d: dict, row_keys: list[str] | None = None,
                                   col_keys: list[str] | None = None) -> NDArray:
            """Convert a dict-of-dicts matrix to a numpy array.

            Handles both nested-list and dict-of-dicts formats.
            """
            if isinstance(d, list):
                return np.array(d, dtype=np.float64)
            # Dict-of-dicts: {row_key: {col_key: value}}
            if row_keys is None:
                row_keys = list(d.keys())
            if col_keys is None:
                first_val = d[row_keys[0]]
                col_keys = list(first_val.keys()) if isinstance(first_val, dict) else None
            if col_keys is None:
                # Dict of lists
                return np.array([d[k] for k in row_keys], dtype=np.float64)
            n_rows = len(row_keys)
            n_cols = len(col_keys)
            mat = np.zeros((n_rows, n_cols), dtype=np.float64)
            for i, rk in enumerate(row_keys):
                row = d[rk]
                for j, ck in enumerate(col_keys):
                    mat[i, j] = float(row[ck])
            return mat

        # ψ_T: Tactic transition
        base_data = _load_json("tactic_transition_base.json")
        if isinstance(base_data, dict):
            # Keyed by tactic names — use tactic_vocab ordering
            base_matrix = _dict_matrix_to_array(base_data, tactic_vocab, tactic_vocab)
        else:
            base_matrix = np.array(base_data, dtype=np.float64)

        desire_data = _load_json("tactic_transition_by_desire.json")
        if isinstance(desire_data, dict):
            # {cluster_id: {from_tactic: {to_tactic: prob}}}
            cluster_keys = sorted(desire_data.keys(), key=lambda x: int(x))
            desire_matrices = np.stack([
                _dict_matrix_to_array(desire_data[ck], tactic_vocab, tactic_vocab)
                for ck in cluster_keys
            ])
        else:
            desire_matrices = np.array(desire_data, dtype=np.float64)

        beta_data = _load_json("persistence_modulation_beta.json")
        persistence_beta = float(beta_data if isinstance(beta_data, (int, float))
                                 else beta_data.get("beta", 1.0))

        centroids = _load_npy("desire_cluster_centroids.npy")

        # ψ_A: Affect transition
        eigvecs = _load_npy("affect_eigenvectors.npy")
        trans_var = _load_npy("affect_transition_variance.npy")
        trans_df = _load_npy("affect_transition_df.npy", required=False)
        if trans_df is None:
            # Try JSON format (dict with per-axis Student-t params)
            trans_df_json = _load_json("affect_transition_df.json", required=False)
            if trans_df_json is not None and isinstance(trans_df_json, dict):
                # Extract df values from per-axis dicts: {"axis_0": {"df": ...}, ...}
                n_axes = len(trans_df_json)
                trans_df = np.array([
                    trans_df_json[f"axis_{i}"]["df"]
                    for i in range(n_axes)
                ], dtype=np.float64)
            else:
                # Default: moderate heavy tails
                trans_df = np.array([5.0, 5.0, 5.0])

        # ψ_D: Desire transition
        desire_trans_data = _load_json("desire_transition_matrix.json")
        desire_trans = _dict_matrix_to_array(desire_trans_data)

        # ψ_arc: Superobjective prior
        so_prior_data = _load_json("superobjective_tactic_prior.json", required=False)
        so_tactic_prior: dict[int, NDArray] = {}
        if so_prior_data is not None:
            for k, v in so_prior_data.items():
                if isinstance(v, dict):
                    # Dict keyed by tactic name: {tactic: prob}
                    arr = np.zeros(n_tactics, dtype=np.float64)
                    for tactic_name, prob in v.items():
                        if tactic_name in tactic_vocab:
                            idx = tactic_vocab.index(tactic_name)
                            arr[idx] = float(prob)
                    # Normalize
                    total = arr.sum()
                    if total > 0:
                        arr /= total
                    else:
                        arr[:] = 1.0 / n_tactics
                    so_tactic_prior[int(k)] = arr
                else:
                    so_tactic_prior[int(k)] = np.array(v, dtype=np.float64)

        so_centroids = _load_npy("superobjective_cluster_centroids.npy", required=False)
        if so_centroids is None:
            so_centroids = np.zeros((0, 384))

        # ψ_emit: Emission profiles
        emit_data = _load_json("tactic_emission_profiles.json", required=False)
        tactic_emission_profiles: dict[str, dict[str, tuple[float, float]]] = {}
        if emit_data is not None:
            for tactic_id, profile in emit_data.items():
                if "mean" in profile and "std" in profile:
                    # Format: {n: int, mean: {feat: val}, std: {feat: val}}
                    means = profile["mean"]
                    stds = profile["std"]
                    tactic_emission_profiles[tactic_id] = {
                        fname: (means[fname], stds.get(fname, 1.0))
                        for fname in means
                    }
                else:
                    # Format: {feat: [mean, std]} or {feat: (mean, std)}
                    tactic_emission_profiles[tactic_id] = {
                        fname: tuple(vals) if isinstance(vals, list) else vals
                        for fname, vals in profile.items()
                    }

        arousal_reg = _load_pkl("arousal_regressor.pkl")

        # ψ_social
        gamma_data = _load_json("status_coupling_gamma.json", required=False)
        gamma = 0.21  # Default from empirical correlation
        if gamma_data is not None:
            gamma = float(gamma_data if isinstance(gamma_data, (int, float))
                          else gamma_data.get("gamma", 0.21))

        # Desire cluster labels
        cluster_labels = _load_json("desire_cluster_labels.json", required=False)
        if cluster_labels is None:
            cluster_labels = [str(i) for i in range(desire_trans.shape[0])]

        return cls(
            tactic_transition_base=base_matrix,
            tactic_transition_by_desire=desire_matrices,
            desire_cluster_centroids=centroids,
            persistence_beta=persistence_beta,
            affect_eigenvectors=eigvecs,
            affect_transition_variance=trans_var,
            affect_transition_df=trans_df,
            desire_transition_matrix=desire_trans,
            superobjective_tactic_prior=so_tactic_prior,
            superobjective_centroids=so_centroids,
            tactic_emission_profiles=tactic_emission_profiles,
            arousal_regressor=arousal_reg,
            status_coupling_gamma=gamma,
            tactic_vocab=tactic_vocab,
            desire_cluster_labels=cluster_labels,
        )

    @classmethod
    def create_uniform(cls, n_tactics: int = 66, n_desire: int = 7) -> FactorParameters:
        """Create a FactorParameters with uniform/non-informative priors.

        Useful for testing and as a fallback when learned parameters
        are not yet available.
        """
        tactic_vocab = [f"TACTIC_{i}" for i in range(n_tactics)]
        desire_labels = [str(i) for i in range(n_desire)]

        uniform_tactic = np.ones((n_tactics, n_tactics)) / n_tactics
        uniform_desire = np.ones((n_desire, n_desire)) / n_desire

        return cls(
            tactic_transition_base=uniform_tactic,
            tactic_transition_by_desire=np.stack([uniform_tactic] * n_desire),
            desire_cluster_centroids=np.zeros((n_desire, 384)),
            persistence_beta=0.0,
            affect_eigenvectors=np.eye(3, 5),  # Identity projection (first 3 dims)
            affect_transition_variance=np.ones(3) * 0.1,
            affect_transition_df=np.array([5.0, 5.0, 5.0]),
            desire_transition_matrix=uniform_desire,
            superobjective_tactic_prior={},
            superobjective_centroids=np.zeros((0, 384)),
            tactic_emission_profiles={},
            arousal_regressor=None,
            status_coupling_gamma=0.21,
            tactic_vocab=tactic_vocab,
            desire_cluster_labels=desire_labels,
        )


# --------------------------------------------------------------------------- #
# CharacterFactorGraph — assembled factors for one character
# --------------------------------------------------------------------------- #

class CharacterFactorGraph:
    """Factor graph for a single character, built from FactorParameters + CharacterBible.

    Assembles all factor nodes with character-specific parameterization:
      - Tactic transition (shared across characters)
      - Affect transition (shared)
      - Desire transition (shared)
      - Superobjective prior (character-specific: depends on SO embedding)
      - Social prior (character-specific: depends on relational profile)
      - Emission model (shared)
      - Status coupling (shared)

    Parameters
    ----------
    params : FactorParameters
        Learned model parameters.
    character_bible : CharacterBible
        The character's bible (superobjective, relational profile, etc.).
    tactic_vocab : list[str]
        Ordered canonical tactic IDs. Typically params.tactic_vocab.
    relational_profiles : dict[str, dict] | None
        Partner-specific social priors: {partner: {mean: [s, w], std: [s, w]}}.
        If None, uses default social prior.
    """

    def __init__(
        self,
        params: FactorParameters,
        character_bible: CharacterBible,
        tactic_vocab: list[str] | None = None,
        relational_profiles: dict[str, dict] | None = None,
    ):
        self.params = params
        self.character_bible = character_bible
        self.tactic_vocab = tactic_vocab or params.tactic_vocab
        self.n_tactics = len(self.tactic_vocab)
        self.desire_labels = params.desire_cluster_labels
        self.n_desire = len(self.desire_labels)

        # --- Build factors ---

        # ψ_T: Tactic transition
        self.tactic_transition = TacticTransitionFactor(
            base_matrix=params.tactic_transition_base,
            desire_matrices=params.tactic_transition_by_desire,
            persistence_beta=params.persistence_beta,
        )

        # ψ_A: Affect transition (Student-t kernel)
        self.affect_transition = AffectTransitionFactor(
            variance=params.affect_transition_variance,
            df=params.affect_transition_df,
        )

        # ψ_D: Desire transition
        self.desire_transition = DesireTransitionFactor(
            transition_matrix=params.desire_transition_matrix,
        )

        # ψ_arc: Superobjective prior (character-specific)
        self.superobjective_prior = self._build_superobjective_prior(params)

        # ψ_S: Social priors (partner-specific)
        self._relational_profiles = relational_profiles or {}
        self._default_social_prior = SocialPrior(
            mean=np.array([0.0, 0.0]),
            std=np.array([0.5, 0.5]),
        )

        # ψ_emit: Emission factor
        self.emission = EmissionFactor(
            tactic_profiles=params.tactic_emission_profiles,
            arousal_regressor=params.arousal_regressor,
        )

        # ψ_social: Status coupling
        self.status_coupling = StatusCouplingFactor(
            gamma=params.status_coupling_gamma,
        )

        # Affect projection matrix
        self.affect_eigenvectors = params.affect_eigenvectors  # (3, 5)

    def _build_superobjective_prior(
        self, params: FactorParameters
    ) -> SuperobjectivePrior | None:
        """Build the superobjective tactic prior for this character.

        Finds the nearest superobjective cluster to this character's
        superobjective embedding and returns the corresponding tactic prior.
        If no superobjective data is available, returns None.
        """
        if not params.superobjective_tactic_prior:
            return None

        # If there's only one cluster or no centroids, use the first available
        if len(params.superobjective_tactic_prior) == 1:
            cluster_id = next(iter(params.superobjective_tactic_prior))
            return SuperobjectivePrior(params.superobjective_tactic_prior[cluster_id])

        # Without an embedding model loaded, we can't compute nearest cluster
        # at construction time. Default to the first cluster; the integration
        # layer can override this with a specific cluster assignment.
        if 0 in params.superobjective_tactic_prior:
            return SuperobjectivePrior(params.superobjective_tactic_prior[0])

        first_key = next(iter(params.superobjective_tactic_prior))
        return SuperobjectivePrior(params.superobjective_tactic_prior[first_key])

    def set_superobjective_cluster(self, cluster_id: int) -> None:
        """Override the superobjective prior with a specific cluster.

        Called by the integration layer after embedding the character's
        superobjective and finding the nearest cluster.
        """
        if cluster_id in self.params.superobjective_tactic_prior:
            self.superobjective_prior = SuperobjectivePrior(
                self.params.superobjective_tactic_prior[cluster_id]
            )

    def get_social_prior(self, partner: str | None = None) -> SocialPrior:
        """Get the social prior for a specific partner, or the default.

        Parameters
        ----------
        partner : str | None
            The interaction partner's name. If None or not in relational
            profiles, returns the default social prior.
        """
        if partner and partner in self._relational_profiles:
            profile = self._relational_profiles[partner]
            return SocialPrior(
                mean=np.array(profile["mean"], dtype=np.float64),
                std=np.array(profile["std"], dtype=np.float64),
            )
        return self._default_social_prior

    def project_affect(self, affect_5d: NDArray) -> NDArray:
        """Project raw 5D affect vector into 3D eigenspace.

        Parameters
        ----------
        affect_5d : (5,) array
            [valence, arousal, certainty, control, vulnerability]

        Returns
        -------
        (3,) array in eigenspace [Disempowerment, Blissful Ignorance, Burdened Power]
        """
        return self.affect_eigenvectors @ np.asarray(affect_5d, dtype=np.float64)

    def initial_tactic_belief(self) -> DiscreteVariable:
        """Create the initial tactic belief from the character's tactic distribution.

        Uses the character bible's tactic_distribution if available,
        otherwise uniform.
        """
        if self.character_bible.tactic_distribution:
            counts = np.zeros(self.n_tactics, dtype=np.float64)
            total = 0.0
            for tactic_id, count in self.character_bible.tactic_distribution.items():
                if tactic_id in self.tactic_vocab:
                    idx = self.tactic_vocab.index(tactic_id)
                    counts[idx] = count
                    total += count
            if total > 0:
                # Add small smoothing
                counts += 1.0
                log_probs = np.log(counts) - np.log(np.sum(counts))
                return DiscreteVariable(self.tactic_vocab, log_probs)

        # Uniform fallback
        return DiscreteVariable(self.tactic_vocab)

    def initial_desire_belief(self) -> DiscreteVariable:
        """Create the initial desire belief (uniform over clusters)."""
        return DiscreteVariable(self.desire_labels)

    def initial_affect(self) -> GaussianVariable:
        """Create the initial affect state in eigenspace (zero mean, moderate uncertainty)."""
        return GaussianVariable.from_diagonal(
            mean=np.zeros(3),
            variance=np.ones(3) * 0.25,
        )

    def initial_social(self, partner: str | None = None) -> GaussianVariable:
        """Create the initial social state from the relational profile prior."""
        prior = self.get_social_prior(partner)
        return prior.as_gaussian()
