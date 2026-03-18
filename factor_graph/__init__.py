"""Factor graph for dramatic state inference (Pass 1.5 and Pass 2).

Core modules:
  - variables: DiscreteVariable, GaussianVariable, PointEstimate
  - factors: TacticTransitionFactor, AffectTransitionFactor, etc.
  - graph: FactorParameters, CharacterFactorGraph
  - inference: ForwardFilter, ForwardBackwardSmoother, PosteriorState
"""
from factor_graph.variables import DiscreteVariable, GaussianVariable, PointEstimate
from factor_graph.graph import FactorParameters, CharacterFactorGraph
from factor_graph.inference import ForwardFilter, ForwardBackwardSmoother, PosteriorState

__all__ = [
    "DiscreteVariable",
    "GaussianVariable",
    "PointEstimate",
    "FactorParameters",
    "CharacterFactorGraph",
    "ForwardFilter",
    "ForwardBackwardSmoother",
    "PosteriorState",
]
