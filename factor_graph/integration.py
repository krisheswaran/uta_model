"""
Factor Graph State Updater — Drop-in replacement for LLM-based state_updater.py

Uses the factor graph's forward filter to update BeatState after each improv
turn, replacing the LLM state updater call with pure numpy inference (<100ms).

INTEGRATION POINT
-----------------
In improv/improvisation_loop.py, line ~360:

    updated_state = update_beat_state(
        beat_state,
        final_candidate.text,
        partner_line=context.partner_line or "",
    )

Replace with:

    from factor_graph.integration import FactorGraphStateUpdater

    # Initialize once per session (in run_session or run_turn setup):
    fg_updater = FactorGraphStateUpdater(play_id="hamlet", character="HAMLET")

    # Then in run_turn, replace the update_beat_state call:
    updated_state = fg_updater.update_state(
        prev_state=beat_state,
        delivered_line=final_candidate.text,
        partner_line=context.partner_line or "",
        characters_present=context.characters_present,
    )

The FactorGraphStateUpdater returns a BeatState identical in structure to what
update_beat_state returns, so no downstream changes are needed. The key
difference: zero LLM calls, <100ms latency, deterministic given the same inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import FACTORS_DIR, PARSED_DIR, VOCAB_DIR
from factor_graph.graph import CharacterFactorGraph, FactorParameters
from factor_graph.inference import (
    ForwardFilter,
    PosteriorState,
    _extract_text_features_from_utterances,
)
from schemas import (
    AffectState,
    BeatState,
    CharacterBible,
    EpistemicState,
    SocialState,
)


class FactorGraphStateUpdater:
    """Drop-in replacement for LLM-based state_updater.

    Uses the factor graph's forward filter to update BeatState
    after each improv turn, replacing the LLM state updater call.

    Parameters
    ----------
    play_id : str
        The play this character belongs to (e.g., "hamlet", "cherry_orchard").
    character : str
        The character name (e.g., "HAMLET", "LOPAKHIN").
    factors_dir : Path | None
        Directory containing learned factor parameters. Defaults to
        config.FACTORS_DIR.
    params : FactorParameters | None
        Pre-loaded parameters. If provided, factors_dir is ignored.
        Useful when running multiple characters from the same parameter set.
    """

    def __init__(
        self,
        play_id: str,
        character: str,
        factors_dir: Path | None = None,
        params: FactorParameters | None = None,
    ):
        self.play_id = play_id
        self.character = character.upper()
        self._last_posterior: Optional[PosteriorState] = None
        self._initialized = False
        self._turn_count = 0

        # Load factor parameters
        if params is not None:
            self._params = params
        else:
            _factors_dir = factors_dir or FACTORS_DIR
            try:
                self._params = FactorParameters.load(_factors_dir)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Factor parameters not found in {_factors_dir}. "
                    "Run the learning pipeline first:\n"
                    "  python -m factor_graph.learning --plays cherry_orchard hamlet "
                    "importance_of_being_earnest"
                ) from e

        # Load character bible from parsed play
        self._bible = self._load_character_bible()

        # Build the factor graph
        self._graph = CharacterFactorGraph(
            params=self._params,
            character_bible=self._bible,
            tactic_vocab=self._params.tactic_vocab,
        )

        # Create forward filter
        self._filter = ForwardFilter(self._graph)

    def _load_character_bible(self) -> CharacterBible:
        """Load the character's bible from the parsed play data."""
        parsed_path = PARSED_DIR / f"{self.play_id}.json"
        if not parsed_path.exists():
            raise FileNotFoundError(
                f"Parsed play not found at {parsed_path}. "
                f"Run the analysis pipeline for '{self.play_id}' first."
            )

        with open(parsed_path) as f:
            play_data = json.load(f)

        for cb_dict in play_data.get("character_bibles", []):
            if cb_dict["character"].upper() == self.character:
                return CharacterBible(**cb_dict)

        # If character not found, provide a helpful error with available characters
        available = [cb["character"] for cb in play_data.get("character_bibles", [])]
        raise ValueError(
            f"Character '{self.character}' not found in {self.play_id}. "
            f"Available characters: {', '.join(available)}"
        )

    def _initialize_from_bible(self, partner: str | None = None) -> PosteriorState:
        """Create the initial posterior from the character bible priors.

        Called on the first update_state() call when no prior posterior exists.
        """
        posterior = self._filter.initialize(partner=partner)
        self._initialized = True
        return posterior

    def update_state(
        self,
        prev_state: BeatState,
        delivered_line: str,
        partner_line: str = "",
        characters_present: list[str] | None = None,
    ) -> BeatState:
        """Update BeatState using the factor graph forward filter.

        Given the previous state and what just happened, predict the next state.
        This replaces the LLM call in state_updater.py.

        Parameters
        ----------
        prev_state : BeatState
            The current beat state before this turn's update.
        delivered_line : str
            The line just delivered by this character.
        partner_line : str
            The most recent line from the scene partner (if any).
        characters_present : list[str] | None
            Characters currently in the scene.

        Returns
        -------
        BeatState
            Updated state reflecting the delivered line's impact. Preserves
            epistemic_state, desire_state, obstacle, superobjective_reminder,
            and defense_state from prev_state (the factor graph models tactic,
            affect, and social dimensions only).
        """
        self._turn_count += 1

        # On first call, initialize from character bible priors
        if self._last_posterior is None:
            primary_partner = None
            if characters_present:
                others = [c for c in characters_present
                          if c.upper() != self.character]
                if others:
                    primary_partner = others[0]
            self._last_posterior = self._initialize_from_bible(partner=primary_partner)

        # Extract desire state for cluster assignment
        desire_state = prev_state.desire_state if prev_state.desire_state else None

        # Combine delivered line and partner line as utterance context
        # The delivered line is the primary signal; partner line provides context
        utterance_text = delivered_line
        if partner_line:
            utterance_text = f"{delivered_line} {partner_line}"

        # Run forward filter step
        try:
            posterior = self._filter.step(
                prev_posterior=self._last_posterior,
                utterance_text=utterance_text,
                desire_state=desire_state,
                characters_present=characters_present,
            )
        except Exception as e:
            # On inference failure, return previous state unchanged
            print(f"  [FactorGraphStateUpdater] WARNING: inference failed "
                  f"(turn {self._turn_count}): {e}")
            return prev_state

        # Store posterior for next step and diagnostics
        self._last_posterior = posterior

        # Convert posterior to BeatState via MAP estimate
        beat_state_from_fg = posterior.to_beat_state(
            method="map",
            affect_eigenvectors=self._params.affect_eigenvectors,
        )

        # Merge: factor graph updates tactic, affect, social;
        # preserve semantic fields from prev_state that the factor graph
        # doesn't model (desire, obstacle, epistemic, defense, etc.)
        updated = prev_state.model_copy(deep=True)

        # Update tactic from factor graph
        updated.tactic_state = beat_state_from_fg.tactic_state
        updated.canonical_tactic = beat_state_from_fg.canonical_tactic

        # Update affect from factor graph
        updated.affect_state = beat_state_from_fg.affect_state

        # Update social from factor graph
        updated.social_state = SocialState(
            status=beat_state_from_fg.social_state.status,
            warmth=beat_state_from_fg.social_state.warmth,
            rationale=prev_state.social_state.rationale,  # preserve rationale
        )

        # Update confidence from the factor graph posterior
        updated.confidence = beat_state_from_fg.confidence

        # Update beat_id for the new turn
        updated.beat_id = f"improv_b{self._turn_count + 1}"

        return updated

    def get_posterior(self) -> Optional[PosteriorState]:
        """Access the full posterior from the last update (for diagnostics).

        Returns None if update_state() has not been called yet.

        The posterior includes full distributions over tactics and desire
        clusters, plus Gaussian parameters for affect and social dimensions.
        This is useful for:
        - Displaying uncertainty to the user
        - Detecting multimodal tactic distributions
        - Logging inference quality metrics
        """
        return self._last_posterior

    @property
    def turn_count(self) -> int:
        """Number of update_state() calls made so far."""
        return self._turn_count

    @property
    def character_bible(self) -> CharacterBible:
        """The loaded character bible."""
        return self._bible

    def reset(self) -> None:
        """Reset the filter state for a new scene.

        Call this when starting a new improv scene with the same character.
        """
        self._last_posterior = None
        self._initialized = False
        self._turn_count = 0
        self._filter = ForwardFilter(self._graph)
