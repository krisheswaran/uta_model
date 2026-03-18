#!/usr/bin/env python3
"""
Pass 1.5c — Forward-backward smoothing over extracted BeatStates.

Runs the factor graph smoother over all characters in a play, producing
refined posterior distributions that reconcile noisy LLM extractions with
learned transition dynamics.

Usage:
    conda run -n uta_model python scripts/run_smoothing.py cherry_orchard
    conda run -n uta_model python scripts/run_smoothing.py hamlet
    conda run -n uta_model python scripts/run_smoothing.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import FACTORS_DIR, PARSED_DIR, SMOOTHED_DIR, VOCAB_DIR, PLAYS
from factor_graph.graph import CharacterFactorGraph, FactorParameters
from factor_graph.inference import ForwardBackwardSmoother, PosteriorState
from schemas import AffectState, BeatState, CharacterBible, Play, SocialState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

MIN_BEATS_FOR_SMOOTHING = 3


def _load_tactic_vocab() -> list[str]:
    """Load the ordered canonical tactic vocabulary."""
    vocab_path = VOCAB_DIR / "tactic_vocabulary.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Tactic vocabulary not found at {vocab_path}. "
            "Run the vocabulary builder first."
        )
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    return sorted(t["canonical_id"] for t in vocab_data.get("tactics", []))


def _load_play(play_id: str) -> dict:
    """Load a parsed play JSON from data/parsed/."""
    path = PARSED_DIR / f"{play_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Parsed play not found at {path}. Run the analysis pipeline first."
        )
    with open(path) as f:
        return json.load(f)


def _build_character_bible(cb_dict: dict) -> CharacterBible:
    """Construct a CharacterBible from a parsed play's character_bible dict."""
    return CharacterBible(**cb_dict)


def _build_beat_state(bs_dict: dict) -> BeatState:
    """Construct a BeatState from a parsed play's beat_state dict."""
    return BeatState(**bs_dict)


def _extract_character_sequences(
    play_data: dict,
) -> dict[str, list[tuple[str, BeatState, list[str]]]]:
    """Extract per-character ordered sequences of (beat_id, BeatState, utterance_texts).

    Groups across all acts/scenes to produce the full play sequence per character.
    Returns {character: [(beat_id, BeatState, [utterance_texts]), ...]}.
    """
    char_sequences: dict[str, list[tuple[int, int, int, int, str, BeatState, list[str]]]] = (
        defaultdict(list)
    )

    for act in play_data["acts"]:
        act_num = act["number"]
        for scene in act["scenes"]:
            scene_num = scene["scene"]
            for beat in scene["beats"]:
                beat_id = beat["id"]
                beat_idx = beat["index"]

                # Build a lookup of utterances by speaker for this beat
                utt_by_speaker: dict[str, list[str]] = defaultdict(list)
                for utt in beat.get("utterances", []):
                    utt_by_speaker[utt["speaker"]].append(utt["text"])

                for bs_dict in beat.get("beat_states", []):
                    character = bs_dict["character"]
                    beat_state = _build_beat_state(bs_dict)
                    utterances = utt_by_speaker.get(character, [])
                    char_sequences[character].append(
                        (act_num, scene_num, beat_idx, 0, beat_id, beat_state, utterances)
                    )

    # Sort each character's sequence by (act, scene, beat_index)
    result: dict[str, list[tuple[str, BeatState, list[str]]]] = {}
    for character, entries in char_sequences.items():
        entries.sort(key=lambda x: (x[0], x[1], x[2]))
        result[character] = [(e[4], e[5], e[6]) for e in entries]

    return result


def _compute_affect_shift(posterior: PosteriorState) -> float:
    """Compute the magnitude of the affect shift in eigenspace (L2 norm of mean)."""
    return float(np.linalg.norm(posterior.affect_trans_mean))


# --------------------------------------------------------------------------- #
# Main smoothing logic
# --------------------------------------------------------------------------- #

def smooth_play(play_id: str, params: FactorParameters, tactic_vocab: list[str]) -> dict:
    """Run forward-backward smoothing for all characters in a play.

    Returns the smoothed output dict ready for JSON serialization.
    """
    print(f"\n{'='*60}")
    print(f"SMOOTHING: {play_id}")
    print(f"{'='*60}")

    # Load parsed play
    play_data = _load_play(play_id)
    print(f"  Loaded play: {play_data.get('title', play_id)}")

    # Build character bible lookup
    bible_lookup: dict[str, CharacterBible] = {}
    for cb_dict in play_data.get("character_bibles", []):
        character = cb_dict["character"]
        bible_lookup[character] = _build_character_bible(cb_dict)

    # Extract per-character beat sequences
    char_sequences = _extract_character_sequences(play_data)
    print(f"  Characters with beat_states: {len(char_sequences)}")
    for char, seq in sorted(char_sequences.items()):
        print(f"    {char}: {len(seq)} beats")

    # Output structure
    output = {
        "play_id": play_id,
        "smoothed_at": datetime.now(timezone.utc).isoformat(),
        "characters": {},
    }

    total_beats_smoothed = 0
    total_tactic_changes = 0

    for character, sequence in sorted(char_sequences.items()):
        n_beats = len(sequence)
        if n_beats < MIN_BEATS_FOR_SMOOTHING:
            print(f"  Skipping {character}: only {n_beats} beats (need >= {MIN_BEATS_FOR_SMOOTHING})")
            continue

        # Get character bible
        bible = bible_lookup.get(character)
        if bible is None:
            print(f"  Skipping {character}: no character bible found")
            continue

        print(f"\n  Smoothing {character} ({n_beats} beats)...")

        # Build factor graph for this character
        try:
            graph = CharacterFactorGraph(
                params=params,
                character_bible=bible,
                tactic_vocab=tactic_vocab,
            )
        except Exception as e:
            print(f"    ERROR building graph for {character}: {e}")
            continue

        # Collect BeatState sequence and utterances
        beat_states = [entry[1] for entry in sequence]
        utterances = [entry[2] for entry in sequence]
        beat_ids = [entry[0] for entry in sequence]

        # Run forward-backward smoother
        try:
            smoother = ForwardBackwardSmoother(graph)
            posteriors = smoother.smooth(beat_states, utterances)
        except Exception as e:
            print(f"    ERROR during smoothing for {character}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if len(posteriors) != n_beats:
            print(f"    WARNING: expected {n_beats} posteriors, got {len(posteriors)}")
            continue

        # Build per-beat output and compute diffs
        char_beats_output = []
        num_tactic_changes = 0
        affect_shifts = []

        for i, (beat_id, bs, posterior) in enumerate(
            zip(beat_ids, beat_states, posteriors)
        ):
            # Use canonical_tactic if available, otherwise uppercase tactic_state
            llm_tactic = bs.canonical_tactic or ""
            if not llm_tactic and bs.tactic_state:
                llm_tactic = bs.tactic_state.upper()
            smoothed_tactic = posterior.tactic_map
            tactic_changed = (
                llm_tactic != ""
                and smoothed_tactic != ""
                and llm_tactic != smoothed_tactic
            )
            if tactic_changed:
                num_tactic_changes += 1

            # Get the probability of the smoothed tactic
            smoothed_tactic_prob = posterior.tactic_distribution.get(smoothed_tactic, 0.0)

            # Affect shift magnitude
            affect_shift = _compute_affect_shift(posterior)
            affect_shifts.append(affect_shift)

            beat_entry = {
                "beat_id": beat_id,
                "llm_tactic": llm_tactic,
                "smoothed_tactic": smoothed_tactic,
                "smoothed_tactic_prob": round(smoothed_tactic_prob, 4),
                "tactic_distribution": {
                    k: round(v, 4) for k, v in sorted(
                        posterior.tactic_distribution.items(),
                        key=lambda x: -x[1],
                    )[:10]  # Top 10 for readability
                },
                "affect_trans_mean": [round(x, 4) for x in posterior.affect_trans_mean.tolist()],
                "affect_trans_std": [round(x, 4) for x in posterior.affect_trans_std.tolist()],
                "arousal": round(posterior.arousal, 4),
                "desire_distribution": {
                    k: round(v, 4) for k, v in sorted(
                        posterior.desire_distribution.items(),
                        key=lambda x: -x[1],
                    )
                },
                "social_mean": [round(x, 4) for x in posterior.social_mean.tolist()],
                "social_std": [round(x, 4) for x in posterior.social_std.tolist()],
                "changed": tactic_changed,
            }
            char_beats_output.append(beat_entry)

        mean_affect_shift = float(np.mean(affect_shifts)) if affect_shifts else 0.0

        output["characters"][character] = {
            "num_beats": n_beats,
            "num_tactic_changes": num_tactic_changes,
            "mean_affect_shift": round(mean_affect_shift, 4),
            "beats": char_beats_output,
        }

        total_beats_smoothed += n_beats
        total_tactic_changes += num_tactic_changes

        print(f"    Smoothed {n_beats} beats, {num_tactic_changes} tactic disagreements, "
              f"mean affect shift: {mean_affect_shift:.4f}")

    return output


def save_smoothed(output: dict, play_id: str) -> Path:
    """Save smoothed output to data/smoothed/{play_id}.json."""
    out_path = SMOOTHED_DIR / f"{play_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved: {out_path} ({size_kb:.1f} KB)")
    return out_path


def print_summary(output: dict) -> None:
    """Print a summary of smoothing results."""
    play_id = output["play_id"]
    characters = output["characters"]

    total_beats = sum(c["num_beats"] for c in characters.values())
    total_changes = sum(c["num_tactic_changes"] for c in characters.values())
    all_shifts = []
    for c in characters.values():
        all_shifts.append(c["mean_affect_shift"])

    mean_shift = float(np.mean(all_shifts)) if all_shifts else 0.0

    print(f"\n{'='*60}")
    print(f"SUMMARY: {play_id}")
    print(f"{'='*60}")
    print(f"  Characters smoothed:       {len(characters)}")
    print(f"  Total beats smoothed:      {total_beats}")
    print(f"  Tactic disagreements:      {total_changes} "
          f"({total_changes/total_beats*100:.1f}%)" if total_beats > 0 else "")
    print(f"  Mean affect shift:         {mean_shift:.4f}")
    print()
    print(f"  Per-character breakdown:")
    for char, data in sorted(characters.items()):
        pct = (data["num_tactic_changes"] / data["num_beats"] * 100
               if data["num_beats"] > 0 else 0.0)
        print(f"    {char:<20} {data['num_beats']:>3} beats, "
              f"{data['num_tactic_changes']:>2} changes ({pct:.0f}%), "
              f"affect shift={data['mean_affect_shift']:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Pass 1.5c: Run forward-backward smoothing over extracted BeatStates"
    )
    parser.add_argument(
        "play_id",
        nargs="?",
        help="Play ID to smooth (e.g., cherry_orchard, hamlet). "
             "Omit if using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Smooth all plays that have parsed data in data/parsed/.",
    )
    parser.add_argument(
        "--factors-dir",
        type=Path,
        default=FACTORS_DIR,
        help=f"Directory containing learned factor parameters (default: {FACTORS_DIR})",
    )
    args = parser.parse_args()

    if not args.play_id and not args.all:
        parser.error("Provide a play_id or use --all")

    # Determine which plays to smooth
    if args.all:
        play_ids = []
        for play_id in PLAYS:
            parsed_path = PARSED_DIR / f"{play_id}.json"
            if parsed_path.exists():
                play_ids.append(play_id)
        if not play_ids:
            print("ERROR: No parsed plays found in data/parsed/.")
            sys.exit(1)
        print(f"Smoothing all available plays: {', '.join(play_ids)}")
    else:
        play_ids = [args.play_id]

    # Load factor parameters
    print(f"\nLoading factor parameters from {args.factors_dir}...")
    try:
        params = FactorParameters.load(args.factors_dir)
        print(f"  Loaded: {len(params.tactic_vocab)} tactics, "
              f"{len(params.desire_cluster_labels)} desire clusters")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nFactor parameters not found. Run the learning pipeline first:")
        print("  conda run -n uta_model python -m factor_graph.learning "
              "--plays cherry_orchard hamlet importance_of_being_earnest")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading factor parameters: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load tactic vocabulary
    print("Loading tactic vocabulary...")
    try:
        tactic_vocab = _load_tactic_vocab()
        print(f"  Loaded: {len(tactic_vocab)} canonical tactics")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Smooth each play
    for play_id in play_ids:
        try:
            output = smooth_play(play_id, params, tactic_vocab)
            save_smoothed(output, play_id)
            print_summary(output)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print(f"Skipping {play_id}.")
            continue
        except Exception as e:
            print(f"\nERROR smoothing {play_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
