"""
Run the full dramatic analysis pipeline (formerly Pass 1) on a play.

Steps:
  1. Parse the raw text into a structured Play object
  2. Segment scenes into beats (LLM annotation, cached)
  3. Extract BeatState per beat per character
  4. Smooth character arcs globally
  5. Build CharacterBible, SceneBible, WorldBible

Usage:
    conda run -n uta_model python scripts/run_analysis.py cherry_orchard
    conda run -n uta_model python scripts/run_analysis.py hamlet
    conda run -n uta_model python scripts/run_analysis.py cherry_orchard --characters LOPAKHIN RANYEVSKAYA
    conda run -n uta_model python scripts/run_analysis.py cherry_orchard --skip-segmentation
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PARSED_DIR, PLAYS, RAW_DIR
from schemas import Play
from ingest.gutenberg_parser import parse_gutenberg_play
from ingest.tei_parser import parse_tei_play
from analysis.segmenter import segment_play
from analysis.extractor import extract_all_beats
from analysis.smoother import smooth_play
from analysis.bible_builder import build_all_bibles


def load_play(play_id: str):
    config = PLAYS[play_id]
    parser = config["parser"]

    if parser == "gutenberg":
        raw_path = RAW_DIR / f"{play_id}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"{raw_path} not found. Run download_plays.py first.")
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        return parse_gutenberg_play(
            play_id,
            raw_text,
            title=config["title"],
            author=config["author"],
            primary_character=config.get("primary_character", ""),
            text_anchor=config.get("text_anchor", ""),
        )
    elif parser == "tei":
        raw_path = RAW_DIR / f"{play_id}.xml"
        if not raw_path.exists():
            raise FileNotFoundError(f"{raw_path} not found. Run download_plays.py first.")
        return parse_tei_play(
            play_id,
            raw_path.read_bytes(),
            title=config["title"],
            author=config["author"],
        )
    else:
        raise ValueError(f"Unknown parser: {parser!r}")


def main():
    parser = argparse.ArgumentParser(description="Run dramatic analysis pipeline on a play")
    parser.add_argument("play_id", choices=list(PLAYS.keys()), help="Play to analyse")
    parser.add_argument("--characters", nargs="*", help="Limit character bibles to these characters")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip beat segmentation (use cached beats or provisional single-beat per scene)")
    parser.add_argument("--skip-smoothing", action="store_true", help="Skip global arc smoothing pass")
    parser.add_argument("--bibles-only", action="store_true",
                        help="Skip steps 1-4; load existing parsed play and build only missing bibles")
    parser.add_argument("--min-beat-states", type=int, default=0,
                        help="Only build bibles for characters with >= N beat states (default: 0 = all)")
    args = parser.parse_args()

    print(f"\n=== Dramatic Analysis: {args.play_id} ===\n")

    if args.bibles_only:
        # Load existing parsed play and build only missing bibles
        parsed_path = PARSED_DIR / f"{args.play_id}.json"
        if not parsed_path.exists():
            raise FileNotFoundError(
                f"{parsed_path} not found. Run the full pipeline first before using --bibles-only.")
        print("Loading existing parsed play (--bibles-only mode)...")
        play = Play.model_validate_json(parsed_path.read_text())
        total_chars = len(play.characters)
        existing_bibles = len(play.character_bibles)
        print(f"  {total_chars} characters, {existing_bibles} existing bibles")

        print("\nBuilding missing bibles...")
        play = build_all_bibles(
            play,
            characters=args.characters,
            skip_scene_bibles=True,
            skip_world_bible=True,
            min_beat_states=args.min_beat_states,
        )

        # Save updated play
        parsed_path.write_text(play.model_dump_json(indent=2))
        print(f"\nUpdated play object saved to {parsed_path}")
        print(f"\n=== Bible building complete for {play.title} ===")
        return

    # Full pipeline
    # Step 1: Parse
    print("Step 1: Parsing raw text...")
    play = load_play(args.play_id)
    total_utterances = sum(1 for _ in play.iter_utterances())
    print(f"  {len(play.acts)} acts, {sum(len(a.scenes) for a in play.acts)} scenes, "
          f"{total_utterances} utterances, {len(play.characters)} characters")

    # Step 2: Beat segmentation
    if not args.skip_segmentation:
        print("\nStep 2: Beat segmentation...")
        play = segment_play(play, use_cache=True)
        total_beats = sum(len(s.beats) for a in play.acts for s in a.scenes)
        print(f"  {total_beats} beats total")
    else:
        print("\nStep 2: Skipping segmentation (using provisional beats)")

    # Step 3: Extract BeatStates
    print("\nStep 3: Extracting dramatic states...")
    play = extract_all_beats(play)

    # Step 4: Smooth arcs
    if not args.skip_smoothing:
        print("\nStep 4: Smoothing character arcs...")
        play = smooth_play(play)
    else:
        print("\nStep 4: Skipping arc smoothing")

    # Step 5: Build bibles
    print("\nStep 5: Building bibles...")
    play = build_all_bibles(play, characters=args.characters)

    # Save parsed play
    parsed_path = PARSED_DIR / f"{args.play_id}.json"
    parsed_path.write_text(play.model_dump_json(indent=2))
    print(f"\nFull play object saved to {parsed_path}")
    print(f"\n=== Analysis complete for {play.title} ===")


if __name__ == "__main__":
    main()
