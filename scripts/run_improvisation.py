"""
Run an improvisation session for a character from a built bible.

Supports three modes:
  single    — one character responds to a series of manually supplied partner lines
  session   — interactive session where you type partner lines in a loop
  crossplay — two-character cross-play scene (e.g. Lopakhin meets Hamlet)

Usage:
    # Single character session (Lopakhin, Cherry Orchard)
    conda run -n uta_model python scripts/run_improvisation.py \
        --character LOPAKHIN \
        --play cherry_orchard \
        --setting "A bare office in Moscow, late night" \
        --stakes "Lopakhin is about to lose everything he built" \
        --mode session

    # Cross-play scene
    conda run -n uta_model python scripts/run_improvisation.py \
        --mode crossplay \
        --character-a LOPAKHIN --play-a cherry_orchard \
        --character-b HAMLET   --play-b hamlet \
        --setting "A crumbling estate in an unnamed country, dusk" \
        --stakes "Both men have come to say goodbye to something they cannot name"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BIBLES_DIR, PLAYS
from schemas import CharacterBible, SceneContext
from improv.improvisation_loop import run_session, initialize_beat_state, run_turn


def load_bible(play_id: str, character: str) -> CharacterBible:
    bible_path = BIBLES_DIR / f"{play_id}_bibles.json"
    if not bible_path.exists():
        raise FileNotFoundError(
            f"No bible found for {play_id}. Run scripts/run_analysis.py {play_id} first."
        )
    data = json.loads(bible_path.read_text())
    for cb_data in data.get("character_bibles", []):
        if cb_data["character"].upper() == character.upper():
            return CharacterBible(**cb_data)
    raise ValueError(f"No CharacterBible for {character!r} in {play_id}")


def mode_session(args):
    """Interactive session: user types partner lines, character responds."""
    bible = load_bible(args.play, args.character)
    setting = args.setting or "An unspecified room"
    stakes = args.stakes or "The stakes are unclear but real"

    print(f"\n=== Improvisation Session: {bible.character} ({bible.play_id}) ===")
    print(f"Setting: {setting}")
    print(f"Stakes:  {stakes}")
    print(f"(Type partner lines and press Enter. Type 'quit' to end.)\n")

    context = SceneContext(
        play_id=bible.play_id,
        character=bible.character,
        setting=setting,
        characters_present=[bible.character, "PARTNER"],
        prior_events=args.prior_events or "The scene begins.",
        stakes=stakes,
    )
    beat_state = initialize_beat_state(bible.character, bible, context)

    turn_index = 0
    while True:
        try:
            partner_line = input("Partner: ").strip()
        except EOFError:
            print("\n(stdin closed — ending session)")
            break
        if partner_line.lower() in ("quit", "exit", "q"):
            break
        if not partner_line:
            continue

        turn_context = context.model_copy(update={"partner_line": partner_line})
        turn, beat_state = run_turn(turn_index + 1, beat_state, bible, turn_context,
                                    min_revisions=args.min_revisions)
        print(f"\n{bible.character}: {turn.final_line}")
        print(f"  [score={turn.scored_line.mean_score:.2f} | tactic={beat_state.tactic_state} | "
              f"revisions={turn.revisions}]\n")
        turn_index += 1


def mode_crossplay(args):
    """Two characters alternating lines in a novel scene."""
    bible_a = load_bible(args.play_a, args.character_a)
    bible_b = load_bible(args.play_b, args.character_b)
    setting = args.setting or "An unspecified space"
    stakes = args.stakes or "The meeting is significant"
    num_turns = args.turns or 6

    print(f"\n=== Cross-Play Scene: {bible_a.character} meets {bible_b.character} ===")
    print(f"Setting: {setting}")
    print(f"Stakes:  {stakes}\n")

    context_a = SceneContext(
        play_id=bible_a.play_id,
        character=bible_a.character,
        setting=setting,
        characters_present=[bible_a.character, bible_b.character],
        prior_events=args.prior_events or "The two characters have just met.",
        stakes=stakes,
        constraint="alternate_universe_same_psyche",
    )
    context_b = SceneContext(
        play_id=bible_b.play_id,
        character=bible_b.character,
        setting=setting,
        characters_present=[bible_a.character, bible_b.character],
        prior_events=args.prior_events or "The two characters have just met.",
        stakes=stakes,
        constraint="alternate_universe_same_psyche",
    )

    state_a = initialize_beat_state(bible_a.character, bible_a, context_a)
    state_b = initialize_beat_state(bible_b.character, bible_b, context_b)

    last_line = None
    for i in range(num_turns):
        # A speaks
        ctx = context_a.model_copy(update={"partner_line": last_line})
        turn_a, state_a = run_turn(i + 1, state_a, bible_a, ctx,
                                   min_revisions=args.min_revisions)
        last_line = turn_a.final_line
        print(f"{bible_a.character}: {last_line}")
        print(f"  [tactic={state_a.tactic_state} | score={turn_a.scored_line.mean_score:.2f}]")

        # B speaks
        ctx = context_b.model_copy(update={"partner_line": last_line})
        turn_b, state_b = run_turn(i + 1, state_b, bible_b, ctx,
                                   min_revisions=args.min_revisions)
        last_line = turn_b.final_line
        print(f"{bible_b.character}: {last_line}")
        print(f"  [tactic={state_b.tactic_state} | score={turn_b.scored_line.mean_score:.2f}]\n")


def main():
    parser = argparse.ArgumentParser(description="Run an improvisation session")
    sub = parser.add_subparsers(dest="mode")

    # Session mode
    session_p = sub.add_parser("session", help="Interactive single-character session")
    session_p.add_argument("--character", required=True)
    session_p.add_argument("--play", required=True, choices=list(PLAYS.keys()))
    session_p.add_argument("--setting", default="")
    session_p.add_argument("--stakes", default="")
    session_p.add_argument("--prior-events", default="")
    session_p.add_argument("--min-revisions", type=int, default=None,
                           help="Minimum revision rounds per turn (overrides config)")

    # Crossplay mode
    cross_p = sub.add_parser("crossplay", help="Two-character cross-play scene")
    cross_p.add_argument("--character-a", required=True)
    cross_p.add_argument("--play-a", required=True, choices=list(PLAYS.keys()))
    cross_p.add_argument("--character-b", required=True)
    cross_p.add_argument("--play-b", required=True, choices=list(PLAYS.keys()))
    cross_p.add_argument("--setting", default="")
    cross_p.add_argument("--stakes", default="")
    cross_p.add_argument("--prior-events", default="")
    cross_p.add_argument("--turns", type=int, default=6)
    cross_p.add_argument("--min-revisions", type=int, default=None,
                           help="Minimum revision rounds per turn (overrides config)")

    args = parser.parse_args()

    if args.mode == "session":
        mode_session(args)
    elif args.mode == "crossplay":
        mode_crossplay(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
