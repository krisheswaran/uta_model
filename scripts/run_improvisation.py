"""
Run an improvisation session for a character from a built bible.

Supports two modes:
  session   — interactive session where you type partner lines in a loop
  crossplay — two-character cross-play scene (e.g. Lopakhin meets Hamlet)

All scenes are saved to data/improv/ as JSON with full revision traces,
dramaturgical feedback, and the configuration used to generate the scene.

Usage:
    # Interactive session (Lopakhin, Cherry Orchard)
    python scripts/run_improvisation.py session \
        --character LOPAKHIN --play cherry_orchard \
        --setting "A bare office in Moscow, late night" \
        --stakes "Lopakhin is about to lose everything he built"

    # Cross-play scene
    python scripts/run_improvisation.py crossplay \
        --character-a LOPAKHIN --play-a cherry_orchard \
        --character-b HAMLET   --play-b hamlet \
        --setting "A crumbling estate in an unnamed country, dusk" \
        --stakes "Both men have come to say goodbye to something they cannot name"
"""
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BIBLES_DIR, IMPROV_DIR, MODEL_CONFIGS, PARSED_DIR, PLAYS,
    MAX_REVISION_ROUNDS, MIN_REVISION_ROUNDS, SCORE_THRESHOLD,
)
from schemas import (
    CharacterBible, ImprovTurn, Play, SceneContext, SceneRecord,
    StatisticalPrior,
)
from improv.improvisation_loop import run_session, initialize_beat_state, run_turn


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _try_load_prior(play_id: str, character: str) -> tuple[StatisticalPrior | None, Play | None]:
    """Try to load statistical priors. Returns (None, None) if not available."""
    try:
        from improv.priors import load_prior_for_character
        prior = load_prior_for_character(play_id, character)
        play = Play.model_validate_json((PARSED_DIR / f"{play_id}.json").read_text())
        return prior, play
    except (FileNotFoundError, Exception) as e:
        print(f"  [info] No statistical priors for {character}: {e}")
        return None, None


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


def _character_info(character: str, play_id: str, prior: StatisticalPrior | None) -> dict:
    """Build a character summary dict for the SceneRecord."""
    info = {"character": character, "play_id": play_id, "has_prior": prior is not None}
    if prior and prior.character_tactic_prior:
        top = max(prior.character_tactic_prior, key=prior.character_tactic_prior.get)
        info["top_tactic"] = top
        info["top_tactic_pct"] = round(prior.character_tactic_prior[top] * 100, 1)
    if prior and prior.relational_profile:
        info["default_warmth"] = prior.relational_profile.default_warmth
        info["default_status"] = prior.relational_profile.default_status_claim
    return info


def _config_snapshot(args) -> dict:
    """Capture the full configuration used for this scene."""
    return {
        "cli_args": {k: v for k, v in vars(args).items() if v is not None and v != ""},
        "model_configs": {
            step: cfg for step, cfg in MODEL_CONFIGS.items()
            if step in ("generation", "critic")
        },
        "pipeline_params": {
            "max_revision_rounds": MAX_REVISION_ROUNDS,
            "min_revision_rounds": args.min_revisions if args.min_revisions is not None else MIN_REVISION_ROUNDS,
            "score_threshold": SCORE_THRESHOLD,
        },
    }


def _save_scene(record: SceneRecord) -> Path:
    """Save a SceneRecord to data/improv/ and return the path."""
    path = IMPROV_DIR / f"{record.scene_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    print(f"\nScene saved to {path}")
    return path


# ────────────────────────────────────────────────────────────────────────────
# Session mode
# ────────────────────────────────────────────────────────────────────────────

def mode_session(args):
    """Interactive session: user types partner lines, character responds."""
    bible = load_bible(args.play, args.character)
    prior, play = _try_load_prior(args.play, args.character)
    setting = args.setting or "An unspecified room"
    stakes = args.stakes or "The stakes are unclear but real"
    scene_id = f"session_{bible.character.lower()}_{uuid.uuid4().hex[:8]}"

    print(f"\n=== Improvisation Session: {bible.character} ({bible.play_id}) ===")
    if prior and prior.character_tactic_prior:
        top = max(prior.character_tactic_prior, key=prior.character_tactic_prior.get)
        print(f"  [priors loaded: top tactic={top}, "
              f"warmth={prior.relational_profile.default_warmth:+.2f}]")
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

    turns: list[ImprovTurn] = []
    transcript: list[dict] = []
    turn_index = 0
    previous_tactic = None

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

        # Record partner line in transcript
        transcript.append({"speaker": "PARTNER", "line": partner_line})

        turn_context = context.model_copy(update={"partner_line": partner_line})
        turn, beat_state = run_turn(
            turn_index + 1, beat_state, bible, turn_context,
            min_revisions=args.min_revisions,
            prior=prior, play=play, previous_tactic=previous_tactic,
        )
        previous_tactic = beat_state.tactic_state
        turns.append(turn)
        transcript.append({
            "speaker": bible.character,
            "line": turn.final_line,
            "tactic": beat_state.tactic_state,
            "mean_score": round(turn.scored_line.mean_score, 2),
        })

        print(f"\n{bible.character}: {turn.final_line}")
        print(f"  [score={turn.scored_line.mean_score:.2f} | tactic={beat_state.tactic_state} | "
              f"revisions={turn.revisions}]\n")
        turn_index += 1

    # Save the scene
    record = SceneRecord(
        scene_id=scene_id,
        mode="session",
        timestamp=datetime.now(timezone.utc).isoformat(),
        setting=setting,
        stakes=stakes,
        prior_events=args.prior_events or "The scene begins.",
        characters=[_character_info(bible.character, bible.play_id, prior)],
        config=_config_snapshot(args),
        turns=turns,
        transcript=transcript,
    )
    _save_scene(record)


# ────────────────────────────────────────────────────────────────────────────
# Crossplay mode
# ────────────────────────────────────────────────────────────────────────────

def mode_crossplay(args):
    """Two characters alternating lines in a novel scene."""
    bible_a = load_bible(args.play_a, args.character_a)
    bible_b = load_bible(args.play_b, args.character_b)
    prior_a, play_a = _try_load_prior(args.play_a, args.character_a)
    prior_b, play_b = _try_load_prior(args.play_b, args.character_b)
    setting = args.setting or "An unspecified space"
    stakes = args.stakes or "The meeting is significant"
    num_turns = args.turns or 6
    scene_id = (f"crossplay_{bible_a.character.lower()}_"
                f"{bible_b.character.lower()}_{uuid.uuid4().hex[:8]}")

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

    turns: list[ImprovTurn] = []
    transcript: list[dict] = []
    last_line = None
    prev_tactic_a = None
    prev_tactic_b = None

    for i in range(num_turns):
        # A speaks
        ctx = context_a.model_copy(update={"partner_line": last_line})
        turn_a, state_a = run_turn(
            i + 1, state_a, bible_a, ctx,
            min_revisions=args.min_revisions,
            prior=prior_a, play=play_a, previous_tactic=prev_tactic_a,
        )
        prev_tactic_a = state_a.tactic_state
        last_line = turn_a.final_line
        turns.append(turn_a)
        transcript.append({
            "speaker": bible_a.character,
            "line": turn_a.final_line,
            "tactic": state_a.tactic_state,
            "mean_score": round(turn_a.scored_line.mean_score, 2),
        })
        print(f"{bible_a.character}: {last_line}")
        print(f"  [tactic={state_a.tactic_state} | score={turn_a.scored_line.mean_score:.2f}]")

        # B speaks
        ctx = context_b.model_copy(update={"partner_line": last_line})
        turn_b, state_b = run_turn(
            i + 1, state_b, bible_b, ctx,
            min_revisions=args.min_revisions,
            prior=prior_b, play=play_b, previous_tactic=prev_tactic_b,
        )
        prev_tactic_b = state_b.tactic_state
        last_line = turn_b.final_line
        turns.append(turn_b)
        transcript.append({
            "speaker": bible_b.character,
            "line": turn_b.final_line,
            "tactic": state_b.tactic_state,
            "mean_score": round(turn_b.scored_line.mean_score, 2),
        })
        print(f"{bible_b.character}: {last_line}")
        print(f"  [tactic={state_b.tactic_state} | score={turn_b.scored_line.mean_score:.2f}]\n")

    # Save the scene
    record = SceneRecord(
        scene_id=scene_id,
        mode="crossplay",
        timestamp=datetime.now(timezone.utc).isoformat(),
        setting=setting,
        stakes=stakes,
        prior_events=args.prior_events or "The two characters have just met.",
        constraint="alternate_universe_same_psyche",
        characters=[
            _character_info(bible_a.character, bible_a.play_id, prior_a),
            _character_info(bible_b.character, bible_b.play_id, prior_b),
        ],
        config=_config_snapshot(args),
        turns=turns,
        transcript=transcript,
    )
    _save_scene(record)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

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
