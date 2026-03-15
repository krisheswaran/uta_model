"""
Run the three-tier evaluation protocol.

Generates lines under three conditions (vanilla LLM, text-only bible,
full reflection loop) for a set of scene prompts, judges each on seven
dimensions, and prints a summary table.

Usage:
    conda run -n uta_model python scripts/run_evaluation.py \
        --character LOPAKHIN --play cherry_orchard \
        --num-scenes 5 --judge-runs 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BIBLES_DIR, PLAYS
from schemas import CharacterBible, SceneContext
from evaluation.judge import evaluate_three_tiers, summarize_ratings

# ── Built-in evaluation scenes for Lopakhin ──────────────────────────────────
LOPAKHIN_EVAL_SCENES: list[SceneContext] = [
    SceneContext(
        play_id="cherry_orchard",
        character="LOPAKHIN",
        setting="A counting-house in St. Petersburg. Lopakhin is being congratulated on a profitable deal.",
        characters_present=["LOPAKHIN", "BUSINESS PARTNER"],
        prior_events="Lopakhin has just sealed a deal that makes him one of the wealthiest men in the province.",
        stakes="He should feel triumphant, but cannot.",
        partner_line="You've done it, Yermolai. You've really done it this time.",
        register="dramatic",
        constraint="alternate_universe_same_psyche",
    ),
    SceneContext(
        play_id="cherry_orchard",
        character="LOPAKHIN",
        setting="The nursery of a grand estate, late autumn, just before auction.",
        characters_present=["LOPAKHIN", "RANYEVSKAYA"],
        prior_events="Lopakhin has presented his plan to save the estate by subdividing it. Ranyevskaya has ignored the plan for months.",
        stakes="The estate will be lost in two days if she does not act.",
        partner_line="I cannot bear to think of cutting down the trees.",
        register="dramatic",
        constraint="faithful_to_arc",
    ),
    SceneContext(
        play_id="cherry_orchard",
        character="LOPAKHIN",
        setting="An empty train platform. A woman from his past is leaving forever.",
        characters_present=["LOPAKHIN", "VARYA"],
        prior_events="Everyone has left the estate. Varya is the last to go.",
        stakes="This is the last chance to say what has never been said.",
        partner_line="Well. Goodbye then.",
        register="dramatic",
        constraint="alternate_universe_same_psyche",
    ),
]


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


def main():
    parser = argparse.ArgumentParser(description="Run three-tier evaluation")
    parser.add_argument("--character", default="LOPAKHIN")
    parser.add_argument("--play", default="cherry_orchard", choices=list(PLAYS.keys()))
    parser.add_argument("--num-scenes", type=int, default=3,
                        help="Number of built-in eval scenes to use")
    parser.add_argument("--judge-runs", type=int, default=3,
                        help="Number of judge passes per line (for reliability)")
    parser.add_argument("--output", default="",
                        help="Path to write JSON results (default: stdout summary only)")
    args = parser.parse_args()

    print(f"\n=== Three-Tier Evaluation: {args.character} ({args.play}) ===\n")
    bible = load_bible(args.play, args.character)

    # Use built-in eval scenes (or a subset)
    scenes = LOPAKHIN_EVAL_SCENES[: args.num_scenes]
    print(f"Using {len(scenes)} evaluation scenes, {args.judge_runs} judge runs per line.\n")

    ratings = evaluate_three_tiers(bible, scenes, num_judge_runs=args.judge_runs)
    summary = summarize_ratings(ratings)

    print("\n=== Results ===\n")
    dims = ["recognizability", "playability", "tactic_fidelity",
            "subtext", "earned_affect", "knowledge_fidelity_pass_rate", "mean_overall"]
    header = f"{'Dimension':<30} {'Vanilla':>10} {'Bible':>10} {'Reflection':>12}"
    print(header)
    print("-" * len(header))
    for dim in dims:
        row = f"{dim:<30}"
        for tier in ("vanilla", "bible", "reflection"):
            val = summary.get(tier, {}).get(dim, "-")
            row += f" {val:>10}"
        print(row)

    if args.output:
        out = {
            "summary": summary,
            "ratings": [r.model_dump() for r in ratings],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
