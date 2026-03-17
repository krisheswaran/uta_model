"""
Regression test for Gutenberg and TEI play parsing.

Run manually (not via pytest) — requires downloaded play texts in data/raw/.
Run this after any changes to ingest/gutenberg_parser.py or ingest/tei_parser.py.

Usage:
    conda run -n uta_model python scripts/regression_test_parsing.py
    conda run -n uta_model python scripts/regression_test_parsing.py cherry_orchard uncle_vanya
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PLAYS, RAW_DIR
from ingest.gutenberg_parser import parse_gutenberg_play
from ingest.tei_parser import parse_tei_play

# ────────────────────────────────────────────────────────────────────────────
# Expected values — update these when parser changes are intentional.
# ────────────────────────────────────────────────────────────────────────────
EXPECTED = {
    "cherry_orchard": {
        "acts": 4,
        "scenes": 4,
        "utterances": 640,
        "characters": 19,
        "first_speaker": "LOPAKHIN",
        "first_text_startswith": "The train",
        "must_include_characters": {"LOPAKHIN", "LUBOV ANDREYEVNA", "ANYA", "VARYA", "GAEV", "TROFIMOV"},
        "must_exclude_characters": set(),
    },
    "uncle_vanya": {
        "acts": 4,
        "scenes": 4,
        "utterances": 539,
        "characters": 11,
        "first_speaker": "MARINA",
        "first_text_startswith": "Take a little tea",
        "must_include_characters": {"VOITSKI", "ASTROFF", "SONIA", "HELENA", "SEREBRAKOFF", "MME. VOITSKAYA"},
        "must_exclude_characters": {"DRAMATIS PERSONAE", "CHARACTERS"},
    },
    "dolls_house": {
        "acts": 3,
        "scenes": 3,
        "utterances": 1290,
        "characters": 12,
        "first_speaker": "NORA",
        "first_text_startswith": "Hide the Christmas Tree",
        "must_include_characters": {"NORA", "HELMER", "DOCTOR RANK", "MRS LINDE", "KROGSTAD"},
        "must_exclude_characters": {"DRAMATIS PERSONAE"},
    },
    "importance_of_being_earnest": {
        "acts": 3,
        "scenes": 3,
        "utterances": 873,
        "characters": 9,
        "first_speaker": "ALGERNON",
        "first_text_startswith": "Did you hear what I was playing",
        "must_include_characters": {"ALGERNON", "JACK", "GWENDOLEN", "CECILY", "LADY BRACKNELL"},
        "must_exclude_characters": {"ACT DROP"},
    },
    "hamlet": {
        "acts": 5,
        "scenes": 20,
        "utterances": 1138,
        "characters": 39,
        "first_speaker": "BARNARDO",
        "first_text_startswith": "Who",
        "must_include_characters": {"HAMLET", "HORATIO", "OPHELIA", "KING", "QUEEN", "LAERTES", "POLONIUS", "GHOST"},
        "must_exclude_characters": set(),
    },
    "macbeth": {
        "acts": 5,
        "scenes": 28,
        "utterances": 649,
        "characters": 42,
        "first_speaker": "FIRST WITCH",
        "first_text_startswith": "When shall we three meet again",
        "must_include_characters": {"MACBETH", "LADY MACBETH", "BANQUO", "MACDUFF", "MALCOLM", "DUNCAN"},
        "must_exclude_characters": set(),
    },
}


def load_play(play_id: str):
    """Load and parse a play using the appropriate parser."""
    config = PLAYS[play_id]
    parser = config["parser"]

    if parser == "gutenberg":
        raw_path = RAW_DIR / f"{play_id}.txt"
        if not raw_path.exists():
            return None, f"raw file not found: {raw_path}"
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        return parse_gutenberg_play(
            play_id,
            raw_text,
            title=config["title"],
            author=config["author"],
            primary_character=config.get("primary_character", ""),
            text_anchor=config.get("text_anchor", ""),
        ), None
    elif parser == "tei":
        raw_path = RAW_DIR / f"{play_id}.xml"
        if not raw_path.exists():
            return None, f"raw file not found: {raw_path}"
        return parse_tei_play(
            play_id,
            raw_path.read_bytes(),
            title=config["title"],
            author=config["author"],
        ), None
    else:
        return None, f"unknown parser: {parser!r}"


def check_play(play_id: str, expected: dict) -> list[str]:
    """Parse a play and compare against expected values. Returns list of failures."""
    play, err = load_play(play_id)
    if err:
        return [f"SKIP ({err})"]

    failures = []
    n_acts = len(play.acts)
    n_scenes = sum(len(a.scenes) for a in play.acts)
    n_utts = sum(1 for _ in play.iter_utterances())
    n_chars = len(play.characters)
    char_set = set(play.characters)

    if n_acts != expected["acts"]:
        failures.append(f"acts: got {n_acts}, expected {expected['acts']}")
    if n_scenes != expected["scenes"]:
        failures.append(f"scenes: got {n_scenes}, expected {expected['scenes']}")
    if n_utts != expected["utterances"]:
        failures.append(f"utterances: got {n_utts}, expected {expected['utterances']}")
    if n_chars != expected["characters"]:
        failures.append(f"characters: got {n_chars}, expected {expected['characters']}")

    first_utt = next(play.iter_utterances(), None)
    if first_utt is None:
        failures.append("no utterances found")
    else:
        if first_utt.speaker != expected["first_speaker"]:
            failures.append(f"first_speaker: got {first_utt.speaker!r}, expected {expected['first_speaker']!r}")
        if not first_utt.text.startswith(expected["first_text_startswith"]):
            failures.append(
                f"first_text: got {first_utt.text[:60]!r}, "
                f"expected starts with {expected['first_text_startswith']!r}"
            )

    missing = expected["must_include_characters"] - char_set
    if missing:
        failures.append(f"missing expected characters: {missing}")

    unwanted = expected["must_exclude_characters"] & char_set
    if unwanted:
        failures.append(f"unwanted characters present: {unwanted}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Regression test for play parsing")
    parser.add_argument("play_ids", nargs="*", help="Play IDs to test (default: all with expectations)")
    args = parser.parse_args()

    targets = args.play_ids or list(EXPECTED.keys())
    total = 0
    passed = 0
    skipped = 0
    failed = 0

    for play_id in targets:
        if play_id not in EXPECTED:
            print(f"  {play_id}: no expected values defined, skipping")
            skipped += 1
            continue

        total += 1
        failures = check_play(play_id, EXPECTED[play_id])

        if len(failures) == 1 and failures[0].startswith("SKIP"):
            print(f"  {play_id}: {failures[0]}")
            skipped += 1
            continue

        if failures:
            failed += 1
            print(f"  FAIL  {play_id}")
            for f in failures:
                print(f"        - {f}")
        else:
            passed += 1
            print(f"  OK    {play_id}")

    print(f"\n{total} tested, {passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
