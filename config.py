"""
Central configuration for the UTA Acting System.
Loads settings from environment variables / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
BEATS_DIR = DATA_DIR / "beats"
BIBLES_DIR = DATA_DIR / "bibles"

for _d in (RAW_DIR, PARSED_DIR, BEATS_DIR, BIBLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Model tiers
ANALYSIS_MODEL = "claude-opus-4-6"       # dramatic analysis pipeline
GENERATION_MODEL = "claude-sonnet-4-6"   # improvisation generation
CRITIC_MODEL = "claude-sonnet-4-6"       # improvisation scoring + feedback
JUDGE_MODEL = "claude-opus-4-6"          # three-tier evaluation judge

# --------------------------------------------------------------------------- #
# Analysis pipeline settings
# --------------------------------------------------------------------------- #
BEAT_CONTEXT_WINDOW = 40        # utterances on each side shown to beat segmenter
MAX_UTTERANCES_PER_EXTRACT = 20 # max utterances fed to per-beat extractor at once
SMOOTH_PASSES = 2               # global arc smoothing iterations

# --------------------------------------------------------------------------- #
# Improvisation settings
# --------------------------------------------------------------------------- #
MAX_REVISION_ROUNDS = 3
SCORE_THRESHOLD = 3.0           # axes below this trigger targeted feedback (scale 1-5)

# --------------------------------------------------------------------------- #
# Plays registry
# --------------------------------------------------------------------------- #
PLAYS = {
    "cherry_orchard": {
        "title": "The Cherry Orchard",
        "author": "Anton Chekhov",
        "translator": "Constance Garnett",
        "source": "gutenberg",
        "gutenberg_id": 7986,
        "primary_character": "LOPAKHIN",
        # Gutenberg #7986 is a multi-play volume; anchor to Cherry Orchard's title line
        "text_anchor": "THE CHERRY ORCHARD",
    },
    "hamlet": {
        "title": "Hamlet",
        "author": "William Shakespeare",
        "source": "dracor_tei",
        "dracor_corpus": "shake",
        "dracor_play": "hamlet",
        "primary_character": "HAMLET",
    },
}
