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
IMPROV_DIR = DATA_DIR / "improv"

for _d in (RAW_DIR, PARSED_DIR, BEATS_DIR, BIBLES_DIR, IMPROV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# --------------------------------------------------------------------------- #
# Model configuration — per-step tiering (Phase B)
#
# Each step of the pipeline can use a different model. The provider field
# enables future multi-provider support (Gemini, OpenAI) but currently
# only "anthropic" is implemented.
# --------------------------------------------------------------------------- #
MODEL_CONFIGS = {
    # Analysis pipeline (Pass 1)
    "segmentation": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "extraction":   {"provider": "anthropic", "model": "claude-opus-4-6"},
    "smoothing":    {"provider": "anthropic", "model": "claude-opus-4-6"},
    "bible":        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "world_bible":  {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    # Improvisation (Pass 2)
    "generation":   {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "critic":       {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    # Evaluation (Pass 3)
    "judge":        {"provider": "anthropic", "model": "claude-opus-4-6"},
}


def get_model(step: str) -> str:
    """Return the model ID for a given pipeline step."""
    return MODEL_CONFIGS[step]["model"]


# Legacy aliases (for backward compatibility during migration)
ANALYSIS_MODEL = get_model("extraction")
GENERATION_MODEL = get_model("generation")
CRITIC_MODEL = get_model("critic")
JUDGE_MODEL = get_model("judge")

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
MIN_REVISION_ROUNDS = 1         # minimum revisions even when scores pass (Phase B)
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
        "source": "folger_tei",
        "dracor_corpus": "shake",
        "dracor_play": "hamlet",
        "primary_character": "HAMLET",
    },
}
