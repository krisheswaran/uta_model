# UTA: A Latent-State Acting System

UTA models theatrical characters as multi-scale latent dynamical systems and generates character-faithful improvised dialogue. It treats dialogue not as text generation, but as the surface emission of hidden dramatic states — objectives, tactics, emotions, knowledge, and relationships — grounded in Stanislavsky-tradition acting theory.

The system works in two passes:

1. **Analysis (Pass 1)**: Parse a play, segment it into beats, extract per-character latent states (desire, affect, tactic, epistemic, social, defense), smooth arcs globally, and build structured character/scene/world bibles.
2. **Improvisation (Pass 2)**: Given a character bible and a novel scene context, initialize a hidden state, generate candidate lines, score them on six axes (voice, tactic, knowledge, relationship, subtext, emotional plausibility), and revise via targeted feedback.

## Setup

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/) (recommended) or a virtual environment
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/uta_model.git
cd uta_model

# Create and activate a conda environment
conda create -n uta_model python=3.11 -y
conda activate uta_model

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with your Anthropic API key:

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

Model selection and pipeline parameters can be adjusted in `config.py`.

## Usage

### 1. Download plays

Download the public-domain play texts (Chekhov from Project Gutenberg, Hamlet from Folger TEI):

```bash
python scripts/download_plays.py cherry_orchard
python scripts/download_plays.py hamlet
```

### 2. Run analysis (Pass 1)

Build character bibles, scene bibles, and beat-level state archives:

```bash
# Analyze all characters in a play
python scripts/run_analysis.py cherry_orchard

# Analyze specific characters only
python scripts/run_analysis.py cherry_orchard --characters LOPAKHIN RANYEVSKAYA

# Skip beat segmentation if already cached
python scripts/run_analysis.py cherry_orchard --skip-segmentation
```

Outputs are saved to `data/bibles/` and `data/beats/`.

> **Note on cost**: Running the full analysis with Claude Opus on Cherry Orchard cost ~$10, and Hamlet cost ~$15.

### 3. Improvise (Pass 2)

Run an interactive improvisation session with a character:

```bash
# Interactive session — you type partner lines, the character responds
python scripts/run_improvisation.py session \
  --character LOPAKHIN --play cherry_orchard \
  --setting "A bare office in Moscow, late night" \
  --stakes "Lopakhin is about to lose everything he built"

# Cross-play scene — two characters from different plays improvise together
python scripts/run_improvisation.py crossplay \
  --character-a LOPAKHIN --play-a cherry_orchard \
  --character-b HAMLET   --play-b hamlet \
  --setting "A crumbling estate in an unnamed country, dusk" \
  --stakes "Both men have come to say goodbye to something they cannot name"
```

### 4. Evaluate

Run the three-tier evaluation protocol (vanilla LLM vs. bible-augmented vs. full reflection loop):

```bash
python scripts/run_evaluation.py \
  --character LOPAKHIN --play cherry_orchard \
  --num-scenes 5 --judge-runs 3
```

## Project structure

```
uta_model/
├── config.py              # Model selection, paths, pipeline parameters
├── schemas.py             # Pydantic data models (Play, Beat, BeatState, CharacterBible, ...)
├── ingest/                # Script parsing (Gutenberg plain text, Folger TEI-XML)
├── analysis/              # Pass 1: beat segmentation, state extraction, arc smoothing, bible building
├── improv/                # Pass 2: state initialization, generation, reflection loop
├── evaluation/            # Three-tier evaluation with LLM-as-judge
├── scripts/               # CLI entry points
├── data/                  # Downloaded texts, parsed plays, cached beats, built bibles
├── viewer/                # Web frontend for browsing pipeline outputs
├── PLAN.md                # Detailed system design and research notes
└── proposal.md            # Original project proposal and design questions
```

## Troubleshooting

### Clearing the beats cache

If you encounter parsing issues after code changes, you may need to regenerate the cached beat segmentation:

```bash
rm data/beats/cherry_orchard_beats.json
```

Then re-run the analysis.

## License

This project is dedicated to the public domain under [CC0 1.0 Universal](LICENSE).
