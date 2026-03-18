# Uta: A Latent-State Acting System

Uta models theatrical characters as multi-scale latent dynamical systems and generates character-faithful improvised dialogue. It treats dialogue not as text generation, but as the surface emission of hidden dramatic states — objectives, tactics, emotions, knowledge, and relationships — grounded in Stanislavsky-tradition acting theory.

The system works in two passes:

1. **Analysis (Pass 1)**: Parse a play, segment it into beats, extract per-character latent states (desire, affect, tactic, epistemic, social, defense), smooth arcs globally, and build structured character/scene/world bibles.
2. **Improvisation (Pass 2)**: Given a character bible and a novel scene context, initialize a hidden state, generate candidate lines, score them on six axes (voice, tactic, knowledge, relationship, subtext, emotional plausibility), and revise via targeted feedback — now with statistical priors providing graduated dramaturgical feedback.

---

## Release Notes

### Phase B.2 — Factor Graph Implementation (2026-03-18)

Phase B.2 introduces a factor graph model for probabilistic inference over dramatic states. The factor graph operates as **Pass 1.5** — after LLM analysis (Pass 1) produces BeatStates, but before improvisation (Pass 2). See [docs/FACTOR_GRAPH_IMPLEMENTATION.md](docs/FACTOR_GRAPH_IMPLEMENTATION.md) for the full design and [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for the experimental findings that informed it.

**Affect eigendecomposition.** The 5D affect state (valence, arousal, certainty, control, vulnerability) decomposes into 3 independent transition axes capturing 91.6% of beat-to-beat variance: Disempowerment (59.9%), Blissful Ignorance (21.7%), and Burdened Power (10.0%). Arousal is near-IID and functions as a stylistic modulator of utterance features. The factor graph uses a 3+1 architecture: 3 latent transition axes + arousal for emission.

**Semantically-informed Dirichlet smoothing.** Transition matrix smoothing uses tactic embedding distances to weight the Dirichlet prior — semantically similar unseen transitions get more mass than distant ones. This reduces mean row entropy by 21% compared to uniform smoothing and drops the smoother disagreement rate from 94.8% to 6.3% on canonical tactics.

**Desire conditioning validated.** Desire similarity (via semantic embeddings) significantly predicts tactic persistence (r=+0.106, p=0.005). Character identity adds no predictive power beyond desire content. The transition factor uses two-part conditioning: continuous desire-similarity scalar + discrete desire-type clustering (k=7).

**Factor graph modules.** New `factor_graph/` package with parameter learning, inference engine (forward filtering + forward-backward smoothing), and improv integration:
- `factor_graph/learning.py` — estimates all factor potentials from the corpus (zero API cost)
- `factor_graph/inference.py` — forward filter (Pass 2) and forward-backward smoother (Pass 1.5)
- `factor_graph/integration.py` — drop-in replacement for the LLM state updater
- `scripts/run_smoothing.py` — CLI for running Pass 1.5 smoothing

**Smoothing pipeline (Pass 1.5).** Forward-backward inference over LLM-extracted BeatStates produces smoothed posterior distributions. Pass 1 outputs in `data/parsed/` are read-only; smoothed results go to `data/smoothed/`. The LLM smoother (Pass 1, semantic) and factor graph smoother (Pass 1.5, statistical) are complementary layers.

**Emission model.** Weak overall (R²<0.07) but structured: tactic-specific text signatures (PROBE→questions, COMMAND→imperatives, SHAME→"you" language) serve as hard emission constraints. Inference is transition-dominated, not observation-dominated.

**Cross-character coupling.** Status is validated as relational/zero-sum (r=-0.20, p<0.0001) across all three plays, supporting the ψ_social coupling factor.

**Superobjective prior.** ψ_arc provides 6.25% information gain for tactic prediction. Modeled as a soft additive mixture (λ=0.06) that strengthens when transition dynamics are uncertain.

### Phase B — Statistical Learning (2026-03-16)

Phase B introduces statistical learning infrastructure over the analyzed corpus, bridging the symbolic Phase A system and the future probabilistic Phase C. See [docs/STATISTICAL_LEARNING_PHASE_DESIGN.md](docs/STATISTICAL_LEARNING_PHASE_DESIGN.md) for the full design.

**Canonical tactic vocabulary.** The 297 free-text tactic strings extracted during analysis are clustered into 66 canonical categories using sentence-transformer embeddings (local, free). Hand-crafted acting-theory definitions for the top 76 tactics ensure high-quality clusters. 881/1114 beat states (79%) are normalized to canonical IDs; the remaining 179 singleton tactics are flagged for future assignment as the corpus grows. The vocabulary grows incrementally — new plays' tactics are either assigned to existing clusters or flagged as unmapped.

**Incremental bible building.** A `--bibles-only` mode on `run_analysis.py` builds CharacterBibles for remaining characters without re-running segmentation, extraction, or smoothing. CharacterBibles are now built for all 36 characters (≥5 beat states) across three plays.

**Model tiering.** Each pipeline step now has its own model configuration via `MODEL_CONFIGS` in `config.py`. Segmentation and bible building use Sonnet, WorldBible uses Haiku, while extraction and smoothing remain on Opus. The provider-agnostic config structure supports future benchmarking with Gemini or OpenAI models. Estimated cost reduction: ~40% per play.

**Relationship modeling.** Directed pairwise `RelationshipEdge` objects are now populated from existing BeatState social_state data (164 edges across three plays). Per-character `RelationalProfile` aggregates capture default warmth, status claim, variance across partners, and per-partner deviations. Profiles are directed — Hamlet's sardonic distance (warmth=-0.01, status=+0.51) is independent of how others relate to him. All numeric aggregation is zero API cost.

**Statistical priors and dramaturgical feedback.** The improvisation loop now loads a `StatisticalPrior` per character (tactic prior, transition matrix, relational profile). Priors act as a **critic, not a constraint** — the LLM generates freely, and deviations from the character's statistical patterns produce graduated director's-note feedback:
- *Tier 1 (on target)*: encouragement and polish notes
- *Tier 2 (mild deviation)*: names the character's typical pattern and asks the actor to justify the shift
- *Tier 3 (sharp deviation)*: demands strong dramatic justification for breaking established patterns

**Revision traces.** `MIN_REVISION_ROUNDS=1` ensures every line gets at least one feedback round. Every revision round is recorded in a `RevisionTrace` (candidate text, per-axis scores, feedback given), enabling analysis of how feedback changes outputs.

**Test suite.** 184 tests covering imports, schemas, config, vocabulary clustering, bible builder logic, data integrity, relationship modeling, statistical priors, factor graph, and dramaturgical feedback. Run with `python -m pytest tests/ -v`.

---

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

### Download pre-computed data

The analyzed play data (parsed plays, character bibles, factor graph parameters, tactic vocabulary) is hosted separately from the repository. Download and extract it:

```bash
./scripts/download_data.sh
```

This populates the `data/` directory (~15MB uncompressed, ~3MB download). If you prefer to re-generate the data from scratch, see sections 1-2 below.

### Configuration

Create a `.env` file in the project root with your Anthropic API key:

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

Model selection and pipeline parameters can be adjusted in `config.py`. Each pipeline step (segmentation, extraction, smoothing, bible building, generation, scoring, judging) has its own model configuration via `MODEL_CONFIGS`, allowing per-step cost/quality tradeoffs.

## Usage

### 1. Download plays

Skip this if you've already run `./scripts/download_data.sh` — the raw texts are included in the data archive.

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

# Build only missing character bibles from an already-analyzed play (skips steps 1-4)
python scripts/run_analysis.py cherry_orchard --bibles-only

# Build bibles only for characters with at least 5 beat states
python scripts/run_analysis.py hamlet --bibles-only --min-beat-states 5

# Build bibles for specific characters in an already-analyzed play
python scripts/run_analysis.py hamlet --bibles-only --characters CLAUDIUS GERTRUDE HORATIO
```

Outputs are saved to `data/parsed/`, `data/bibles/`, and `data/beats/`.

> **Note on cost**: Running the full analysis on Cherry Orchard costs ~$10, and Hamlet ~$15. The `--bibles-only` mode costs ~$0.18 per character since it reuses existing beat extractions.

### 2b. Run factor graph smoothing (Pass 1.5)

After analysis, run the factor graph to produce smoothed posterior distributions:

```bash
# Learn factor parameters from the corpus (run once, or when new plays are added)
python -m factor_graph.learning --plays cherry_orchard hamlet importance_of_being_earnest

# Run forward-backward smoothing on a play
python scripts/run_smoothing.py cherry_orchard

# Smooth all plays
python scripts/run_smoothing.py --all
```

Outputs are saved to `data/smoothed/`. Pass 1 outputs in `data/parsed/` are never modified.

### 3. Improvise (Pass 2)

Run an interactive improvisation session with a character. Statistical priors (tactic distributions, transition matrices, relational profiles) are automatically loaded when available and provide graduated dramaturgical feedback during revision rounds. All scenes are automatically saved to `data/improv/` as JSON.

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

# Force at least 2 revision rounds per turn (useful for studying feedback dynamics)
python scripts/run_improvisation.py session \
  --character LOPAKHIN --play cherry_orchard \
  --setting "A bare office" --stakes "Everything is at risk" \
  --min-revisions 2

# Use factor graph for state updates instead of LLM state updater (faster, free)
python scripts/run_improvisation.py session \
  --character LOPAKHIN --play cherry_orchard \
  --setting "A bare office" --stakes "Everything is at risk" \
  --factor-graph
```

Each saved scene (`data/improv/{scene_id}.json`) includes:
- **Full turn data** with all revision drafts, per-axis scores, and feedback (including dramaturgical notes from priors)
- **Configuration snapshot**: CLI flags, model configs, pipeline parameters (min/max revisions, score threshold)
- **Character info**: play, whether priors were loaded, top tactic, default warmth/status
- **Transcript**: flat speaker/line list for easy reading

### 4. Build canonical tactic vocabulary (Phase B)

Cluster the free-text tactic strings extracted during analysis into a canonical vocabulary:

```bash
# Build the vocabulary from all parsed plays
python analysis/vocabulary.py build

# Display clusters for human review
python analysis/vocabulary.py show

# Assign canonical tactic IDs to all BeatStates in parsed plays
python analysis/vocabulary.py normalize

# Ingest new tactics from a freshly analyzed play into the existing vocabulary
python analysis/vocabulary.py ingest cherry_orchard
```

The vocabulary is saved to `data/vocab/tactic_vocabulary.json`. The `build` command accepts `--threshold` (default 0.45) to control cluster granularity — lower values produce more clusters, higher values produce fewer.

### 5. Build relationship profiles (Phase B)

Compute directed pairwise relationship edges and per-character relational profiles from existing BeatState data:

```bash
# Build relationship edges and profiles (no API calls for numerics)
python analysis/relationship_builder.py cherry_orchard
python analysis/relationship_builder.py hamlet

# Include LLM-generated summaries for each relationship (~$0.02/pair)
python analysis/relationship_builder.py cherry_orchard --summaries
```

Outputs: relationship edges are added to the parsed play JSON, and relational profiles are saved to `data/vocab/{play_id}_relational_profiles.json`.

### 6. Run tests

```bash
python -m pytest tests/ -v
```

184 tests covering imports, schemas, config, vocabulary, bible builder, data integrity, relationships, and priors.

### 7. Evaluate

Run the three-tier evaluation protocol (vanilla LLM vs. bible-augmented vs. full reflection loop):

```bash
python scripts/run_evaluation.py \
  --character LOPAKHIN --play cherry_orchard \
  --num-scenes 5 --judge-runs 3
```

## Project structure

```
uta_model/
├── config.py              # Per-step model selection, paths, pipeline parameters
├── schemas.py             # Pydantic data models (Play, BeatState, CharacterBible, StatisticalPrior, ...)
├── ingest/                # Script parsing (Gutenberg plain text, Folger TEI-XML)
├── analysis/              # Pass 1: segmentation, extraction, smoothing, bible building
│   ├── vocabulary.py      # Canonical tactic vocabulary — clustering + normalization (Phase B)
│   └── relationship_builder.py  # Pairwise edges + relational profiles (Phase B)
├── improv/                # Pass 2: state initialization, generation, reflection loop
│   └── priors.py          # Statistical priors + graduated dramaturgical feedback (Phase B)
├── factor_graph/          # Pass 1.5: factor graph learning, inference, smoothing
├── evaluation/            # Three-tier evaluation with LLM-as-judge
├── scripts/               # CLI entry points
├── tests/                 # Test suite (184 tests)
├── data/                  # Pre-computed pipeline outputs (download via scripts/download_data.sh)
│   ├── parsed/            # Pass 1 output: full play JSONs with BeatStates
│   ├── factors/           # Pass 1.5a: learned factor parameters
│   ├── smoothed/          # Pass 1.5c: factor-graph-smoothed posteriors
│   ├── vocab/             # Tactic vocabulary, relational profiles
│   ├── bibles/            # Character/scene/world bibles
│   └── improv/            # Saved improvisation scenes
├── docs/                  # Architecture and design documents
│   ├── LATENT_STATE_ARCHITECTURE.md
│   └── STATISTICAL_LEARNING_PHASE_DESIGN.md
├── viewer/                # Web frontend for browsing pipeline outputs
├── PLAN.md                # Detailed system design and research notes
└── proposal.md            # Original project proposal and design questions
```

## Viewer

A local web interface for browsing pipeline outputs — beats, character bibles, scene bibles, and parsed plays — with affect-space visualizations, beat segmentation strips, and epistemic state inspection.

### Prerequisites

- Node.js 18+ and npm

### Setup

```bash
cd viewer
npm install
```

### Workflow

Run this after each analysis pass to copy the latest outputs into the viewer's asset directory and regenerate the play index:

```bash
npm run sync
```

Then start the dev server:

```bash
npm run dev
# → http://localhost:5173
```

**Full example** — after analyzing Cherry Orchard for the first time:

```bash
# 1. Run analysis (from project root)
conda activate uta_model
python scripts/run_analysis.py cherry_orchard --characters LOPAKHIN

# 2. Sync outputs into the viewer (from viewer/)
cd viewer
npm run sync

# 3. Start the viewer
npm run dev
```

Re-run `npm run sync` any time you add a new play or re-run analysis on existing characters. The dev server hot-reloads on code changes; a sync + browser refresh picks up new data.

### Pages

| URL | Contents |
|-----|----------|
| `/` | Play browser — all synced plays |
| `/plays/:playId` | World bible, character grid, scene list, beat segmentation chart |
| `/plays/:playId/characters/:character` | Character bible + Arc / Affect Space / Tactics / Arc-by-Scene tabs |
| `/plays/:playId/scenes/:act/:scene` | Scene bible + beat-by-beat state inspector |

## Troubleshooting

### Clearing the beats cache

If you encounter parsing issues after code changes, you may need to regenerate the cached beat segmentation:

```bash
rm data/beats/cherry_orchard_beats.json
```

Then re-run the analysis.

## License

This project is dedicated to the public domain under [CC0 1.0 Universal](LICENSE).
