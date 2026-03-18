# Statistical Learning Phase (Phase B) — Design Document

This document describes the design of Phase B of the Uta Acting System: the introduction of statistical learning over a growing corpus of analyzed plays. Phase B bridges the current symbolic-only system (Phase A) and the future factor-graph / learned-embedding system (Phase C) described in [LATENT_STATE_ARCHITECTURE.md](LATENT_STATE_ARCHITECTURE.md).

---

## 1. Motivation and Scope

### What Phase A gave us
- Point-estimate BeatStates extracted by single LLM calls
- Free-text tactic and defense labels (297 unique tactics, many synonyms)
- CharacterBibles for 36 characters across 3 plays (Cherry Orchard 12, Hamlet 15, Earnest 9)
- All analysis uses Opus at $10–15/play
- No learned priors, no transition models, no relationship tracking
- Improvisation generates freely with no statistical grounding

### What Phase B adds (implemented)
- **Canonical vocabulary**: 66 canonical tactic clusters from 297 raw strings, with hand-crafted acting-theory definitions for the top 76. 79% of beat states normalized. (`analysis/vocabulary.py`)
- **Model tiering**: per-step model selection via `MODEL_CONFIGS` — segmentation and bible building on Sonnet, WorldBible on Haiku, extraction and smoothing on Opus. ~40% cost reduction per play. (`config.py`)
- **Incremental bible building**: `--bibles-only` mode builds CharacterBibles for remaining characters without re-running steps 1–4. All 25 significant characters now have bibles. (`scripts/run_analysis.py`, `analysis/bible_builder.py`)
- **Relationship modeling**: 164 directed pairwise edges across 3 plays (68+66+30), 40 relational profiles with default warmth/status, variance, and per-partner deviations. Zero API cost. (`analysis/relationship_builder.py`)
- **Dramaturgical feedback loop**: `StatisticalPrior` loaded per character at improv time. Deviations from tactic priors and affect baselines generate graduated director's-note feedback (3 tiers). (`improv/priors.py`)
- **Revision traces**: `MIN_REVISION_ROUNDS=1` ensures every line gets feedback. Full `RevisionTrace` recorded per round. (`improv/improvisation_loop.py`)
- **Test suite**: 108 tests across 7 test files. (`tests/`)

### Not yet implemented (Phase B.2)
- **Ensemble extraction**: multiple model calls per beat → distributional estimates with calibrated uncertainty. Schemas ready (`BeatStateEstimate`), implementation deferred to when new plays are analyzed.

---

## 2. Canonical Tactic Vocabulary

### 2.1 The problem

The current 297 unique tactic strings include morphological variants ("commanding" / "command"), near-synonyms ("plead" / "implore" / "beg"), and genuinely distinct rare tactics. The `tactic_distribution` in CharacterBible counts exact strings, making distributions sparse and incomparable across characters.

### 2.2 Tactic sentence expansion

Raw tactic strings are single words that risk polysemy when embedded. Before clustering, each tactic is expanded to a sentence anchoring it as a theatrical action verb:

```
"deflect" → "deflect — redirect attention away from a threatening topic to avoid confrontation"
"mock"    → "mock — use ridicule or imitation to diminish the other person's position"
"plead"   → "plead — appeal to the other person's compassion or mercy from a position of need"
```

Expansion is done via a single batch Haiku call (~$0.01) for the long tail, with the top 50 tactics hand-reviewed for accuracy.

### 2.3 Clustering methodology (implemented)

1. Two-pass approach: cluster well-defined "seed" tactics (hand-crafted definitions or count ≥ 3), then assign long-tail singletons to nearest seed cluster with a tighter threshold
2. Embed all expanded tactic sentences with `all-MiniLM-L6-v2` (sentence-transformers, local, free)
3. Agglomerative clustering with cosine distance, threshold=0.45 for seeds, 0.7× threshold for long-tail assignment
4. Result: **66 canonical tactics**, 179 unmapped singletons. 881/1114 beat states (79%) normalized
5. Human-in-the-loop review via `vocabulary.py show`

### 2.4 Incremental vocabulary growth

When a new play is analyzed:
1. New tactic strings are expanded to sentences and embedded
2. Each is compared to existing cluster centroids
3. If within threshold → assigned to nearest cluster, added to `members`
4. If outside threshold → added to `unmapped` for review
5. Periodically re-cluster the full vocabulary (every ~5 plays)

### 2.5 Schema

```python
class CanonicalTactic(BaseModel):
    canonical_id: str          # "DEFLECT"
    canonical_verb: str        # "deflect"
    description: str           # acting-theory definition sentence
    members: list[str]         # raw strings mapping here
    category: str = ""         # optional super-category (e.g., "avoidance")

class TacticVocabulary(BaseModel):
    version: int
    tactics: list[CanonicalTactic]
    unmapped: list[str]        # new strings awaiting assignment
```

### 2.6 Integration

- `BeatState` gains optional `canonical_tactic: str` field (populated by normalization pass)
- `CharacterBible.tactic_distribution` keys become canonical IDs
- Same approach applied to defense mechanisms (fewer unique values)

---

## 3. Incremental Bible Building (implemented)

### 3.1 The `--bibles-only` pipeline mode

BeatStates are already extracted for ALL characters in both plays. A `--bibles-only` flag on `run_analysis.py` builds only missing CharacterBibles:

1. Loads the Play object from `data/parsed/{play_id}.json`
2. Skips steps 1–4 (parse, segment, extract, smooth)
3. Identifies characters missing bibles (or uses `--characters` filter)
4. Calls `build_character_bible()` only for missing characters
5. Skips scene/world bible rebuild (already present)
6. Saves updated Play object

### 3.2 Character selection criteria

Not all characters warrant bibles. Threshold: characters with ≥ 5 beat states (`--min-beat-states 5`). This yields 12 for Cherry Orchard, 15 for Hamlet, and 9 for Earnest. All 36 are now built.

### 3.3 Cost

~$0.04/character with Sonnet (per model tiering). For 25 characters: ~$1 total.

---

## 4. Model Tiering Strategy (implemented)

### 4.1 Per-step model configuration

| Step | Current | Recommended | Rationale |
|---|---|---|---|
| Segmentation | Opus | Sonnet | Structured boundary detection; constrained int-array output |
| Extraction | Opus | Opus (or 3×Sonnet ensemble) | Most demanding: 9-dim structured JSON |
| Smoothing | Opus | Opus | Global long-context reasoning over full arc |
| CharacterBible | Opus | Sonnet | Synthesis from structured data |
| SceneBible | Opus | Sonnet | Simpler synthesis |
| WorldBible | Opus | Haiku | Recalling known facts about the play |

### 4.2 Multi-provider configuration

```python
MODEL_CONFIGS = {
    "segmentation": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "extraction":   {"provider": "anthropic", "model": "claude-opus-4-6"},
    "smoothing":    {"provider": "anthropic", "model": "claude-opus-4-6"},
    "bible":        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "world_bible":  {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "generation":   {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "critic":       {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "judge":        {"provider": "anthropic", "model": "claude-opus-4-6"},
}
```

The provider-agnostic config makes it straightforward to swap in Gemini Pro/Flash or GPT-4o/4o-mini per step after benchmarking.

### 4.3 Cost projections

| Step | All-Opus cost | Tiered cost | Savings |
|---|---|---|---|
| Segmentation | $2–3 | $0.40–0.60 | ~80% |
| Extraction | $6–8 | $4–5 (ensemble) or $6–8 (Opus) | 0–40% |
| Smoothing | $1–2 | $1–2 | 0% |
| Bible building | $1–2 | $0.20–0.40 | ~80% |
| **Total per play** | **$10–15** | **$5.60–8.00** | **~40%** |

---

## 5. Ensemble Extraction and Likelihood Calibration

### 5.1 Multi-model/temperature ensemble

Instead of a single Opus extraction per beat, run K extractions across models and/or temperatures. The ensemble strategy:

- **Sweet spot**: 3×Sonnet at temperatures 0.3, 0.6, 0.9
- **Cost**: ~$0.14/beat (vs $0.18 for 1×Opus)
- **Output**: distributional estimates rather than point estimates

### 5.2 Aggregation

**Discrete variables** (tactic, defense): Voting across K runs. The empirical distribution IS the likelihood:
```
P(tactic = "deflect" | text) ≈ count("deflect" across K runs) / K
```

**Continuous variables** (affect 5D, social 2D): Treat K runs as samples. Compute mean vector and per-dimension standard deviation. The std captures calibrated uncertainty.

### 5.3 Calibration protocol

Using existing Opus point estimates as pseudo-ground-truth:
1. Sample ~50 beats per play
2. Run the ensemble on each
3. Measure: agreement rate (discrete), RMSE (continuous)
4. Check whether ensemble confidence correlates with Opus agreement
5. When the ensemble disagrees, that IS the uncertainty signal

### 5.4 Schema

```python
class BeatStateEstimate(BaseModel):
    beat_id: str
    character: str
    estimates: list[BeatState]          # K individual estimates
    model_ids: list[str]                # which model produced each
    temperatures: list[float]           # temperature for each
    tactic_posterior: dict[str, float]  # canonical_tactic → P
    affect_mean: AffectState            # mean across estimates
    affect_std: dict[str, float]        # per-dimension std
    consensus_confidence: float         # agreement measure
```

### 5.5 Update from experiments

**Update from experiments**: The eigendecomposition of affect transition covariance (EXPERIMENT_LOG.md) shows that the 5D affect space reduces to 3 independent axes (Disempowerment, Blissful Ignorance, Burdened Power) capturing 91.6% of variance. Ensemble calibration should focus on these rotated axes rather than raw dimensions. Arousal is near-IID and can be estimated from text features rather than calibrated via ensemble. The continuous affect aggregation (mean vector, per-dimension std) should be computed in the rotated basis for better conditioning.

### 5.6 Bridge to the factor graph

Ensemble estimates provide the observation likelihoods needed by the factor graph described in LATENT_STATE_ARCHITECTURE.md §9. The `tactic_posterior` becomes the emission factor, the `affect_mean`/`affect_std` become the Gaussian observation model, and `consensus_confidence` modulates the weight of each observation in belief propagation.

---

## 6. Relationship Modeling

### 6.1 Populating RelationshipEdges from existing BeatStates (implemented)

For each pair of characters co-occurring in ≥3 beats, aggregate the `social_state` from each character's BeatState:
- `temperature_by_beat`: character A's `warmth` toward B at each beat
- `power_by_beat`: character A's `status` relative to B at each beat
- `summary`: optional Sonnet call reading the time series (`--summaries` flag, ~$0.02/pair)

Result: **68 directed edges** for Cherry Orchard, **66 for Hamlet**, **30 for Earnest**. Cost for numeric aggregation: $0.

### 6.2 RelationalProfile: character-level social tendencies (implemented)

Per character, aggregate social_state across ALL partners. **40 profiles** built across 3 plays:

```python
class RelationalProfile(BaseModel):
    character: str
    play_id: str
    default_status_claim: float      # mean status across all interactions
    default_warmth: float            # mean warmth across all interactions
    status_variance: float           # how much status varies by partner
    warmth_variance: float           # how much warmth varies by partner
    partner_deviations: dict[str, dict[str, float]]  # partner → {status_delta, warmth_delta}
```

A character with low `warmth_variance` treats everyone similarly (Hamlet's sardonic edge is nearly universal). A character with high `warmth_variance` is context-dependent (Lopakhin is warm to Ranyevskaya but distant with Varya).

### 6.3 Cross-play initialization via directed profile pooling

RelationalProfiles are **directed** — character A's profile describes how A relates to others, independently of how others relate to A. When A meets stranger B in cross-play:
- A's initial social state uses A's `default_status_claim` and `default_warmth`
- B independently uses B's profile
- These are asymmetric: Lopakhin may default to low-status deference while Hamlet defaults to sardonic distance

If B's superobjective/role resembles a known partner of A (by embedding similarity), A's profile for that partner provides a more specific initialization.

---

## 7. Integration with Improvisation

### 7.1 Design principle: priors as dramaturgical critic, not generative constraint (implemented)

The statistical priors do NOT go into generation prompts. The LLM generates freely. The priors are used *after* generation to evaluate consistency with the character's established patterns and to produce **director's-note style feedback** when deviations occur. This mirrors how a real director works: the actor makes a choice, then the director responds.

Implementation: `improv/priors.py` provides `generate_dramaturgical_feedback()` which is called by the improvisation loop after each scoring round, appending feedback to the scorer's notes.

### 7.2 StatisticalPrior object (implemented)

```python
class StatisticalPrior(BaseModel):
    tactic_vocabulary: TacticVocabulary
    character_tactic_prior: dict[str, float]      # canonical_id → P(tactic | character)
    tactic_transition_matrix: dict[str, dict[str, float]]  # P(next | current)
    relational_profile: RelationalProfile
```

Loaded once per character at session start via `load_prior_for_character()`. Auto-loaded by `run_improvisation.py` when vocabulary and profiles exist on disk.

### 7.3 Integration points (implemented)

**State initialization**: The LLM infers the full BeatState freely. The relational profile provides the social baseline only (warmth/status for unknown partners).

**Generation**: Unchanged from Phase A. No probability language in the prompt. The canonical vocabulary is used only for internal bookkeeping (mapping the generated tactic to a canonical ID).

**Scoring + feedback** (`improv/priors.py`): After generation, two deviation analyses run:
- **Tactic deviation**: maps generated tactic to canonical ID, computes P(tactic | character) and P(tactic | previous_tactic), assigns tier 1/2/3
- **Affect deviation**: computes z-scores for valence, arousal, vulnerability against character's historical range

**Experimental finding**: Desire content is a stronger predictor of tactic choice than character identity (18.2% vs 13.6% accuracy). The `character_tactic_prior` P(tactic | character) should be supplemented or replaced with a desire-conditioned prior. Desire similarity between consecutive beats modulates tactic persistence (8.1% → 15.8%), which should inform the graduated feedback thresholds.

**State update**: The previous tactic is tracked across turns for transition analysis. Beat-shift detection via low-probability transitions is supported in the deviation analysis.

### 7.4 Graduated dramaturgical feedback (implemented)

Feedback is templated with graduated intensity based on deviation magnitude. Templates name the character's specific characteristic behavior and scale from encouragement to strong pushback.

**Tier 1 — On target** (P(tactic | character) ≥ 5%, or MIN_REVISION_ROUNDS forcing a pass):
> "Good instinct — the deflection feels characteristic of {character}. See if you can let the subtext breathe a little more through the rhythm of the line."

**Tier 2 — Mild deviation** (2% ≤ P < 5%, or low transition probability):
> "{character} typically operates through {top_tactic} in moments like this — the shift to {generated_tactic} is noticeable. If this is a genuine departure, let the audience feel the cost of that shift."

**Tier 3 — Sharp deviation** (P < 2%):
> "This is a significant break from {character}'s established pattern. {character} has used {top_tactic} in {pct}% of comparable moments — jumping to {generated_tactic} needs a very strong provocation. What in this specific moment forces {character} out of their comfort zone? If you can't point to it, return to {top_tactic}."

**Affect deviation tiers** (based on z-scores against historical range):
- Tier 2 (z > 1.5): names the deviant dimension and typical range
- Tier 3 (z > 2.5): flags the shift as needing to be earned beat by beat

Templates are populated from the `StatisticalPrior` and generated by `improv/priors.py:generate_dramaturgical_feedback()`.

---

## 8. Revision Loop Enhancements (implemented)

### 8.1 Minimum revision rounds

`MIN_REVISION_ROUNDS = 1` (configurable via `config.py` or `--min-revisions` CLI flag). Every line gets at least one round of feedback, even when initial scores are above threshold. This ensures the dramaturgical feedback system always engages.

### 8.2 Revision trace recording

Every revision round is captured in `ImprovTurn.revision_trace`:

```python
class RevisionTrace(BaseModel):
    round: int
    candidate_text: str
    scores: dict[str, float]      # axis → score
    feedback: list[str]           # feedback notes for this round
```

`ImprovTurn` gains a `revision_trace: list[RevisionTrace]` field.

### 8.3 Analysis of revision value

With MIN_REVISION_ROUNDS=1+, the revision traces reveal:
- Which axes improve most from revision (expected: subtext, tactic fidelity)
- Which axes plateau or regress (possible: voice fidelity — over-revision → generic)
- The marginal value of each additional revision round
- How graduated feedback intensity correlates with revision effectiveness

---

## 9. Implementation Status

### Phase B.1 — Complete
1. Canonical tactic vocabulary — `analysis/vocabulary.py`, 66 clusters, 79% coverage
2. Incremental bible building — `--bibles-only` mode, 36 characters built across 3 plays
3. MIN_REVISION_ROUNDS + RevisionTrace — default 1, full trace recording
4. Model tiering — `MODEL_CONFIGS` with per-step `{provider, model}`, ~40% savings

### Phase B.3 — Complete
5. Relationship modeling — `analysis/relationship_builder.py`, 164 directed edges, 40 profiles
6. StatisticalPrior integration — `improv/priors.py`, graduated dramaturgical feedback in revision loop
7. Test suite — 108 tests across `tests/`

### Phase B.2 — Not yet implemented
8. Ensemble extraction mode (`analysis/ensemble.py`) — deferred until new plays are analyzed
9. Calibration protocol (`analysis/calibration.py`) — depends on ensemble extraction

---

## Appendix A: Cost Model

| Component | Model | Per-unit cost | Units per play |
|---|---|---|---|
| Beat segmentation | Sonnet | ~$0.01 | ~10–20 scenes |
| BeatState extraction | Opus or 3×Sonnet | ~$0.14–0.18 | ~40–60 beats × characters |
| Arc smoothing | Opus | ~$0.10 | 2 passes × characters |
| CharacterBible | Sonnet | ~$0.04 | 1 per character |
| SceneBible | Sonnet | ~$0.02 | 1 per scene |
| WorldBible | Haiku | ~$0.002 | 1 per play |
| Relationship summaries | Sonnet | ~$0.02 | ~50 pairs per play |

## Appendix B: Migration from Phase A Schemas

All schema changes are additive — existing fields are preserved, new fields have defaults:
- `BeatState.canonical_tactic: Optional[str] = None` — populated by normalization pass
- `BeatStateEstimate` — new model wrapping multiple `BeatState` estimates
- `RelationalProfile` — new model, computed from existing `SocialState` data
- `RevisionTrace` — new model
- `ImprovTurn.revision_trace: list[RevisionTrace] = []` — new field with empty default
- `CanonicalTactic`, `TacticVocabulary` — new models, persisted to `data/vocab/`

No existing data files need to be regenerated. Canonical tactic assignment and relationship profiling are computed from existing BeatStates.
