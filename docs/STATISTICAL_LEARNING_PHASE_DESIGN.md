# Statistical Learning Phase (Phase B) — Design Document

This document describes the design of Phase B of the UTA Acting System: the introduction of statistical learning over a growing corpus of analyzed plays. Phase B bridges the current symbolic-only system (Phase A) and the future factor-graph / learned-embedding system (Phase C) described in [LATENT_STATE_ARCHITECTURE.md](LATENT_STATE_ARCHITECTURE.md).

---

## 1. Motivation and Scope

### What Phase A gives us
- Point-estimate BeatStates extracted by single LLM calls
- Free-text tactic and defense labels (297 unique tactics, many synonyms)
- CharacterBibles for 2 of 58 characters across 2 plays
- All analysis uses Opus at $10–15/play
- No learned priors, no transition models, no relationship tracking
- Improvisation generates freely with no statistical grounding

### What Phase B adds
- **Canonical vocabulary**: clustered, deduplicated tactic/defense taxonomy with acting-theory definitions
- **Model tiering**: per-step model selection (Opus only where needed, Sonnet/Haiku elsewhere)
- **Ensemble extraction**: multiple model calls per beat → distributional estimates with calibrated uncertainty
- **Relationship modeling**: pairwise edges populated from existing data + character-level relational profiles
- **Dramaturgical feedback loop**: statistical priors used as a critic (not a constraint) — deviations from character patterns generate graduated director's-note feedback during improvisation
- **Revision trace recording**: every revision round captured for analysis of feedback dynamics

### Prerequisites
- 2 plays fully analyzed (Cherry Orchard, Hamlet) with BeatStates for all characters
- Incremental bible building for remaining characters (~$4.50)
- `sentence-transformers` for local embeddings

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

### 2.3 Clustering methodology

1. Embed all expanded tactic sentences with `all-MiniLM-L6-v2` (sentence-transformers, local, free)
2. Agglomerative clustering with cosine distance, threshold-based (not fixed k)
3. Starting granularity: ~30–40 canonical tactics
4. Human-in-the-loop review: export clusters, allow manual splits/merges

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

## 3. Incremental Bible Building

### 3.1 The `--bibles-only` pipeline mode

BeatStates are already extracted for ALL characters in both plays. Only CharacterBible synthesis is missing for ~56 of 58 characters. A new `--bibles-only` flag on `run_analysis.py`:

1. Loads the Play object from `data/parsed/{play_id}.json`
2. Skips steps 1–4 (parse, segment, extract, smooth)
3. Identifies characters missing bibles (or uses `--characters` filter)
4. Calls `build_character_bible()` only for missing characters
5. Skips scene/world bible rebuild (already present)
6. Saves updated Play object

### 3.2 Character selection criteria

Not all characters warrant bibles. Threshold: characters with ≥ 5 beat states. This yields ~12 for Cherry Orchard and ~15 for Hamlet.

### 3.3 Cost

~$0.18/character (one Sonnet call for bible synthesis). For ~25 characters: ~$4.50 total.

---

## 4. Model Tiering Strategy

### 4.1 Per-step model recommendations

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

### 5.5 Bridge to the factor graph

Ensemble estimates provide the observation likelihoods needed by the factor graph described in LATENT_STATE_ARCHITECTURE.md §9. The `tactic_posterior` becomes the emission factor, the `affect_mean`/`affect_std` become the Gaussian observation model, and `consensus_confidence` modulates the weight of each observation in belief propagation.

---

## 6. Relationship Modeling

### 6.1 Populating RelationshipEdges from existing BeatStates

For each pair of characters co-occurring in beats, aggregate the `social_state` from each character's BeatState:
- `temperature_by_beat`: character A's `warmth` toward B at each beat
- `power_by_beat`: character A's `status` relative to B at each beat
- `summary`: one Sonnet call reading the time series (~$0.02/pair)

Cost for numeric aggregation: $0. Cost for summaries: ~$1 for ~50 significant pairs.

### 6.2 RelationalProfile: character-level social tendencies

Per character, aggregate social_state across ALL partners:

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

### 7.1 Design principle: priors as dramaturgical critic, not generative constraint

The statistical priors do NOT go into generation prompts. The LLM generates freely. The priors are used *after* generation to evaluate consistency with the character's established patterns and to produce **director's-note style feedback** when deviations occur. This mirrors how a real director works: the actor makes a choice, then the director responds.

### 7.2 StatisticalPrior object

```python
class StatisticalPrior(BaseModel):
    tactic_vocabulary: TacticVocabulary
    character_tactic_prior: dict[str, float]      # canonical_id → P(tactic | character)
    tactic_transition_matrix: dict[str, dict[str, float]]  # P(next | current)
    relational_profile: RelationalProfile
```

Loaded once per character at the start of an improv session.

### 7.3 Integration points

**State initialization**: The LLM infers the full BeatState freely. The relational profile provides the social baseline only (warmth/status for unknown partners).

**Generation**: Unchanged from Phase A. No probability language in the prompt. The canonical vocabulary is used only for internal bookkeeping (mapping the generated tactic to a canonical ID).

**Scoring + feedback**: After generation, the scorer maps the tactic to a canonical ID and checks:
- Is this tactic significantly unlikely for this character?
- Is this tactic transition improbable given the previous tactic?
- Does the affect state deviate from the character's typical range?

**State update**: Transition priors inform beat-shift detection. A low-probability transition (bottom 10%) signals a beat boundary → flag `beat_shifted = True`. The state updater itself uses the LLM freely; the prior is diagnostic.

### 7.4 Graduated dramaturgical feedback

Feedback is templated with graduated intensity based on deviation magnitude. Templates name the character's specific characteristic behavior and scale from encouragement to strong pushback.

**Tier 1 — On target** (deviation within normal range, or MIN_REVISION_ROUNDS forcing a pass):
> "Good instinct — the deflection feels characteristic of {character}. See if you can let the underlying vulnerability bleed through just slightly more in the rhythm of the line."

**Tier 2 — Mild deviation** (outside typical range but not extreme):
> "{character} typically operates through {top_tactic} in moments like this — the shift to {generated_tactic} is noticeable. If this is a genuine departure, let the audience feel the cost of that shift."

**Tier 3 — Sharp deviation** (bottom 5% probability):
> "This is a significant break from {character}'s established pattern. {character} has used {top_tactic} in {pct}% of comparable moments — jumping to {generated_tactic} needs a very strong provocation. What in this specific moment forces {character} out of their comfort zone? If you can't point to it, return to {top_tactic}."

Templates are populated from the `StatisticalPrior` and generated by `improv/priors.py`.

---

## 8. Revision Loop Enhancements

### 8.1 Minimum revision rounds

`MIN_REVISION_ROUNDS = 1` (configurable). Every line gets at least one round of feedback, even when initial scores are above threshold. This ensures the dramaturgical feedback system always engages.

### 8.2 Revision trace recording

Every revision round is captured:

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

## 9. Implementation Sequence

### Phase B.1 — Immediate, low cost, high value
1. Canonical tactic vocabulary (pure computation + local embeddings)
2. Incremental bible building for remaining characters (~$4.50)
3. MIN_REVISION_ROUNDS + RevisionTrace

### Phase B.2 — Short-term, moderate effort
4. Model tiering config refactor
5. Ensemble extraction mode + calibration protocol

### Phase B.3 — Medium-term
6. Relationship modeling (pairwise edges + relational profiles)
7. Full StatisticalPrior integration with improvisation (graduated feedback)

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
