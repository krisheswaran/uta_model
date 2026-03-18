# Factor Graph Implementation Plan

This document describes how to implement the factor graph model specified in [LATENT_STATE_ARCHITECTURE.md](LATENT_STATE_ARCHITECTURE.md) §9.2, informed by the experimental findings in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). It covers learning the model parameters from existing data and running inference over a sequence of utterances at inference time.

---

## 1. Architecture Overview

### 1.1 What we're building

A factor graph that models the joint distribution over a character's latent dramatic state across a sequence of beats. It operates as **Pass 1.5** in the pipeline — after the existing LLM analysis (Pass 1) produces BeatStates, bibles, and relationship edges, but before improvisation (Pass 2).

```
Pass 1 (LLM analysis)  →  Pass 1.5 (factor graph)  →  Pass 2 (improv)
   parse, segment,           learn parameters,            generate lines
   extract, smooth,          build graph,                 with factor graph
   build bibles              run forward-backward         driving state updates
                             over extracted BeatStates
```

**Pass 1 is untouched.** It remains the pre-processing step that produces the raw data: BeatStates (noisy LLM point estimates), CharacterBibles, SceneBibles, WorldBibles, RelationshipEdges. The LLM smoother (Step 4) stays in Pass 1 — it catches semantic issues (knowledge leaks, unmotivated contradictions) that a statistical model cannot.

**Pass 1.5 treats Pass 1 outputs as observations.** The factor graph takes the LLM-extracted BeatStates as noisy measurements of the true latent state, learns factor potentials (transition matrices, covariance kernels, emission parameters) from the corpus, and runs forward-backward inference to produce smoothed posterior distributions. This is a new computational layer — no LLM calls, pure numpy/scipy.

**Pass 2 uses factor graph posteriors.** During improvisation, the factor graph runs forward inference to update the character's latent state beat-by-beat, replacing the LLM state updater (`state_updater.py`).

### 1.2 Variable nodes

From the experimental findings, the latent state decomposes as:

```
Z_c(t) = { D(t), T(t), A_trans(t), A_emit(t), S(t), K(t) }

D(t)        desire type         discrete, k=7 clusters (from desire embedding k-means)
T(t)        tactic              discrete, 66 canonical tactics
A_trans(t)  transition affect   continuous, 3D (Disempowerment, Blissful Ignorance, Burdened Power)
A_emit(t)   emissive affect     continuous, scalar (Arousal)
S(t)        social state        continuous, 2D (status, warmth)
K(t)        epistemic state     structured (sets of propositions — not modeled probabilistically in v1)
```

Defense/contradiction (Δ) is dropped from the factor graph v1 — it's a derivative of the other states and doesn't have independent transition dynamics.

### 1.3 Factor nodes

```
ψ_T     tactic transition         P(T(t) | T(t-1), D(t), desire_similarity)
ψ_A     affect transition         P(A_trans(t) | A_trans(t-1))
ψ_D     desire transition         P(D(t) | D(t-1))
ψ_S     social prior              P(S(t) | character, partner)
ψ_arc   superobjective prior      P(T(t) | superobjective embedding)
ψ_emit  emission likelihood       P(text_features(t) | T(t), A_emit(t))
ψ_social cross-character coupling P(S_c1(t), S_c2(t))
```

### 1.4 What already exists in the codebase

| Component | Status | Location |
|---|---|---|
| Tactic vocabulary (66 clusters) | Production-ready | `analysis/vocabulary.py`, `data/vocab/tactic_vocabulary.json` |
| Tactic transition matrix P(T\|T_{t-1}) | Computed on-demand | `improv/priors.py` |
| Relational profiles (status/warmth priors) | Production-ready | `analysis/relationship_builder.py`, `data/vocab/*_relational_profiles.json` |
| Affect transition covariance | Computed in experiments | `scripts/experiments/affect_eigendecomposition.py` |
| Desire embeddings + k=7 clusters | Computed in experiments | `scripts/experiments/desire_conditioning.py` |
| Utterance feature extraction (9 features) | Computed in experiments | `scripts/experiments/emission_model.py` |
| Superobjective embeddings | Computed in experiments | `scripts/experiments/superobjective_predictiveness.py` |
| Belief propagation solver | **Not implemented** | — |
| Factor wrappers (message-passing interface) | **Not implemented** | — |

### 1.5 What needs to be built

1. **Parameter learning** (Pass 1.5a) — estimate all factor potentials from the corpus of Pass 1 outputs
2. **Factor graph engine** (Pass 1.5b) — variable nodes, factor nodes, message passing
3. **Smoothing** (Pass 1.5c) — forward-backward over extracted BeatStates, producing refined posteriors
4. **Integration** (Pass 2) — forward inference during improv, replacing `state_updater.py`

---

## 2. Parameter Learning (from corpus)

All learning runs over the 3 parsed plays in `data/parsed/`. No LLM calls required — this is pure computation over existing BeatStates.

### 2.1 ψ_T: Tactic transition factor

**What to learn**: P(T(t) | T(t-1), D(t), desire_similarity)

**Method**:

1. **Base transition matrix** — pool across all characters and plays:
   - For each character's sequence of beats within a scene, extract (canonical_tactic_{t-1}, canonical_tactic_t) bigrams
   - Count matrix C[i,j] = number of times tactic i → tactic j observed
   - Apply **semantically-informed Dirichlet smoothing** (validated in EXPERIMENT_LOG.md):
     ```
     α_ij = α_base · scale_i · exp(-D[i,j] / τ)
     P[i,j] = (C[i,j] + α_ij) / Σ_j(C[i,j] + α_ij)
     ```
     where D[i,j] is cosine distance between tactic embeddings (all-MiniLM-L6-v2 on full description strings), τ=0.7, and scale_i is proportional to observed row entropy (hub/terminal distinction). This concentrates smoothing mass on semantically plausible unseen transitions while preserving observed dramatic co-occurrence patterns including non-local jumps (e.g., MOCK→PLEAD)

2. **Desire-type conditioning** — from desire embedding clusters (k=7):
   - Embed all desire_state strings with all-MiniLM-L6-v2
   - Run k-means (k=7) on desire embeddings; save cluster centroids
   - For each desire cluster d, compute a separate transition matrix P_d[i,j]
   - At inference time: classify the current desire into a cluster, use the corresponding matrix

3. **Desire-similarity persistence modulation** — continuous scalar:
   - Compute cosine similarity between desire embeddings at t-1 and t
   - Multiply the self-transition probability P[i,i] by a modulation factor: `mod = 1 + β * (similarity - 0.5)` where β is learned from data (experiments show persistence ~doubles from low to high similarity, so β ≈ 1.0)
   - Renormalize the row

**Output artifacts** (save to `data/factors/`):
- `tactic_transition_base.json` — 66×66 smoothed transition matrix
- `tactic_transition_by_desire.json` — 7 × 66×66 matrices (one per desire cluster)
- `desire_cluster_centroids.npy` — 7×384 matrix of cluster centroids
- `persistence_modulation_beta.json` — learned β coefficient

**Open question 1**: Should the base transition matrix be pooled across all plays (more data, less specific) or estimated per-play (more specific, sparser)? The experiment log recommends pooling at n=3, but we could also try leave-one-play-out cross-validation to check whether play-specific matrices outperform pooled on held-out data.

> **ANSWER**: We have too little data to not pool the base transition matrix across plays. It's worthwhile running the leave-one-play-out cross-validation experiment once we've built our parameter learning pipeline.

### 2.2 ψ_A: Affect transition factor

**What to learn**: P(A_trans(t) | A_trans(t-1)) as a Gaussian transition kernel

**Method**:

1. **Compute eigenvectors** from the affect transition covariance:
   - For each character's consecutive beats in a scene, compute deltas: ΔA = A(t) - A(t-1) for all 5 raw affect dimensions
   - Compute 5×5 covariance matrix Σ of the deltas
   - Eigendecompose: Σ = V Λ V^T
   - Take top 3 eigenvectors (columns of V) as the rotation matrix R (3×5)

2. **Project all affect vectors** into eigenspace:
   - A_trans(t) = R · [valence, arousal, certainty, control, vulnerability]^T → 3D vector

3. **Estimate transition kernel** in eigenspace:
   - Compute deltas in eigenspace: ΔA_trans = A_trans(t) - A_trans(t-1)
   - The transition kernel is: A_trans(t) ~ N(A_trans(t-1), Σ_trans)
   - Σ_trans is diagonal in eigenspace (by construction) — just 3 variance parameters

**Output artifacts**:
- `affect_eigenvectors.npy` — 3×5 rotation matrix R
- `affect_eigenvalues.npy` — 3 eigenvalues
- `affect_transition_variance.npy` — 3 diagonal variances for the transition kernel

**Open question 2**: Should the transition kernel be strictly Gaussian, or should we use a heavier-tailed distribution (e.g., Student-t) to accommodate occasional large affect jumps (revelations, betrayals)? A Gaussian penalizes large jumps quadratically, which might over-constrain dramatic moments. We could check the empirical distribution of ΔA_trans — if it has heavier tails than Gaussian, Student-t would be more appropriate.

> **ANSWER**: My thought is a heavier-tail, but it's worth checking the empirical distribution to verify. If inconclusive, let's go with heavier-tail, especially given how small our corpus is, we don't want to over-constrain things.

### 2.3 A_emit: Arousal estimation from text

**What to learn**: P(A_emit(t) | text_features(t)) — a simple regression

**Method**:
- Arousal is near-IID (lag-1 r=+0.036), so no transition model needed
- Train a Ridge regression: arousal ~ word_count + question_density + exclamation_density + imperative_density + mean_sentence_length + lexical_diversity + first_person_rate + second_person_rate + sentiment_polarity
- This gives a point estimate + residual variance for the emission factor

**Output artifacts**:
- `arousal_regressor.pkl` — fitted Ridge model
- `arousal_residual_variance.json` — σ² of the residuals

**Open question 3**: Should arousal be estimated purely from text features (as proposed), or should we also allow a weak character-level prior? The variance decomposition shows only 24.5% of arousal variance is between-character (ICC=0.214), but for characters like Hamlet (consistently high arousal) that prior might help. A simple approach: arousal ~ text_features + character_mean_arousal, where character_mean_arousal is a constant from the CharacterBible.

> **ANSWER**: I like the idea of computing a character mean arousal.

### 2.4 ψ_D: Desire transition factor

**What to learn**: P(D(t) | D(t-1)) — transition matrix over desire clusters

**Method**:
1. Embed all desire_state strings, assign to k=7 clusters (same centroids as §2.1)
2. Extract desire-cluster bigrams per character per scene
3. Smooth and normalize to get a 7×7 transition matrix

Desire transitions tend to be sticky within a scene (same objective) with occasional shifts at dramatic turning points. The transition matrix should reflect this — high diagonal, with off-diagonal entries concentrated on specific shift patterns.

**Output artifacts**:
- `desire_transition_matrix.json` — 7×7 smoothed matrix
- Reuses `desire_cluster_centroids.npy` from §2.1

### 2.5 ψ_S: Social state prior

**What to learn**: P(S(t) | character, partner) — Gaussian prior on (status, warmth)

**Method**:
- Already computed in RelationalProfiles: `default_status_claim`, `default_warmth`, `status_variance`, `warmth_variance`, `partner_deviations`
- For character c addressing partner p:
  - μ_status = default_status_claim + partner_deviations[p].status_delta
  - μ_warmth = default_warmth + partner_deviations[p].warmth_delta
  - σ_status = sqrt(status_variance)
  - σ_warmth = sqrt(warmth_variance)

This is not a transition model (social state doesn't have strong temporal dynamics within a scene) — it's a per-beat prior conditioned on who is in the scene.

**Output artifacts**: Reuses existing `data/vocab/*_relational_profiles.json`

### 2.6 ψ_arc: Superobjective prior

**What to learn**: P(T(t) | superobjective embedding) — soft bias on tactic distribution

**Method**:
1. Embed all superobjectives with all-MiniLM-L6-v2
2. For each superobjective cluster (from experiment: natural clusters emerge), compute the empirical tactic distribution
3. At inference time: embed the character's superobjective, find nearest cluster, blend the cluster's tactic prior with the base transition matrix

The blending weight should be small — superobjective provides only 6.25% information gain, so it's a gentle nudge, not a strong constraint.

**Output artifacts**:
- `superobjective_tactic_prior.json` — mapping from SO cluster → tactic distribution
- `superobjective_cluster_centroids.npy` — cluster centroids

**Open question 4**: How should the superobjective prior blend with the transition prior? Options:
- (a) Multiplicative: P_final(T) ∝ P_transition(T) × P_arc(T)^λ, where λ controls influence
- (b) Additive mixture: P_final(T) = (1-λ) P_transition(T) + λ P_arc(T)
- (c) Only as initialization: use the SO prior for the first beat's tactic distribution, then let transitions take over

> **ANSWER**: Help me understand the tradeoffs here to provide a more informed decision.

**Tradeoff analysis**:

- **(a) Multiplicative** acts as a persistent filter at every beat — the SO reshapes the tactic distribution always, even when the transition model is confident. Risk: suppresses valid dramatic departures at climactic moments. Benefit: ensures long-range coherence.
- **(b) Additive mixture** acts as a fallback — nearly invisible when the transition model is confident (one tactic dominates), but fills the gap when the transition model is uncertain (flat distribution after a desire shift). Self-regulating: kicks in exactly when you'd want a character-level prior. Risk: with λ=0.06 (matching 6.25% IG), the effect is very subtle. Benefit: matches the data — beat-level SO consistency is only 0.516, suggesting the SO should be a gentle background influence.
- **(c) Initialization only** sets the starting point then lets transitions take over. Weakest form — maximum freedom for scene dynamics, but no pull-back toward the SO over long scenes. Given the 0.516 consistency, this might overfit to scene-level noise.

**Recommendation**: (b) additive mixture with small λ. Most self-regulating, strongest when uncertainty is high, invisible when transitions are confident.

### 2.7 ψ_emit: Emission factor

**What to learn**: P(text_features(t) | T(t), A_emit(t))

**Method**:

Two components:

1. **Tactic-specific emission profiles** — from the emission model experiment:
   - For each tactic, compute mean and std of each text feature across all beats where that tactic is active
   - This gives per-tactic Gaussian emission parameters: P(feature_j | tactic_i) ~ N(μ_{ij}, σ_{ij})
   - For distinctive tactic-feature pairs (z > 1.5), treat as hard constraints (high log-likelihood bonus)

2. **Arousal emission** — from the arousal regressor (§2.3):
   - P(text_features | A_emit) is implicit in the regression model
   - At inference time, the predicted arousal from text features IS the A_emit estimate

**Output artifacts**:
- `tactic_emission_profiles.json` — 66 tactics × 9 features, each with (μ, σ)
- Reuses `arousal_regressor.pkl` from §2.3

### 2.8 ψ_social: Cross-character status coupling

**What to learn**: The coupling strength between co-present characters' status claims

**Method**:
- Already validated: Pearson r=-0.20 (p<0.0001) across all plays
- Model as a pairwise factor: P(S_c1.status, S_c2.status) ∝ exp(-γ · (S_c1.status + S_c2.status)²)
- This penalizes both characters claiming high status simultaneously
- γ is estimated from the empirical correlation: γ = |r| / (1 - r²) ≈ 0.21

**Output artifacts**:
- `status_coupling_gamma.json` — single scalar

---

## 3. Inference

The factor graph serves two distinct inference modes: **smoothing** (Pass 1.5) and **forward filtering** (Pass 2). Both use the same engine and learned parameters, but differ in what's observed and what's inferred.

### 3.1 Pass 1.5: Smoothing over extracted BeatStates

**Input**: The full sequence of LLM-extracted BeatStates for a character across a play (from Pass 1), plus the corresponding utterances.

**What it does**: Runs forward-backward to produce smoothed posterior distributions at every beat. The LLM extractions are treated as noisy observations of the true latent state — the factor graph reconciles them with learned transition dynamics and cross-character coupling to produce refined estimates.

**Why it's valuable**: The Pass 1 extractor sees only ~3 beats of context per extraction call. The factor graph smoother propagates information across the entire play: a later revelation can retroactively refine the interpretation of an earlier ambiguous beat. This complements (not replaces) the LLM smoother in Pass 1 Step 4, which catches semantic issues the factor graph cannot.

```
For each character c in play:
    FORWARD PASS (t = 1 to T):
        For each beat t:
            1. PREDICT: compute prior over Z(t) from transition factors
               - P(T(t) | T(t-1), D(t)) from ψ_T
               - P(A_trans(t) | A_trans(t-1)) from ψ_A (Student-t kernel)
               - P(D(t) | D(t-1)) from ψ_D
               - P(S(t)) from ψ_S (relational profile prior)

            2. CONDITION: blend superobjective prior into tactic distribution (ψ_arc)

            3. OBSERVE: update with LLM-extracted BeatState as observation
               - Tactic observation: P(extracted_tactic | true_tactic) — confusion model
               - Affect observation: P(extracted_affect | true_affect) — Gaussian noise model
               - Text features: P(features | T, A_emit) from ψ_emit

            4. COUPLE: apply ψ_social for co-present characters

            5. Store forward message α(t) = P(Z(t) | observations 1:t)

    BACKWARD PASS (t = T down to 1):
        For each beat t:
            - Compute backward message β(t) from β(t+1) through transition factors
            - Smoothed posterior: P(Z(t) | all observations) ∝ α(t) × β(t)

    OUTPUT: Smoothed posteriors for every beat — refined tactic distributions,
            affect trajectories, desire sequences. These are the "true"
            latent states, reconciling noisy LLM extractions with learned dynamics.
```

The smoothed posteriors can be saved alongside the raw BeatStates (as a `PosteriorState` per beat) for use in Pass 2 and for analysis.

### 3.2 Pass 2: Forward filtering during improv

**Input**: A character's CharacterBible, the previous beat's posterior state, and the utterance just delivered.

**What it does**: Runs a single forward step to predict the character's state for the next beat. This replaces `state_updater.py`.

```
For each improv turn:
    1. PREDICT: compute prior over Z(t) from Z(t-1) via transition factors
    2. CONDITION: blend ψ_arc
    3. OBSERVE: extract text features from the delivered utterance
       - Estimate A_emit from text features (arousal regressor)
       - Update tactic posterior using tactic-specific emission profiles
    4. COUPLE: apply ψ_social if other characters present
    5. POSTERIOR: P(Z(t) | history)
       - Convert to BeatState for next generation prompt
```

No backward pass — improv is purely forward. The posterior from each step becomes the prior for the next.

### 3.3 Key difference between the two modes

| | Pass 1.5 (smoothing) | Pass 2 (improv) |
|---|---|---|
| Observations | LLM-extracted BeatStates (noisy) + utterances | Utterance text features only |
| Direction | Forward + backward (uses future context) | Forward only (causal) |
| Purpose | Refine extracted latent states for the full play | Predict next state for generation |
| Cost | One-time computation per play | Per-beat during improv |
| LLM calls | Zero | Zero (replaces LLM state updater) |

### 3.4 Handling mixed discrete/continuous state

The state has both discrete (T, D) and continuous (A_trans, A_emit, S) components.

**Decision: Factored inference** (v1)
- Discrete: exact forward-backward over T×D (66×7 = 462 states — tractable)
- Continuous: Kalman-filter-style updates for A_trans and S, conditioned on the discrete state (with Student-t transition kernel for heavier tails)
- A_emit: point estimate from text features via arousal regressor (no inference needed — near-IID)

Particle-based inference is an option for v2 if factored inference proves too constraining.

### 3.5 Posterior → BeatState conversion (Pass 2)

**Decision**: Use MAP for generation (most probable tactic, mean affect, etc.), but also sample from the posterior periodically to discover multimodal states. If the posterior has multiple modes (e.g., DEFLECT 0.4, CHALLENGE 0.35), the generation prompt could frame this as a choice rather than a directive.

### 3.6 Relationship to the LLM smoother

Pass 1's LLM smoother (Step 4, `analysis/smoother.py`) remains part of Pass 1 — it catches semantic issues (knowledge leaks, unmotivated contradictions) that the statistical factor graph cannot see. Pass 1.5's factor graph smoother is a *complementary* layer that enforces statistical coherence on top of the LLM's semantic corrections. They are not in competition:

- **LLM smoother** (Pass 1, Step 4): "Hamlet shouldn't know about the letter yet in Act 3" — semantic
- **Factor graph smoother** (Pass 1.5): "This affect trajectory has an implausible jump that the transition kernel penalizes" — statistical

Future (v2): the factor graph could also run incremental backward smoothing during improv to produce **retroactive dramaturgical notes** — reinterpreting earlier beats in light of how the scene developed. But this is a refinement, not a v1 requirement.

---

## 4. Code Structure

### 4.1 New modules

```
uta_model/
├── factor_graph/
│   ├── __init__.py
│   ├── variables.py          # Variable node types (Discrete, Gaussian, PointEstimate)
│   ├── factors.py            # Factor node types (Transition, Emission, Prior, Coupling)
│   ├── graph.py              # FactorGraph container — build, connect, validate
│   ├── inference.py          # Forward, forward-backward, message passing
│   ├── learning.py           # Estimate all factor potentials from corpus
│   └── integration.py        # Interface with improv loop (replaces state_updater)
│
├── data/
│   ├── parsed/                # Pass 1 output (READ ONLY for Pass 1.5 — never modified)
│   │   ├── cherry_orchard.json
│   │   ├── hamlet.json
│   │   └── importance_of_being_earnest.json
│   │
│   ├── factors/               # Pass 1.5a output — learned model parameters
│   │   ├── tactic_transition_base.json
│   │   ├── tactic_transition_by_desire.json
│   │   ├── desire_cluster_centroids.npy
│   │   ├── desire_transition_matrix.json
│   │   ├── affect_eigenvectors.npy
│   │   ├── affect_transition_variance.npy
│   │   ├── arousal_regressor.pkl
│   │   ├── tactic_emission_profiles.json
│   │   ├── superobjective_tactic_prior.json
│   │   └── status_coupling_gamma.json
│   │
│   └── smoothed/              # Pass 1.5c output — smoothed posteriors
│       ├── cherry_orchard.json
│       ├── hamlet.json
│       └── importance_of_being_earnest.json
```

**Non-destructive data flow**: Pass 1.5 reads from `data/parsed/` but never writes to it. Learned parameters go to `data/factors/`, smoothed posteriors go to `data/smoothed/`. The raw LLM extractions in `data/parsed/` are preserved as the ground-truth observations that the factor graph was trained on and smoothed against. This separation is essential: it lets the viewer show both layers side-by-side, it makes the smoothing diff computable at any time, and it means re-running Pass 1.5 with different parameters doesn't destroy Pass 1 output.

### 4.2 Key interfaces

```python
# Learning
class FactorLearner:
    def learn_from_corpus(self, play_ids: list[str]) -> FactorParameters:
        """Estimate all factor potentials from parsed plays."""
        ...

# Inference
class FactorGraphInference:
    def __init__(self, params: FactorParameters, character_bible: CharacterBible):
        """Build factor graph for a specific character."""
        ...

    def forward_step(self, prev_state: BeatState, utterance: str,
                     characters_present: list[str]) -> PosteriorState:
        """One-step filtering: predict + observe."""
        ...

    def smooth(self, beat_states: list[BeatState],
               utterances: list[str]) -> list[PosteriorState]:
        """Full forward-backward smoothing over a sequence."""
        ...

# Integration
class PosteriorState:
    tactic_distribution: dict[str, float]    # canonical_tactic → probability
    tactic_map: str                          # most probable tactic
    affect_trans_mean: np.ndarray            # 3D mean in eigenspace
    affect_trans_std: np.ndarray             # 3D std
    arousal: float                           # point estimate from text features
    desire_distribution: dict[int, float]    # cluster_id → probability
    social_mean: np.ndarray                  # 2D (status, warmth)
    social_std: np.ndarray                   # 2D

    def to_beat_state(self, method: str = "map") -> BeatState:
        """Convert posterior to BeatState for generation prompt."""
        ...
```

### 4.3 Entry points

```python
# Pass 1.5a: Learn parameters (run once, or when corpus grows)
python -m factor_graph.learning --plays cherry_orchard hamlet importance_of_being_earnest

# Pass 1.5c: Smooth extracted BeatStates (run after Pass 1 analysis)
python scripts/run_smoothing.py cherry_orchard
python scripts/run_smoothing.py hamlet
python scripts/run_smoothing.py importance_of_being_earnest

# Pass 2: Improv with factor graph (integrated into improv loop)
python scripts/run_improvisation.py --character HAMLET --play hamlet --factor-graph

# In improvisation_loop.py, the --factor-graph flag replaces:
#   updated_state = update_state(prev_state, line, context)
# With:
#   posterior = fg_inference.forward_step(prev_state, line, characters_present)
#   updated_state = posterior.to_beat_state(method="map")
```

---

## 5. Implementation Sequence

### Phase 1: Learn parameters (Pass 1.5a)

Write `factor_graph/learning.py`. Reads the parsed plays (Pass 1 outputs) and computes all factor parameters, saving them to `data/factors/`. Reuses existing experiment code (eigendecomposition, desire clustering, emission profiling) but packages the results as persisted artifacts.

**Deliverable**: `data/factors/` populated with all learned parameters. Can be validated by inspecting the matrices and comparing to experiment log numbers.

### Phase 2: Build inference engine (Pass 1.5b)

Write `factor_graph/variables.py`, `factors.py`, `graph.py`, `inference.py`. Core factor graph implementation — variable nodes, factor nodes, message passing, forward/forward-backward algorithms.

No external library needed — the graph is small (6 variable types, ~8 factor types) and the inference is exact for the discrete components (462 states) and Gaussian for the continuous components. NumPy + SciPy suffice.

**Deliverable**: A standalone inference engine that can take a sequence of LLM-extracted BeatStates + utterances and produce smoothed posterior distributions. Testable independently of the improv loop.

### Phase 3: Smoothing pipeline (Pass 1.5c)

Write a `scripts/run_smoothing.py` entry point that runs forward-backward over the extracted BeatStates for all characters in a play. This is the full Pass 1.5 pipeline: load Pass 1 outputs → load learned parameters → smooth → save posterior states.

**Deliverable**: `data/smoothed/{play_id}.json` or equivalent, containing `PosteriorState` per (beat, character). Can be compared to the raw LLM extractions to see where the factor graph disagrees.

### Phase 4: Integration with improv (Pass 2)

Write `factor_graph/integration.py`. Wire the forward inference step into `improvisation_loop.py` as an alternative to `state_updater.py`. Add a `--factor-graph` flag to `scripts/run_improvisation.py`.

**Deliverable**: The improv loop can run with either LLM-based or factor-graph-based state updates. Side-by-side comparison possible on the same scenes.

### Future: Viewer integration (Pass 1.5 → viewer/)

The `viewer/` app (Redwood SDK, Vite + React) currently consumes Pass 1 output — the parsed play JSON with raw LLM-extracted BeatStates. Once Pass 1.5 produces smoothed posteriors, the viewer should surface both layers and the diff between them.

**What the viewer should show:**

1. **Side-by-side BeatState view**: For each (beat, character), show the raw LLM extraction alongside the smoothed posterior. Where they agree, display normally. Where they disagree, highlight the diff — this is where the factor graph's learned dynamics override the LLM's point estimate.

2. **Posterior distributions, not just point estimates**: The smoothed output includes tactic *distributions* (e.g., DEFLECT 0.6, MOCK 0.2, CHALLENGE 0.1), not just a single tactic. The viewer should render these — a small bar chart or probability list per beat. This surfaces the model's uncertainty and reveals beats where the character's state is genuinely ambiguous (multiple high-probability tactics), which are often the most dramaturgically interesting moments.

3. **Affect trajectories in eigenspace**: The viewer currently shows raw affect dimensions (valence, arousal, certainty, control, vulnerability) over time. It should also show the rotated eigenspace axes (Disempowerment, Blissful Ignorance, Burdened Power) as time series, with the smoothed trajectories overlaid on the raw LLM trajectories. Divergences reveal where the factor graph's transition dynamics pulled the affect trajectory toward a more plausible arc.

4. **Diff summary per character**: A high-level report of what the factor graph changed — how many beats had tactic corrections, the mean affect shift magnitude, which scenes had the most corrections. This answers "what does smoothing materially change?" at a glance.

5. **Factor attribution**: When the smoothed posterior diverges from the raw extraction, *which factor* drove the change? Was it the transition prior (this tactic is unlikely given the previous one), the cross-character coupling (the other character's status contradicts this one's), or the emission model (the text features don't match this tactic's profile)? This level of explainability is a unique advantage of the factor graph over the LLM smoother, which gives corrections without mechanistic attribution.

**Data format**: The smoothed output (`data/smoothed/{play_id}.json`) should include both the `PosteriorState` per (beat, character) and a `SmoothingDiff` object that records the raw-vs-smoothed delta for each field. The viewer can then render diffs without recomputing them.

**Not part of the initial implementation phases** — this depends on Phase 3 (smoothing pipeline) producing stable output first. But the smoothed data format should be designed with viewer consumption in mind from the start, so the `PosteriorState` schema includes everything the viewer needs.

---

## 6. Open Questions Summary

| # | Question | Decision |
|---|---|---|
| 1 | Pooled vs per-play transition matrices | **Pooled** (too little data to split). Run leave-one-out CV once learning pipeline is built. |
| 2 | Gaussian vs heavy-tailed affect transitions | **Heavy-tailed** (Student-t). Verify empirically; default to heavy-tail given small corpus. |
| 3 | Character-level arousal prior | **Yes** — include character mean arousal as a feature in the regression. |
| 4 | Superobjective blending method | **Additive mixture** (b) with small λ. Self-regulating, matches SO consistency of 0.516. |
| 5 | Factored vs particle-based inference | **Factored** for v1. Exact discrete + Kalman continuous. |
| 6 | Posterior → BeatState conversion | **MAP** for generation, but also **sample** to discover multimodal posteriors. If modes found, frame choices deterministically in the prompt. |
| 7 | Factor graph smoother vs LLM smoother | **Complementary layers.** LLM smoother stays in Pass 1 (semantic). Factor graph smoother is Pass 1.5 (statistical). No replacement needed. |

---

## 7. Validation Plan

### 7.1 Parameter validation

After learning (Phase 1), verify:
- Transition matrix row sums = 1 (within tolerance)
- Eigenvalues match experiment log (59.9%, 21.7%, 10.0%)
- Desire cluster centroids produce the same cluster assignments as the experiment
- Arousal regressor R² matches experiment (~0.08)

### 7.2 Smoothing validation (Pass 1.5)

After building the smoothing pipeline (Phase 3), test on each play:
- **Tactic prediction accuracy**: Given beats 1..t-1, predict T(t) via forward filtering. Compare to:
  - Majority baseline (14.5%)
  - Character prior only (13.6%)
  - Full factor graph (target: >18%, matching desire-conditioned model)
- **Affect prediction**: Given A_trans(t-1), predict A_trans(t). Compare MSE to:
  - Constant baseline (predict same as t-1)
  - Student-t kernel (should be better for large jumps)
- **Smoothing vs raw extraction**: Compare smoothed posteriors to raw LLM BeatStates. Where do they disagree? Are the smoothed versions more plausible? Compare to LLM smoother corrections — do the factor graph and LLM smoother flag the same beats?
- **Cross-character coupling**: Do smoothed status trajectories for co-present characters show stronger inverse correlation than raw extractions? (They should, since ψ_social enforces this.)

### 7.3 Improv integration validation (Pass 2)

After integration (Phase 4):
- Run the same improv scenes with LLM state updater vs factor graph forward inference
- Compare 6-axis scores (voice fidelity, tactic fidelity, etc.)
- Compare latency (factor graph should be <100ms vs ~5s for LLM)
- Compare cost (factor graph: $0 per turn vs ~$0.01 for LLM)
- Check posterior multimodality: how often does the tactic posterior have >1 mode with p>0.2? When it does, is the scene dramatically richer?
