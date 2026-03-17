# Latent State Architecture

This document describes the current implementation of the UTA Acting System's latent state representation: how dramatic psychology is modeled in code, how the pipeline populates it, and where the current design has limitations.

---

## 1. Core Design Principle: Factored State

The system models character psychology as a **factored latent state** — not a single monolithic hidden variable, but independent dimensions that can change asynchronously. This mirrors how actors work: a character can shift tactics while maintaining the same want, or escalate emotionally while keeping the same relationship stance.

The atomic unit is the **BeatState** (`schemas.py:75`): one character's full psychological snapshot at one beat.

---

## 2. Schema Hierarchy

All schemas are defined in `schemas.py`. The structural hierarchy:

```
Play
  ├── Act[]
  │     └── Scene[]
  │           ├── beats: Beat[]
  │           │     ├── utterances: Utterance[]
  │           │     ├── beat_states: BeatState[]   (one per character present)
  │           │     ├── beat_summary: str
  │           │     └── characters_present: str[]
  │           └── bible: SceneBible (optional)
  │
  ├── characters: str[]
  ├── character_bibles: CharacterBible[]
  ├── scene_bibles: SceneBible[]
  ├── world_bible: WorldBible
  └── relationship_edges: RelationshipEdge[]
```

### Utterance

The atomic dialogue unit. Fields: `speaker`, `text`, `stage_direction` (optional), `addressee` (optional). Identified by `play_id`, `act`, `scene`, `index`.

### Beat

A unit of consistent dramatic action, defined by Stanislavski/Meisner theory: a beat ends when a character's objective or tactic shifts, significant new information enters, or the emotional register sharply changes.

Parsers produce one provisional beat per scene. The segmenter (`analysis/segmenter.py`) refines these into real beats via LLM annotation, cached to `data/beats/{play_id}_beats.json`.

---

## 3. The BeatState: Factored Latent Dimensions

`BeatState` (`schemas.py:75-106`) has nine factored dimensions plus confidence/ambiguity fields:

### Symbolic dimensions (free-text strings)

| Field | What it captures | Example |
|---|---|---|
| `desire_state` | What the character wants right now (scene want) | "Convince Ranyevskaya to accept the dacha plan" |
| `superobjective_reminder` | How the scene want connects to the deepest, play-long want | "This is his chance to transcend his serf origins" |
| `obstacle` | What blocks the objective | "Her sentimental attachment to the estate" |
| `tactic_state` | Action verb — what the character is doing TO the other person | "persuade", "deflect", "mock", "plead" |
| `defense_state` | Active psychological defense mechanism | "intellectualization" |
| `psychological_contradiction` | Active internal conflict | "Wants to save her while also wanting to own what she owns" |

### Continuous sub-models

**AffectState** (`schemas.py:39-50`) — emotion as a 5D vector, extending Russell's circumplex:

| Dimension | Range | Anchors |
|---|---|---|
| `valence` | [-1, 1] | negative ↔ positive |
| `arousal` | [-1, 1] | calm ↔ activated |
| `certainty` | [-1, 1] | uncertain ↔ certain |
| `control` | [-1, 1] | powerless ↔ dominant |
| `vulnerability` | [0, 1] | guarded ↔ exposed |

Each has a `rationale: str` explaining the rating.

**SocialState** (`schemas.py:53-58`) — relational stance toward the addressee:

| Dimension | Range | Anchors |
|---|---|---|
| `status` | [-1, 1] | low status claim ↔ high status claim |
| `warmth` | [-1, 1] | hostile ↔ warm |

**EpistemicState** (`schemas.py:61-68`) — what the character knows, hides, and misunderstands:

| Field | Type | Purpose |
|---|---|---|
| `known_facts` | `list[str]` | Facts the character knows at this beat |
| `hidden_secrets` | `list[str]` | Facts the character is actively concealing |
| `false_beliefs` | `list[str]` | Things the character believes that are wrong |

### Confidence and ambiguity

| Field | Type | Purpose |
|---|---|---|
| `confidence` | `float` [0, 1] | How certain the interpretation is |
| `alternative_hypothesis` | `str` | An alternative reading if ambiguous |

These exist to preserve interpretive ambiguity — good dramatic analysis admits multiple readings. In practice, the extractor sets `confidence` and `alternative_hypothesis` is often empty.

---

## 4. Bible Models: Aggregated Character Knowledge

Bibles are synthesis outputs — they aggregate beat-level states into durable character knowledge used at inference time.

### CharacterBible (`schemas.py:148-177`)

Built by `analysis/bible_builder.py` from all BeatStates for a character.

**Enduring psychology:**
- `superobjective` — the character's deepest, play-long want
- `wounds_fears_needs` — psychological wounds, fears, and unmet needs
- `recurring_tactics` — action verbs the character uses repeatedly
- `preferred_defense_mechanisms`
- `psychological_contradictions`

**Voice:**
- `speech_style` — prose description of how they speak
- `lexical_signature` — characteristic words/phrases
- `rhetorical_patterns`
- `few_shot_lines` — up to 10 canonical lines sampled evenly across the play

**Knowledge:**
- `known_facts`, `secrets`

**Arc:**
- `arc_by_scene` — `dict[scene_id, str]`: brief note on what changes for this character in each scene

**Derived statistics:**
- `tactic_distribution` — `dict[str, int]`: tactic → count, computed directly from BeatState `tactic_state` values

### SceneBible (`schemas.py:129-136`)

Per-scene summary: `dramatic_pressure`, `what_changes`, `hidden_tensions`, `beat_map` (prose summary of beat progression).

### WorldBible (`schemas.py:139-145`)

Per-play: `era`, `genre`, `social_norms`, `factual_timeline`, `genre_constraints`.

**Caveat:** WorldBible is generated from LLM knowledge of the play title/author — it does not read the parsed text. For well-known plays this works; for obscure plays it would need to be text-grounded.

### RelationshipEdge (`schemas.py:180-188`)

Tracks pairwise character dynamics: `temperature_by_beat` (warmth float per beat), `power_by_beat` (A's power over B per beat), and a `summary`.

**Caveat:** RelationshipEdges are defined in the schema but not currently populated by the pipeline. The `build_all_bibles` function does not build them. They would need a dedicated extraction step.

---

## 5. Pass 1: How Latent States Are Populated

The pipeline (`scripts/run_analysis.py`) runs five steps:

### Step 1: Parse (`ingest/`)

- **Gutenberg** (`gutenberg_parser.py`): Regex-based parsing of Constance Garnett formatting (`SPEAKER. dialogue`). Handles multi-play files via `text_anchor`. Produces one provisional beat per scene.
- **TEI-XML** (`tei_parser.py`): Parses `<sp>`, `<speaker>`, `<stage>`, `<div type="act|scene">` with namespace handling. Also one beat per scene initially.

Both parsers extract `Utterance` objects and build the `Play → Act → Scene → Beat` hierarchy. Stage directions are captured when available. Addressee estimation is not implemented — `addressee` is always `None`.

### Step 2: Segment beats (`analysis/segmenter.py`)

An LLM (Opus) reads each scene's utterances and returns a JSON array of beat boundary indices. The system prompt defines beats in Stanislavski/Meisner terms. Results are cached to `data/beats/` and reused on subsequent runs.

Scenes with fewer than 3 utterances are not segmented.

### Step 3: Extract BeatStates (`analysis/extractor.py`)

For each beat, the LLM receives:
- Play title, act, scene, beat index
- Prior context (summaries of last 3 beats in the scene)
- The beat's utterances
- Characters present

It returns structured JSON with all BeatState fields for each character. The code parses the JSON, constructs `AffectState`, `SocialState`, `EpistemicState` sub-models, and attaches the resulting `BeatState` objects to the beat.

**Error handling:** If JSON parsing fails, the code tries a regex fallback to extract a JSON object. If that fails, the beat gets no BeatStates. Individual characters that fail to parse are silently skipped.

### Step 4: Smooth arcs (`analysis/smoother.py`)

For each character, the LLM reads their full BeatState trajectory across the play and returns a JSON array of corrections. Each correction specifies `beat_id`, `field`, `old_value`, `new_value`, `rationale`. Corrections are applied via `setattr`. Runs `SMOOTH_PASSES` times (default: 2).

The smoother looks for: unmotivated tactic jumps, knowledge inconsistencies, abrupt emotional transitions, superobjective drift, missing contradictions.

### Step 5: Build bibles (`analysis/bible_builder.py`)

- **CharacterBible**: LLM reads the character's beat-by-beat arc + sample canonical lines → synthesizes the bible fields. `tactic_distribution` is counted directly from `BeatState.tactic_state`. `few_shot_lines` are sampled from actual utterances (up to 10, evenly distributed).
- **SceneBible**: LLM reads scene utterances + beat states → synthesizes the four scene bible fields.
- **WorldBible**: LLM is given only play title + author → relies on its training knowledge.

Output: Complete `Play` object saved to `data/parsed/{play_id}.json` and `data/bibles/{play_id}_bibles.json`.

---

## 6. Pass 2: How Latent States Drive Improvisation

The improv system (`improv/`) uses the bible + scene context to generate lines controlled by a live BeatState that evolves turn-by-turn.

### State initialization (`improvisation_loop.py:87-162`)

Given a `CharacterBible` and `SceneContext`, the LLM infers an initial `BeatState`: what the character would want, which tactic they'd default to, initial affect/social state. The epistemic state falls back to the bible's `known_facts` and `secrets` if the LLM doesn't specify them.

Continuous values are clamped to their valid ranges via a `_clamp` helper.

### Line generation (`improvisation_loop.py:221-276`)

The LLM receives a structured control prompt containing all nine BeatState dimensions, the character's speech style, up to 6 few-shot canonical lines, and the scene context. It returns a `CandidateLine` with `text` and `internal_reasoning`.

**What the generation prompt includes from the BeatState:**
- `desire_state`, `obstacle`, `tactic_state`, `superobjective_reminder`
- `affect_state.valence`, `affect_state.arousal`, `affect_state.vulnerability`
- `social_state.status`, `social_state.warmth`
- `defense_state`, `psychological_contradiction`
- `epistemic_state.hidden_secrets` (up to 3)

**What it omits:** `affect_state.certainty`, `affect_state.control`, `epistemic_state.known_facts`, `epistemic_state.false_beliefs`, `confidence`, `alternative_hypothesis`.

### Six-axis scoring (`improv/scorer.py`)

A critic LLM scores the candidate on:

| Axis | What it checks |
|---|---|
| `voice_fidelity` | Does this sound like the character? |
| `tactic_fidelity` | Does the action verb match the specified tactic? |
| `knowledge_fidelity` | Does the character reveal only what they should know? |
| `relationship_fidelity` | Is status/warmth appropriate? |
| `subtext_richness` | Is something happening beneath the words? |
| `emotional_transition_plausibility` | Is the affect shift motivated? |

Each axis is rated 1-5. Feedback is targeted and state-based (e.g., "This line is too direct; the tactic should be 'deflect'").

### Revision loop (`improvisation_loop.py:283-332`)

If `mean_score < SCORE_THRESHOLD` (3.0), the feedback is injected into the generation prompt and the line is regenerated. Up to `MAX_REVISION_ROUNDS` (3) retries. After the final round, the line is accepted regardless of score.

### State update (`improv/state_updater.py`)

After each turn, a critic LLM reads the previous BeatState + the line delivered + partner response and returns an updated state. Changes are incremental. The updater also reports `beat_shifted` (boolean) and `beat_shift_reason`, though these are not currently used to alter the flow.

**What the state updater receives from the previous state:** `desire_state`, `tactic_state`, affect (valence, arousal, vulnerability only), social (status, warmth), `defense_state`.

**What it omits:** `obstacle`, `superobjective_reminder`, `epistemic_state`, `confidence`, `alternative_hypothesis`, `affect_state.certainty`, `affect_state.control`.

---

## 7. Evaluation: Three-Tier Comparison

`evaluation/judge.py` implements the three-tier protocol:

| Tier | What the LLM receives | Implementation |
|---|---|---|
| **Vanilla** | Zero-shot prompt: "You are [CHARACTER]." + scene context | `generate_vanilla()` |
| **Bible** | CharacterBible as system prompt + scene context | `generate_with_bible()` |
| **Reflection** | Full BeatState initialization + improv loop with scoring/revision | `generate_with_reflection()` — delegates to `run_turn()` |

A judge LLM (Opus) rates each generated line on 7 dimensions (recognizability, playability, tactic_fidelity, subtext, earned_affect, knowledge_fidelity_pass/note, identified_tactic). Each judge prompt runs `num_runs` times (default: 3) with `temperature=0.3` and scores are averaged.

**Current state:** Only Lopakhin evaluation scenes are hardcoded in `scripts/run_evaluation.py` (3 scenes). The PLAN calls for 20 scenes per condition x 3 conditions x 3 familiarity levels. The low-familiarity (Lomov) and synthetic character conditions are not yet implemented.

---

## 8. Data Flow Summary

```
Raw text/XML
    │
    ▼
[Parser] ──→ Play with provisional beats (1 per scene)
    │
    ▼
[Segmenter] ──→ Refined beats (cached to data/beats/)
    │
    ▼
[Extractor] ──→ BeatState[] per beat (all 9 dimensions)
    │
    ▼
[Smoother] ──→ Corrected BeatStates (arc-level coherence)
    │
    ▼
[Bible Builder] ──→ CharacterBible, SceneBible, WorldBible
    │
    ▼
[Full Play JSON] ──→ data/parsed/{play_id}.json
                      data/bibles/{play_id}_bibles.json

At inference time:

CharacterBible + SceneContext
    │
    ▼
[State Initializer] ──→ Initial BeatState
    │
    ▼
[Generator] ──→ CandidateLine (controlled by BeatState)
    │
    ▼
[Scorer] ──→ ScoredLine (6 axes + feedback)
    │
    ├── if below threshold ──→ back to Generator with feedback
    │
    ▼
[State Updater] ──→ Updated BeatState for next turn
```

---

## 9. Roadmap: From Symbolic States to Probabilistic Inference

The current system (Phase A) uses Option A: every BeatState is a point estimate — one tactic, one desire, one affect vector — produced by a single LLM call. This section describes how the existing data structures map onto probabilistic representations that become viable as the corpus grows.

### 9.1 Current implementation: what is actually wired up

The current system does not use a factor graph. State transitions are LLM-generated point estimates, and the statistical components operate as standalone modules rather than as factors in a joint model. The diagram below shows the actual dependencies between components.

```mermaid
graph TD
    subgraph "t-1"
        D1["D(t-1)"]
        T1["T(t-1)"]
        A1["A(t-1)"]
        S1["S(t-1)"]
        K1["K(t-1)"]
        Delta1["Δ(t-1)"]
        U1(["U(t-1) — observed utterance"])
    end

    subgraph "Scorer — emission proxy"
        score{{"6-axis scorer"}}
    end

    D1 -. "voice fidelity" .-> score
    T1 -. "tactic fidelity" .-> score
    A1 -. "emotional transition" .-> score
    S1 -. "relationship fidelity" .-> score
    K1 -. "knowledge fidelity" .-> score
    U1 --> score

    subgraph "Statistical priors (standalone)"
        tp{{"P(T|character)"}}
        tm{{"P(T(t)|T(t-1))"}}
        az{{"affect z-score"}}
    end

    T1 --> tm
    T1 --> tp
    A1 --> az

    subgraph "LLM state updater"
        upd["LLM call"]
    end

    D1 --> upd
    T1 --> upd
    A1 --> upd
    S1 --> upd
    Delta1 --> upd
    U1 --> upd

    subgraph "t"
        D2["D(t)"]
        T2["T(t)"]
        A2["A(t)"]
        S2["S(t)"]
        K2["K(t) ⚠ not updated"]
        Delta2["Δ(t)"]
    end

    upd --> D2
    upd --> T2
    upd --> A2
    upd --> S2
    upd --> Delta2
    K1 -. "copied unchanged" .-> K2
```

**Key gaps relative to the target factor graph (§9.2):**

| Component | Target | Current |
|---|---|---|
| Transition model | Per-variable factors (ψ_D, ψ_T, ...) with explicit dependencies | Single LLM call updates D, T, A, S, Δ jointly; no factored structure |
| ψ_T dependencies | ψ_T(T(t-1), T(t), D(t)) — conditions on desire | P(T(t)\|T(t-1)) bigram only; no conditioning on D(t) |
| Epistemic transitions | ψ_K(K(t-1), K(t), U(t-1)) — updates from utterance | K(t) is never updated; carried forward unchanged |
| Emission model | ψ_emit as explicit likelihood P(U\|state) | 6-axis scorer evaluates all state dimensions but returns ordinal scores (1–5), not likelihoods |
| Cross-character coupling | ψ_social, ψ_epistemic link characters in shared beats | No cross-character factors; characters updated independently |
| Inference | Belief propagation over factor graph | Point estimates from LLM; statistical priors feed dramaturgical feedback but not inference |

The migration to Option B means replacing the monolithic LLM state updater with factored transition kernels and connecting the scorer/priors into a joint graphical model.

### 9.2 Factor graph over the factored state

The factored BeatState design was chosen specifically to enable this migration. Each dimension becomes a variable node in a factor graph. The key insight is that these dimensions are **conditionally independent given certain shared contexts**, but not marginally independent — which is exactly the structure a factor graph can exploit.

For a single character `c` at beat `t`, the factored state is:

```
Z_c(t) = { D(t), T(t), A(t), S(t), K(t), Δ(t) }

where:
  D(t)  = desire / objective       (discrete: finite vocabulary of wants)
  T(t)  = tactic                   (discrete: finite vocabulary of action verbs)
  A(t)  = affect                   (continuous: 5D vector in [-1,1]^4 × [0,1])
  S(t)  = social state             (continuous: 2D vector in [-1,1]^2)
  K(t)  = epistemic state          (structured: sets of propositions)
  Δ(t)  = defense / contradiction  (discrete or structured)
```

The factor graph connects these through several types of factors:

```mermaid
graph TD
    subgraph "t-1"
        D1["D(t-1)"]
        T1["T(t-1)"]
        A1["A(t-1)"]
        S1["S(t-1)"]
        K1["K(t-1)"]
        Delta1["Δ(t-1)"]
        emit1{{"ψ_emit(t-1)"}}
        U1(["U(t-1) — observed utterance"])
        D1 --- emit1
        T1 --- emit1
        A1 --- emit1
        emit1 --- U1
    end

    subgraph "Transition factors"
        pD{{"ψ_D"}}
        pT{{"ψ_T"}}
        pA{{"ψ_A"}}
        pS{{"ψ_S"}}
        pK{{"ψ_K"}}
        pDelta{{"ψ_Δ"}}
    end

    D1 --- pD
    T1 --- pT
    A1 --- pA
    S1 --- pS
    K1 --- pK
    Delta1 --- pDelta
    U1 --- pK

    subgraph "t"
        D2["D(t)"]
        T2["T(t)"]
        A2["A(t)"]
        S2["S(t)"]
        K2["K(t)"]
        Delta2["Δ(t)"]
        emit2{{"ψ_emit(t)"}}
        U2(["U(t) — observed utterance"])
        D2 --- emit2
        T2 --- emit2
        A2 --- emit2
        emit2 --- U2
    end

    pD --- D2
    pT --- T2
    D2 --- pT
    pA --- A2
    pS --- S2
    pK --- K2
    pDelta --- Delta2
```

**Factor types:**

1. **Transition factors** ψ_D, ψ_T, ψ_A, ψ_S, ψ_K, ψ_Δ: encode how each dimension evolves between beats. These capture the acting-theory constraint that state changes should be motivated.
   - ψ_T(T(t-1), T(t), D(t)): tactic transitions depend on whether the objective changed — same want often → same tactic; new want → likely new tactic.
   - ψ_A(A(t-1), A(t)): affect transitions are smooth (small Δ per beat) unless a major event occurs. Implementable as a Gaussian transition kernel with learned variance.
   - ψ_K(K(t-1), K(t), U(t-1)): epistemic state is monotonic for facts (you can't un-learn something you heard) with possible revelation events when secrets are exposed.

2. **Emission factors** ψ_emit: P(utterance | state). These encode the probability of the observed dialogue given the hidden state. In the current system, the LLM acts as an implicit emission model — it generates text conditioned on state. In a probabilistic version, this becomes an explicit likelihood: how well does this utterance fit this tactic, this affect, this speech style?

3. **Cross-character factors**: connect states across characters present in the same beat.
   - ψ_social(S_c1(t), S_c2(t)): status is relational — if c1 claims high status, c2's status claim is constrained.
   - ψ_epistemic(K_c1(t), K_c2(t), U(t)): if c1 reveals a secret in utterance U(t), c2's known_facts must update.
   - ψ_relationship(R(c1,c2,t)): the `RelationshipEdge` schema already models this — `temperature_by_beat` and `power_by_beat` become time-varying edge potentials.

4. **Global factors**: enforce play-level constraints.
   - ψ_arc(D(1), D(2), ..., D(T)): the character's desire trajectory should trace a coherent arc (the superobjective).
   - ψ_genre(all states): genre constraints (comedy vs. tragedy) bias the transition priors.

### 9.3 Multi-agent POMDP formulation

At inference time (Pass 2 / improvisation), the factor graph becomes a **multi-agent POMDP**. Each character is an agent with:

```
For character c:
  State:        Z_c(t) = { D(t), T(t), A(t), S(t), K(t), Δ(t) }
  Observation:  O_c(t) = partner's utterance + visible stage actions
  Action:       a_c(t) = (tactic choice, utterance) — what the character does/says
  Reward:       R(t) = alignment with superobjective + tactic effectiveness
  Transition:   Z_c(t+1) = f(Z_c(t), a_c(t), O_c(t))
  Belief:       b_c(t) = P(Z_partner | history of observations)
```

The POMDP framing captures what acting actually is:
- Each character has **private beliefs and objectives** (the epistemic state, the desire state)
- Each **observes imperfectly** — you see what the other person says and does, but not what they feel or want
- Each **chooses tactics** as actions designed to affect the other person's state
- Utterances are **actions that affect others' beliefs and relationships**
- The character must maintain a **belief distribution over the partner's hidden state** — "I think she's deflecting, but she might be genuinely indifferent"

The current system approximates this with point estimates and LLM-as-policy. The POMDP formulation would make the belief tracking explicit and allow for principled action selection (tactic choice) under uncertainty about the partner's state.

### 9.4 What the current schemas already support

The existing data structures were designed with this migration in mind:

| Current schema element | Role in probabilistic model |
|---|---|
| `BeatState` per character per beat | Becomes posterior marginal P(Z_c(t) \| observations 1:t) |
| `confidence` field | Seeds the posterior uncertainty; currently a scalar, would become per-dimension |
| `alternative_hypothesis` field | Second mode of a multimodal posterior |
| `tactic_distribution` in CharacterBible | Empirical prior P(tactic \| character) — already computed |
| `arc_by_scene` in CharacterBible | Constraints on the global factor ψ_arc |
| `RelationshipEdge.temperature_by_beat` | Time series for the cross-character social factor |
| `EpistemicState.known_facts/hidden_secrets` | Structured belief state for the POMDP observation model |
| `AffectState` 5D vector | Continuous state for Gaussian transition kernels |
| `ScoredLine` six-axis scores | Proxy for emission likelihood P(utterance \| state) on each dimension |

### 9.5 Migration path: what changes at each phase

**Phase A (current):** LLM produces point-estimate BeatStates. The smoother is an LLM review pass. No formal probabilistic structure.

**Phase B (5+ plays):** Introduce the factor graph.

1. **Discretize tactics and defenses** using the free-text labels accumulated across the corpus. Cluster the LLM's tactic outputs (e.g., "deflect", "deflection", "avoid" → DEFLECT) to build a canonical vocabulary. This gives a finite state space for the tactic variable.
2. **Learn transition priors** from the corpus: P(T(t) | T(t-1), D(t)) from observed tactic sequences, P(A(t) | A(t-1)) from affect trajectories. These replace the implicit priors currently embedded in LLM prompts.
3. **Calibrate LLM outputs as likelihoods.** Treat the LLM's per-beat extraction as a noisy observation, not ground truth. Run multiple LLM extractions at different temperatures and calibrate P(LLM_output | true_state).
4. **Run belief propagation** (sum-product or max-product) over the factor graph to produce smoothed posterior distributions over states. This replaces the current LLM-based smoother with a principled algorithm.
5. **Use posterior distributions in generation prompts.** Instead of "tactic: deflect", the prompt becomes "tactic distribution: deflect 0.6, conceal 0.3, test 0.1" — letting the LLM choose from the distribution and introducing motivated ambiguity into the generated lines.

**Phase C (many plays):** Replace hand-designed factors with learned potentials. Train character-specific emission models. Learn latent embeddings with symbolic heads (Option C). The factor graph structure is preserved but the potentials are parameterized by neural networks.

### 9.6 What this buys over the current system

The probabilistic formulation addresses several current limitations directly:

- **Limitation #1 (single-shot extraction):** Factor graph inference aggregates evidence across beats and characters, not just one LLM call per beat.
- **Limitation #2 (uncalibrated floats):** Transition priors learned from corpus provide empirical grounding for affect/social dimensions.
- **Limitation #3 (no temporal smoothing):** Belief propagation is a formal smoothing algorithm — it replaces the LLM review pass.
- **Limitation #4 (unpopulated RelationshipEdges):** Cross-character factors make relationship tracking a structural part of inference, not a separate aggregation step.
- **Limitation #15 (no tactic taxonomy):** The discretization step in Phase B produces a canonical vocabulary grounded in observed usage.

---

## 10. Limitations and Caveats (Current Implementation)

### Latent state representation

1. **All extraction is single-shot LLM inference.** There is no probabilistic inference, belief propagation, or factor graph structure. Each BeatState is a point estimate from a single LLM call. The `confidence` and `alternative_hypothesis` fields exist in the schema but are underutilized — the LLM typically sets `confidence: 1.0` and leaves `alternative_hypothesis` empty.

2. **Continuous dimensions are LLM-estimated, not grounded.** Affect and social state floats (valence, arousal, status, warmth, etc.) are produced by asking the LLM to output numbers. There is no calibration, no anchoring protocol, and no inter-annotator agreement baseline. The numbers are plausible but not validated.

3. **No temporal smoothing model.** The smoother (`analysis/smoother.py`) is an LLM review pass, not a statistical smoothing algorithm. There is no HMM, no Kalman filter, no belief propagation over time. State trajectories are not formally modeled as dynamical systems — they are corrected post hoc by a second LLM call.

4. **RelationshipEdges are not populated.** The schema exists but `build_all_bibles` does not build them. Pairwise relationship dynamics are partially captured in `SocialState` (warmth/status per beat) but not aggregated into relationship trajectories.

5. **Addressee estimation is not implemented.** `Utterance.addressee` is always `None`. The system prompts do not leverage addressee information, and the `SocialState` (status/warmth "toward addressee") has no explicit addressee binding when multiple characters are present.

### Information loss in the pipeline

6. **Generation prompt omits several BeatState fields.** The generation prompt (`improvisation_loop.py:192-218`) sends `affect_state.valence`, `arousal`, and `vulnerability` but omits `certainty` and `control`. It omits `epistemic_state.known_facts` and `false_beliefs`. The state updater similarly operates on a subset of the full state. This means some extracted dimensions influence bible synthesis but not line generation.

7. **State updater does not update epistemic state.** After each improv turn, the updater adjusts desire, tactic, affect, social, and defense — but does not track changes to what the character knows, what secrets have been revealed, or what false beliefs have been corrected. Over a long improv session, the epistemic state drifts from reality.

8. **WorldBible is not text-grounded, and creates a contamination channel.** The WorldBible is generated from the LLM's knowledge of the play title and author, not from the parsed text. This works in two directions:
   - For **obscure plays**, the LLM lacks the knowledge to produce an accurate WorldBible, making the output unreliable.
   - For **well-known plays**, the WorldBible becomes a conduit for the contamination problem described in PLAN.md §3. The LLM already "knows" the world of Hamlet or Cherry Orchard from pretraining. When the WorldBible (era, social norms, genre constraints, factual timeline) is fed back into the generation and scoring prompts, it is unclear whether the system's structured state representation is doing the work, or whether the LLM is simply pattern-matching against memorized knowledge about these plays. This compresses the delta between tiers and makes it harder to attribute improvements to the system architecture rather than the LLM's prior familiarity with the material. The same concern applies to CharacterBible and BeatState extraction — the LLM's pretraining on canonical literary analysis may dominate its outputs for well-known characters, regardless of what the prompts specify. The low-familiarity and synthetic character conditions in the evaluation design exist precisely to control for this.

### Evaluation gaps

9. **Only 3 hardcoded evaluation scenes.** The PLAN calls for 20 scenes x 3 tiers x 3 familiarity levels. The current implementation has 3 Lopakhin scenes. There are no Hamlet evaluation scenes, no low-familiarity character (Lomov / The Proposal), and no synthetic character.

10. **No inter-rater reliability reporting.** The judge runs each prompt 3 times and averages, but does not report variance or compute agreement metrics. The PLAN specifies reporting variance and using 2 judge models as ensemble; neither is implemented.

11. **Evaluation uses the same model family for generation and judging.** Both generation and scoring/judging use Claude models. There is no cross-model evaluation to control for model-specific biases.

### Parsing limitations

12. **Gutenberg parser is format-specific.** It handles the Constance Garnett convention (`SPEAKER. dialogue`) but may fail on other Gutenberg formatting styles. The standalone-speaker-name fallback is a heuristic that can false-positive on all-caps headings.

13. **Stage directions are discarded during Gutenberg parsing.** The Gutenberg parser recognizes stage directions (`[brackets]` or `(parentheses)`) but drops them rather than attaching them to utterances. The TEI parser captures them.

### Structural issues

14. **Beat summary is not populated during extraction.** `Beat.beat_summary` exists but is never set by the extractor or segmenter. It remains empty for all beats. The prior-context formatting in the extractor falls back to showing the first utterance of each prior beat.

15. **No formal taxonomy for tactics or defenses.** `tactic_state` and `defense_state` are free-text strings. The LLM is prompted with example action verbs, but there is no controlled vocabulary, no canonical list, and no normalization. The same tactic may appear as "deflect", "deflection", or "avoid". The `tactic_distribution` in CharacterBible counts exact string matches.

16. **No `__init__.py` files.** The `ingest/`, `analysis/`, `improv/`, and `evaluation/` directories lack `__init__.py`. Imports work via `sys.path.insert(0, ...)` at the top of every file. This is fragile and prevents the project from being installed as a package.

---

## 11. Current State of Pipeline Outputs

Both plays have been fully processed:

| Play | Raw source | Beats cached | Full parse | Bibles |
|---|---|---|---|---|
| Cherry Orchard | `data/raw/cherry_orchard.txt` (419 KB) | Yes | Yes (1.7 MB) | Yes |
| Hamlet | `data/raw/hamlet.xml` (583 KB) | Yes | Yes (2.7 MB) | Yes |

The parsed JSON files contain the complete `Play` objects with all BeatStates, CharacterBibles, SceneBibles, and WorldBibles. These are viewable through the Redwood SDK viewer app (`viewer/`).
