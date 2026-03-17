# Factor Graph Migration — Opportunities with Three Plays

This document assesses the gap between the current system (Phase A + partial Phase B) and the target factor graph / POMDP described in [LATENT_STATE_ARCHITECTURE.md](LATENT_STATE_ARCHITECTURE.md) §9, identifies concrete opportunities that three parsed plays now unlock, and proposes an experiment-first roadmap for gaining the insights needed before committing to final architecture.

---

## 0. Where We Stand

| Metric | Cherry Orchard | Hamlet | Earnest | Total |
|---|---|---|---|---|
| Characters with bibles | 12 | 15 | 9 | 36 |
| Beats | 469 | 645 | 191 | 1,305 |
| Superobjectives populated | 0/12 | 3/15 | 4/9 | 7/36 |
| Relationship edges | 0 | 0 | 0 | 0 |
| World bible populated | No | No | Yes | 1/3 |
| Canonical tactic coverage | 79% (66 clusters from 297 raw strings) ||||

Three plays gives us a modest but structurally interesting corpus: a Chekhovian ensemble drama, a Shakespearean tragedy, and a Wildean farce. The genre diversity is more useful than three plays of the same type would be — it lets us distinguish genre-invariant patterns from genre-specific ones.

### Data gaps to resolve before experiments begin

Two data gaps block downstream work and should be investigated first:

1. **Relationship edges are empty for all three plays.** The `relationship_builder.py` module exists and was documented as having built 134 directed edges, but the bibles JSON files show empty arrays. This may be a serialization/persistence bug, or an agent may have run a local experiment whose results were never written back. This needs debugging — cross-character factors (ψ_social, ψ_epistemic) require pairwise data.

2. **Superobjectives are missing for 29/36 characters.** The three populated superobjectives in Hamlet are for minor characters (Barnardo, Marcellus, Reynaldo), which is the opposite of expected. This suggests a bug in the bible builder or a missing flag when the pipeline was run. Needs investigation before arc factor work can begin.

---

## 1. Calibrating LLM-Based Outputs

The core problem: every BeatState dimension is a single-shot LLM point estimate with no empirical grounding. Three plays give us enough repeated structure to begin calibrating.

### 1.1 Tactic calibration via cross-play consistency

With 1,305 beats and 66 canonical tactics, we can now ask: **does the LLM assign tactics consistently across plays?** If "deflect" in Cherry Orchard looks the same as "deflect" in Hamlet in terms of surrounding affect/social state, that's evidence the label is calibrated. If not, we've found a systematic bias.

**Concrete step:** For each canonical tactic with ≥15 occurrences across the corpus, compute the mean affect vector (valence, arousal, vulnerability) and social vector (status, warmth) when that tactic is active. Test whether the within-tactic variance is smaller than the between-tactic variance (a basic discriminant validity check). Weight plays equally (not by beat count) to avoid Earnest being washed out — its comedic register is exactly where calibration is most likely to break.

Genre will not be treated as a covariate at n=3 (too few plays, and genre is confounded with playwright). However, if the discriminant analysis shows that specific tactics have systematically different affect profiles across plays, that pattern is worth noting — it becomes a hypothesis to test with more data (see §6, H3).

### 1.2 Affect dimension anchoring

The 5D affect vector (valence, arousal, certainty, control, vulnerability) is extracted as floats with no anchoring protocol. Three plays let us build empirical baselines:

- **Per-play affect distributions:** What is the mean and variance of each affect dimension across all beats in each play? Hamlet should skew lower valence and higher arousal than Earnest. If it doesn't, the LLM's affect extraction is poorly calibrated.
- **Per-character affect baselines:** Already partially captured in the `StatisticalPrior`'s z-score mechanism. With three plays we can ask whether z-score thresholds (1.5 for tier 2, 2.5 for tier 3) are empirically reasonable or need per-genre adjustment.

**Concrete step:** Compute descriptive statistics (mean, std, quartiles) for each affect dimension, grouped by play and by character. Visualize as violin plots or ridge plots. This is cheap (zero API cost, pure computation over existing data) and immediately informative.

The certainty and control dimensions are extracted but never used in generation or scoring (see [LATENT_STATE_ARCHITECTURE.md](LATENT_STATE_ARCHITECTURE.md) §6, limitation #6). Part of this analysis should explicitly test whether they earn their keep — do they discriminate characters or predict tactic choice? If not, dropping them would save tokens. If they do, that justifies continuing to extract them. This is testable locally (see §6, H1).

### 1.3 Ensemble extraction: deferred

The `BeatStateEstimate` schema is defined but unimplemented (Phase B.2). A calibration study is viable:

1. Sample ~30 beats per play (90 total), stratified by scene length and number of characters present
2. Run 3×Sonnet ensemble at temperatures 0.3/0.6/0.9
3. Compare ensemble tactic posteriors to the existing Opus point estimates
4. Measure: (a) agreement rate for discrete variables, (b) RMSE for continuous variables, (c) whether ensemble disagreement predicts smoother corrections (i.e., beats where the ensemble disagrees are the same beats the smoother flagged)

This gives us the observation likelihoods the factor graph needs (§5.5 of the Statistical Learning Phase Design). However, it requires LLM budget (~$10–15). **Deferred until local experiments have identified which dimensions earn their keep** — no point calibrating dimensions we may drop. Logged as a future experiment (see §6, E1).

---

## 2. Sanity-Check Evals: Expected Clustering

If we construct latent-space models of characters and relationship pairs, certain clusters should emerge as basic sanity checks. If they don't, something is wrong with the representation. Here are the expected groupings, organized by what they test.

### 2.1 Character clustering by tactic distribution

Cluster characters by their normalized tactic distribution (66-dim probability vector). Expected structure:

**Servants / low-status pragmatists** — should cluster:
- Lane (Earnest), Merriman (Earnest), Fiers (Cherry Orchard), Yasha (Cherry Orchard)
- Shared signature: high DEFER, COMPLY, INFORM; low DOMINATE, CHALLENGE
- *But* Yasha should be an outlier within this group — he's aspirational and contemptuous, more DISMISS and MOCK than the others. If he clusters tightly with Lane, the tactic extraction is missing his cruelty.

**Social dominators** — should cluster:
- Lady Bracknell (Earnest), King Claudius (Hamlet), Polonius (Hamlet)
- Shared signature: high COMMAND, INTERROGATE, DOMINATE, DISMISS
- *But* Lady Bracknell's dominance is comedic (PRONOUNCE, DISMISS with absurdist logic) while Claudius's is political (MANIPULATE, CHARM, COERCE). If they're indistinguishable, the tactic vocabulary may be too coarse to capture register.

**Romantic idealists / naifs** — should cluster:
- Anya (Cherry Orchard), Cecily (Earnest), Ophelia (Hamlet)
- Shared signature: high AFFIRM, EMBRACE, PLEAD; low DEFLECT, MANIPULATE
- This cluster tests whether the system sees the structural similarity across genres.

**Wit-driven deflectors** — should cluster:
- Algernon (Earnest), Hamlet, Trofimov (Cherry Orchard)
- Shared signature: high MOCK, DEFLECT, CHALLENGE, PROVOKE
- This is the most interesting test: these characters use intellect and wit as a defense mechanism, but in radically different dramatic contexts. If they cluster, the system is capturing something real about dramatic function.

The **wit-driven deflectors** and **romantic idealists / naifs** are the most load-bearing sanity checks — they test cross-genre structural similarity, which is harder to get right than within-genre clustering.

Hamlet is a particularly interesting case: his tactic entropy may be much higher than Algernon's, since he shifts modes dramatically across the play. Whether he clusters with the wit-driven deflectors — or fails to cluster anywhere — is informative either way. A sub-experiment worth running: cluster Hamlet's tactic distribution *per scene* (e.g., Act 2 Scene 2 with Polonius vs. Act 3 Scene 1's soliloquy) to see whether scene-level slices of Hamlet land in different clusters.

Run character clustering first (cheap, zero API cost). Use results to decide whether to invest in relationship-pair clustering.

### 2.2 Character clustering by affect trajectory shape

Instead of tactic distribution, cluster by the *shape* of the affect trajectory across the play. Represent each character's arc as a time series of (valence, arousal, vulnerability) values, then use DTW (dynamic time warping) or a simpler summary (mean, trend slope, variance) for distance.

Expected structure:

**Descending arc** (high → low valence, rising vulnerability):
- Lubov (Cherry Orchard), Ophelia (Hamlet), possibly Gaev (Cherry Orchard)

**Ascending or stable-high arc** (rising or sustained positive valence):
- Jack (Earnest, comedic resolution), Cecily (Earnest), Anya (Cherry Orchard, hopeful ending)

**Volatile / high-entropy arc** (large swings, high arousal variance):
- Hamlet, Lopakhin (Cherry Orchard), Algernon (Earnest, though comedic volatility)

For comparing trajectories across plays of different lengths, windowed summaries (per-act or fixed-length windows) are preferable to raw normalization — it's unclear whether longer plays pace differently than shorter ones, and windowing avoids letting granularity differences dominate the distance metric. Whether pacing differs across our n=3 plays is itself testable (see §6, H4).

### 2.3 Relationship pair clustering

Once relationship edges are populated (currently empty — see §0), cluster directed pairs by their warmth/status trajectories.

Expected structure:

**Asymmetric devotion** (A warm + low-status toward B; B cool + high-status toward A):
- Lopakhin → Lubov, Ophelia → Hamlet, Horatio → Hamlet

**Adversarial parity** (both low warmth, contested status):
- Hamlet ↔ King, Hamlet ↔ Laertes, Jack ↔ Algernon (comedic version)

**Comedic courtship** (oscillating warmth, playful status games):
- Jack ↔ Gwendolen, Algernon ↔ Cecily
- These should be structurally similar to each other (Wilde designed them as mirrors) but different from the tragic pairs above.

Blocked on populating relationship edges.

### 2.4 Cross-genre discriminant validity

A meta-test: if we cluster all 36 characters in a shared embedding space (tactic distribution + mean affect + mean social state), do the plays separate? They should partially — Earnest characters should occupy a distinct region (higher valence, more MOCK/CHARM/DISMISS, less PLEAD/LAMENT) — but with bridges. The bridging characters are the interesting ones:

- Trofimov (Cherry Orchard) might drift toward Earnest characters (intellectual, ironic, low stakes)
- Horatio (Hamlet) might drift toward Cherry Orchard characters (observing, low-tactic, warm)

If the three plays form three tight, non-overlapping clusters with no bridges, the representation is capturing genre more than character, which would mean the LLM's extraction is heavily influenced by its knowledge of each play's genre rather than by the text of individual scenes.

---

## 3. Grounding Analysis for Unknown Plays

### 3.1 The contamination problem, sharpened

The three plays we have span a spectrum of LLM familiarity:

| Play | LLM familiarity | Risk |
|---|---|---|
| Hamlet | Extremely high | LLM's pretraining on centuries of Hamlet criticism dominates extraction; hard to attribute results to our pipeline vs. memorized analysis |
| Cherry Orchard | High | Same risk, somewhat less extreme |
| Earnest | High | Same, but comedic register may be underrepresented in training data's dramatic analysis |

All three are well-known. This is useful for sanity-checking (we can compare our BeatStates to established criticism) but dangerous for claims about the system's analytical capability. When we add an obscure play — or when a user brings their own unpublished script — the system must still work.

### 3.2 WorldBible grounding: from LLM recall to text-derived

Currently, WorldBible generation relies on the LLM knowing the play (`LATENT_STATE_ARCHITECTURE.md` §5, step 5). Only Earnest has a populated WorldBible; Cherry Orchard and Hamlet have empty ones. For unknown plays, this approach fails entirely.

**Proposed approach: gestalt-first WorldBible extraction**

Give the LLM the full parsed play text (all utterances + stage directions, no BeatStates) and ask it to derive the WorldBible fields from the text alone, as if encountering the work for the first time:

1. **Era/setting inference**: From vocabulary, proper nouns, social conventions mentioned in dialogue, stage directions describing setting
2. **Genre inference**: From dramatic structure (does it end in marriage or death?), tactic distribution (farce has more MOCK/CHARM; tragedy has more PLEAD/LAMENT/CHALLENGE), tone of stage directions
3. **Social norms**: Inferred from how characters address each other, what causes scandal or approval, who has power and why
4. **Factual timeline**: Events referenced as having happened before the play begins, reconstructed from dialogue
5. **Genre constraints**: What the play's internal logic permits (Earnest: coincidence is fine; Hamlet: actions have consequences)

This is a single long-context LLM call. For Hamlet (~583KB XML → ~150K tokens of utterances), it fits in a 200K context window. For shorter plays like Earnest, easily.

**Calibration approach**: Instructing the LLM to "pretend it doesn't know the play" is unenforceable — it will recognize Hamlet regardless. The better approach is to run both text-grounded and knowledge-grounded WorldBible extraction on all three plays and measure the delta. This reveals what the text alone can and cannot supply, and which text-grounded fields are likely contaminated by training knowledge. For truly unknown plays, the text-grounded version is all we'd have, and the delta tells us what kinds of information are typically missing from text-only analysis.

### 3.3 The gestalt question: what can an LLM see in a single read?

Beyond the structured WorldBible fields, there's a richer question: **what dramaturgical gestalt can an LLM extract from a cold read of the full play?**

A human dramaturg, reading a play for the first time, picks up on:
- **Structural rhythm**: How scenes alternate in energy/length; where the playwright placed the climax
- **Character gravity**: Which characters the play orbits around (not always the one with the most lines)
- **Thematic obsessions**: Recurring images, metaphors, topics that the play keeps circling back to
- **The unsaid**: What the play conspicuously avoids mentioning, which often reveals its deepest concerns
- **Tonal signature**: Whether the play trusts its characters (Chekhov) or holds them at ironic distance (Wilde) or traps them (Shakespeare's tragedies)

An LLM, given the full text in a single context window, should be able to approximate several of these — especially structural rhythm (measurable from scene lengths and speaker distributions) and thematic recurrence (detectable from lexical patterns). Tonal signature is harder but potentially detectable from the ratio of sincere to ironic utterances.

**Proposed experiment**: For each play, prompt a long-context LLM with the full text and ask for a "dramaturgical first impression" — a structured response covering the items above. Then compare this gestalt to the CharacterBibles and SceneBibles we've already built bottom-up from beat-level extraction. Where they agree, our pipeline is consistent with a holistic reading. Where they disagree, either the gestalt missed beat-level nuance or the bottom-up extraction missed forest-for-trees patterns.

Architecturally, the gestalt could serve as a **top-down prior** constraining the factor graph — the global factors ψ_arc and ψ_genre described in LATENT_STATE_ARCHITECTURE.md §9.2. Whether to design the gestalt output as directly consumable factor potentials or keep it as a validation-only signal depends on how much information gain it provides over the bottom-up pipeline. This needs experimental validation before committing architecturally (see §6, H5).

### 3.4 Bootstrapping analysis of an unknown play

Combining the above, here's a proposed pipeline for a play the LLM has never seen:

```
Full play text
    │
    ├──→ [Gestalt extraction] ──→ Top-down priors (genre, arc shape, character gravity)
    │
    ├──→ [Text-grounded WorldBible] ──→ Era, social norms, genre constraints
    │
    ├──→ [Standard pipeline: parse → segment → extract → smooth → bibles]
    │         │
    │         └──→ Bottom-up BeatStates, CharacterBibles, SceneBibles
    │
    └──→ [Reconciliation step] ──→ Flag disagreements between top-down and bottom-up
```

The reconciliation step should be deferred until experiments validate that the gestalt extraction produces meaningfully different information from the bottom-up pipeline. If they always agree, reconciliation is unnecessary. If they always disagree, one of them is broken. The interesting case — partial disagreement on specific dimensions — is where dramaturgical insight lives, but we need to observe it first.

---

## 4. Factor Graph Migration: What Three Plays Unlock

### 4.1 Transition priors from data

With 1,305 beats and 66 canonical tactics, we can now estimate empirical transition matrices:

- **Tactic bigrams**: P(T(t) | T(t-1)) from observed sequences. With ~1,305 × (avg characters per beat) observations, the bigram matrix should be reasonably populated for the top 20–30 tactics, sparse for the tail.
- **Tactic-desire conditioning**: P(T(t) | T(t-1), D_changed). Split transitions by whether the desire state changed at this beat. Theory predicts: same desire → higher tactic persistence; new desire → more tactic diversity. If the data shows this, we have the first empirically validated factor (ψ_T from §9.2).
- **Affect transition kernels**: Fit a simple Gaussian P(A(t) | A(t-1)) = N(A(t-1), Σ) from observed affect trajectories. The covariance Σ tells us which affect dimensions co-vary (e.g., does arousal increase predict vulnerability increase?). Three plays give us ~36 character trajectories to estimate Σ.

Transition priors will be pooled across all plays. With n=3 plays (one per genre), per-genre estimation would give n=1 per genre — too few for reliable estimates, and genre is confounded with playwright. Genre conditioning can be revisited at 6+ plays. The information gain from tactic bigrams to tactic-desire conditioning is itself worth measuring as an experiment (see §6, H6).

### 4.2 Cross-character factors: now estimable

With three plays featuring multi-character scenes, we can begin to estimate the cross-character coupling factors from §9.2:

- **ψ_social**: When character A claims high status, is B's status claim inversely correlated? Measure the correlation between co-present characters' status values per beat. If status is truly relational (zero-sum), we should see negative correlation. If positive, characters are "matching" rather than competing.
- **ψ_epistemic**: When a secret is revealed in a beat (character A's `hidden_secrets` at t-1 contains item X, which appears in character B's `known_facts` at t), does B's behavior change? This requires the epistemic state to actually be updated — which it currently isn't (limitation #7). But we can check whether the existing (static) epistemic states at least contain the right information.

### 4.3 Emission model approximation

The 6-axis scorer already functions as an implicit emission model — it evaluates P(utterance | state). With three plays' worth of scored beats, we can begin to characterize this:

- Collect scorer outputs (if available from evaluation runs) and correlate with BeatState dimensions
- Even without scorer data, we can use the extracted BeatStates as a noisy P(state | utterance) and test whether the mapping is invertible: given a BeatState, can we predict features of the corresponding utterance (length, question density, imperative density, sentiment)?

### 4.4 Superobjective and arc factors

Only 7/36 characters have populated superobjectives. Before we can estimate the global arc factor ψ_arc, we need superobjectives for at least the major characters. This is a straightforward gap to close — see §0 for the investigation needed.

---

## 5. Prioritized Roadmap

Items are ordered by (information value × feasibility), not by architectural elegance. The philosophy: cheap local experiments first → insights → architecture decisions. Expensive LLM experiments are logged for later once we know which dimensions earn their keep.

### Tier 0: Data gap investigation (prerequisite)

1. **Debug relationship edge persistence** (§0) — find where the 134 edges went and ensure they're written to bibles JSON
2. **Investigate missing superobjectives** (§0) — determine whether the bible builder is failing or a flag was missing at runtime

### Tier 1: Zero-cost diagnostics (existing data, no API calls)

3. **Tactic discriminant analysis** (§1.1) — validates whether our 66-cluster vocabulary is empirically grounded; equal play weighting
4. **Affect distribution profiling** (§1.2) — violin plots per play, per character; includes testing whether certainty/control dimensions discriminate
5. **Character clustering by tactic distribution** (§2.1) — focus on wit-driven deflectors and romantic idealists as primary sanity checks; include per-scene Hamlet sub-experiment
6. **Character clustering by affect trajectory** (§2.2) — use windowed summaries (per-act) for cross-play comparison
7. **Cross-genre discriminant validity** (§2.4) — are we capturing character or genre?
8. **Transition prior estimation** (§4.1) — tactic bigrams, affect Gaussian, desire-conditioned transitions (pooled across plays)
9. **Cross-character factor estimation** (§4.2) — status correlation in multi-character scenes

### Tier 2: Low-cost data completion (~$2–5)

10. **Populate relationship edges** for all three plays (§2.3) — prerequisite for relationship pair clustering and cross-character factors
11. **Populate missing superobjectives** (§4.4) — prerequisite for arc factors
12. **Text-grounded WorldBible** for Cherry Orchard and Hamlet (§3.2) — fills the gap and enables contamination delta measurement

### Tier 3: Insight-dependent experiments (~$5–15, deferred)

These are logged for execution once Tier 1 results clarify which dimensions and structures earn their keep:

13. **Gestalt extraction** for all three plays (§3.3) — establishes top-down signal; defer until bottom-up pipeline insights are in hand
14. **Ensemble extraction calibration** (§1.3) — 90-beat sample across 3 plays; defer until we know which dimensions to calibrate
15. **Gestalt vs. bottom-up comparison** (§3.3) — quantify agreement/disagreement patterns; depends on #13
16. **Text-grounded vs. knowledge-grounded WorldBible comparison** (§3.2) — measure contamination delta; depends on #12

---

## 6. Hypotheses to Test and Deferred Experiments

Organized into testable hypotheses (cheap, local) and deferred experiments (require LLM budget). Each hypothesis includes a falsification criterion.

### Testable hypotheses (Tier 1 experiments)

**H1: Certainty and control dimensions discriminate characters and/or predict tactic choice.**
- Test: For each affect dimension, compute between-character variance vs. within-character variance across all beats. Run a logistic regression predicting tactic class from the 5D affect vector; compare AUC with and without certainty/control.
- Falsified if: Dropping certainty and control does not reduce discriminative power. In that case, consider removing them to save extraction tokens.

**H2: Wit-driven deflectors cluster across genres.**
- Test: Compute tactic distributions for Algernon, Hamlet, and Trofimov. Measure pairwise cosine similarity and compare to within-cluster similarity of other groupings (servants, dominators).
- Falsified if: These three characters are no more similar to each other than to random characters from different clusters.
- Sub-experiment: Compute tactic distributions for Hamlet per-scene (e.g., Act 2 Scene 2 vs. Act 3 Scene 1). Do scene-level Hamlet slices land in different expected clusters?

**H3: Genre systematically shifts tactic affect profiles.**
- Test: For tactics with ≥15 occurrences across the corpus, compare mean affect vectors per play. If "deflect" has significantly different affect in Earnest vs. Hamlet, that's a genre effect.
- Falsified if: Within-tactic affect profiles are consistent across plays (low between-play variance relative to within-play variance). If falsified, genre-as-covariate is unnecessary and can be dropped permanently rather than deferred.

**H4: Longer plays pace differently than shorter ones.**
- Test: Compute beat-level affect variance within fixed-length windows across plays. Compare the distribution of per-window variance in Hamlet (645 beats) vs. Earnest (191 beats).
- Falsified if: Per-window affect variance distributions are similar across plays, suggesting pacing is scale-invariant and simple normalization would suffice for trajectory comparison.

**H5: Gestalt extraction captures information not present in bottom-up pipeline.**
- Test (requires Tier 3 budget): Compare gestalt "dramaturgical first impression" to bottom-up CharacterBibles and SceneBibles. Measure agreement on character gravity rankings, arc shape, and tonal signature.
- Falsified if: Gestalt and bottom-up agree on all major dimensions — in which case the gestalt adds no information and the reconciliation step is unnecessary. If not falsified, the magnitude of disagreement determines whether gestalt should feed directly into factor graph global factors.

**H6: Desire-state changes predict tactic transitions beyond what tactic bigrams capture.**
- Test: Compute tactic transition entropy conditioned on desire-change vs. no-desire-change. Theory predicts higher entropy (more tactic diversity) when desire changes.
- Falsified if: Tactic transition distributions are indistinguishable regardless of desire state. Would suggest desire tracking adds no predictive value for the transition factor.

### Deferred experiments (require LLM budget)

**E1: Ensemble extraction calibration** (~$10–15)
- 90-beat sample, 3×Sonnet at temperatures 0.3/0.6/0.9 vs. Opus point estimates
- Deferred until H1 clarifies which dimensions to calibrate

**E2: Text-grounded vs. knowledge-grounded WorldBible delta** (~$2–5)
- Run both extraction modes on all three plays; measure what text-only analysis misses
- Depends on populating WorldBibles for Cherry Orchard and Hamlet first

**E3: Gestalt extraction for factor graph input** (~$5–8)
- Full dramaturgical gestalt for all three plays; compare to bottom-up pipeline
- Deferred until Tier 1 insights clarify whether top-down signal is architecturally needed

---

## 7. Decisions and Conventions

### Sanity-check failure protocol

When clustering experiments fail to produce expected groupings, the response is diagnostic, not corrective. Ask *why*: Do subsets of characters cluster separately? (Possibly still valid.) Are there megaclusters? (Could indicate tactic vocabulary grouping issues, data limitations, or extraction bias.) Dramaturgical theory serves as an important inductive bias given the size of our corpus and is not something we'd abandon in the short term. However, if specific aspects of the theory don't provide sufficient information gain, we may simplify the corresponding model components for now.

### Unknown play strategy: LLM-generated synthetic plays

Rather than searching for obscure published scripts, generate a series of short 1-act plays via LLM. Use a hierarchical generation approach to keep output context constrained: first generate an outline (characters, overall arc), then expand scene by scene. This approach offers several advantages over sourcing obscure published works:

- Covers a wider variety of genres, periods, and dramatic conventions on demand
- Prompts can specify novelty and uniqueness of voice to reduce the chance of mimicking a known author's style
- Scales cheaply — we can generate as many as needed for statistical power
- Guaranteed to be outside the LLM's training data (at least for extraction, though the generating LLM's biases will still be present)

This can run in parallel with Tier 1 experiments — no need to wait for full pipeline calibration.

**Ground-truth advantage**: Because generation is hierarchical (outline → scenes), the outline itself serves as ground truth for gestalt, WorldBible, character arcs, and superobjectives — something we lack entirely for the canonical plays. This enables direct evaluation of how well the bottom-up pipeline recovers top-down structure, without the contamination problem that plagues our three known plays.

### Experiment log format

Experiment results will be recorded in a structured Markdown file (`docs/EXPERIMENT_LOG.md`) with entries following this format:

- **Hypothesis**: Which hypothesis (H1–H6) or deferred experiment (E1–E3) this addresses
- **Method**: What was computed or run
- **Result**: What was observed
- **Implication**: What this means for the architecture, and how much it resolves the corresponding hypothesis

Each entry should explicitly note how much entropy remains on the relevant hypothesis — the goal is to reduce uncertainty until we are confident enough to commit to an overall design.
