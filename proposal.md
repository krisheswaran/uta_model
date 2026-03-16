Acting System Proposal

## **High-level idea**

Treat the play as a multi-scale latent dynamical system:

* Global play state: era, world constraints, genre pressure, social structure, prior events, stakes

* Scene state: who is present, what just changed, tension level, immediate conflict, subtextual frame

* Character state for each character:

  * objective / superobjective

  * scene want

  * obstacle

  * tactic / action verb

  * emotional valence / arousal / regulation

  * knowledge / beliefs / secrets

  * relationship stance toward each other character

  * self-presentation mask vs private state

* Utterance-level realization: wording, rhythm, interruption style, lexical choices, silence, evasion, etc.

So the dialogue is not just text generation. It is the surface emission of hidden dramatic states.

## **Architecture in two passes**

### **Pass 1: build the dramatic world and character models from the script**

This is an offline analysis pass over the corpus.

#### **1\. Parse the script into a structured representation**

Represent:

* acts / scenes / beats

* entrances / exits

* speaker turns

* stage directions

* addressee estimates

* interruptions / overlaps / pauses if available

* references to offstage events

* physical actions

* props / locations / time

Output a graph-like structure:

* Play

* Scene

* Beat

* Utterance

* Character

* RelationshipEdge

* WorldFact

* BeliefState

* ObjectiveHypothesis

A beat is important here. You do not want to model only at line level. A lot of acting analysis lives at the beat level.

#### **2\. Construct an ontology of hidden dramatic variables**

This is where Stanislavsky / Hagen / Meisner become computable.

For each character and beat, infer latent fields such as:

* Want: what do they want right now?

* Obstacle: what blocks them?

* Tactic / action: to seduce, deflect, shame, reassure, provoke, test, conceal, dominate, plead…

* Emotion: not just label, but vector

  * valence

  * arousal

  * certainty

  * control

  * vulnerability

* Given circumstances

* Inner object / imagined partner framing

* Relationship temperature

* Status / power

* Truthfulness / concealment

* Knowledge state

* Commitment strength

* Urgency

* Psychological contradiction: e.g. “wants intimacy while defending against it”

I would keep these as a mix of:

* categorical variables

* continuous variables

* structured text summaries

#### **3\. Use layered inference, not one-shot labeling**

Do not ask one model once, “what is Hamlet feeling?”

Use a staged pipeline:

1. Literal extraction

   * who says what

   * what claims are made

   * what actions happen

   * who knows what

2. Beat segmentation

   * where objectives or tactics shift

3. Local latent inference

   * per utterance / per beat

4. Temporal smoothing

   * infer state trajectories over scenes

5. Global consistency optimization

   * enforce character-level coherence across the play

This is where your HMM intuition fits well.

### **Suggested probabilistic structure**

A plain HMM is too weak, but a hierarchical switching state-space model or dynamic Bayesian network is a good fit.

Something like:

* Z\_play(t) \= global dramatic regime

* Z\_scene(s) \= scene regime

* Z\_char(c,t) \= character hidden state at time t

* R(c1,c2,t) \= relationship state

* K(c,t) \= knowledge / belief state

* U(t) \= observed utterance

With dependencies like:

* Z\_char(c,t) depends on:

  * previous Z\_char(c,t-1)

  * current scene state

  * relationship states

  * what others just said

* U(t) emitted from:

  * current speaker’s hidden state

  * addressee relation

  * style prior of the character

This gets you a principled latent-state model.

#### **4\. Add an optimization layer for long-horizon rigor**

This is the key part.

Local line-by-line interpretations are noisy. So define a scoring objective over the whole play:

* Emission fit: do the inferred hidden states explain the words and actions?

* Temporal coherence: do state transitions make sense?

* Character coherence: does the character behave consistently with an evolving but legible arc?

* Relationship coherence: do changes in relationships track events?

* Acting-theory constraints:

  * people usually act through tactics toward objectives

  * emotional flips should usually be motivated

  * contradictions are allowed, but should recur in recognizable patterns

* Script-specific evidence:

  * stage directions

  * repeated motifs

  * callbacks

  * lexical patterns

Then do iterative refinement:

* initialize with LLM analyses

* refine with dynamic programming / belief propagation / variational inference / beam search over latent trajectories

* re-score globally

* re-ask the LLM only on ambiguous spans

So the LLM is a proposal generator plus semantic evaluator, while the algorithmic layer enforces consistency over long horizons.

#### **5\. Learn character-specific emission models**

Each character should get a distinct model.

For each character build:

* lexical profile

* syntactic profile

* rhetorical profile

* turn-taking profile

* interruption profile

* metaphor profile

* emotional transition tendencies

* tactic distribution

* preferred defense mechanisms

* relationship-conditioned language shifts

For example:

* Character A may conceal via irony

* Character B may escalate via blunt repetition

* Character C may seek reassurance through questions disguised as accusations

This becomes the character’s “embodiment prior.”

#### **6\. Build explicit memory objects**

At the end of pass 1, produce:

### **Character bible**

* superobjective

* enduring wounds / fears / needs

* recurring tactics

* typical contradictions

* speech style

* known facts / secrets

* relationship map

* arc by act / scene

### **Scene bible**

* objective dramatic pressure

* what changes in the scene

* hidden tensions

* beat map

### **World bible**

* social norms

* genre constraints

* factual timeline

### **Formal latent state archive**

A machine-readable version:

* hidden states by beat

* state transition probabilities

* confidence intervals

* alternative hypotheses where ambiguity remains

That last part matters. Good acting analysis often admits multiple readings.

---

## **Pass 2: augmented reflection-based improvisation system**

Now assume you want the character to improvise in a new setting.

The goal is not “generate lines that sound similar.”

The goal is “generate lines consistent with the latent dramatic engine of that character.”

### **Inference-time loop**

#### **Step 1\. Define the new scene context**

Input:

* new setting

* who is present

* what happened just before

* stakes

* what the character wants now

* optional constraints:

  * comedic / tragic register

  * faithful to canonical arc

  * alternate-universe but same psyche

  * post-play continuation vs in-play insertion

#### **Step 2\. Initialize the hidden state**

Use the pass-1 character model to infer:

* what would this character likely want here?

* what would threaten them?

* which tactics would they default to?

* how would they frame the other person?

* what emotional contradiction is likely activated?

This produces an inferred initial latent state:

Z\_char\*, R\*, K\*, etc.

#### **Step 3\. Generate a candidate response with the LLM**

Prompt the LLM not only with backstory text, but with structured control state:

* objective

* obstacle

* tactic

* emotional vector

* relationship stance

* secrets being protected

* known facts

* speech-style constraints

* beat purpose

#### **Step 4\. Run reflection / adjudication against the character model**

Now score the candidate along several axes:

* Voice fidelity

* Objective fidelity

* Tactic fidelity

* Relationship fidelity

* Knowledge fidelity

* Arc plausibility

* Emotional transition plausibility

* Subtext richness

* Canon contradiction penalty

This can be a weighted ensemble:

* discriminative classifier

* character-specific style model

* LLM critic

* latent-state consistency checker

* rule-based canon checker

#### **Step 5\. Revise by targeted feedback, not generic “improve this”**

Feedback should be state-based, e.g.:

* “This line is too direct; the character usually deflects before admitting vulnerability.”

* “The tactic shifted from testing to pleading without trigger.”

* “The character reveals knowledge they should not have.”

* “Lexically this sounds right, but status behavior is too low.”

* “Emotional intensity is too high for this phase.”

Then regenerate.

#### **Step 6\. Update latent state after each turn**

Improvisation should be stateful.

After each generated utterance:

* update scene state

* update speaker state

* update listener state

* update relationship state

* track beat shift

Then generate the next turn with the updated state.

This makes the improv session a controlled rollout through latent dramatic space.

---

## **A practical implementation path**

## **Phase A: symbolic \+ LLM hybrid prototype**

Start simple.

### **Representations**

Use Pydantic-like schemas or JSON for:

* UtteranceAnalysis

* BeatAnalysis

* CharacterState

* RelationshipState

* SceneState

### **Pipeline**

1. parse script

2. segment beats

3. run LLM extractors

4. store candidate states

5. run smoothing / consistency passes

6. build character bible

7. build inference-time reflection loop

This alone could work surprisingly well.

## **Phase B: statistical learning over a corpus**

If you have many plays or scenes:

* cluster tactics

* learn transition priors

* train character/state classifiers

* learn emission models for style

* train retrieval over analogous beats

Then your inference system can say:

“this new moment resembles beats 14, 52, and 89 where the character masked vulnerability with wit.”

## **Phase C: richer dynamical model**

Move to a learned latent model:

* hierarchical HMM

* HSMM if beat durations matter

* switching linear dynamical systems if you want continuous state

* POMDP framing if you want agents choosing tactics under partial observability

* factor graph over characters and relationships

For acting logic, a multi-agent POMDP flavored model is actually very natural:

* each character has private beliefs and objectives

* each observes imperfectly

* each chooses tactics

* utterances are actions that affect others’ beliefs and relationships

That may fit even better than an HMM.

---

## **What to optimize for**

You probably want four separate objectives:

1. Interpretive rigor

   * consistent hidden-state analysis over the source text

2. Embodiment fidelity

   * generated improv feels like the same character

3. Dramatic usefulness

   * outputs are playable by an actor, not just analytically plausible

4. Controlled variation

   * character remains recognizable in new contexts, without becoming a parody

That last one is crucial. Too much fidelity becomes imitation; too little becomes generic dialogue.

---

## **Evaluation**

You’ll need both intrinsic and human evaluation.

### **Intrinsic**

* next-utterance likelihood under character-conditioned model

* hidden-state consistency score

* canon contradiction count

* style distance to source character

* retrieval agreement with similar canonical beats

### **Human**

Ask actors / dramaturgs / directors to rate:

* is this character recognizable?

* is the objective playable?

* does the tactic track?

* is the subtext alive?

* does the emotional shift feel earned?

* would this be useful in rehearsal?

You might also do blind comparisons:

* baseline LLM

* LLM \+ text-only character bible

* LLM \+ latent-state reflection system

### Training data contamination caveat

Well-known characters (Hamlet, major Chekhov roles) present a contamination problem: the LLM has likely seen extensive character analyses during pretraining, inflating baseline performance and making it hard to attribute improvements to the system rather than memorized knowledge. The evaluation must include **low-familiarity characters** (e.g., minor one-acts, obscure plays) and ideally a **synthetic character** (hand-authored bible, no canonical source) as controls. The strongest evidence for the system comes from the delta between evaluation tiers on characters the LLM cannot already "play" from pretraining alone.

---

## **Important modeling choice**

I would not force “emotion / want / action” into one monolithic hidden state.

Use a factored state:

* desire state

* affect state

* social state

* epistemic state

* tactic state

* self-concept / defense state

Why?

Because actors often keep one thing constant while another changes:

* same want, new tactic

* same tactic, rising panic

* same words, different relationship temperature

A factored model is much more interpretable and controllable.

---

## **Minimal viable system**

If you wanted an MVP, I’d do this:

### **Offline**

* parse play

* beat segmentation

* LLM extracts for each beat:

  * objective

  * obstacle

  * tactic

  * emotion

  * relationship shift

  * revealed knowledge

* global refinement pass over each character arc

* build character bibles \+ state transition priors

### **Online**

* new scene context in

* infer initial hidden state from character bible \+ context

* generate candidate line

* reflect with:

  * voice checker

  * tactic checker

  * knowledge checker

  * relationship checker

* revise 2–4 rounds

* update state

* continue

That is probably enough to demonstrate the idea.

---

## **Key ambiguities / questions**

These are the main points where design choices matter:

1. What is the target use case?

ANSWER:

*  a literary analysis engine (outputs should include separable and modular models for ( character ), ( relationship ), and ( environment i.e. play, scene, world) , etc.)... each of these should be understandable

* improv bot \- a system that when given character models and some text prompts, can improvise a scene. Specificity can be at the level of (text-only model bible) or actual models for each components, resulting in a more involved system.

2. What is the corpus?

ANSWER: one play at a time… once enough plays are built up,

3. How much do you want the model to generalize?
   ANSWER: for now, preserve one specific character in alternate situations. We don’t have enough data to learn reusable dramatic priors across many characters, so we will need to incorporate certain baseline inductive biases.

4. How formal do you want the latent variables to be?

ANSWER: Not sure… walk me through the tradeoffs of each of these options and their consequences

* mostly symbolic labels with text rationales

  * probabilistic latent states

  * fully learned embeddings constrained by symbolic heads

5. Do you want strict canon fidelity or creative elasticity?

   ANSWER: prefer creative elasticity, e.g. “What would a Hamlet-like psychology say?”, strict canon fidelity like iambic pentameter could likely be reproduced intuitively by the LLM itself if given a sampling of few shot typical phrases of the character, so doesn’t need to be directly modeled beyond collection of the few shot phrases.

6. Should the model optimize for literary truth or performative truth?

   ANSWER: playable objective

7. Do you want character models to include bodily behavior and silence?

   ANSWER: Inputs and Output will be a text corpus, which may not include bodily behavior and silence, so this seems unlikely to be easily captured.

8. How important is pairwise chemistry?

   ANSWER: At inference time, we may pair characters with characters with whom they had no prior relationship, so this isn’t important in an MVP. Once we’ve built up a large enough corpus, we may bring characters together, mix and match, so that could be a longer term target.

9. Do you want a single best interpretation or maintained ambiguity?

   ANSWER: If the model is a Bayesian factor graph, the ambiguity would come from the dynamics of the model and whether one is choosing the “maximum likelihood” estimator or something else. Since the output would be the models, the ambiguity should be captured in the models, but we should also be able to recover the single best interpretation implicitly via some kind of maximum likelihood.

10. At inference time, is the LLM generating only one character’s lines, or the full scene including partners?

    ANSWER:The LLM would only be responsible for one character’s lines.

# Assessment

* offline: factor-graph / hierarchical latent analysis of dramatic states

* online: constrained multi-step generation with structured self-critique against the latent character model

# Next steps

Review tradeoffs in question 5, and once we have resolved that, turn this into a more formal system design with concrete schemas, algorithms, and a training/inference loop.