"""
Core Pydantic schemas for the UTA Acting System.

Hierarchy:
  Play → Act → Scene → Beat → Utterance
  Play → CharacterBible[]
  Play → RelationshipEdge[]
  Beat → BeatState[]  (one per character present)

Pass 1 output: Play object fully populated with BeatStates + Bibles.
Pass 2 input:  CharacterBible + SceneContext → improvised lines.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Utterance — atomic unit of dialogue
# --------------------------------------------------------------------------- #

class Utterance(BaseModel):
    id: str                              # e.g. "ham_1_2_u3"
    play_id: str
    act: int
    scene: int
    index: int                           # position within the scene
    speaker: str                         # normalized uppercase name
    text: str
    stage_direction: Optional[str] = None  # inline stage direction if present
    addressee: Optional[str] = None      # estimated addressee (may be None)


# --------------------------------------------------------------------------- #
# Affect state — Russell circumplex + extra dimensions
# --------------------------------------------------------------------------- #

class AffectState(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0,
                           description="Negative-to-positive (-1 to 1)")
    arousal: float = Field(0.0, ge=-1.0, le=1.0,
                           description="Calm-to-activated (-1 to 1)")
    certainty: float = Field(0.0, ge=-1.0, le=1.0,
                             description="Uncertain-to-certain (-1 to 1)")
    control: float = Field(0.0, ge=-1.0, le=1.0,
                           description="Powerless-to-dominant (-1 to 1)")
    vulnerability: float = Field(0.0, ge=0.0, le=1.0,
                                 description="0 = guarded, 1 = exposed")
    rationale: str = ""


class SocialState(BaseModel):
    status: float = Field(0.0, ge=-1.0, le=1.0,
                          description="Low-to-high social status claim (-1 to 1)")
    warmth: float = Field(0.0, ge=-1.0, le=1.0,
                          description="Hostile-to-warm toward addressee (-1 to 1)")
    rationale: str = ""


class EpistemicState(BaseModel):
    known_facts: list[str] = Field(default_factory=list,
                                   description="Facts the character knows at this beat")
    hidden_secrets: list[str] = Field(default_factory=list,
                                      description="Facts character is concealing")
    false_beliefs: list[str] = Field(default_factory=list,
                                     description="Things character believes that are wrong")
    rationale: str = ""


# --------------------------------------------------------------------------- #
# BeatState — full factored hidden state for one character at one beat
# --------------------------------------------------------------------------- #

class BeatState(BaseModel):
    beat_id: str
    character: str

    # Factored state components
    desire_state: str = Field("", description="What the character wants right now (scene want)")
    superobjective_reminder: str = Field("", description="How this want connects to their superobjective")
    obstacle: str = Field("", description="What is blocking them")
    tactic_state: str = Field("", description="Action verb: what they're doing TO the other person")
    canonical_tactic: Optional[str] = Field(None, description="Canonical tactic ID from vocabulary (Phase B)")
    affect_state: AffectState = Field(default_factory=AffectState)
    social_state: SocialState = Field(default_factory=SocialState)
    epistemic_state: EpistemicState = Field(default_factory=EpistemicState)
    defense_state: str = Field("", description="Active defense mechanism if any")
    psychological_contradiction: str = Field("", description="Any active internal contradiction")

    # Confidence / ambiguity
    confidence: float = Field(1.0, ge=0.0, le=1.0,
                              description="Confidence in this interpretation")
    alternative_hypothesis: str = Field("", description="Alternative reading if ambiguous")

    class Config:
        json_schema_extra = {
            "example": {
                "beat_id": "cherry_1_1_b2",
                "character": "LOPAKHIN",
                "desire_state": "Convince Ranyevskaya to accept the dacha plan",
                "obstacle": "Her sentimental attachment to the estate makes her deaf to practicality",
                "tactic_state": "persuade",
                "defense_state": "intellectualization",
                "psychological_contradiction": "Wants to save her while also wanting to own what she owns",
            }
        }


# --------------------------------------------------------------------------- #
# Beat — a unit of consistent dramatic action
# --------------------------------------------------------------------------- #

class Beat(BaseModel):
    id: str                              # e.g. "cherry_1_1_b3"
    play_id: str
    act: int
    scene: int
    index: int                           # beat number within scene
    utterances: list[Utterance] = Field(default_factory=list)
    beat_states: list[BeatState] = Field(default_factory=list)
    beat_summary: str = ""               # brief description of what changes in this beat
    characters_present: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Scene, Act, Play
# --------------------------------------------------------------------------- #

class SceneBible(BaseModel):
    play_id: str
    act: int
    scene: int
    dramatic_pressure: str = ""
    what_changes: str = ""
    hidden_tensions: str = ""
    beat_map: str = ""                   # prose summary of beat progression


class WorldBible(BaseModel):
    play_id: str
    era: str = ""
    genre: str = ""
    social_norms: list[str] = Field(default_factory=list)
    factual_timeline: list[str] = Field(default_factory=list)
    genre_constraints: list[str] = Field(default_factory=list)


class CharacterBible(BaseModel):
    play_id: str
    character: str

    # Enduring psychology
    superobjective: str = ""
    wounds_fears_needs: str = ""
    recurring_tactics: list[str] = Field(default_factory=list)
    preferred_defense_mechanisms: list[str] = Field(default_factory=list)
    psychological_contradictions: list[str] = Field(default_factory=list)

    # Voice
    speech_style: str = ""
    lexical_signature: list[str] = Field(default_factory=list,
                                         description="Characteristic words/phrases")
    rhetorical_patterns: list[str] = Field(default_factory=list)
    few_shot_lines: list[str] = Field(default_factory=list,
                                      description="Canonical lines for voice reference")

    # Knowledge & relationships
    known_facts: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)

    # Arc
    arc_by_scene: dict[str, str] = Field(default_factory=dict,
                                         description="scene_id → brief arc note")

    # Derived statistics
    tactic_distribution: dict[str, int] = Field(default_factory=dict,
                                                description="tactic → count across play")


class RelationshipEdge(BaseModel):
    play_id: str
    character_a: str
    character_b: str
    # temperature over beats: beat_id → warmth float
    temperature_by_beat: dict[str, float] = Field(default_factory=dict)
    power_by_beat: dict[str, float] = Field(default_factory=dict,
                                            description="A's power over B at each beat")
    summary: str = ""


class Scene(BaseModel):
    id: str                              # e.g. "cherry_1_1"
    play_id: str
    act: int
    scene: int
    beats: list[Beat] = Field(default_factory=list)
    bible: Optional[SceneBible] = None


class Act(BaseModel):
    id: str
    play_id: str
    number: int
    scenes: list[Scene] = Field(default_factory=list)


class Play(BaseModel):
    id: str                              # e.g. "cherry_orchard"
    title: str
    author: str
    acts: list[Act] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    character_bibles: list[CharacterBible] = Field(default_factory=list)
    scene_bibles: list[SceneBible] = Field(default_factory=list)
    world_bible: Optional[WorldBible] = None
    relationship_edges: list[RelationshipEdge] = Field(default_factory=list)

    def get_character_bible(self, character: str) -> Optional[CharacterBible]:
        for cb in self.character_bibles:
            if cb.character.upper() == character.upper():
                return cb
        return None

    def get_scene_bible(self, act: int, scene: int) -> Optional[SceneBible]:
        for sb in self.scene_bibles:
            if sb.act == act and sb.scene == scene:
                return sb
        return None

    def iter_utterances(self):
        for act in self.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    yield from beat.utterances


# --------------------------------------------------------------------------- #
# Pass 2 — inference-time schemas
# --------------------------------------------------------------------------- #

class SceneContext(BaseModel):
    """Input to the Pass 2 improv loop for a single scene."""
    play_id: str
    character: str                       # which character we are generating for
    setting: str                         # description of the new situation
    characters_present: list[str]        # all characters in the scene
    prior_events: str                    # what happened just before
    stakes: str                          # what's at risk
    partner_line: Optional[str] = None   # most recent line from the other character
    dramatic_register: str = "dramatic"  # comedic / tragic / dramatic / tragicomic
    constraint: str = "alternate_universe_same_psyche"
    # Options: faithful_to_arc | alternate_universe_same_psyche | post_play_continuation


class CandidateLine(BaseModel):
    text: str
    internal_reasoning: str = ""         # the character's unspoken thought process


class ScoredLine(BaseModel):
    candidate: CandidateLine
    voice_fidelity: float = Field(ge=1.0, le=5.0)
    tactic_fidelity: float = Field(ge=1.0, le=5.0)
    knowledge_fidelity: float = Field(ge=1.0, le=5.0)
    relationship_fidelity: float = Field(ge=1.0, le=5.0)
    subtext_richness: float = Field(ge=1.0, le=5.0)
    emotional_transition_plausibility: float = Field(ge=1.0, le=5.0)
    feedback: list[str] = Field(default_factory=list)  # targeted state-based notes

    @property
    def mean_score(self) -> float:
        dims = [
            self.voice_fidelity, self.tactic_fidelity, self.knowledge_fidelity,
            self.relationship_fidelity, self.subtext_richness,
            self.emotional_transition_plausibility,
        ]
        return sum(dims) / len(dims)

    @property
    def passed(self) -> bool:
        from config import SCORE_THRESHOLD
        return self.mean_score >= SCORE_THRESHOLD


class RevisionTrace(BaseModel):
    """One round of the revision loop: candidate, scores, and feedback given."""
    round: int
    candidate_text: str
    scores: dict[str, float] = Field(default_factory=dict,
                                     description="axis → score for this round")
    feedback: list[str] = Field(default_factory=list,
                                description="Feedback notes (including dramaturgical) for this round")


class ImprovTurn(BaseModel):
    turn_index: int
    context: SceneContext
    initial_beat_state: BeatState
    final_line: str
    revisions: int
    scored_line: ScoredLine
    updated_beat_state: BeatState
    revision_trace: list[RevisionTrace] = Field(default_factory=list,
                                                description="Full trace of all revision rounds")


class ImprovSession(BaseModel):
    session_id: str
    character: str
    character_bible: CharacterBible
    turns: list[ImprovTurn] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase B — Statistical Learning schemas
# --------------------------------------------------------------------------- #

class CanonicalTactic(BaseModel):
    """A canonical tactic in the vocabulary, grouping synonymous raw tactic strings."""
    canonical_id: str                    # e.g. "DEFLECT"
    canonical_verb: str                  # e.g. "deflect"
    description: str                     # acting-theory definition sentence
    members: list[str] = Field(default_factory=list,
                               description="Raw tactic strings that map to this canonical")
    category: str = ""                   # optional super-category (e.g. "avoidance")


class TacticVocabulary(BaseModel):
    """The canonical tactic vocabulary, built by clustering observed tactics."""
    version: int = 1
    tactics: list[CanonicalTactic] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list,
                                description="New tactic strings awaiting cluster assignment")

    def lookup(self, raw_tactic: str) -> Optional[str]:
        """Return canonical_id for a raw tactic string, or None if unmapped."""
        raw_lower = raw_tactic.lower().strip()
        for ct in self.tactics:
            if raw_lower in [m.lower() for m in ct.members]:
                return ct.canonical_id
        return None


class BeatStateEstimate(BaseModel):
    """Multiple extraction estimates for a single character at a single beat (Phase B ensemble)."""
    beat_id: str
    character: str
    estimates: list[BeatState] = Field(default_factory=list)
    model_ids: list[str] = Field(default_factory=list)
    temperatures: list[float] = Field(default_factory=list)
    tactic_posterior: dict[str, float] = Field(default_factory=dict,
                                               description="canonical_tactic → probability")
    affect_mean: Optional[AffectState] = None
    affect_std: dict[str, float] = Field(default_factory=dict,
                                         description="Per affect dimension std dev")
    consensus_confidence: float = Field(1.0, ge=0.0, le=1.0)


class RelationalProfile(BaseModel):
    """Character-level social tendencies aggregated across all interaction partners.

    Directed: describes how THIS character relates to others, not how others relate to them.
    """
    character: str
    play_id: str
    default_status_claim: float = Field(0.0, description="Mean status across all interactions")
    default_warmth: float = Field(0.0, description="Mean warmth across all interactions")
    status_variance: float = Field(0.0, description="How much status varies by partner")
    warmth_variance: float = Field(0.0, description="How much warmth varies by partner")
    partner_deviations: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="partner → {status_delta, warmth_delta} deviation from default"
    )


class StatisticalPrior(BaseModel):
    """Learned priors from corpus analysis, loaded at improv time for one character."""
    tactic_vocabulary: TacticVocabulary = Field(default_factory=TacticVocabulary)
    character_tactic_prior: dict[str, float] = Field(
        default_factory=dict, description="canonical_id → P(tactic | character)")
    tactic_transition_matrix: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="P(next_tactic | current_tactic)")
    relational_profile: Optional[RelationalProfile] = None


# --------------------------------------------------------------------------- #
# Evaluation schemas
# --------------------------------------------------------------------------- #

class JudgeRating(BaseModel):
    """LLM-as-judge rating for a single generated line."""
    line: str
    tier: str                            # "vanilla" | "bible" | "reflection"
    character: str
    scene_context: str

    recognizability: float = Field(ge=1.0, le=5.0)
    playability: float = Field(ge=1.0, le=5.0)
    tactic_fidelity: float = Field(ge=1.0, le=5.0)
    subtext: float = Field(ge=1.0, le=5.0)
    earned_affect: float = Field(ge=1.0, le=5.0)
    knowledge_fidelity_pass: bool = True
    knowledge_fidelity_note: str = ""
    identified_tactic: str = ""          # judge's open-text tactic identification

    @property
    def mean_score(self) -> float:
        dims = [self.recognizability, self.playability, self.tactic_fidelity,
                self.subtext, self.earned_affect]
        kf = 5.0 if self.knowledge_fidelity_pass else 1.0
        return (sum(dims) + kf) / (len(dims) + 1)
