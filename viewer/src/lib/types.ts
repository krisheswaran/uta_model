// ─── Affect / Social / Epistemic State ──────────────────────────────────────

export interface AffectState {
  valence: number;       // [-1, 1]  negative → positive
  arousal: number;       // [-1, 1]  calm → activated
  certainty: number;     // [-1, 1]  uncertain → certain
  control: number;       // [-1, 1]  powerless → dominant
  vulnerability: number; // [0, 1]   guarded → exposed
  rationale: string;
}

export interface SocialState {
  status: number;  // [-1, 1]
  warmth: number;  // [-1, 1]
  rationale: string;
}

export interface EpistemicState {
  known_facts: string[];
  hidden_secrets: string[];
  false_beliefs: string[];
  rationale: string;
}

// ─── Beat ────────────────────────────────────────────────────────────────────

export interface BeatState {
  beat_id: string;
  character: string;
  desire_state: string;
  superobjective_reminder: string;
  obstacle: string;
  tactic_state: string;
  affect_state: AffectState;
  social_state: SocialState;
  epistemic_state: EpistemicState;
  defense_state: string;
  psychological_contradiction: string;
  confidence: number; // [0, 1]
  alternative_hypothesis: string;
}

export interface Utterance {
  id: string;
  play_id: string;
  act: number;
  scene: number;
  index: number;
  speaker: string;
  text: string;
  stage_direction: string | null;
  addressee: string | null;
}

export interface Beat {
  id: string;
  play_id: string;
  act: number;
  scene: number;
  index: number;
  utterances: Utterance[];
  beat_states: BeatState[];
  beat_summary?: string;
  characters_present?: string[];
}

// ─── Structure ───────────────────────────────────────────────────────────────

export interface Scene {
  id: string;
  play_id: string;
  act: number;
  scene: number;
  beats: Beat[];
}

export interface Act {
  id: string;
  play_id: string;
  number: number;
  scenes: Scene[];
}

// ─── Bibles ──────────────────────────────────────────────────────────────────

export interface CharacterBible {
  play_id: string;
  character: string;
  superobjective: string;
  wounds_fears_needs: string;
  recurring_tactics: string[];
  preferred_defense_mechanisms: string[];
  psychological_contradictions: string[];
  speech_style: string;
  lexical_signature: string[];
  rhetorical_patterns: string[];
  few_shot_lines: string[];
  known_facts: string[];
  secrets: string[];
  arc_by_scene: Record<string, string>;
  tactic_distribution: Record<string, number>;
}

export interface SceneBible {
  play_id: string;
  act: number;
  scene: number;
  dramatic_pressure: string;
  what_changes: string;
  hidden_tensions: string;
  beat_map: string;
}

export interface WorldBible {
  play_id: string;
  era: string;
  genre: string;
  social_norms: string[];
  factual_timeline: string[];
  genre_constraints: string[];
}

export interface RelationshipEdge {
  play_id: string;
  character_a: string;
  character_b: string;
  trajectory: string;
}

// ─── Play ─────────────────────────────────────────────────────────────────────

export interface Play {
  id: string;
  title: string;
  author: string;
  characters: string[];
  acts: Act[];
  character_bibles: CharacterBible[];
  scene_bibles: SceneBible[];
  world_bible: WorldBible | null;
  relationship_edges: RelationshipEdge[];
}

// ─── Index ────────────────────────────────────────────────────────────────────

export interface PlayIndexEntry {
  id: string;
  title: string;
  author: string;
  characters: string[];
  actCount: number;
}

export interface PlayIndex {
  plays: PlayIndexEntry[];
}

// ─── Tactic Vocabulary ────────────────────────────────────────────────────────

export interface TacticEntry {
  canonical_id: string;   // e.g. "DEFLECT"
  canonical_verb: string; // e.g. "deflect"
  description: string;
  members: string[];      // synonym verbs that map here
  category: string;
}

export interface TacticVocabulary {
  version: number;
  tactics: TacticEntry[];
  unmapped: string[];
}

// ─── Relational Profiles ──────────────────────────────────────────────────────

export interface PartnerDeviation {
  warmth_delta: number;
  status_delta: number;
}

export interface RelationalProfile {
  character: string;
  play_id: string;
  default_status_claim: number;
  default_warmth: number;
  status_variance: number;
  warmth_variance: number;
  partner_deviations: Record<string, PartnerDeviation>;
}

// ─── Improv Session ───────────────────────────────────────────────────────────

export interface ImprovCharacterInfo {
  character: string;
  play_id: string;
  has_prior: boolean;
  top_tactic: string;
  top_tactic_pct: number;
  default_warmth: number;
  default_status: number;
}

export interface RevisionScores {
  voice_fidelity: number;
  tactic_fidelity: number;
  knowledge_fidelity: number;
  relationship_fidelity: number;
  subtext_richness: number;
  emotional_transition_plausibility: number;
}

export interface RevisionTrace {
  round: number;
  candidate_text: string;
  scores: RevisionScores;
  feedback: string[];
}

export interface ScoredLine extends RevisionScores {
  candidate: { text: string; internal_reasoning: string };
  feedback: string[];
}

export interface ImprovTurnContext {
  play_id: string;
  character: string;
  setting: string;
  characters_present: string[];
  prior_events: string;
  stakes: string;
  partner_line: string | null;
  dramatic_register: string;
  constraint: string;
}

export interface ImprovTurn {
  turn_index: number;
  context: ImprovTurnContext;
  initial_beat_state: BeatState;
  final_line: string;
  revisions: number;
  scored_line: ScoredLine;
  updated_beat_state: BeatState;
  revision_trace: RevisionTrace[];
}

export interface TranscriptLine {
  speaker: string;
  line: string;
  tactic?: string;
  mean_score?: number;
}

export interface ImprovSession {
  scene_id: string;
  mode: string;
  timestamp: string;
  setting: string;
  stakes: string;
  prior_events: string;
  dramatic_register: string;
  constraint: string;
  characters: ImprovCharacterInfo[];
  config: {
    cli_args: Record<string, unknown>;
    model_configs: Record<string, { provider: string; model: string }>;
    pipeline_params: {
      max_revision_rounds: number;
      min_revision_rounds: number;
      score_threshold: number;
    };
  };
  turns: ImprovTurn[];
  transcript: TranscriptLine[];
}

export interface ImprovSessionMeta {
  scene_id: string;
  mode: string;
  timestamp: string;
  character: string;
  play_id: string;
  setting: string;
  turn_count: number;
  mean_score: number;
}

export interface ImprovIndex {
  sessions: ImprovSessionMeta[];
}
