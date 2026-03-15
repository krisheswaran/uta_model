"""
Improvisation Loop — Improvisation System

Orchestrates a full improv session for one character:

  For each turn:
    1. Initialize BeatState from CharacterBible + SceneContext
    2. Generate candidate line (LLM with structured control prompt)
    3. Score candidate on six axes
    4. If below threshold: send targeted feedback, regenerate (max N rounds)
    5. Update BeatState for the next turn
    6. Record the turn in an ImprovSession

The LLM generating lines receives a structured control prompt specifying:
  objective, obstacle, tactic, affect vector, relationship stance,
  epistemic state, speech-style constraints, and few-shot examples.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import GENERATION_MODEL, ANTHROPIC_API_KEY, MAX_REVISION_ROUNDS, SCORE_THRESHOLD
from schemas import (
    AffectState, BeatState, CandidateLine, CharacterBible, ImprovSession,
    ImprovTurn, SceneContext, ScoredLine, SocialState, EpistemicState,
)
from improv.scorer import score_candidate
from improv.state_updater import update_beat_state

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ────────────────────────────────────────────────────────────────────────────
# State initialization
# ────────────────────────────────────────────────────────────────────────────

_INIT_SYSTEM = """\
You are initializing the hidden dramatic state for a character about to enter
an improvised scene. Given the character bible and the new scene context, infer:
  - what the character most likely wants in this situation
  - what tactic they would default to
  - their initial affect and social state

Return ONLY valid JSON matching the BeatState schema (exclude beat_id and character):
{
  "desire_state": "...",
  "superobjective_reminder": "...",
  "obstacle": "...",
  "tactic_state": "action verb",
  "defense_state": "...",
  "psychological_contradiction": "...",
  "affect_state": {"valence": 0.0, "arousal": 0.0, "certainty": 0.0,
                   "control": 0.0, "vulnerability": 0.0, "rationale": "..."},
  "social_state": {"status": 0.0, "warmth": 0.0, "rationale": "..."},
  "epistemic_state": {"known_facts": [], "hidden_secrets": [], "false_beliefs": [],
                      "rationale": "..."}
}
"""

_INIT_USER = """\
CHARACTER BIBLE:
  Superobjective: {superobjective}
  Wounds/fears/needs: {wounds}
  Recurring tactics: {tactics}
  Defense mechanisms: {defenses}
  Psychological contradictions: {contradictions}
  Secrets: {secrets}
  Known facts: {known_facts}

NEW SCENE CONTEXT:
  Setting: {setting}
  Who is present: {present}
  Prior events: {prior_events}
  Stakes: {stakes}
  Register: {register}
  Constraint: {constraint}

Initialize the hidden state for {character}. Return JSON only.
"""


def initialize_beat_state(
    character: str,
    bible: CharacterBible,
    context: SceneContext,
    beat_id: str = "improv_b1",
) -> BeatState:
    """Infer initial hidden state for the character from the bible + scene context."""
    prompt = _INIT_USER.format(
        superobjective=bible.superobjective,
        wounds=bible.wounds_fears_needs,
        tactics=", ".join(bible.recurring_tactics),
        defenses=", ".join(bible.preferred_defense_mechanisms),
        contradictions="; ".join(bible.psychological_contradictions),
        secrets="; ".join(bible.secrets[:5]),
        known_facts="; ".join(bible.known_facts[:10]),
        setting=context.setting,
        present=", ".join(context.characters_present),
        prior_events=context.prior_events,
        stakes=context.stakes,
        register=context.dramatic_register,
        constraint=context.constraint,
        character=character,
    )
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=_INIT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    def _clamp(v, lo=-1.0, hi=1.0):
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return 0.0

    affect = data.get("affect_state", {})
    social = data.get("social_state", {})
    epistemic = data.get("epistemic_state", {})

    return BeatState(
        beat_id=beat_id,
        character=character,
        desire_state=data.get("desire_state", ""),
        superobjective_reminder=data.get("superobjective_reminder", ""),
        obstacle=data.get("obstacle", ""),
        tactic_state=data.get("tactic_state", ""),
        defense_state=data.get("defense_state", ""),
        psychological_contradiction=data.get("psychological_contradiction", ""),
        affect_state=AffectState(
            valence=_clamp(affect.get("valence", 0)),
            arousal=_clamp(affect.get("arousal", 0)),
            certainty=_clamp(affect.get("certainty", 0)),
            control=_clamp(affect.get("control", 0)),
            vulnerability=_clamp(affect.get("vulnerability", 0), lo=0.0),
            rationale=affect.get("rationale", ""),
        ),
        social_state=SocialState(
            status=_clamp(social.get("status", 0)),
            warmth=_clamp(social.get("warmth", 0)),
            rationale=social.get("rationale", ""),
        ),
        epistemic_state=EpistemicState(
            known_facts=epistemic.get("known_facts", bible.known_facts[:10]),
            hidden_secrets=epistemic.get("hidden_secrets", bible.secrets),
            false_beliefs=epistemic.get("false_beliefs", []),
            rationale=epistemic.get("rationale", ""),
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ────────────────────────────────────────────────────────────────────────────

_GENERATION_SYSTEM = """\
You are playing a theatrical character. Your job is to produce ONE line of dialogue.

You will be given:
  - Your character's full dramatic state (objective, tactic, affect, etc.)
  - The scene context
  - Your character's speech style and few-shot examples

RULES:
  - Stay true to the tactic — it is what you are DOING TO the other person
  - Do NOT be generic or polite unless that IS the tactic
  - Let the subtext live beneath the words; do not say everything you feel
  - Honour the affect vector: if vulnerability=0.8, something is breaking through
  - Keep it to 1–3 sentences unless the character tends toward longer speeches
  - Do NOT add stage directions or speaker attribution

Return JSON:
{
  "text": "the line of dialogue",
  "internal_reasoning": "the character's unspoken thought process (1-2 sentences)"
}
"""

_GENERATION_USER = """\
CHARACTER: {character}

DRAMATIC STATE:
  Want: {desire_state}
  Obstacle: {obstacle}
  Tactic (what you are DOING TO the other person): {tactic_state}
  Superobjective link: {superobjective_reminder}
  Affect: valence={valence:.1f} arousal={arousal:.1f} vulnerability={vulnerability:.1f}
  Status claim: {status:.1f} | Warmth toward partner: {warmth:.1f}
  Active defense: {defense_state}
  Inner contradiction: {contradiction}
  Secrets to protect: {secrets}

SCENE:
  Setting: {setting}
  Partner's last line: {partner_line}
  Stakes: {stakes}

SPEECH STYLE: {speech_style}

FEW-SHOT CANONICAL LINES FROM THIS CHARACTER:
{few_shot}

{feedback_block}
Generate the line now. Return JSON only.
"""


def generate_candidate(
    beat_state: BeatState,
    bible: CharacterBible,
    context: SceneContext,
    feedback: list[str] | None = None,
) -> CandidateLine:
    """Generate one candidate line using structured state as control prompt."""
    feedback_block = ""
    if feedback:
        feedback_block = (
            "REVISION NOTES — address these before generating:\n"
            + "\n".join(f"  - {f}" for f in feedback)
            + "\n"
        )

    few_shot = "\n".join(f'  "{l}"' for l in bible.few_shot_lines[:6])

    prompt = _GENERATION_USER.format(
        character=beat_state.character,
        desire_state=beat_state.desire_state,
        obstacle=beat_state.obstacle,
        tactic_state=beat_state.tactic_state,
        superobjective_reminder=beat_state.superobjective_reminder,
        valence=beat_state.affect_state.valence,
        arousal=beat_state.affect_state.arousal,
        vulnerability=beat_state.affect_state.vulnerability,
        status=beat_state.social_state.status,
        warmth=beat_state.social_state.warmth,
        defense_state=beat_state.defense_state,
        contradiction=beat_state.psychological_contradiction,
        secrets="; ".join(beat_state.epistemic_state.hidden_secrets[:3]),
        setting=context.setting,
        partner_line=context.partner_line or "(scene opening — no line yet)",
        stakes=context.stakes,
        speech_style=bible.speech_style,
        few_shot=few_shot,
        feedback_block=feedback_block,
    )

    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=512,
        system=_GENERATION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
    try:
        data = json.loads(raw)
        return CandidateLine(
            text=data.get("text", raw),
            internal_reasoning=data.get("internal_reasoning", ""),
        )
    except json.JSONDecodeError:
        return CandidateLine(text=raw)


# ────────────────────────────────────────────────────────────────────────────
# Improv session orchestrator
# ────────────────────────────────────────────────────────────────────────────

def run_turn(
    turn_index: int,
    beat_state: BeatState,
    bible: CharacterBible,
    context: SceneContext,
) -> tuple[ImprovTurn, BeatState]:
    """
    Run one improv turn: generate → score → (revise) → update state.
    Returns the completed ImprovTurn and the updated BeatState for the next turn.
    """
    initial_state = beat_state.model_copy(deep=True)
    feedback: list[str] = []
    scored: ScoredLine | None = None
    final_candidate = None

    for revision_round in range(MAX_REVISION_ROUNDS + 1):
        candidate = generate_candidate(beat_state, bible, context, feedback or None)
        scored = score_candidate(candidate, beat_state, bible, context)

        if scored.passed or revision_round == MAX_REVISION_ROUNDS:
            final_candidate = candidate
            break

        # Collect feedback for next round
        feedback = scored.feedback or [
            f"Axis scores: voice={scored.voice_fidelity:.1f} "
            f"tactic={scored.tactic_fidelity:.1f} "
            f"knowledge={scored.knowledge_fidelity:.1f} "
            f"relationship={scored.relationship_fidelity:.1f} "
            f"subtext={scored.subtext_richness:.1f} "
            f"affect={scored.emotional_transition_plausibility:.1f}. "
            "Revise to improve the lowest-scoring dimensions."
        ]

    updated_state = update_beat_state(
        beat_state,
        final_candidate.text,
        partner_line=context.partner_line or "",
    )

    turn = ImprovTurn(
        turn_index=turn_index,
        context=context,
        initial_beat_state=initial_state,
        final_line=final_candidate.text,
        revisions=revision_round,
        scored_line=scored,
        updated_beat_state=updated_state,
    )
    return turn, updated_state


def run_session(
    bible: CharacterBible,
    scene_prompts: list[SceneContext],
    session_id: str | None = None,
) -> ImprovSession:
    """
    Run a full improv session across a list of scene prompts.
    Each prompt is one turn; the state carries over between turns.
    """
    session_id = session_id or str(uuid.uuid4())[:8]
    session = ImprovSession(
        session_id=session_id,
        character=bible.character,
        character_bible=bible,
    )

    beat_state = initialize_beat_state(
        bible.character, bible, scene_prompts[0], beat_id=f"{session_id}_b1"
    )

    for turn_index, context in enumerate(scene_prompts):
        print(f"  Turn {turn_index + 1}: {context.partner_line or '(opening)'}")
        turn, beat_state = run_turn(turn_index + 1, beat_state, bible, context)
        session.turns.append(turn)
        print(f"    → {turn.final_line[:80]}... "
              f"(score={turn.scored_line.mean_score:.2f}, revisions={turn.revisions})")
        # Carry partner context forward
        if turn_index + 1 < len(scene_prompts) and not scene_prompts[turn_index + 1].partner_line:
            scene_prompts[turn_index + 1] = scene_prompts[turn_index + 1].model_copy(
                update={"partner_line": turn.final_line}
            )

    return session
