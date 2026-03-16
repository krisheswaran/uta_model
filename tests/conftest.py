"""Shared fixtures for UTA test suite."""
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so all imports resolve
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_beat_state():
    """A minimal valid BeatState for testing."""
    from schemas import AffectState, BeatState, EpistemicState, SocialState
    return BeatState(
        beat_id="test_1_1_b1",
        character="TEST_CHAR",
        desire_state="To understand the situation",
        superobjective_reminder="To find truth",
        obstacle="Confusion",
        tactic_state="probe",
        affect_state=AffectState(valence=0.2, arousal=0.5, certainty=-0.3,
                                 control=0.1, vulnerability=0.4, rationale="testing"),
        social_state=SocialState(status=0.0, warmth=0.3, rationale="neutral"),
        epistemic_state=EpistemicState(
            known_facts=["fact1"], hidden_secrets=["secret1"],
            false_beliefs=[], rationale=""),
        defense_state="intellectualization",
        psychological_contradiction="Wants truth but fears it",
        confidence=0.85,
    )


@pytest.fixture
def sample_character_bible():
    """A minimal valid CharacterBible for testing."""
    from schemas import CharacterBible
    return CharacterBible(
        play_id="test_play",
        character="TEST_CHAR",
        superobjective="To find truth",
        wounds_fears_needs="Fear of abandonment",
        recurring_tactics=["probe", "deflect", "mock"],
        preferred_defense_mechanisms=["intellectualization"],
        psychological_contradictions=["Seeks truth but avoids it"],
        speech_style="Measured, precise, with sudden bursts of passion",
        rhetorical_patterns=["rhetorical questions"],
        few_shot_lines=["What do you mean by that?", "I see through you."],
        known_facts=["fact1"],
        secrets=["secret1"],
        arc_by_scene={"test_1_1": "Begins cautious, grows bold"},
        tactic_distribution={"probe": 10, "deflect": 8, "mock": 5},
    )


@pytest.fixture
def sample_scene_context():
    """A minimal valid SceneContext for testing."""
    from schemas import SceneContext
    return SceneContext(
        play_id="test_play",
        character="TEST_CHAR",
        setting="A dimly lit room",
        characters_present=["TEST_CHAR", "PARTNER"],
        prior_events="They have just met.",
        stakes="Something unspoken hangs between them",
        partner_line="I don't think you understand what's happening here.",
    )


@pytest.fixture
def sample_vocabulary():
    """A small TacticVocabulary for testing."""
    from schemas import CanonicalTactic, TacticVocabulary
    return TacticVocabulary(
        version=1,
        tactics=[
            CanonicalTactic(
                canonical_id="DEFLECT",
                canonical_verb="deflect",
                description="deflect — redirect attention away from a threatening topic",
                members=["deflect", "redirect", "avoid"],
            ),
            CanonicalTactic(
                canonical_id="MOCK",
                canonical_verb="mock",
                description="mock — use ridicule to diminish the other person",
                members=["mock", "ridicule", "taunt"],
            ),
            CanonicalTactic(
                canonical_id="PROBE",
                canonical_verb="probe",
                description="probe — ask pointed questions to uncover hidden information",
                members=["probe", "test", "investigate"],
            ),
        ],
        unmapped=["consecrate"],
    )
