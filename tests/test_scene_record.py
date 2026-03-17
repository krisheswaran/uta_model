"""Test SceneRecord construction and serialization."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    BeatState, CandidateLine, ImprovTurn, RevisionTrace,
    SceneContext, SceneRecord, ScoredLine,
)


def _make_turn(turn_index: int, character: str, line: str, tactic: str,
               score: float = 3.5, drafts: int = 2) -> ImprovTurn:
    """Build a minimal ImprovTurn with revision trace."""
    ctx = SceneContext(
        play_id="test", character=character, setting="A room",
        characters_present=[character, "PARTNER"], prior_events="",
        stakes="high",
    )
    bs = BeatState(beat_id=f"b{turn_index}", character=character, tactic_state=tactic)
    scored = ScoredLine(
        candidate=CandidateLine(text=line),
        voice_fidelity=score, tactic_fidelity=score, knowledge_fidelity=score,
        relationship_fidelity=score, subtext_richness=score,
        emotional_transition_plausibility=score,
    )
    trace = [
        RevisionTrace(
            round=0, candidate_text=f"Draft 1 of {line}",
            scores={"voice_fidelity": score - 0.5},
            feedback=["Try harder"],
        ),
        RevisionTrace(
            round=1, candidate_text=line,
            scores={"voice_fidelity": score},
            feedback=["Good instinct — the deflection feels characteristic."],
        ),
    ]
    return ImprovTurn(
        turn_index=turn_index, context=ctx,
        initial_beat_state=bs, final_line=line,
        revisions=drafts - 1, scored_line=scored,
        updated_beat_state=bs, revision_trace=trace,
    )


class TestSceneRecord:
    def test_construction(self):
        turn = _make_turn(1, "HAMLET", "To be or not to be.", "probe")
        record = SceneRecord(
            scene_id="test_001",
            mode="session",
            timestamp="2026-03-16T12:00:00Z",
            setting="A bare stage",
            stakes="Everything",
            characters=[{"character": "HAMLET", "play_id": "hamlet", "has_prior": True}],
            config={"cli_args": {"character": "HAMLET"}, "model_configs": {}},
            turns=[turn],
            transcript=[
                {"speaker": "PARTNER", "line": "What troubles you?"},
                {"speaker": "HAMLET", "line": "To be or not to be.", "tactic": "probe", "mean_score": 3.5},
            ],
        )
        assert record.scene_id == "test_001"
        assert record.mode == "session"
        assert len(record.turns) == 1
        assert len(record.transcript) == 2

    def test_revision_trace_preserved_in_turns(self):
        turn = _make_turn(1, "HAMLET", "Final line.", "mock")
        record = SceneRecord(
            scene_id="test_002", mode="session",
            timestamp="2026-03-16T12:00:00Z",
            setting="", stakes="",
            turns=[turn], transcript=[],
        )
        assert len(record.turns[0].revision_trace) == 2
        assert record.turns[0].revision_trace[0].candidate_text == "Draft 1 of Final line."
        assert "Try harder" in record.turns[0].revision_trace[0].feedback

    def test_dramaturgical_feedback_in_trace(self):
        turn = _make_turn(1, "HAMLET", "Line.", "deflect")
        # Simulate dramaturgical feedback in the second round
        turn.revision_trace[1].feedback = [
            "Good instinct — the deflection feels characteristic of HAMLET.",
        ]
        record = SceneRecord(
            scene_id="test_003", mode="crossplay",
            timestamp="2026-03-16T12:00:00Z",
            setting="", stakes="",
            turns=[turn], transcript=[],
        )
        feedback = record.turns[0].revision_trace[1].feedback
        assert any("characteristic" in f for f in feedback)

    def test_config_snapshot_included(self):
        record = SceneRecord(
            scene_id="test_004", mode="session",
            timestamp="2026-03-16T12:00:00Z",
            setting="", stakes="",
            config={
                "cli_args": {"character": "LOPAKHIN", "play": "cherry_orchard",
                             "min_revisions": 2},
                "model_configs": {
                    "generation": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                },
                "pipeline_params": {
                    "max_revision_rounds": 3, "min_revision_rounds": 2,
                    "score_threshold": 3.0,
                },
            },
        )
        assert record.config["cli_args"]["min_revisions"] == 2
        assert record.config["pipeline_params"]["min_revision_rounds"] == 2
        assert "generation" in record.config["model_configs"]

    def test_roundtrip_json(self):
        turn = _make_turn(1, "HAMLET", "A line.", "mock", drafts=2)
        record = SceneRecord(
            scene_id="test_005", mode="crossplay",
            timestamp="2026-03-16T12:00:00Z",
            setting="Elsinore", stakes="The crown",
            characters=[
                {"character": "HAMLET", "play_id": "hamlet", "has_prior": True,
                 "top_tactic": "MOCK", "default_warmth": -0.01},
                {"character": "LOPAKHIN", "play_id": "cherry_orchard", "has_prior": True,
                 "top_tactic": "DEFLECT", "default_warmth": 0.22},
            ],
            config={"cli_args": {}, "model_configs": {}, "pipeline_params": {}},
            turns=[turn],
            transcript=[{"speaker": "HAMLET", "line": "A line.", "tactic": "mock", "mean_score": 3.5}],
        )
        json_str = record.model_dump_json(indent=2)
        restored = SceneRecord.model_validate_json(json_str)
        assert restored.scene_id == "test_005"
        assert len(restored.turns) == 1
        assert len(restored.turns[0].revision_trace) == 2
        assert restored.characters[0]["top_tactic"] == "MOCK"

    def test_save_and_load(self, tmp_path):
        turn = _make_turn(1, "X", "Hello.", "probe")
        record = SceneRecord(
            scene_id="test_save", mode="session",
            timestamp="2026-03-16T12:00:00Z",
            setting="", stakes="",
            turns=[turn], transcript=[],
        )
        path = tmp_path / f"{record.scene_id}.json"
        path.write_text(record.model_dump_json(indent=2))
        loaded = SceneRecord.model_validate_json(path.read_text())
        assert loaded.scene_id == record.scene_id
        assert len(loaded.turns) == 1
