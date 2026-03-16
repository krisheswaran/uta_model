"""Test that all modules import without errors.

These are smoke tests — they verify that the code compiles and that
cross-module dependencies resolve correctly. No API calls are made.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoreImports:
    """Core modules that should always import cleanly."""

    def test_config(self):
        from config import MODEL_CONFIGS, get_model, MIN_REVISION_ROUNDS, MAX_REVISION_ROUNDS

    def test_schemas_core(self):
        from schemas import (
            Play, Act, Scene, Beat, Utterance,
            BeatState, AffectState, SocialState, EpistemicState,
            CharacterBible, SceneBible, WorldBible, RelationshipEdge,
        )

    def test_schemas_improv(self):
        from schemas import (
            SceneContext, CandidateLine, ScoredLine,
            ImprovTurn, ImprovSession, RevisionTrace,
        )

    def test_schemas_phase_b(self):
        from schemas import (
            CanonicalTactic, TacticVocabulary,
            BeatStateEstimate, RelationalProfile, StatisticalPrior,
        )

    def test_schemas_evaluation(self):
        from schemas import JudgeRating


class TestAnalysisImports:
    """Analysis pipeline modules."""

    def test_segmenter(self):
        from analysis.segmenter import segment_play

    def test_extractor(self):
        from analysis.extractor import extract_all_beats

    def test_smoother(self):
        from analysis.smoother import smooth_play

    def test_bible_builder(self):
        from analysis.bible_builder import (
            build_all_bibles, build_character_bible,
            build_scene_bible, build_world_bible,
            _count_character_beat_states, _existing_bible_characters,
        )

    def test_vocabulary(self):
        from analysis.vocabulary import (
            collect_raw_tactics, expand_tactic_to_sentence,
            embed_tactics, cluster_tactics, assign_new_tactics,
            normalize_play, normalize_play_json,
            save_vocabulary, load_vocabulary,
        )


class TestImprovImports:
    """Improvisation pipeline modules."""

    def test_improvisation_loop(self):
        from improv.improvisation_loop import (
            initialize_beat_state, generate_candidate,
            run_turn, run_session,
        )

    def test_scorer(self):
        from improv.scorer import score_candidate

    def test_state_updater(self):
        from improv.state_updater import update_beat_state


class TestIngestImports:
    """Ingest/parsing modules."""

    def test_gutenberg_parser(self):
        from ingest.gutenberg_parser import parse_gutenberg_play

    def test_tei_parser(self):
        from ingest.tei_parser import parse_tei_play


class TestEvaluationImports:
    """Evaluation module."""

    def test_judge(self):
        from evaluation.judge import evaluate_three_tiers, judge_line, summarize_ratings


class TestScriptImports:
    """Script entry points — import the module but don't run main()."""

    def test_run_analysis(self):
        import scripts.run_analysis

    def test_run_improvisation(self):
        import scripts.run_improvisation

    def test_run_evaluation(self):
        import scripts.run_evaluation
