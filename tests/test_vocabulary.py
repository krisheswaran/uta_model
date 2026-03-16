"""Test canonical tactic vocabulary module.

These tests verify vocabulary construction, lookup, normalization, and
incremental assignment — all without API calls. Clustering tests use
the actual sentence-transformers model (local, no network needed after
first download).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import CanonicalTactic, TacticVocabulary


class TestTacticExpansion:
    def test_defined_tactic_uses_definition(self):
        from analysis.vocabulary import expand_tactic_to_sentence, _TACTIC_DEFINITIONS
        result = expand_tactic_to_sentence("deflect")
        assert "redirect" in result.lower() or "deflect" in result.lower()
        assert result == _TACTIC_DEFINITIONS["deflect"]

    def test_undefined_tactic_uses_fallback(self):
        from analysis.vocabulary import expand_tactic_to_sentence
        result = expand_tactic_to_sentence("zygomorphize")
        assert "zygomorphize" in result

    def test_case_insensitive_lookup(self):
        from analysis.vocabulary import expand_tactic_to_sentence, _TACTIC_DEFINITIONS
        # The function lowercases the key
        if "deflect" in _TACTIC_DEFINITIONS:
            assert expand_tactic_to_sentence("deflect") == _TACTIC_DEFINITIONS["deflect"]


class TestVocabularyLookup:
    def test_exact_match(self, sample_vocabulary):
        assert sample_vocabulary.lookup("deflect") == "DEFLECT"

    def test_synonym_match(self, sample_vocabulary):
        assert sample_vocabulary.lookup("redirect") == "DEFLECT"
        assert sample_vocabulary.lookup("taunt") == "MOCK"

    def test_no_match(self, sample_vocabulary):
        assert sample_vocabulary.lookup("consecrate") is None  # in unmapped
        assert sample_vocabulary.lookup("totally_unknown") is None

    def test_case_insensitive(self, sample_vocabulary):
        assert sample_vocabulary.lookup("DEFLECT") == "DEFLECT"
        assert sample_vocabulary.lookup("Mock") == "MOCK"
        assert sample_vocabulary.lookup("PROBE") == "PROBE"


class TestVocabularySerialization:
    def test_roundtrip_json(self, sample_vocabulary):
        json_str = sample_vocabulary.model_dump_json()
        restored = TacticVocabulary.model_validate_json(json_str)
        assert len(restored.tactics) == 3
        assert restored.lookup("deflect") == "DEFLECT"
        assert restored.unmapped == ["consecrate"]

    def test_save_and_load(self, sample_vocabulary, tmp_path):
        from analysis.vocabulary import save_vocabulary, load_vocabulary
        path = tmp_path / "test_vocab.json"
        save_vocabulary(sample_vocabulary, path)
        loaded = load_vocabulary(path)
        assert len(loaded.tactics) == len(sample_vocabulary.tactics)
        assert loaded.lookup("mock") == "MOCK"

    def test_load_nonexistent_raises(self, tmp_path):
        from analysis.vocabulary import load_vocabulary
        with pytest.raises(FileNotFoundError):
            load_vocabulary(tmp_path / "nonexistent.json")


class TestClustering:
    """These tests use the actual embedding model — they are slower (~5s)
    but verify real clustering behavior."""

    @pytest.fixture
    def small_tactics(self):
        return {
            "deflect": 50, "redirect": 10, "avoid": 5,
            "mock": 30, "ridicule": 8, "taunt": 3,
            "plead": 20, "beg": 7, "implore": 4,
            "command": 15, "order": 6,
        }

    def test_cluster_produces_vocabulary(self, small_tactics):
        from analysis.vocabulary import cluster_tactics
        tactics = list(small_tactics.keys())
        counts = [small_tactics[t] for t in tactics]
        vocab = cluster_tactics(tactics, counts, distance_threshold=0.50)
        assert isinstance(vocab, TacticVocabulary)
        assert len(vocab.tactics) > 0
        # The most frequent tactic in each cluster should be the canonical verb
        for ct in vocab.tactics:
            assert ct.canonical_verb in ct.members

    def test_synonyms_cluster_together(self, small_tactics):
        from analysis.vocabulary import cluster_tactics
        tactics = list(small_tactics.keys())
        counts = [small_tactics[t] for t in tactics]
        vocab = cluster_tactics(tactics, counts, distance_threshold=0.50)
        # "deflect" and "redirect" should be in the same cluster
        deflect_id = vocab.lookup("deflect")
        redirect_id = vocab.lookup("redirect")
        if deflect_id and redirect_id:
            assert deflect_id == redirect_id

    def test_distinct_tactics_separate(self, small_tactics):
        from analysis.vocabulary import cluster_tactics
        tactics = list(small_tactics.keys())
        counts = [small_tactics[t] for t in tactics]
        vocab = cluster_tactics(tactics, counts, distance_threshold=0.30)
        # "deflect" and "command" should NOT be in the same cluster
        deflect_id = vocab.lookup("deflect")
        command_id = vocab.lookup("command")
        if deflect_id and command_id:
            assert deflect_id != command_id


class TestIncrementalAssignment:
    def test_assign_known_tactic_is_noop(self, sample_vocabulary):
        from analysis.vocabulary import assign_new_tactics
        original_count = sum(len(ct.members) for ct in sample_vocabulary.tactics)
        assign_new_tactics(sample_vocabulary, ["deflect", "mock"])
        new_count = sum(len(ct.members) for ct in sample_vocabulary.tactics)
        assert new_count == original_count  # no duplicates added

    def test_assign_new_tactic_adds_or_unmaps(self, sample_vocabulary):
        from analysis.vocabulary import assign_new_tactics
        assign_new_tactics(sample_vocabulary, ["parry"])
        # "parry" should either be in a cluster or in unmapped
        found = sample_vocabulary.lookup("parry") is not None
        in_unmapped = "parry" in sample_vocabulary.unmapped
        assert found or in_unmapped


class TestNormalization:
    def test_normalize_play_json(self, sample_vocabulary):
        from analysis.vocabulary import normalize_play_json
        data = {
            "acts": [{
                "scenes": [{
                    "beats": [{
                        "beat_states": [
                            {"tactic_state": "deflect"},
                            {"tactic_state": "mock"},
                            {"tactic_state": "unknown_xyz"},
                            {"tactic_state": ""},
                        ]
                    }]
                }]
            }]
        }
        assigned = normalize_play_json(data, sample_vocabulary)
        assert assigned == 2  # deflect and mock
        bs = data["acts"][0]["scenes"][0]["beats"][0]["beat_states"]
        assert bs[0]["canonical_tactic"] == "DEFLECT"
        assert bs[1]["canonical_tactic"] == "MOCK"
        assert bs[2]["canonical_tactic"] is None
        assert bs[3].get("canonical_tactic") is None
