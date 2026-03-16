"""Test configuration module."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_CONFIGS, get_model, MIN_REVISION_ROUNDS, MAX_REVISION_ROUNDS


class TestModelConfigs:
    def test_all_steps_have_configs(self):
        expected_steps = [
            "segmentation", "extraction", "smoothing",
            "bible", "world_bible",
            "generation", "critic", "judge",
        ]
        for step in expected_steps:
            assert step in MODEL_CONFIGS, f"Missing config for step: {step}"

    def test_all_configs_have_required_fields(self):
        for step, cfg in MODEL_CONFIGS.items():
            assert "provider" in cfg, f"{step} missing 'provider'"
            assert "model" in cfg, f"{step} missing 'model'"

    def test_get_model_returns_string(self):
        for step in MODEL_CONFIGS:
            model = get_model(step)
            assert isinstance(model, str)
            assert len(model) > 0

    def test_get_model_invalid_step(self):
        with pytest.raises(KeyError):
            get_model("nonexistent_step")

    def test_model_tiers_make_sense(self):
        """Extraction and smoothing should use the most capable model."""
        extraction = get_model("extraction")
        smoothing = get_model("smoothing")
        assert "opus" in extraction.lower()
        assert "opus" in smoothing.lower()

    def test_cheaper_steps_use_cheaper_models(self):
        """Segmentation and bible building should not use Opus."""
        segmentation = get_model("segmentation")
        bible = get_model("bible")
        assert "opus" not in segmentation.lower()
        assert "opus" not in bible.lower()


class TestImprovConfig:
    def test_min_revision_rounds_exists(self):
        assert isinstance(MIN_REVISION_ROUNDS, int)
        assert MIN_REVISION_ROUNDS >= 0

    def test_max_greater_than_min(self):
        assert MAX_REVISION_ROUNDS >= MIN_REVISION_ROUNDS

    def test_min_revision_default_is_one(self):
        assert MIN_REVISION_ROUNDS == 1
