"""Unit tests for configuration loading and validation."""

import pytest
import yaml

from axa_ml.config import PipelineConfig, load_config


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_loads_valid_config(self, tmp_path) -> None:
        config_data = {
            "data": {
                "url": "https://example.com/data.rda",
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
            },
            "features": {
                "target_column": "Numtppd",
                "drop_columns": ["Numtppd"],
                "categorical_columns": ["Gender"],
            },
            "model": {
                "test_size": 0.2,
                "random_seed": 42,
                "n_trials": 5,
            },
            "artifacts": {
                "metrics_dir": "artifacts/metrics",
                "model_dir": "artifacts/models",
            },
        }
        config_file = tmp_path / "test_config.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert isinstance(config, PipelineConfig)
        assert config.data.url == "https://example.com/data.rda"
        assert config.model.random_seed == 42

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_test_size_raises(self, tmp_path) -> None:
        config_data = {
            "data": {"url": "https://example.com/data.rda"},
            "model": {"test_size": 1.5},  # invalid: must be < 1
        }
        config_file = tmp_path / "bad_config.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(Exception):  # noqa: B017
            load_config(config_file)

    def test_defaults_applied(self, tmp_path) -> None:
        config_data = {
            "data": {"url": "https://example.com/data.rda"},
        }
        config_file = tmp_path / "minimal.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.model.random_seed == 42
        assert config.model.test_size == 0.2
        assert config.model.n_trials == 20

    def test_hyperparameter_space_defaults(self, tmp_path) -> None:
        config_data = {
            "data": {"url": "https://example.com/data.rda"},
        }
        config_file = tmp_path / "minimal.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        hp = config.model.hyperparameter_space
        assert hp.n_estimators == (10, 200)
        assert hp.learning_rate == (0.001, 0.3)
