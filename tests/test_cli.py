"""Unit tests for the CLI entry-point."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from axa_ml.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    """CLI test runner."""
    return CliRunner()


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
    """Create a minimal valid YAML config file and return its path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config_data = {
        "data": {
            "url": "https://example.com/data.rda",
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "processed"),
            "dataset_filename": "pg15training.csv",
        },
        "model": {
            "test_size": 0.2,
            "random_seed": 42,
            "n_trials": 2,
        },
        "features": {
            "target_column": "Numtppd",
            "drop_columns": ["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"],
            "categorical_columns": ["Gender", "Type"],
        },
        "artifacts": {
            "metrics_dir": str(tmp_path / "metrics"),
            "model_dir": str(tmp_path / "models"),
        },
    }
    config_file = tmp_path / "test_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)
    return config_file


class TestCliHelp:
    """Tests that CLI commands are reachable and display help."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Insurance claims" in result.stdout

    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout

    def test_download_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.stdout

    def test_train_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0

    def test_evaluate_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0


class TestDownloadCommand:
    """Tests for the download CLI command."""

    def test_download_success(
        self, runner: CliRunner, minimal_config: Path, tmp_path: Path
    ) -> None:
        fake_path = tmp_path / "raw" / "pg15training.csv"
        with patch("axa_ml.cli.download_dataset", return_value=fake_path):
            result = runner.invoke(app, ["download", "--config", str(minimal_config)])
        assert result.exit_code == 0
        assert "Dataset saved to" in result.stdout

    def test_download_with_force(
        self, runner: CliRunner, minimal_config: Path, tmp_path: Path
    ) -> None:
        fake_path = tmp_path / "raw" / "pg15training.csv"
        with patch("axa_ml.cli.download_dataset", return_value=fake_path) as mock_dl:
            result = runner.invoke(app, ["download", "--config", str(minimal_config), "--force"])
        assert result.exit_code == 0
        mock_dl.assert_called_once()
        # Verify force=True was passed
        _, kwargs = mock_dl.call_args
        assert kwargs["force"] is True


class TestEvaluateCommand:
    """Tests for the evaluate CLI command."""

    def test_evaluate_missing_model(self, runner: CliRunner, minimal_config: Path) -> None:
        result = runner.invoke(app, ["evaluate", "--config", str(minimal_config)])
        assert result.exit_code == 1
        assert "Model not found" in result.output

    def test_evaluate_missing_splits(
        self, runner: CliRunner, minimal_config: Path, tmp_path: Path
    ) -> None:
        # Create dummy model so it passes the model check
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "model.joblib").write_text("dummy")

        result = runner.invoke(app, ["evaluate", "--config", str(minimal_config)])
        assert result.exit_code == 1
        assert "splits not found" in result.output


class TestRunCommand:
    """Tests for the run CLI command."""

    def test_run_calls_pipeline(self, runner: CliRunner, minimal_config: Path) -> None:
        fake_metrics = {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75}
        with patch("axa_ml.cli.run_pipeline", return_value=fake_metrics):
            result = runner.invoke(app, ["run", "--config", str(minimal_config)])
        assert result.exit_code == 0
        assert "Pipeline complete" in result.stdout
