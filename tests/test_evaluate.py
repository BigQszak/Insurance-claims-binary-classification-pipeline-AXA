"""Unit tests for model evaluation functions."""

import json

import numpy as np

from axa_ml.model.evaluate import compute_metrics, save_metrics


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0

    def test_returns_float_values(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)

        for value in metrics.values():
            assert isinstance(value, float)

    def test_expected_keys(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_metrics(y_true, y_pred)

        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_partial_predictions(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.5
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5


class TestSaveMetrics:
    """Tests for the save_metrics function."""

    def test_creates_json_file(self, tmp_path) -> None:
        metrics = {"accuracy": 0.95, "precision": 0.9}
        filepath = save_metrics(metrics, tmp_path)

        assert filepath.exists()
        assert filepath.name == "metrics.json"

    def test_json_content_matches(self, tmp_path) -> None:
        metrics = {"accuracy": 0.95, "precision": 0.9}
        filepath = save_metrics(metrics, tmp_path)

        with filepath.open() as f:
            loaded = json.load(f)
        assert loaded == metrics

    def test_creates_directories(self, tmp_path) -> None:
        nested = tmp_path / "nested" / "dir"
        filepath = save_metrics({"accuracy": 0.5}, nested)
        assert filepath.exists()
