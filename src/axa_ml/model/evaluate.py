"""Model evaluation — compute, log, and persist classification metrics."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = structlog.get_logger(__name__)


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, and ``f1``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
    }


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log metrics via structlog."""
    logger.info("evaluation_metrics", **metrics)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Persist metrics as a JSON file.

    Args:
        metrics: Dictionary of metric values.
        path: Directory to save the JSON file into.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "metrics.json"
    with filepath.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("metrics_saved", path=str(filepath))
    return filepath
