"""Model evaluation — compute, log, and persist classification metrics."""

import json
from pathlib import Path
from typing import Any

import structlog
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = structlog.get_logger(__name__)


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Compute standard binary classification metrics including confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``,
        and ``confusion_matrix`` (as a nested list ``[[TN, FP], [FN, TP]]``).
    """
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "confusion_matrix": cm.tolist(),
    }


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log metrics via structlog."""
    logger.info("evaluation_metrics", **metrics)


def save_metrics(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist metrics as a JSON file.

    Args:
        metrics: Dictionary of metric values.
        output_dir: Directory to save the JSON file into.

    Returns:
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "metrics.json"
    with filepath.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("metrics_saved", path=str(filepath))
    return filepath
