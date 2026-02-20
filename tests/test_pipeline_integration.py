"""Integration test — full pipeline with synthetic data (no network access)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from axa_ml.config import PipelineConfig
from axa_ml.pipeline import run_pipeline


def _make_synthetic_csv(path: Path) -> Path:
    """Create a small synthetic CSV that mirrors the real dataset schema."""
    import numpy as np

    rng = np.random.RandomState(42)
    n = 200

    df = pd.DataFrame(
        {
            "Numtppd": rng.choice([0, 0, 0, 0, 1, 2], size=n),
            "Numtpbi": rng.choice([0, 0, 0, 1], size=n),
            "Indtppd": rng.uniform(0, 1000, size=n).round(2),
            "Indtpbi": rng.uniform(0, 500, size=n).round(2),
            "Exposure": rng.uniform(0.1, 1.0, size=n).round(2),
            "Power": rng.randint(40, 120, size=n),
            "CalYear": rng.choice(["2010", "2011"], size=n),
            "Gender": rng.choice(["M", "F"], size=n),
            "Type": rng.choice(["A", "B"], size=n),
            "Category": rng.choice(["C1", "C2"], size=n),
            "Occupation": rng.choice(["O1", "O2"], size=n),
            "SubGroup2": rng.choice(["S1", "S2"], size=n),
            "Group2": rng.choice(["G1", "G2"], size=n),
            "Group1": rng.choice(["GG1", "GG2"], size=n),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.mark.integration
def test_full_pipeline(tmp_path: Path) -> None:
    """Run the pipeline end-to-end on synthetic data and assert outputs."""
    raw_dir = tmp_path / "raw"
    csv_path = raw_dir / "pg15training.csv"
    _make_synthetic_csv(csv_path)

    config = PipelineConfig(
        data={
            "url": "https://unused.example.com",
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "processed"),
        },
        features={
            "target_column": "Numtppd",
            "drop_columns": ["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"],
            "categorical_columns": [
                "CalYear",
                "Gender",
                "Type",
                "Category",
                "Occupation",
                "SubGroup2",
                "Group2",
                "Group1",
            ],
        },
        model={
            "test_size": 0.2,
            "random_seed": 42,
            "n_trials": 3,  # keep fast for tests
        },
        artifacts={
            "metrics_dir": str(tmp_path / "metrics"),
            "model_dir": str(tmp_path / "models"),
        },
    )

    # Patch download_dataset so it returns the pre-created CSV path
    with patch("axa_ml.pipeline.download_dataset", return_value=csv_path):
        metrics = run_pipeline(config)

    # Assert metrics are sane
    assert isinstance(metrics, dict)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    # Assert artifacts were created
    assert (tmp_path / "metrics" / "metrics.json").exists()
    assert (tmp_path / "models" / "model.joblib").exists()
