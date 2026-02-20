"""Pipeline configuration models with Pydantic validation."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data source configuration."""

    url: str
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    target_column: str = "Numtppd"
    drop_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)


class HyperparameterSpace(BaseModel):
    """Bounds for each hyperparameter (min, max)."""

    n_estimators: tuple[int, int] = (10, 200)
    learning_rate: tuple[float, float] = (0.001, 0.3)
    max_depth: tuple[int, int] = (3, 10)
    num_leaves: tuple[int, int] = (20, 150)
    min_child_samples: tuple[int, int] = (5, 100)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.5, 1.0)


class ModelConfig(BaseModel):
    """Model training configuration."""

    test_size: Annotated[float, Field(gt=0, lt=1)] = 0.2
    random_seed: int = 42
    n_trials: Annotated[int, Field(gt=0)] = 20
    hyperparameter_space: HyperparameterSpace = Field(default_factory=HyperparameterSpace)


class ArtifactsConfig(BaseModel):
    """Output artifacts paths."""

    metrics_dir: str = "artifacts/metrics"
    model_dir: str = "artifacts/models"


class PipelineConfig(BaseModel):
    """Root configuration that aggregates all sub-configs."""

    data: DataConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)


def load_config(path: Path | str) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ``PipelineConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    return PipelineConfig.model_validate(raw)
