"""Pipeline orchestration — download → preprocess → train → evaluate."""

import random

import numpy as np
import pandas as pd
import structlog

from axa_ml.config import PipelineConfig
from axa_ml.data.download import download_dataset
from axa_ml.data.preprocessing import (
    create_target,
    drop_columns,
    encode_categoricals,
    save_splits,
    split_data,
)
from axa_ml.model.evaluate import compute_metrics, log_metrics, save_metrics
from axa_ml.model.train import (
    optimize_hyperparameters,
    save_model,
    train_final_model,
)

logger = structlog.get_logger(__name__)


def _set_global_seeds(seed: int) -> None:
    """Pin all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info("seeds_set", seed=seed)


def load_and_preprocess(
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data, apply preprocessing, split, and persist splits.

    This is the single shared helper that eliminates duplicated preprocessing
    logic across the pipeline and CLI commands.

    Steps:
        1. Read raw CSV.
        2. Create binary target column.
        3. Drop leakage / unused columns.
        4. One-hot encode categoricals.
        5. Stratified train/test split.
        6. Save splits to ``config.data.processed_dir``.

    Args:
        config: Validated pipeline configuration.

    Returns:
        ``(X_train, X_test, y_train, y_test)``

    Raises:
        FileNotFoundError: If the raw CSV does not exist.
    """
    from pathlib import Path

    csv_path = Path(config.data.raw_dir) / config.data.dataset_filename
    if not csv_path.exists():
        msg = f"Raw data not found: {csv_path}. Run `axa-ml download` first."
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    df = create_target(df, config.features.target_column)
    df = drop_columns(df, config.features.drop_columns)
    df = encode_categoricals(df, config.features.categorical_columns)

    x_train, x_test, y_train, y_test = split_data(
        df, "target", config.model.test_size, config.model.random_seed
    )

    # Persist splits so `evaluate` can reload them without re-splitting
    save_splits(x_train, x_test, y_train, y_test, config.data.processed_dir)

    return x_train, x_test, y_train, y_test


def run_pipeline(config: PipelineConfig) -> dict[str, float]:
    """Execute the full ML pipeline.

    Steps:
        1. Set global random seeds.
        2. Download dataset (idempotent).
        3. Preprocess and split data.
        4. Hyperparameter optimization with Optuna.
        5. Train final model with best parameters.
        6. Evaluate and persist metrics + model.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    # 1. Reproducibility
    _set_global_seeds(config.model.random_seed)

    # 2. Download
    download_dataset(
        config.data.url,
        config.data.raw_dir,
        filename=config.data.dataset_filename,
        rda_key=config.data.rda_key,
    )

    # 3. Preprocess + split + save splits
    x_train, x_test, y_train, y_test = load_and_preprocess(config)

    # 4. Optimize
    best_params = optimize_hyperparameters(
        x_train,
        y_train,
        random_seed=config.model.random_seed,
        n_trials=config.model.n_trials,
        hp_space=config.model.hyperparameter_space,
    )

    # 5. Train final model
    model = train_final_model(x_train, y_train, best_params, random_seed=config.model.random_seed)

    # 6. Evaluate
    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)  # pyright: ignore[reportArgumentType]
    log_metrics(metrics)

    # Persist artifacts
    save_metrics(metrics, config.artifacts.metrics_dir)
    save_model(model, config.artifacts.model_dir)

    logger.info("pipeline_complete")
    return metrics
