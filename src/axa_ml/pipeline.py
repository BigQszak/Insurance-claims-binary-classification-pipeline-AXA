"""Pipeline orchestration — download → preprocess → train → evaluate."""

import random

import numpy as np
import pandas as pd
import structlog

from axa_ml.config import PipelineConfig
from axa_ml.data.download import download_dataset
from axa_ml.data.preprocessing import create_target, drop_columns, encode_categoricals, split_data
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


def run_pipeline(config: PipelineConfig) -> dict[str, float]:
    """Execute the full ML pipeline.

    Steps:
        1. Set global random seeds.
        2. Download dataset (idempotent).
        3. Preprocess (target creation, column dropping, encoding).
        4. Train/test split.
        5. Hyperparameter optimization with Optuna.
        6. Train final model with best parameters.
        7. Evaluate and persist metrics + model.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    # 1. Reproducibility
    _set_global_seeds(config.model.random_seed)

    # 2. Download
    csv_path = download_dataset(config.data.url, config.data.raw_dir)

    # 3. Preprocess
    df = pd.read_csv(csv_path)
    df = create_target(df, config.features.target_column)
    df = drop_columns(df, config.features.drop_columns)
    df = encode_categoricals(df, config.features.categorical_columns)

    # 4. Split
    x_train, x_test, y_train, y_test = split_data(
        df, "target", config.model.test_size, config.model.random_seed
    )

    # 5. Optimize
    best_params = optimize_hyperparameters(
        x_train,
        y_train,
        random_seed=config.model.random_seed,
        n_trials=config.model.n_trials,
        hp_space=config.model.hyperparameter_space,
    )

    # 6. Train final model
    model = train_final_model(x_train, y_train, best_params, random_seed=config.model.random_seed)

    # 7. Evaluate
    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)  # pyright: ignore[reportArgumentType]  # predict() always returns ndarray
    log_metrics(metrics)

    # Persist artifacts
    save_metrics(metrics, config.artifacts.metrics_dir)
    save_model(model, config.artifacts.model_dir)

    logger.info("pipeline_complete")
    return metrics
