"""Model training — Optuna hyperparameter optimization and LightGBM fitting."""

from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
import structlog
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from axa_ml.config import HyperparameterSpace

logger = structlog.get_logger(__name__)


def _create_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
    hp_space: HyperparameterSpace,
) -> Any:
    """Return an Optuna objective function bound to the given data and search space."""

    def objective(trial: optuna.Trial) -> float:
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=random_seed, stratify=y_train
        )

        params = {
            "objective": "binary",
            "n_jobs": 1,
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", *hp_space.n_estimators),
            "learning_rate": trial.suggest_float(
                "learning_rate", *hp_space.learning_rate, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", *hp_space.max_depth),
            "num_leaves": trial.suggest_int("num_leaves", *hp_space.num_leaves),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *hp_space.min_child_samples
            ),
            "subsample": trial.suggest_float("subsample", *hp_space.subsample),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *hp_space.colsample_bytree
            ),
        }

        model = LGBMClassifier(**params)
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)
        return float(accuracy_score(y_val, y_pred))

    return objective


def optimize_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int = 42,
    n_trials: int = 20,
    hp_space: HyperparameterSpace | None = None,
) -> dict[str, Any]:
    """Run Optuna study to find the best LightGBM hyperparameters.

    Args:
        x_train: Training features.
        y_train: Training labels.
        random_seed: Seed for the Optuna sampler and validation split.
        n_trials: Number of optimization trials.
        hp_space: Search space bounds.

    Returns:
        Dictionary of best hyperparameters.
    """
    if hp_space is None:
        hp_space = HyperparameterSpace()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = _create_objective(x_train, y_train, random_seed, hp_space)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info("optimization_complete", best_value=study.best_value, best_params=best)
    return best


def train_final_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
    *,
    random_seed: int = 42,
) -> LGBMClassifier:
    """Train the final LightGBM model using the best hyperparameters.

    Args:
        x_train: Full training features.
        y_train: Full training labels.
        best_params: Hyperparameters from optimization.
        random_seed: Random state for the model.

    Returns:
        Fitted ``LGBMClassifier``.
    """
    model = LGBMClassifier(**best_params, verbosity=-1, n_jobs=1, random_state=random_seed)
    model.fit(x_train, y_train)
    logger.info("model_trained", n_features=x_train.shape[1])
    return model


def save_model(model: LGBMClassifier, output_dir: str | Path) -> Path:
    """Persist a trained model to disk using joblib.

    Args:
        model: Fitted model.
        output_dir: Directory to save the model file into.

    Returns:
        Path to the saved model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "model.joblib"
    joblib.dump(model, filepath)
    logger.info("model_saved", path=str(filepath))
    return filepath


def load_model(path: str | Path) -> LGBMClassifier:
    """Load a persisted model from disk.

    Args:
        path: Path to the model ``.joblib`` file.

    Returns:
        Loaded ``LGBMClassifier``.
    """
    path = Path(path)
    model: LGBMClassifier = joblib.load(path)
    logger.info("model_loaded", path=str(path))
    return model
