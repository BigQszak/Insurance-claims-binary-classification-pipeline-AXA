"""CLI entry-point for the AXA ML pipeline."""

import random
from pathlib import Path
from typing import Annotated

import numpy as np
import structlog
import typer

from axa_ml.config import PipelineConfig, load_config
from axa_ml.data.download import download_dataset
from axa_ml.data.preprocessing import load_splits
from axa_ml.model.evaluate import compute_metrics, log_metrics, save_metrics
from axa_ml.model.train import load_model, optimize_hyperparameters, save_model, train_final_model
from axa_ml.pipeline import load_and_preprocess, run_pipeline

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

app = typer.Typer(
    name="axa-ml",
    help="Insurance claims classification pipeline — AXA ML Engineering.",
    add_completion=False,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the YAML configuration file."),
]


def _set_global_seeds(config: PipelineConfig) -> None:
    """Set global seeds for standalone CLI commands."""
    random.seed(config.model.random_seed)
    np.random.seed(config.model.random_seed)


@app.command()
def run(
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Run the full pipeline: download → preprocess → train → evaluate."""
    cfg = load_config(config)
    metrics = run_pipeline(cfg)
    typer.echo(f"\nPipeline complete. Metrics: {metrics}")


@app.command()
def download(
    config: ConfigOption = Path("configs/default.yaml"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if file exists."),
) -> None:
    """Download the dataset only."""
    cfg = load_config(config)
    path = download_dataset(
        cfg.data.url,
        cfg.data.raw_dir,
        filename=cfg.data.dataset_filename,
        rda_key=cfg.data.rda_key,
        force=force,
    )
    typer.echo(f"Dataset saved to {path}")


@app.command()
def train(
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Train the model (assumes data is already downloaded)."""
    cfg = load_config(config)
    _set_global_seeds(cfg)

    x_train, _x_test, y_train, _y_test = load_and_preprocess(cfg)

    best_params = optimize_hyperparameters(
        x_train,
        y_train,
        random_seed=cfg.model.random_seed,
        n_trials=cfg.model.n_trials,
        hp_space=cfg.model.hyperparameter_space,
    )
    model = train_final_model(x_train, y_train, best_params, random_seed=cfg.model.random_seed)
    save_model(model, cfg.artifacts.model_dir)
    typer.echo("Model trained and saved.")


@app.command()
def evaluate(
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Evaluate a saved model on the test set.

    Loads the pre-saved train/test splits from ``processed_dir`` to guarantee
    that evaluation uses the exact same data split as training.
    """
    cfg = load_config(config)

    model_path = Path(cfg.artifacts.model_dir) / "model.joblib"
    if not model_path.exists():
        typer.echo("Model not found. Run `axa-ml train` first.", err=True)
        raise typer.Exit(code=1)

    try:
        _x_train, x_test, _y_train, y_test = load_splits(cfg.data.processed_dir)
    except FileNotFoundError:
        typer.echo(
            "Processed splits not found. Run `axa-ml run` or `axa-ml train` first.",
            err=True,
        )
        raise typer.Exit(code=1)  # noqa: B904

    model = load_model(model_path)
    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)  # pyright: ignore[reportArgumentType]
    log_metrics(metrics)
    save_metrics(metrics, cfg.artifacts.metrics_dir)
    typer.echo(f"Evaluation complete. Metrics: {metrics}")


if __name__ == "__main__":
    app()
