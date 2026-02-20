"""CLI entry-point for the AXA ML pipeline."""

from pathlib import Path
from typing import Annotated

import structlog
import typer

from axa_ml.config import load_config
from axa_ml.data.download import download_dataset
from axa_ml.pipeline import run_pipeline

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
    path = download_dataset(cfg.data.url, cfg.data.raw_dir, force=force)
    typer.echo(f"Dataset saved to {path}")


@app.command()
def train(
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Train the model (assumes data is already downloaded)."""
    import pandas as pd

    from axa_ml.data.preprocessing import (
        create_target,
        drop_columns,
        encode_categoricals,
        split_data,
    )
    from axa_ml.model.train import optimize_hyperparameters, save_model, train_final_model

    cfg = load_config(config)
    csv_path = Path(cfg.data.raw_dir) / "pg15training.csv"
    if not csv_path.exists():
        typer.echo("Data not found. Run `axa-ml download` first.", err=True)
        raise typer.Exit(code=1)

    df = pd.read_csv(csv_path)
    df = create_target(df, cfg.features.target_column)
    df = drop_columns(df, cfg.features.drop_columns)
    df = encode_categoricals(df, cfg.features.categorical_columns)

    x_train, _x_test, y_train, _y_test = split_data(
        df, "target", cfg.model.test_size, cfg.model.random_seed
    )

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
    """Evaluate a saved model on the test set."""
    import pandas as pd

    from axa_ml.data.preprocessing import (
        create_target,
        drop_columns,
        encode_categoricals,
        split_data,
    )
    from axa_ml.model.evaluate import compute_metrics, log_metrics, save_metrics
    from axa_ml.model.train import load_model

    cfg = load_config(config)

    csv_path = Path(cfg.data.raw_dir) / "pg15training.csv"
    model_path = Path(cfg.artifacts.model_dir) / "model.joblib"

    if not csv_path.exists():
        typer.echo("Data not found. Run `axa-ml download` first.", err=True)
        raise typer.Exit(code=1)
    if not model_path.exists():
        typer.echo("Model not found. Run `axa-ml train` first.", err=True)
        raise typer.Exit(code=1)

    df = pd.read_csv(csv_path)
    df = create_target(df, cfg.features.target_column)
    df = drop_columns(df, cfg.features.drop_columns)
    df = encode_categoricals(df, cfg.features.categorical_columns)

    _x_train, x_test, _y_train, y_test = split_data(
        df, "target", cfg.model.test_size, cfg.model.random_seed
    )

    model = load_model(model_path)
    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)  # pyright: ignore[reportArgumentType]  # predict() always returns ndarray
    log_metrics(metrics)
    save_metrics(metrics, cfg.artifacts.metrics_dir)
    typer.echo(f"Evaluation complete. Metrics: {metrics}")


if __name__ == "__main__":
    app()
