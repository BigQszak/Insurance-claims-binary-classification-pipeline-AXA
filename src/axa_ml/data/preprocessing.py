"""Data preprocessing — target creation, encoding, train/test split, and persistence."""

from pathlib import Path

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger(__name__)


def create_target(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Create a binary ``target`` column (1 if *column* != 0, else 0).

    Args:
        df: Input dataframe.
        column: Source column for target derivation.

    Returns:
        DataFrame with an added ``target`` column.
    """
    df = df.copy()
    df["target"] = (df[column] != 0).astype(int)
    logger.info("target_created", positive_rate=f"{df['target'].mean():.4f}")
    return df


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop the specified columns from the dataframe.

    Missing columns are silently ignored.
    """
    existing = [c for c in columns if c in df.columns]
    df = df.drop(columns=existing)
    logger.info("columns_dropped", dropped=existing)
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode the specified categorical columns.

    Args:
        df: Input dataframe.
        columns: Columns to encode.

    Returns:
        DataFrame with encoded columns replacing the originals.
    """
    existing = [c for c in columns if c in df.columns]
    df = pd.get_dummies(df, columns=existing)
    logger.info("categoricals_encoded", encoded=existing)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test features and labels with stratification.

    Uses stratified splitting to preserve the class distribution in both
    the training and test sets — important for imbalanced datasets.

    Args:
        df: Preprocessed dataframe containing the *target_col*.
        target_col: Name of the target column.
        test_size: Fraction of data reserved for testing.
        random_seed: Random state for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``
    """
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    logger.info(
        "data_split",
        train_size=len(x_train),
        test_size=len(x_test),
        train_positive_rate=f"{y_train.mean():.4f}",
        test_positive_rate=f"{y_test.mean():.4f}",
    )
    return x_train, x_test, y_train, y_test  # pyright: ignore[reportReturnType]


def save_splits(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
) -> Path:
    """Persist train/test splits as CSV files for reproducible evaluation.

    Saves four files into *output_dir*:
    ``x_train.csv``, ``x_test.csv``, ``y_train.csv``, ``y_test.csv``.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        output_dir: Directory to save the CSV files into.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(output_dir / "x_train.csv", index=False)
    x_test.to_csv(output_dir / "x_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)

    logger.info("splits_saved", output_dir=str(output_dir))
    return output_dir


def load_splits(
    input_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load previously saved train/test splits from CSV files.

    Args:
        input_dir: Directory containing the split CSV files.

    Returns:
        ``(X_train, X_test, y_train, y_test)``

    Raises:
        FileNotFoundError: If any of the expected split files is missing.
    """
    input_dir = Path(input_dir)

    expected_files = ["x_train.csv", "x_test.csv", "y_train.csv", "y_test.csv"]
    for fname in expected_files:
        if not (input_dir / fname).exists():
            msg = f"Split file not found: {input_dir / fname}"
            raise FileNotFoundError(msg)

    x_train = pd.read_csv(input_dir / "x_train.csv")
    x_test = pd.read_csv(input_dir / "x_test.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(input_dir / "y_test.csv").squeeze("columns")

    logger.info(
        "splits_loaded",
        input_dir=str(input_dir),
        train_size=len(x_train),
        test_size=len(x_test),
    )
    return x_train, x_test, y_train, y_test
