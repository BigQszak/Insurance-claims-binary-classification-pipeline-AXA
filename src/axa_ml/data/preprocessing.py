"""Data preprocessing — target creation, encoding, and train/test split."""

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
    """Split into train/test features and labels.

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
        x, y, test_size=test_size, random_state=random_seed
    )
    logger.info(
        "data_split",
        train_size=len(x_train),
        test_size=len(x_test),
    )
    return x_train, x_test, y_train, y_test  # pyright: ignore[reportReturnType]
