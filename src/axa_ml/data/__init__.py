"""Data subpackage — download and preprocessing."""

from axa_ml.data.download import download_dataset
from axa_ml.data.preprocessing import (
    create_target,
    drop_columns,
    encode_categoricals,
    load_splits,
    save_splits,
    split_data,
)

__all__ = [
    "create_target",
    "download_dataset",
    "drop_columns",
    "encode_categoricals",
    "load_splits",
    "save_splits",
    "split_data",
]
