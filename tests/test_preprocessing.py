"""Unit tests for data preprocessing functions."""

import pandas as pd

from axa_ml.data.preprocessing import create_target, drop_columns, encode_categoricals, split_data


class TestCreateTarget:
    """Tests for the create_target function."""

    def test_binary_target_values(self, sample_dataframe: pd.DataFrame) -> None:
        result = create_target(sample_dataframe, "Numtppd")
        assert "target" in result.columns
        assert set(result["target"].unique()).issubset({0, 1})

    def test_nonzero_mapped_to_one(self, sample_dataframe: pd.DataFrame) -> None:
        result = create_target(sample_dataframe, "Numtppd")
        nonzero_mask = sample_dataframe["Numtppd"] != 0
        assert (result.loc[nonzero_mask, "target"] == 1).all()

    def test_zero_mapped_to_zero(self, sample_dataframe: pd.DataFrame) -> None:
        result = create_target(sample_dataframe, "Numtppd")
        zero_mask = sample_dataframe["Numtppd"] == 0
        assert (result.loc[zero_mask, "target"] == 0).all()

    def test_does_not_modify_original(self, sample_dataframe: pd.DataFrame) -> None:
        original_cols = set(sample_dataframe.columns)
        create_target(sample_dataframe, "Numtppd")
        assert set(sample_dataframe.columns) == original_cols


class TestDropColumns:
    """Tests for the drop_columns function."""

    def test_drops_existing_columns(self, sample_dataframe: pd.DataFrame) -> None:
        result = drop_columns(sample_dataframe, ["Numtppd", "Numtpbi"])
        assert "Numtppd" not in result.columns
        assert "Numtpbi" not in result.columns

    def test_ignores_missing_columns(self, sample_dataframe: pd.DataFrame) -> None:
        result = drop_columns(sample_dataframe, ["nonexistent_column"])
        assert len(result.columns) == len(sample_dataframe.columns)

    def test_preserves_other_columns(self, sample_dataframe: pd.DataFrame) -> None:
        result = drop_columns(sample_dataframe, ["Numtppd"])
        assert "Exposure" in result.columns
        assert "Power" in result.columns


class TestEncodeCategoricals:
    """Tests for the encode_categoricals function."""

    def test_original_columns_removed(self, sample_dataframe: pd.DataFrame) -> None:
        result = encode_categoricals(sample_dataframe, ["Gender", "Type"])
        assert "Gender" not in result.columns
        assert "Type" not in result.columns

    def test_dummy_columns_created(self, sample_dataframe: pd.DataFrame) -> None:
        result = encode_categoricals(sample_dataframe, ["Gender"])
        gender_cols = [c for c in result.columns if c.startswith("Gender_")]
        assert len(gender_cols) == 2  # M and F

    def test_row_count_preserved(self, sample_dataframe: pd.DataFrame) -> None:
        result = encode_categoricals(sample_dataframe, ["Gender", "Type"])
        assert len(result) == len(sample_dataframe)


class TestSplitData:
    """Tests for the split_data function."""

    def test_split_sizes(self, sample_dataframe: pd.DataFrame) -> None:
        df = create_target(sample_dataframe, "Numtppd")
        df = drop_columns(df, ["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"])
        x_train, x_test, y_train, y_test = split_data(df, "target", 0.2, 42)
        assert len(x_train) + len(x_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)

    def test_reproducibility(self, sample_dataframe: pd.DataFrame) -> None:
        df = create_target(sample_dataframe, "Numtppd")
        df = drop_columns(df, ["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"])

        x1, _, _, _ = split_data(df, "target", 0.2, 42)
        x2, _, _, _ = split_data(df, "target", 0.2, 42)
        pd.testing.assert_frame_equal(x1, x2)

    def test_target_column_excluded_from_features(self, sample_dataframe: pd.DataFrame) -> None:
        df = create_target(sample_dataframe, "Numtppd")
        df = drop_columns(df, ["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"])
        x_train, _, _, _ = split_data(df, "target", 0.2, 42)
        assert "target" not in x_train.columns
