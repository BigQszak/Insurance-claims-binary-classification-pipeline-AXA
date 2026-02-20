"""Unit tests for dataset download with mocked network access."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from axa_ml.data.download import download_dataset


def _make_fake_rda_response(rda_key: str = "pg15training") -> MagicMock:
    """Create a mocked requests.Response containing a minimal R data frame."""
    # Build a tiny CSV, then simulate what rdata.read_rda would return.
    # Since we can't easily create a real .rda in tests, we mock
    # rdata.read_rda instead of the full HTTP content.
    response = MagicMock()
    response.status_code = 200
    response.content = b"fake-rda-bytes"
    response.raise_for_status = MagicMock()
    return response


class TestDownloadDataset:
    """Tests for the download_dataset function."""

    def test_skips_if_file_exists(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pg15training.csv"
        csv_path.write_text("col1,col2\n1,2\n")

        result = download_dataset(
            url="https://example.com/data.rda",
            output_dir=tmp_path,
        )
        assert result == csv_path

    def test_force_redownloads(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pg15training.csv"
        csv_path.write_text("old data")

        fake_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        mock_response = _make_fake_rda_response()

        with (
            patch("axa_ml.data.download.requests.get", return_value=mock_response),
            patch("axa_ml.data.download.rdata.read_rda", return_value={"pg15training": fake_df}),
        ):
            result = download_dataset(
                url="https://example.com/data.rda",
                output_dir=tmp_path,
                force=True,
            )

        assert result == csv_path
        # Verify the file was overwritten with new data
        loaded = pd.read_csv(result)
        assert len(loaded) == 2

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "nested" / "raw"

        fake_df = pd.DataFrame({"x": [10]})
        mock_response = _make_fake_rda_response()

        with (
            patch("axa_ml.data.download.requests.get", return_value=mock_response),
            patch("axa_ml.data.download.rdata.read_rda", return_value={"pg15training": fake_df}),
        ):
            result = download_dataset(
                url="https://example.com/data.rda",
                output_dir=nested_dir,
            )

        assert result.exists()
        assert result.parent == nested_dir

    def test_uses_custom_filename(self, tmp_path: Path) -> None:
        fake_df = pd.DataFrame({"a": [1]})
        mock_response = _make_fake_rda_response()

        with (
            patch("axa_ml.data.download.requests.get", return_value=mock_response),
            patch("axa_ml.data.download.rdata.read_rda", return_value={"mykey": fake_df}),
        ):
            result = download_dataset(
                url="https://example.com/data.rda",
                output_dir=tmp_path,
                filename="custom_data.csv",
                rda_key="mykey",
            )

        assert result.name == "custom_data.csv"
        assert result.exists()

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with (
            patch("axa_ml.data.download.requests.get", return_value=mock_response),
            pytest.raises(requests.HTTPError),
        ):
            download_dataset(
                url="https://example.com/bad.rda",
                output_dir=tmp_path,
            )
