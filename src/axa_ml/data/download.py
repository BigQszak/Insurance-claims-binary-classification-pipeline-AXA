"""Download the insurance claims dataset from a remote URL."""

import io
from pathlib import Path

import rdata
import requests
import structlog

logger = structlog.get_logger(__name__)

_FILENAME = "pg15training.csv"


def download_dataset(url: str, output_dir: str | Path, *, force: bool = False) -> Path:
    """Download an ``.rda`` file, convert to CSV, and save locally.

    The download is idempotent — if the CSV already exists, it is skipped
    unless *force* is ``True``.

    Args:
        url: Remote URL pointing to the ``.rda`` file.
        output_dir: Local directory to save the CSV into.
        force: Re-download even if the file already exists.

    Returns:
        Path to the saved CSV file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / _FILENAME

    if csv_path.exists() and not force:
        logger.info("dataset_already_exists", path=str(csv_path))
        return csv_path

    logger.info("downloading_dataset", url=url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    buf = io.BytesIO(response.content)
    r_data = rdata.read_rda(buf)["pg15training"]

    r_data.to_csv(csv_path, index=False)
    logger.info("dataset_saved", path=str(csv_path), rows=len(r_data))

    return csv_path
