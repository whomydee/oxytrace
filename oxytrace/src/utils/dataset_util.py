from pathlib import Path

import gdown

from .custom_logger import LOGGER
from .env_config import AppConfig


def download_dataset(output_filename: str = "dataset.csv") -> str:
    """
    Downloads a CSV dataset from Google Drive and stores it in oxytrace/dataset directory.

    Args:
        output_filename: Name of the output file (default: "dataset.csv")

    Returns:
        str: Path to the downloaded file

    Raises:
        ValueError: If DATASET_URL is not configured in environment variables
    """
    config = AppConfig()

    if not config.dataset_url:
        raise ValueError("DATASET_URL is not configured in environment variables")

    # Create dataset directory if it doesn't exist
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Full path for the output file
    output_path = dataset_dir / output_filename

    LOGGER.info("Downloading dataset from Google Drive", url=config.dataset_url, output_path=str(output_path))

    # Download the file
    gdown.download(config.dataset_url, str(output_path), quiet=False, fuzzy=True)

    LOGGER.info("Dataset downloaded successfully", output_path=str(output_path))
    return str(output_path)


if __name__ == "__main__":
    # Test the download function
    download_dataset()
