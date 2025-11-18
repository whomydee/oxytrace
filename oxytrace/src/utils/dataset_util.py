from pathlib import Path
from typing import Optional, Tuple
import zipfile

import gdown
import pandas as pd

from .custom_logger import LOGGER
from .env_config import AppConfig


class DatasetUtil:
    """Utility class for downloading and loading datasets."""

    @staticmethod
    def get_dataset_path(filename: str = "dataset.csv") -> Path:
        """
        Get the path to the dataset file.

        Args:
            filename: Name of the dataset file

        Returns:
            Path: Path to the dataset file
        """
        return Path(__file__).parent.parent.parent / "dataset" / filename

    @staticmethod
    def download_dataset(output_filename: str = "dataset.csv") -> str:
        """
        Downloads a CSV dataset from Google Drive and stores it in oxytrace/dataset directory.
        If the downloaded file is a ZIP archive, it will be extracted automatically.

        Args:
            output_filename: Name of the output file (default: "dataset.csv")

        Returns:
            str: Path to the downloaded CSV file

        Raises:
            ValueError: If DATASET_URL is not configured in environment variables
        """
        config = AppConfig()

        if not config.dataset_url:
            raise ValueError("DATASET_URL is not configured in environment variables")

        # Create dataset directory if it doesn't exist
        dataset_dir = Path(__file__).parent.parent.parent / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download to a temporary file first
        temp_download = dataset_dir / "temp_download"

        LOGGER.info("Downloading dataset from Google Drive", url=config.dataset_url)

        # Download the file
        gdown.download(config.dataset_url, str(temp_download), quiet=False, fuzzy=True)

        # Check if it's a ZIP file
        if zipfile.is_zipfile(temp_download):
            LOGGER.info("Downloaded file is a ZIP archive, extracting...")

            with zipfile.ZipFile(temp_download, "r") as zip_ref:
                # List files in ZIP
                file_list = zip_ref.namelist()
                LOGGER.info("Files in ZIP archive", files=file_list)

                # Find CSV files
                csv_files = [f for f in file_list if f.endswith(".csv")]

                if not csv_files:
                    raise ValueError("No CSV file found in the ZIP archive")

                # Extract the first CSV file
                csv_file = csv_files[0]
                LOGGER.info("Extracting CSV file", csv_file=csv_file)

                # Extract to dataset directory
                zip_ref.extract(csv_file, dataset_dir)

                # Rename to output_filename
                extracted_path = dataset_dir / csv_file
                output_path = dataset_dir / output_filename

                if extracted_path != output_path:
                    extracted_path.rename(output_path)

                LOGGER.info("CSV file extracted successfully", output_path=str(output_path))

            # Remove the temp ZIP file
            temp_download.unlink()
        else:
            # Not a ZIP, just rename to output filename
            output_path = dataset_dir / output_filename
            temp_download.rename(output_path)
            LOGGER.info("Dataset downloaded successfully", output_path=str(output_path))

        return str(output_path)

    @staticmethod
    def load_dataset(
        filename: str = "dataset.csv",
        num_rows: Optional[int] = None,
        percent_of_data: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file with optional row limiting.

        Args:
            filename: Name of the dataset file (default: "dataset.csv")
            num_rows: Number of rows to load (takes precedence over percent_of_data)
            percent_of_data: Percentage of data to load (0-100)

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If percent_of_data is not between 0 and 100
        """
        dataset_path = DatasetUtil.get_dataset_path(filename)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {dataset_path}. " f"Please run DatasetUtil.download_dataset() first."
            )

        # Load full dataset if no restrictions
        if num_rows is None and percent_of_data is None:
            LOGGER.info("Loading full dataset", path=str(dataset_path))
            df = pd.read_csv(dataset_path)
            LOGGER.info("Dataset loaded successfully", rows=len(df), columns=len(df.columns))
            return df

        # Calculate number of rows to load
        rows_to_load = num_rows

        if rows_to_load is None and percent_of_data is not None:
            if not 0 < percent_of_data <= 100:
                raise ValueError("percent_of_data must be between 0 and 100")

            # Get total rows count efficiently - read just first column
            temp_df = pd.read_csv(dataset_path, usecols=[0])
            total_rows = len(temp_df)
            rows_to_load = int((percent_of_data / 100) * total_rows)
            LOGGER.info(
                "Loading dataset by percentage",
                path=str(dataset_path),
                percent=percent_of_data,
                total_rows=total_rows,
                rows_to_load=rows_to_load,
            )

        # Load only the required number of rows
        if rows_to_load is not None:
            LOGGER.info("Loading partial dataset", path=str(dataset_path), rows_to_load=rows_to_load)
            df = pd.read_csv(dataset_path, nrows=rows_to_load)
            LOGGER.info("Dataset loaded successfully", rows=len(df), columns=len(df.columns))
            return df

        # Fallback
        df = pd.read_csv(dataset_path)
        return df
    
    @staticmethod
    def preprocess_dataset(df: pd.DataFrame, remove_null_oxygen: bool = True) -> pd.DataFrame:
        """
        Preprocess the dataset for time series analysis.
        
        Args:
            df: Raw dataframe with 'time' and 'Oxygen[%sat]' columns
            validate: Whether to perform validation checks
            remove_null_oxygen: Whether to filter out rows with NULL oxygen values
            
        Returns:
            pd.DataFrame: Preprocessed dataframe with datetime index and features
        """
        LOGGER.info("Starting dataset preprocessing", rows=len(df))
        
        df_processed = df.copy()
        
        # Convert time column to datetime
        if 'time' in df_processed.columns:
            df_processed['time'] = pd.to_datetime(df_processed['time'])
            LOGGER.info("Converted 'time' column to datetime")
        else:
            raise ValueError("Dataset must contain 'time' column")
        
        # Filter NULL oxygen values BEFORE sorting
        if remove_null_oxygen and 'Oxygen[%sat]' in df_processed.columns:
            null_count = df_processed['Oxygen[%sat]'].isna().sum()
            null_pct = (null_count / len(df_processed)) * 100
            
            LOGGER.info(
                "Filtering NULL oxygen values",
                null_count=null_count,
                null_percentage=f"{null_pct:.2f}%",
                rows_before=len(df_processed)
            )
            
            df_processed = df_processed[df_processed['Oxygen[%sat]'].notna()].copy()
            
            LOGGER.info(
                "NULL oxygen values removed",
                rows_after=len(df_processed),
                rows_removed=null_count
            )
            
            if len(df_processed) == 0:
                raise ValueError("No valid oxygen readings found in dataset!")
        
        # Sort by time
        df_processed = df_processed.sort_values('time').reset_index(drop=True)
        
        LOGGER.info(
            "Preprocessing complete",
            rows=len(df_processed),
            columns=list(df_processed.columns)
        )
        
        return df_processed

    @staticmethod
    def split_train_test_val(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets chronologically.
        
        Args:
            df: Preprocessed dataframe sorted by time
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        LOGGER.info(
            "Split dataset",
            total_rows=n,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df)
        )
        
        return train_df, val_df, test_df