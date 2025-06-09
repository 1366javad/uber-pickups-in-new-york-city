"""
data_ingestion.py

This module handles the data ingestion process for the Uber stock dataset.
It performs the following steps:
- Downloads the dataset from a cloud storage bucket
- Cleans and preprocesses the raw stock data (e.g., date conversion, handling missing values)
- Splits the data into chronological train, validation, and test sets
- Saves the processed datasets into CSV files
"""

import logging
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# No need for sklearn.model_selection.train_test_split as we'll do time-based split
# No need for sklearn.preprocessing.LabelEncoder for stock data

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    A class to manage the full data ingestion pipeline for stock data, including downloading,
    cleaning, splitting, and saving datasets.
    """

    def __init__(self, config):
        """
        Initializes the DataIngestion instance with configuration.

        Args:
            config (dict): Configuration dictionary parsed from YAML file.
        """
        self.data_ingestion_config = config["data_ingestion"]
        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.object_name = self.data_ingestion_config["object_name"]
        self.storage_path = self.data_ingestion_config["storage_path"]

        # Adjust ratios for time-based split
        # We'll use these to determine split points based on percentage of total data
        self.train_ratio = self.data_ingestion_config.get(
            "train_ratio", 0.7
        )  # default to 70%
        self.val_ratio = self.data_ingestion_config.get(
            "val_ratio", 0.15
        )  # default to 15% (remaining 15% for test)
        # Ensure train_ratio + val_ratio < 1 for test set to exist

        # Construct the full URL
        # Assuming storage_path is something like 'storage.arvancloud.com'
        # And object_name is the path within the bucket, like 'uber_stock_data.csv'
        self.url = f"https://{self.bucket_name}.{self.storage_path}/{self.object_name}"

        # Define artifact directories
        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.artifact_dir = artifact_dir
        self.raw_dir = artifact_dir / "raw"
        self.processed_dir = artifact_dir / "processed"  # For train/val/test splits

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self) -> pd.DataFrame:
        """
        Downloads the raw CSV stock data from the cloud URL.

        Returns:
            pd.DataFrame: The raw data loaded into a DataFrame.
        """
        logger.info("Downloading raw stock data...")
        try:
            with urlopen(self.url) as response:
                raw_data = response.read().decode("utf-8")
                df = pd.read_csv(StringIO(raw_data))
                logger.info(
                    f"Downloaded dataset with {df.shape[0]} rows and {df.shape[1]} columns."
                )
                return df
        except Exception as e:
            logger.error(f"Error downloading data from {self.url}: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the raw stock dataset.
        - Converts 'Date' column to datetime and sets as index.
        - Sorts data by date.
        - Removes rows with any missing values.
        - Selects only relevant stock columns.

        Args:
            df (pd.DataFrame): Raw stock dataset.

        Returns:
            pd.DataFrame: Cleaned and prepared stock dataset.
        """
        logger.info("Cleaning and preprocessing stock data...")

        # Ensure 'Date' column exists and convert to datetime
        if "Date" not in df.columns:
            logger.error("'Date' column not found in the raw data.")
            raise ValueError("'Date' column is required for stock data processing.")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Remove rows where Date conversion failed (NaT)
        df = df.dropna(subset=["Date"])

        df = df.set_index("Date")

        # Sort data by date
        df = df.sort_index()

        # Remove any other missing values that might exist in financial columns
        df.dropna(inplace=True)

        # Select relevant stock columns
        # Assuming you have 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        # Check if all expected columns are present
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Missing expected stock columns: {missing_cols}. Attempting to proceed with available columns."
            )
            # Adjust expected_cols to only include columns actually present in df
            cols_to_keep = [col for col in expected_cols if col in df.columns]
        else:
            cols_to_keep = expected_cols

        df_clean = df[cols_to_keep].copy()

        logger.info(
            f"Cleaned data with {df_clean.shape[0]} rows and {df_clean.shape[1]} columns."
        )
        return df_clean

    def split_data(self, df: pd.DataFrame):
        """
        Splits the dataset into train, validation, and test sets chronologically.

        Args:
            df (pd.DataFrame): Cleaned stock dataset.

        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        logger.info(
            "Splitting data into chronological train, validation, and test sets..."
        )

        total_rows = len(df)
        train_end_idx = int(total_rows * self.train_ratio)
        val_end_idx = int(total_rows * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end_idx].copy()
        val_df = df.iloc[train_end_idx:val_end_idx].copy()
        test_df = df.iloc[val_end_idx:].copy()

        logger.info(
            f"Train size: {len(train_df)} - Val size: {len(val_df)} - Test size: {len(test_df)}"
        )

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            logger.warning(
                "One or more split datasets are empty. Check your ratios and data size."
            )

        return train_df, val_df, test_df

    def save_to_csv_files(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ):
        """
        Saves the train, validation, and test datasets to CSV files in the processed directory.

        Args:
            train (pd.DataFrame): Training dataset.
            val (pd.DataFrame): Validation dataset.
            test (pd.DataFrame): Test dataset.
        """
        logger.info("Saving processed data to CSV files...")

        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "validation.csv"
        test_path = self.processed_dir / "test.csv"

        train.to_csv(train_path, index=True)  # Keep index (Date)
        val.to_csv(val_path, index=True)  # Keep index (Date)
        test.to_csv(test_path, index=True)  # Keep index (Date)

        logger.info(f"Train data saved to: {train_path}")
        logger.info(f"Validation data saved to: {val_path}")
        logger.info(f"Test data saved to: {test_path}")
        logger.info("Processed data saved successfully.")

    def run(self):
        """
        Runs the full data ingestion pipeline.
        """
        logger.info("Starting data ingestion pipeline for Uber stock data.")
        df = self.download_raw_data()

        # Save raw data for inspection/debugging if needed
        raw_output_path = self.raw_dir / self.object_name
        df.to_csv(
            raw_output_path, index=False
        )  # raw data often doesn't need date as index yet
        logger.info(f"Raw downloaded data saved to: {raw_output_path}")

        df_clean = self.clean_data(df)
        train, val, test = self.split_data(df_clean)
        self.save_to_csv_files(train, val, test)
        logger.info("Data ingestion completed successfully for Uber stock data.")
