"""
data_ingestion.py

This module handles the data ingestion process for the Uber pickups dataset.
It performs the following steps:
- Downloads the dataset from a cloud storage bucket
- Cleans and preprocesses the data
- Extracts useful features
- Splits the data into train, validation, and test sets
- Saves the processed datasets into CSV files
"""

import logging
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    A class to manage the full data ingestion pipeline, including downloading,
    cleaning, feature engineering, splitting, and saving datasets.
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
        self.train_ratio = self.data_ingestion_config["train_ration"]

        self.url = f"https://{self.bucket_name}.{self.storage_path}/{self.object_name}"

        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.artifact_dir = artifact_dir
        self.raw_dir = artifact_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self) -> pd.DataFrame:
        """
        Downloads the raw CSV data from the cloud URL.

        Returns:
            pd.DataFrame: The raw data loaded into a DataFrame.
        """
        logger.info("Downloading raw data...")
        with urlopen(self.url) as response:
            raw_data = response.read().decode("utf-8")
            df = pd.read_csv(StringIO(raw_data))
            logger.info(
                f"Downloaded dataset with {df.shape[0]} rows and {df.shape[1]} columns."
            )
            return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the raw dataset.

        - Removes missing values
        - Encodes categorical addresses
        - Creates datetime and extracts time features
        - Creates binary target for cancellation

        Args:
            df (pd.DataFrame): Raw dataset.

        Returns:
            pd.DataFrame: Cleaned and feature-enriched dataset.
        """
        logger.info("Cleaning and preprocessing data...")

        df = df.dropna()

        df["is_cancelled"] = df["Status"].apply(
            lambda x: 1 if str(x).strip().lower() == "cancelled" else 0
        )

        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek

        le_pu = LabelEncoder()
        le_do = LabelEncoder()
        df["PU_Address_encoded"] = le_pu.fit_transform(df["PU_Address"])
        df["DO_Address_encoded"] = le_do.fit_transform(df["DO_Address"])

        df_clean = df[
            [
                "hour",
                "day_of_week",
                "PU_Address_encoded",
                "DO_Address_encoded",
                "is_cancelled",
            ]
        ].copy()
        logger.info("Data cleaned and features extracted.")
        return df_clean

    def split_data(self, df: pd.DataFrame):
        """
        Splits the dataset into train, validation, and test sets.

        Args:
            df (pd.DataFrame): Cleaned dataset.

        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        logger.info("Splitting data into train, validation, and test sets...")
        train_val, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.25, random_state=42)
        logger.info(
            f"Train size: {len(train)} - Val size: {len(val)} - Test size: {len(test)}"
        )
        return train, val, test

    def save_to_csv_files(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ):
        """
        Saves the train, validation, and test datasets to CSV files.

        Args:
            train (pd.DataFrame): Training dataset.
            val (pd.DataFrame): Validation dataset.
            test (pd.DataFrame): Test dataset.
        """
        logger.info("Saving data to CSV files...")
        train.to_csv(self.artifact_dir / "train.csv", index=False)
        val.to_csv(self.artifact_dir / "validation.csv", index=False)
        test.to_csv(self.artifact_dir / "test.csv", index=False)
        logger.info("Data saved successfully.")

    def run(self):
        """
        Runs the full data ingestion pipeline.
        """
        df = self.download_raw_data()
        df_clean = self.clean_data(df)
        train, val, test = self.split_data(df_clean)
        self.save_to_csv_files(train, val, test)
        logger.info("Data ingestion completed.")
