"""
data_preprocessing.py

This module handles the data collection, cleaning, and preprocessing steps for mutual fund analysis.
Author: Satej
"""

import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean the data by handling missing values and ensuring time-series alignment.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Check for missing values
    missing_summary = df.isnull().sum()
    print(f"Missing Values Summary:\n{missing_summary}")

    # Fill missing values with forward-fill for time-series consistency
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df


def preprocess_data(df):
    """
    Perform additional preprocessing steps, such as setting datetime indices
    and sorting by time.

    Args:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    if 'date' not in df.columns:
        raise KeyError("Dataset must contain a 'date' column.")

    # Convert 'date' column to datetime format and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Ensure the data is sorted by time
    df.sort_index(inplace=True)

    return df


def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.

    Args:
        df (pd.DataFrame): Cleaned and preprocessed dataset.
        output_path (str): Path to save the processed data.
    """
    df.to_csv(output_path, index=True)
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    input_file = "satej/mutual_fund_data.csv"
    output_file = "satej/cleaned_mutual_fund_data.csv"

    print("Loading data...")
    data = load_data(input_file)

    print("Cleaning data...")
    cleaned_data = clean_data(data)

    print("Preprocessing data...")
    preprocessed_data = preprocess_data(cleaned_data)

    print("Saving cleaned data...")
    save_cleaned_data(preprocessed_data, output_file)
