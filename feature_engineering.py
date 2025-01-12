"""
feature_engineering.py

This module performs feature engineering to enhance the predictive accuracy of mutual fund forecasting models.
Author: Satej
"""

import pandas as pd
import numpy as np

def calculate_rolling_features(df, column, window_sizes):
    """
    Calculate rolling averages and rolling standard deviations for specified window sizes.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        column (str): Column name for which rolling features are calculated.
        window_sizes (list): List of window sizes for rolling calculations.

    Returns:
        pd.DataFrame: Dataset with new rolling feature columns added.
    """
    for window in window_sizes:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
    return df


def calculate_volatility(df, column):
    """
    Calculate volatility as the percentage change in NAV over time.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        column (str): Column name for which volatility is calculated.

    Returns:
        pd.DataFrame: Dataset with a new 'volatility' column added.
    """
    df['volatility'] = df[column].pct_change().rolling(window=10).std()
    return df


def calculate_sharpe_ratio(df, return_column, risk_free_rate=0.02):
    """
    Calculate the Sharpe ratio for the dataset.

    Args:
        df (pd.DataFrame): Dataset with a return column.
        return_column (str): Column representing returns.
        risk_free_rate (float): Risk-free rate for calculating Sharpe ratio.

    Returns:
        pd.DataFrame: Dataset with a new 'sharpe_ratio' column added.
    """
    df['sharpe_ratio'] = (df[return_column] - risk_free_rate) / df['volatility']
    return df


def create_lag_features(df, column, lags):
    """
    Create lag features to capture the influence of previous values.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        column (str): Column name for which lag features are created.
        lags (int): Number of lag features to create.

    Returns:
        pd.DataFrame: Dataset with new lag feature columns added.
    """
    for lag in range(1, lags + 1):
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


if __name__ == "__main__":
    # Example usage
    input_file = "satej/cleaned_mutual_fund_data.csv"
    output_file = "satej/engineered_mutual_fund_data.csv"

    print("Loading cleaned data...")
    data = pd.read_csv(input_file, index_col='date', parse_dates=True)

    print("Calculating rolling features...")
    data = calculate_rolling_features(data, column='nav', window_sizes=[7, 30, 90])

    print("Calculating volatility...")
    data = calculate_volatility(data, column='nav')

    print("Calculating Sharpe ratio...")
    data = calculate_sharpe_ratio(data, return_column='nav')

    print("Creating lag features...")
    data = create_lag_features(data, column='nav', lags=5)

    print("Saving engineered data...")
    data.to_csv(output_file)
    print(f"Engineered data saved to: {output_file}")
