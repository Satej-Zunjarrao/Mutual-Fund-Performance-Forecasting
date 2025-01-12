"""
eda_visualization.py

This module performs Exploratory Data Analysis (EDA) and generates visualizations for mutual fund analysis.
Author: Satej
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_nav_trends(df):
    """
    Plot the trends of Net Asset Value (NAV) over time.

    Args:
        df (pd.DataFrame): Preprocessed dataset with a datetime index and 'nav' column.
    """
    if 'nav' not in df.columns:
        raise KeyError("Dataset must contain an 'nav' column.")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['nav'], label='NAV', color='blue')
    plt.title("NAV Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Net Asset Value (NAV)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_correlation_matrix(df):
    """
    Plot a heatmap of the correlation matrix for the dataset.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
    """
    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def plot_rolling_average(df, window_size=30):
    """
    Plot the rolling average of NAV for a given window size.

    Args:
        df (pd.DataFrame): Preprocessed dataset with a 'nav' column.
        window_size (int): Window size for calculating the rolling average.
    """
    if 'nav' not in df.columns:
        raise KeyError("Dataset must contain a 'nav' column.")

    rolling_avg = df['nav'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['nav'], label='NAV', color='blue')
    plt.plot(df.index, rolling_avg, label=f'Rolling Average (Window={window_size})', color='orange')
    plt.title(f"Rolling Average of NAV (Window Size = {window_size})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    input_file = "satej/cleaned_mutual_fund_data.csv"

    print("Loading cleaned data...")
    data = pd.read_csv(input_file, index_col='date', parse_dates=True)

    print("Plotting NAV trends...")
    plot_nav_trends(data)

    print("Plotting correlation matrix...")
    plot_correlation_matrix(data)

    print("Plotting rolling average...")
    plot_rolling_average(data, window_size=30)
