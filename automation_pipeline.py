"""
automation_pipeline.py

This module automates the entire pipeline, including data updates, model retraining, and dashboard refresh.
Author: Satej
"""

import os
import pandas as pd
from data_preprocessing import load_data, clean_data, preprocess_data
from feature_engineering import calculate_rolling_features, calculate_volatility, calculate_sharpe_ratio, create_lag_features
from model_training import train_linear_regression, train_arima, train_lstm
from hyperparameter_tuning import tune_arima

def update_data(source_file, cleaned_file):
    """
    Automate the process of updating and cleaning new data.

    Args:
        source_file (str): Path to the raw data file.
        cleaned_file (str): Path to save the cleaned data.
    """
    print("Updating data...")
    raw_data = load_data(source_file)
    cleaned_data = clean_data(raw_data)
    preprocessed_data = preprocess_data(cleaned_data)
    preprocessed_data.to_csv(cleaned_file)
    print(f"Data updated and saved to {cleaned_file}")


def engineer_features(cleaned_file, engineered_file):
    """
    Automate the process of feature engineering.

    Args:
        cleaned_file (str): Path to the cleaned data file.
        engineered_file (str): Path to save the feature-engineered data.
    """
    print("Engineering features...")
    data = pd.read_csv(cleaned_file, index_col='date', parse_dates=True)

    data = calculate_rolling_features(data, column='nav', window_sizes=[7, 30, 90])
    data = calculate_volatility(data, column='nav')
    data = calculate_sharpe_ratio(data, return_column='nav')
    data = create_lag_features(data, column='nav', lags=5)

    data.to_csv(engineered_file)
    print(f"Features engineered and saved to {engineered_file}")


def retrain_models(engineered_file, model_save_path):
    """
    Automate the process of retraining models.

    Args:
        engineered_file (str): Path to the feature-engineered data.
        model_save_path (str): Path to save trained models.
    """
    print("Retraining models...")
    data = pd.read_csv(engineered_file, index_col='date', parse_dates=True)
    features = [col for col in data.columns if 'lag' in col or 'rolling' in col]
    X = data[features].dropna()
    y = data['nav'][X.index]

    # Retrain Linear Regression
    print("Training Linear Regression model...")
    lr_model = train_linear_regression(X, y)

    # Retrain ARIMA with hyperparameter tuning
    print("Training ARIMA model...")
    arima_results = tune_arima(y, p_values=[1, 2, 3], d_values=[0, 1], q_values=[1, 2])
    arima_model = arima_results['model']

    # Save models
    os.makedirs(model_save_path, exist_ok=True)
    pd.to_pickle(lr_model, os.path.join(model_save_path, "linear_regression.pkl"))
    arima_model.save(os.path.join(model_save_path, "arima_model.pkl"))

    print(f"Models saved to {model_save_path}")


if __name__ == "__main__":
    source_file = "satej/mutual_fund_data.csv"
    cleaned_file = "satej/cleaned_mutual_fund_data.csv"
    engineered_file = "satej/engineered_mutual_fund_data.csv"
    model_save_path = "satej/models/"

    print("Starting automation pipeline...")
    update_data(source_file, cleaned_file)
    engineer_features(cleaned_file, engineered_file)
    retrain_models(engineered_file, model_save_path)
    print("Pipeline execution completed.")
