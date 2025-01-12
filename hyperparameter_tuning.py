"""
hyperparameter_tuning.py

This module performs hyperparameter tuning to optimize predictive models for mutual fund analysis.
Author: Satej
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
import numpy as np


def tune_linear_regression(X_train, y_train):
    """
    Perform hyperparameter tuning for Linear Regression using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target variable for training.

    Returns:
        dict: Best parameters and trained model.
    """
    # For Linear Regression, no hyperparameters to tune, just return the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return {"model": model}


def tune_arima(train_series, p_values, d_values, q_values):
    """
    Perform hyperparameter tuning for ARIMA.

    Args:
        train_series (pd.Series): Time-series data for training.
        p_values (list): List of AR values to test.
        d_values (list): List of differencing values to test.
        q_values (list): List of MA values to test.

    Returns:
        dict: Best parameters and trained model.
    """
    best_score = np.inf
    best_params = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    if aic < best_score:
                        best_score = aic
                        best_params = (p, d, q)
                        best_model = model_fit
                except Exception as e:
                    print(f"Error with ARIMA({p},{d},{q}): {e}")

    return {"model": best_model, "best_params": best_params}


if __name__ == "__main__":
    input_file = "satej/engineered_mutual_fund_data.csv"

    print("Loading engineered data...")
    data = pd.read_csv(input_file, index_col='date', parse_dates=True)

    print("Preparing data for hyperparameter tuning...")
    features = [col for col in data.columns if 'lag' in col or 'rolling' in col]
    X = data[features].dropna()
    y = data['nav'][X.index]

    X_train = X
    y_train = y

    print("Tuning Linear Regression...")
    lr_results = tune_linear_regression(X_train, y_train)
    print(f"Best Linear Regression Model: {lr_results['model']}")

    print("Tuning ARIMA...")
    arima_results = tune_arima(y_train, p_values=[1, 2, 3], d_values=[0, 1], q_values=[1, 2])
    print(f"Best ARIMA Parameters: {arima_results['best_params']}")
