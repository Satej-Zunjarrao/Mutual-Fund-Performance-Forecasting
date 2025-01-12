"""
model_training.py

This module handles the training and evaluation of predictive models for mutual fund analysis.
Author: Satej
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_percentage_error

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target variable for training.

    Returns:
        model: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_arima(train_series, order):
    """
    Train an ARIMA model.

    Args:
        train_series (pd.Series): Time-series data for training.
        order (tuple): ARIMA order (p, d, q).

    Returns:
        model: Trained ARIMA model.
    """
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit


def train_lstm(X_train, y_train, input_shape):
    """
    Train an LSTM model.

    Args:
        X_train (np.array): Feature matrix for training.
        y_train (np.array): Target variable for training.
        input_shape (tuple): Shape of the input data for LSTM.

    Returns:
        model: Trained LSTM model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    return model


if __name__ == "__main__":
    input_file = "satej/engineered_mutual_fund_data.csv"

    print("Loading engineered data...")
    data = pd.read_csv(input_file, index_col='date', parse_dates=True)

    print("Preparing data for model training...")
    features = [col for col in data.columns if 'lag' in col or 'rolling' in col]
    X = data[features].dropna()
    y = data['nav'][X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    print(f"Linear Regression MAPE: {mean_absolute_percentage_error(y_test, lr_predictions):.4f}")

    print("Training ARIMA model...")
    arima_model = train_arima(y_train, order=(5, 1, 0))
    arima_predictions = arima_model.forecast(len(y_test))
    print(f"ARIMA MAPE: {mean_absolute_percentage_error(y_test, arima_predictions):.4f}")

    print("Training LSTM model...")
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = train_lstm(X_train_lstm, y_train.values, input_shape=(X_train_lstm.shape[1], 1))
    lstm_predictions = lstm_model.predict(X_test_lstm).flatten()
    print(f"LSTM MAPE: {mean_absolute_percentage_error(y_test, lstm_predictions):.4f}")
