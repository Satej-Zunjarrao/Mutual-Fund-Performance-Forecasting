"""
utils.py

This module provides utility functions for common tasks such as data transformations,
error metrics, and visualization helpers.
Author: Satej
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

    Returns:
        float: MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_predictions(y_true, y_pred, title="Predictions vs True Values"):
    """
    Plot predicted vs true values.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='orange')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_plot_as_image(plot_func, output_path, *args, **kwargs):
    """
    Save a generated plot as an image file.

    Args:
        plot_func (function): Function to generate the plot.
        output_path (str): Path to save the plot image.
        *args: Arguments to pass to the plotting function.
        **kwargs: Keyword arguments to pass to the plotting function.
    """
    plot_func(*args, **kwargs)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def scale_features(X, scaler):
    """
    Scale features using a provided scaler (e.g., MinMaxScaler or StandardScaler).

    Args:
        X (pd.DataFrame or np.array): Feature matrix to be scaled.
        scaler (object): Scaler instance (e.g., sklearn.preprocessing.MinMaxScaler).

    Returns:
        np.array: Scaled features.
    """
    return scaler.fit_transform(X)


if __name__ == "__main__":
    # Example usage
    true_values = [100, 200, 300, 400, 500]
    predicted_values = [110, 190, 290, 420, 510]

    print("Calculating MAPE...")
    mape = calculate_mape(true_values, predicted_values)
    print(f"MAPE: {mape:.2f}%")

    print("Plotting predictions...")
    plot_predictions(true_values, predicted_values)

    print("Saving plot...")
    save_plot_as_image(plot_predictions, "satej/predictions_plot.png", true_values, predicted_values)
