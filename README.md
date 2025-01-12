# Mutual-Fund-Performance-Prediction

Built a time-series-based system to analyze mutual fund performance and forecast future trends.

# Mutual Fund Performance Prediction System

## Overview
The **Mutual Fund Performance Prediction System** is a Python-based solution designed to analyze and forecast the performance of mutual funds using historical financial data. The system utilizes advanced machine learning techniques and time-series analysis to assist investors in making informed decisions by identifying trends, returns, and risk factors.

This project includes a modular and scalable pipeline for data collection, preprocessing, feature engineering, model training, evaluation, visualization, and automation.

---

## Key Features
- **Data Collection**: Extracts historical mutual fund data (NAV, returns, and indices) from SQL databases.
- **Data Cleaning**: Handles missing values and preprocesses time-series data for analysis.
- **Exploratory Data Analysis (EDA)**: Visualizes NAV trends, correlations, and historical returns.
- **Feature Engineering**: Creates advanced features like rolling averages, volatility, lag features, and Sharpe ratio.
- **Model Training**: Implements and trains ARIMA, Linear Regression, and LSTM models for NAV and return forecasting.
- **Visualization**: Generates interactive dashboards for visualizing forecasts, risks, and performance.
- **Automation**: Automates the pipeline for periodic updates and retraining.

---

## Directory Structure

```plaintext
project/
│
├── data_preprocessing.py       # Handles data loading, cleaning, and preprocessing
├── eda_visualization.py        # Generates visualizations and insights
├── feature_engineering.py      # Creates advanced features for predictive modeling
├── model_training.py           # Trains and evaluates machine learning models
├── hyperparameter_tuning.py    # Optimizes models for better performance
├── dashboard.py                # Builds an interactive dashboard for insights
├── automation_pipeline.py      # Automates the pipeline for data updates and retraining
├── utils.py                    # Provides helper functions for scaling, metrics, etc.
├── main.py                     # Orchestrates the entire system workflow
├── README.md                   # Project documentation
```
# Modules

## 1. data_preprocessing.py
- Loads raw historical mutual fund data from files or databases.
- Cleans data by handling missing values and aligning time-series indices.
- Outputs a preprocessed dataset for feature engineering.

## 2. eda_visualization.py
- Visualizes NAV trends, correlations, and rolling averages.
- Provides insights into mutual fund performance through heatmaps and line plots.

## 3. feature_engineering.py
- Creates rolling averages, rolling standard deviations, volatility, and Sharpe ratio features.
- Generates lag features to incorporate historical NAV data into the predictive model.

## 4. model_training.py
- Trains ARIMA, Linear Regression, and LSTM models for forecasting.
- Evaluates models using MAPE and other performance metrics.

## 5. hyperparameter_tuning.py
- Performs grid search and optimization for ARIMA and other models.
- Identifies the best model parameters for improved forecasting accuracy.

## 6. dashboard.py
- Builds an interactive dashboard using Dash.
- Visualizes forecasts, risk metrics, and historical performance for stakeholders.

## 7. automation_pipeline.py
- Automates the pipeline to update data, engineer features, and retrain models.
- Ensures the system remains current with the latest market changes.

## 8. utils.py
- Contains helper functions for scaling, calculating MAPE, and visualizing predictions.
- Includes utilities for saving and loading models or plots.

## 9. main.py
- Serves as the entry point to orchestrate the entire workflow.
- Executes data updates, model training, and launches the dashboard.

---

# Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com  
