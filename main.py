"""
main.py

This script serves as the entry point for the mutual fund performance prediction system. 
It orchestrates the entire pipeline, including data preprocessing, feature engineering, 
model training, and dashboard initialization.
Author: Satej
"""

import os
from automation_pipeline import update_data, engineer_features, retrain_models
from dashboard import app as dashboard_app

def main():
    """
    Main function to execute the mutual fund prediction system.
    """
    # Define file paths
    source_file = "satej/mutual_fund_data.csv"
    cleaned_file = "satej/cleaned_mutual_fund_data.csv"
    engineered_file = "satej/engineered_mutual_fund_data.csv"
    model_save_path = "satej/models/"

    # Ensure directories exist
    os.makedirs(os.path.dirname(cleaned_file), exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Step 1: Update and preprocess data
    print("Step 1: Updating and preprocessing data...")
    update_data(source_file, cleaned_file)

    # Step 2: Feature engineering
    print("Step 2: Engineering features...")
    engineer_features(cleaned_file, engineered_file)

    # Step 3: Retrain models
    print("Step 3: Retraining models...")
    retrain_models(engineered_file, model_save_path)

    # Step 4: Launch interactive dashboard
    print("Step 4: Launching interactive dashboard...")
    dashboard_app.run_server(debug=True, port=8050)


if __name__ == "__main__":
    print("Starting Mutual Fund Performance Prediction System...")
    main()
