import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

# Directory path to your data folder
data_dir = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data'

# Output CSV to store results
output_csv = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/SARIMA/output_sarimax_cesr.csv'

# Initialize output file
output_df = pd.DataFrame(columns=['Actual_CESR', 'Predicted_CESR'])
output_df.to_csv(output_csv, index=False)

# Find all CSV files inside the directory structure
all_csv_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.csv'):
            all_csv_files.append(os.path.join(root, file))

# Loop over each CSV file
for file_path in tqdm(all_csv_files, desc="Processing SARIMAX files"):
    try:
        df = pd.read_csv(file_path)

        # Create datetime index with hourly frequency
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        df = df.set_index('datetime')
        df = df.asfreq('H')  # Set frequency to hourly

        # Select only the last 60% of the data
        last_60pct_index = int(len(df) * 0.4)
        df_recent = df.iloc[last_60pct_index:]

        # Target and exogenous features
        y = df_recent['CESR']
        exog_features = ['AirTemp', 'Wind', 'Tdp', 'Solar']
        exog = df_recent[exog_features]

        # Train-test split (80-20) on the last 60%
        train_size = int(len(y) * 0.8)
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]

        # Fit SARIMAX model
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(2, 0, 1),
            seasonal_order=(1, 1, 0, 24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, low_memory=True)

        # Forecast for the test period
        forecast = results.get_forecast(steps=len(y_test), exog=exog_test)
        forecast_mean = forecast.predicted_mean

        # Append results to output CSV
        temp_df = pd.DataFrame({
            'Actual_CESR': y_test.values,
            'Predicted_CESR': forecast_mean.values
        })
        temp_df.to_csv(output_csv, mode='a', header=False, index=False)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print("✅ All SARIMAX files processed. Results saved to:", output_csv)
