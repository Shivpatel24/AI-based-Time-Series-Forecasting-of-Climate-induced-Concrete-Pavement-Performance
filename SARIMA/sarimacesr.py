import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Directory path to your data folder
data_dir = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data'

# Output CSV to store results
output_csv = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/SARIMA/output_sarimacesr.csv'

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
for file_path in tqdm(all_csv_files, desc="Processing files"):
    try:
        df = pd.read_csv(file_path)

        # Create datetime column
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        cesr_series = pd.Series(data=df['CESR'].values, index=df['datetime'], name='CESR')

        # Train-test split (80-20)
        train_size = int(len(cesr_series) * 0.8)
        train, test = cesr_series[:train_size], cesr_series[train_size:]

        # Fit SARIMA model
        model = SARIMAX(
            train,
            order=(2, 0, 1),
            seasonal_order=(1, 1, 0, 24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, low_memory=True)

        # Forecast for the test period
        forecast = results.get_forecast(steps=len(test))
        forecast_mean = forecast.predicted_mean

        # Append results to output DataFrame
        temp_df = pd.DataFrame({
            'Actual_CESR': test.values,
            'Predicted_CESR': forecast_mean.values
        })
        temp_df.to_csv(output_csv, mode='a', header=False, index=False)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print("All files processed. Results saved to:", output_csv)
