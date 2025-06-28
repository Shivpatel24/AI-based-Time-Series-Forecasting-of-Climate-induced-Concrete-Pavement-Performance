import os
import glob
import re
import numpy as np
import pandas as pd
import joblib
import random
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from concurrent.futures import ThreadPoolExecutor

# --- SETTINGS ---
data_root = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data'
model_save_path = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Xgboost/models/xgb_cesr_final.model'
scaler_save_path = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Xgboost/scalers/xgb_cesr_final.save'
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

random.seed(42)
np.random.seed(42)

# --- Feature Engineering Function ---
def extract_features_from_csv(fp, depth, albedo):
    try:
        df = pd.read_csv(fp)
        if df.empty or 'CESR' not in df.columns:
            return None

        # Time encoding
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']], errors='coerce')
        df = df.dropna(subset=['Date'])

        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['HourOfDay'] = df['Date'].dt.hour
        df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        df['sin_hour'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)

        df['Depth'] = depth
        df['Albedo'] = albedo

        features = ['AirTemp', 'Wind', 'Tdp', 'Solar', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour', 'Depth', 'Albedo']
        df = df[features + ['CESR']].dropna()

        # Train-test split within the file to preserve time structure
        split = int(0.8 * len(df))
        return df.iloc[:split], df.iloc[split:]
    except:
        return None

# --- Load All Files in Parallel ---
all_folders = glob.glob(os.path.join(data_root, 'Output_hourly_h*_alb*'))
tasks = []

for folder in all_folders:
    match = re.search(r'Output_hourly_h(\d+)_alb(\d+)', folder)
    if not match:
        continue
    depth = int(match.group(1))
    albedo = int(match.group(2)) / 100.0

    for file in glob.glob(os.path.join(folder, '*.csv')):
        tasks.append((file, depth, albedo))

train_frames = []
test_frames = []

with ThreadPoolExecutor(max_workers=8) as executor:
    for result in executor.map(lambda args: extract_features_from_csv(*args), tasks):
        if result:
            train_df, test_df = result
            train_frames.append(train_df)
            test_frames.append(test_df)

# --- Merge All Data ---
train_df = pd.concat(train_frames, ignore_index=True)
test_df = pd.concat(test_frames, ignore_index=True)

X_train = train_df.drop(columns='CESR').values
y_train = train_df['CESR'].values
X_test = test_df.drop(columns='CESR').values
y_test = test_df['CESR'].values

# --- Scale Features ---
scalers = []
X_train_scaled = np.zeros_like(X_train)
X_test_scaled = np.zeros_like(X_test)

for i in range(X_train.shape[1]):
    scaler = RobustScaler()
    X_train_scaled[:, i] = scaler.fit_transform(X_train[:, i].reshape(-1, 1)).flatten()
    X_test_scaled[:, i] = scaler.transform(X_test[:, i].reshape(-1, 1)).flatten()
    scalers.append(scaler)

# Save scalers
joblib.dump(scalers, scaler_save_path)

# --- Train XGBoost Model ---
model = XGBRegressor(
    n_estimators=152,
    learning_rate=0.1564,
    max_depth=7,
    subsample=0.778,
    gamma=4.75,
    min_child_weight = 5,
    colsample_bytree=0.75,
    random_state=42,
    tree_method='gpu_hist' if joblib.cpu_count() > 1 else 'auto'
)

model.fit(X_train_scaled, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- XGBoost Model Evaluation ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# --- Save Model ---
joblib.dump(model, model_save_path)
print(f"Model saved to {model_save_path}")
print(f"Scalers saved to {scaler_save_path}")
