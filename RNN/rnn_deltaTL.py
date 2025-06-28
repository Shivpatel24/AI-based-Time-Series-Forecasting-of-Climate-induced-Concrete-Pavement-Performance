import os
import glob
import re
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from concurrent.futures import ThreadPoolExecutor

# SETTINGS
data_root = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data'
model_save_path = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/RNN/Model/rnn_tl_seq3_albx.pth'
scaler_folder = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/RNN/scalers'
scaler_save_path = os.path.join(scaler_folder, 'rnn_tl_seq3_albx.save')

batch_size = 1024  # Increased for faster utilization
epochs = 3
learning_rate = 0.01
seq_len = 3
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def make_sequences(df, seq_len):
    arr = df.drop(columns='TL').to_numpy()
    targets = df['TL'].to_numpy()
    n_samples = len(df) - seq_len
    X = np.stack([arr[i:i+seq_len] for i in range(n_samples)])
    y = targets[seq_len:]
    return X, y

# === LOAD AND PROCESS DATA IN PARALLEL ===
def process_file(fp, depth, alb, seq_len=2):
    try:
        df = pd.read_csv(fp)
        if df.empty or {'Year','Month','Day','Hour','AirTemp','Wind','Tdp','Solar','TL'}.difference(df.columns):
            return None

        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['HourOfDay'] = df['Date'].dt.hour
        df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)
        df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)
        df['sin_hour'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
        df = df[['AirTemp','Wind','Tdp','Solar','sin_doy','cos_doy','sin_hour','cos_hour','TL']].dropna().reset_index(drop=True)
        df['Depth'] = depth
        df['Albedo'] = alb

        if len(df) < seq_len + 2:
            return None

        split_idx = int(0.8 * len(df))
        X_train, y_train = make_sequences(df.iloc[:split_idx], seq_len)
        X_test, y_test = make_sequences(df.iloc[split_idx:], seq_len)

        return (X_train, y_train, X_test, y_test)
    except Exception as e:
        return None

all_folders = glob.glob(os.path.join(data_root, 'Output_hourly_h*_alb*'))
tasks = []
for folder in all_folders:
    m = re.search(r'Output_hourly_h(\d+)_alb(\d+)', os.path.basename(folder))
    if not m:
        continue
    depth = int(m.group(1))
    alb = int(m.group(2)) / 100.0
    for fp in glob.glob(os.path.join(folder, '*.csv')):
        tasks.append((fp, depth, alb))

train_seq_list, test_seq_list = [], []
with ThreadPoolExecutor(max_workers=8) as executor:
    for result in executor.map(lambda args: process_file(*args), tasks):
        if result:
            X_train, y_train, X_test, y_test = result
            if X_train.size and y_train.size:
                train_seq_list.append((X_train, y_train))
            if X_test.size and y_test.size:
                test_seq_list.append((X_test, y_test))

if not train_seq_list or not test_seq_list:
    raise ValueError("No data found for training or testing!")

# === Aggregate arrays ===
X_train = np.vstack([x for x, _ in train_seq_list])
y_train = np.hstack([y for _, y in train_seq_list])
X_test = np.vstack([x for x, _ in test_seq_list])
y_test = np.hstack([y for _, y in test_seq_list])

# === Shuffle train ===
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train, y_train = X_train[indices], y_train[indices]

# === SCALE FEATURES ===
n_features = X_train.shape[2]
scalers = []
for i in range(n_features):
    scaler = RobustScaler()
    scaler.fit(X_train[:, :, i])
    X_train[:, :, i] = scaler.transform(X_train[:, :, i])
    X_test[:, :, i] = scaler.transform(X_test[:, :, i])
    scalers.append(scaler)

os.makedirs(scaler_folder, exist_ok=True)
joblib.dump(scalers, scaler_save_path)

# === DATA TO TENSORS ===
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# === MODEL ===
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=12, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

model = RNNModel(input_size=n_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# === TRAINING ===
model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().numpy() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_loss:.6f}", flush=True)

# === EVALUATION ===
model.eval()
y_pred = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device, non_blocking=True)
        y_pred.append(model(xb).cpu().numpy())
y_pred = np.concatenate(y_pred).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation on Test Split ---", flush=True)
print(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}", flush=True)

torch.save(model.state_dict(), model_save_path)
print(f"Saved model to: {model_save_path}\nSaved scaler(s) to: {scaler_save_path}", flush=True)
