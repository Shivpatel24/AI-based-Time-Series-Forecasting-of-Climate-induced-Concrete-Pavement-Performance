import os
import glob
import re
import random
import numpy as np
import pandas as pd
import joblib
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from concurrent.futures import ThreadPoolExecutor

# === SETTINGS ===
data_root = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data'
model_save_path = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/LSTM/Model/lstm_tl_seq3_albx_2layer.pth'
scaler_folder = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/LSTM/Scalers'
scaler_save_path = os.path.join(scaler_folder, 'lstm_tl_seq3_albx_2layer.save')

batch_size = 1024
epochs = 3
learning_rate = 0.01
seq_len = 24
seed = 42
num_workers = 4

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def make_sequences(df, seq_len):
    arr = df.drop(columns='TL').to_numpy(dtype=np.float32)
    targets = df['TL'].to_numpy(dtype=np.float32)
    n_samples = len(df) - seq_len
    X = np.stack([arr[i:i+seq_len] for i in range(n_samples)], dtype=np.float32)
    y = targets[seq_len:]
    return X, y

def process_file(fp, depth, alb):
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

        del df
        gc.collect()
        return (X_train, y_train, X_test, y_test)
    except:
        return None

# === Prepare tasks ===
all_folders = glob.glob(os.path.join(data_root, 'Output_hourly_h*_alb*'))
tasks = []
for folder in all_folders:
    m = re.search(r'Output_hourly_h(\d+)_alb(\d+)', os.path.basename(folder))
    if m:
        depth = int(m.group(1))
        alb = int(m.group(2)) / 100.0
        for fp in glob.glob(os.path.join(folder, '*.csv')):
            tasks.append((fp, depth, alb))

# === Parallel Processing ===
train_seq_list, test_seq_list = [], []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
X_train = np.vstack([x for x, _ in train_seq_list]).astype(np.float32)
y_train = np.hstack([y for _, y in train_seq_list]).astype(np.float32)
X_test = np.vstack([x for x, _ in test_seq_list]).astype(np.float32)
y_test = np.hstack([y for _, y in test_seq_list]).astype(np.float32)

train_seq_list, test_seq_list = None, None
gc.collect()

# === Shuffle training set ===
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train, y_train = X_train[indices], y_train[indices]

# === Feature scaling ===
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

# === Dataset and DataLoader ===
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

# Free memory after dataset preparation
del X_train, X_test, y_train, y_test
gc.collect()

# === MODEL ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=12, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = LSTMModel(input_size=n_features, hidden_size=12, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# === TRAIN ===
model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_loss:.6f}", flush=True)

# === EVALUATE ===
model.eval()
y_pred = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        y_pred.append(preds)
y_pred = np.concatenate(y_pred).flatten()

rmse = np.sqrt(mean_squared_error(test_dataset.y.numpy(), y_pred))
mae = mean_absolute_error(test_dataset.y.numpy(), y_pred)
r2 = r2_score(test_dataset.y.numpy(), y_pred)

print("\n--- Evaluation on Test Split ---", flush=True)
print(f"RMSE: {rmse:.4f}", flush=True)
print(f"MAE:  {mae:.4f}", flush=True)
print(f"R²:   {r2:.4f}", flush=True)

torch.save(model.state_dict(), model_save_path)
print(f"Saved model to: {model_save_path}", flush=True)
print(f"Saved scaler(s) to: {scaler_save_path}", flush=True)
