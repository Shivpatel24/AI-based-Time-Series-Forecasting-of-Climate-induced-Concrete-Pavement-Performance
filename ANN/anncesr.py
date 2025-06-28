import os
import glob
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# SETTINGS
data_root = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data'
model_save_path = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/ANN/Model/ann_cesr_.pth'
scaler_folder = '/home/bel/Desktop/Shiv_SRIP/ATenLoc/ANN/scalersann'
scaler_save_path = os.path.join(scaler_folder, 'ann_cesr_.save')

batch_size = 512
epochs = 3
learning_rate = 1e-2
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === LOAD + FEATURE ENGINEER DATA FROM ALL DEPTH-ALBEDO FOLDERS ===
train_data, test_data = [], []
all_folders = glob.glob(os.path.join(data_root, 'Output_hourly_h*_alb*'))

for folder in all_folders:
    m = re.search(r'Output_hourly_h(\d+)_alb(\d+)', os.path.basename(folder))
    if not m:
        continue
    depth = int(m.group(1))
    alb = int(m.group(2)) / 100.0

    all_files = glob.glob(os.path.join(folder, '*.csv'))
    if not all_files:
        continue

    for fp in all_files:
        df = pd.read_csv(fp)
        if df.empty or {'Year','Month','Day','Hour','AirTemp','Wind','Tdp','Solar','CESR'}.difference(df.columns):
            continue  # skip empty or invalid files

        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['HourOfDay'] = df['Date'].dt.hour
        df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)
        df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)
        df['sin_hour'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)

        df = df[['AirTemp','Wind','Tdp','Solar','sin_doy','cos_doy','sin_hour','cos_hour','CESR']].dropna().reset_index(drop=True)
        df['Depth'] = depth
        df['Albedo'] = alb

        if df.shape[0] < 10:
            continue

        split_idx = int(0.8 * len(df))
        train_part = df.iloc[:split_idx]
        test_part = df.iloc[split_idx:]

        train_part = train_part.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle only the training part
        train_data.append(train_part)
        test_data.append(test_part)

if not train_data or not test_data:
    raise ValueError("No data found for training or testing!")

final_train_df = pd.concat(train_data, ignore_index=True)
final_test_df = pd.concat(test_data, ignore_index=True)

X_train = final_train_df.drop(columns='CESR').values
y_train = final_train_df['CESR'].values
X_test = final_test_df.drop(columns='CESR').values
y_test = final_test_df['CESR'].values

# === SCALE FEATURES ===
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
os.makedirs(scaler_folder, exist_ok=True)
joblib.dump(scaler, scaler_save_path)

# === Prepare Data for PyTorch ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === MODEL DEFINITION (with tanh activation) ===
class ANN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 12),
            nn.Tanh(),
            nn.BatchNorm1d(12),
            nn.Linear(12, 1)
        )
    def forward(self, x):
        return self.net(x)

model = ANN(X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.MSELoss()

# === TRAINING LOOP ===
model.train()
for epoch in range(epochs):
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
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), model_save_path)
print(f"Saved model to: {model_save_path}")
print(f"Saved scaler to: {scaler_save_path}")

# === EVALUATION ===
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy().flatten()
        y_true.extend(yb.numpy().flatten())
        y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")
