import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from neuralprophet import NeuralProphet, save, load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def check_cuda():
    if torch.cuda.is_available():
        print(f"  CUDA is available. NeuralProphet will use GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(" CUDA is NOT available. NeuralProphet will use CPU.")

def preprocess_file(file_path, station_id):
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    df = df[['ds', 'AirTemp', 'Wind', 'Tdp', 'Solar', 'TL']]
    df = df.sort_values('ds').drop_duplicates(subset='ds')
    df = df.rename(columns={'TL': 'y'})
    df['ID'] = station_id
    return df

def load_all_data(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_dfs = []
    for idx, file in enumerate(csv_files):
        station_id = f"station_{idx+1}"
        df = preprocess_file(os.path.join(folder_path, file), station_id)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def train_global_model(df,
                       model_save_path="/home/bel/Desktop/Shiv_SRIP/Saved Models/global_model_TL.np",
                       save_train_data_path="training_data_processed.csv"):
    check_cuda()  # Just check and print whether CUDA is available

    train_list, test_list = [], []
    for station_id in df['ID'].unique():
        df_station = df[df['ID'] == station_id].sort_values('ds')
        split_index = int(len(df_station) * 0.8)
        train_list.append(df_station.iloc[:split_index])
        test_list.append(df_station.iloc[split_index:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    train_df.to_csv(save_train_data_path, index=False)
    print(f"Training data saved to: {save_train_data_path}")

    model = NeuralProphet(
        trend_global_local="global",
        season_global_local="global",
        changepoints_range=0.8,
        epochs=20,
        trend_reg=5,
        learning_rate=0.1,
        batch_size=1024,
        loss_func="Huber",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.add_lagged_regressor(names=['AirTemp', 'Wind', 'Tdp', 'Solar'])

    print("Training model...")
    model.fit(train_df, freq='h')

    save(model, model_save_path)
    print(f"Model saved to: {model_save_path}")

    forecast = model.predict(test_df)
    test_df['yhat1'] = forecast['yhat1']
    eval_df = test_df.dropna(subset=['y', 'yhat1'])

    mae = mean_absolute_error(eval_df['y'], eval_df['yhat1'])
    mse = mean_squared_error(eval_df['y'], eval_df['yhat1'])
    r2 = r2_score(eval_df['y'], eval_df['yhat1'])

    print("\nTest Set Evaluation:")
    print(f"MAE = {mae:.4f}, MSE = {mse:.4f}, R² = {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(eval_df['y'], eval_df['yhat1'], alpha=0.5)
    plt.plot([eval_df['y'].min(), eval_df['y'].max()],
             [eval_df['y'].min(), eval_df['y'].max()], 'r--')
    plt.xlabel("Actual TL")
    plt.ylabel("Predicted TL")
    plt.title("Test Set: Actual vs Predicted")
    plt.grid(True)
    plt.show()

    return model, eval_df

def evaluate_on_new_data(new_file_path, model_path):
    check_cuda()

    df_new = pd.read_csv(new_file_path)
    df_new['ds'] = pd.to_datetime(df_new[['Year', 'Month', 'Day', 'Hour']])
    df_new = df_new[['ds', 'AirTemp', 'Wind', 'Tdp', 'Solar', 'TL']]
    df_new = df_new.sort_values('ds').drop_duplicates(subset='ds')
    df_new = df_new.rename(columns={'TL': 'y'})
    df_new['ID'] = 'new_stationx'

    model = load(model_path)
    model.config_normalization.unknown_data_normalization = 'global'
    print("Model loaded with global normalization for new data")

    forecast = model.predict(df_new)
    df_new['yhat1'] = forecast['yhat1']
    eval_df = df_new.dropna(subset=['y', 'yhat1'])

    mae = mean_absolute_error(eval_df['y'], eval_df['yhat1'])
    mse = mean_squared_error(eval_df['y'], eval_df['yhat1'])
    r2 = r2_score(eval_df['y'], eval_df['yhat1'])

    print("\nNew Station Evaluation:")
    print(f"MAE = {mae:.4f}, MSE = {mse:.4f}, R² = {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(eval_df['y'], eval_df['yhat1'], alpha=0.5)
    plt.plot([eval_df['y'].min(), eval_df['y'].max()],
             [eval_df['y'].min(), eval_df['y'].max()], 'r--')
    plt.xlabel("Actual TL")
    plt.ylabel("Predicted TL")
    plt.title("New Station: Actual vs Predicted")
    plt.grid(True)
    plt.show()

    return eval_df

if __name__ == "__main__":
    data_folder = "/home/bel/Desktop/Shiv_SRIP/ATenLoc/Data/Output_hourly_h300_alb30"
    df_all = load_all_data(data_folder)
    model, _ = train_global_model(df_all,
                                  model_save_path="/home/bel/Desktop/Shiv_SRIP/ATenLoc/Prophet/model/300_30_model_TL.np",
                                  save_train_data_path="train_data.csv")

