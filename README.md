# AI-based-Time-Series-Forecasting-of-Climate-induced-Concrete-Pavement-Performance

This repository presents a comprehensive modeling pipeline developed to predict temperature-induced stresses—namely, Thermal Load (TL) and Critical Eigenstress Ratio (CESR)—in concrete pavements. These stresses arise due to non-uniform and non-linear temperature distributions influenced by atmospheric conditions such as solar radiation, air temperature, humidity, and wind.

Traditionally, these stresses are computed using complex numerical models, which, while accurate, are often computationally expensive and time-consuming to run, especially when simulations are required over large geographic areas or extended periods. This project addresses this challenge by developing statistical and machine learning models that aim to approximate or outperform numerical simulations in terms of accuracy, efficiency, and scalability.

## Project Objective

The core objective is to develop robust and generalizable forecasting models that predict TL and CESR values across diverse spatial locations and varying environmental conditions using only surface-level meteorological and physical features. These include:

- Air Temperature (AirTemp)
- Wind Speed (Wind)
- Dew Point Temperature (Tdp)
- Solar Irradiance (Solar)
- Hour of the day and Day of the year (to capture temporal periodicity)
- Albedo (surface reflectivity)
- Depth (to capture slab thickness effects)

These features allow the models to learn the influence of environmental variability on internal stress development in pavements.

## Models Implemented

A range of models were implemented and evaluated, from classical time series models to advanced deep learning architectures. The models are listed below in order of increasing performance.

### 1. ARIMA

Autoregressive Integrated Moving Average, used as a baseline model for time series forecasting under stationarity assumptions.

### 2. SARIMA

Extends ARIMA by incorporating seasonal trends using (p,d,q)(P,D,Q,s) parameters. Suitable for series with repeating seasonal patterns.

### 3. SARIMAX

Further extends SARIMA by incorporating exogenous regressors such as solar irradiance and temperature. Improved performance in presence of external variability.

### 4. NeuralProphet

A neural network extension of Facebook’s Prophet model. It supports non-linear trend fitting, multiple seasonalities, and external regressors. Temporal and meteorological features were included to improve accuracy.

### 5. XGBoost

A gradient-boosted decision tree model capable of capturing non-linear feature interactions. Hyperparameter tuning was performed using RandomizedSearchCV. Although fast and accurate, it does not retain temporal memory.

### 6. Artificial Neural Network (ANN)

Feedforward neural networks were trained using selected meteorological and temporal features. Simpler networks often outperformed deeper architectures, particularly after including "hour of the day" and "day of year" as periodic inputs.

### 7. Recurrent Neural Network (RNN)

Recurrent networks were introduced for sequence modeling. With time-encoded inputs and short sequence lengths, RNNs learned temporal dependencies better than feedforward architectures.

### 8. Long Short-Term Memory (LSTM)

The best-performing model, achieving R² scores exceeding 0.98 for TL prediction. LSTM's gating mechanisms allow it to model both short- and long-term dependencies effectively. It was also the most robust model across spatial locations and varying depths.

## Key Highlights

- Detailed feature engineering was performed to capture the physical, spatial, and temporal dynamics influencing TL and CESR.
- Generalization was tested using geographically distinct sites (Delhi and Kanyakumari).
- Evaluation was performed using R² score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
- LSTM was particularly effective for TL, while CESR prediction required careful handling due to its lower range and sensitivity to scaling.
- The project demonstrates the potential for machine learning to serve as a faster alternative to traditional numerical simulations.


## Model Performance Summary

| Model         | TL R² Score | CESR R² Score | Key Insights                             |
|---------------|-------------|----------------|-------------------------------------------|
| LSTM          | > 0.98      | ~ 0.91         | Best performance across all conditions    |
| RNN           | ~ 0.95      | ~ 0.89         | Strong sequence learning on shorter terms |
| ANN           | ~ 0.96      | ~ 0.86         | Effective with key engineered features    |
| XGBoost       | ~ 0.94      | ~ 0.85         | Fast and accurate, lacks sequential memory|
| NeuralProphet | ~ 0.91      | ~ 0.82         | Flexible with seasonality and regressors  |
| SARIMAX       | ~ 0.88      | ~ 0.80         | Leverages exogenous inputs                |
| SARIMA        | ~ 0.85      | ~ 0.77         | Good for seasonal univariate series       |
| ARIMA         | ~ 0.79      | ~ 0.72         |As the Data is Seasonal it was not able to perform                     |


