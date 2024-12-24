import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Generate demo CSV file
def generate_demo_csv():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    demand = np.random.randint(50, 200, size=len(dates))
    temperature = np.random.uniform(10, 35, size=len(dates))
    precipitation = np.random.uniform(0, 50, size=len(dates))
    weather_condition = np.random.choice(["Sunny", "Cloudy", "Rainy"], size=len(dates))
    holiday = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
    festival_name = np.where(holiday == 1, "Festival", "")
    product_launch = np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05])
    
    data = {
        "Date": dates,
        "Demand": demand,
        "Temperature": temperature,
        "Precipitation": precipitation,
        "Weather Condition": weather_condition,
        "Holiday": holiday,
        "Festival Name": festival_name,
        "Product Launch": product_launch,
    }
    df = pd.DataFrame(data)
    df.to_csv("demand_data.csv", index=False)
    print("Demo CSV file generated as 'demand_data.csv'.")

# Preprocess data
def preprocess_data(data, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler):
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    max_error = np.max(np.abs(y_test - predictions))
    print(f"MAE: {mae}, MAPE: {mape}, Max Error: {max_error}")
    
    return predictions, mae, mape, max_error

# Compare with naive baseline
def naive_forecasting_baseline(y_test, predictions):
    baseline = y_test[:-1]
    naive_mae = mean_absolute_error(y_test[1:], baseline)
    naive_mape = mean_absolute_percentage_error(y_test[1:], baseline)
    print(f"Naive Baseline MAE: {naive_mae}, MAPE: {naive_mape}")

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Demand")
    plt.plot(predictions, label="LSTM Predictions")
    plt.title("Demand Forecasting")
    plt.legend()
    plt.show()

# Main pipeline
def main_pipeline():
    generate_demo_csv()
    file_path = "demand_data.csv"
    lookback = 30

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    data = df['Demand'].values.reshape(-1, 1)

    X, y, scaler = preprocess_data(data, lookback)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    predictions, mae, mape, max_error = train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler)

    naive_forecasting_baseline(scaler.inverse_transform(y_test), predictions)
    plot_predictions(scaler.inverse_transform(y_test), predictions)

if __name__ == "__main__":
    main_pipeline()
