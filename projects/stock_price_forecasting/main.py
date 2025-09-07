import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense
from keras.models import Sequential
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Step 1: Download and Save Historical Stock Data ---
# We will use the yfinance library to download data for a specific stock.
# Let's use Apple Inc. (AAPL) as our example.

stock_ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
file_path = f'{stock_ticker.lower()}_data.csv'

try:
    # Try to load the data from a local CSV file
    stock_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print(f"Data loaded successfully from {file_path}")

except FileNotFoundError:
    print(f"File '{file_path}' not found. Downloading data...")
    # If the file does not exist, download it from Yahoo Finance
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    print("Data downloaded successfully.")
    # Save the downloaded data to a CSV for future use
    stock_data.to_csv(file_path)
    print(f"Data saved to {file_path}")

# # Explicitly set the frequency of the index to daily ('D')
# stock_data.index = pd.to_datetime(stock_data.index)
# stock_data.index.freq = 'D'
# --- Step 2: Explore and Visualize the Data ---
print("\nFirst 5 rows of the dataset:")
print(stock_data.head())

print("\nDataFrame info:")
stock_data.info()

# # We will plot the 'Close' price to see the trend over time
# plt.figure(figsize=(14, 7))
# stock_data['Close'].plot(title=f'{stock_ticker} Stock Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.grid(True)
# plt.show()

# --- Step 3: Check for Stationarity and Perform Differencing ---
# The Augmented Dickey-Fuller (ADF) test is a statistical test for stationarity.
print("\nPerforming Augmented Dickey-Fuller Test for Stationarity...")
result = adfuller(stock_data['Close'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# Interpret the p-value
if result[1] <= 0.05:
    print("Conclusion: The time series is likely stationary.")
else:
    print("Conclusion: The time series is likely NOT stationary. We will perform differencing.")
    
    # Perform first-order differencing
    stock_data['Close_Differenced'] = stock_data['Close'].diff().dropna()
    
    # # Plot the differenced series
    # plt.figure(figsize=(14, 7))
    # stock_data['Close_Differenced'].plot(title=f'{stock_ticker} Differenced Stock Price')
    # plt.xlabel('Date')
    # plt.ylabel('Change in Closing Price (USD)')
    # plt.grid(True)
    # plt.show()

    # Re-run the ADF test on the differenced data
    print("\nRe-running ADF Test on differenced data...")
    result_diff = adfuller(stock_data['Close_Differenced'].dropna())
    print('ADF Statistic: %f' % result_diff[0])
    print('p-value: %f' % result_diff[1])
    if result_diff[1] <= 0.05:
        print("Conclusion: The differenced time series is now likely stationary.")
    else:
        print("Conclusion: The differenced time series is still not stationary.")

# --- Step 4: Split Data and Build an ARIMA Model ---
print("\nSplitting data and fitting an ARIMA model...")
# We will use 80% of the data for training and 20% for testing
train_size = int(len(stock_data) * 0.8)
train_data = stock_data['Close'][:train_size]
test_data = stock_data['Close'][train_size:]


# # Fit the ARIMA model to the training data.
# # We will use p=1, d=1, and q=1 as a starting point.
# try:
#     model = ARIMA(train_data, order=(1, 1, 1))
#     fitted_model = model.fit()
#     print("ARIMA model fitted successfully.")
    
#     # --- Step 5: Make Predictions and Visualize Results ---
#     # Make predictions on the test set
#     predictions = fitted_model.forecast(steps=len(test_data))
#     predictions.index = test_data.index
    
#     # Plot the results
#     plt.figure(figsize=(14, 7))
#     plt.plot(train_data.index, train_data, label='Training Data')
#     plt.plot(test_data.index, test_data, label='Actual Price', color='red')
#     plt.plot(predictions.index, predictions, label='Predicted Price', color='green', linestyle='--')
#     plt.title(f'{stock_ticker} Stock Price Prediction with ARIMA(1,1,1)')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price (USD)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
# except Exception as e:
#     print(f"An error occurred while fitting the ARIMA model: {e}")
#     print("Please make sure you have the 'statsmodels' and 'sklearn' libraries installed.")
#     print("You can install them using: pip install statsmodels scikit-learn")


# # --- Step 6: Prepare Data for an LSTM Model ---
# print("\nPreparing data for LSTM model...")
# # LSTMs are sensitive to the scale of the data, so we need to normalize it.
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# # Split the data into training and testing sets
# train_size_lstm = int(len(scaled_data) * 0.8)
# train_data_lstm = scaled_data[:train_size_lstm, :]
# test_data_lstm = scaled_data[train_size_lstm:, :]

# # Create a function to convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=60):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i:(i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return np.array(dataX), np.array(dataY)

# # Use a look-back period of 60 days to predict the next day's price
# look_back = 60
# X_train, y_train = create_dataset(train_data_lstm, look_back)
# X_test, y_test = create_dataset(test_data_lstm, look_back)

# # Reshape the input to be [samples, time steps, features] which is a requirement for LSTMs
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # --- Step 7: Build and Train the LSTM Model ---
# print("\nBuilding and training the LSTM model...")
# # Check if TensorFlow and GPU are available
# print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")

# # Build the LSTM model
# lstm_model = Sequential()
# lstm_model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
# lstm_model.add(LSTM(50))
# lstm_model.add(Dense(1))

# # Compile the model
# lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
# print("LSTM model trained successfully.")

# # --- Step 8: Make Predictions and Visualize Results (LSTM) ---
# print("\nMaking predictions with the LSTM model...")
# # Make predictions on the test set
# lstm_predictions = lstm_model.predict(X_test)

# # Invert the predictions to the original scale
# lstm_predictions = scaler.inverse_transform(lstm_predictions)

# # Create a full prediction array with NaNs to plot the entire series without a gap
# train_predict_plot = np.empty_like(scaled_data)
# train_predict_plot[:] = np.nan
# train_predict_plot[look_back:len(y_train) + look_back] = scaler.inverse_transform(y_train.reshape(-1, 1))

# test_predict_plot = np.empty_like(scaled_data)
# test_predict_plot[:] = np.nan
# # Fix the ValueError by removing .flatten(), as lstm_predictions already has the correct shape for broadcasting
# test_predict_plot[len(train_data_lstm) + look_back + 1:len(scaled_data)] = lstm_predictions

# # Plot the results
# plt.figure(figsize=(14, 7))
# plt.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='red')
# plt.plot(stock_data.index, train_predict_plot, label='LSTM Training Prediction', color='green', linestyle='--')
# plt.plot(stock_data.index, test_predict_plot, label='LSTM Test Prediction', color='orange', linestyle='--')
# plt.title(f'{stock_ticker} Stock Price Prediction with LSTM')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.legend()
# plt.grid(True)
# plt.show()


# # --- Step 9: Model Evaluation ---
# print("\nEvaluating LSTM Model Performance...")
# # The y_test variable contains the scaled values, so we need to inverse transform it
# y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# # Calculate the Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# # Calculate the Mean Absolute Error (MAE)
# mae = mean_absolute_error(y_test_actual, lstm_predictions)
# print(f"Mean Absolute Error (MAE): {mae:.2f}")

# # --- Step 10: Future Predictions ---
# print("\nMaking future predictions for the next 30 days...")
# # Get the last 'look_back' days of data from the entire dataset
# last_60_days = scaled_data[-look_back:]
# future_predictions = []

# # Loop to predict the next 30 days
# num_prediction_days = 30
# for _ in range(num_prediction_days):
#     # Reshape the last 60 days to the required format for the model
#     input_data = last_60_days[-look_back:].reshape(1, look_back, 1)
    
#     # Predict the next day's price
#     predicted_price = lstm_model.predict(input_data, verbose=0)
    
#     # Append the prediction to the future_predictions list
#     future_predictions.append(predicted_price[0])
    
#     # Add the predicted price to the sequence for the next prediction
#     last_60_days = np.vstack([last_60_days, predicted_price])

# # Invert the future predictions back to the original scale
# future_predictions = scaler.inverse_transform(future_predictions)

# # Create a new index for the future dates
# last_date = stock_data.index[-1]
# future_dates = [last_date + timedelta(days=i) for i in range(1, num_prediction_days + 1)]

# # Plot the results including the future predictions
# plt.figure(figsize=(14, 7))
# plt.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='red')
# plt.plot(stock_data.index[train_size:], test_data, label='Actual Test Data', color='blue') # Added to clearly show test data
# plt.plot(stock_data.index, train_predict_plot, label='LSTM Training Prediction', color='green', linestyle='--')
# plt.plot(stock_data.index, test_predict_plot, label='LSTM Test Prediction', color='orange', linestyle='--')
# plt.plot(future_dates, future_predictions, label='LSTM Future Predictions', color='purple', linestyle='--')
# plt.title(f'{stock_ticker} Stock Price Forecasting with LSTM')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.legend()
# plt.grid(True)
# plt.show()


# --- Step 6: Prepare Data for an LSTM Model ---
print("\nPreparing data for LSTM model...")
# LSTMs are sensitive to the scale of the data, so we need to normalize it.
# We will now use multiple features: Open, High, Low, Volume, and Close
features = ['Open', 'High', 'Low', 'Volume', 'Close']
data_for_lstm = stock_data[features].values
num_features = len(features)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_for_lstm)

# Split the data into training and testing sets
train_size_lstm = int(len(scaled_data) * 0.8)
train_data_lstm = scaled_data[:train_size_lstm, :]
test_data_lstm = scaled_data[train_size_lstm:, :]

# Create a function to convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        # Use all features in the look_back window
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        # The prediction target is still just the 'Close' price (last column)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)

# Use a look-back period of 60 days to predict the next day's price
look_back = 60
X_train, y_train = create_dataset(train_data_lstm, look_back)
X_test, y_test = create_dataset(test_data_lstm, look_back)

# Reshape the input to be [samples, time steps, features] which is a requirement for LSTMs
# This is already handled by our new create_dataset function, but we keep it for clarity
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))

# --- Step 7: Build and Train the LSTM Model ---
print("\nBuilding and training the LSTM model...")
# Check if TensorFlow and GPU are available
print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")

# Build the LSTM model, now with input_shape considering multiple features
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(look_back, num_features)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
print("LSTM model trained successfully.")

# --- Step 8: Make Predictions and Visualize Results (LSTM) ---
print("\nMaking predictions with the LSTM model...")
# Make predictions on the test set
lstm_predictions = lstm_model.predict(X_test)

# To inverse transform, we need a full-width array.
# The scaler was trained on all features, so we can't just transform the single predicted value.
# We create a temporary array to hold our predicted values.
temp_predictions = np.zeros((len(lstm_predictions), num_features))
temp_predictions[:, -1] = lstm_predictions.flatten()

# Invert the predictions to the original scale
lstm_predictions = scaler.inverse_transform(temp_predictions)[:, -1]

# Create a temporary array to hold y_train to inverse transform
temp_y_train = np.zeros((len(y_train), num_features))
temp_y_train[:, -1] = y_train.flatten()

# Inverse transform the training predictions
training_predictions_unscaled = scaler.inverse_transform(temp_y_train)[:, -1]

# Create a full prediction array with NaNs to plot the entire series without a gap
train_predict_plot = np.empty_like(scaled_data[:, -1])
train_predict_plot[:] = np.nan
train_predict_plot[look_back:len(y_train) + look_back] = training_predictions_unscaled

test_predict_plot = np.empty_like(scaled_data[:, -1])
test_predict_plot[:] = np.nan
test_predict_plot[len(train_data_lstm) + look_back + 1:len(scaled_data)] = lstm_predictions

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='red')
plt.plot(stock_data.index, train_predict_plot, label='LSTM Training Prediction', color='green', linestyle='--')
plt.plot(stock_data.index, test_predict_plot, label='LSTM Test Prediction', color='orange', linestyle='--')
plt.title(f'{stock_ticker} Stock Price Prediction with LSTM (Multi-Feature)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# --- Step 9: Model Evaluation ---
print("\nEvaluating LSTM Model Performance...")
# Create a temporary array to hold y_test to inverse transform
temp_y_test = np.zeros((len(y_test), num_features))
temp_y_test[:, -1] = y_test.flatten()

# The y_test variable contains the scaled values, so we need to inverse transform it
y_test_actual = scaler.inverse_transform(temp_y_test)[:, -1]

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_actual, lstm_predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# --- Step 10: Future Predictions ---
print("\nMaking future predictions for the next 30 days...")
# Get the last 'look_back' days of data from the entire dataset
last_60_days = scaled_data[-look_back:]
future_predictions = []

# Loop to predict the next 30 days
num_prediction_days = 30
for _ in range(num_prediction_days):
    # Reshape the last 60 days to the required format for the model
    input_data = last_60_days[-look_back:].reshape(1, look_back, num_features)
    
    # Predict the next day's price
    predicted_price = lstm_model.predict(input_data, verbose=0)
    
    # We now have a single predicted value for the 'Close' price.
    # We create a new row with the last values of the other features and our new prediction.
    # A simplified approach is to assume the other features will have the same scaled values as the last day.
    new_row = np.copy(last_60_days[-1, :])
    new_row[-1] = predicted_price.flatten()
    
    # Append the predicted price to the future_predictions list
    future_predictions.append(predicted_price[0])
    
    # Add the new row to the sequence for the next prediction
    last_60_days = np.vstack([last_60_days, new_row])

# Invert the future predictions back to the original scale
# We need to perform the same temporary array trick to inverse transform
future_predictions_scaled = np.zeros((len(future_predictions), num_features))
future_predictions_scaled[:, -1] = np.array(future_predictions).flatten()
future_predictions = scaler.inverse_transform(future_predictions_scaled)[:, -1]

# Create a new index for the future dates
last_date = stock_data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, num_prediction_days + 1)]

# Plot the results including the future predictions
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='red')
plt.plot(stock_data.index[train_size:], test_data, label='Actual Test Data', color='blue') # Added to clearly show test data
plt.plot(stock_data.index, train_predict_plot, label='LSTM Training Prediction', color='green', linestyle='--')
plt.plot(stock_data.index, test_predict_plot, label='LSTM Test Prediction', color='orange', linestyle='--')
plt.plot(future_dates, future_predictions, label='LSTM Future Predictions', color='purple', linestyle='--')
plt.title(f'{stock_ticker} Stock Price Forecasting with LSTM (Multi-Feature)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
