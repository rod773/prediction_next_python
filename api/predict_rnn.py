import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. Data Collection ---
# Fetch real data for a given symbol
while True:
    symbol = input("Enter the symbol you want to predict (e.g., BTC-USD, AAPL, AUDCAD=X): ")
    print(f"Fetching data for {symbol}...")
    df = yf.Ticker(symbol).history(period="60d", interval="1h")

    if not df.empty:
        break

    print(f"\nError: No data found for symbol '{symbol}'.")
    print("Please check if the symbol is correct. For Forex, try appending '=X' (e.g., AUDCAD=X).")
    print("Try again...\n")

# Resample 1-hour data to 4-hour candles (H4)
df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
df.columns = df.columns.str.lower() # Convert columns to lowercase (open, high, low...)
df = df.dropna()

print("--- Initial Data ---")
print(df.tail())

# --- 2. Feature Engineering ---
# Create a rich set of features that the model can learn from.
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, append=True)
df.ta.atr(length=14, append=True)
df.ta.bbands(length=20, append=True)
df.ta.ema(length=50, append=True, col_names='EMA_50')

# Create custom lagged features
for i in range(1, 6):
    df[f'price_change_lag_{i}'] = df['close'].diff(i)

df.dropna(inplace=True)

# --- 3. Target Definition & Data Splitting ---
prediction_day_date = df.index.max().normalize()
historical_df = df[df.index < prediction_day_date].copy()
prediction_df = df[df.index.normalize() == prediction_day_date].copy()

if prediction_df.empty:
    print("\nError: No data available for the current day to make a prediction.")
    print("This can happen on market holidays or if the script is run before trading begins.")
    exit()

daily_close_map = historical_df.resample('D')['close'].last()
historical_df['target_eod_close'] = historical_df.index.normalize().map(daily_close_map)
historical_df.dropna(inplace=True)

# --- 4. Data Preparation for RNN Model ---
features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 'target_eod_close']
features = [col for col in historical_df.columns if col not in features_to_exclude]
X = historical_df[features]
y = historical_df[['target_eod_close']] # Keep as DataFrame for scaler

# Split data into training and testing sets before scaling
train_size = int(len(X) * 0.8)
X_train_df, X_test_df = X[:train_size], X[train_size:]
y_train_df, y_test_df = y[:train_size], y[train_size:]

# Scale features and target
# Fit scalers ONLY on the training data to prevent data leakage
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train_df)
X_test_scaled = x_scaler.transform(X_test_df)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train_df)
y_test_scaled = y_scaler.transform(y_test_df)

# Create sequences for the RNN
sequence_length = 10 # Use the last 10 time steps (4-hour candles) to predict the next

def create_sequences(X_data, y_data, seq_length):
    """
    Transforms 2D time series data into 3D sequences for RNNs.
    For a sequence from t to t+seq_length-1, the target is y at t+seq_length.
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:(i + seq_length)])
        y_seq.append(y_data[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

print(f"\n--- Data Shapes for RNN ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 5. Model Training ---
print("\n--- Training RNN Model ---")
model = Sequential([
    SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    SimpleRNN(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1) # Output layer, linear activation is default
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 6. Evaluation & Prediction ---
# Evaluate the model on the unseen historical test set
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_inv = y_scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inv, y_pred)
r2 = r2_score(y_test_inv, y_pred)

print("\n--- Model Evaluation on Historical Test Set ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"(MAE means the EOD predictions were off by an average of ${mae:.4f} on historical data)")

# Predict today's close using the last available sequence of data.
# We need the last 'sequence_length' data points from the full dataframe.
full_features_df = df.drop(columns=features_to_exclude, errors='ignore')

# Get the last `sequence_length` rows
latest_sequence_df = full_features_df.iloc[-sequence_length:]

# Scale these features using the already-fitted x_scaler
latest_sequence_scaled = x_scaler.transform(latest_sequence_df)

# Reshape for RNN input: (1, sequence_length, n_features)
input_for_prediction = np.expand_dims(latest_sequence_scaled, axis=0)

# Make the prediction
todays_eod_prediction_scaled = model.predict(input_for_prediction)

# Inverse transform the prediction to get the actual price
todays_eod_prediction = y_scaler.inverse_transform(todays_eod_prediction_scaled)

print("\n--- Prediction for Today's Close ---")
last_known_price = prediction_df['close'].iloc[-1]
print(f"Last known price for today: {last_known_price:.4f}")
print(f"Predicted Today's End-of-Day Close: {todays_eod_prediction[0][0]:.4f}")
