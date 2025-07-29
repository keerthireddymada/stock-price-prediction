import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

ticker = "AAPL"
start_date = "2018-01-01"
end_date = "2025-07-25"
df = yf.download(ticker, start=start_date, end=end_date)

df['SMA_10'] = df['Close'].rolling(window=10).mean()

delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

features = ['Close', 'SMA_10', 'RSI_14']
data = df[features].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

X_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    X_train.append(train_data[i-sequence_length:i])
    y_train.append(train_data[i][0])  

X_train = np.array(X_train)
y_train = np.array(y_train)


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

model.save("stock_lstm_model.keras")


import os
print("Model saved at:", os.path.abspath("stock_lstm_model.keras"))






'''
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# No imports from train_model.py itself!

def build_and_train_model(X, y, model_path='model.h5'):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    import sys; sys.stdout.flush()
    model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=1
    )

    model.save(model_path)
    return model

# ----- Load + preprocess data -----
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-12-31")
data = df[['Close']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
seq_len = 60

for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i - seq_len:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(f"Training data shape: X={X.shape}, y={y.shape}")

# ----- Train model -----
model = build_and_train_model(X, y, model_path="model.h5")
print("✅ Model trained and saved to model.h5")


'''