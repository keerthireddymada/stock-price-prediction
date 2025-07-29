import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, sequence_length=60):
    df = df[['Close', 'SMA', 'RSI']]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])  # sequence of past days
        y.append(scaled_data[i, 0])  # predict the 'Close' price

    X, y = np.array(X), np.array(y)
    return X, y, scaler
