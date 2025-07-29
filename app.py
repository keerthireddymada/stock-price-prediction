import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="Stock Price Predictor", layout="wide")

@st.cache_resource
def load_lstm_model():
    return load_model("stock_lstm_model.keras")

model = load_lstm_model()


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


st.title("Stock Price Prediction with LSTM")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker (e.g. AAPL, TSLA):", value="AAPL")
with col2:
    predict_next_day = st.checkbox("Predict Next Day's Closing Price")

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.button("Predict"):
    with st.spinner("Fetching and processing data..."):

        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for this ticker and date range.")
            st.stop()

        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['RSI_14'] = compute_rsi(df['Close'])
        df.dropna(inplace=True)

        features = ['Close', 'SMA_10', 'RSI_14']
        data = df[features].values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        sequence_length = 60
        X = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
        X = np.array(X)

        predicted_scaled = model.predict(X)
        reconstructed = np.zeros((predicted_scaled.shape[0], 3))
        reconstructed[:, 0] = predicted_scaled[:, 0]
        reconstructed[:, 1:] = scaled_data[sequence_length:, 1:]

        predicted_prices = scaler.inverse_transform(reconstructed)[:, 0]
        actual_prices = df['Close'].values[sequence_length:]

        
        st.subheader("Actual vs Predicted Stock Prices")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(actual_prices, label="Actual", color='blue')
        ax1.plot(predicted_prices, label="Predicted", color='red')
        ax1.set_title(f"{ticker} Price Prediction")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)

        
        if predict_next_day:
            st.subheader("Next Day Prediction")

            last_60 = scaled_data[-60:]
            last_60 = np.expand_dims(last_60, axis=0)

            next_day_scaled = model.predict(last_60)
            next_day_full = np.zeros((1, 3))
            next_day_full[0, 0] = next_day_scaled[0, 0]
            next_day_full[0, 1:] = scaled_data[-1, 1:]

            next_day_price = scaler.inverse_transform(next_day_full)[0, 0]

            
            st.success(f"Predicted next closing price: **${next_day_price:.2f}**")

            
            last_10 = df['Close'].values[-10:]
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(range(10), last_10, label="Last 10 Actual", color='blue')
            ax2.scatter(10, next_day_price, color='red', label="Predicted Next Day", s=80)
            ax2.set_title("Last 10 Days vs Predicted Next Day")
            ax2.set_xlabel("Day Index")
            ax2.set_ylabel("Price")
            ax2.legend()
            st.pyplot(fig2)







