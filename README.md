# 📈 Stock Price Predictor with LSTM

This is a web-based application that predicts stock prices using a pre-trained LSTM (Long Short-Term Memory) model. It allows users to view actual vs predicted prices, and also forecast the next day's closing price based on historical data.

---

## 🚀 Features
- Input any stock ticker (e.g., AAPL, MSFT, TSLA)
- Select a date range for historical analysis
- Visualize actual vs predicted closing prices
- Predict the next day's closing price
- Simple and interactive Streamlit UI

---

## 🔧 Tools Used
- **Python**, **Streamlit**, **Pandas**, **NumPy**
- **TensorFlow/Keras** (LSTM model)
- **yfinance** (for live stock data)
- **Matplotlib** (for plotting)
- **scikit-learn** (for scaling)

---

## 🔄 Workflow

1. **User Input:**
   - Ticker Symbol
   - Start & End Date
   - Option to predict next day's price

2. **Data Collection:**
   - Live stock data is fetched using `yfinance`

3. **Preprocessing:**
   - Data is scaled and features like SMA & RSI are computed

4. **Prediction:**
   - The LSTM model (`stock_lstm_model.keras`) is used to predict stock prices
   - Visual plots and predicted values are displayed

5. **Next-Day Forecast:**
   - The model uses recent history to forecast the next closing price

---

## 🌐 Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
