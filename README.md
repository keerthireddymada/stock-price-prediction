# 📈 Stock Price Prediction using LSTM

This project uses historical stock data to predict future stock prices using a Long Short-Term Memory (LSTM) neural network. The implementation is done using TensorFlow and Keras, and the focus is on closing price prediction based on past trends.

---

## ✅ Project Goals

- Predict next-day stock prices using the past 60 days' data.
- Train an LSTM model on historical price data.
- Visualize predicted vs actual prices.
- Extend model to forecast future prices beyond available data.

---

## 🔧 Tools & Libraries

- **Data Collection**: yfinance  
- **Processing**: pandas, numpy, sklearn  
- **Modeling**: TensorFlow, Keras  
- **Visualization**: matplotlib  

---

## 📌 Work Done So Far

1. **Collected historical stock data** (e.g., Apple - AAPL) from 2015–2025 using `yfinance`.
2. **Selected and normalized** the closing price data for training.
3. **Created time series sequences**: for each 60-day window, the model predicts the next (61st) day's price.
4. **Built and trained** a stacked LSTM model with dropout layers to reduce overfitting.
5. **Made predictions and plotted results** showing predicted vs actual prices using test data.

---

## 🔮 Next Steps

- Predicting future prices by feeding recent 60-day sequences to the model recursively.
- Saving and exporting the model for deployment or future reuse.
- Building a simple GUI using Streamlit or Tkinter to visualize live predictions.

---

## 🌱 Possible Extensions

- Including other features like volume, open/high/low prices.
- Adding sentiment analysis or external indicators to improve accuracy.
- Experimenting with GRU, attention mechanisms, or hybrid models.

---
