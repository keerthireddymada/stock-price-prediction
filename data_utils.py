import yfinance as yf
import pandas as pd
import ta

def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)

    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=14)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    df.dropna(inplace=True)
    return df
