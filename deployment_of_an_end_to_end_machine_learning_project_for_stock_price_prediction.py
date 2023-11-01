
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Fetch stock data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    fig.show()

def forecast_data(data, period):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    fig1 = plot_plotly(m, forecast)
    fig1.show()

    return forecast

if __name__ == '__main__':
    # Define stock options
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'ARKK', 'VXX', 'BABA')
    selected_stock = input(f"Select dataset for prediction {stocks}: ")
    n_years = int(input("Years of prediction (1-4): "))
    period = n_years * 365

    # Load and display data
    data = load_data(selected_stock)
    print(data.tail())
    plot_raw_data(data)

    # Forecast and display results
    forecast = forecast_data(data, period)
    print(forecast.tail())
