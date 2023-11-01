import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Streamlit setup
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .reportview-container {
            background: #F0EDEE;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('📈 Stock Forecast App 🚀')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'ARKK', 'VXX', 'BABA')
selected_stock = st.selectbox('🌐 Select dataset for prediction', stocks)

n_years = st.slider('📅 Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

with st.spinner('🔄 Loading data...'):
    data = load_data(selected_stock)

st.subheader('Raw data 📊')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", line=dict(color='red')))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data 📈')
st.write(forecast.tail())
    
st.write(f'🔮 Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Forecast components 📉")
fig2 = m.plot_components(forecast)
st.write(fig2)