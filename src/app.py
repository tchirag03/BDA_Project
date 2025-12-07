import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page Config
st.set_page_config(page_title="Equity Analysis System", layout="wide")

# Load Data
@st.cache_data
def load_data():
    try:
        fund_df = pd.read_parquet("database/fundamentals.parquet")
        tech_df = pd.read_parquet("database/technicals.parquet")
        ratings_df = pd.read_parquet("database/ratings.parquet")
        return fund_df, tech_df, ratings_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

fund_df, tech_df, ratings_df = load_data()

if tech_df.empty:
    st.warning("No data available. Please run the ETL pipeline.")
    st.stop()

# Sidebar
st.sidebar.title("Equity Analysis")
tickers = sorted(tech_df['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

# Filter Data
ticker_tech = tech_df[tech_df['Ticker'] == selected_ticker].sort_values('Date')
ticker_fund = fund_df[fund_df['Ticker'] == selected_ticker].sort_values('Date', ascending=False)
ticker_ratings = ratings_df[ratings_df['Ticker'] == selected_ticker]

if ticker_tech.empty:
    st.error(f"No data for {selected_ticker}")
    st.stop()

latest_tech = ticker_tech.iloc[-1]
latest_rating = ticker_ratings.iloc[-1] if not ticker_ratings.empty else None

# Header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.title(selected_ticker)
with col2:
    st.metric("Price", f"₹{latest_tech['Close']:.2f}", f"{latest_tech['DailyReturn']*100:.2f}%")
with col3:
    if latest_rating is not None:
        st.metric("Overall Score", f"{latest_rating['OverallScore']:.1f}/5")
with col4:
    ml_label = latest_tech['Label'] if latest_tech['Label'] else "N/A"
    ml_score = latest_tech['Technical_ML_Score']
    st.metric("ML Prediction", ml_label, f"Score: {ml_score:.1f}" if ml_score else None)

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Technicals", "Fundamentals"])

with tab1:
    st.subheader("Score Cards")
    if latest_rating is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"Balance Sheet: {latest_rating['BalanceSheetScore']:.1f}")
        c2.info(f"P&L: {latest_rating['PLScore']:.1f}")
        c3.info(f"Valuation: {latest_rating['ValuationScore']:.1f}")
        c4.info(f"Technical: {latest_rating['TechnicalScore']:.1f}")
    else:
        st.write("No ratings available.")

with tab2:
    st.subheader("Price & Volume")
    
    # Candlestick with SMAs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                        row_width=[0.2, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=ticker_tech['Date'],
                                 open=ticker_tech['Open'],
                                 high=ticker_tech['High'],
                                 low=ticker_tech['Low'],
                                 close=ticker_tech['Close'],
                                 name='OHLC'), row=1, col=1)
    
    # SMAs
    fig.add_trace(go.Scatter(x=ticker_tech['Date'], y=ticker_tech['SMA50'], line=dict(color='orange', width=1), name='SMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_tech['Date'], y=ticker_tech['SMA200'], line=dict(color='blue', width=1), name='SMA200'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=ticker_tech['Date'], y=ticker_tech['Volume'], name='Volume'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Fundamental Data")
    if not ticker_fund.empty:
        st.dataframe(ticker_fund.drop(columns=['Ticker']).style.format(precision=2))
    else:
        st.write("No fundamental data available.")
