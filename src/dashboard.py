import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Equity Analyzer Pro", layout="wide")

@st.cache_data
def load_data():
    path = "dataset/spark_processed/processed_data.parquet"
    if not os.path.exists(path):
        st.error(f"{path} not found.")
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data
def load_predictions():
    path = "dataset/spark_processed/predictions.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

st.title("🔥 Equity Analyzer Pro")
st.markdown("### Spark-Based Equity Analysis & Prediction Platform")

df = load_data()
df_pred = load_predictions()

if not df.empty:
    # Sidebar
    st.sidebar.header("Filters")
    tickers = st.sidebar.multiselect("Select Tickers", options=df['Ticker'].unique(), default=df['Ticker'].unique()[:5])
    
    if tickers:
        df_filtered = df[df['Ticker'].isin(tickers)]
    else:
        df_filtered = df

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Technical Analysis", "ML Predictions"])

    with tab1:
        st.subheader("Market Overview")
        st.dataframe(df_filtered.head(100))
        
        col1, col2 = st.columns(2)
        with col1:
             st.metric("Total Rows", len(df_filtered))
        with col2:
             avg_risk = df_filtered['Risk_Score'].mean()
             st.metric("Avg Market Risk", f"{avg_risk:.2f}")

    with tab2:
        st.subheader("Trend Analysis")
        # Line chart
        ticker = st.selectbox("Select Stock for Chart", options=tickers if tickers else df['Ticker'].unique())
        chart_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        fig = px.line(chart_data, x="Date", y=["Close", "50_SMA", "200_SMA"], title=f"{ticker} Price & MA")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Random Forest Predictions (Test Set)")
        st.write("Prediction: 1 (Up), 0 (Down)")
        
        if not df_pred.empty:
             pred_filtered = df_pred[df_pred['Ticker'].isin(tickers)] if tickers else df_pred
             fig = px.scatter(pred_filtered, x="Risk_Score", y="Probability", 
                              color="Prediction", hover_data=["Ticker"],
                              title="Risk vs Prediction Probability")
             st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("No predictions found.")

# Note: Date might be missing in predictions.parquet.
# For the chart, I should load the FULL dataset or ensure Date is passed.
# I'll assume for now I can just plot what I have or use Index.
