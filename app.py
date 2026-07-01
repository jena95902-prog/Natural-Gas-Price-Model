import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Natural Gas Price Estimator", layout="wide")
st.title("JPMC Task 1: Natural Gas Price Analysis")

@st.cache_data
def load_and_train(csv_path="data/gas_prices.csv"):
    df = pd.read_csv(csv_path)
    # Forage CSV usually has columns: Date, Price
    df.columns = [c.strip().capitalize() for c in df.columns] 
    if 'Date' not in df.columns or 'Price' not in df.columns:
        st.error(f"CSV must have 'Date' and 'Price' columns. Found: {list(df.columns)}")
        st.stop()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date').asfreq('M') # Monthly
    
    # Drop any NaNs before fitting
    df = df.dropna()

    # Holt-Winters: additive trend + additive 12-month seasonality
    model = ExponentialSmoothing(
        df['Price'], trend='add', seasonal='add', seasonal_periods=12
    ).fit(optimized=True)
    return df, model

def get_price_for_date(target_date, model, df):
    last_date = df.index.max()
    months_ahead = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    
    if months_ahead <= 0:
        # In-sample: return fitted value
        return float(model.fittedvalues.asof(target_date))
    
    # Out-of-sample: forecast
    forecast = model.forecast(steps=months_ahead)
    return float(forecast.iloc[-1])

# --- App UI ---
try:
    df, model = load_and_train()
except FileNotFoundError:
    st.error("Could not find `data/gas_prices.csv`. Add it to a `data/` folder in your repo.")
    st.stop()

tab1, tab2 = st.tabs(["📈 Plot", "🔮 Estimate"])

with tab1:
    st.subheader("Historical Data + 12 Month Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Price'], label="Historical")
    
    future_idx = pd.date_range(start=df.index.max() + relativedelta(months=1), periods=12, freq='M')
    forecast = model.forecast(steps=12)
    ax.plot(future_idx, forecast, label="Forecast", linestyle='--')
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Natural Gas Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.caption("Model: Holt-Winters with 12-month seasonality. Peaks usually align with winter demand.")

with tab2:
    st.subheader("Get Estimated Price")
    min_d = df.index.min().date()
    max_d = (df.index.max() + relativedelta(months=12)).date()
    
    user_date = st.date_input("Select any date", value=max_d, min_value=min_d, max_value=max_d)
    
    if st.button("Estimate Price"):
        est_price = get_price_for_date(pd.to_datetime(user_date), model, df)
        st.success(f"Estimated Price on {user_date}: ${est_price:.2f}")
