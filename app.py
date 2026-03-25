import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import time
from datetime import datetime
from transformers import pipeline
import plotly.graph_objects as go
from yfinance.exceptions import YFRateLimitError

st.set_page_config(
    page_title="FinSentiment Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
.metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #2d3250;
}
.metric-label { font-size: 12px; color: #8b9ab0; }
.metric-value { font-size: 26px; font-weight: bold; }
.up { color: #2ecc71; }
.down { color: #e74c3c; }
.neutral { color: #f39c12; }
</style>
""", unsafe_allow_html=True)

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_finbert():
    return pipeline("text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/xgb_sentiment_model.pkl")
        features = joblib.load("model/feature_columns.pkl")
        return model, features
    except:
        return None, None

# -------------------- FETCH PRICE DATA (FIXED) --------------------
@st.cache_data(ttl=300)
def fetch_price_data(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="1mo", progress=False)

            if df.empty:
                return None, None

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            closes = df["Close"]

            features = {
                "daily_return": (latest["Close"] - prev["Close"]) / prev["Close"],
                "price_vs_ma5": latest["Close"] / closes.tail(5).mean() - 1,
                "price_vs_ma10": latest["Close"] / closes.tail(10).mean() - 1,
                "volatility_5d": closes.pct_change().tail(5).std(),
            }

            delta = closes.diff()
            gain = delta.clip(lower=0).tail(14).mean()
            loss = (-delta.clip(upper=0)).tail(14).mean()
            rs = gain / (loss + 1e-9)
            features["rsi"] = 100 - (100 / (1 + rs))

            price_info = {
                "current_price": round(latest["Close"], 2),
                "change_pct": round((latest["Close"] - prev["Close"]) / prev["Close"] * 100, 2),
                "history": df["Close"].tail(30)
            }

            return features, price_info

        except YFRateLimitError:
            time.sleep(2)

    return None, None

# -------------------- FETCH NEWS --------------------
def fetch_news(ticker):
    if not NEWS_API_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        return [a["title"] for a in data.get("articles", []) if a.get("title")]
    except:
        return []

# -------------------- SENTIMENT --------------------
def analyze_sentiment(headlines, finbert):
    sentiments = []
    for h in headlines:
        try:
            r = finbert(h[:512])[0]
            sentiments.append(r["label"])
        except:
            continue
    return sentiments

# -------------------- UI --------------------
st.title("📈 FinSentiment Stock Predictor")

STOCKS = ["AAPL", "TSLA", "MSFT", "GOOGL"]
ticker = st.selectbox("Select Stock", STOCKS)

if st.button("Analyze"):

    # PRICE
    price_features, price_info = fetch_price_data(ticker)

    if price_info is None:
        st.warning("⚠️ Rate limit reached. Please wait and try again.")
        st.stop()

    # NEWS
    headlines = fetch_news(ticker)

    # SENTIMENT
    finbert = load_finbert()
    sentiments = analyze_sentiment(headlines, finbert)

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Price", f"${price_info['current_price']}")
    col2.metric("Change %", f"{price_info['change_pct']}%")
    col3.metric("News Count", len(headlines))

    # ---------------- CHART ----------------
    fig = go.Figure()
    hist = price_info["history"]

    fig.add_trace(go.Scatter(x=hist.index, y=hist.values))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- SENTIMENT ----------------
    if sentiments:
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        neu = sentiments.count("neutral")

        st.write("### Sentiment")
        st.write(f"🟢 Positive: {pos}")
        st.write(f"🔴 Negative: {neg}")
        st.write(f"🟡 Neutral: {neu}")
    else:
        st.info("No news found.")

    # ---------------- MODEL ----------------
model, feature_cols = load_model()

if model:
    df = pd.DataFrame([price_features])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    # ✅ CRITICAL FIX
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    df = df.astype(np.float32)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    if pred == 1:
        st.success(f"📈 UP ({round(max(prob)*100,1)}%)")
    else:
        st.error(f"📉 DOWN ({round(max(prob)*100,1)}%)")

else:
    st.warning("Model not found.")
