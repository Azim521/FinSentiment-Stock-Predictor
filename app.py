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

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception  # fallback for older yfinance

st.set_page_config(
    page_title="FinSentiment Stock Predictor",
    page_icon="📈",
    layout="wide"
)

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

@st.cache_resource
def load_finbert():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1  # force CPU
    )

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/xgb_sentiment_model.pkl")
        features = joblib.load("model/feature_columns.pkl")
        return model, features
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None

@st.cache_data(ttl=300)
def fetch_price_data(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="1mo", progress=False)
            if df.empty:
                return None, None

            # ✅ FIX: Flatten MultiIndex columns (yfinance 0.2.x+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df.columns:
                return None, None

            closes = df["Close"].squeeze()  # ensure Series
            latest_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2]) if len(closes) > 1 else latest_close

            if pd.isna(latest_close) or pd.isna(prev_close):
                return None, None

            pct_changes = closes.pct_change().dropna()

            features = {
                "daily_return": (latest_close - prev_close) / (prev_close + 1e-9),
                "return_2d": (latest_close - float(closes.iloc[-3])) / float(closes.iloc[-3]) if len(closes) >= 3 else 0,
                "return_5d": (latest_close - float(closes.iloc[-6])) / float(closes.iloc[-6]) if len(closes) >= 6 else 0,
                "price_vs_ma5": latest_close / (closes.tail(5).mean() + 1e-9) - 1,
                "price_vs_ma10": latest_close / (closes.tail(10).mean() + 1e-9) - 1,
                "price_vs_ma20": latest_close / (closes.tail(20).mean() + 1e-9) - 1,
                "volatility_5d": pct_changes.tail(5).std() if len(pct_changes) >= 5 else 0,
                "volatility_10d": pct_changes.tail(10).std() if len(pct_changes) >= 10 else 0,
            }

            # Volume features
            if "Volume" in df.columns:
                vol = df["Volume"].squeeze()
                features["volume_change"] = float(vol.iloc[-1]) / (float(vol.iloc[-2]) + 1e-9) - 1 if len(vol) > 1 else 0
                features["vol_vs_ma5"] = float(vol.iloc[-1]) / (vol.tail(5).mean() + 1e-9) - 1
            else:
                features["volume_change"] = 0
                features["vol_vs_ma5"] = 0

            # RSI
            delta = closes.diff()
            gain = delta.clip(lower=0).tail(14).mean()
            loss = (-delta.clip(upper=0)).tail(14).mean()
            rs = gain / (loss + 1e-9)
            features["rsi"] = float(100 - (100 / (1 + rs)))

            # High-Low range
            if "High" in df.columns and "Low" in df.columns:
                features["high_low_range"] = float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])
            else:
                features["high_low_range"] = 0

            price_info = {
                "current_price": round(latest_close, 2),
                "change_pct": round(((latest_close - prev_close) / (prev_close + 1e-9)) * 100, 2),
                "history": closes.tail(30)
            }
            return features, price_info

        except YFRateLimitError:
            time.sleep(2)
        except Exception as e:
            st.warning(f"Price fetch error: {e}")
            return None, None

    return None, None

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

def analyze_sentiment(headlines, finbert):
    results = []
    for h in headlines:
        try:
            r = finbert(h[:512])[0]
            results.append(r)
        except:
            continue
    return results

# -------------------- UI --------------------
st.title("📈 FinSentiment Stock Predictor")

STOCKS = ["AAPL", "TSLA", "MSFT", "GOOGL"]
ticker = st.selectbox("Select Stock", STOCKS)

if st.button("Analyze"):
    with st.spinner("Fetching price data..."):
        price_features, price_info = fetch_price_data(ticker)

    if price_info is None or price_features is None:
        st.warning("⚠️ Failed to fetch price data. yfinance may be rate-limited. Try again in a few seconds.")
        st.stop()

    with st.spinner("Fetching news..."):
        headlines = fetch_news(ticker)

    with st.spinner("Running FinBERT sentiment analysis..."):
        finbert = load_finbert()
        sentiment_results = analyze_sentiment(headlines, finbert)

    # ---- METRICS ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${price_info['current_price']:.2f}")
    col2.metric("Change %", f"{price_info['change_pct']:.2f}%")
    col3.metric("News Count", len(headlines))

    # ---- CHART ----
    hist = price_info["history"].squeeze()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values.flatten(), mode="lines"))
    fig.update_layout(template="plotly_dark", title=f"{ticker} — Last 30 Days")
    st.plotly_chart(fig, use_container_width=True)

    # ---- SENTIMENT ----
    if sentiment_results:
        labels = [r["label"] for r in sentiment_results]
        scores = [r["score"] for r in sentiment_results]
        pos = labels.count("positive")
        neg = labels.count("negative")
        neu = labels.count("neutral")
        total = len(labels)

        st.write("### 📰 News Sentiment")
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Positive", pos)
        col2.metric("🔴 Negative", neg)
        col3.metric("🟡 Neutral", neu)

        # Show individual headlines
        for i, (h, r) in enumerate(zip(headlines, sentiment_results)):
            color = "🟢" if r["label"] == "positive" else ("🔴" if r["label"] == "negative" else "🟡")
            st.write(f"{color} **{r['label'].upper()}** ({r['score']*100:.0f}%) — {h}")

        # Build sentiment features for model
        pos_scores = [scores[i] for i, l in enumerate(labels) if l == "positive"]
        neg_scores = [scores[i] for i, l in enumerate(labels) if l == "negative"]
        neu_scores = [scores[i] for i, l in enumerate(labels) if l == "neutral"]

        price_features["avg_positive"] = np.mean(pos_scores) if pos_scores else 0
        price_features["avg_negative"] = np.mean(neg_scores) if neg_scores else 0
        price_features["avg_neutral"] = np.mean(neu_scores) if neu_scores else 0
        price_features["pos_ratio"] = pos / total if total > 0 else 0
        price_features["sentiment_score"] = (pos - neg) / total if total > 0 else 0
        price_features["article_count"] = total
    else:
        st.info("ℹ️ No news found. Check that NEWS_API_KEY is set in Streamlit secrets.")
        price_features["avg_positive"] = 0
        price_features["avg_negative"] = 0
        price_features["avg_neutral"] = 0
        price_features["pos_ratio"] = 0
        price_features["sentiment_score"] = 0
        price_features["article_count"] = 0

    # ---- MODEL PREDICTION ----
    model, feature_cols = load_model()
    if model is None:
        st.warning("⚠️ Model not found.")
        st.stop()

    df_pred = pd.DataFrame([price_features])
    for col in feature_cols:
        if col not in df_pred.columns:
            df_pred[col] = 0
    df_pred = df_pred[feature_cols]
    df_pred = df_pred.apply(pd.to_numeric, errors='coerce')
    df_pred = df_pred.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)

    pred = model.predict(df_pred)[0]
    prob = model.predict_proba(df_pred)[0]
    confidence = round(max(prob) * 100, 1)

    st.write("### 🤖 Model Prediction")
    if pred == 1:
        st.success(f"📈 Predicted: **UP** — {confidence}% confidence")
    else:
        st.error(f"📉 Predicted: **DOWN** — {confidence}% confidence")
    st.caption("Note: ROC-AUC 0.529 — slight edge over random. Not financial advice.")
