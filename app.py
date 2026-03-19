import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import os
from datetime import datetime, timedelta
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FinSentiment Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-label {
        font-size: 12px;
        color: #8b9ab0;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .up    { color: #2ecc71; }
    .down  { color: #e74c3c; }
    .neutral { color: #f39c12; }
    .section-header {
        font-size: 12px;
        font-weight: 600;
        color: #8b9ab0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 20px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2d3250;
    }
    .news-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid #2d3250;
    }
    .news-title { font-size: 13px; color: #c9d1d9; line-height: 1.4; }
    .news-meta  { font-size: 11px; color: #8b9ab0; margin-top: 4px; }
    .pos { border-left-color: #2ecc71 !important; }
    .neg { border-left-color: #e74c3c !important; }
    .neu { border-left-color: #f39c12 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load API keys from Streamlit secrets
# ─────────────────────────────────────────
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

# ─────────────────────────────────────────
# Load FinBERT
# ─────────────────────────────────────────
@st.cache_resource
def load_finbert():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        return_all_scores=True
    )

# ─────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/xgb_sentiment_model.pkl")
        features = joblib.load("model/feature_columns.pkl")
        return model, features
    except:
        return None, None

# ─────────────────────────────────────────
# Fetch news headlines
# ─────────────────────────────────────────
def fetch_news(ticker, company_name):
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} OR {company_name} stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "from": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "apiKey": NEWS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a["title"],
                "source": a["source"]["name"],
                "publishedAt": a["publishedAt"][:10]
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a["title"]
        ]
    except:
        return []

# ─────────────────────────────────────────
# Run FinBERT on headlines
# ─────────────────────────────────────────
def analyze_sentiment(headlines, finbert):
    results = []
    for h in headlines:
        try:
            scores = finbert(h["title"][:512])[0]
            score_dict = {s["label"]: s["score"] for s in scores}
            label = max(score_dict, key=score_dict.get)
            results.append({
                "title": h["title"],
                "source": h["source"],
                "date": h["publishedAt"],
                "sentiment": label,
                "positive": score_dict.get("positive", 0),
                "negative": score_dict.get("negative", 0),
                "neutral": score_dict.get("neutral", 0),
                "confidence": score_dict.get(label, 0)
            })
        except:
            continue
    return pd.DataFrame(results)

# ─────────────────────────────────────────
# Fetch price data + technical features
# ─────────────────────────────────────────
def fetch_price_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo")
    if df.empty:
        return None, None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # Technical features
    closes = df["Close"]
    features = {
        "daily_return": (latest["Close"] - prev["Close"]) / prev["Close"],
        "volume_change": (latest["Volume"] - prev["Volume"]) / (prev["Volume"] + 1),
        "ma5":  closes.tail(5).mean(),
        "ma10": closes.tail(10).mean(),
        "ma20": closes.tail(20).mean(),
        "price_vs_ma5":  latest["Close"] / closes.tail(5).mean() - 1,
        "price_vs_ma20": latest["Close"] / closes.tail(20).mean() - 1,
        "volatility":    closes.pct_change().tail(10).std(),
        "high_low_range": (latest["High"] - latest["Low"]) / latest["Close"],
    }

    # RSI
    delta = closes.diff()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    rs = gain / (loss + 1e-9)
    features["rsi"] = 100 - (100 / (1 + rs))

    price_info = {
        "current_price": round(latest["Close"], 2),
        "prev_close": round(prev["Close"], 2),
        "change_pct": round((latest["Close"] - prev["Close"]) / prev["Close"] * 100, 2),
        "high": round(latest["High"], 2),
        "low": round(latest["Low"], 2),
        "volume": int(latest["Volume"]),
        "history": df["Close"].tail(30)
    }
    return features, price_info

# ─────────────────────────────────────────
# Build feature vector for prediction
# ─────────────────────────────────────────
def build_features(price_features, sentiment_df):
    if sentiment_df is None or sentiment_df.empty:
        avg_pos = avg_neg = avg_neu = pos_ratio = 0.0
        n_articles = 0
    else:
        avg_pos = sentiment_df["positive"].mean()
        avg_neg = sentiment_df["negative"].mean()
        avg_neu = sentiment_df["neutral"].mean()
        pos_ratio = (sentiment_df["sentiment"] == "positive").mean()
        n_articles = len(sentiment_df)

    features = {**price_features,
                "avg_positive": avg_pos,
                "avg_negative": avg_neg,
                "avg_neutral": avg_neu,
                "positive_ratio": pos_ratio,
                "n_articles": n_articles,
                "sentiment_score": avg_pos - avg_neg}
    return pd.DataFrame([features])

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
STOCKS = {
    "AAPL — Apple": ("AAPL", "Apple"),
    "TSLA — Tesla": ("TSLA", "Tesla"),
    "MSFT — Microsoft": ("MSFT", "Microsoft"),
    "GOOGL — Alphabet": ("GOOGL", "Google Alphabet"),
}

with st.sidebar:
    st.markdown("## 📈 FinSentiment Predictor")
    st.markdown("Real-time news sentiment + price movement prediction using FinBERT.")
    st.markdown("---")
    selected = st.selectbox("Select Stock", list(STOCKS.keys()))
    ticker, company = STOCKS[selected]
    analyze_btn = st.button("🔍 Analyze Now")
    st.markdown("---")
    st.markdown("**Model:** XGBoost + FinBERT  \n**Data:** NewsAPI + yfinance  \n**Sentiment:** ProsusAI/finbert")

# ─────────────────────────────────────────
# Main area
# ─────────────────────────────────────────
st.markdown("# 📈 FinSentiment Stock Predictor")
st.markdown("Real-time financial news sentiment analysis + next-day price movement prediction.")
st.markdown("---")

if not analyze_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Sentiment Model</div>
            <div class="metric-value" style="font-size:16px;color:#667eea;">ProsusAI/FinBERT</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Stocks Covered</div>
            <div class="metric-value" style="color:#667eea;">4</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Data Sources</div>
            <div class="metric-value" style="font-size:16px;color:#667eea;">NewsAPI + yfinance</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Select a stock from the sidebar and click **Analyze Now** to get a sentiment-based prediction.")

else:
    with st.spinner(f"Fetching data for {ticker}..."):
        price_features, price_info = fetch_price_data(ticker)
        news = fetch_news(ticker, company)

    if price_info is None:
        st.error("Could not fetch price data. Please try again.")
        st.stop()

    with st.spinner("Running FinBERT sentiment analysis..."):
        finbert = load_finbert()
        sentiment_df = analyze_sentiment(news, finbert) if news else pd.DataFrame()

    # ── Price metrics ──
    change_color = "up" if price_info["change_pct"] >= 0 else "down"
    change_arrow = "▲" if price_info["change_pct"] >= 0 else "▼"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value" style="color:#c9d1d9;">${price_info['current_price']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Day Change</div>
            <div class="metric-value {change_color}">{change_arrow} {abs(price_info['change_pct'])}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        rsi_val = round(price_features.get("rsi", 50), 1)
        rsi_color = "down" if rsi_val > 70 else "up" if rsi_val < 30 else "neutral"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value {rsi_color}">{rsi_val}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        n_articles = len(sentiment_df) if not sentiment_df.empty else 0
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">News Articles</div>
            <div class="metric-value" style="color:#667eea;">{n_articles}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price chart + Sentiment breakdown ──
    col_chart, col_sent = st.columns([3, 2])

    with col_chart:
        st.markdown("#### 📉 30-Day Price History")
        fig = go.Figure()
        hist = price_info["history"]
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            mode='lines', line=dict(color='#667eea', width=2),
            fill='tozeroy', fillcolor='rgba(102,126,234,0.1)',
            name=ticker
        ))
        fig.update_layout(
            paper_bgcolor='#1e2130', plot_bgcolor='#1e2130',
            font=dict(color='#8b9ab0', size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='#2d3250', showgrid=True),
            yaxis=dict(gridcolor='#2d3250', showgrid=True, tickprefix='$'),
            height=220, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_sent:
        st.markdown("#### 🧠 Sentiment Breakdown")
        if not sentiment_df.empty:
            sent_counts = sentiment_df["sentiment"].value_counts()
            colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"}
            fig2 = go.Figure(go.Pie(
                labels=sent_counts.index,
                values=sent_counts.values,
                hole=0.5,
                marker=dict(colors=[colors.get(l, "#667eea") for l in sent_counts.index]),
                textfont=dict(size=12)
            ))
            fig2.update_layout(
                paper_bgcolor='#1e2130', plot_bgcolor='#1e2130',
                font=dict(color='#8b9ab0'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
                legend=dict(font=dict(color='#c9d1d9'))
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No news articles found for the selected period.")

    # ── Prediction ──
    st.markdown("---")
    st.markdown("#### 🎯 Movement Prediction")

    model, feature_cols = load_model()
    feature_df = build_features(price_features, sentiment_df)

    if model is not None and feature_cols is not None:
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df[feature_cols]
        prob = model.predict_proba(feature_df)[0]
        pred = model.predict(feature_df)[0]
        confidence = round(max(prob) * 100, 1)

        pred_label = "UP ▲" if pred == 1 else "DOWN ▼"
        pred_color = "up" if pred == 1 else "down"
        pred_interpretation = (
            "Positive sentiment and technical indicators suggest upward movement tomorrow."
            if pred == 1
            else "Negative sentiment or weak technical signals suggest downward pressure tomorrow."
        )

        col_pred, col_conf, col_sent_score = st.columns(3)
        with col_pred:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Predicted Movement</div>
                <div class="metric-value {pred_color}">{pred_label}</div>
            </div>""", unsafe_allow_html=True)
        with col_conf:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color:#667eea;">{confidence:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with col_sent_score:
            sent_score = round(
                (sentiment_df["positive"].mean() - sentiment_df["negative"].mean()) * 100
                if not sentiment_df.empty else 0, 1
            )
            score_color = "up" if sent_score > 0 else "down" if sent_score < 0 else "neutral"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Sentiment Score</div>
                <div class="metric-value {score_color}">{sent_score:+.1f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#1e2130;border-radius:10px;padding:16px 20px;border-left:4px solid {'#2ecc71' if pred==1 else '#e74c3c'};">
            <strong>{'📈' if pred==1 else '📉'} {pred_label} — {confidence}% confidence</strong><br><br>
            {pred_interpretation}<br><br>
            <span style="font-size:11px;color:#8b9ab0;">⚠️ This is a research tool. Not financial advice. Always do your own research before investing.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Trained model not found. Run the EDA notebook to train and save the model first.")

    # ── News feed ──
    st.markdown("---")
    st.markdown(f"#### 📰 Latest News — {ticker}")

    if not sentiment_df.empty:
        for _, row in sentiment_df.head(8).iterrows():
            sent_class = {"positive": "pos", "negative": "neg", "neutral": "neu"}.get(
                row["sentiment"], "neu")
            sent_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(
                row["sentiment"], "🟡")
            conf_pct = round(row["confidence"] * 100, 1)
            st.markdown(f"""
            <div class="news-card {sent_class}">
                <div class="news-title">{row['title']}</div>
                <div class="news-meta">{sent_emoji} {row['sentiment'].title()} ({conf_pct}%) · {row['source']} · {row['date']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No recent news articles found.")

    st.markdown("---")
    st.caption("Model: XGBoost + ProsusAI/FinBERT · Data: NewsAPI + yfinance · Built by Azim Sadath · Not financial advice")
