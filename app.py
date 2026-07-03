import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="BIST Elmas Screener",
    layout="wide"
)

st.title("BIST Elmas Screener")
st.caption("Dip bölgesi, momentum dönüşü ve trend teyidi arayan basit BIST tarayıcı.")

BIST_SYMBOLS = [
    "THYAO.IS", "ASELS.IS", "KCHOL.IS", "SISE.IS", "TUPRS.IS",
    "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "ISCTR.IS", "SAHOL.IS",
    "SASA.IS", "HEKTS.IS", "EREGL.IS", "KRDMD.IS", "BIMAS.IS",
    "FROTO.IS", "TOASO.IS", "DOAS.IS", "KONTR.IS", "ASTOR.IS",
    "PASEU.IS", "ALARK.IS", "ENKAI.IS", "PETKM.IS", "KOZAL.IS"
]

with st.sidebar:
    st.header("Filtreler")
    interval_label = st.selectbox("Periyot", ["Günlük", "Haftalık", "Aylık"])
    min_score = st.slider("Minimum Elmas Skoru", 0, 100, 50)
    selected_symbol = st.selectbox("Grafikte gösterilecek hisse", BIST_SYMBOLS)

interval_map = {
    "Günlük": "1d",
    "Haftalık": "1wk",
    "Aylık": "1mo"
}

interval = interval_map[interval_label]


@st.cache_data(ttl=60 * 60)
def get_data(symbol, interval):
    df = yf.download(
        symbol,
        period="5y",
        interval=interval,
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


def calculate_indicators(df):
    data = df.copy()

    data["MA14"] = data["Close"].rolling(14).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    data["STD50"] = data["Close"].rolling(50).std()
    data["Upper"] = data["MA50"] + 2 * data["STD50"]
    data["Lower"] = data["MA50"] - 2 * data["STD50"]

    data["Momentum14"] = data["Close"] / data["Close"].shift(14) - 1
    data["Perf60"] = data["Close"] / data["Close"].shift(60) - 1

    data["Range20"] = (
        data["High"].rolling(20).max() - data["Low"].rolling(20).min()
    ) / data["Close"]

    return data


def analyze_symbol(symbol):
    df = get_data(symbol, interval)

    if df.empty or len(df) < 220:
        return None

    df = calculate_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    ma14 = float(last["MA14"])
    ma50 = float(last["MA50"])
    ma200 = float(last["MA200"])
    lower = float(last["Lower"])
    momentum14 = float(last["Momentum14"])
    perf60 = float(last["Perf60"])
    range20 = float(last["Range20"])

    if np.isnan([close, ma14, ma50, ma200, lower, momentum14, perf60, range20]).any():
        return None

    # Dip skoru: fiyat alt banda yaklaştıkça skor artar
    distance_to_lower = (close - lower) / close
    dip_score = 100 - (distance_to_lower * 250)
    dip_score = max(0, min(100, dip_score))

    # Sıkışma skoru: son 20 bar aralığı daraldıkça skor artar
    squeeze_score = 100 - (range20 * 250)
    squeeze_score = max(0, min(100, squeeze_score))

    # Momentum skoru
    momentum_score = 50 + (momentum14 * 250)
    momentum_score = max(0, min(100, momentum_score))

    golden_cross = ma50 > ma200
    ma14_up = last["MA14"] > prev["MA14"]

    white_angel = (
        close > ma14 and
        ma14_up and
        momentum14 > 0 and
        close < ma50 * 1.20
    )

    cluster = 0
    if dip_score >= 60:
        cluster += 1
    if squeeze_score >= 50:
        cluster += 1
    if momentum14 > 0:
        cluster += 1
    if close > ma14:
        cluster += 1
    if golden_cross:
        cluster += 1
    if white_angel:
        cluster += 1

    elmas_score = (
        dip_score * 0.35 +
        squeeze_score * 0.20 +
        momentum_score * 0.25 +
        cluster * 5
    )

    if white_angel:
        elmas_score += 10

    if golden_cross:
        elmas_score += 5

    elmas_score = max(0, min(100, elmas_score))

    return {
        "Sembol": symbol.replace(".IS", ""),
        "Fiyat": round(close, 2),
        "Elmas Skoru": round(elmas_score, 1),
        "Dip Skoru": round(dip_score, 1),
        "Sıkışma": round(squeeze_score, 1),
        "Momentum 14 %": round(momentum14 * 100, 2),
        "60 Bar Perf %": round(perf60 * 100, 2),
        "Golden Cross": "Evet" if golden_cross else "Hayır",
        "White Angel": "Evet" if white_angel else "Hayır",
        "Cluster": cluster
    }


def make_chart(symbol):
    df = get_data(symbol, interval)

    if df.empty:
        st.warning("Veri alınamadı.")
        return

    df = calculate_indicators(df).dropna()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Fiyat"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["MA14"], name="MA14"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Üst Bant"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Alt Bant"))

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=symbol.replace(".IS", "")
    )

    st.plotly_chart(fig, use_container_width=True)


with st.spinner("BIST hisseleri taranıyor..."):
    rows = []
    for symbol in BIST_SYMBOLS:
        result = analyze_symbol(symbol)
        if result:
            rows.append(result)

df_result = pd.DataFrame(rows)

if df_result.empty:
    st.error("Sonuç üretilemedi. Veri kaynağı cevap vermemiş olabilir.")
else:
    df_result = df_result.sort_values("Elmas Skoru", ascending=False)
    filtered = df_result[df_result["Elmas Skoru"] >= min_score]

    st.subheader("Screener")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Grafik")
    make_chart(selected_symbol)

st.info("Bu uygulama yatırım tavsiyesi değildir. Sadece teknik tarama ve eğitim amaçlıdır.")
