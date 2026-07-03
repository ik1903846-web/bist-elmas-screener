import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="BIST Elmas Screener",
    layout="wide"
)

st.title("BIST Elmas Screener")
st.caption("Dip bölgesi, sıkışma, momentum dönüşü ve trend teyidi arayan basit BIST tarayıcı.")


# =========================
# SYMBOL LIST
# =========================

@st.cache_data(ttl=60 * 60)
def load_symbols():
    try:
        symbols_df = pd.read_csv("symbols.csv")
        symbols = symbols_df["symbol"].dropna().astype(str).tolist()
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        return symbols
    except Exception:
        return [
            "THYAO.IS", "ASELS.IS", "KCHOL.IS", "SISE.IS", "TUPRS.IS",
            "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "ISCTR.IS", "SAHOL.IS",
            "SASA.IS", "HEKTS.IS", "EREGL.IS", "KRDMD.IS", "BIMAS.IS",
            "FROTO.IS", "TOASO.IS", "DOAS.IS", "KONTR.IS", "ASTOR.IS",
            "PASEU.IS", "ALARK.IS", "ENKAI.IS", "PETKM.IS", "KOZAL.IS",
            "PGSUS.IS", "TAVHL.IS", "TCELL.IS", "TTKOM.IS", "OYAKC.IS",
            "GUBRF.IS", "ODAS.IS", "CANTE.IS", "SMRTG.IS", "GESAN.IS",
            "KCAER.IS", "CWENE.IS", "MIATK.IS", "EUPWR.IS", "QUAGR.IS"
        ]


BIST_SYMBOLS = load_symbols()


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("Filtreler")

    interval_label = st.selectbox(
        "Periyot",
        ["Günlük", "Haftalık", "Aylık"],
        index=1
    )

    min_score = st.slider(
        "Minimum Elmas Skoru",
        min_value=0,
        max_value=100,
        value=40
    )

    selected_symbol = st.selectbox(
        "Grafikte gösterilecek hisse",
        BIST_SYMBOLS
    )

    show_only_elmas = st.checkbox(
        "Sadece Elmas sinyali olanları göster",
        value=False
    )

    show_only_white_angel = st.checkbox(
        "Sadece White Angel sinyali olanları göster",
        value=False
    )

    st.divider()
    st.caption("Bu uygulama yatırım tavsiyesi değildir.")


interval_map = {
    "Günlük": "1d",
    "Haftalık": "1wk",
    "Aylık": "1mo"
}

interval = interval_map[interval_label]


# =========================
# PERIOD SETTINGS
# =========================

def get_period_settings(interval):
    """
    Günlük/Haftalık:
    MA14 / MA50 / MA200

    Aylık:
    MA6 / MA12 / MA24
    """

    if interval == "1mo":
        return {
            "data_period": "10y",
            "fast_len": 6,
            "mid_len": 12,
            "long_len": 24,
            "std_len": 24,
            "range_len": 12,
            "momentum_len": 6,
            "min_bars": 36,
            "fast_name": "MA6",
            "mid_name": "MA12",
            "long_name": "MA24",
            "perf_name": "24 Bar Perf %",
        }

    return {
        "data_period": "5y",
        "fast_len": 14,
        "mid_len": 50,
        "long_len": 200,
        "std_len": 50,
        "range_len": 20,
        "momentum_len": 14,
        "min_bars": 220,
        "fast_name": "MA14",
        "mid_name": "MA50",
        "long_name": "MA200",
        "perf_name": "60 Bar Perf %",
    }


settings = get_period_settings(interval)


# =========================
# DATA
# =========================

@st.cache_data(ttl=60 * 60)
def get_data(symbol, interval):
    try:
        period_settings = get_period_settings(interval)

        df = yf.download(
            symbol,
            period=period_settings["data_period"],
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close"]

        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[required_cols].dropna()

        return df

    except Exception:
        return pd.DataFrame()


# =========================
# INDICATORS
# =========================

def calculate_indicators(df):
    data = df.copy()
    period_settings = get_period_settings(interval)

    fast_len = period_settings["fast_len"]
    mid_len = period_settings["mid_len"]
    long_len = period_settings["long_len"]
    std_len = period_settings["std_len"]
    range_len = period_settings["range_len"]
    momentum_len = period_settings["momentum_len"]

    data["MA_FAST"] = data["Close"].rolling(fast_len).mean()
    data["MA_MID"] = data["Close"].rolling(mid_len).mean()
    data["MA_LONG"] = data["Close"].rolling(long_len).mean()

    data["STD"] = data["Close"].rolling(std_len).std()
    data["Upper"] = data["MA_MID"] + 2 * data["STD"]
    data["Lower"] = data["MA_MID"] - 2 * data["STD"]

    data["Momentum"] = data["Close"] / data["Close"].shift(momentum_len) - 1

    perf_len = 24 if interval == "1mo" else 60
    data["Performance"] = data["Close"] / data["Close"].shift(perf_len) - 1

    data["Range"] = (
        data["High"].rolling(range_len).max() - data["Low"].rolling(range_len).min()
    ) / data["Close"]

    # Dip skoru: fiyat alt banda yaklaştıkça yükselir
    data["DipScoreRaw"] = 100 - (((data["Close"] - data["Lower"]) / data["Close"]) * 250)
    data["DipScoreRaw"] = data["DipScoreRaw"].clip(0, 100)

    # Sıkışma skoru: range daraldıkça yükselir
    data["SqueezeScoreRaw"] = 100 - (data["Range"] * 250)
    data["SqueezeScoreRaw"] = data["SqueezeScoreRaw"].clip(0, 100)

    # Momentum skoru
    data["MomentumScoreRaw"] = 50 + (data["Momentum"] * 250)
    data["MomentumScoreRaw"] = data["MomentumScoreRaw"].clip(0, 100)

    # Ortalama yönü
    data["MAFastUp"] = data["MA_FAST"] > data["MA_FAST"].shift(1)

    # Golden Cross benzeri: orta ortalama uzun ortalamanın üstünde
    data["GoldenCross"] = data["MA_MID"] > data["MA_LONG"]

    # Dip sinyali
    data["DipSignal"] = (
        (data["DipScoreRaw"] >= 65) &
        (data["Close"] <= data["MA_MID"] * 1.15)
    )

    # Momentum dönüş sinyali
    data["MomentumSignal"] = (
        (data["Close"] > data["MA_FAST"]) &
        (data["MAFastUp"]) &
        (data["Momentum"] > 0)
    )

    # White Angel benzeri toparlanma sinyali
    data["WhiteAngelSignal"] = (
        (data["Close"] > data["MA_FAST"]) &
        (data["MAFastUp"]) &
        (data["Momentum"] > 0) &
        (data["Close"] < data["MA_MID"] * 1.20)
    )

    # Elmas sinyali
    data["ElmasSignal"] = (
        (data["DipScoreRaw"] >= 55) &
        (data["SqueezeScoreRaw"] >= 40) &
        (data["Momentum"] > 0) &
        (data["Close"] > data["MA_FAST"]) &
        (data["MAFastUp"])
    )

    return data


# =========================
# ANALYSIS
# =========================

def analyze_symbol(symbol):
    df = get_data(symbol, interval)
    period_settings = get_period_settings(interval)

    if df.empty or len(df) < period_settings["min_bars"]:
        return None

    df = calculate_indicators(df).dropna()

    if df.empty or len(df) < 2:
        return None

    last = df.iloc[-1]

    close = float(last["Close"])
    ma_fast = float(last["MA_FAST"])
    ma_mid = float(last["MA_MID"])
    momentum = float(last["Momentum"])
    performance = float(last["Performance"])

    dip_score = float(last["DipScoreRaw"])
    squeeze_score = float(last["SqueezeScoreRaw"])
    momentum_score = float(last["MomentumScoreRaw"])

    golden_cross = bool(last["GoldenCross"])
    ma_fast_up = bool(last["MAFastUp"])
    white_angel = bool(last["WhiteAngelSignal"])
    elmas_signal = bool(last["ElmasSignal"])
    dip_signal = bool(last["DipSignal"])
    momentum_signal = bool(last["MomentumSignal"])

    cluster = 0

    if dip_score >= 60:
        cluster += 1

    if squeeze_score >= 50:
        cluster += 1

    if momentum > 0:
        cluster += 1

    if close > ma_fast:
        cluster += 1

    if ma_fast_up:
        cluster += 1

    if golden_cross:
        cluster += 1

    if white_angel:
        cluster += 1

    if elmas_signal:
        cluster += 1

    elmas_score = (
        dip_score * 0.35 +
        squeeze_score * 0.20 +
        momentum_score * 0.25 +
        cluster * 4
    )

    if white_angel:
        elmas_score += 8

    if golden_cross:
        elmas_score += 5

    if elmas_signal:
        elmas_score += 12

    elmas_score = max(0, min(100, elmas_score))

    return {
        "Sembol": symbol.replace(".IS", ""),
        "Fiyat": round(close, 2),
        "Elmas Skoru": round(elmas_score, 1),
        "Dip Skoru": round(dip_score, 1),
        "Sıkışma": round(squeeze_score, 1),
        "Momentum %": round(momentum * 100, 2),
        period_settings["perf_name"]: round(performance * 100, 2),
        period_settings["fast_name"]: round(ma_fast, 2),
        period_settings["mid_name"]: round(ma_mid, 2),
        "Golden Cross": "Evet" if golden_cross else "Hayır",
        "White Angel": "Evet" if white_angel else "Hayır",
        "Dip": "Evet" if dip_signal else "Hayır",
        "Momentum": "Evet" if momentum_signal else "Hayır",
        "Elmas": "Evet" if elmas_signal else "Hayır",
        "Cluster": cluster
    }


# =========================
# CHART
# =========================

def make_chart(symbol):
    df = get_data(symbol, interval)
    period_settings = get_period_settings(interval)

    if df.empty:
        st.warning("Veri alınamadı.")
        return

    df = calculate_indicators(df).dropna()

    if df.empty:
        st.warning("Grafik için yeterli veri yok.")
        return

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Fiyat"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA_FAST"],
        name=period_settings["fast_name"],
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA_MID"],
        name=period_settings["mid_name"],
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA_LONG"],
        name=period_settings["long_name"],
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Upper"],
        name="Üst Bant",
        mode="lines",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Lower"],
        name="Alt Bant",
        mode="lines",
        line=dict(dash="dot")
    ))

    dip_points = df[df["DipSignal"]]
    momentum_points = df[df["MomentumSignal"]]
    white_angel_points = df[df["WhiteAngelSignal"]]
    elmas_points = df[df["ElmasSignal"]]

    fig.add_trace(go.Scatter(
        x=dip_points.index,
        y=dip_points["Low"] * 0.98,
        mode="markers",
        name="Dip",
        marker=dict(size=8, symbol="circle")
    ))

    fig.add_trace(go.Scatter(
        x=momentum_points.index,
        y=momentum_points["Low"] * 0.96,
        mode="markers",
        name="Momentum",
        marker=dict(size=9, symbol="triangle-up")
    ))

    fig.add_trace(go.Scatter(
        x=white_angel_points.index,
        y=white_angel_points["Low"] * 0.94,
        mode="markers",
        name="White Angel",
        marker=dict(size=10, symbol="star")
    ))

    fig.add_trace(go.Scatter(
        x=elmas_points.index,
        y=elmas_points["Low"] * 0.92,
        mode="markers+text",
        name="Elmas",
        text=["Elmas"] * len(elmas_points),
        textposition="bottom center",
        marker=dict(size=13, symbol="diamond")
    ))

    title_text = f"{symbol.replace('.IS', '')} - {interval_label}"

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=title_text,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=10, r=10, t=80, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# MAIN APP
# =========================

with st.spinner("BIST hisseleri taranıyor..."):
    rows = []

    for symbol in BIST_SYMBOLS:
        result = analyze_symbol(symbol)

        if result:
            rows.append(result)

df_result = pd.DataFrame(rows)

if df_result.empty:
    st.error("Sonuç üretilemedi. Veri kaynağı cevap vermemiş olabilir veya seçili periyotta yeterli veri yok.")

else:
    df_result = df_result.sort_values("Elmas Skoru", ascending=False)

    filtered = df_result[df_result["Elmas Skoru"] >= min_score]

    if show_only_elmas:
        filtered = filtered[filtered["Elmas"] == "Evet"]

    if show_only_white_angel:
        filtered = filtered[filtered["White Angel"] == "Evet"]

    st.subheader("Genel Durum")

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)

    with top_col1:
        st.metric("Taranan Hisse", len(df_result))

    with top_col2:
        st.metric("Filtrelenen", len(filtered))

    with top_col3:
        if not filtered.empty:
            st.metric("En Yüksek Skor", filtered["Elmas Skoru"].max())
        else:
            st.metric("En Yüksek Skor", "-")

    with top_col4:
        elmas_count = len(df_result[df_result["Elmas"] == "Evet"])
        st.metric("Elmas Sinyali", elmas_count)

    st.subheader("Aktif Periyot Ayarları")

    ayar_col1, ayar_col2, ayar_col3, ayar_col4 = st.columns(4)

    with ayar_col1:
        st.metric("Kısa Ortalama", settings["fast_name"])

    with ayar_col2:
        st.metric("Orta Ortalama", settings["mid_name"])

    with ayar_col3:
        st.metric("Uzun Ortalama", settings["long_name"])

    with ayar_col4:
        st.metric("Veri Süresi", settings["data_period"])

    st.subheader("Screener")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Seçili Hisse Analizi")

    selected_clean = selected_symbol.replace(".IS", "")
    selected_row = df_result[df_result["Sembol"] == selected_clean]

    if not selected_row.empty:
        row = selected_row.iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Fiyat", row["Fiyat"])

        with c2:
            st.metric("Elmas Skoru", row["Elmas Skoru"])

        with c3:
            st.metric("Cluster", row["Cluster"])

        c4, c5, c6 = st.columns(3)

        with c4:
            st.metric("Dip Skoru", row["Dip Skoru"])

        with c5:
            st.metric("Momentum %", row["Momentum %"])

        with c6:
            st.metric("Sıkışma", row["Sıkışma"])

        c7, c8, c9 = st.columns(3)

        with c7:
            st.metric("Golden Cross", row["Golden Cross"])

        with c8:
            st.metric("White Angel", row["White Angel"])

        with c9:
            st.metric("Elmas", row["Elmas"])

    else:
        st.warning("Seçili hisse bu periyotta analiz edilemedi. Veri eksik olabilir.")

    st.subheader("Grafik")
    make_chart(selected_symbol)

st.info(
    "Bu uygulama yatırım tavsiyesi değildir. "
    "Sadece teknik tarama, eğitim ve fikir üretme amacıyla hazırlanmıştır."
)
