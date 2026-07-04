import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="BIST Elmas Radar V3",
    layout="wide"
)

st.title("BIST Elmas Radar V3")
st.caption(
    "Elmas sinyalinin nerede doğduğunu, kaç mum önce oluştuğunu "
    "ve sinyal sonrası trendin devam edip etmediğini tarar."
)


# ============================================================
# SYMBOL LIST
# ============================================================

@st.cache_data(ttl=60 * 60)
def load_symbols():
    try:
        symbols_df = pd.read_csv("symbols.csv")

        if "symbol" not in symbols_df.columns:
            raise ValueError("symbols.csv içinde 'symbol' kolonu bulunamadı.")

        symbols = (
            symbols_df["symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        symbols = [
            symbol if symbol.endswith(".IS") else f"{symbol}.IS"
            for symbol in symbols
            if symbol
        ]

        return sorted(list(set(symbols)))

    except Exception:
        return [
            "THYAO.IS",
            "ASELS.IS",
            "KCHOL.IS",
            "SISE.IS",
            "TUPRS.IS",
            "AKBNK.IS",
            "GARAN.IS",
            "YKBNK.IS",
            "ISCTR.IS",
            "SAHOL.IS",
            "SASA.IS",
            "HEKTS.IS",
            "EREGL.IS",
            "KRDMD.IS",
            "BIMAS.IS",
            "FROTO.IS",
            "TOASO.IS",
            "DOAS.IS",
            "KONTR.IS",
            "ASTOR.IS",
        ]


BIST_SYMBOLS = load_symbols()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Radar Filtreleri")

    st.metric("Yüklü Sembol", len(BIST_SYMBOLS))

    interval_label = st.selectbox(
        "Periyot",
        ["Günlük", "Haftalık", "Aylık"],
        index=1
    )

    min_score = st.slider(
        "Minimum Sinyal Elmas Skoru",
        min_value=0,
        max_value=100,
        value=45
    )

    max_since = st.slider(
        "Maksimum Kaç Mum Önce",
        min_value=0,
        max_value=120,
        value=52
    )

    status_filter = st.multiselect(
        "Durum",
        [
            "Yeni Elmas",
            "Güncel Aday",
            "Trendde",
            "Geç Kalındı",
            "Bozuldu",
            "Elmas Yok"
        ],
        default=[
            "Yeni Elmas",
            "Güncel Aday",
            "Trendde"
        ]
    )

    selected_symbol = st.selectbox(
        "Grafikte Gösterilecek Hisse",
        BIST_SYMBOLS
    )

    show_only_recent = st.checkbox(
        "Sadece geçmişte Elmas bulunanları göster",
        value=True
    )

    st.divider()

    st.caption(
        "Bu uygulama yatırım tavsiyesi değildir. "
        "Teknik araştırma amacıyla hazırlanmıştır."
    )


# ============================================================
# INTERVAL MAP
# ============================================================

interval_map = {
    "Günlük": "1d",
    "Haftalık": "1wk",
    "Aylık": "1mo"
}

interval = interval_map[interval_label]


# ============================================================
# PERIOD SETTINGS
# ============================================================

def get_period_settings(selected_interval):
    if selected_interval == "1mo":
        return {
            "data_period": "10y",
            "fast_len": 6,
            "mid_len": 12,
            "long_len": 24,
            "std_len": 24,
            "range_len": 12,
            "momentum_len": 6,
            "scan_bars": 24,
            "min_bars": 36,
            "fast_name": "MA6",
            "mid_name": "MA12",
            "long_name": "MA24",
        }

    if selected_interval == "1wk":
        return {
            "data_period": "5y",
            "fast_len": 14,
            "mid_len": 50,
            "long_len": 200,
            "std_len": 50,
            "range_len": 20,
            "momentum_len": 14,
            "scan_bars": 52,
            "min_bars": 220,
            "fast_name": "MA14",
            "mid_name": "MA50",
            "long_name": "MA200",
        }

    return {
        "data_period": "5y",
        "fast_len": 14,
        "mid_len": 50,
        "long_len": 200,
        "std_len": 50,
        "range_len": 20,
        "momentum_len": 14,
        "scan_bars": 120,
        "min_bars": 220,
        "fast_name": "MA14",
        "mid_name": "MA50",
        "long_name": "MA200",
    }


settings = get_period_settings(interval)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=60 * 60)
def get_data(symbol, selected_interval):
    try:
        period_settings = get_period_settings(selected_interval)

        df = yf.download(
            symbol,
            period=period_settings["data_period"],
            interval=selected_interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for column in required_columns:
            if column not in df.columns:
                return pd.DataFrame()

        df = df[required_columns].copy()
        df = df.dropna()

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df, selected_interval):
    data = df.copy()

    period_settings = get_period_settings(selected_interval)

    fast_len = period_settings["fast_len"]
    mid_len = period_settings["mid_len"]
    long_len = period_settings["long_len"]
    std_len = period_settings["std_len"]
    range_len = period_settings["range_len"]
    momentum_len = period_settings["momentum_len"]

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    data["MA_FAST"] = (
        data["Close"]
        .rolling(fast_len)
        .mean()
    )

    data["MA_MID"] = (
        data["Close"]
        .rolling(mid_len)
        .mean()
    )

    data["MA_LONG"] = (
        data["Close"]
        .rolling(long_len)
        .mean()
    )

    # --------------------------------------------------------
    # STANDARD DEVIATION CHANNEL
    # --------------------------------------------------------

    data["STD"] = (
        data["Close"]
        .rolling(std_len)
        .std()
    )

    data["Upper"] = (
        data["MA_MID"] +
        (2.0 * data["STD"])
    )

    data["Lower"] = (
        data["MA_MID"] -
        (2.0 * data["STD"])
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    data["Momentum"] = (
        data["Close"] /
        data["Close"].shift(momentum_len)
    ) - 1.0

    # --------------------------------------------------------
    # RANGE / SQUEEZE
    # --------------------------------------------------------

    highest_range = (
        data["High"]
        .rolling(range_len)
        .max()
    )

    lowest_range = (
        data["Low"]
        .rolling(range_len)
        .min()
    )

    data["Range"] = (
        highest_range -
        lowest_range
    ) / data["Close"]

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    data["DipScore"] = (
        100.0 -
        (
            (
                data["Close"] -
                data["Lower"]
            ) /
            data["Close"]
        ) *
        250.0
    )

    data["DipScore"] = (
        data["DipScore"]
        .clip(0, 100)
    )

    data["SqueezeScore"] = (
        100.0 -
        (
            data["Range"] *
            250.0
        )
    )

    data["SqueezeScore"] = (
        data["SqueezeScore"]
        .clip(0, 100)
    )

    data["MomentumScore"] = (
        50.0 +
        (
            data["Momentum"] *
            250.0
        )
    )

    data["MomentumScore"] = (
        data["MomentumScore"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # TREND CONDITIONS
    # --------------------------------------------------------

    data["MAFastUp"] = (
        data["MA_FAST"] >
        data["MA_FAST"].shift(1)
    )

    data["GoldenCross"] = (
        data["MA_MID"] >
        data["MA_LONG"]
    )

    # --------------------------------------------------------
    # BASE SIGNALS
    # --------------------------------------------------------

    data["DipBase"] = (
        (data["DipScore"] >= 65) &
        (
            data["Close"] <=
            data["MA_MID"] * 1.12
        )
    )

    data["DipSignal"] = (
        data["DipBase"] &
        ~data["DipBase"].shift(1).fillna(False)
    )

    data["MomentumBase"] = (
        (data["Close"] > data["MA_FAST"]) &
        data["MAFastUp"] &
        (data["Momentum"] > 0) &
        (
            data["Close"] <
            data["MA_MID"] * 1.15
        )
    )

    data["MomentumSignal"] = (
        data["MomentumBase"] &
        ~data["MomentumBase"].shift(1).fillna(False)
    )

    data["WhiteAngelBase"] = (
        data["MomentumBase"] &
        (data["DipScore"] >= 35) &
        (data["SqueezeScore"] >= 25)
    )

    data["WhiteAngelSignal"] = (
        data["WhiteAngelBase"] &
        ~data["WhiteAngelBase"].shift(1).fillna(False)
    )

    # --------------------------------------------------------
    # SIMPLE W STRUCTURE
    # --------------------------------------------------------

    rolling_low = (
        data["Low"]
        .rolling(range_len)
        .min()
    )

    data["WBase"] = (
        (
            data["Low"] <=
            rolling_low * 1.03
        ) &
        (
            data["Close"] >
            data["Open"]
        ) &
        (
            data["Close"] >
            data["Close"].shift(1)
        )
    )

    # --------------------------------------------------------
    # CLUSTER
    # --------------------------------------------------------

    cluster = pd.Series(
        0,
        index=data.index,
        dtype="int64"
    )

    cluster += (
        data["DipScore"] >= 60
    ).astype(int)

    cluster += (
        data["SqueezeScore"] >= 50
    ).astype(int)

    cluster += (
        data["Momentum"] > 0
    ).astype(int)

    cluster += (
        data["Close"] >
        data["MA_FAST"]
    ).astype(int)

    cluster += (
        data["MAFastUp"]
    ).astype(int)

    cluster += (
        data["GoldenCross"]
    ).astype(int)

    cluster += (
        data["WhiteAngelBase"]
    ).astype(int)

    cluster += (
        data["WBase"]
    ).astype(int)

    data["Cluster"] = cluster

    # --------------------------------------------------------
    # ELMAS BASE
    # --------------------------------------------------------

    data["ElmasBase"] = (
        (data["DipScore"] >= 55) &
        (data["SqueezeScore"] >= 40) &
        (data["Momentum"] > 0) &
        (data["Close"] > data["MA_FAST"]) &
        data["MAFastUp"] &
        (data["Cluster"] >= 5) &
        (
            data["Close"] <
            data["MA_MID"] * 1.25
        )
    )

    # --------------------------------------------------------
    # FIRST ELMAS TRIGGER
    #
    # Burada amaç:
    # Elmas şartı 8 hafta boyunca devam etse bile
    # yalnızca ilk oluştuğu mumu sinyal kabul etmek.
    # --------------------------------------------------------

    data["ElmasSignal"] = (
        data["ElmasBase"] &
        ~data["ElmasBase"].shift(1).fillna(False)
    )

    # --------------------------------------------------------
    # ELMAS SCORE
    # --------------------------------------------------------

    data["ElmasScore"] = (
        data["DipScore"] * 0.35 +
        data["SqueezeScore"] * 0.20 +
        data["MomentumScore"] * 0.25 +
        data["Cluster"] * 4.0
    )

    data["ElmasScore"] += np.where(
        data["WhiteAngelBase"],
        8.0,
        0.0
    )

    data["ElmasScore"] += np.where(
        data["GoldenCross"],
        5.0,
        0.0
    )

    data["ElmasScore"] += np.where(
        data["ElmasBase"],
        12.0,
        0.0
    )

    data["ElmasScore"] = (
        data["ElmasScore"]
        .clip(0, 100)
    )

    return data


# ============================================================
# STATUS LOGIC
# ============================================================

def calculate_status(
    since,
    perf,
    max_perf,
    trend_alive,
    current_elmas
):
    if current_elmas:
        return "Yeni Elmas"

    if since is None:
        return "Elmas Yok"

    if not trend_alive:
        return "Bozuldu"

    if since <= 3 and perf <= 15:
        return "Yeni Elmas"

    if since <= 10 and perf <= 35:
        return "Güncel Aday"

    if perf <= 100 and trend_alive:
        return "Trendde"

    if perf > 100 or max_perf > 150:
        return "Geç Kalındı"

    return "Trendde"


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(symbol):
    df = get_data(
        symbol,
        interval
    )

    period_settings = get_period_settings(
        interval
    )

    if df.empty:
        return None

    if len(df) < period_settings["min_bars"]:
        return None

    data = calculate_indicators(
        df,
        interval
    )

    data = data.dropna()

    if data.empty:
        return None

    if len(data) < 2:
        return None

    last = data.iloc[-1]

    scan_bars = period_settings["scan_bars"]

    scan_window = data.tail(
        min(
            scan_bars,
            len(data)
        )
    )

    elmas_rows = scan_window[
        scan_window["ElmasSignal"]
    ]

    current_elmas = bool(
        last["ElmasSignal"]
    )

    # --------------------------------------------------------
    # NO HISTORICAL ELMAS FOUND
    # --------------------------------------------------------

    if elmas_rows.empty:
        current_status = calculate_status(
            since=None,
            perf=0,
            max_perf=0,
            trend_alive=False,
            current_elmas=current_elmas
        )

        return {
            "Sembol": symbol.replace(".IS", ""),
            "Fiyat": round(float(last["Close"]), 2),
            "Aktif Elmas": "Evet" if current_elmas else "Hayır",
            "Son Elmas": "Yok",
            "Elmas Since": None,
            "Sinyal Fiyatı": None,
            "Elmas Perf %": None,
            "Max Perf %": None,
            "Sinyal Skoru": None,
            "Sinyal Cluster": None,
            "Güncel Skor": round(float(last["ElmasScore"]), 1),
            "Güncel Cluster": int(last["Cluster"]),
            "Momentum %": round(float(last["Momentum"]) * 100, 2),
            "Trend": "Hayır",
            "Durum": current_status
        }

    # --------------------------------------------------------
    # FIND MOST RECENT ELMAS
    # --------------------------------------------------------

    signal_index = elmas_rows.index[-1]

    signal_position = data.index.get_loc(
        signal_index
    )

    current_position = len(data) - 1

    since = (
        current_position -
        signal_position
    )

    signal_row = data.loc[
        signal_index
    ]

    signal_price = float(
        signal_row["Close"]
    )

    signal_score = float(
        signal_row["ElmasScore"]
    )

    signal_cluster = int(
        signal_row["Cluster"]
    )

    current_price = float(
        last["Close"]
    )

    perf = (
        (
            current_price /
            signal_price
        ) -
        1.0
    ) * 100.0

    after_signal = data.loc[
        signal_index:
    ]

    max_high = float(
        after_signal["High"].max()
    )

    max_perf = (
        (
            max_high /
            signal_price
        ) -
        1.0
    ) * 100.0

    # --------------------------------------------------------
    # TREND ALIVE
    # --------------------------------------------------------

    trend_alive = bool(
        (
            last["Close"] >
            last["MA_FAST"]
        ) and
        (
            last["MA_FAST"] >
            last["MA_FAST"].shift(1)
            if False
            else last["MAFastUp"]
        )
    )

    # Daha güçlü trend kontrolü
    trend_alive = bool(
        (
            last["Close"] >
            last["MA_FAST"]
        ) and
        bool(last["MAFastUp"]) and
        (
            last["Momentum"] >
            -0.05
        )
    )

    status = calculate_status(
        since=since,
        perf=perf,
        max_perf=max_perf,
        trend_alive=trend_alive,
        current_elmas=current_elmas
    )

    return {
        "Sembol": symbol.replace(".IS", ""),
        "Fiyat": round(current_price, 2),
        "Aktif Elmas": "Evet" if current_elmas else "Hayır",
        "Son Elmas": signal_index.strftime("%Y-%m-%d"),
        "Elmas Since": int(since),
        "Sinyal Fiyatı": round(signal_price, 2),
        "Elmas Perf %": round(perf, 2),
        "Max Perf %": round(max_perf, 2),
        "Sinyal Skoru": round(signal_score, 1),
        "Sinyal Cluster": signal_cluster,
        "Güncel Skor": round(float(last["ElmasScore"]), 1),
        "Güncel Cluster": int(last["Cluster"]),
        "Momentum %": round(float(last["Momentum"]) * 100, 2),
        "Trend": "Evet" if trend_alive else "Hayır",
        "Durum": status
    }


# ============================================================
# CHART
# ============================================================

def make_chart(symbol):
    df = get_data(
        symbol,
        interval
    )

    if df.empty:
        st.warning(
            "Grafik verisi alınamadı."
        )
        return

    data = calculate_indicators(
        df,
        interval
    )

    data = data.dropna()

    if data.empty:
        st.warning(
            "Grafik için yeterli veri yok."
        )
        return

    period_settings = get_period_settings(
        interval
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # CANDLESTICKS
    # --------------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Fiyat"
        )
    )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA_FAST"],
            name=period_settings["fast_name"],
            mode="lines"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA_MID"],
            name=period_settings["mid_name"],
            mode="lines"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA_LONG"],
            name=period_settings["long_name"],
            mode="lines"
        )
    )

    # --------------------------------------------------------
    # STD BANDS
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Upper"],
            name="Üst Bant",
            mode="lines",
            line=dict(
                dash="dot"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Lower"],
            name="Alt Bant",
            mode="lines",
            line=dict(
                dash="dot"
            )
        )
    )

    # --------------------------------------------------------
    # SIGNAL POINTS
    # --------------------------------------------------------

    dip_points = data[
        data["DipSignal"]
    ]

    momentum_points = data[
        data["MomentumSignal"]
    ]

    wa_points = data[
        data["WhiteAngelSignal"]
    ]

    elmas_points = data[
        data["ElmasSignal"]
    ]

    fig.add_trace(
        go.Scatter(
            x=dip_points.index,
            y=dip_points["Low"] * 0.98,
            mode="markers",
            name="Dip",
            marker=dict(
                size=7,
                symbol="circle"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=momentum_points.index,
            y=momentum_points["Low"] * 0.96,
            mode="markers",
            name="Momentum",
            marker=dict(
                size=8,
                symbol="triangle-up"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=wa_points.index,
            y=wa_points["Low"] * 0.94,
            mode="markers+text",
            name="White Angel",
            text=[
                "WA"
            ] * len(wa_points),
            textposition="bottom center",
            marker=dict(
                size=9,
                symbol="star"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=elmas_points.index,
            y=elmas_points["Low"] * 0.91,
            mode="markers+text",
            name="Elmas",
            text=[
                "ELMAS"
            ] * len(elmas_points),
            textposition="bottom center",
            marker=dict(
                size=14,
                symbol="diamond"
            )
        )
    )

    # --------------------------------------------------------
    # LATEST ELMAS LEVELS
    # --------------------------------------------------------

    if not elmas_points.empty:
        latest_signal_index = (
            elmas_points.index[-1]
        )

        latest_signal = data.loc[
            latest_signal_index
        ]

        signal_price = float(
            latest_signal["Close"]
        )

        support_price = float(
            data.loc[
                :latest_signal_index,
                "Low"
            ]
            .tail(
                period_settings["range_len"]
            )
            .min()
        )

        upper_target = float(
            latest_signal["Upper"]
        )

        fig.add_hline(
            y=support_price,
            line_dash="dash",
            annotation_text="Elmas Destek"
        )

        fig.add_hline(
            y=signal_price,
            line_dash="dash",
            annotation_text="Elmas Tetik"
        )

        fig.add_hline(
            y=upper_target,
            line_dash="dot",
            annotation_text="İlk Üst Bant"
        )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=(
            f"{symbol.replace('.IS', '')} "
            f"- {interval_label}"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(
            l=10,
            r=10,
            t=80,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SCAN
# ============================================================

with st.spinner(
    f"{len(BIST_SYMBOLS)} sembol taranıyor..."
):
    rows = []

    for symbol in BIST_SYMBOLS:
        result = analyze_symbol(
            symbol
        )

        if result is not None:
            rows.append(
                result
            )


df_result = pd.DataFrame(
    rows
)


# ============================================================
# MAIN APP
# ============================================================

if df_result.empty:
    st.error(
        "Sonuç üretilemedi. "
        "Veri kaynağı cevap vermemiş olabilir "
        "veya periyotta yeterli veri bulunmuyor."
    )

else:
    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filtered = df_result.copy()

    if show_only_recent:
        filtered = filtered[
            filtered["Elmas Since"].notna()
        ]

    filtered = filtered[
        (
            filtered["Sinyal Skoru"].fillna(0) >=
            min_score
        )
    ]

    filtered = filtered[
        (
            filtered["Elmas Since"].fillna(999) <=
            max_since
        )
    ]

    if status_filter:
        filtered = filtered[
            filtered["Durum"].isin(
                status_filter
            )
        ]

    # --------------------------------------------------------
    # RADAR PRIORITY
    # --------------------------------------------------------

    status_priority = {
        "Yeni Elmas": 1,
        "Güncel Aday": 2,
        "Trendde": 3,
        "Geç Kalındı": 4,
        "Bozuldu": 5,
        "Elmas Yok": 6
    }

    filtered["Durum Öncelik"] = (
        filtered["Durum"]
        .map(status_priority)
        .fillna(99)
    )

    filtered = filtered.sort_values(
        by=[
            "Durum Öncelik",
            "Elmas Since",
            "Sinyal Skoru"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )

    # --------------------------------------------------------
    # GENERAL STATUS
    # --------------------------------------------------------

    st.subheader(
        "Radar Genel Durum"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Taranan Hisse",
            len(df_result)
        )

    with c2:
        st.metric(
            "Radardaki Aday",
            len(filtered)
        )

    with c3:
        recent_count = len(
            df_result[
                df_result["Elmas Since"].notna()
            ]
        )

        st.metric(
            "Elmas Geçmişi Bulunan",
            recent_count
        )

    with c4:
        new_count = len(
            df_result[
                df_result["Durum"] ==
                "Yeni Elmas"
            ]
        )

        st.metric(
            "Yeni Elmas",
            new_count
        )

    # --------------------------------------------------------
    # PERIOD INFO
    # --------------------------------------------------------

    st.subheader(
        "Aktif Tarama Mantığı"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "Elmas Arama Penceresi",
            f'{settings["scan_bars"]} Mum'
        )

    with p2:
        st.metric(
            "Kısa Ortalama",
            settings["fast_name"]
        )

    with p3:
        st.metric(
            "Orta Ortalama",
            settings["mid_name"]
        )

    with p4:
        st.metric(
            "Uzun Ortalama",
            settings["long_name"]
        )

    # --------------------------------------------------------
    # SCREENER TABLE
    # --------------------------------------------------------

    st.subheader(
        "Elmas Radar"
    )

    visible_columns = [
        "Sembol",
        "Fiyat",
        "Aktif Elmas",
        "Elmas Since",
        "Sinyal Fiyatı",
        "Elmas Perf %",
        "Max Perf %",
        "Sinyal Skoru",
        "Sinyal Cluster",
        "Güncel Skor",
        "Momentum %",
        "Trend",
        "Durum"
    ]

    st.dataframe(
        filtered[
            visible_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SELECTED STOCK ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "Seçili Hisse Analizi"
    )

    selected_clean = (
        selected_symbol
        .replace(".IS", "")
    )

    selected_row = df_result[
        df_result["Sembol"] ==
        selected_clean
    ]

    if selected_row.empty:
        st.warning(
            "Seçili hisse analiz edilemedi."
        )

    else:
        row = selected_row.iloc[0]

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric(
                "Fiyat",
                row["Fiyat"]
            )

        with s2:
            st.metric(
                "Durum",
                row["Durum"]
            )

        with s3:
            st.metric(
                "Aktif Elmas",
                row["Aktif Elmas"]
            )

        s4, s5, s6 = st.columns(3)

        with s4:
            st.metric(
                "Elmas Since",
                (
                    row["Elmas Since"]
                    if pd.notna(row["Elmas Since"])
                    else "-"
                )
            )

        with s5:
            st.metric(
                "Sinyal Fiyatı",
                (
                    row["Sinyal Fiyatı"]
                    if pd.notna(row["Sinyal Fiyatı"])
                    else "-"
                )
            )

        with s6:
            st.metric(
                "Elmas Perf %",
                (
                    row["Elmas Perf %"]
                    if pd.notna(row["Elmas Perf %"])
                    else "-"
                )
            )

        s7, s8, s9 = st.columns(3)

        with s7:
            st.metric(
                "Max Perf %",
                (
                    row["Max Perf %"]
                    if pd.notna(row["Max Perf %"])
                    else "-"
                )
            )

        with s8:
            st.metric(
                "Sinyal Skoru",
                (
                    row["Sinyal Skoru"]
                    if pd.notna(row["Sinyal Skoru"])
                    else "-"
                )
            )

        with s9:
            st.metric(
                "Sinyal Cluster",
                (
                    row["Sinyal Cluster"]
                    if pd.notna(row["Sinyal Cluster"])
                    else "-"
                )
            )

        s10, s11, s12 = st.columns(3)

        with s10:
            st.metric(
                "Güncel Skor",
                row["Güncel Skor"]
            )

        with s11:
            st.metric(
                "Momentum %",
                row["Momentum %"]
            )

        with s12:
            st.metric(
                "Trend",
                row["Trend"]
            )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader(
        "Grafik"
    )

    make_chart(
        selected_symbol
    )


st.info(
    "Elmas Radar yalnızca matematiksel ve teknik koşulları tarar. "
    "Bir sinyalin geçmişte oluşmuş olması gelecekte yükseliş garantisi değildir."
)