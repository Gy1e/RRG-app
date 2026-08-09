"""
Liquidity Rotation RRG — Relative Rotation Graph for asset classes & sectors.

Run with:
    streamlit run rrg_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Liquidity Rotation RRG", layout="wide")

# ---------------------------------------------------------------------------
# Preset universes
# ---------------------------------------------------------------------------

ASSET_CLASS_UNIVERSE = {
    "Stocks (SPY)": "SPY",
    "S&P 500 Futures (ES=F)": "ES=F",
    "Nasdaq-100 (^NDX)": "^NDX",
    "Nasdaq-100 Futures (NQ=F)": "NQ=F",
    "Commodities (DBC)": "DBC",
    "Gold (GLD)": "GLD",
    "Crypto (BTC)": "BTC-USD",
    "US Dollar (UUP)": "UUP",
    "Long Bonds (TLT)": "TLT",
    "Cash / T-Bills (BIL)": "BIL",
}

SECTOR_UNIVERSE = {
    "Technology (XLK)": "XLK",
    "Financials (XLF)": "XLF",
    "Health Care (XLV)": "XLV",
    "Energy (XLE)": "XLE",
    "Consumer Discretionary (XLY)": "XLY",
    "Consumer Staples (XLP)": "XLP",
    "Industrials (XLI)": "XLI",
    "Materials (XLB)": "XLB",
    "Utilities (XLU)": "XLU",
    "Communication Services (XLC)": "XLC",
    "Real Estate (XLRE)": "XLRE",
}

# interval -> allowed lookback periods (kept within yfinance's real limits)
INTERVAL_OPTIONS = {
    "1 Hour (intraday)": {"interval": "1h", "periods": ["5d", "1mo", "3mo", "6mo", "1y", "2y"]},
    "1 Day (swing)": {"interval": "1d", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y"]},
    "1 Week (position)": {"interval": "1wk", "periods": ["6mo", "1y", "2y", "5y", "10y"]},
}

QUADRANT_COLORS = {
    "Leading": "rgba(46, 160, 67, 0.15)",
    "Weakening": "rgba(210, 168, 22, 0.15)",
    "Lagging": "rgba(219, 55, 55, 0.15)",
    "Improving": "rgba(31, 111, 235, 0.15)",
}
POINT_COLORS = {
    "Leading": "#2ea043",
    "Weakening": "#d2a816",
    "Lagging": "#db3737",
    "Improving": "#1f6feb",
}


def get_status(x: float, y: float) -> str:
    if x >= 100 and y >= 100:
        return "Leading"
    if x >= 100 and y < 100:
        return "Weakening"
    if x < 100 and y < 100:
        return "Lagging"
    return "Improving"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_prices(tickers: tuple, period: str, interval: str) -> pd.DataFrame:
    """Download close prices for a list of tickers, returning a wide DataFrame."""
    tickers = list(tickers)
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
            close = raw[field]
        else:
            close = raw

    close = close.dropna(how="all")
    # Forward-fill so 24/7 assets (crypto) reconcile with market-hours-only
    # assets (stocks/ETFs) — e.g. SPY's last known price carries over during
    # hours/days it's closed, rather than leaving a gap.
    close = close.ffill()
    return close


def build_composite_benchmark(prices: pd.DataFrame) -> pd.Series:
    """Equal-weighted benchmark built by rebasing every asset to 100 at the
    first common date, then averaging — avoids letting a high-priced asset
    (e.g. BTC) dominate a simple price average."""
    common = prices.dropna()
    if common.empty:
        raise ValueError(
            "No overlapping data across the selected assets for this period/interval. "
            "Try a longer lookback period."
        )
    rebased = common / common.iloc[0] * 100
    return rebased.mean(axis=1)


# ---------------------------------------------------------------------------
# RRG calculation
# ---------------------------------------------------------------------------

def compute_rrg(prices: pd.DataFrame, benchmark: pd.Series, window: int) -> dict:
    """Returns {ticker: DataFrame[RS-Ratio, RS-Momentum]} aligned on date."""
    out = {}
    for col in prices.columns:
        rs = 100 * (prices[col] / benchmark)
        rs_mean = rs.rolling(window=window).mean()
        rs_std = rs.rolling(window=window).std(ddof=0)
        rsr = (100 + (rs - rs_mean) / rs_std).dropna()

        if rsr.empty:
            continue

        rsr_roc = 100 * (rsr / rsr.shift(1) - 1)
        roc_mean = rsr_roc.rolling(window=window).mean()
        roc_std = rsr_roc.rolling(window=window).std(ddof=0)
        rsm = (101 + (rsr_roc - roc_mean) / roc_std).dropna()

        df = pd.DataFrame({"RS-Ratio": rsr, "RS-Momentum": rsm}).dropna()
        if not df.empty:
            out[col] = df
    return out


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("RRG Controls")

st.sidebar.header("1. Universe")
universe_choice = st.sidebar.radio(
    "What are you tracking?",
    ["Asset Classes", "Sectors (S&P 500)", "Custom"],
    help="Asset Classes = liquidity rotation view. Sectors = classic S&P 500 sector rotation.",
)

if universe_choice == "Asset Classes":
    preset = ASSET_CLASS_UNIVERSE
elif universe_choice == "Sectors (S&P 500)":
    preset = SECTOR_UNIVERSE
else:
    preset = {}

if universe_choice != "Custom":
    selected_names = st.sidebar.multiselect(
        "Assets to include", list(preset.keys()), default=list(preset.keys())
    )
    tickers = [preset[n] for n in selected_names]
else:
    custom_input = st.sidebar.text_area(
        "Enter tickers, comma-separated", "SPY,DBC,BTC-USD,UUP,TLT"
    )
    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

st.sidebar.header("2. Benchmark")
bench_mode = st.sidebar.radio(
    "Benchmark type",
    ["Equal-weighted composite of selected assets", "Specific ticker"],
    help="Composite = each asset judged against the group itself (best for spotting "
    "who's rotating into strength at whose expense). Specific ticker = classic "
    "sector-vs-SPY style benchmark.",
)

default_bench_ticker = "SPY" if universe_choice == "Sectors (S&P 500)" else (tickers[0] if tickers else "SPY")
if bench_mode == "Specific ticker":
    benchmark_ticker = st.sidebar.text_input("Benchmark ticker", default_bench_ticker).strip().upper()
else:
    benchmark_ticker = None

st.sidebar.header("3. Timeframe")
tf_choice = st.sidebar.selectbox(
    "Interval",
    list(INTERVAL_OPTIONS.keys()),
    index=1,
    help="Switch to 1 Hour for intraday day-trading signals, 1 Day/1 Week for swing "
    "or position-level rotation.",
)
interval = INTERVAL_OPTIONS[tf_choice]["interval"]
period = st.sidebar.selectbox("Lookback period", INTERVAL_OPTIONS[tf_choice]["periods"], index=1)

st.sidebar.header("4. Calculation")
window = st.sidebar.slider("Smoothing window (bars)", 5, 30, 14)
tail = st.sidebar.slider("Tail length (bars shown per asset)", 2, 20, 5)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Liquidity Rotation RRG")
st.caption(
    "Relative Rotation Graph — watch assets move between quadrants to catch "
    "rotation before it's obvious on a raw price chart."
)

if not tickers:
    st.warning("Add at least two tickers in the sidebar to build the chart.")
    st.stop()

fetch_list = list(tickers)
if benchmark_ticker and benchmark_ticker not in fetch_list:
    fetch_list.append(benchmark_ticker)

with st.spinner("Loading price data..."):
    try:
        prices = load_prices(tuple(fetch_list), period, interval)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

if prices.empty:
    st.error("No data returned. Try a different period/interval or check your tickers.")
    st.stop()

missing = [t for t in fetch_list if t not in prices.columns]
if missing:
    st.warning(f"No data for: {', '.join(missing)} — dropped from the chart.")

asset_cols = [t for t in tickers if t in prices.columns]
if len(asset_cols) < 2:
    st.error("Need at least 2 valid assets with data to plot an RRG.")
    st.stop()

if bench_mode == "Specific ticker":
    if benchmark_ticker not in prices.columns:
        st.error(f"Benchmark ticker '{benchmark_ticker}' has no data.")
        st.stop()
    benchmark_series = prices[benchmark_ticker]
    asset_cols = [c for c in asset_cols if c != benchmark_ticker]
else:
    try:
        benchmark_series = build_composite_benchmark(prices[asset_cols])
    except ValueError as e:
        st.error(str(e))
        st.stop()

rrg_data = compute_rrg(prices[asset_cols], benchmark_series, window)

if not rrg_data:
    st.warning(
        "Not enough bars to compute the RRG with this window/period combo. "
        "Try a shorter window or a longer lookback period."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

all_x = pd.concat([d["RS-Ratio"].tail(tail) for d in rrg_data.values()])
all_y = pd.concat([d["RS-Momentum"].tail(tail) for d in rrg_data.values()])
pad_x = max((all_x.max() - all_x.min()) * 0.25, 1)
pad_y = max((all_y.max() - all_y.min()) * 0.25, 1)
x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
y_min, y_max = all_y.min() - pad_y, all_y.max() + pad_y

fig = go.Figure()

# quadrant backgrounds
fig.add_shape(type="rect", x0=100, x1=x_max, y0=100, y1=y_max,
              fillcolor=QUADRANT_COLORS["Leading"], line_width=0, layer="below")
fig.add_shape(type="rect", x0=100, x1=x_max, y0=y_min, y1=100,
              fillcolor=QUADRANT_COLORS["Weakening"], line_width=0, layer="below")
fig.add_shape(type="rect", x0=x_min, x1=100, y0=y_min, y1=100,
              fillcolor=QUADRANT_COLORS["Lagging"], line_width=0, layer="below")
fig.add_shape(type="rect", x0=x_min, x1=100, y0=100, y1=y_max,
              fillcolor=QUADRANT_COLORS["Improving"], line_width=0, layer="below")
fig.add_vline(x=100, line_width=1, line_color="gray")
fig.add_hline(y=100, line_width=1, line_color="gray")

fig.add_annotation(x=x_max, y=y_max, text="LEADING", showarrow=False, font=dict(color="#2ea043", size=12), xanchor="right", yanchor="top")
fig.add_annotation(x=x_max, y=y_min, text="WEAKENING", showarrow=False, font=dict(color="#d2a816", size=12), xanchor="right", yanchor="bottom")
fig.add_annotation(x=x_min, y=y_min, text="LAGGING", showarrow=False, font=dict(color="#db3737", size=12), xanchor="left", yanchor="bottom")
fig.add_annotation(x=x_min, y=y_max, text="IMPROVING", showarrow=False, font=dict(color="#1f6feb", size=12), xanchor="left", yanchor="top")

for ticker, df in rrg_data.items():
    tail_df = df.tail(tail)
    if tail_df.empty:
        continue
    last_x, last_y = tail_df["RS-Ratio"].iloc[-1], tail_df["RS-Momentum"].iloc[-1]
    status = get_status(last_x, last_y)
    color = POINT_COLORS[status]

    fig.add_trace(go.Scatter(
        x=tail_df["RS-Ratio"], y=tail_df["RS-Momentum"],
        mode="lines+markers",
        line=dict(color=color, width=1.5),
        marker=dict(size=6, color=color, opacity=0.5),
        name=ticker,
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[last_x], y=[last_y],
        mode="markers+text",
        marker=dict(size=14, color=color, line=dict(width=1.5, color="white")),
        text=[ticker], textposition="top center",
        name=ticker,
        hovertemplate=f"<b>{ticker}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
    ))

fig.update_layout(
    xaxis_title="RS-Ratio",
    yaxis_title="RS-Momentum",
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max]),
    height=650,
    showlegend=False,
    margin=dict(l=40, r=40, t=20, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

rows = []
for ticker, df in rrg_data.items():
    if df.empty:
        continue
    last_x, last_y = df["RS-Ratio"].iloc[-1], df["RS-Momentum"].iloc[-1]
    status = get_status(last_x, last_y)
    last_price = prices[ticker].iloc[-1]
    first_price = prices[ticker].dropna().iloc[0]
    pct_change = (last_price / first_price - 1) * 100
    rows.append({
        "Ticker": ticker,
        "Price": round(last_price, 2),
        "Change over window (%)": round(pct_change, 2),
        "RS-Ratio": round(last_x, 2),
        "RS-Momentum": round(last_y, 2),
        "Status": status,
    })

status_df = pd.DataFrame(rows).sort_values("Status")
st.dataframe(status_df, use_container_width=True, hide_index=True)

bench_label = "Equal-weighted composite" if bench_mode != "Specific ticker" else benchmark_ticker
st.caption(f"Benchmark: {bench_label} · Interval: {interval} · Period: {period} · Window: {window} · Tail: {tail}")
