# Liquidity Rotation Graph

A Streamlit app for visualizing Liquidity Rotation Graphs across asset
classes (stocks, commodities, gold, crypto, dollar, bonds/cash) or classic
S&P 500 sectors — with adjustable timeframes.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run rrg_app.py
```

This opens a local web page (usually `http://localhost:8501`) in your
browser. It's running on your own machine — nothing is deployed publicly
unless you choose to host it elsewhere.

## What you can adjust live, no code changes needed

- **Universe**: Asset Classes / Sectors (S&P 500) / Custom ticker list
- **Benchmark**: equal-weighted composite of your selected assets, or a
  specific ticker (e.g. SPY)
- **Interval**: 1 Hour (intraday), 1 Day (swing), 1 Week (position) — each
  has its own valid lookback-period options, since intraday data only goes
  back so far on Yahoo Finance
- **Lookback period**: e.g. 5d, 1mo, 1y, 5y depending on interval
- **Smoothing window** and **tail length** (how many trailing bars are drawn
  per asset)

## Notes

- Data comes from Yahoo Finance via `yfinance`. Free intraday data (1h bars)
  is limited to roughly the last 2 years by Yahoo, and finer intervals (1m,
  5m, etc.) go back even less — this app sticks to 1h/1d/1wk to stay within
  what's reliably available.
- The composite benchmark rebases every asset to 100 at the first common
  date before averaging, so a high-priced asset like BTC doesn't dominate
  a simple price average.
- Click **Refresh data** in the sidebar to clear the cache and re-pull
  prices (data is otherwise cached for 5 minutes).
