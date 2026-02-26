"""
regime/data.py
==============
Fetches all data needed for regime classification via yfinance.

TIMESTAMP SEMANTICS:
  - All data fetched is EOD (post-close)
  - available_at = T+0 post-close
  - regime output is valid for T+1 trading decisions (shift applied in classifier)

KNOWN LIMITATIONS (yfinance):
  - No point-in-time guarantee — yfinance can silently backfill adjusted prices
  - Breadth computed from static index samples — update lists quarterly
  - VIX from CBOE via Yahoo — occasional missing days, forward-filled here
  - For production: replace with Norgate or Tiingo for clean adjusted data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ── SPX sample (~100 liquid members) ──────────────────────────────────────────
# Broad market health. Update quarterly.
SPX_BREADTH_SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","JPM","AVGO",
    "XOM","TSLA","UNH","JNJ","V","PG","MA","HD","COST","MRK","ABBV","CVX",
    "CRM","BAC","NFLX","AMD","PEP","KO","TMO","ADBE","WMT","ACN","MCD","ABT",
    "LIN","DHR","TXN","NEE","PM","ORCL","QCOM","HON","UPS","AMGN","LOW",
    "INTU","CAT","GE","IBM","SPGI","GS","AXP","MS","RTX","ISRG","SYK","BLK",
    "GILD","ELV","VRTX","MDT","PLD","CI","CB","REGN","ADI","LRCX","PANW",
    "AMAT","MU","KLAC","SNPS","CDNS","APH","MCO","MMC","ITW","PH","EMR",
    "ETN","CME","ICE","AON","MSCI","IDXX","A","ZBH","ZTS","DXCM","WST",
    "MTD","RMD","COO","EW","HOLX","ALGN","PODD","TECH","BAX","BDX","BSX",
]

# ── NDX sample (~60 core Nasdaq-100 members) ──────────────────────────────────
# Tech/growth health. More relevant for a tech-heavy portfolio.
# NDX is concentrated — 60 names covers >85% of index weight.
# WHY NDX MATTERS: SPX can look "fine" (financials/energy holding up)
# while NDX tech is quietly rolling over. 2022 Q1 is the textbook example.
# Update quarterly — NDX rebalances March/June/September/December.
NDX_BREADTH_SAMPLE = [
    # Mega-cap tech (dominate index weight)
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    # Semiconductors
    "AMD","QCOM","TXN","AMAT","MU","LRCX","KLAC","MRVL","ON","MCHP",
    # Software / cloud
    "ADBE","CRM","INTU","ORCL","PANW","SNPS","CDNS","FTNT","WDAY","TEAM",
    # Next-gen SaaS / cybersecurity
    "CRWD","DDOG","ZS","NET","SNOW","MDB","HUBS","NOW","VEEV","TTD",
    # Consumer tech / platforms
    "NFLX","ABNB","PYPL","MELI","PDD",
    # Biotech on NDX
    "AMGN","REGN","VRTX","GILD","BIIB","IDXX","DXCM","ILMN",
    # Other NDX constituents
    "ISRG","MDLZ","MNST","FAST","PAYX","ODFL","CTAS","FANG",
]


def fetch_all(
    lookback_days: int = 504,
    spx_ticker: str = "^GSPC",
    qqq_ticker: str = "QQQ",
    vix_ticker: str = "^VIX",
    vix_short_ticker: str = "^VIX9D",
    vix_long_ticker: str = "^VIX3M",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Fetch SPX, QQQ, VIX, VIX term-structure legs, SPX breadth, and NDX breadth.

    Returns (9-tuple):
        spx             : SPX close
        qqq             : QQQ close
        vix             : VIX spot close (30-day)
        spx_breadth_50  : % SPX sample above 50d SMA
        spx_breadth_200 : % SPX sample above 200d SMA
        ndx_breadth_50  : % NDX sample above 50d SMA
        ndx_breadth_200 : % NDX sample above 200d SMA
        vix_short       : 9-day VIX (^VIX9D) — None series if unavailable
        vix_long        : 3-month VIX (^VIX3M) — None series if unavailable

    VIX term structure: contango when vix_short < vix_long (low near-term fear),
    backwardation when vix_short > vix_long (near-term fear elevated vs medium-term).
    """
    start = (date.today() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching market data {start} -> {end}")

    # --- SPX ---
    spx_raw = yf.download(spx_ticker, start=start, end=end, progress=False, auto_adjust=True)
    if spx_raw.empty:
        raise ValueError(f"No SPX data for {spx_ticker}")
    spx = spx_raw["Close"].squeeze()
    spx.index = pd.to_datetime(spx.index).normalize()

    # --- QQQ ---
    qqq_raw = yf.download(qqq_ticker, start=start, end=end, progress=False, auto_adjust=True)
    if qqq_raw.empty:
        raise ValueError(f"No QQQ data for {qqq_ticker}")
    qqq = qqq_raw["Close"].squeeze()
    qqq.index = pd.to_datetime(qqq.index).normalize()
    qqq = qqq.reindex(spx.index).ffill()

    # --- VIX spot ---
    vix_raw = yf.download(vix_ticker, start=start, end=end, progress=False, auto_adjust=False)
    if vix_raw.empty:
        raise ValueError(f"No VIX data for {vix_ticker}")
    vix = vix_raw["Close"].squeeze()
    vix.index = pd.to_datetime(vix.index).normalize()
    vix = vix.reindex(spx.index).ffill()

    # --- VIX term structure (best-effort — degrade to None if unavailable) ---
    vix_short = _fetch_vix_leg(vix_short_ticker, start, end, spx.index, label="VIX9D")
    vix_long  = _fetch_vix_leg(vix_long_ticker,  start, end, spx.index, label="VIX3M")

    # --- SPX breadth ---
    logger.info(f"Fetching SPX breadth ({len(SPX_BREADTH_SAMPLE)} tickers)...")
    spx_b50, spx_b200 = _compute_breadth(SPX_BREADTH_SAMPLE, start, end, spx.index, label="SPX")

    # --- NDX breadth ---
    logger.info(f"Fetching NDX breadth ({len(NDX_BREADTH_SAMPLE)} tickers)...")
    ndx_b50, ndx_b200 = _compute_breadth(NDX_BREADTH_SAMPLE, start, end, spx.index, label="NDX")

    # --- Validation ---
    if len(spx) < 210:
        raise ValueError(f"Only {len(spx)} SPX bars — need at least 210")

    logger.info(f"Data ready: {len(spx)} days, latest={spx.index[-1].date()}")
    return spx, qqq, vix, spx_b50, spx_b200, ndx_b50, ndx_b200, vix_short, vix_long


def _fetch_vix_leg(
    ticker: str,
    start: str,
    end: str,
    target_index: pd.DatetimeIndex,
    label: str = "",
) -> Optional[pd.Series]:
    """Fetch a single VIX index series; returns None if unavailable."""
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if raw.empty:
            logger.warning(f"  {label} ({ticker}): no data — term structure will show n/a")
            return None
        s = raw["Close"].squeeze()
        s.index = pd.to_datetime(s.index).normalize()
        s = s.reindex(target_index).ffill()
        logger.info(f"  {label} ({ticker}): fetched, latest={float(s.dropna().iloc[-1]):.2f}")
        return s
    except Exception as e:
        logger.warning(f"  {label} ({ticker}): fetch failed ({e}) — term structure will show n/a")
        return None


def _compute_breadth(
    tickers: list,
    start: str,
    end: str,
    target_index: pd.DatetimeIndex,
    label: str = "",
) -> Tuple[pd.Series, pd.Series]:
    """Batch download and compute % above 50d/200d SMA."""
    raw = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=True)["Close"]
    raw.index = pd.to_datetime(raw.index).normalize()

    # Drop sparse tickers
    raw = raw.loc[:, raw.isna().mean() < 0.20]
    valid_n = raw.shape[1]
    logger.info(f"  {label} breadth: {valid_n}/{len(tickers)} tickers valid")

    if valid_n < 15:
        raise ValueError(f"Too few valid {label} tickers ({valid_n})")

    sma50 = raw.rolling(50).mean()
    sma200 = raw.rolling(200).mean()

    b50 = (raw > sma50).astype(float).mean(axis=1).reindex(target_index).ffill()
    b200 = (raw > sma200).astype(float).mean(axis=1).reindex(target_index).ffill()

    return b50, b200
