"""
regime/classifier.py
====================
Regime classification logic.

REGIMES (think of it as a 4-state traffic light):
  bull_trend   (GREEN)        → all momentum strategies active, full sizing
  chop         (YELLOW-EARLY) → mixed signals, reduce size or skip new entries
  distribution (YELLOW-LATE)  → market weakening, no new entries, manage exits
  crash        (RED)          → severe downtrend/vol spike, flat, protect capital

TIMESTAMP CRITICAL:
  - Inputs: EOD data, available_at = T+0 post-close
  - Output `regime_for_trading`: shifted +1 day — valid for T+1 decisions
  - NEVER use regime[T] to make a trade on T — that's look-ahead
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date
from typing import Optional

from regime_dashboard import RegimeSnapshot, render_dashboard

logger = logging.getLogger(__name__)

REGIME_ORDER = ["bull_trend", "chop", "distribution", "crash"]
REGIME_NUMERIC = {"bull_trend": 1, "chop": 0, "distribution": -1, "crash": -2}
REGIME_EMOJI = {"bull_trend": "🟢", "chop": "🟡", "distribution": "🟠", "crash": "🔴"}


@dataclass
class RegimeConfig:
    # Bull — SPX conditions (all must be true)
    spx_above_50ma: bool = True
    spx_above_200ma: bool = True
    breadth_50ma_bull: float = 0.60      # SPX breadth threshold
    breadth_200ma_bull: float = 0.50     # SPX breadth threshold
    vix_bull_max: float = 20.0

    # Bull — NDX/tech conditions (both must be true for bull_trend)
    # Lower thresholds than SPX — NDX is more concentrated, fewer names
    ndx_breadth_50ma_bull: float = 0.55  # >55% NDX above 50d SMA
    ndx_breadth_200ma_bull: float = 0.45 # >45% NDX above 200d SMA
    qqq_above_50ma: bool = True          # QQQ must be above its own 50MA

    # Crash (any sufficient)
    vix_crash_min: float = 30.0
    breadth_200ma_crash: float = 0.35

    # Distribution
    breadth_50ma_dist: float = 0.45

    # Smoothing
    label_smoothing_days: int = 3

    @classmethod
    def from_yaml(cls, path: str) -> "RegimeConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        t = cfg["thresholds"]
        return cls(
            spx_above_50ma=t["spx_above_50ma"],
            spx_above_200ma=t["spx_above_200ma"],
            breadth_50ma_bull=t["breadth_50ma_bull"],
            breadth_200ma_bull=t["breadth_200ma_bull"],
            vix_bull_max=t["vix_bull_max"],
            ndx_breadth_50ma_bull=t.get("ndx_breadth_50ma_bull", 0.55),
            ndx_breadth_200ma_bull=t.get("ndx_breadth_200ma_bull", 0.45),
            qqq_above_50ma=t.get("qqq_above_50ma", True),
            vix_crash_min=t["vix_crash_min"],
            breadth_200ma_crash=t["breadth_200ma_crash"],
            breadth_50ma_dist=t["breadth_50ma_dist"],
            label_smoothing_days=t["label_smoothing_days"],
        )


def classify_regime(
    spx: pd.Series,
    vix: pd.Series,
    breadth_50: pd.Series,
    breadth_200: pd.Series,
    qqq: Optional[pd.Series] = None,
    ndx_breadth_50: Optional[pd.Series] = None,
    ndx_breadth_200: Optional[pd.Series] = None,
    cfg: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """
    Classify market regime for each trading day.

    Args:
        spx           : SPX close
        vix           : VIX close
        breadth_50    : % SPX members above 50d SMA
        breadth_200   : % SPX members above 200d SMA
        qqq           : QQQ close (optional — enables QQQ MA gate)
        ndx_breadth_50 : % NDX members above 50d SMA (optional)
        ndx_breadth_200: % NDX members above 200d SMA (optional)
        cfg           : RegimeConfig (uses defaults if None)

    Returns DataFrame columns:
        spx, vix, qqq, breadth_50, breadth_200       : SPX inputs
        ndx_breadth_50, ndx_breadth_200              : NDX inputs (if provided)
        spx_50ma, spx_200ma, qqq_50ma                : computed MAs
        spx_above_50, spx_above_200, qqq_above_50    : boolean flags
        raw_regime                                    : unsmoothed label
        regime                                        : smoothed label
        regime_numeric                                : 1/0/-1/-2
        regime_for_trading                            : T+1 shifted — USE THIS
        regime_numeric_for_trading                    : numeric of above
    """
    if cfg is None:
        cfg = RegimeConfig()

    df = pd.DataFrame({
        "spx": spx,
        "vix": vix,
        "breadth_50": breadth_50,
        "breadth_200": breadth_200,
    }).dropna(subset=["spx", "vix"])

    # QQQ — optional
    if qqq is not None:
        df["qqq"] = qqq.reindex(df.index).ffill()
        df["qqq_50ma"] = df["qqq"].rolling(50).mean()
        df["qqq_above_50"] = df["qqq"] > df["qqq_50ma"]
    else:
        df["qqq"] = np.nan
        df["qqq_50ma"] = np.nan
        df["qqq_above_50"] = True  # don't gate if not provided

    # NDX breadth — optional
    if ndx_breadth_50 is not None:
        df["ndx_breadth_50"] = ndx_breadth_50.reindex(df.index).ffill()
        df["ndx_breadth_200"] = ndx_breadth_200.reindex(df.index).ffill()
    else:
        df["ndx_breadth_50"] = np.nan
        df["ndx_breadth_200"] = np.nan

    # SPX MAs
    df["spx_50ma"] = df["spx"].rolling(50).mean()
    df["spx_200ma"] = df["spx"].rolling(200).mean()
    df["spx_above_50"] = df["spx"] > df["spx_50ma"]
    df["spx_above_200"] = df["spx"] > df["spx_200ma"]

    df["raw_regime"] = df.apply(lambda row: _classify_row(row, cfg), axis=1)
    df["regime"] = _smooth_labels(df["raw_regime"], cfg.label_smoothing_days)
    df["regime_numeric"] = df["regime"].map(REGIME_NUMERIC)

    # === CRITICAL: shift for trading — T+1 only ===
    df["regime_for_trading"] = df["regime"].shift(1)
    df["regime_numeric_for_trading"] = df["regime_numeric"].shift(1)

    return df


def _classify_row(row: pd.Series, cfg: RegimeConfig) -> str:
    """
    Single-row classification. Priority: crash > bull > distribution > chop.

    BULL logic (AND gate):
      SPX conditions  : above 50MA, above 200MA, SPX breadth healthy, VIX calm
      NDX/tech gate   : QQQ above 50MA AND NDX breadth healthy (if data available)

    WHY NDX GATE: SPX can be "fine" (energy/financials holding) while tech rolls over.
    Your portfolio is tech-heavy — a bull label when NDX is weak is a false green.
    """
    if pd.isna(row["spx_50ma"]) or pd.isna(row["spx_200ma"]):
        return "chop"

    # ── CRASH: VIX spike + SPX below 200MA + breadth collapsed ──────────────
    if (row["vix"] >= cfg.vix_crash_min and
            not row["spx_above_200"] and
            row["breadth_200"] < cfg.breadth_200ma_crash):
        return "crash"

    # Extreme VIX regardless (2020-style spike: VIX > 40)
    if row["vix"] >= cfg.vix_crash_min + 10:
        return "crash"

    # ── BULL: all gates must pass ─────────────────────────────────────────────
    spx_ok = (
        row["spx_above_50"] and
        row["spx_above_200"] and
        row["breadth_50"] > cfg.breadth_50ma_bull and
        row["breadth_200"] > cfg.breadth_200ma_bull and
        row["vix"] < cfg.vix_bull_max
    )

    # NDX/tech gate — only apply if data is present
    ndx_data_available = not pd.isna(row.get("ndx_breadth_50", np.nan))
    if ndx_data_available:
        tech_ok = (
            bool(row.get("qqq_above_50", True)) and
            row["ndx_breadth_50"] > cfg.ndx_breadth_50ma_bull and
            row["ndx_breadth_200"] > cfg.ndx_breadth_200ma_bull
        )
    else:
        tech_ok = True  # graceful fallback if NDX data missing

    if spx_ok and tech_ok:
        return "bull_trend"

    # SPX bull but tech lagging → chop (not distribution — market isn't broken yet)
    if spx_ok and not tech_ok:
        return "chop"

    # ── DISTRIBUTION: structural breakdown ───────────────────────────────────
    if (not row["spx_above_50"] or
            row["breadth_50"] < cfg.breadth_50ma_dist):
        return "distribution"

    return "chop"


def _smooth_labels(labels: pd.Series, n: int) -> pd.Series:
    """
    Require N consecutive identical labels to flip confirmed regime state.
    Prevents whipsaw on single-day events.

    Analogy: thermostat with hysteresis — doesn't flip until signal is sustained.

    Key distinction:
      confirmed = stable output (what we report)
      candidate = what raw signal is currently showing
      streak    = consecutive days candidate has held
    confirmed only flips when streak >= n.
    """
    if n <= 1:
        return labels.copy()

    smoothed = labels.copy()
    confirmed = labels.iloc[0]   # stable output state
    candidate = labels.iloc[0]   # current raw signal being tracked
    streak = 1

    for i in range(1, len(labels)):
        if labels.iloc[i] == candidate:
            streak += 1
        else:
            candidate = labels.iloc[i]   # new signal — reset streak
            streak = 1

        if streak >= n:
            confirmed = candidate        # flip confirmed state

        smoothed.iloc[i] = confirmed

    return smoothed


def get_current_regime(df: pd.DataFrame) -> dict:
    """
    Returns the latest regime dict for strategy consumption.
    Uses regime_for_trading (already T+1 shifted).
    """
    latest = df.dropna(subset=["regime_for_trading"]).iloc[-1]
    regime = latest["regime_for_trading"]

    out = {
        "as_of_date": str(df.index[-1].date()),
        "valid_for_trading_date": "next trading day",
        "regime": regime,
        "regime_numeric": int(REGIME_NUMERIC[regime]),
        "emoji": REGIME_EMOJI[regime],
        "spx_close": round(float(latest["spx"]), 2),
        "vix_close": round(float(latest["vix"]), 2),
        "spx_above_50ma": bool(latest["spx_above_50"]),
        "spx_above_200ma": bool(latest["spx_above_200"]),
        "spx_breadth_50": round(float(latest["breadth_50"]), 3),
        "spx_breadth_200": round(float(latest["breadth_200"]), 3),
    }

    # NDX fields — include if available
    if "qqq" in latest and not pd.isna(latest.get("qqq", np.nan)):
        out["qqq_close"] = round(float(latest["qqq"]), 2)
        out["qqq_above_50ma"] = bool(latest.get("qqq_above_50", True))
    if "ndx_breadth_50" in latest and not pd.isna(latest.get("ndx_breadth_50", np.nan)):
        out["ndx_breadth_50"] = round(float(latest["ndx_breadth_50"]), 3)
        out["ndx_breadth_200"] = round(float(latest["ndx_breadth_200"]), 3)

    return out


def _build_snapshot(
    df: pd.DataFrame,
    current: dict,
    vix_short: Optional[pd.Series] = None,
    vix_long: Optional[pd.Series] = None,
) -> RegimeSnapshot:
    """Convert the classified df + current regime dict into a RegimeSnapshot."""
    # 12-month return from available history (up to 252 trading days)
    spx_12m_return = None
    lookback = min(252, len(df) - 1)
    if lookback > 50:
        past = float(df["spx"].iloc[-(lookback + 1)])
        now  = float(df["spx"].iloc[-1])
        if past > 0:
            spx_12m_return = (now - past) / past

    # Near ATH — within 5% of all-time high in the dataset
    spx_near_ath = None
    if len(df) > 10:
        ath = float(df["spx"].max())
        spx_near_ath = float(df["spx"].iloc[-1]) >= ath * 0.95

    # Breadth stored as 0–1 in df/current; RegimeSnapshot expects 0–100
    def _pct(key: str) -> Optional[float]:
        v = current.get(key)
        return v * 100 if v is not None else None

    # Latest scalar from a Series; None if series absent or last value NaN
    def _latest(s: Optional[pd.Series]) -> Optional[float]:
        if s is None:
            return None
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else None

    return RegimeSnapshot(
        data_date        = current["as_of_date"],
        regime           = current["regime"],
        spx              = current["spx_close"],
        qqq              = current.get("qqq_close"),
        vix              = current["vix_close"],
        vix_short        = _latest(vix_short),
        vix_long         = _latest(vix_long),
        spx_above_50ma   = current["spx_above_50ma"],
        spx_above_200ma  = current["spx_above_200ma"],
        qqq_above_50ma   = current.get("qqq_above_50ma"),
        spx_breadth_50d  = _pct("spx_breadth_50"),
        spx_breadth_200d = _pct("spx_breadth_200"),
        ndx_breadth_50d  = _pct("ndx_breadth_50"),
        ndx_breadth_200d = _pct("ndx_breadth_200"),
        spx_12m_return   = spx_12m_return,
        spx_near_ath     = spx_near_ath,
        spx_high_low     = None,
        qqq_ret          = None,
        qqqe_ret         = None,
        economic_regime  = None,
        sentiment_regime = None,
    )


def write_outputs(
    df: pd.DataFrame,
    cfg_dict: dict,
    vix_short: Optional[pd.Series] = None,
    vix_long: Optional[pd.Series] = None,
) -> None:
    """Write CSV labels, latest JSON, and print enriched dashboard to stdout."""
    out = Path(cfg_dict["output"]["label_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)

    # Full history CSV
    df.to_csv(out)
    logger.info(f"Labels written to {out}")

    # Latest JSON — consumed by other strategies
    current = get_current_regime(df)
    json_path = Path(cfg_dict["output"]["latest_json"])
    json_path.write_text(json.dumps(current, indent=2))
    logger.info(f"Latest regime written to {json_path}")

    # Enriched console dashboard
    snapshot = _build_snapshot(df, current, vix_short=vix_short, vix_long=vix_long)
    render_dashboard(snapshot)
