"""
tests/test_classifier.py
========================
Unit tests for regime classifier.
Run: pytest tests/ -v
"""

import pandas as pd
import numpy as np
import pytest
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from regime.classifier import (
    RegimeConfig, classify_regime, _smooth_labels, get_current_regime
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_data(n=300, spx_trend="up", vix_level=15.0, breadth_50=0.75, breadth_200=0.65):
    """Helper to generate synthetic market data."""
    idx = pd.bdate_range("2021-01-01", periods=n)

    if spx_trend == "up":
        spx_vals = np.linspace(3700, 4800, n)
    elif spx_trend == "down":
        spx_vals = np.linspace(4000, 2600, n)
    elif spx_trend == "flat":
        spx_vals = np.full(n, 4000.0)
    else:
        spx_vals = np.array(spx_trend)

    spx = pd.Series(spx_vals, index=idx)
    vix = pd.Series(np.full(n, vix_level), index=idx)
    b50 = pd.Series(np.full(n, breadth_50), index=idx)
    b200 = pd.Series(np.full(n, breadth_200), index=idx)
    return spx, vix, b50, b200


# ── Tests: Classification ─────────────────────────────────────────────────────

def test_clean_bull_market():
    """Strong uptrend + low VIX + broad breadth → mostly bull_trend after warmup.
    No NDX data passed → falls back to SPX-only gate."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up", vix_level=14, breadth_50=0.72, breadth_200=0.62)
    df = classify_regime(spx, vix, b50, b200)  # no NDX — graceful fallback
    post_warmup = df["regime"].iloc[210:]
    bull_pct = (post_warmup == "bull_trend").mean()
    assert bull_pct > 0.85, f"Expected >85% bull, got {bull_pct:.1%}"


def test_crash_market():
    """SPX down 35% + VIX=45 + breadth collapsed → mostly crash/distribution."""
    n = 300
    # SPX starts flat then crashes hard (below 200MA by end)
    spx_vals = np.concatenate([np.linspace(4000, 4000, 50), np.linspace(4000, 2600, 250)])
    vix_vals = np.concatenate([np.full(50, 18.0), np.linspace(18, 48, 250)])
    b50_vals = np.concatenate([np.full(50, 0.70), np.linspace(0.70, 0.12, 250)])
    b200_vals = np.concatenate([np.full(50, 0.60), np.linspace(0.60, 0.10, 250)])
    idx = pd.bdate_range("2021-01-01", periods=n)

    df = classify_regime(
        pd.Series(spx_vals, index=idx),
        pd.Series(vix_vals, index=idx),
        pd.Series(b50_vals, index=idx),
        pd.Series(b200_vals, index=idx),
    )
    # Check late period — by day 250+ SPX is ~2800, 200MA is ~3500 → clearly below
    crash_period = df["regime"].iloc[260:]
    bad_pct = crash_period.isin(["crash", "distribution"]).mean()
    assert bad_pct > 0.75, f"Expected >75% crash/dist in deep crash, got {bad_pct:.1%}"


def test_chop_is_middle_ground():
    """SPX flat + moderate breadth + moderate VIX → chop (not bull, not crash)."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="flat", vix_level=22,
                                     breadth_50=0.52, breadth_200=0.52)
    df = classify_regime(spx, vix, b50, b200)
    post_warmup = df["regime"].iloc[210:]
    assert (post_warmup == "bull_trend").mean() < 0.30
    assert (post_warmup == "crash").mean() < 0.20


# ── Tests: Timestamp safety ───────────────────────────────────────────────────

def test_regime_for_trading_is_t_plus_1():
    """regime_for_trading[i] must equal regime[i-1] — no look-ahead."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up")
    df = classify_regime(spx, vix, b50, b200)
    clean = df.dropna(subset=["regime", "regime_for_trading"])

    for i in range(1, min(20, len(clean))):
        idx_today = clean.index[i]
        idx_yesterday = clean.index[i - 1]
        assert clean.loc[idx_today, "regime_for_trading"] == clean.loc[idx_yesterday, "regime"], \
            f"Look-ahead violation at {idx_today}"


def test_first_row_regime_for_trading_is_nan():
    """First row of regime_for_trading must be NaN (no prior day)."""
    spx, vix, b50, b200 = make_data(n=300)
    df = classify_regime(spx, vix, b50, b200)
    assert pd.isna(df["regime_for_trading"].iloc[0])


# ── Tests: Smoothing ──────────────────────────────────────────────────────────

def test_smoothing_prevents_single_day_flip():
    """A single day anomaly should not flip a 3-day smoothed regime."""
    n = 300
    spx, vix, b50, b200 = make_data(n=n, spx_trend="up", vix_level=15,
                                     breadth_50=0.75, breadth_200=0.65)
    # Inject single-day crash signals on day 250
    vix.iloc[250] = 40.0
    b200.iloc[250] = 0.20

    cfg = RegimeConfig(label_smoothing_days=3)
    df = classify_regime(spx, vix, b50, b200, cfg=cfg)

    # Day 251 should still be bull (not crash) — single day not enough to flip
    regime_251 = df["regime"].iloc[251]
    assert regime_251 != "crash", \
        f"Single-day spike should not flip to crash with smoothing=3, got {regime_251}"


def test_sustained_crash_does_flip():
    """3+ consecutive crash days must flip the smoothed label to crash."""
    n = 300
    # Need SPX below 200MA for crash condition — use flat/declining SPX
    spx_vals = np.concatenate([np.linspace(4000, 4000, 200), np.linspace(4000, 2800, 100)])
    idx = pd.bdate_range("2021-01-01", periods=n)
    spx = pd.Series(spx_vals, index=idx)
    vix = pd.Series(np.concatenate([np.full(200, 15.0), np.full(100, 15.0)]), index=idx)
    b50 = pd.Series(np.concatenate([np.full(200, 0.75), np.full(100, 0.75)]), index=idx)
    b200 = pd.Series(np.concatenate([np.full(200, 0.65), np.full(100, 0.65)]), index=idx)

    # Inject 5 consecutive crash signals at day 250 (SPX well below 200MA by then)
    for i in range(250, 256):
        vix.iloc[i] = 40.0
        b200.iloc[i] = 0.20
        b50.iloc[i] = 0.20

    cfg = RegimeConfig(label_smoothing_days=3)
    df = classify_regime(spx, vix, b50, b200, cfg=cfg)

    # By day 255, should be crash or distribution (SPX below 200MA + VIX=40 + breadth collapsed)
    regime_255 = df["regime"].iloc[255]
    assert regime_255 in ("crash", "distribution"), \
        f"Sustained bad signals should flip to crash/distribution, got {regime_255}"


def test_smooth_labels_basic():
    """_smooth_labels: verify hysteresis logic directly."""
    labels = pd.Series(["bull_trend"] * 5 + ["crash"] + ["bull_trend"] * 5 +
                       ["crash"] * 3 + ["bull_trend"] * 5)
    smoothed = _smooth_labels(labels, n=3)
    # Single "crash" in first block should not flip
    assert smoothed.iloc[5] == "bull_trend"
    # 3 consecutive "crash" should flip
    assert smoothed.iloc[13] == "crash"


# ── Tests: Regime numeric ─────────────────────────────────────────────────────

def test_regime_numeric_mapping():
    """Check numeric mapping is correct and consistent."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up")
    df = classify_regime(spx, vix, b50, b200)
    expected_map = {"bull_trend": 1, "chop": 0, "distribution": -1, "crash": -2}
    for label, num in expected_map.items():
        mask = df["regime"] == label
        if mask.any():
            assert (df.loc[mask, "regime_numeric"] == num).all()


# ── Tests: get_current_regime ─────────────────────────────────────────────────

def test_get_current_regime_structure():
    """Current regime dict must have all required keys."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up")
    df = classify_regime(spx, vix, b50, b200)
    result = get_current_regime(df)
    required_keys = ["as_of_date", "regime", "regime_numeric", "spx_close",
                     "vix_close", "spx_above_50ma", "spx_above_200ma",
                     "spx_breadth_50", "spx_breadth_200"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_get_current_regime_valid_label():
    """Current regime must be one of the four valid labels."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up")
    df = classify_regime(spx, vix, b50, b200)
    result = get_current_regime(df)
    assert result["regime"] in ["bull_trend", "chop", "distribution", "crash"]


# ── Tests: NDX / QQQ gate ─────────────────────────────────────────────────────

def test_bull_with_healthy_ndx():
    """SPX bull + healthy NDX breadth → bull_trend."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up", vix_level=14,
                                     breadth_50=0.72, breadth_200=0.62)
    n = 300
    idx = spx.index
    # QQQ trending up with SPX
    qqq = pd.Series(np.linspace(300, 420, n), index=idx)
    ndx_b50 = pd.Series(np.full(n, 0.68), index=idx)   # well above 0.55 threshold
    ndx_b200 = pd.Series(np.full(n, 0.58), index=idx)  # well above 0.45 threshold

    df = classify_regime(spx, vix, b50, b200, qqq=qqq,
                         ndx_breadth_50=ndx_b50, ndx_breadth_200=ndx_b200)
    post_warmup = df["regime"].iloc[210:]
    bull_pct = (post_warmup == "bull_trend").mean()
    assert bull_pct > 0.85, f"Expected >85% bull with healthy NDX, got {bull_pct:.1%}"


def test_spx_bull_but_ndx_weak_gives_chop():
    """
    2022-Q1 scenario: SPX still near highs but NDX already rolling over.
    Should produce chop, NOT bull_trend.
    This is the key test for tech-heavy portfolios.
    """
    n = 300
    idx = pd.bdate_range("2021-01-01", periods=n)

    # SPX looks fine — still above both MAs
    spx = pd.Series(np.linspace(4200, 4500, n), index=idx)
    vix = pd.Series(np.full(n, 17.0), index=idx)
    b50 = pd.Series(np.full(n, 0.65), index=idx)   # SPX breadth still ok
    b200 = pd.Series(np.full(n, 0.55), index=idx)

    # QQQ rolling over — below 50MA in second half
    qqq_vals = np.concatenate([np.linspace(380, 380, 150),
                                np.linspace(380, 330, 150)])  # drops hard
    qqq = pd.Series(qqq_vals, index=idx)

    # NDX breadth deteriorating
    ndx_b50 = pd.Series(np.concatenate([np.full(150, 0.65),
                                          np.linspace(0.65, 0.35, 150)]), index=idx)
    ndx_b200 = pd.Series(np.concatenate([np.full(150, 0.58),
                                           np.linspace(0.58, 0.30, 150)]), index=idx)

    df = classify_regime(spx, vix, b50, b200, qqq=qqq,
                         ndx_breadth_50=ndx_b50, ndx_breadth_200=ndx_b200)

    # In the second half (NDX broken), should NOT be bull_trend
    second_half = df["regime"].iloc[200:]
    bull_pct = (second_half == "bull_trend").mean()
    assert bull_pct < 0.20, \
        f"SPX bull + weak NDX should NOT produce bull labels, got {bull_pct:.1%}"


def test_ndx_data_in_output_columns():
    """classify_regime output must contain NDX columns when data is passed."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up")
    idx = spx.index
    qqq = pd.Series(np.linspace(300, 420, 300), index=idx)
    ndx_b50 = pd.Series(np.full(300, 0.65), index=idx)
    ndx_b200 = pd.Series(np.full(300, 0.55), index=idx)

    df = classify_regime(spx, vix, b50, b200, qqq=qqq,
                         ndx_breadth_50=ndx_b50, ndx_breadth_200=ndx_b200)

    for col in ["qqq", "qqq_50ma", "qqq_above_50", "ndx_breadth_50", "ndx_breadth_200"]:
        assert col in df.columns, f"Missing expected column: {col}"


def test_ndx_missing_falls_back_gracefully():
    """When no NDX data passed, classifier should not crash and still label regimes."""
    spx, vix, b50, b200 = make_data(n=300, spx_trend="up", vix_level=14,
                                     breadth_50=0.72, breadth_200=0.62)
    # No NDX args — should use SPX-only logic
    df = classify_regime(spx, vix, b50, b200)
    assert df["regime"].iloc[210:].notna().all()
    assert (df["regime"].iloc[210:] == "bull_trend").mean() > 0.80
