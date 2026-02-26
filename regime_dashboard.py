"""
regime_dashboard.py
===================
Extended regime dashboard with trend-cycle phase, breadth labels,
VIX term structure, divergence detection, and risk-control hints.

Standalone — no dependency on regime/classifier.py.
Integrate by calling render_dashboard(snapshot) from write_outputs().
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RegimeSnapshot:
    # ── Core prices ──────────────────────────────────────────────────────────
    data_date: str = ""
    regime: str = ""                        # from classifier (bull_trend / chop / distribution / crash)
    spx: Optional[float] = None
    qqq: Optional[float] = None
    vix: Optional[float] = None

    # ── VIX term structure ───────────────────────────────────────────────────
    vix_short: Optional[float] = None       # e.g. 9-day or front-month futures
    vix_long: Optional[float] = None        # e.g. 30-day or second-month futures

    # ── Moving average flags ─────────────────────────────────────────────────
    spx_above_50ma: Optional[bool] = None
    spx_above_200ma: Optional[bool] = None
    qqq_above_50ma: Optional[bool] = None

    # ── Breadth (0–100 scale) ─────────────────────────────────────────────────
    spx_breadth_50d: Optional[float] = None
    spx_breadth_200d: Optional[float] = None
    ndx_breadth_50d: Optional[float] = None
    ndx_breadth_200d: Optional[float] = None

    # ── Trend-cycle inputs ────────────────────────────────────────────────────
    spx_12m_return: Optional[float] = None  # e.g. 0.22 for +22%
    spx_near_ath: Optional[bool] = None     # True if within ~5% of ATH

    # ── Divergence inputs ────────────────────────────────────────────────────
    spx_high_low: Optional[float] = None    # net new 52-week highs minus lows
    qqq_ret: Optional[float] = None         # QQQ cap-weighted return (any period)
    qqqe_ret: Optional[float] = None        # QQQE equal-weighted return (same period)

    # ── Pluggable external context ───────────────────────────────────────────
    economic_regime: Optional[str] = None   # e.g. "late-cycle", "recession-risk"
    sentiment_regime: Optional[str] = None  # e.g. "fear", "neutral", "greed"


# ---------------------------------------------------------------------------
# Breadth labeling
# ---------------------------------------------------------------------------

def label_breadth_level(percent: Optional[float]) -> str:
    if percent is None:
        return "n/a"
    if percent >= 70:
        return "strong"
    if percent >= 40:
        return "neutral"
    return "weak"


# ---------------------------------------------------------------------------
# Divergence: cap-weighted vs equal-weighted
# ---------------------------------------------------------------------------

def detect_divergence(
    main_index_ret: Optional[float],
    equal_weight_ret: Optional[float],
    threshold: float = 0.03,
) -> str:
    """
    Compare cap-weighted vs equal-weighted performance.
    threshold = minimum gap (in return units) to call "narrow leadership".
    """
    if main_index_ret is None or equal_weight_ret is None:
        return "n/a"
    gap = main_index_ret - equal_weight_ret
    if gap > threshold:
        return "narrow leadership"
    if equal_weight_ret >= main_index_ret:
        return "broad participation"
    return "no signal"


# ---------------------------------------------------------------------------
# VIX term structure + volatility regime
# ---------------------------------------------------------------------------

def classify_vix_term_structure(
    vix_short: Optional[float],
    vix_long: Optional[float],
    threshold: float = 0.5,
) -> str:
    """
    contango  : front < back — normal, low-fear environment
    backwardation: front > back — elevated near-term fear
    flat      : within threshold
    """
    if vix_short is None or vix_long is None:
        return "n/a"
    diff = vix_short - vix_long
    if diff < -threshold:
        return "contango"
    if diff > threshold:
        return "backwardation"
    return "flat"


def classify_vol_regime(vix: Optional[float], term_structure: str) -> str:
    """
    calm     : VIX < 15 and market structure healthy (contango)
    normal   : VIX < 25 and no backwardation
    stressed : VIX >= 25 or backwardation
    """
    if vix is None:
        return "n/a"
    if vix >= 25 or term_structure == "backwardation":
        return "stressed"
    if vix < 15 and term_structure == "contango":
        return "calm"
    return "normal"


# ---------------------------------------------------------------------------
# Trend-cycle phase
# ---------------------------------------------------------------------------

def classify_trend_cycle(
    spx_12m_return: Optional[float],
    spx_near_ath: Optional[bool],
    spx_breadth_50d: Optional[float],
    spx_breadth_200d: Optional[float],
) -> tuple[str, str]:
    """
    Returns (phase, cycle_context).

    phase         : Accumulation | Markup | Distribution | Markdown
    cycle_context : early-cycle | mid-cycle | late-cycle

    Heuristic (transparent thresholds):
      strongly_positive : 12m return > +15%
      near_ath          : passed in directly
      breadth_strong    : SPX 50d breadth >= 60% AND 200d >= 60%
      breadth_weak      : SPX 50d breadth <  45% AND 200d <  45%
    """
    if spx_12m_return is None:
        return "n/a", "n/a"

    near_ath        = bool(spx_near_ath) if spx_near_ath is not None else False
    b50             = spx_breadth_50d  or 0.0
    b200            = spx_breadth_200d or 0.0
    strongly_pos    = spx_12m_return > 0.15
    negative        = spx_12m_return < 0.0
    breadth_strong  = b50 >= 60 and b200 >= 60
    breadth_weak    = b50 < 45 and b200 < 45

    if strongly_pos and near_ath and breadth_strong:
        return "Markup", "mid-cycle"

    if strongly_pos and near_ath and not breadth_strong:
        return "Distribution", "late-cycle"

    if negative and breadth_weak:
        return "Markdown", "late-cycle"

    if not strongly_pos and breadth_weak and not near_ath:
        if spx_12m_return >= 0:
            return "Accumulation", "early-cycle"
        return "Markdown", "late-cycle"

    if spx_12m_return >= 0 and not near_ath and not breadth_strong:
        return "Accumulation", "early-cycle"

    return "Markup", "mid-cycle"


# ---------------------------------------------------------------------------
# Risk-control hint
# ---------------------------------------------------------------------------

def compute_risk_control_hint(
    spx_breadth_50d: Optional[float],
    ndx_breadth_50d: Optional[float],
    vix: Optional[float],
) -> str:
    if spx_breadth_50d is None or ndx_breadth_50d is None:
        return "n/a"

    both_very_weak = spx_breadth_50d < 30 and ndx_breadth_50d < 30
    reduce         = spx_breadth_50d < 50 and ndx_breadth_50d < 30

    if both_very_weak and vix is not None and vix > 25:
        return "Defensive posture (both breadths < 30% & VIX > 25)"
    if reduce:
        return "Reduce leverage (SPX 50d breadth < 50% and NDX 50d breadth < 30%)"
    return "OK"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

W = 17  # label column width


def _row(label: str, value: str, comment: str = "") -> str:
    suffix = f"  → {comment}" if comment else ""
    return f"  {label:<{W}}: {value}{suffix}"


def _fmt(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _icon(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "✅" if value else "❌"


def _breadth_row(label: str, pct: Optional[float]) -> str:
    val_str = f"{pct:.1f}%" if pct is not None else "n/a"
    lbl     = label_breadth_level(pct)
    comment = lbl if pct is not None else ""
    return _row(label, val_str, comment)


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

SEP = "=" * 55

REGIME_EMOJI = {
    "bull_trend":   "🟢",
    "chop":         "🟡",
    "distribution": "🟠",
    "crash":        "🔴",
}


def render_dashboard(s: RegimeSnapshot) -> None:
    # Derived labels
    term_str  = classify_vix_term_structure(s.vix_short, s.vix_long)
    vol_lbl   = classify_vol_regime(s.vix, term_str)
    phase, cx = classify_trend_cycle(
        s.spx_12m_return, s.spx_near_ath,
        s.spx_breadth_50d, s.spx_breadth_200d,
    )
    divergence = detect_divergence(s.qqq_ret, s.qqqe_ret)
    risk_hint  = compute_risk_control_hint(
        s.spx_breadth_50d, s.ndx_breadth_50d, s.vix,
    )

    # Formatted sub-strings
    emoji      = REGIME_EMOJI.get(s.regime, "⚪")
    regime_str = f"{emoji} {s.regime.upper()}" if s.regime else "n/a"

    trend_cycle_str = f"{cx} ({phase})" if phase != "n/a" else "n/a"

    vix_term_str = (
        f"{term_str} ({vol_lbl})"
        if term_str != "n/a"
        else (vol_lbl if vol_lbl != "n/a" else "n/a")
    )

    hl_str = _fmt(s.spx_high_low, 0) if s.spx_high_low is not None else "n/a"

    print(SEP)
    print("REGIME DETECTOR — CURRENT STATUS")
    print(SEP)
    print(_row("Data through",   s.data_date or "n/a"))
    print(_row("Regime",         regime_str))
    print(_row("Trend cycle",    trend_cycle_str))
    print(SEP)
    print("  EQUITY")
    print(_row("SPX",            _fmt(s.spx)))
    print(_row("SPX above 50MA", _icon(s.spx_above_50ma)))
    print(_row("SPX above 200MA",_icon(s.spx_above_200ma)))
    print(_row("QQQ",            _fmt(s.qqq)))
    print(_row("QQQ above 50MA", _icon(s.qqq_above_50ma)))
    print(SEP)
    print("  BREADTH")
    print(_breadth_row("SPX breadth 50d",  s.spx_breadth_50d))
    print(_breadth_row("SPX breadth 200d", s.spx_breadth_200d))
    print(_breadth_row("NDX breadth 50d",  s.ndx_breadth_50d))
    print(_breadth_row("NDX breadth 200d", s.ndx_breadth_200d))
    print(_row("New highs/lows", hl_str))
    print(_row("QQQ vs QQQE",    divergence))
    print(SEP)
    print("  VOLATILITY")
    print(_row("VIX",            _fmt(s.vix)))
    print(_row("VIX term",       vix_term_str))
    print(SEP)
    print("  CONTEXT")
    print(_row("Economic regime", s.economic_regime or "n/a"))
    print(_row("Sentiment",       s.sentiment_regime or "n/a"))
    print(SEP)
    print("  RISK")
    print(_row("Risk-control",    risk_hint))
    print(SEP)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    snapshot = RegimeSnapshot(
        data_date        = "2026-02-25",
        regime           = "distribution",
        spx              = 6946.13,
        qqq              = 616.68,
        vix              = 17.93,
        vix_short        = 16.50,   # front-month VIX futures
        vix_long         = 18.20,   # second-month VIX futures
        spx_above_50ma   = True,
        spx_above_200ma  = True,
        qqq_above_50ma   = True,
        spx_breadth_50d  = 57.1,
        spx_breadth_200d = 68.6,
        ndx_breadth_50d  = 34.9,
        ndx_breadth_200d = 39.5,
        spx_12m_return   = 0.22,    # +22% over trailing 12 months
        spx_near_ath     = True,    # SPX within ~5% of ATH
        spx_high_low     = None,    # not available — prints n/a
        qqq_ret          = 0.18,    # QQQ cap-weighted trailing return
        qqqe_ret         = 0.09,    # QQQE equal-weighted trailing return
        economic_regime  = "late-cycle",
        sentiment_regime = "mildly bullish",
    )

    render_dashboard(snapshot)
