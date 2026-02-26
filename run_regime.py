"""
run_regime.py
=============
Main entry point. Run manually or via cron.

Usage:
  python run_regime.py                  # full run + chart
  python run_regime.py --no-chart       # faster, skip chart (use for cron)
  python run_regime.py --validate       # historical validation only
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from regime.data import fetch_all
from regime.classifier import RegimeConfig, classify_regime, write_outputs
from regime.chart import build_dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/regime_run.log", mode="a"),
    ]
)
logger = logging.getLogger("run_regime")


def main():
    parser = argparse.ArgumentParser(description="Regime Detector")
    parser.add_argument("--config", default="config/regime.yaml")
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--narrative", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = RegimeConfig.from_yaml(args.config)
    Path("output").mkdir(exist_ok=True)

    logger.info(f"Starting regime run — {datetime.now().isoformat()}")

    spx, qqq, vix, spx_b50, spx_b200, ndx_b50, ndx_b200, vix_short, vix_long = fetch_all(
        lookback_days=cfg_dict["data"]["lookback_days"],
        spx_ticker=cfg_dict["data"]["spx_ticker"],
        qqq_ticker=cfg_dict["data"].get("qqq_ticker", "QQQ"),
        vix_ticker=cfg_dict["data"]["vix_ticker"],
    )

    df = classify_regime(
        spx, vix, spx_b50, spx_b200,
        qqq=qqq,
        ndx_breadth_50=ndx_b50,
        ndx_breadth_200=ndx_b200,
        cfg=cfg,
    )

    if args.validate:
        _print_validation(df)
        return

    write_outputs(df, cfg_dict, vix_short=vix_short, vix_long=vix_long)

    if not args.no_chart:
        build_dashboard(
            df,
            output_path=cfg_dict["output"]["chart_html"],
            title="Market Regime Dashboard",
        )

    if args.narrative:
        run_narrative(output_dir="output")

    logger.info("Regime run complete.")


def _print_validation(df):
    print("\n=== REGIME DISTRIBUTION ===")
    total = len(df.dropna(subset=["regime"]))
    for regime, count in df["regime"].value_counts().items():
        print(f"  {regime:<16}: {count:>5} days ({count/total:.1%})")

    print("\n=== BEAR PERIODS (expect crash/distribution) ===")
    bears = [
        ("2007-10-01","2009-03-31","GFC"),
        ("2020-02-15","2020-04-15","COVID crash"),
        ("2022-01-01","2022-06-30","2022 NDX bear"),
        ("2022-01-01","2022-12-31","2022 full year"),
    ]
    for start, end, label in bears:
        period = df.loc[start:end, "regime"].dropna()
        if len(period) == 0:
            print(f"  {label}: no data in window")
            continue
        pct = period.isin(["crash","distribution"]).mean()
        print(f"  {label:<20}: {pct:.0%} crash/dist ({len(period)}d)")

    print("\n=== BULL PERIODS (expect bull_trend) ===")
    bulls = [
        ("2020-11-01","2021-12-31","Post-COVID bull"),
        ("2023-01-01","2023-12-31","2023 recovery"),
        ("2024-01-01","2024-10-31","2024 bull"),
    ]
    for start, end, label in bulls:
        period = df.loc[start:end, "regime"].dropna()
        if len(period) == 0:
            print(f"  {label}: no data in window")
            continue
        pct = (period == "bull_trend").mean()
        print(f"  {label:<20}: {pct:.0%} bull_trend ({len(period)}d)")


def run_narrative(output_dir="output"):
    """Load today's regime JSON and generate AI narrative."""
    from regime.narrative import generate_narrative
    import json
    from pathlib import Path as P

    json_path = P(output_dir) / "regime_latest.json"
    if not json_path.exists():
        print("regime_latest.json not found — run without --narrative first")
        return

    regime_data = json.loads(json_path.read_text())
    narrative = generate_narrative(regime_data, output_dir=output_dir)

    if narrative:
        print("\n" + "="*55)
        print("  AI NARRATIVE")
        print("="*55)
        print(narrative)
        print("="*55 + "\n")


if __name__ == "__main__":
    main()


# (duplicate run_narrative removed — definition above at line 122 is used)
