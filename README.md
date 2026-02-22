# regimecheck

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-live-brightgreen)

> Daily market regime classifier built for tech-heavy portfolios. Runs on cron at 8am Mon–Fri, classifies the market into one of four states, and publishes a structured JSON that downstream strategies consume as an entry gate.

---

## What it does

- Fetches **SPX, QQQ, VIX + breadth data** daily via `yfinance`
- Applies a **dual-gate classifier** (SPX + NDX) tuned for tech-heavy exposure
- Classifies into one of four regimes:

| Label | Signal |
|-------|--------|
| 🟢 `bull_trend` | Trend up, conditions favorable — full exposure |
| 🟡 `chop` | No clear direction — stay selective, smaller size |
| 🟠 `distribution` | Topping behavior — reduce exposure |
| 🔴 `crash` | Risk-off — defensive posture |

- Optionally calls **Claude Haiku API** to generate a plain-English trading narrative
- Writes structured output files for downstream strategies to consume

---

## Output files

```
output/
├── regime_latest.json           # Current regime + all metrics (consumed by other strategies)
├── regime_labels.csv            # Full regime history
├── regime_dashboard.html        # Plotly visualization
├── narrative_YYYY-MM-DD.txt     # AI-generated trading brief (requires --narrative flag)
└── cron.log                     # Cron run history
```

`regime_latest.json` schema:

```json
{
  "regime": "bull_trend",
  "spx_above_50ma": true,
  "ndx_above_50ma": true,
  "vix": 16.2,
  "timestamp": "2026-02-22T08:01:33"
}
```

---

## Usage

```bash
# Standard run — fetches data, classifies, generates chart
python run_regime.py

# Full run with AI narrative via Claude Haiku
python run_regime.py --narrative

# Dry-run — validates data pipeline only, no chart, no API call
python run_regime.py --validate

# Skip chart generation
python run_regime.py --no-chart
```

If you have the [`quant` CLI](https://github.com/StokedDude/quant) installed:

```bash
quant regime                # standard run
quant regime --narrative    # run + AI brief
quant regime --validate     # dry-run data check
quant regime --show         # print today's saved narrative
```

---

## Setup

```bash
git clone https://github.com/StokedDude/regimecheck.git
cd regimecheck
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment variable (required for `--narrative` only):**

```bash
# Add to ~/.zshrc or ~/.zshenv
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Requirements

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.26
plotly>=5.20
anthropic>=0.26
rich>=13.0
```

---

## Cron setup

Runs automatically at 8am Mon–Fri:

```bash
crontab -e
```

```
0 8 * * 1-5 cd ~/projects/regimecheck && ./venv/bin/python run_regime.py >> output/cron.log 2>&1
```

---

## Integration — part of the `quant` stack

`regimecheck` is the **regime engine** that anchors everything else. Other strategies read `regime_latest.json` as an entry gate before executing.

```
quant/                ← CLI wrapper — single entry point for all strategies
regimecheck/          ← This repo — regime engine, runs on cron
canslim-booster/      ← Stock screener, reads regime_latest.json as gate
family-office-os/     ← 10-factor Qlib pipeline (in development)
alert-bridge/         ← Real-time watchlist scanner (planned)
```

---

## License

MIT
