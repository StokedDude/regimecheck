"""
regime/narrative.py
===================
Calls Claude API to generate daily trading narrative from regime data.
Saves to output/narrative_YYYY-MM-DD.txt
"""

import json
import os
import urllib.request
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def generate_narrative(regime_json: dict, output_dir: str = "output") -> str:
    """
    Takes regime_latest.json dict, calls Claude, saves daily narrative.
    Returns the narrative text.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set — skipping narrative")
        return ""

    prompt = _build_prompt(regime_json)
    narrative = _call_claude(prompt, api_key)

    # Save to dated file
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    filename = out / f"narrative_{date.today().isoformat()}.txt"
    filename.write_text(narrative)
    print(f"✅ Narrative saved → {filename}")

    return narrative


def _build_prompt(d: dict) -> str:
    emoji = d.get("emoji", "")
    return f"""You are a quant trading assistant analyzing daily market regime data for a CANSLIM / momentum equity trader with a tech-heavy portfolio.

Today's regime data:
- Date: {d['as_of_date']} (valid for next trading day)
- Regime: {emoji} {d['regime'].upper()}
- SPX: {d['spx_close']} | Above 50MA: {d['spx_above_50ma']} | Above 200MA: {d['spx_above_200ma']}
- QQQ: {d['qqq_close']} | Above 50MA: {d['qqq_above_50ma']}
- VIX: {d['vix_close']}
- SPX Breadth 50d: {d['spx_breadth_50']:.1%} | 200d: {d['spx_breadth_200']:.1%}
- NDX Breadth 50d: {d['ndx_breadth_50']:.1%} | 200d: {d['ndx_breadth_200']:.1%}

Regime thresholds for reference:
- BULL requires: SPX above both MAs, SPX breadth 50d >60%, NDX breadth 50d >55%, QQQ above 50MA, VIX <20
- DISTRIBUTION: SPX below 50MA or SPX breadth 50d <45%
- CRASH: VIX >30 + SPX below 200MA + breadth collapsed

Write a concise daily briefing with exactly three sections:

1. REGIME INTERPRETATION (3-4 sentences)
What the current regime means. Which signals are bullish vs bearish. How close or far we are from a regime change. Be specific about the numbers.

2. KEY LEVELS TO WATCH (bullet points)
What specific conditions would flip the regime to bull or deeper into distribution/crash. Give concrete breadth percentages and price levels where possible.

3. CANSLIM / MOMENTUM ACTION (2-3 sentences)
Specific actionable guidance for a momentum trader today. New entries yes/no, position sizing, what to do with existing positions.

Be direct. No fluff. Write like a quant, not a newsletter."""


def _call_claude(prompt: str, api_key: str) -> str:
    """Calls Claude Haiku via raw urllib — no SDK dependency needed."""
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        error = e.read().decode()
        print(f"❌ Claude API error {e.code}: {error}")
        return ""
    except Exception as e:
        print(f"❌ Narrative generation failed: {e}")
        return ""
