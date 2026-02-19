"""
regime/chart.py
===============
Generates interactive Plotly HTML dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

REGIME_COLORS = {
    "bull_trend"  : "rgba(0,200,100,0.13)",
    "chop"        : "rgba(255,200,0,0.15)",
    "distribution": "rgba(255,140,0,0.20)",
    "crash"       : "rgba(220,50,50,0.25)",
}


def build_dashboard(df, output_path="output/regime_dashboard.html",
                    title="Market Regime Dashboard"):

    plot = df.dropna(subset=["spx_50ma", "regime"])

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.32, 0.17, 0.17, 0.20, 0.14],
        vertical_spacing=0.025,
        subplot_titles=[
            "SPX + Moving Averages",
            "QQQ + 50d SMA",
            "VIX",
            "Breadth — % Above MA (SPX solid, NDX dashed)",
            "Regime",
        ],
    )

    # Panel 1: SPX
    fig.add_trace(go.Scatter(x=plot.index, y=plot["spx"],
        name="SPX", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot.index, y=plot["spx_50ma"],
        name="SPX 50MA", line=dict(color="#ff7f0e", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot.index, y=plot["spx_200ma"],
        name="SPX 200MA", line=dict(color="#d62728", width=1, dash="dash")), row=1, col=1)

    # Panel 2: QQQ
    if "qqq" in plot.columns and plot["qqq"].notna().any():
        fig.add_trace(go.Scatter(x=plot.index, y=plot["qqq"],
            name="QQQ", line=dict(color="#17becf", width=1.5)), row=2, col=1)
    if "qqq_50ma" in plot.columns and plot["qqq_50ma"].notna().any():
        fig.add_trace(go.Scatter(x=plot.index, y=plot["qqq_50ma"],
            name="QQQ 50MA", line=dict(color="#ff7f0e", width=1, dash="dot")), row=2, col=1)

    # Panel 3: VIX
    fig.add_trace(go.Scatter(x=plot.index, y=plot["vix"],
        name="VIX", fill="tozeroy",
        fillcolor="rgba(148,103,189,0.10)",
        line=dict(color="#9467bd", width=1.5)), row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#00c864",
                  annotation_text="Bull max", annotation_position="right",
                  row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#dc3232",
                  annotation_text="Crash min", annotation_position="right",
                  row=3, col=1)

    # Panel 4: Breadth
    if "spx_b50" in plot.columns:
        fig.add_trace(go.Scatter(x=plot.index, y=plot["spx_b50"]*100,
            name="SPX % >50MA", line=dict(color="#ff7f0e", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["spx_b200"]*100,
            name="SPX % >200MA", line=dict(color="#d62728", width=1.5)), row=4, col=1)
    if "ndx_b50" in plot.columns:
        fig.add_trace(go.Scatter(x=plot.index, y=plot["ndx_b50"]*100,
            name="NDX % >50MA",
            line=dict(color="#ff7f0e", width=1.5, dash="dash")), row=4, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["ndx_b200"]*100,
            name="NDX % >200MA",
            line=dict(color="#d62728", width=1.5, dash="dash")), row=4, col=1)
    for lvl, lbl in [(60,"Bull 50d"),(45,"Dist trigger"),(35,"Crash trigger")]:
        fig.add_hline(y=lvl, line_dash="dot",
                      annotation_text=lbl, annotation_position="right",
                      row=4, col=1)

    # Panel 5: Regime
    num_map = {"bull_trend":1,"chop":0,"distribution":-1,"crash":-2}
    plot2 = plot.copy()
    plot2["rn"] = plot2["regime"].map(num_map)
    fig.add_trace(go.Scatter(
        x=plot2.index, y=plot2["rn"], name="Regime",
        mode="lines", fill="tozeroy",
        fillcolor="rgba(100,100,100,0.08)",
        line=dict(color="#333", width=2),
        text=plot2["regime"],
        hovertemplate="<b>%{text}</b><extra></extra>",
    ), row=5, col=1)
    fig.update_yaxes(
        tickvals=[-2,-1,0,1],
        ticktext=["🔴 Crash","🟠 Dist","🟡 Chop","🟢 Bull"],
        row=5, col=1
    )

    # Regime shading across all panels
    _add_shading(fig, plot, n_rows=5)

    fig.update_layout(
        title=dict(text=title, font_size=16),
        height=1000, template="plotly_white",
        hovermode="x unified",
        margin=dict(r=140, t=60),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(out), include_plotlyjs="cdn", full_html=True)
    print(f"Dashboard saved → {out.resolve()}")


def _add_shading(fig, df, n_rows=5):
    col = df["regime"].fillna("chop")
    changes = col.ne(col.shift()).cumsum()
    for _, blk in df.groupby(changes):
        color = REGIME_COLORS.get(blk["regime"].iloc[0],
                                   "rgba(200,200,200,0.10)")
        for row in range(1, n_rows + 1):
            fig.add_vrect(
                x0=blk.index[0], x1=blk.index[-1],
                fillcolor=color, layer="below", line_width=0,
                row=row, col=1,
            )
