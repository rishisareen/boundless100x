"""The Plotly figures a dashboard embeds, and nothing else.

Seven builders and the one function that decides which of them to run. Each
takes the data it charts and returns an HTML fragment; none of them touched
`self` even while they were methods, which is what made the extraction
mechanical rather than a judgement call — they were module functions wearing
method clothing, and `ReportGenerator` was their namespace rather than their
owner.

**A builder that cannot draw returns `""`, and that is a contract.** Missing
columns, an empty frame, a metric that did not compute: every one of them comes
back as the empty string rather than an exception or a half-drawn figure, so a
data gap costs one panel and never the report. The template pairs that with
`{% if chart %}`, so an empty fragment removes its container and its card
instead of rendering a blank box — which is also what makes the golden
comparison notice when a chart stops rendering at all, since the container
count falls. See `tests/test_report_lane_status.py`.

Kept out of `report_generator.py` because a Plotly figure is a different kind
of thing from a report section: four hundred lines of trace assembly, axis
configuration and colour choices, none of which the sections that surround them
need to read.
"""

import logging

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)


def _drawn(name: str, builder, *args) -> str:
    """One builder's fragment, or `""` when it could not draw at all.

    The contract above — a data gap costs one panel and never the report — is
    honoured by every builder for the gaps it *anticipates*: a missing column,
    an empty frame, a metric that did not compute. This extends the same
    guarantee to the ones none of them anticipated, because the failure it was
    written for is precisely the unanticipated kind:
    `pe_band_chart_historical` reached `interpolate(method="time")` with a
    `NaT` in its index and pandas raised `NotImplementedError` from three
    frames down, which took out a CAPLIPOINT report whose JSON exports, copied
    annual reports and *paid two-pass LLM analysis* were already on disk. One
    panel was worth losing. The document was not.

    Logged at `exception` rather than swallowed: a chart that silently stops
    drawing is a defect that hides, and the golden comparison only notices the
    missing container if someone reads the count. This buys the report a
    degraded panel and a stack trace — it does not make a broken builder
    correct.
    """
    try:
        return builder(*args)
    except Exception:
        logger.exception("Chart %r failed to render — omitting that panel", name)
        return ""


def render_charts(result) -> dict:
    charts = {}

    # SQGLP radar removed — element scores table is sufficient

    ratios = result.data.get("ratios")
    if ratios is not None and not ratios.empty:
        charts["roce_trend"] = _drawn("roce_trend", roce_trend_chart, ratios)

    price = result.data.get("price")
    metrics = result.metrics
    if price is not None and not price.empty:
        charts["pe_band"] = _drawn("pe_band", pe_band_chart, price, metrics)

    charts["growth"] = _drawn("growth", growth_chart, metrics)

    # Shareholding: uses HTML table now, no chart needed

    # Feature 4: DCF gauge
    dcf_chart = _drawn("dcf_gauge", dcf_visualization, result)
    if dcf_chart:
        charts["dcf_gauge"] = dcf_chart

    # Feature 5: Cash flow quality
    cf_chart = _drawn("cashflow_quality", cashflow_quality_chart, result)
    if cf_chart:
        charts["cashflow_quality"] = cf_chart

    # Feature 7: Historical PE band
    pe_hist = _drawn("pe_band_historical", pe_band_chart_historical, result)
    if pe_hist:
        charts["pe_band_historical"] = pe_hist

    return charts


def roce_trend_chart(ratios) -> str:
    if "roce" not in ratios.columns or "year" not in ratios.columns:
        return ""

    df = ratios[~ratios["year"].astype(str).str.contains("TTM", case=False, na=False)].copy()
    df["roce_num"] = pd.to_numeric(df["roce"], errors="coerce")
    df = df.dropna(subset=["roce_num"])

    if df.empty:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"].astype(str),
        y=df["roce_num"],
        mode="lines+markers",
        name="RoCE %",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=8),
    ))
    # Add 15% threshold line
    fig.add_hline(y=15, line_dash="dash", line_color="#dc2626",
                  annotation_text="15% threshold")
    fig.update_layout(
        title="RoCE Trend (10yr)",
        yaxis_title="RoCE %",
        margin=dict(l=50, r=30, t=50, b=30),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def pe_band_chart(price, metrics: dict) -> str:
    pe_result = metrics.get("pe_vs_historical")
    if not pe_result or not pe_result.ok:
        return ""

    meta = pe_result.metadata or {}
    percentile = pe_result.value
    pe_current = metrics.get("pe_ttm")
    if not pe_current or not pe_current.ok:
        return ""

    # Simple PE gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pe_current.value,
        title={"text": f"PE TTM ({percentile:.0f}th percentile)"},
        gauge=dict(
            axis=dict(range=[0, min(pe_current.value * 2, 150)]),
            bar=dict(color="#2563eb"),
            steps=[
                dict(range=[0, meta.get("p25", 30)], color="#dcfce7"),
                dict(range=[meta.get("p25", 30), meta.get("p75", 60)], color="#fef9c3"),
                dict(range=[meta.get("p75", 60), min(pe_current.value * 2, 150)], color="#fecaca"),
            ],
            threshold=dict(
                line=dict(color="#dc2626", width=2),
                value=meta.get("median", 45),
                thickness=0.75,
            ),
        ),
    ))
    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def growth_chart(metrics: dict) -> str:
    labels = []
    values = []

    growth_metrics = [
        ("revenue_cagr_5yr", "Rev CAGR 5yr"),
        ("pat_cagr_5yr", "PAT CAGR 5yr"),
        ("pat_cagr_3yr", "PAT CAGR 3yr"),
        ("eps_cagr_5yr", "EPS CAGR 5yr"),
    ]

    for mid, label in growth_metrics:
        result = metrics.get(mid)
        if result and result.ok and isinstance(result.value, (int, float)):
            labels.append(label)
            values.append(result.value)

    if not labels:
        return ""

    colors = ["#2563eb" if v >= 15 else "#f59e0b" if v >= 0 else "#dc2626" for v in values]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=15, line_dash="dash", line_color="#16a34a",
                  annotation_text="15% compounder threshold")
    fig.update_layout(
        title="Growth Metrics",
        yaxis_title="CAGR %",
        margin=dict(l=50, r=30, t=50, b=30),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)

# ── Growth Decomposition ──


def dcf_visualization(result) -> str:
    """Create a Plotly gauge chart for DCF margin of safety."""
    dcf = result.metrics.get("dcf_margin_of_safety")
    if not dcf or not dcf.ok or dcf.value is None:
        return ""

    margin_pct = dcf.value
    # Clamp display range
    display_val = max(-50, min(50, margin_pct))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        number={"suffix": "%"},
        title={"text": "DCF Margin of Safety"},
        gauge=dict(
            axis=dict(range=[-50, 50], tickvals=[-50, -25, 0, 25, 50]),
            bar=dict(color="#2563eb" if margin_pct >= 0 else "#dc2626"),
            steps=[
                dict(range=[-50, -10], color="#fecaca"),
                dict(range=[-10, 0], color="#fef9c3"),
                dict(range=[0, 20], color="#dcfce7"),
                dict(range=[20, 50], color="#bbf7d0"),
            ],
            threshold=dict(
                line=dict(color="#16a34a", width=2),
                value=0,
                thickness=0.75,
            ),
        ),
    ))
    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def cashflow_quality_chart(result) -> str:
    """Create a dual-line Plotly chart: CFO vs PAT over 10 years."""
    financials = result.data.get("financials")
    cashflow = result.data.get("cashflow")
    if financials is None or cashflow is None:
        return ""

    def _annual(df):
        if df is None or df.empty or "year" not in df.columns:
            return pd.DataFrame()
        mask = df["year"].astype(str).str.startswith("Mar", na=False)
        return df[mask].copy()

    df_fin = _annual(financials)
    df_cf = _annual(cashflow)

    if df_fin.empty or df_cf.empty or "pat" not in df_fin.columns or "cfo" not in df_cf.columns:
        return ""

    # Merge on year
    merged = pd.merge(
        df_fin[["year", "pat"]],
        df_cf[["year", "cfo"]],
        on="year", how="inner",
    )
    merged["pat_num"] = pd.to_numeric(merged["pat"], errors="coerce")
    merged["cfo_num"] = pd.to_numeric(merged["cfo"], errors="coerce")
    merged = merged.dropna(subset=["pat_num", "cfo_num"])

    if len(merged) < 3:
        return ""

    years = merged["year"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=merged["cfo_num"], name="CFO",
        mode="lines+markers",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=merged["pat_num"], name="PAT",
        mode="lines+markers",
        line=dict(color="#16a34a", width=2),
        marker=dict(size=6),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title="CFO vs PAT (Cash Flow Quality)",
        yaxis_title="INR Crores",
        margin=dict(l=50, r=30, t=50, b=30),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def pe_band_chart_historical(result) -> str:
    """Create a price chart with PE-band lines using interpolated annual EPS.

    Shows stock price overlaid with coloured PE-band lines (EPS × N).
    EPS is linearly interpolated between fiscal year-ends for smooth bands.
    """
    price_df = result.data.get("price")
    financials = result.data.get("financials")
    if price_df is None or price_df.empty or financials is None or financials.empty:
        return ""

    # --- Extract annual EPS ---
    def _annual(df):
        if "year" not in df.columns:
            return pd.DataFrame()
        mask = df["year"].astype(str).str.startswith("Mar", na=False)
        return df[mask].copy()

    df_fin = _annual(financials)
    if df_fin.empty or "eps" not in df_fin.columns:
        return ""

    df_fin["eps_num"] = pd.to_numeric(df_fin["eps"], errors="coerce")
    df_fin = df_fin.dropna(subset=["eps_num"])
    df_fin = df_fin[df_fin["eps_num"] > 0]

    if len(df_fin) < 3:
        return ""

    # Parse fiscal year end dates (e.g., "Mar 2023" → 2023-03-31)
    #
    # `errors="coerce"` makes an unparseable label `NaT`, and a `NaT` reaching
    # the index is not a cosmetic gap: `interpolate(method="time")` below raises
    # `NotImplementedError` on it. That took the whole HTML report down *after*
    # the JSON exports and a paid LLM run had already been written to disk.
    #
    # The label that gets this far is Screener's **fiscal-year-transition
    # stub**. A company that moves its year end renders a part-year column —
    # Caplin Point, which went from a June to a March year end, has
    # `Mar 20169m` for the 9-month stub — and that starts with "Mar", so the
    # `_annual` filter above admits it where it correctly drops `TTM`. Matching
    # a prefix is not the same as parsing a date, which is the whole lesson.
    #
    # Dropping the row is the right *reading*, not just the safe one: a stub
    # period has no well-defined year end to interpolate between, and its EPS
    # covers nine months of earnings beside neighbours covering twelve.
    parsed = pd.to_datetime(df_fin["year"], format="%b %Y", errors="coerce")
    df_fin = df_fin[parsed.notna()]
    eps_dates = parsed.dropna() + pd.offsets.MonthEnd(0)

    # Re-checked after the drop, not just before it: the count that matters is
    # how many points are left to interpolate between.
    if len(df_fin) < 3:
        return ""

    eps_series = pd.Series(df_fin["eps_num"].values, index=eps_dates).sort_index()

    # --- Standardize price data ---
    if isinstance(price_df.index, pd.DatetimeIndex):
        prices = price_df.copy()
    elif "date" in price_df.columns:
        prices = price_df.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices.set_index("date")
    elif "Date" in price_df.columns:
        prices = price_df.copy()
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
        prices = prices.set_index("Date")
    else:
        return ""

    close_col = None
    for col in ["Close", "close", "Adj Close"]:
        if col in prices.columns:
            close_col = col
            break
    if close_col is None:
        return ""

    prices = prices[[close_col]].dropna()
    prices.columns = ["close"]
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna().sort_index()

    if len(prices) < 100:
        return ""

    # --- Interpolate EPS to daily (smooth, not step-function) ---
    # Reindex to daily price dates, then interpolate between annual values
    daily_eps = eps_series.reindex(
        eps_series.index.union(prices.index)
    ).interpolate(method="time")
    daily_eps = daily_eps.reindex(prices.index).dropna()

    if len(daily_eps) < 100:
        return ""

    aligned = prices.loc[daily_eps.index].copy()
    aligned["eps"] = daily_eps.values

    # --- Determine PE band multiples ---
    # Use actual trailing PE to pick sensible bands
    aligned["pe"] = aligned["close"] / aligned["eps"]
    pe_clipped = aligned["pe"].clip(upper=100)
    pe_median = float(pe_clipped.median())

    candidate_bands = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    price_max = float(aligned["close"].max())
    eps_max = float(aligned["eps"].max())

    # Select bands whose MAX band price (across all years) stays within 1.5× price max.
    # This ensures bands visually bracket the price, not dwarf it.
    selected_bands = [
        n for n in candidate_bands
        if eps_max * n <= price_max * 1.8
    ]
    # Ensure at least one band above the price too
    if not selected_bands:
        selected_bands = [n for n in candidate_bands if n <= pe_median * 2]
    if not selected_bands:
        selected_bands = [10, 15, 20]
    if len(selected_bands) > 5:
        step = max(1, len(selected_bands) // 5)
        selected_bands = selected_bands[::step][:5]

    # --- Build chart ---
    band_colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]

    # Convert to plain lists for reliable Plotly serialization
    x_dates = aligned.index.strftime("%Y-%m-%d").tolist()
    y_close = aligned["close"].tolist()

    fig = go.Figure()

    # PE band lines (drawn first so price is on top)
    for i, n in enumerate(selected_bands):
        band_vals = (aligned["eps"] * n).tolist()
        color = band_colors[i % len(band_colors)]
        fig.add_trace(go.Scatter(
            x=x_dates, y=band_vals,
            name=f"{n}x PE", mode="lines",
            line=dict(width=1.5, color=color, dash="dot"),
        ))

    # Price line on top (solid, prominent)
    fig.add_trace(go.Scatter(
        x=x_dates, y=y_close,
        name="Price", mode="lines",
        line=dict(color="#2563eb", width=2.5),
    ))

    # Current PE annotation
    current_pe = float(aligned["pe"].iloc[-1])
    fig.add_annotation(
        x=x_dates[-1], y=y_close[-1],
        text=f"PE: {current_pe:.1f}x",
        showarrow=True, arrowhead=2, arrowsize=1,
        ax=40, ay=-30,
        font=dict(size=11, color="#2563eb"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#2563eb",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(text="Historical PE Band Chart", font=dict(size=16)),
        yaxis_title="Price (₹)",
        margin=dict(l=50, r=30, t=50, b=50),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, font_size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)
