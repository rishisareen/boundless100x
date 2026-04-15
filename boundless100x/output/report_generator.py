"""Report Generator — HTML dashboards, markdown summaries, and JSON exports."""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.growth import compute_lever_decomposition_table

logger = logging.getLogger(__name__)


def _safe_numeric(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        v = float(val)
        if pd.isna(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


# ── Human-readable flag labels ──
# Maps raw flag strings to (display_label, sentiment) where sentiment ∈ {good, bad, neutral}
FLAG_LABELS: dict[str, tuple[str, str]] = {
    # Growth
    "growth_quality_high_quality": ("High-Quality Growth", "good"),
    "growth_quality_moderate": ("Moderate Growth Quality", "neutral"),
    "growth_quality_low_quality": ("Low-Quality Growth", "bad"),
    "very_short_history_unreliable": ("Very Short History — Unreliable", "bad"),
    "bonus_split_adjusted": ("Bonus/Split Adjusted", "neutral"),
    "high_dilution": ("Significant Equity Dilution", "bad"),
    "minimal_dilution": ("Minimal Equity Dilution", "good"),
    # Profitability
    "consistently_high_roce": ("Consistently High RoCE", "good"),
    "exceptional_roce": ("Exceptional RoCE (>25%)", "good"),
    "improving_roce": ("Improving RoCE Trend", "good"),
    "declining_roce": ("Declining RoCE Trend", "bad"),
    "high_operating_margin": ("High Operating Margin", "good"),
    "improving_margins": ("Improving Margins", "good"),
    "cash_cow": ("Cash Cow — Strong Cash Conversion", "good"),
    "cfi_dominated_by_acquisitions": ("Capex Dominated by Acquisitions", "bad"),
    "volatile_tax_rate": ("Volatile Tax Rate", "neutral"),
    # Valuation
    "very_expensive_pe": ("Very Expensive PE (>80x)", "bad"),
    "expensive_pe": ("Expensive PE (>50x)", "bad"),
    "cheap_pe": ("Cheap PE (<15x)", "good"),
    "attractively_valued_peg": ("Attractively Valued (PEG < 1)", "good"),
    "expensive_peg": ("Expensive PEG (>2.5x)", "bad"),
    "attractive_trailing_peg": ("Attractive Trailing PEG", "good"),
    "pe_above_historical_75th": ("PE Above 75th Percentile — Expensive", "bad"),
    "pe_below_historical_25th": ("PE Below 25th Percentile — Cheap", "good"),
    "dcf_undervalued": ("DCF: Undervalued", "good"),
    "dcf_overvalued": ("DCF: Overvalued", "bad"),
    "negative_average_fcf": ("Negative Average Free Cash Flow", "bad"),
    "negative_fcf_even_after_outlier_removal": ("Negative FCF Even After Outlier Removal", "bad"),
    "reverse_dcf_overpriced": ("Market Overpricing Growth (Reverse DCF)", "bad"),
    "reverse_dcf_underpriced": ("Market Underpricing Growth (Reverse DCF)", "good"),
    "earnings_yield_above_gsec": ("Earnings Yield Beats G-Sec", "good"),
    "gsec_more_attractive": ("G-Sec More Attractive Than Earnings Yield", "bad"),
    # Leverage
    "debt_risk": ("High Debt Risk", "bad"),
    "virtually_debt_free": ("Virtually Debt-Free", "good"),
    "low_interest_coverage": ("Weak Interest Coverage", "bad"),
    "strong_interest_coverage": ("Strong Interest Coverage", "good"),
    # Efficiency
    "improving_working_capital": ("Improving Working Capital", "good"),
    "worsening_working_capital": ("Worsening Working Capital", "bad"),
    # Size
    "small_cap": ("Small Cap", "neutral"),
    "mid_cap": ("Mid Cap", "neutral"),
    "large_cap": ("Large Cap", "neutral"),
    "micro_cap": ("Micro Cap", "neutral"),
    "low_institutional_ownership": ("Low Institutional Ownership", "neutral"),
    "heavily_institutional": ("Heavily Institutional", "neutral"),
    "under_researched": ("Under-Researched (<5 Analysts)", "neutral"),
    "lightly_covered": ("Lightly Covered (5–10 Analysts)", "neutral"),
    "promoter_increasing_stake": ("Promoter Increasing Stake", "good"),
    "promoter_reducing_stake": ("Promoter Reducing Stake", "bad"),
    "promoter_pledge_red_flag": ("Promoter Pledge — Red Flag", "bad"),
    # Longevity
    "wide_moat_cap": ("Wide Moat (Market Cap Proxy)", "good"),
    "moderate_moat_cap": ("Moderate Moat (Market Cap Proxy)", "neutral"),
    "highly_stable_margins": ("Highly Stable Margins", "good"),
    "volatile_margins": ("Volatile Margins", "bad"),
    "heavy_reinvestment": ("Heavy Reinvestment", "neutral"),
    "consistent_fcf_generator": ("Consistent Free Cash Flow Generator", "good"),
    "consistent_organic_fcf_generator": ("Consistent Organic FCF (Excl. M&A)", "good"),
    # Composite
    "possible_bonus_split": ("Possible Bonus/Split Event Detected", "neutral"),
}

# ── Metric-to-element mapping with display labels ──
# Used for SQGLP score drill-down: maps metric_id → (element, display_name)
METRIC_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    # Size
    "market_cap": ("size", "Market Cap"),
    "institutional_holding": ("size", "FII + DII Holding"),
    "analyst_coverage": ("size", "Analyst Coverage"),
    "daily_turnover_ratio": ("size", "Daily Turnover Ratio"),
    # Quality Business
    "roce_5yr_avg": ("quality_business", "RoCE 5yr Avg"),
    "roe_5yr_avg": ("quality_business", "ROE 5yr Avg"),
    "operating_margin_5yr": ("quality_business", "OPM 5yr Avg"),
    "dupont_margin": ("quality_business", "DuPont: Net Margin"),
    "dupont_turnover": ("quality_business", "DuPont: Asset Turnover"),
    "dupont_equity_multiplier": ("quality_business", "DuPont: Equity Multiplier"),
    "cash_conversion": ("quality_business", "Cash Conversion"),
    "fcf_yield": ("quality_business", "FCF Yield"),
    "debt_equity": ("quality_business", "Debt/Equity"),
    "interest_coverage": ("quality_business", "Interest Coverage"),
    "working_capital_days_trend": ("quality_business", "Working Capital Days"),
    # Quality Management
    "promoter_holding_trend": ("quality_management", "Promoter Holding Trend"),
    "promoter_pledge": ("quality_management", "Promoter Pledge %"),
    "owner_operator_signal": ("quality_management", "Owner-Operator Signal"),
    "equity_dilution": ("quality_management", "Equity Dilution 10yr"),
    "dividend_consistency": ("quality_management", "Dividend Consistency"),
    "effective_tax_rate_variance": ("quality_management", "Tax Rate Consistency"),
    # Growth
    "revenue_cagr_5yr": ("growth", "Revenue CAGR 5yr"),
    "pat_cagr_5yr": ("growth", "PAT CAGR 5yr"),
    "eps_cagr_5yr": ("growth", "EPS CAGR 5yr"),
    "pat_cagr_3yr": ("growth", "PAT CAGR 3yr"),
    "operating_leverage": ("growth", "Operating Leverage"),
    "financial_leverage_ratio": ("growth", "Financial Leverage"),
    "growth_quality_grade": ("growth", "Growth Quality Grade"),
    "revenue_growth_consistency": ("growth", "Revenue Growth Consistency"),
    "revenue_cagr_3yr": ("growth", "Revenue CAGR 3yr"),
    "ebit_cagr_5yr": ("growth", "EBIT CAGR 5yr"),
    "ebit_cagr_3yr": ("growth", "EBIT CAGR 3yr"),
    "price_lever_signal": ("growth", "Pricing Power"),
    # Longevity
    "roce_consistency": ("longevity", "RoCE >15% Years"),
    "cap_proxy": ("longevity", "CAP Proxy"),
    "revenue_growth_streak": ("longevity", "Growth Streak"),
    "gross_margin_stability": ("longevity", "Margin Stability"),
    "reinvestment_rate": ("longevity", "Reinvestment Rate"),
    "fcf_consistency": ("longevity", "FCF+ Years"),
    # Price
    "pe_ttm": ("price", "PE TTM"),
    "peg_ratio": ("price", "PEG Ratio"),
    "trailing_peg": ("price", "Trailing PEG"),
    "ev_ebitda": ("price", "EV/EBITDA"),
    "pe_vs_historical": ("price", "PE Percentile"),
    "dcf_margin_of_safety": ("price", "DCF Margin of Safety"),
    "reverse_dcf_growth": ("price", "Reverse DCF Implied"),
    "earnings_yield_vs_gsec": ("price", "EY Spread vs G-Sec"),
}

# ── Flag-to-SQGLP element mapping ──
# Maps raw flag strings to their SQGLP element for per-section grouping
FLAG_ELEMENT_MAP: dict[str, str] = {
    # Growth
    "growth_quality_high_quality": "growth",
    "growth_quality_moderate": "growth",
    "growth_quality_low_quality": "growth",
    "very_short_history_unreliable": "growth",
    "bonus_split_adjusted": "growth",
    "high_dilution": "growth",
    "minimal_dilution": "growth",
    # Quality Business (Profitability + Leverage + Efficiency)
    "consistently_high_roce": "quality_business",
    "exceptional_roce": "quality_business",
    "improving_roce": "quality_business",
    "declining_roce": "quality_business",
    "high_operating_margin": "quality_business",
    "improving_margins": "quality_business",
    "cash_cow": "quality_business",
    "cfi_dominated_by_acquisitions": "quality_business",
    "volatile_tax_rate": "quality_business",
    "debt_risk": "quality_business",
    "virtually_debt_free": "quality_business",
    "low_interest_coverage": "quality_business",
    "strong_interest_coverage": "quality_business",
    "improving_working_capital": "quality_business",
    "worsening_working_capital": "quality_business",
    # Price (Valuation)
    "very_expensive_pe": "price",
    "expensive_pe": "price",
    "cheap_pe": "price",
    "attractively_valued_peg": "price",
    "expensive_peg": "price",
    "attractive_trailing_peg": "price",
    "pe_above_historical_75th": "price",
    "pe_below_historical_25th": "price",
    "dcf_undervalued": "price",
    "dcf_overvalued": "price",
    "negative_average_fcf": "price",
    "negative_fcf_even_after_outlier_removal": "price",
    "reverse_dcf_overpriced": "price",
    "reverse_dcf_underpriced": "price",
    "earnings_yield_above_gsec": "price",
    "gsec_more_attractive": "price",
    # Size
    "small_cap": "size",
    "mid_cap": "size",
    "large_cap": "size",
    "micro_cap": "size",
    "low_institutional_ownership": "size",
    "heavily_institutional": "size",
    "under_researched": "size",
    "lightly_covered": "size",
    # Quality Management
    "promoter_increasing_stake": "quality_management",
    "promoter_reducing_stake": "quality_management",
    "promoter_pledge_red_flag": "quality_management",
    # Longevity
    "wide_moat_cap": "longevity",
    "moderate_moat_cap": "longevity",
    "highly_stable_margins": "longevity",
    "volatile_margins": "longevity",
    "heavy_reinvestment": "longevity",
    "consistent_fcf_generator": "longevity",
    "consistent_organic_fcf_generator": "longevity",
    # Composite
    "possible_bonus_split": "composite",
}

# ── SQGLP element display config ──
ELEMENT_CONFIG: dict[str, dict] = {
    "size": {"label": "Size", "short": "S", "weight": "10%"},
    "quality_business": {"label": "Quality — Business", "short": "QB", "weight": "20%"},
    "quality_management": {"label": "Quality — Management", "short": "QM", "weight": "10%"},
    "growth": {"label": "Growth", "short": "G", "weight": "25%"},
    "longevity": {"label": "Longevity", "short": "L", "weight": "20%"},
    "price": {"label": "Price", "short": "P", "weight": "15%"},
}

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _md_inline(text: str) -> str:
    """Convert basic markdown inline formatting to HTML.

    Handles: **bold**, *italic*, `code`.
    Escapes HTML entities first to prevent XSS.
    """
    from markupsafe import escape

    text = str(escape(text))
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Italic: *text* or _text_ (but not inside words like file_name)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
    # Inline code: `code`
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _paragraphize(text: str) -> Markup:
    """Format long-form text into readable HTML paragraphs.

    Strategy:
    1. If text has \\n\\n paragraph breaks, split on those.
    2. Otherwise, group sentences into ~2-3 sentence paragraphs for readability.
    Single-line newlines are treated as soft breaks within a paragraph.
    Markdown bold/italic/code is converted to HTML.
    """
    if not text:
        return Markup("")

    # Step 1: Split on explicit double-newline paragraph breaks
    raw_paragraphs = re.split(r"\n\n+", text.strip())

    # Step 2: For any paragraph that's a long single block (>300 chars, 3+ sentences),
    # split into smaller groups of 2-3 sentences for readability
    final_paragraphs: list[str] = []
    for para in raw_paragraphs:
        para = para.strip().replace("\n", " ")  # collapse soft newlines
        if not para:
            continue
        if len(para) > 300:
            # Split on sentence boundaries: period/exclamation/question followed by space+uppercase
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
            if len(sentences) >= 2:
                # Group into chunks — flush when chunk reaches 2+ sentences
                # and accumulated text exceeds ~200 chars
                chunk: list[str] = []
                for sent in sentences:
                    # If this single sentence is already long, flush current chunk first
                    if chunk and (len(" ".join(chunk)) + len(sent)) > 350:
                        final_paragraphs.append(" ".join(chunk))
                        chunk = []
                    chunk.append(sent)
                    if len(chunk) >= 2 and len(" ".join(chunk)) > 200:
                        final_paragraphs.append(" ".join(chunk))
                        chunk = []
                if chunk:
                    final_paragraphs.append(" ".join(chunk))
                continue
        final_paragraphs.append(para)

    # Convert markdown inline formatting (bold, italic, code) to HTML
    html = "".join(f"<p>{_md_inline(p)}</p>" for p in final_paragraphs if p)
    return Markup(html)


def _sanitize_filename(name: str, max_length: int = 40) -> str:
    """Sanitize a string for use in filenames."""
    clean = re.sub(r"[^\w\s-]", "", name)  # Remove special chars
    clean = re.sub(r"[\s]+", "_", clean.strip())  # Spaces → underscores
    return clean[:max_length]


class ReportGenerator:
    """Generate HTML dashboard, markdown summary, and JSON data exports."""

    def __init__(self, output_dir: str | None = None):
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters["paragraphize"] = _paragraphize
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "reports"

    def generate(self, result, formats: list[str] | None = None) -> Path:
        """Generate all requested report formats.

        Args:
            result: AnalysisResult from the service layer.
            formats: List of formats to generate (html, md, json). Default: all.

        Returns:
            Path to the report directory.
        """
        formats = formats or ["html", "md", "json"]
        metadata = result.data.get("metadata", {})
        company_name = metadata.get("name", result.ticker)
        report_dir = self._make_report_dir(result.ticker, company_name)

        # Compute report data
        growth_decomposition = self._compute_growth_decomposition(result)
        executive_summary = self._build_executive_summary(result)
        financial_snapshot = self._build_financial_snapshot(result)
        sector_context = self._build_sector_context(result)
        dcf_summary = self._build_dcf_summary(result)
        cashflow_quality = self._build_cashflow_quality(result)
        pe_band_summary = self._build_pe_band_summary(result)
        score_drilldown = self._build_score_drilldown(result)
        flags = self._collect_flags(result.metrics)
        element_summaries = self._build_element_summaries(result, score_drilldown, flags)

        if "json" in formats:
            self._export_json(result, report_dir, growth_decomposition)
            logger.info(f"JSON exports saved to {report_dir}")

        # Copy annual reports to the report folder
        self._copy_annual_reports(result, report_dir)

        # Pre-render charts for HTML
        charts = self._render_charts(result)

        shareholding_data = self._prepare_shareholding_data(result)

        if "html" in formats:
            html = self._render_html(
                result, charts, growth_decomposition,
                executive_summary=executive_summary,
                financial_snapshot=financial_snapshot,
                sector_context=sector_context,
                dcf_summary=dcf_summary,
                cashflow_quality=cashflow_quality,
                shareholding_data=shareholding_data,
                score_drilldown=score_drilldown,
                element_summaries=element_summaries,
                flags_precomputed=flags,
            )
            path = report_dir / f"{result.ticker}_dashboard.html"
            path.write_text(html)
            logger.info(f"HTML dashboard: {path}")

        if "md" in formats:
            md = self._render_markdown(
                result, growth_decomposition,
                executive_summary=executive_summary,
                financial_snapshot=financial_snapshot,
                shareholding_data=shareholding_data,
                sector_context=sector_context,
                dcf_summary=dcf_summary,
                cashflow_quality=cashflow_quality,
                pe_band_summary=pe_band_summary,
                score_drilldown=score_drilldown,
                element_summaries=element_summaries,
                flags_precomputed=flags,
            )
            path = report_dir / f"{result.ticker}_report.md"
            path.write_text(md)
            logger.info(f"Markdown report: {path}")

        return report_dir

    # ── HTML ──

    def _render_html(self, result, charts: dict, growth_decomposition: dict | None = None,
                     executive_summary: dict | None = None,
                     financial_snapshot: list | None = None,
                     sector_context: dict | None = None,
                     dcf_summary: dict | None = None,
                     cashflow_quality: dict | None = None,
                     shareholding_data: list | None = None,
                     score_drilldown: dict | None = None,
                     element_summaries: dict | None = None,
                     flags_precomputed: list | None = None) -> str:
        template = self.env.get_template("sqglp_report.html.j2")
        flags = flags_precomputed if flags_precomputed is not None else self._collect_flags(result.metrics)
        return template.render(
            ticker=result.ticker,
            metadata=result.data.get("metadata", {}),
            scores=result.scores,
            metrics=self._metrics_to_display(result.metrics),
            flags=flags,
            llm_analysis=result.llm_analysis,
            growth=growth_decomposition,
            executive_summary=executive_summary or {},
            snapshot=financial_snapshot or [],
            sector_context=sector_context or {},
            dcf_summary=dcf_summary or {},
            cashflow_quality=cashflow_quality or {},
            radar_chart=charts.get("radar", ""),
            roce_trend_chart=charts.get("roce_trend", ""),
            pe_band_chart=charts.get("pe_band", ""),
            growth_chart=charts.get("growth", ""),
            shareholding_data=shareholding_data or [],
            dcf_gauge_chart=charts.get("dcf_gauge", ""),
            cashflow_quality_chart=charts.get("cashflow_quality", ""),
            pe_band_historical_chart=charts.get("pe_band_historical", ""),
            score_drilldown=score_drilldown or {},
            element_summaries=element_summaries or {},
            element_config=ELEMENT_CONFIG,
            errors=result.errors,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    # ── Markdown ──

    def _render_markdown(self, result, growth_decomposition: dict | None = None,
                         executive_summary: dict | None = None,
                         financial_snapshot: list | None = None,
                         shareholding_data: list | None = None,
                         sector_context: dict | None = None,
                         dcf_summary: dict | None = None,
                         cashflow_quality: dict | None = None,
                         pe_band_summary: dict | None = None,
                         score_drilldown: dict | None = None,
                         element_summaries: dict | None = None,
                         flags_precomputed: list | None = None) -> str:
        template = self.env.get_template("sqglp_report.md.j2")
        flags = flags_precomputed if flags_precomputed is not None else self._collect_flags(result.metrics)
        return template.render(
            ticker=result.ticker,
            metadata=result.data.get("metadata", {}),
            scores=result.scores,
            metrics=self._metrics_to_display(result.metrics),
            flags=flags,
            llm_analysis=result.llm_analysis,
            growth=growth_decomposition,
            executive_summary=executive_summary or {},
            snapshot=financial_snapshot or [],
            shareholding_data=shareholding_data or [],
            sector_context=sector_context or {},
            dcf_summary=dcf_summary or {},
            cashflow_quality=cashflow_quality or {},
            pe_band_summary=pe_band_summary or {},
            score_drilldown=score_drilldown or {},
            element_summaries=element_summaries or {},
            element_config=ELEMENT_CONFIG,
            errors=result.errors,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    # ── JSON Export ──

    def _export_json(self, result, report_dir: Path, growth_decomposition: dict | None = None):
        # raw_metrics.json
        metrics_export = {}
        for mid, mr in result.metrics.items():
            metrics_export[mid] = {
                "value": mr.value if mr.ok else None,
                "error": mr.error,
                "flags": mr.flags,
                "metadata": mr.metadata,
            }
        self._write_json(report_dir / "raw_metrics.json", metrics_export)

        # scores.json
        self._write_json(report_dir / "scores.json", result.scores)

        # growth_decomposition.json
        if growth_decomposition:
            self._write_json(report_dir / "growth_decomposition.json", growth_decomposition)

        # llm_analysis.json
        if result.llm_analysis:
            self._write_json(report_dir / "llm_analysis.json", result.llm_analysis)

    # ── Charts ──

    def _render_charts(self, result) -> dict:
        charts = {}

        # SQGLP radar removed — element scores table is sufficient

        ratios = result.data.get("ratios")
        if ratios is not None and not ratios.empty:
            charts["roce_trend"] = self._roce_trend_chart(ratios)

        price = result.data.get("price")
        metrics = result.metrics
        if price is not None and not price.empty:
            charts["pe_band"] = self._pe_band_chart(price, metrics)

        charts["growth"] = self._growth_chart(metrics)

        # Shareholding: uses HTML table now, no chart needed

        # Feature 4: DCF gauge
        dcf_chart = self._dcf_visualization(result)
        if dcf_chart:
            charts["dcf_gauge"] = dcf_chart

        # Feature 5: Cash flow quality
        cf_chart = self._cashflow_quality_chart(result)
        if cf_chart:
            charts["cashflow_quality"] = cf_chart

        # Feature 7: Historical PE band
        pe_hist = self._pe_band_chart_historical(result)
        if pe_hist:
            charts["pe_band_historical"] = pe_hist

        # Peer radar removed — peer comparison table is clearer

        return charts

    def _radar_chart(self, elements: dict) -> str:
        categories = [
            "Size", "Quality\n(Business)", "Quality\n(Mgmt)",
            "Growth", "Longevity", "Price",
        ]
        keys = [
            "size", "quality_business", "quality_management",
            "growth", "longevity", "price",
        ]
        values = [elements.get(k, 0) or 0 for k in keys]
        values.append(values[0])  # Close the polygon
        categories.append(categories[0])

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line_color="#2563eb",
            fillcolor="rgba(37, 99, 235, 0.2)",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10]),
            ),
            showlegend=False,
            margin=dict(l=60, r=60, t=30, b=30),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    def _roce_trend_chart(self, ratios) -> str:
        import pandas as pd
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

    def _pe_band_chart(self, price, metrics: dict) -> str:
        import pandas as pd
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

    def _growth_chart(self, metrics: dict) -> str:
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

    def _compute_growth_decomposition(self, result) -> dict | None:
        """Compute 4-lever growth decomposition from result data."""
        try:
            financials = result.data.get("financials")
            if financials is None or financials.empty:
                return None

            decomposition = compute_lever_decomposition_table(result.data)

            # Enrich valuation check with PE from metrics if not in financials
            val_check = decomposition.get("valuation_check", {})
            if val_check.get("current_pe") is None:
                pe_result = result.metrics.get("pe_ttm")
                if pe_result and pe_result.ok:
                    val_check["current_pe"] = pe_result.value
                    pat_5 = val_check.get("pat_cagr_5yr")
                    if pat_5 and pat_5 > 0:
                        val_check["trailing_peg"] = pe_result.value / pat_5
                        from boundless100x.compute_engine.metrics.builtin.growth import _peg_verdict
                        quality = decomposition.get("growth_synthesis", {}).get("quality_flag", "moderate")
                        val_check["verdict"] = _peg_verdict(val_check["trailing_peg"], quality)

            return decomposition
        except Exception as e:
            logger.warning(f"Growth decomposition failed: {e}")
            return None

    # ── Executive Summary ──

    def _build_executive_summary(self, result) -> dict:
        """Build executive summary data for the decision dashboard."""
        metadata = result.data.get("metadata", {})
        scores = result.scores
        llm = result.llm_analysis

        summary = {
            "composite_score": scores.get("composite"),
            "element_scores": scores.get("elements", {}),
            "company_name": metadata.get("name", result.ticker),
            "sector": metadata.get("sector", "N/A"),
            "market_cap": metadata.get("Market Cap"),
            "has_llm": False,
            "suggested_action": None,
            "conviction_level": None,
            "thesis": None,
            "holding_period": None,
            "kill_risks": [],
            "key_highlights": [],
        }

        # LLM-enriched fields
        if llm and not llm.get("skipped"):
            p2 = llm.get("pass2", {})
            if p2 and not p2.get("error") and not p2.get("skipped"):
                summary["has_llm"] = True
                summary["suggested_action"] = p2.get("suggested_action")
                summary["conviction_level"] = p2.get("conviction_level")
                summary["thesis"] = p2.get("thesis")
                summary["holding_period"] = p2.get("target_holding_period")
                kill = p2.get("kill_the_thesis", [])
                summary["kill_risks"] = kill[:3]

        # Key metric highlights for --no-llm fallback
        highlights = []
        metric_picks = [
            ("roce_5yr_avg", "RoCE 5yr", "%"),
            ("pat_cagr_5yr", "PAT CAGR 5yr", "%"),
            ("pe_ttm", "PE TTM", "x"),
            ("debt_equity", "D/E", "x"),
            ("fcf_consistency", "FCF+ Years", "yrs"),
        ]
        for mid, label, unit in metric_picks:
            mr = result.metrics.get(mid)
            if mr and mr.ok and mr.value is not None:
                highlights.append({"label": label, "value": mr.value, "unit": unit})
        summary["key_highlights"] = highlights

        # ── Red flags: top 3 "bad" sentiment flags for exec summary ──
        red_flags = []
        seen_flags: set[str] = set()
        for mid, mr in result.metrics.items():
            if mr.ok and mr.flags:
                for f in mr.flags:
                    if f in seen_flags:
                        continue
                    seen_flags.add(f)
                    label, sentiment = FLAG_LABELS.get(f, (None, None))
                    if sentiment == "bad" and label:
                        red_flags.append(label)
        summary["red_flags"] = red_flags[:5]

        # ── Quality-Growth Quadrant badge ──
        qg = result.metrics.get("quality_growth_quadrant")
        if qg and qg.ok and qg.value:
            qg_meta = qg.metadata or {}
            QUADRANT_LABELS = {
                "compounder": ("Compounder", "good", "High quality + High growth"),
                "quality_trap": ("Quality Trap", "bad", "High quality but low growth"),
                "growth_trap": ("Growth Trap", "bad", "High growth but low quality"),
                "dog": ("Dog", "bad", "Low quality + Low growth"),
                "emerging_compounder": ("Emerging Compounder", "neutral", "Improving quality & growth"),
            }
            raw_val = qg.value
            label, sentiment, desc = QUADRANT_LABELS.get(
                raw_val, (raw_val.replace("_", " ").title(), "neutral", "")
            )
            summary["quadrant"] = {
                "label": label,
                "sentiment": sentiment,
                "description": desc,
                "avg_roce": qg_meta.get("avg_roce"),
                "pat_cagr": qg_meta.get("pat_cagr"),
            }

        # ── Analyst target price cross-check ──
        ac = result.data.get("analyst_coverage", {})
        if ac and ac.get("avg_target") and ac.get("count"):
            dcf = result.metrics.get("dcf_margin_of_safety")
            current_price = result.data.get("metadata", {}).get("Current Price")
            analyst_info = {
                "count": ac["count"],
                "avg_target": ac["avg_target"],
                "consensus": ac.get("consensus", "—"),
            }
            if current_price:
                analyst_info["current_price"] = current_price
                analyst_info["upside"] = (ac["avg_target"] - current_price) / current_price * 100
            if dcf and dcf.ok and dcf.metadata:
                analyst_info["dcf_intrinsic"] = dcf.metadata.get("intrinsic_per_share")
            summary["analyst"] = analyst_info

        return summary

    # ── Score Drill-Down ──

    def _build_score_drilldown(self, result) -> dict:
        """Build per-element drill-down showing which sub-metrics drive each score.

        Returns dict keyed by element: {
            "growth": [
                {"name": "Revenue CAGR 5yr", "value": "17.0%", "score": 5.0, "weight": "12%", "contribution": "good|mid|low"},
                ...
            ]
        }
        """
        details = result.scores.get("details", {})
        if not details:
            return {}

        drilldown: dict[str, list[dict]] = {}

        for metric_id, info in details.items():
            if not isinstance(info, dict):
                continue
            weight = info.get("weight", 0)
            if weight == 0:
                continue

            element, display_name = METRIC_DISPLAY_NAMES.get(metric_id, (None, None))
            if element is None:
                continue

            score = info.get("score")
            value = info.get("value")

            # Format value for display
            if value is None:
                val_str = "—"
            elif isinstance(value, str):
                val_str = value.replace("_", " ").title()
            elif isinstance(value, (int, float)):
                if abs(value) >= 100:
                    val_str = f"{value:,.0f}"
                elif abs(value) >= 1:
                    val_str = f"{value:.1f}"
                else:
                    val_str = f"{value:.2f}"
            else:
                val_str = str(value)

            # Score contribution level
            if score is None:
                contribution = "none"
            elif score >= 0.7:
                contribution = "good"
            elif score >= 0.4:
                contribution = "mid"
            else:
                contribution = "low"

            entry = {
                "name": display_name,
                "value": val_str,
                "score_pct": f"{score * 100:.0f}%" if score is not None else "—",
                "weight": f"{weight * 100:.0f}%",
                "contribution": contribution,
            }

            drilldown.setdefault(element, [])
            drilldown[element].append(entry)

        # Sort each element's metrics by weight descending
        for el in drilldown:
            drilldown[el].sort(key=lambda x: float(x["weight"].rstrip("%")), reverse=True)

        return drilldown

    def _build_element_summaries(self, result, score_drilldown: dict, flags: list[dict]) -> dict[str, str]:
        """Generate a data-driven 1-2 sentence summary for each SQGLP element.

        Uses metric scores and flags to build a narrative without needing LLM.
        """
        summaries: dict[str, str] = {}

        for element, config in ELEMENT_CONFIG.items():
            parts = []
            drilldown = score_drilldown.get(element, [])
            el_flags = [f for f in flags if f.get("element") == element]

            # Identify top strengths and weaknesses from drilldown
            strengths = [m for m in drilldown if m["contribution"] == "good"]
            weaknesses = [m for m in drilldown if m["contribution"] == "low"]

            if strengths:
                top = strengths[:3]
                names = [f"{m['name']} ({m['value']})" for m in top]
                if len(names) == 1:
                    parts.append(f"Strong on {names[0]}.")
                else:
                    parts.append(f"Strong on {', '.join(names[:-1])} and {names[-1]}.")

            if weaknesses:
                bottom = weaknesses[:2]
                names = [f"{m['name']} ({m['value']})" for m in bottom]
                if len(names) == 1:
                    parts.append(f"Weak on {names[0]}.")
                else:
                    parts.append(f"Weak on {' and '.join(names)}.")

            # Add notable flags
            good_flags = [f["label"] for f in el_flags if f["sentiment"] == "good"]
            bad_flags = [f["label"] for f in el_flags if f["sentiment"] == "bad"]

            if good_flags and not strengths:
                parts.append(f"{', '.join(good_flags[:2])}.")
            if bad_flags and not weaknesses:
                parts.append(f"Watch: {', '.join(bad_flags[:2])}.")

            # Fallback if no drilldown data
            if not parts:
                el_score = result.scores.get("elements", {}).get(element)
                if el_score is not None:
                    if el_score >= 7:
                        parts.append(f"Scores well at {el_score:.1f}/10.")
                    elif el_score >= 4:
                        parts.append(f"Average at {el_score:.1f}/10.")
                    else:
                        parts.append(f"Below average at {el_score:.1f}/10.")

            if parts:
                summaries[element] = " ".join(parts)

        return summaries

    # ── Financial Snapshot ──

    def _build_financial_snapshot(self, result) -> list[dict]:
        """Build 10-year financial snapshot by joining multiple DataFrames."""
        financials = result.data.get("financials")
        if financials is None or financials.empty:
            return []

        def annual_only(df):
            if df is None or df.empty or "year" not in df.columns:
                return pd.DataFrame()
            mask = df["year"].astype(str).str.startswith("Mar", na=False)
            return df[mask].copy()

        df_fin = annual_only(financials)
        if df_fin.empty:
            return []

        # Build snapshot from financials
        snapshot = {}
        for _, row in df_fin.iterrows():
            yr = str(row["year"])
            snapshot[yr] = {
                "year": yr,
                "revenue": _safe_numeric(row.get("revenue")),
                "pat": _safe_numeric(row.get("pat")),
                "eps": _safe_numeric(row.get("eps")),
                "opm": _safe_numeric(row.get("opm_pct")),
            }

        # Merge RoCE from ratios
        df_rat = annual_only(result.data.get("ratios"))
        if not df_rat.empty and "roce" in df_rat.columns:
            for _, row in df_rat.iterrows():
                yr = str(row["year"])
                if yr in snapshot:
                    snapshot[yr]["roce"] = _safe_numeric(row.get("roce"))

        # Merge D/E from balance_sheet
        df_bs = annual_only(result.data.get("balance_sheet"))
        if not df_bs.empty:
            for _, row in df_bs.iterrows():
                yr = str(row["year"])
                if yr in snapshot:
                    borrowings = _safe_numeric(row.get("borrowings"))
                    equity = _safe_numeric(row.get("equity_capital"))
                    reserves = _safe_numeric(row.get("reserves"))
                    if borrowings is not None and equity is not None and reserves is not None:
                        total_equity = equity + reserves
                        snapshot[yr]["de"] = borrowings / total_equity if total_equity > 0 else None
                    else:
                        snapshot[yr]["de"] = None

        # Merge CFO from cashflow
        df_cf = annual_only(result.data.get("cashflow"))
        if not df_cf.empty and "cfo" in df_cf.columns:
            for _, row in df_cf.iterrows():
                yr = str(row["year"])
                if yr in snapshot:
                    snapshot[yr]["cfo"] = _safe_numeric(row.get("cfo"))

        # Fill missing keys and sort
        all_keys = ["year", "revenue", "pat", "eps", "opm", "roce", "de", "cfo"]
        result_list = []
        for yr in sorted(snapshot.keys()):
            entry = snapshot[yr]
            for k in all_keys:
                entry.setdefault(k, None)
            result_list.append(entry)

        # Compute trend arrows: compare latest year to 3 years ago for key metrics
        if len(result_list) >= 4:
            latest = result_list[-1]
            compare_to = result_list[-4]  # 3 years ago
            trends = {}
            # higher_is_better metrics
            for key in ["revenue", "pat", "eps", "opm", "roce", "cfo"]:
                v_now = latest.get(key)
                v_then = compare_to.get(key)
                if v_now is not None and v_then is not None and v_then != 0:
                    pct_change = (v_now - v_then) / abs(v_then) * 100
                    if pct_change > 5:
                        trends[key] = "up"
                    elif pct_change < -5:
                        trends[key] = "down"
                    else:
                        trends[key] = "flat"
                else:
                    trends[key] = None
            # D/E: lower_is_better (inverted)
            de_now = latest.get("de")
            de_then = compare_to.get("de")
            if de_now is not None and de_then is not None and de_then != 0:
                de_change = (de_now - de_then) / abs(de_then) * 100
                if de_change < -5:
                    trends["de"] = "up"  # de down = good = green arrow
                elif de_change > 5:
                    trends["de"] = "down"  # de up = bad = red arrow
                else:
                    trends["de"] = "flat"
            else:
                trends["de"] = None
            # Attach trends to the list (as metadata)
            for entry in result_list:
                entry["_trends"] = trends

        return result_list

    # ── Shareholding ──

    def _shareholding_chart(self, result) -> str:
        """Build Plotly shareholding chart with two vertically-stacked panels.

        Top panel: Promoter & Public (large, stable holdings — narrow y-range).
        Bottom panel: FII & DII (institutional flows — zoomed in to show real changes).
        This avoids the "flat line" problem where Promoter at ~50% squishes
        FII/DII movements into invisibility.
        """
        from plotly.subplots import make_subplots

        sh_bse = result.data.get("shareholding_bse")
        sh_screener = result.data.get("shareholding")

        if sh_bse is not None and not sh_bse.empty and len(sh_bse) >= 2:
            df = sh_bse.copy()
            has_pledge = "promoter_pledge_pct" in df.columns
        elif sh_screener is not None and not sh_screener.empty and len(sh_screener) >= 2:
            df = sh_screener.copy()
            has_pledge = False
        else:
            return ""

        if "quarter" not in df.columns:
            return ""

        df["_sort_date"] = pd.to_datetime(df["quarter"], format="%b %Y", errors="coerce")
        df = df.sort_values("_sort_date").reset_index(drop=True)
        df = df.drop(columns=["_sort_date"])
        quarters = df["quarter"].astype(str)

        def _make_label(name: str, vals) -> str:
            start_val, end_val = float(vals.iloc[0]), float(vals.iloc[-1])
            delta = end_val - start_val
            delta_str = f" ({delta:+.1f}pp)" if abs(delta) >= 0.1 else ""
            return f"{name} {end_val:.1f}%{delta_str}"

        def _y_range(vals, padding_pct: float = 0.3) -> list:
            """Compute y-axis range with padding so lines don't hug the edges."""
            vmin, vmax = float(vals.min()), float(vals.max())
            spread = max(vmax - vmin, 1.0)
            return [vmin - spread * padding_pct, vmax + spread * padding_pct]

        # Determine which categories have data
        top_traces = []  # (col, name, color) — Promoter & Public
        bot_traces = []  # (col, name, color) — FII & DII & Govt
        trace_defs = [
            ("promoter_pct", "Promoter", "#2563eb", "top"),
            ("public_pct", "Public", "#9ca3af", "top"),
            ("fii_pct", "FII", "#16a34a", "bot"),
            ("dii_pct", "DII", "#f59e0b", "bot"),
            ("govt_pct", "Government", "#8b5cf6", "bot"),
        ]
        for col, name, color, panel in trace_defs:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                if vals.sum() > 0:
                    bucket = top_traces if panel == "top" else bot_traces
                    bucket.append((col, name, color, vals))

        if not top_traces and not bot_traces:
            return ""

        # If only one panel has data, use single chart
        has_both = bool(top_traces) and bool(bot_traces)
        if has_both:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=("Promoter & Public", "Institutional (FII & DII)"),
                row_heights=[0.45, 0.55],
            )
        else:
            fig = go.Figure()

        # ── Top panel: Promoter & Public ──
        for col, name, color, vals in top_traces:
            row = 1 if has_both else None
            fig.add_trace(
                go.Scatter(
                    x=quarters, y=vals,
                    name=_make_label(name, vals),
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=4),
                    hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>",
                    legendgroup="top",
                ),
                row=row, col=1 if has_both else None,
            )

        # ── Bottom panel: FII, DII, Govt ──
        for col, name, color, vals in bot_traces:
            row = 2 if has_both else None
            fig.add_trace(
                go.Scatter(
                    x=quarters, y=vals,
                    name=_make_label(name, vals),
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5),
                    hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>",
                    legendgroup="bot",
                ),
                row=row, col=1 if has_both else None,
            )

        # ── Promoter pledge overlay on top panel ──
        if has_pledge:
            pledge = pd.to_numeric(df["promoter_pledge_pct"], errors="coerce").fillna(0)
            if pledge.sum() > 0:
                row = 1 if has_both else None
                fig.add_trace(
                    go.Scatter(
                        x=quarters, y=pledge,
                        name=_make_label("Pledge", pledge),
                        mode="lines+markers",
                        line=dict(color="#dc2626", width=2, dash="dot"),
                        marker=dict(size=5, symbol="x"),
                        hovertemplate="Pledge: %{y:.1f}%<extra></extra>",
                        legendgroup="top",
                    ),
                    row=row, col=1 if has_both else None,
                )

        # ── Y-axis ranges: zoom into actual data range so changes are visible ──
        if has_both:
            top_all = pd.concat([v for _, _, _, v in top_traces])
            bot_all = pd.concat([v for _, _, _, v in bot_traces])
            fig.update_yaxes(range=_y_range(top_all), title_text="Holding %", row=1, col=1,
                             gridcolor="rgba(128,128,128,0.15)")
            fig.update_yaxes(range=_y_range(bot_all), title_text="Holding %", row=2, col=1,
                             gridcolor="rgba(128,128,128,0.15)")
            fig.update_xaxes(showgrid=False, row=1, col=1)
            fig.update_xaxes(showgrid=False, row=2, col=1)

        fig.update_layout(
            title="Shareholding Pattern Trend",
            margin=dict(l=50, r=50, t=60, b=50),
            height=520 if has_both else 350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        if not has_both:
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(title_text="Holding %", gridcolor="rgba(128,128,128,0.15)")

        return pio.to_html(fig, include_plotlyjs=False, full_html=False)

    def _prepare_shareholding_data(self, result) -> list[dict]:
        """Prepare shareholding data as list of dicts for Markdown table."""
        sh_bse = result.data.get("shareholding_bse")
        sh_screener = result.data.get("shareholding")

        if sh_bse is not None and not sh_bse.empty:
            df = sh_bse.copy()
        elif sh_screener is not None and not sh_screener.empty:
            df = sh_screener.copy()
        else:
            return []

        if "quarter" not in df.columns:
            return []

        # Parse quarter strings (e.g. "Mar 2023") to dates for proper chronological sort
        df["_sort_date"] = pd.to_datetime(df["quarter"], format="%b %Y", errors="coerce")
        df = df.sort_values("_sort_date").reset_index(drop=True)
        df = df.drop(columns=["_sort_date"])

        records = []
        for _, row in df.iterrows():
            records.append({
                "quarter": str(row.get("quarter", "")),
                "promoter_pct": _safe_numeric(row.get("promoter_pct")),
                "fii_pct": _safe_numeric(row.get("fii_pct")),
                "dii_pct": _safe_numeric(row.get("dii_pct")),
                "public_pct": _safe_numeric(row.get("public_pct")),
                "govt_pct": _safe_numeric(row.get("govt_pct")),
                "promoter_pledge_pct": _safe_numeric(row.get("promoter_pledge_pct")),
            })
        return records

    # ── Feature 8: Sector-Relative Context ──

    # Metrics where lower values are better (valuation, leverage, volatility)
    LOWER_IS_BETTER_METRICS = {
        "pe_ttm", "peg_ratio", "trailing_peg", "ev_ebitda",
        "debt_equity", "gross_margin_stability", "working_capital_days_trend",
        "revenue_growth_consistency", "effective_tax_rate_variance",
        "dupont_equity_multiplier",
    }

    def _build_sector_context(self, result) -> dict:
        """Compute peer median/quartile context for key metrics.

        Returns dict keyed by metric_id with:
            {median, p25, p75, target_value, rank, total, vs_median, sentiment}
        where sentiment accounts for metric direction (higher/lower is better).
        """
        comparison = result.comparison
        if not comparison or not comparison.get("companies"):
            return {}

        companies = comparison["companies"]
        ranks = comparison.get("ranks", {})
        total = len(companies)
        context = {}

        for mid in comparison.get("metrics", []):
            values = [
                companies[t].get(mid)
                for t in companies
                if companies[t].get(mid) is not None
            ]
            if len(values) < 2:
                continue

            arr = np.array(values, dtype=float)
            median = float(np.nanmedian(arr))
            p25 = float(np.nanpercentile(arr, 25))
            p75 = float(np.nanpercentile(arr, 75))

            target_val = companies.get(result.ticker, {}).get(mid)
            target_rank = ranks.get(result.ticker, {}).get(mid)

            if target_val is not None:
                if target_val > median * 1.02:
                    vs = "above"
                elif target_val < median * 0.98:
                    vs = "below"
                else:
                    vs = "at"
            else:
                vs = None

            # Determine sentiment: is being above/below median good or bad?
            lower_better = mid in self.LOWER_IS_BETTER_METRICS
            if vs == "above":
                sentiment = "bad" if lower_better else "good"
            elif vs == "below":
                sentiment = "good" if lower_better else "bad"
            else:
                sentiment = "neutral"

            context[mid] = {
                "median": median,
                "p25": p25,
                "p75": p75,
                "target_value": target_val,
                "rank": target_rank,
                "total": total,
                "vs_median": vs,
                "sentiment": sentiment,
            }

        return context

    # ── Feature 4: DCF Visualization ──

    def _dcf_visualization(self, result) -> str:
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

    def _build_dcf_summary(self, result) -> dict:
        """Build DCF summary data for template rendering."""
        dcf = result.metrics.get("dcf_margin_of_safety")
        rdcf = result.metrics.get("reverse_dcf_growth")

        summary = {}
        if dcf and dcf.ok and dcf.value is not None:
            meta = dcf.metadata or {}
            summary["intrinsic_per_share"] = meta.get("intrinsic_per_share")
            summary["current_price"] = meta.get("current_price")
            summary["margin_pct"] = dcf.value
            summary["fcf_growth_assumed"] = meta.get("fcf_growth_assumed")

        if rdcf and rdcf.ok and rdcf.value is not None:
            meta_r = rdcf.metadata or {}
            summary["reverse_dcf_implied"] = rdcf.value
            summary["actual_cagr"] = meta_r.get("actual_cagr")
            if summary.get("reverse_dcf_implied") is not None and summary.get("actual_cagr") is not None:
                summary["reverse_dcf_gap"] = summary["reverse_dcf_implied"] - summary["actual_cagr"]

        return summary if summary.get("intrinsic_per_share") is not None else {}

    # ── Feature 5: Cash Flow Quality ──

    def _cashflow_quality_chart(self, result) -> str:
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

    def _build_cashflow_quality(self, result) -> dict:
        """Build cash flow quality summary metrics."""
        financials = result.data.get("financials")
        cashflow = result.data.get("cashflow")
        if financials is None or cashflow is None:
            return {}

        def _annual(df):
            if df is None or df.empty or "year" not in df.columns:
                return pd.DataFrame()
            mask = df["year"].astype(str).str.startswith("Mar", na=False)
            return df[mask].copy()

        df_fin = _annual(financials)
        df_cf = _annual(cashflow)

        if df_fin.empty or df_cf.empty:
            return {}

        merged = pd.merge(
            df_fin[["year", "pat"]],
            df_cf[["year", "cfo"]],
            on="year", how="inner",
        )
        merged["pat_num"] = pd.to_numeric(merged["pat"], errors="coerce")
        merged["cfo_num"] = pd.to_numeric(merged["cfo"], errors="coerce")
        merged = merged.dropna(subset=["pat_num", "cfo_num"])

        if merged.empty:
            return {}

        yearly_data = []
        ratios = []
        for _, row in merged.iterrows():
            cfo = float(row["cfo_num"])
            pat = float(row["pat_num"])
            ratio = (cfo / pat * 100) if pat > 0 else None
            yearly_data.append({"year": str(row["year"]), "cfo": cfo, "pat": pat, "ratio": ratio})
            if ratio is not None:
                ratios.append(ratio)

        cum_cfo = float(merged["cfo_num"].sum())
        cum_pat = float(merged["pat_num"].sum())
        cum_ratio = (cum_cfo / cum_pat * 100) if cum_pat > 0 else 0

        return {
            "avg_cfo_pat_ratio": float(np.mean(ratios)) if ratios else 0,
            "cumulative_cfo": cum_cfo,
            "cumulative_pat": cum_pat,
            "cumulative_ratio": cum_ratio,
            "yearly_data": yearly_data,
        }

    # ── Feature 7: Historical PE Band Chart ──

    def _pe_band_chart_historical(self, result) -> str:
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
        eps_dates = pd.to_datetime(df_fin["year"], format="%b %Y", errors="coerce") + pd.offsets.MonthEnd(0)
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

    def _build_pe_band_summary(self, result) -> dict:
        """Build PE band summary for markdown."""
        pe_hist = result.metrics.get("pe_vs_historical")
        pe_ttm = result.metrics.get("pe_ttm")

        if not pe_hist or not pe_hist.ok or not pe_ttm or not pe_ttm.ok:
            return {}

        # Compute simple PE range from financials
        financials = result.data.get("financials")
        meta = result.data.get("metadata", {})
        current_price = meta.get("Current Price")
        if financials is None or current_price is None:
            return {}

        mask = financials["year"].astype(str).str.startswith("Mar", na=False)
        df = financials[mask].copy()
        df["eps_num"] = pd.to_numeric(df["eps"], errors="coerce")
        valid_eps = df[df["eps_num"] > 0]["eps_num"]

        if valid_eps.empty:
            return {}

        implied_pes = [current_price / e for e in valid_eps]

        return {
            "percentile": pe_hist.value,
            "current_pe": pe_ttm.value,
            "pe_min": float(min(implied_pes)),
            "pe_max": float(max(implied_pes)),
        }

    # ── Feature 6: Peer Radar Overlay ──

    def _peer_radar_chart(self, result) -> str:
        """Create a radar chart overlaying target vs top peers on 6 normalized metrics."""
        comparison = result.comparison
        if not comparison or not comparison.get("companies"):
            return ""

        companies = comparison["companies"]
        if len(companies) < 2:
            return ""

        # 6 radar axes: metric_id, display_label, direction
        radar_axes = [
            ("roce_5yr_avg", "RoCE 5yr", "higher"),
            ("pat_cagr_5yr", "PAT CAGR 5yr", "higher"),
            ("operating_margin_5yr", "OPM 5yr", "higher"),
            ("pe_ttm", "Valuation\n(PE inv)", "lower"),
            ("debt_equity", "Leverage\n(D/E inv)", "lower"),
            ("fcf_consistency", "FCF Consistency", "higher"),
        ]

        # Check that at least 3 axes have data for the target
        target = result.ticker
        target_data = companies.get(target, {})
        available_axes = [ax for ax in radar_axes if target_data.get(ax[0]) is not None]
        if len(available_axes) < 3:
            return ""

        # Use available axes
        axes = available_axes

        # Select companies: target + up to 3 peers
        peer_tickers = [t for t in companies if t != target][:3]
        selected = [target] + peer_tickers

        # Normalize each metric to 0-10 across all selected companies
        normalized = {t: [] for t in selected}
        labels = []

        for mid, label, direction in axes:
            vals = [companies[t].get(mid) for t in selected]
            numeric_vals = [v for v in vals if v is not None]

            if len(numeric_vals) < 2:
                # Not enough data for normalization, skip this axis
                continue

            labels.append(label)
            vmin = min(numeric_vals)
            vmax = max(numeric_vals)
            spread = vmax - vmin if vmax != vmin else 1.0

            for t in selected:
                v = companies[t].get(mid)
                if v is not None:
                    norm = (v - vmin) / spread * 10
                    if direction == "lower":
                        norm = 10 - norm
                    normalized[t].append(max(0, min(10, norm)))
                else:
                    normalized[t].append(0)

        if len(labels) < 3:
            return ""

        # Close polygon
        labels_closed = labels + [labels[0]]

        colors = ["#2563eb", "#16a34a", "#f59e0b", "#8b5cf6"]
        fig = go.Figure()

        # Build actual values per company for hover
        actual_values = {t: [] for t in selected}
        for mid, label, direction in axes:
            for t in selected:
                v = companies[t].get(mid)
                actual_values[t].append(f"{v:.1f}" if v is not None else "N/A")

        for i, t in enumerate(selected):
            vals = normalized[t] + [normalized[t][0]]
            hover_vals = actual_values[t] + [actual_values[t][0]]
            hover_labels = labels + [labels[0]]
            hover_text = [f"{t}<br>{l}: {v}" for l, v in zip(hover_labels, hover_vals)]
            is_target = (t == target)
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels_closed,
                name=t,
                fill="toself" if is_target else "none",
                text=hover_text,
                hoverinfo="text",
                line=dict(
                    color=colors[i % len(colors)],
                    width=2.5 if is_target else 1.5,
                    dash="solid" if is_target else "dash",
                ),
                fillcolor=f"rgba(37, 99, 235, 0.15)" if is_target else None,
                opacity=1.0 if is_target else 0.7,
            ))

        fig.update_layout(
            title="Competitive Positioning",
            polar=dict(radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(l=60, r=60, t=50, b=50),
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return pio.to_html(fig, include_plotlyjs=False, full_html=False)

    # ── Helpers ──

    def _metrics_to_display(self, metrics: dict[str, MetricResult]) -> dict:
        """Convert metrics to display-friendly dict."""
        display = {}
        for mid, result in metrics.items():
            if result.ok:
                display[mid] = {
                    "value": result.value,
                    "flags": result.flags,
                    "metadata": result.metadata,
                }
            else:
                display[mid] = {
                    "value": None,
                    "error": result.error,
                }
        return display

    def _collect_flags(self, metrics: dict[str, MetricResult]) -> list[dict]:
        """Collect all flags from metrics and humanize them.

        Returns list of dicts: [{"label": "High-Quality Growth", "sentiment": "good", "raw": "growth_quality_high_quality"}, ...]
        """
        flags = []
        seen = set()
        for mid, result in metrics.items():
            if result.ok and result.flags:
                for f in result.flags:
                    if f in seen:
                        continue
                    seen.add(f)
                    label, sentiment = FLAG_LABELS.get(f, (None, None))
                    if label is None:
                        # Auto-humanize: replace underscores with spaces, title case
                        label = f.replace("_", " ").title()
                        sentiment = "neutral"
                    element = FLAG_ELEMENT_MAP.get(f, "composite")
                    flags.append({"label": label, "sentiment": sentiment, "raw": f, "element": element})

        # Sort: good first, then bad, then neutral
        order = {"good": 0, "bad": 1, "neutral": 2}
        flags.sort(key=lambda x: order.get(x["sentiment"], 2))
        return flags

    def _make_report_dir(self, ticker: str, company_name: str = "") -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        dir_name = f"{ticker}_{date_str}"
        report_dir = self.output_dir / dir_name
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _copy_annual_reports(self, result, report_dir: Path):
        """Copy downloaded annual report PDFs and extracted text to the report directory."""
        bse_code = result.data.get("metadata", {}).get("bse_code")
        if not bse_code:
            return

        raw_data_dir = Path(__file__).parent.parent / "data_fetcher" / "raw_data"
        ar_source = raw_data_dir / bse_code / "annual_reports"

        if not ar_source.exists():
            return

        ar_dest = report_dir / "annual_reports"
        ar_dest.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src_file in sorted(ar_source.iterdir()):
            if src_file.suffix in (".pdf", ".txt"):
                dest_file = ar_dest / src_file.name
                if not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
                    copied += 1

        if copied:
            logger.info(f"Copied {copied} annual report files to {ar_dest}")

    def _write_json(self, path: Path, data):
        def default_serializer(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return str(obj)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=default_serializer)
