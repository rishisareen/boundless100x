"""Report Generator — HTML dashboards, markdown summaries, and JSON exports."""

import json
import logging
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader

from boundless100x.compute_engine.metrics.base import MetricResult

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportGenerator:
    """Generate HTML dashboard, markdown summary, and JSON data exports."""

    def __init__(self, output_dir: str | None = None):
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
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
        report_dir = self._make_report_dir(result.ticker)

        if "json" in formats:
            self._export_json(result, report_dir)
            logger.info(f"JSON exports saved to {report_dir}")

        # Pre-render charts for HTML
        charts = self._render_charts(result)

        if "html" in formats:
            html = self._render_html(result, charts)
            path = report_dir / "sqglp_dashboard.html"
            path.write_text(html)
            logger.info(f"HTML dashboard: {path}")

        if "md" in formats:
            md = self._render_markdown(result)
            path = report_dir / "sqglp_report.md"
            path.write_text(md)
            logger.info(f"Markdown report: {path}")

        return report_dir

    # ── HTML ──

    def _render_html(self, result, charts: dict) -> str:
        template = self.env.get_template("sqglp_report.html.j2")
        return template.render(
            ticker=result.ticker,
            metadata=result.data.get("metadata", {}),
            scores=result.scores,
            metrics=self._metrics_to_display(result.metrics),
            flags=self._collect_flags(result.metrics),
            peers=result.peers,
            comparison=result.comparison,
            llm_analysis=result.llm_analysis,
            radar_chart=charts.get("radar", ""),
            roce_trend_chart=charts.get("roce_trend", ""),
            pe_band_chart=charts.get("pe_band", ""),
            growth_chart=charts.get("growth", ""),
            errors=result.errors,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    # ── Markdown ──

    def _render_markdown(self, result) -> str:
        template = self.env.get_template("sqglp_report.md.j2")
        return template.render(
            ticker=result.ticker,
            metadata=result.data.get("metadata", {}),
            scores=result.scores,
            metrics=self._metrics_to_display(result.metrics),
            flags=self._collect_flags(result.metrics),
            peers=result.peers,
            comparison=result.comparison,
            llm_analysis=result.llm_analysis,
            errors=result.errors,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    # ── JSON Export ──

    def _export_json(self, result, report_dir: Path):
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

        # peer_comparison.json
        if result.comparison:
            # Convert for serialization (values are already native types)
            self._write_json(report_dir / "peer_comparison.json", result.comparison)

        # peer_discovery.json
        if result.peers:
            discovery = {
                "direct_competitors": result.peers.direct_competitors,
                "sector_peers": result.peers.sector_peers,
                "financial_peers": result.peers.financial_peers,
                "peer_data": result.peers.peer_data,
                "discovery_metadata": result.peers.discovery_metadata,
            }
            self._write_json(report_dir / "peer_discovery.json", discovery)

        # llm_analysis.json
        if result.llm_analysis:
            self._write_json(report_dir / "llm_analysis.json", result.llm_analysis)

    # ── Charts ──

    def _render_charts(self, result) -> dict:
        charts = {}

        scores = result.scores.get("elements", {})
        if scores:
            charts["radar"] = self._radar_chart(scores)

        ratios = result.data.get("ratios")
        if ratios is not None and not ratios.empty:
            charts["roce_trend"] = self._roce_trend_chart(ratios)

        price = result.data.get("price")
        metrics = result.metrics
        if price is not None and not price.empty:
            charts["pe_band"] = self._pe_band_chart(price, metrics)

        charts["growth"] = self._growth_chart(metrics)

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

    def _collect_flags(self, metrics: dict[str, MetricResult]) -> list[str]:
        flags = []
        for mid, result in metrics.items():
            if result.ok and result.flags:
                for f in result.flags:
                    flags.append(f"[{mid}] {f}")
        return flags

    def _make_report_dir(self, ticker: str) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        report_dir = self.output_dir / f"{ticker}_{date_str}"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _write_json(self, path: Path, data):
        def default_serializer(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return str(obj)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=default_serializer)
