"""Report Generator — HTML dashboards, markdown summaries, and JSON exports."""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.growth import compute_lever_decomposition_table
# The shipped tax and slippage rates, so a lane section handed no assumptions
# still states the numbers the model would actually have applied rather than a
# blank. `lifecycle/friction.py` imports nothing from this layer, so the
# direction is one-way.
from boundless100x.lifecycle.friction import config_from as friction_config_from
# One statement of what the fast lane is called. `lifecycle/advance.py` and
# `lifecycle/lane_view.py` both gate on this constant; a literal here would be a
# third spelling with nothing keeping it in step, and it decides whether a whole
# report section renders. Read from `lifecycle/states.py`, which is where the
# lifecycle's vocabulary lives, rather than from the store that persists it.
from boundless100x.lifecycle.states import RERATING_LANE
# The fallback basis for a usage block with no `cost_basis` key — reports
# written before the field existed are estimates by history, not by a stated
# reading. `cli.py`, `service.py`, `sweep.py` and `orchestrator.py` all default
# to this same constant; a literal `'estimated'` here would be a second
# spelling with nothing keeping it in step with a rename or a third basis value.
from boundless100x.llm_layer.transport import COST_BASIS_ESTIMATED

# What the report *calls* things, and the figures it draws, each in its own
# module. Re-exported here because this is where every one of these names was
# published — `FLAG_ELEMENT_MAP`, `LANE_VERDICT_LABELS`, `FRICTION_*` and the
# rest are imported from `report_generator` by the templates' callers and by the
# test suite, and a caller that reaches for one here is not wrong about where it
# means something.
from boundless100x.output.report_charts import render_charts
from boundless100x.output.report_vocabulary import (
    BREAKEVEN_CAVEAT,
    BREAKEVEN_ESTIMATE,
    BREAKEVEN_STATEMENT,
    ELEMENT_CONFIG,
    FLAG_ELEMENT_MAP,
    FLAG_LABELS,
    FORWARD_SIGNALS,
    FORWARD_SIGNALS_DISCLAIMER,
    FORWARD_SIGNALS_ELEMENT,  # noqa: F401 - re-export; the tests import it here
    FRICTION_BASIS_LABELS,
    FRICTION_NOTE,
    FRICTION_UNAVAILABLE_LABEL,
    LANE_LABELS,
    LANE_VERDICT_LABELS,
    METRIC_DISPLAY_NAMES,
    MOMENTUM_UNAVAILABLE_LABEL,
)

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

    def generate(self, result, formats: list[str] | None = None,
                 lane_context: dict | None = None) -> Path:
        """Generate all requested report formats.

        Args:
            result: AnalysisResult from the service layer.
            formats: List of formats to generate (html, md, json). Default: all.
            lane_context: `lifecycle.lane_view.build_lane_context` output for a
                watchlisted company, or None. **None by default, so every
                existing call site is untouched** — a ticker analysed outside
                the watchlist renders exactly what it rendered before the Lane
                & Friction section existed (KTD9).

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
        dcf_summary = self._build_dcf_summary(result)
        cashflow_quality = self._build_cashflow_quality(result)
        pe_band_summary = self._build_pe_band_summary(result)
        score_drilldown = self._build_score_drilldown(result)
        forward_signals = self._build_forward_signals(result)
        lane_status = self._build_lane_status(lane_context)
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
                dcf_summary=dcf_summary,
                cashflow_quality=cashflow_quality,
                shareholding_data=shareholding_data,
                score_drilldown=score_drilldown,
                element_summaries=element_summaries,
                flags_precomputed=flags,
                forward_signals=forward_signals,
                lane_status=lane_status,
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
                dcf_summary=dcf_summary,
                cashflow_quality=cashflow_quality,
                pe_band_summary=pe_band_summary,
                score_drilldown=score_drilldown,
                element_summaries=element_summaries,
                flags_precomputed=flags,
                forward_signals=forward_signals,
                lane_status=lane_status,
            )
            path = report_dir / f"{result.ticker}_report.md"
            path.write_text(md)
            logger.info(f"Markdown report: {path}")

        return report_dir

    # ── HTML ──

    def _render_charts(self, result) -> dict:
        """The figures this report embeds. Built in `report_charts`.

        A method rather than a direct call at the one call site, because it is
        the seam a test patches to render a report with no charts in it.
        """
        return render_charts(result)

    def _render_html(self, result, charts: dict, growth_decomposition: dict | None = None,
                     executive_summary: dict | None = None,
                     financial_snapshot: list | None = None,
                     dcf_summary: dict | None = None,
                     cashflow_quality: dict | None = None,
                     shareholding_data: list | None = None,
                     score_drilldown: dict | None = None,
                     element_summaries: dict | None = None,
                     flags_precomputed: list | None = None,
                     forward_signals: dict | None = None,
                     lane_status: dict | None = None) -> str:
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
            forward_signals=forward_signals or {},
            lane_status=lane_status or {},
            cost_basis_estimated=COST_BASIS_ESTIMATED,
            errors=result.errors,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    # ── Markdown ──

    def _render_markdown(self, result, growth_decomposition: dict | None = None,
                         executive_summary: dict | None = None,
                         financial_snapshot: list | None = None,
                         shareholding_data: list | None = None,
                         dcf_summary: dict | None = None,
                         cashflow_quality: dict | None = None,
                         pe_band_summary: dict | None = None,
                         score_drilldown: dict | None = None,
                         element_summaries: dict | None = None,
                         flags_precomputed: list | None = None,
                         forward_signals: dict | None = None,
                         lane_status: dict | None = None) -> str:
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
            dcf_summary=dcf_summary or {},
            cashflow_quality=cashflow_quality or {},
            pe_band_summary=pe_band_summary or {},
            score_drilldown=score_drilldown or {},
            element_summaries=element_summaries or {},
            element_config=ELEMENT_CONFIG,
            forward_signals=forward_signals or {},
            lane_status=lane_status or {},
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

        # eligibility.json
        if getattr(result, "eligibility", None):
            self._write_json(report_dir / "eligibility.json", result.eligibility)

        # llm_analysis.json
        if result.llm_analysis:
            self._write_json(report_dir / "llm_analysis.json", result.llm_analysis)

    # ── Charts ──





    def _compute_growth_decomposition(self, result) -> dict | None:
        """The 4-lever decomposition, preferring the one the service computed.

        Recomputing here produced a second table built without the macro config
        and then patched with a P/E the service's copy never received, so the
        report and LLM Pass 2 could disagree about the same company. The
        service's table is authoritative; this only builds one when the caller
        used ReportGenerator directly.
        """
        if getattr(result, "growth_decomposition", None):
            return result.growth_decomposition

        try:
            financials = result.data.get("financials")
            if financials is None or financials.empty:
                return None
            return compute_lever_decomposition_table(result.data)
        except Exception as e:
            logger.warning(f"Growth decomposition failed: {e}")
            return None

    # ── Executive Summary ──

    ELIGIBILITY_BADGES = {
        "eligible": ("100x Candidate", "good",
                     "Clears every eligibility gate"),
        "not_eligible": ("Not a 100x Candidate", "bad",
                         "Fails at least one necessary condition"),
        "indeterminate": ("Eligibility Unknown", "neutral",
                          "A gate could not be evaluated from available data"),
    }

    def _build_eligibility_badge(self, result) -> dict:
        """The 100x verdict, kept separate from the composite it must not dilute."""
        eligibility = getattr(result, "eligibility", None)
        if not eligibility:
            return {}

        label, sentiment, description = self.ELIGIBILITY_BADGES.get(
            eligibility.get("verdict"),
            ("Eligibility Unknown", "neutral", ""),
        )
        gates = eligibility.get("gates", {})
        return {
            "verdict": eligibility.get("verdict"),
            "label": label,
            "sentiment": sentiment,
            "description": description,
            "failed_reasons": [
                gates[g]["reason"] for g in eligibility.get("failed", []) if g in gates
            ],
            "unknown_reasons": [
                gates[g]["reason"] for g in eligibility.get("indeterminate", []) if g in gates
            ],
            "gates": [
                {"id": gid, **detail} for gid, detail in gates.items()
            ],
        }

    @staticmethod
    def _resolve_action(result) -> dict:
        """The action this report may display, guarded at the render boundary.

        Always derived from `llm_analysis`, `eligibility` and `scores` — a
        pre-populated `final_action` is never trusted. Trusting it would make
        the guard only as strong as whoever set the field: a stale decision
        left behind by a rescore, or one attached to a hand-built
        AnalysisResult, would render straight through beside a verdict it
        contradicts. Recomputing is a pure dict operation, so there is no
        reason to accept that risk to save it.
        """
        from boundless100x.action_policy import resolve_for_result

        decision = resolve_for_result(result) or {}

        stored = getattr(result, "final_action", None)
        if stored and stored.get("action") != decision.get("action"):
            logger.warning(
                "Stored final_action (%s) disagrees with the action recomputed "
                "at render time (%s) — rendering the recomputed one. The stored "
                "decision is stale or was not produced by the action policy.",
                stored.get("action"), decision.get("action"),
            )

        return decision

    # ── Forward Signals (Phase 2) ──

    @staticmethod
    def _forward_band(config: dict, value) -> str:
        """Which interpretation band a value falls in."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return ""
        for threshold, label in config["bands"]:
            if value >= threshold:
                return label
        return config["low_label"]

    def _build_forward_signals(self, result) -> dict:
        """The Phase 2 signals, ready to render, or `{}` when there are none.

        Deliberately **not** part of `_build_score_drilldown`. That path skips
        every `weight == 0` metric and reads display names from
        `METRIC_DISPLAY_NAMES`, so reusing it would mean either faking a weight
        these metrics must never have or silently dropping all five.

        Momentum is read from `result.momentum` — the report never reaches into
        score history itself, which would bypass the per-caller history
        redirect the service owns.
        """
        metrics = result.metrics or {}
        present = [mid for mid in FORWARD_SIGNALS if mid in metrics]
        momentum = getattr(result, "momentum", None)

        if not present and not momentum:
            # A ticker analysed before this phase. Nothing to say, and saying
            # "unknown" five times would imply a measurement was attempted.
            return {}

        signals = []
        for metric_id in present:
            config = FORWARD_SIGNALS[metric_id]
            outcome = metrics[metric_id]
            entry = {
                "id": metric_id,
                "name": config["name"],
                "direction": config["direction"],
                "meaning": config["meaning"],
                "available": bool(getattr(outcome, "ok", False)),
                "value": None,
                "formatted": "—",
                "band": "",
                "reason": "",
                "metadata": {},
            }
            if entry["available"]:
                entry["value"] = outcome.value
                entry["metadata"] = outcome.metadata or {}
                try:
                    entry["formatted"] = config["format"].format(outcome.value)
                except (ValueError, TypeError):
                    entry["formatted"] = str(outcome.value)
                # A metric that declared its own band wins: headroom's are
                # owner-editable in YAML params, and a tuned band must beat a
                # default declared in this file.
                entry["band"] = (entry["metadata"].get("band")
                                 or self._forward_band(config, outcome.value))
            else:
                # Indeterminate renders as unknown *with its reason*, the same
                # way an eligibility gate does. A blank cell would read as
                # "nothing to report" rather than "could not be measured".
                entry["reason"] = getattr(outcome, "error", "") or "not available"
            signals.append(entry)

        return {
            "signals": signals,
            "momentum": self._build_momentum(momentum),
            "disclaimer": FORWARD_SIGNALS_DISCLAIMER,
        }

    @staticmethod
    def _build_momentum(momentum: dict | None) -> dict | None:
        """Score trajectory for the reader, with absence kept distinct from zero.

        A zero delta means flat; no delta means unknown. They look identical in
        a table and mean opposite things, so the unavailable case renders with
        its own label and the reason it is unavailable — never as 0.0.
        """
        if not momentum:
            return None

        latest = momentum.get("latest")
        if not latest:
            return {
                "available": False,
                "label": MOMENTUM_UNAVAILABLE_LABEL,
                "reason": momentum.get("reason", ""),
                "composite_delta": None,
                "element_deltas": [],
            }

        return {
            "available": True,
            "label": "Composite trajectory",
            "reason": "",
            "composite_delta": latest["composite_delta"],
            "composite_from": latest["composite_from"],
            "composite_to": latest["composite_to"],
            "from_date": latest["from_date"],
            "to_date": latest["to_date"],
            "span": latest["span"],
            "element_deltas": sorted(
                (
                    {"element": element,
                     "label": ELEMENT_CONFIG.get(element, {}).get("label", element),
                     "delta": delta}
                    for element, delta in (latest.get("element_deltas") or {}).items()
                ),
                key=lambda e: e["label"],
            ),
        }

    # ── Lane & Friction (Phase 3) ──

    def _build_lane_status(self, lane_context: dict | None) -> dict | None:
        """The lane section, ready to render, or None when there is nothing to show.

        The same shape `_build_forward_signals` has, and gated the same way: the
        templates ask `{% if lane_status %}`, so a company with no lane context
        renders byte-identically to the way it did before this section existed.
        That is the whole of KTD9's "unchanged" claim — it is a claim about
        *untracked* reports, never a claim that a tracked core entry shows
        nothing. A tracked core entry does show its lane and state, because
        that is what it has.

        Nothing here is computed from `result`. Everything arrives assembled
        from `lifecycle/lane_view.build_lane_context`, so the report and the
        terminal cannot drift into rendering two different readings of one
        position.
        """
        if not lane_context:
            return None

        lane = lane_context.get("lane")
        state = str(lane_context.get("state") or "")
        gate_result = lane_context.get("lane_gates") or {}
        gates = gate_result.get("gates") or {}
        verdict = gate_result.get("verdict")
        label, sentiment, description = LANE_VERDICT_LABELS.get(verdict, ("", "", ""))

        return {
            "lane": lane,
            "lane_label": LANE_LABELS.get(lane, str(lane)),
            "state": state,
            # The machine word with its underscore softened, not a title-cased
            # rewrite: `exit_review` is the vocabulary every other surface uses
            # for this state, and a report inventing "Exit Review" would make an
            # owner search for a state the CLI never prints.
            "state_label": state.replace("_", " "),
            "as_of": lane_context.get("as_of"),
            "verdict": verdict,
            "verdict_label": label,
            "sentiment": sentiment,
            "description": description,
            "gates": [{"id": gate_id, **detail} for gate_id, detail in gates.items()],
            "failed_reasons": [
                gates[g]["reason"] for g in gate_result.get("failed", []) if g in gates
            ],
            "unknown_reasons": [
                gates[g]["reason"]
                for g in gate_result.get("indeterminate", [])
                if g in gates
            ],
            "catalyst": self._build_catalyst(lane_context.get("catalyst")),
            "friction": self._build_lane_friction(lane_context.get("friction")),
            "breakeven": self._build_breakeven(
                lane, lane_context.get("friction_assumptions")
            ),
        }

    @staticmethod
    def _build_catalyst(catalyst: dict | None) -> dict | None:
        """The recorded catalyst, with its overdue flag as display only.

        §13 keeps the system advisory: an active catalyst whose window has
        passed is worth a reader's attention and is not a transition. The flag
        is computed upstream and rendered here; nothing on this path can move a
        company's state.
        """
        if not catalyst:
            return None
        return {
            "description": catalyst.get("description", ""),
            "expected_by": catalyst.get("expected_by", ""),
            "status": catalyst.get("status", ""),
            "overdue": bool(catalyst.get("overdue")),
        }

    @staticmethod
    def _build_lane_friction(reading: dict | None) -> dict | None:
        """Gross beside net, or the reason there is neither.

        Three outcomes stay distinguishable, and the middle one is the reason
        this is a builder rather than a template expression. **None** means no
        modeled position exists — nothing was ever bought — and the section
        simply carries no friction subsection. **Unavailable** means a position
        exists and could not be priced, and it renders its reason with *no
        numeric field at all*: a rendered zero says the position went nowhere,
        which is a measurement, and an unreadable input is not one. A reading
        renders gross and net together (R5), never one without the other.
        """
        if not reading:
            return None

        basis = reading.get("basis", "estimate")
        if not reading.get("available"):
            return {
                "available": False,
                "basis": basis,
                "label": FRICTION_UNAVAILABLE_LABEL,
                "reason": reading.get("reason", "no reason given"),
            }

        # A reading that claims to be available and is missing half the pair is
        # refused rather than rendered short. Two reasons, and both are real:
        # R5 has no half — a gross figure with no net beside it is precisely
        # the one-without-the-other it forbids — and an `exited` payload is
        # read back off a hand-editable JSON store, so a missing figure would
        # otherwise reach a format filter and take the whole report down with a
        # TypeError.
        missing = [
            name for name in ("gross_return_pct", "net_return_pct")
            if not isinstance(reading.get(name), (int, float))
            or isinstance(reading.get(name), bool)
        ]
        if missing:
            return {
                "available": False,
                "basis": basis,
                "label": FRICTION_UNAVAILABLE_LABEL,
                "reason": (
                    f"the recorded reading is incomplete — {', '.join(missing)} "
                    f"is missing or not a number, and gross and net are only "
                    f"ever shown together"
                ),
            }

        return {
            "available": True,
            "basis": basis,
            "label": FRICTION_BASIS_LABELS.get(basis, FRICTION_BASIS_LABELS["estimate"]),
            "gross_return_pct": reading.get("gross_return_pct"),
            "after_slippage_pct": reading.get("after_slippage_pct"),
            "net_return_pct": reading.get("net_return_pct"),
            "holding_days": reading.get("holding_days"),
            "tax_regime": str(reading.get("tax_regime", "")).upper(),
            "tax_pct": reading.get("tax_pct"),
            "slippage_bps": reading.get("slippage_bps"),
            "entry_date": reading.get("entry_date"),
            "exit_date": reading.get("exit_date"),
            "note": FRICTION_NOTE,
        }

    @staticmethod
    def _build_breakeven(lane, assumptions: dict | None) -> dict | None:
        """§8.2's break-even statement — fast lane only, and never a computed hurdle.

        A core position is held; it does not pay a round trip per cycle, so the
        statement would be about a cost it never bears. See `BREAKEVEN_CAVEAT`
        for why no number is derived from the rates listed here.

        The rates come from the context rather than from this file, because a
        rendered assumption must be the one that was actually applied — an
        owner who edited `friction:` in config and still read the shipped rates
        here would be reading a different model than the one that produced the
        figures beside it.
        """
        if lane != RERATING_LANE:
            return None

        settings = assumptions or friction_config_from(None)
        return {
            "estimate": BREAKEVEN_ESTIMATE,
            "statement": BREAKEVEN_STATEMENT,
            "caveat": BREAKEVEN_CAVEAT,
            "assumptions": [
                f"Short-term capital gains {settings['stcg_pct']}%",
                f"Long-term capital gains {settings['ltcg_pct']}% "
                f"at or beyond {settings['ltcg_holding_days']} days",
                f"Round-trip slippage {settings['slippage_bps']} bps "
                f"(entry and exit together)",
            ],
        }

    def _build_executive_summary(self, result) -> dict:
        """Build executive summary data for the decision dashboard."""
        metadata = result.data.get("metadata", {})
        scores = result.scores
        llm = result.llm_analysis

        summary = {
            "composite_score": scores.get("composite"),
            "element_scores": scores.get("elements", {}),
            "eligibility": self._build_eligibility_badge(result),
            "coverage": scores.get("coverage", {}),
            "company_name": metadata.get("name", result.ticker),
            "sector": metadata.get("sector", "N/A"),
            "market_cap": metadata.get("Market Cap"),
            "has_llm": False,
            "suggested_action": None,
            "action_constraint": {},
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
                # Never the raw p2 action: it has not been checked against the
                # eligibility verdict rendered beside it. Recomputed here when
                # the service did not supply one, so the guard holds for any
                # AnalysisResult handed to the generator.
                decision = self._resolve_action(result)
                summary["has_llm"] = True
                summary["suggested_action"] = decision.get("action")
                summary["action_constraint"] = decision
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
            # Keys must match the values compute_qg_quadrant emits.
            QUADRANT_LABELS = {
                "true_wealth_creator": ("True Wealth Creator", "good", "High quality + High growth"),
                "quality_trap": ("Quality Trap", "bad", "High quality but low growth"),
                "growth_trap": ("Growth Trap", "bad", "High growth but low quality"),
                "wealth_destroyer": ("Wealth Destroyer", "bad", "Low quality + Low growth"),
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

    # ── Feature 4: DCF Visualization ──


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
