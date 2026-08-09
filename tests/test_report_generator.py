"""Report generation regression tests.

The peer-comparison removal left `_build_sector_context` reading a deleted
`AnalysisResult.comparison` field, which raised AttributeError for every
ticker. These tests pin the generator against that class of regression.

`TestTheLegacyReportIsFrozen` at the foot of this file is the other kind of
pin, and it is a freeze rather than an assertion: **the HTML and Markdown the
generator produces today, byte for byte after normalisation.** R16 says the
current report keeps being generated unchanged while a second, clearer report
is added beside it, and "unchanged" is not a claim a targeted assertion can
make — a section that quietly stopped rendering, a number that gained a digit,
a label that lost a word all pass every test in this file that names something
specific. Only a whole-document comparison sees them.

Regenerating the goldens is a deliberate act, never a way to make a red test
green. When a change to the legacy report *is* intended:

    venv/bin/python -m pytest tests/test_report_generator.py -q  # read the diff
    venv/bin/python -c "from tests.test_report_generator import write_goldens; write_goldens()"
"""

from pathlib import Path

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import friction as friction_module
from boundless100x.output.report_generator import ReportGenerator
from tests.conftest import make_result
# The two things that legitimately differ between two identical runs — the
# generation timestamp and the uuid Plotly stamps on every chart div — are
# already solved there, including the part that is easy to get wrong: an empty
# chart container must not normalise to the same string as a full one, or a
# report that silently lost every figure would compare equal.
from tests.test_report_lane_status import normalise

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "boundless100x"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

GOLDEN_HTML = GOLDEN_DIR / "pre_report_clarity_dashboard.html"
GOLDEN_MD = GOLDEN_DIR / "pre_report_clarity_report.md"


@pytest.fixture
def generator(tmp_path):
    return ReportGenerator(output_dir=str(tmp_path))


def test_generate_completes_for_current_analysis_result(generator, analysis_result):
    report_dir = generator.generate(analysis_result, formats=["html", "md", "json"])

    assert report_dir.exists()
    assert (report_dir / f"{analysis_result.ticker}_dashboard.html").exists()
    assert (report_dir / "raw_metrics.json").exists()


def test_generate_html_only_completes(generator, analysis_result):
    report_dir = generator.generate(analysis_result, formats=["html"])
    html = (report_dir / f"{analysis_result.ticker}_dashboard.html").read_text()

    assert "Test Co" in html


def test_generate_without_llm_analysis_completes(generator):
    """--no-llm runs leave llm_analysis as None; the report must still render."""
    result = make_result()
    assert result.llm_analysis is None

    report_dir = generator.generate(result, formats=["html", "md"])

    assert (report_dir / f"{result.ticker}_dashboard.html").exists()


def test_no_references_to_removed_comparison_field():
    """`AnalysisResult.comparison` was deleted with the peer feature."""
    offenders = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        if "result.comparison" in path.read_text()
    ]

    assert offenders == []


def test_no_references_to_removed_peers_field():
    offenders = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        if "result.peers" in path.read_text()
    ]

    assert offenders == []


class TestTheReportLayerSplit:
    """Where the report layer's three modules draw their lines.

    `report_generator.py` had grown past two thousand lines, and two of the
    things inside it were not report *sections*: four hundred lines of display
    vocabulary that grows every time a metric or flag is added, and four
    hundred lines of Plotly trace assembly. Both moved out. What stayed is the
    class that decides what a report contains.

    The split is pinned rather than trusted because both halves have a way of
    creeping back — a new label is easiest to type next to the section that
    renders it, and a new chart next to the builder that calls it.
    """

    def source(self, name: str) -> str:
        import boundless100x.output as package

        return (Path(package.__file__).parent / name).read_text()

    def test_the_vocabulary_module_is_data_only(self):
        """No logic, no rendering, no I/O — only what a reader sees. A function
        here would be a report section that had drifted into the dictionary of
        names for report sections."""
        import ast

        tree = ast.parse(self.source("report_vocabulary.py"))
        definitions = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        assert definitions == []

    def test_no_plotly_import_survives_in_the_generator(self):
        """The charts moved wholesale. A `go` or `pio` left behind would mean a
        figure still being assembled among the sections."""
        source = self.source("report_generator.py")

        assert "plotly" not in source

    def test_every_chart_builder_is_free_of_self(self):
        """Why the extraction was mechanical rather than a judgement call: they
        were module functions wearing method clothing, and the class was their
        namespace rather than their owner. A builder that starts reading
        generator state would have to come back."""
        import ast

        tree = ast.parse(self.source("report_charts.py"))
        signatures = [
            node.args.args[0].arg
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.args.args
        ]

        assert "self" not in signatures

    def test_the_names_the_suite_imports_still_resolve_from_the_generator(self):
        """Re-exported deliberately. These are imported from
        `report_generator` by the tests and by KTD6's registry-derived flag
        check; the split must not move them out from under anybody."""
        from boundless100x.output import report_generator, report_vocabulary

        for name in ("FLAG_LABELS", "FLAG_ELEMENT_MAP", "FORWARD_SIGNALS",
                     "FORWARD_SIGNALS_DISCLAIMER", "FORWARD_SIGNALS_ELEMENT",
                     "FRICTION_UNAVAILABLE_LABEL", "LANE_VERDICT_LABELS",
                     "METRIC_DISPLAY_NAMES", "ELEMENT_CONFIG"):
            assert getattr(report_generator, name) is getattr(
                report_vocabulary, name
            ), name

    def test_the_generator_still_renders_every_format(self, tmp_path):
        """The whole point, asserted past the module boundaries: a real render
        still produces all three artefacts with real figures in the HTML."""
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        report_dir = Path(
            ReportGenerator(output_dir=str(tmp_path)).generate(
                make_result("TEST"), formats=["html", "md", "json"]
            )
        )
        html = (report_dir / "TEST_dashboard.html").read_text()

        assert (report_dir / "TEST_report.md").exists()
        assert (report_dir / "scores.json").exists()
        assert html.count("Plotly.newPlot") >= 3


# ──────────────────────────────────────────────────────────────────────────
# The frozen legacy report
#
# A golden is only worth its green if the fixture behind it reaches the report.
# `make_result()` on its own renders a report with no LLM view, no eligibility
# verdict, no drill-down, no DCF section, no forward signals and no lane — over
# half the template, unexercised, and a change to any of it would land under a
# passing golden. So the fixture below deliberately lights up every section the
# generator can build, including the paths that are easiest to break quietly:
# an errored metric, an indeterminate gate, an indeterminate forward signal, a
# capped action, and a run that finished with warnings.
#
# Everything here is a literal. Nothing is read off `raw_data/`, nothing is
# computed by the engine, and no value is derived from another test module's
# fixtures — a golden whose inputs move for reasons unrelated to the report is
# a golden nobody trusts.
# ──────────────────────────────────────────────────────────────────────────


def rich_metrics() -> dict[str, MetricResult]:
    """Metrics across all six elements, plus the zero-weight forward signals.

    Flags are drawn from `FLAG_LABELS` rather than invented, so the report
    renders real labels with real sentiments in all three buckets (strength,
    concern, context) — an auto-humanised unknown flag would pin the fallback
    path instead of the registered one.
    """
    return {
        # ── Size ──
        "market_cap": MetricResult(value=5000.0, flags=["mid_cap"]),
        "institutional_holding": MetricResult(
            value=13.0, flags=["low_institutional_ownership"]
        ),
        "analyst_coverage": MetricResult(value=12.0),
        # ── Quality — Business ──
        "roce_5yr_avg": MetricResult(value=22.0, flags=["consistently_high_roce"]),
        "operating_margin_5yr": MetricResult(
            value=25.0, flags=["high_operating_margin"]
        ),
        "debt_equity": MetricResult(value=0.02, flags=["virtually_debt_free"]),
        # A metric that could not be computed. `ok` is False, so it renders
        # through the error branch of `_metrics_to_display` and contributes
        # nothing to the drill-down — the quiet path a golden exists to hold.
        "interest_coverage": MetricResult(
            error="interest expense is zero in 4 of 10 years"
        ),
        # ── Quality — Management ──
        "promoter_holding_trend": MetricResult(value=0.0),
        "equity_dilution": MetricResult(value=0.4, flags=["minimal_dilution"]),
        # ── Growth ──
        "revenue_cagr_5yr": MetricResult(value=20.0),
        "pat_cagr_5yr": MetricResult(value=25.0),
        "growth_quality_grade": MetricResult(
            value="moderate", flags=["growth_quality_moderate"]
        ),
        # ── Longevity ──
        "fcf_consistency": MetricResult(value=9.0, flags=["consistent_fcf_generator"]),
        "cap_proxy": MetricResult(value=6.0, flags=["moderate_moat_cap"]),
        # ── Price ──
        "pe_ttm": MetricResult(value=30.0),
        # Carries the band metadata `compute_pe_percentile` always returns.
        # Without it the fixture rendered a valuation range the real metric
        # could never produce, and the golden froze that fiction — the
        # percentile below sits inside this band, which is the whole point.
        "pe_vs_historical": MetricResult(
            value=82.0,
            flags=["pe_above_historical_75th"],
            raw_series=[14.2, 18.6, 22.4, 26.1, 27.8, 29.3, 31.5, 33.0, 35.4, 38.2],
            metadata={
                "years_used": 10,
                "pe_min": 14.2,
                "pe_max": 38.2,
                "pe_median": 28.55,
                "current_pe": 30.0,
                "price_basis": "raw_close",
            },
        ),
        "dcf_margin_of_safety": MetricResult(
            value=-12.5,
            flags=["dcf_overvalued"],
            metadata={
                "intrinsic_per_share": 262.5,
                "current_price": 300.0,
                "fcf_growth_assumed": 14.0,
            },
        ),
        "reverse_dcf_growth": MetricResult(
            value=31.0,
            flags=["reverse_dcf_overpriced"],
            metadata={"actual_cagr": 25.0},
        ),
        # ── Composite (display only) ──
        "quality_growth_quadrant": MetricResult(
            value="true_wealth_creator",
            metadata={"avg_roce": 22.0, "pat_cagr": 25.0},
        ),
        # ── Forward signals (Phase 2, zero weight) ──
        "rerating_headroom": MetricResult(
            value=42.0,
            flags=["rerating_headroom_favourable"],
            metadata={"band": "favourable", "justified_multiple": 43.0,
                      "current_multiple": 30.3},
        ),
        "promises_kept_ratio": MetricResult(value=75.0, metadata={"kept": 3, "due": 4}),
        # Indeterminate: renders as unknown *with its reason*, never as a zero.
        "tam_runway": MetricResult(
            error="no numeric addressable-market figure in the MD&A"
        ),
        "quarterly_momentum": MetricResult(
            value=-5.4,
            flags=["quarterly_growth_decelerating"],
            metadata={"yoy_pct": [30.0, 20.0, 12.0]},
        ),
    }


def rich_scores() -> dict:
    """A scorer payload in the shape `SQGLPScorer.score` returns.

    `low_data_coverage` and a sub-1.0 composite coverage are both set, so the
    executive summary renders the "scored on N% of metric weight" line and the
    action policy has a coverage constraint to state alongside the eligibility
    one.
    """
    return {
        "composite": 6.52,
        "elements": {
            "size": 5.0, "quality_business": 7.4, "quality_management": 6.0,
            "growth": 7.2, "longevity": 6.1, "price": 4.3,
        },
        "flags": ["low_data_coverage"],
        "coverage": {
            "composite": 0.82,
            "elements": {
                "size": 0.9, "quality_business": 0.75, "quality_management": 0.8,
                "growth": 0.85, "longevity": 0.8, "price": 0.8,
            },
            "unscored": ["interest_coverage"],
        },
        "details": {
            "market_cap": {"value": 5000.0, "score": 0.5, "weight": 0.06,
                           "flags": ["mid_cap"]},
            "institutional_holding": {"value": 13.0, "score": 0.5, "weight": 0.04,
                                      "flags": ["low_institutional_ownership"]},
            "roce_5yr_avg": {"value": 22.0, "score": 0.82, "weight": 0.12,
                             "flags": ["consistently_high_roce"]},
            "operating_margin_5yr": {"value": 25.0, "score": 0.7, "weight": 0.05,
                                     "flags": ["high_operating_margin"]},
            "debt_equity": {"value": 0.02, "score": 0.95, "weight": 0.03,
                            "flags": ["virtually_debt_free"]},
            "interest_coverage": {"value": None, "score": None, "weight": 0,
                                  "error": "interest expense is zero in 4 of 10 years"},
            "promoter_holding_trend": {"value": 0.0, "score": 0.5, "weight": 0.05,
                                       "flags": []},
            "equity_dilution": {"value": 0.4, "score": 0.9, "weight": 0.05,
                                "flags": ["minimal_dilution"]},
            "revenue_cagr_5yr": {"value": 20.0, "score": 0.75, "weight": 0.1,
                                 "flags": []},
            "pat_cagr_5yr": {"value": 25.0, "score": 0.85, "weight": 0.1,
                             "flags": []},
            "growth_quality_grade": {"value": "moderate", "score": 0.5,
                                     "weight": 0.05, "flags": ["growth_quality_moderate"]},
            "fcf_consistency": {"value": 9.0, "score": 0.9, "weight": 0.1,
                                "flags": ["consistent_fcf_generator"]},
            "cap_proxy": {"value": 6.0, "score": 0.6, "weight": 0.1, "flags": []},
            "pe_ttm": {"value": 30.0, "score": 0.35, "weight": 0.06, "flags": []},
            "pe_vs_historical": {"value": 82.0, "score": 0.2, "weight": 0.04,
                                 "flags": ["pe_above_historical_75th"]},
            "dcf_margin_of_safety": {"value": -12.5, "score": 0.3, "weight": 0.05,
                                     "flags": ["dcf_overvalued"]},
            # Zero weight: skipped by the drill-down by construction, and here
            # so the golden would notice if it ever stopped being skipped.
            "rerating_headroom": {"value": 42.0, "score": None, "weight": 0,
                                  "flags": ["rerating_headroom_favourable"]},
        },
    }


def rich_eligibility() -> dict:
    """A 100x verdict in `EligibilityEvaluator.evaluate`'s shape.

    `not_eligible` with one failed gate *and* one indeterminate gate, because
    the two render differently (fail / unknown) and are read by two different
    lists in the badge builder.
    """
    gates = {
        "small_enough_to_multiply": {
            "label": "Small enough to multiply", "rationale": "",
            "passed": False,
            "reason": ("Small enough to multiply not met: market_cap 5000.00 "
                       "lte 3000"),
            "conditions": [],
        },
        "reinvestment_runway": {
            "label": "Reinvestment runway", "rationale": "",
            "passed": True,
            "reason": "Reinvestment runway met: roce_5yr_avg 22.00 gte 18",
            "conditions": [],
        },
        "market_not_already_pricing_it": {
            "label": "Market not already pricing it", "rationale": "",
            "passed": None,
            "reason": ("Market not already pricing it indeterminate: "
                       "reverse_dcf_growth is unavailable"),
            "conditions": [],
        },
    }
    return {
        "eligible": False,
        "verdict": "not_eligible",
        "gates": gates,
        "failed": ["small_enough_to_multiply"],
        "indeterminate": ["market_not_already_pricing_it"],
    }


def rich_llm_analysis() -> dict:
    """Both passes plus a usage block.

    `suggested_action: buy` against a `not_eligible` verdict is the case the
    action guard exists for: the report must show WATCHLIST with the model's
    own answer preserved beside it, and the golden is what says it still does.
    """
    return {
        "pass1": {
            "management_integrity_score": 8,
            "management_competence_score": 7,
            "growth_mindset_score": 8,
            "moat_type": "Brand + distribution",
            "moat_strength": 7,
            "red_flags": [
                "Related-party sales rose to 4% of revenue without explanation.",
                "Capex guidance was revised twice in the same financial year.",
            ],
        },
        "pass2": {
            "suggested_action": "buy",
            "conviction_level": "medium",
            "target_holding_period": "5-7 years",
            "thesis": (
                "A capital-light compounder with a durable distribution edge, "
                "priced for the growth it has already delivered."
            ),
            "bull_case": (
                "Category penetration is early and the incumbent's pricing "
                "power has held through two input-cost cycles."
            ),
            "bear_case": (
                "The multiple assumes the current growth rate persists for a "
                "decade; a single flat year would re-rate it hard."
            ),
            "kill_the_thesis": [
                "RoCE falls below 15% for two consecutive years.",
                "Promoter holding drops below 50%.",
                "Receivable days exceed 90 for three consecutive quarters.",
            ],
            "key_monitorables": [
                "Quarterly OPM against the 24% base.",
                "Capacity commissioning at the new plant.",
            ],
            "reasoning": (
                "The composite rests on a strong quality reading and a weak "
                "price one, which is the usual shape for this business at this "
                "point in its cycle."
            ),
        },
        "usage": {
            "total_tokens": 34820,
            "total_cached_input_tokens": 29110,
            "estimated_cost_usd": 0.2841,
            "cost_basis": "actual",
            "provider": "claude_cli",
            "total_seconds": 41.2,
            "failed_calls": 1,
        },
    }


def rich_momentum() -> dict:
    """Score trajectory in `trajectory.momentum`'s shape — the available case."""
    return {
        "ticker": "TEST",
        "status": "ok",
        "reason": "",
        "latest": {
            "from_date": "2026-01-01", "to_date": "2026-04-01",
            "interval_days": 90, "span": "90 days",
            "composite_from": 6.12, "composite_to": 6.52, "composite_delta": 0.4,
            "element_deltas": {"growth": 0.4, "price": -0.2, "longevity": 0.0},
            "synthetic": False, "config_hash": "abc123abc123",
        },
        "regimes": [],
    }


def rich_lane_context() -> dict:
    """A fast-lane watchlist entry, in `lane_view.build_lane_context`'s shape.

    Written out here rather than imported from `test_report_lane_status`: that
    file's fixtures exist to be varied per test, and a golden whose inputs move
    when an unrelated test is edited fails for reasons that have nothing to do
    with the report. The friction assumptions come from `config_from(None)`,
    which is the shipped-defaults path — the same numbers a run with no
    `friction:` block in config would render.
    """
    def gate(label, passed, reason):
        return {"label": label, "rationale": "", "passed": passed,
                "reason": reason, "conditions": []}

    gates = {
        "quality_floor": gate(
            "Quality floor", True,
            "Quality floor met: score composite 6.52 gte 5.5"),
        "valuation_discount": gate(
            "Valuation discount", True,
            "Valuation discount met: pe_vs_historical 82.00 lte 90"),
        "growth_intact": gate(
            "Growth intact", True,
            "Growth intact met: ttm_growth_vs_cagr 4.00 gte 0"),
        "institutional_accumulation": gate(
            "Institutional accumulation", None,
            "Institutional accumulation indeterminate: "
            "institutional_accumulation_streak is unavailable"),
        "catalyst_identified": gate(
            "Catalyst identified", True,
            "Catalyst identified met: catalyst is active"),
        "liquidity_floor": gate(
            "Liquidity floor", False,
            "Liquidity floor not met: daily_turnover_ratio 0.01 gte 0.02"),
    }
    return {
        "lane": "rerating",
        "state": "probe",
        "as_of": "2026-08-07",
        "catalyst": {
            "description": "Demerger of the packaging arm",
            "expected_by": "2026-01-31",
            "status": "active",
            "overdue": True,
        },
        "lane_gates": {
            "verdict": "not_qualified",
            "qualifies": False,
            "gates": gates,
            "failed": ["liquidity_floor"],
            "indeterminate": ["institutional_accumulation"],
        },
        "friction": {
            "available": True,
            "basis": friction_module.BASIS_ESTIMATE,
            "gross_return_pct": 48.0,
            "after_slippage_pct": 47.0,
            "net_return_pct": 41.125,
            "holding_days": 420,
            "tax_regime": "ltcg",
            "tax_pct": 12.5,
            "taxed": True,
            "ltcg_holding_days": 365,
            "slippage_bps": 100,
            "entry_date": "2025-06-10",
            "exit_date": "2026-08-04",
            "price_series": "adj_close",
        },
        "friction_assumptions": friction_module.config_from(None),
    }


def rich_result():
    """An AnalysisResult that reaches every section the legacy report renders."""
    result = make_result(ticker="TEST", metrics=rich_metrics(), scores=rich_scores())
    # `make_data` builds the metadata, so the fields the header and the analyst
    # cross-check read are set on the built dict rather than passed through.
    result.data["metadata"]["sector"] = "Chemicals"
    result.data["analyst_coverage"] = {
        "count": 12, "avg_target": 420.0, "consensus": "Buy",
    }
    # Deliberately no `bse_code`: with one, `_copy_annual_reports` reaches into
    # the real gitignored `raw_data/` and copies whatever that machine happens
    # to hold, which would make the goldens differ between checkouts.
    assert "bse_code" not in result.data["metadata"]
    result.eligibility = rich_eligibility()
    result.llm_analysis = rich_llm_analysis()
    result.momentum = rich_momentum()
    # A run that finished with a warning: the Appendix's errors block is the
    # only place this reaches a reader.
    result.errors = ["shareholding: BSE returned no data for Sep 2025"]
    return result


def render_legacy(output_dir) -> tuple[str, str]:
    """The legacy HTML and Markdown, normalised, from the full `generate` path.

    Through `generate` rather than the two private renderers, because the
    format gating, the shared builders and the order they run in are all part
    of what is being frozen.
    """
    report_dir = ReportGenerator(output_dir=str(output_dir)).generate(
        rich_result(), formats=["html", "md", "json"],
        lane_context=rich_lane_context(),
    )
    return (
        normalise((report_dir / "TEST_dashboard.html").read_text()),
        normalise((report_dir / "TEST_report.md").read_text()),
    )


def write_goldens() -> None:
    """Regenerate both goldens. Deliberate act — see this module's docstring."""
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        html, md = render_legacy(Path(scratch))

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_HTML.write_text(html)
    GOLDEN_MD.write_text(md)


class TestTheLegacyReportIsFrozen:
    """R16: the current report keeps being generated, unchanged in content.

    Two later units modify `report_generator.py` to add a second report beside
    this one. These two comparisons are what say they did not disturb it.
    """

    def test_the_markdown_report_matches_its_golden(self, tmp_path):
        _, md = render_legacy(tmp_path)

        assert md == GOLDEN_MD.read_text(), (
            "the legacy Markdown report has changed. R16 requires it to keep "
            "rendering unchanged while the new report is added beside it — "
            "regenerate the golden only for a change you meant to make."
        )

    def test_the_html_dashboard_matches_its_golden(self, tmp_path):
        html, _ = render_legacy(tmp_path)

        assert html == GOLDEN_HTML.read_text(), (
            "the legacy HTML dashboard has changed. R16 requires it to keep "
            "rendering unchanged while the new report is added beside it — "
            "regenerate the golden only for a change you meant to make."
        )

    def test_every_json_side_export_lands_beside_them(self, tmp_path):
        """The third format, which no golden covers because it is not prose.

        `eligibility.json` and `llm_analysis.json` are written only when the
        result carries those fields, so the minimal fixtures above never reach
        them — the same blind spot the rich fixture exists to close.
        """
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            rich_result(), formats=["json"], lane_context=rich_lane_context()
        )

        for name in ("raw_metrics.json", "scores.json", "eligibility.json",
                     "llm_analysis.json", "growth_decomposition.json"):
            assert (report_dir / name).exists(), name

    def test_two_runs_of_the_same_fixture_normalise_identically(self, tmp_path):
        """The property a golden rests on, asserted rather than assumed.

        If anything in the report varied run to run past the timestamp and the
        Plotly uuids, the comparisons above would be flaky rather than wrong,
        and a flaky golden gets deleted instead of read.
        """
        first = render_legacy(tmp_path / "a")
        second = render_legacy(tmp_path / "b")

        assert first == second

    @pytest.mark.parametrize("marker", [
        # A section from each part of the report, so normalisation eating one
        # of them is a failure here rather than a silent hole in the goldens.
        "## Executive Summary",
        "**100x Eligibility: Not a 100x Candidate**",
        "| Small enough to multiply | **fail** |",
        "| Market not already pricing it | unknown |",
        "**Action capped to WATCHLIST**",
        # The drill-down now labels metrics from the registry rather than the
        # hand-maintained table, so this row reads "RoCE (5yr Avg)".
        "| RoCE (5yr Avg) | 22.0 | 82% | 12% |",
        "### Cash Flow Quality",
        "### Shareholding Trend",
        "### The 4-Lever Earnings Decomposition",
        "### DCF Summary",
        "### Valuation Reality Check",
        "### Historical Valuation Range",
        "## Forward Signals",
        "### Score Trajectory",
        "## Lane & Friction",
        "### Friction",
        "## Investment Thesis",
        "### Qualitative Assessment",
        "### 10-Year Financial Snapshot",
        "### Warnings & Errors",
    ])
    def test_the_markdown_golden_carries_every_section(self, marker):
        """A golden that normalised away its content would still compare equal
        to itself. These are the sections the fixture was built to reach."""
        assert marker in GOLDEN_MD.read_text()

    @pytest.mark.parametrize("marker", [
        "<h2>Forward Signals</h2>",
        "<h2>Lane &amp; Friction</h2>",
        "<h2>Investment Thesis</h2>",
        "<h2>Appendix</h2>",
        "Not a 100x Candidate",
        "True Wealth Creator",
        "Capped from BUY",
        "(+29,110 cached)",
    ])
    def test_the_html_golden_carries_every_section(self, marker):
        assert marker in GOLDEN_HTML.read_text()

    def test_the_html_golden_still_holds_its_charts(self):
        """`normalise` replaces a chart's payload but records that there was
        one. A report that lost every figure must not compare equal to this."""
        from tests.test_report_lane_status import CHART_EMPTY, CHART_RENDERED

        golden = GOLDEN_HTML.read_text()

        assert golden.count(CHART_RENDERED) >= 3
        assert CHART_EMPTY not in golden

    def test_the_numbers_survive_normalisation(self):
        """Only the timestamp and the chart uuids are stripped. If a figure or
        a label were normalised away too, the goldens would pass through a
        change to the very thing they exist to hold."""
        md = GOLDEN_MD.read_text()

        assert "6.52" in md          # the composite
        assert "82% of metric weight" in md  # coverage
        assert "₹262" in md          # DCF intrinsic value
        assert "+42%" in md          # a forward signal's reading
        assert "+48.0%" in md        # the gross friction figure
        assert "<generated>" in md   # and the one thing that is stripped


# ──────────────────────────────────────────────────────────────────────────
# The research note (U10)
#
# The fourth format, beside the three frozen above. Everything below asserts
# something about the *new* report; that the old two are untouched is what
# `TestTheLegacyReportIsFrozen` already says, and the two claims are kept in
# separate classes so a failure names which of them broke.
# ──────────────────────────────────────────────────────────────────────────


def clarity_result():
    """The golden fixture, moved into the sector that makes it interesting.

    `rich_result()` fires no expansion trigger — no metric scores exactly zero
    and "Chemicals" is not a sector anyone has declared applicability for — so
    it renders AE5's all-collapsed shape and says nothing about AE1's. Setting
    the sector to Finance turns on the one trigger the plan was written about:
    asset turnover, the equity multiplier and free cash flow stop measuring
    anything, and three sections earn their space.

    It mutates a fresh `rich_result()` rather than editing the fixture, because
    the fixture is the goldens' input and moving it would move them.
    """
    result = rich_result()
    result.data["metadata"]["sector"] = "Finance"
    return result


def render_note(output_dir, result=None) -> tuple[str, str]:
    """The note's two surfaces, from the full `generate` path.

    Through `generate` for the reason `render_legacy` goes through it: the
    format gating and the shared builders are part of what is being asserted.
    """
    report_dir = ReportGenerator(output_dir=str(output_dir)).generate(
        result if result is not None else clarity_result(),
        formats=["clarity"],
    )
    ticker = (result or clarity_result()).ticker
    return (
        (report_dir / f"{ticker}_note.html").read_text(),
        (report_dir / f"{ticker}_note.md").read_text(),
    )


def visible_text(markup: str) -> str:
    """What a reader of the HTML actually sees, with the markup taken off.

    R14 is a claim about *content*, so a comparison that matched strings would
    be asserting the wrong thing — the two surfaces are supposed to differ in
    markup. Tags come off, entities are unescaped, and whitespace collapses, so
    what is left is the words. Anything a template wrapped in `<strong>` or in
    `**` reads the same afterwards.
    """
    import html as html_module
    import re as re_module

    text = re_module.sub(r"<script.*?</script>|<style.*?</style>", " ", markup,
                         flags=re_module.S)
    text = re_module.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    return re_module.sub(r"\s+", " ", text)


def plain_text(markdown: str) -> str:
    """The Markdown's words, on the same footing as `visible_text`'s.

    Only whitespace is collapsed. Emphasis markers are deliberately left
    alone — `**Cheap PE**` still *contains* "Cheap PE", so stripping them buys
    nothing, and stripping asterisks would also eat the ones inside a
    declaration that reads "profit *per share*", turning a content match into a
    punctuation match.
    """
    import re as re_module

    return re_module.sub(r"\s+", " ", markdown)


def headings(html: str, markdown: str) -> tuple[list[str], list[str]]:
    """The section headings each surface actually opened, in order.

    Extracted rather than searched for, because a title can legitimately appear
    in running text — the opening line names the sections that expanded — and a
    `find()` on the whole document would report that mention as the section.
    """
    import html as html_module
    import re as re_module

    from_html = [
        html_module.unescape(match).strip()
        for match in re_module.findall(r"<h2>(.*?)</h2>", html, flags=re_module.S)
    ]
    from_md = [
        line[3:].strip() for line in markdown.splitlines()
        if line.startswith("## ")
    ]
    return from_html, from_md


def content_of(context: dict) -> list[str]:
    """Every string the two surfaces were given, walked off the model itself.

    This is what makes the R14 assertion below a content comparison rather than
    a string comparison: the expectations come from the `Section` objects both
    renderers received, not from either rendering.
    """
    from boundless100x.output import report_surfaces as rs

    pieces: list[str] = []
    sections = [context["lead"], *context["sections"]]
    if context["unscored"]:
        sections.append(context["unscored"])

    for section in sections:
        pieces.append(section.title)
        headline, body, qualifier = rs.reading_line(section.reading)
        pieces.extend(p for p in (headline, body, qualifier) if p)
        if not section.expanded:
            continue
        for finding in section.findings:
            pieces.extend(p for p in rs.finding_line(finding) if p)
        for row in section.rows:
            pieces.extend(rs.metric_cells(row))
        for unknown in section.unknowns:
            pieces.extend((unknown.subject, unknown.reason))
        for caveat in section.caveats:
            pieces.append(rs.caveat_line(caveat))

    appendix = context["appendix"]
    for finding in appendix["signals"]:
        pieces.extend(p for p in rs.finding_line(finding) if p)
    for unknown in appendix["signal_unknowns"]:
        pieces.extend((unknown.subject, unknown.reason))
    for disclosure in appendix["disclosures"]:
        pieces.extend((disclosure.title, disclosure.body))
    for caveat in appendix["caveats"]:
        pieces.append(rs.caveat_line(caveat))

    import re as re_module

    return [re_module.sub(r"\s+", " ", p).strip() for p in pieces if str(p).strip()]


class TestOneRunProducesBothReports:
    """R16 from the other side: the note is *added*, and the old two stay."""

    def test_a_single_generate_call_writes_all_four_artefacts(self, tmp_path):
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            clarity_result(), formats=["html", "md", "clarity", "json"],
        )

        for name in ("TEST_dashboard.html", "TEST_report.md",
                     "TEST_note.html", "TEST_note.md", "scores.json"):
            assert (report_dir / name).exists(), name

    def test_the_note_joins_the_default_format_list(self, tmp_path):
        """`formats=None` means "everything this generator produces".

        A default that listed three of four reports would be the silent
        omission this whole plan is about — and every production caller passes
        `formats=` explicitly, so joining it changes nothing that runs today.
        """
        from boundless100x.output.report_generator import CLARITY, DEFAULT_FORMATS

        assert CLARITY in DEFAULT_FORMATS

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            clarity_result()
        )

        assert (report_dir / "TEST_note.md").exists()

    def test_the_cli_asks_for_the_note_explicitly(self):
        """KTD3's footnote: `cli.py` passes `formats=` and never sees the
        default above, so the token has to be in its option string too."""
        from boundless100x.output.report_generator import CLARITY

        source = (Path(__file__).resolve().parent.parent
                  / "boundless100x" / "cli.py").read_text()

        assert f'"html,md,{CLARITY},json"' in source

    def test_asking_only_for_the_note_leaves_the_legacy_files_unwritten(self, tmp_path):
        """The new block gates on its own token and nothing else."""
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            clarity_result(), formats=["clarity"],
        )

        assert (report_dir / "TEST_note.md").exists()
        assert not (report_dir / "TEST_report.md").exists()
        assert not (report_dir / "TEST_dashboard.html").exists()

    def test_the_two_reports_agree_on_every_element_score(self, tmp_path):
        """One of the plan's three success criteria, asserted rather than hoped.

        The note reads its figures off the same `result` the dashboard does, so
        a disagreement would mean one of them recomputed something.
        """
        result = clarity_result()
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            result, formats=["md", "clarity"],
        )
        legacy = (report_dir / "TEST_report.md").read_text()
        note = (report_dir / "TEST_note.md").read_text()

        for score in result.scores["elements"].values():
            assert f"{score:.1f}/10" in legacy
            assert f"{score:.1f} / 10" in note
        assert str(result.scores["composite"]) in legacy
        assert f"{result.scores['composite']:.1f} / 10" in note


class TestBothSurfacesCarryTheSameContent:
    """R14, proven against the model rather than against either rendering."""

    def test_every_string_the_model_carried_reaches_both_surfaces(self, tmp_path):
        generator = ReportGenerator(output_dir=str(tmp_path))
        result = clarity_result()
        context = generator._clarity_context(result)
        html, md = render_note(tmp_path, result)

        seen_html, seen_md = visible_text(html), plain_text(md)
        missing = [
            piece for piece in content_of(context)
            if piece not in seen_html or piece not in seen_md
        ]

        assert missing == []

    def test_the_two_surfaces_open_the_same_sections(self, tmp_path):
        """"A section present in one surface is present in all three."

        Compared as a set of titles, in order, because that is the claim — not
        that the headings are spelled the same way in HTML and Markdown.
        """
        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        html, md = render_note(tmp_path, result)

        titles = [context["lead"].title, *[s.title for s in context["sections"]]]
        if context["unscored"]:
            titles.append(context["unscored"].title)
        titles.append("Appendix")

        from_html, from_md = headings(html, md)

        assert from_html == titles
        assert from_md == titles

    def test_both_surfaces_render_every_member_of_the_closed_set(self):
        """R13's mechanism, asserted where it can be read as the requirement.

        The decorator already refuses an incomplete renderer at class-creation
        time; this is the statement that the two surfaces R14 names for a
        report have both landed and are both complete.
        """
        from boundless100x.output import report_components as rc
        from boundless100x.output import report_surfaces as rs

        assert rc.SURFACES["html"] is rs.HtmlComponents
        assert rc.SURFACES["markdown"] is rs.MarkdownComponents
        assert rc.missing_members(rs.HtmlComponents) == ()
        assert rc.missing_members(rs.MarkdownComponents) == ()


class TestASectionIsAsLongAsItHasSomethingToSay:
    """R5, R6, R7 as the reader meets them."""

    def test_a_collapsed_section_is_a_score_and_one_line(self, tmp_path):
        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        _, md = render_note(tmp_path, result)

        collapsed = [s for s in context["sections"] if not s.expanded]
        assert collapsed, "the fixture must leave something collapsed"

        for section in collapsed:
            start = md.index(f"## {section.title}")
            body = md[start:].split("\n## ", 1)[0]
            assert "|" not in body, section.title      # no table
            assert "\n- " not in body, section.title   # no findings

    def test_an_expanded_section_names_every_reason_it_expanded(self, tmp_path):
        """AE1: the three lender readings, stated before the table that scores
        all three at zero."""
        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        _, md = render_note(tmp_path, result)

        quality = next(s for s in context["sections"]
                       if s.key == "quality_business")
        body = md[md.index(f"## {quality.title}"):].split("\n## ", 1)[0]

        mismatches = [
            finding.text for finding in quality.findings
            if finding.source == "sector_mismatch"
        ]

        assert quality.expanded
        assert len(mismatches) == 3, mismatches
        for text in mismatches:
            assert text in body
        assert body.index("does not measure anything") < body.index("| Metric |")

    def test_the_opening_states_how_many_sections_expanded(self, tmp_path):
        """AE5's other half: the contrast has to be legible without reading."""
        _, md = render_note(tmp_path)

        assert "sections have something to explain" in md

    def test_a_company_that_fires_nothing_collapses_everywhere(self, tmp_path):
        """AE5. `rich_result()` is that company — no zero score, and a sector
        nobody has declared applicability for."""
        generator = ReportGenerator(output_dir=str(tmp_path))
        context = generator._clarity_context(rich_result())
        _, md = render_note(tmp_path, rich_result())

        assert not any(s.expanded for s in context["sections"])
        assert "No section needed more than its score and one line" in md
        assert len(md) < len(render_note(tmp_path / "wide")[1])


class TestEveryReadingIsPresentWithoutAModel:
    """AE3, KD2: the reading layer is pure, so `--no-llm` loses nothing."""

    def test_a_run_with_no_llm_still_carries_every_section_reading(self, tmp_path):
        result = make_result()
        assert result.llm_analysis is None

        generator = ReportGenerator(output_dir=str(tmp_path))
        context = generator._clarity_context(result)
        html, md = render_note(tmp_path, result)

        from boundless100x.output.report_surfaces import MarkdownComponents

        surface = MarkdownComponents()
        for section in [context["lead"], *context["sections"]]:
            rendered = surface.render_reading(section.reading)
            assert rendered.strip(), section.title
            assert rendered.splitlines()[0] in md, section.title
            assert visible_text(html).count(section.title) >= 1

    def test_no_section_opens_on_a_blank(self, tmp_path):
        """The failure AE3 names: a heading with nothing under it."""
        _, md = render_note(tmp_path, make_result())

        lines = md.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("## "):
                continue
            following = [text for text in lines[index + 1:index + 4] if text.strip()]
            assert following, line

    def test_the_note_names_no_action_when_no_model_ran(self, tmp_path):
        _, md = render_note(tmp_path, make_result())

        assert "Action:" not in md


class TestAnAbsenceIsAlwaysExplained:
    """R4 and R12 in the rendered table — AE4."""

    def test_a_metric_with_no_declaration_renders_unknown_with_its_reason(
        self, tmp_path
    ):
        """Stripped in memory rather than in YAML: every shipped metric carries
        a declaration after U3, so the only honest way to render AE4's case is
        to take one away for the length of a test."""
        from boundless100x.compute_engine.engine import ComputeEngine

        engine = ComputeEngine()
        engine.metrics["dupont_margin"].pop("presentation", None)

        generator = ReportGenerator(output_dir=str(tmp_path))
        generator._registry = engine
        context = generator._clarity_context(clarity_result())

        quality = next(s for s in context["sections"]
                       if s.key == "quality_business")
        stripped = next(row for row in quality.rows
                        if row.metric_id == "dupont_margin")

        # A row, not a dropped metric and not a bare number: the label is
        # there, the reading cell carries the reason, and there is no value to
        # show because a metric with no declared unit has nothing R12 would let
        # it render.
        assert not stripped.known
        assert stripped.label
        assert "nothing declares how to read this metric" in stripped.unknown.reason

    def test_a_row_with_no_figure_says_so_rather_than_showing_a_dash(
        self, tmp_path
    ):
        """R4's "never an empty cell", at the one place it is easy to leave one."""
        from boundless100x.output.report_vocabulary import NO_FIGURE_LABEL

        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        _, md = render_note(tmp_path, result)

        rows = [row for section in context["sections"] for row in section.rows]
        assert any(not row.value for row in rows), "the fixture must have one"
        assert NO_FIGURE_LABEL in md
        assert "| — |" not in md

    def test_every_rendered_row_carries_all_three_of_its_cells(self, tmp_path):
        from boundless100x.output.report_surfaces import metric_cells

        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(clarity_result())

        for section in context["sections"]:
            for row in section.rows:
                assert all(cell.strip() for cell in metric_cells(row)), row.label


class TestCoverageIsStatedWhereItMatters:
    """R18 — AE7."""

    def test_a_thin_element_states_its_coverage_in_the_one_line_reading(
        self, tmp_path
    ):
        result = clarity_result()
        # The number AE7 names, set on the fixture so the assertion is about
        # the rendering rather than about whatever the fixture happened to hold.
        result.scores["coverage"]["elements"]["quality_business"] = 0.32

        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        _, md = render_note(tmp_path, result)

        quality = next(s for s in context["sections"]
                       if s.key == "quality_business")

        assert "32%" in quality.reading.qualifier
        assert quality.reading.qualifier in md

    def test_an_element_above_the_bar_recites_no_number(self, tmp_path):
        result = clarity_result()
        result.scores["coverage"]["elements"]["growth"] = 1.0

        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)

        growth = next(s for s in context["sections"] if s.key == "growth")

        assert growth.reading.qualifier == ""


class TestTheMultiYearTablesLiveInTheAppendix:
    """R10 — the twelve rows of cash-flow history that outweighed the score."""

    APPENDIX_TABLES = ("Ten-year snapshot", "Cash-flow history",
                       "Shareholding history")

    @pytest.mark.parametrize("heading", APPENDIX_TABLES)
    def test_each_table_renders_after_the_appendix_heading(self, tmp_path, heading):
        html, md = render_note(tmp_path)

        for surface, marker in ((md, "## Appendix"), (visible_text(html), "Appendix")):
            assert heading in surface, heading
            assert surface.index(marker) < surface.index(heading), heading

    def test_no_element_section_body_holds_one_of_them(self, tmp_path):
        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        _, md = render_note(tmp_path, result)

        appendix_at = md.index("## Appendix")
        for section in context["sections"]:
            start = md.index(f"## {section.title}")
            body = md[start:].split("\n## ", 1)[0]
            assert start < appendix_at
            for heading in self.APPENDIX_TABLES:
                assert heading not in body, (section.title, heading)


class TestTheExplanationIsReachableAndNeverInline:
    """R3's two halves, and the second is what makes the first safe."""

    def test_every_row_reference_resolves_to_a_body_in_the_appendix(self, tmp_path):
        from boundless100x.output.report_surfaces import anchor_id

        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        html, md = render_note(tmp_path, result)

        bodies = {d.anchor for d in context["appendix"]["disclosures"]}
        # Only the rows a reader can actually see: a collapsed section renders
        # no rows and therefore no references, while its explanations are in
        # the appendix all the same — R3 is about every metric being reachable,
        # not about every metric being on the page.
        referenced = {
            row.disclosure.anchor
            for section in [*context["sections"], context["unscored"]]
            if section and section.expanded
            for row in section.rows
            if row.disclosure
        }

        assert referenced, "the fixture must reference at least one explanation"
        assert referenced <= bodies
        for anchor in referenced:
            assert f'id="{anchor_id(anchor)}"' in html
            assert f"(#{anchor_id(anchor)})" in md

    def test_no_explanation_body_appears_inside_a_section(self, tmp_path):
        """The half a type system enforces: `Section.flow` excludes the bodies,
        so this asserts the template did not go around it."""
        result = clarity_result()
        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._clarity_context(result)
        _, md = render_note(tmp_path, result)

        appendix_at = md.index("## Appendix")
        before_appendix = md[:appendix_at]

        for disclosure in context["appendix"]["disclosures"]:
            assert disclosure.body not in before_appendix, disclosure.title


class TestTheNoteStatesAFlagOnce:
    """One flag, one appearance. Not KD4's rule, which is about something else.

    KD4 deliberately lets several *sections* state the same finding
    independently — each of them reached it, and deduplicating into a roll-up
    would leave a reader unable to tell which section the finding was about.
    This is the other thing: `build_section` turns every element-mapped flag
    into a section finding, and the appendix used to re-derive its Signals list
    from the whole flag list with no filter, so most flags printed twice in one
    document — once under their element and once at the foot.
    """

    @staticmethod
    def _context(tmp_path, result=None):
        return ReportGenerator(output_dir=str(tmp_path))._clarity_context(
            result if result is not None else clarity_result()
        )

    def test_no_appendix_signal_repeats_one_an_expanded_section_showed(
        self, tmp_path
    ):
        context = self._context(tmp_path)

        shown = {
            finding.headline
            for section in context["sections"] if section.expanded
            for finding in section.findings
        }
        appendix = [f.headline for f in context["appendix"]["signals"]]

        assert shown, "the fixture must expand at least one section with a flag"
        assert appendix, "and must leave something for the appendix to carry"
        assert not shown & set(appendix)

    def test_the_rendered_document_prints_each_label_once(self, tmp_path):
        """The claim measured on the page rather than on the model: a flag from
        an expanded section appears exactly once in each surface."""
        result = clarity_result()
        generator = ReportGenerator(output_dir=str(tmp_path))
        raw_flags = {f["raw"] for f in generator._collect_flags(result.metrics)}
        context = generator._clarity_context(result)
        html, md = render_note(tmp_path, result)

        labels = [
            finding.headline
            for section in context["sections"] if section.expanded
            for finding in section.findings
            if finding.source in raw_flags
        ]

        assert labels, "the fixture must expand a section carrying a flag"
        for label in labels:
            assert visible_text(html).count(label) == 1, label
            assert plain_text(md).count(label) == 1, label

    def test_a_flag_from_a_collapsed_section_still_reaches_the_appendix(
        self, tmp_path
    ):
        """The filter is "what was rendered", never "what has an element".

        A collapsed section renders no findings at all — both templates gate on
        `section.expanded` — so dropping its flags from the appendix too would
        delete them from the note entirely, which is a worse failure than
        printing them twice.
        """
        result = clarity_result()
        context = self._context(tmp_path, result)
        html, md = render_note(tmp_path, result)

        collapsed = {
            finding.headline
            for section in context["sections"] if not section.expanded
            for finding in section.findings
        }
        appendix = {f.headline for f in context["appendix"]["signals"]}

        assert collapsed, "the fixture must collapse a section that carries a flag"
        assert collapsed <= appendix
        for label in collapsed:
            assert label in visible_text(html), label
            assert label in plain_text(md), label

    def test_a_flag_belonging_to_no_section_is_untouched(self, tmp_path):
        """The zero-weight signals map to `forward_signals`, which is not one of
        the six scored sections, so nothing ever showed them and the appendix is
        the only place they can appear."""
        context = self._context(tmp_path)

        assert "Re-rating Headroom — Favourable" in {
            f.headline for f in context["appendix"]["signals"]
        }


class TestTheSubjectIsNotOneOfItsOwnComparables:
    """R8 asks how *other* companies read this metric.

    `generate()` writes the run's own `scores.json` before the clarity block
    and into the same directory `load_scored_corpus` then scans, so without an
    exclusion the ticker being rendered arrives as one of the votes on whether
    its own zero is corpus-wide.
    """

    def test_the_note_passes_its_own_ticker_as_the_exclusion(self, tmp_path,
                                                             monkeypatch):
        from boundless100x.output import report_generator as module

        seen: dict = {}
        real = module.load_scored_corpus

        def spy(reports_dir=None, **kwargs):
            seen.update(kwargs)
            return real(reports_dir, **kwargs)

        monkeypatch.setattr(module, "load_scored_corpus", spy)

        ReportGenerator(output_dir=str(tmp_path)).generate(
            clarity_result(), formats=["json", "clarity"],
        )

        assert seen.get("exclude") == "TEST"

    def test_the_self_written_scores_would_otherwise_be_in_the_corpus(
        self, tmp_path
    ):
        """The precondition, asserted rather than assumed: the vote really is
        on disk by the time the note is built, so the exclusion is doing work
        rather than guarding against something that cannot happen."""
        from boundless100x.output.report_expansion import load_scored_corpus

        ReportGenerator(output_dir=str(tmp_path)).generate(
            clarity_result(), formats=["json", "clarity"],
        )

        assert load_scored_corpus(tmp_path).tickers == ("TEST",)
        assert load_scored_corpus(tmp_path, exclude="TEST").tickers == ()


class TestTheNoteEscapesWhatItInterpolates:
    """The slots no component ever guarded, on both surfaces.

    `guard_text` refuses markup at the point a component is constructed, and
    `report_surfaces` rests its "Markdown escapes nothing" on exactly that. The
    masthead and the appendix tables interpolate values that never became
    components — a company name, a sector, a scraped quarter label — so they
    reached the page raw. Seven of the twenty-six cached tickers carry an
    ampersand in one or the other.
    """

    AMPERSAND_NAME = "Indian Railway Catering & Tourism Corporation Ltd"
    AMPERSAND_SECTOR = "Chemicals & Petrochemicals"

    def result_with_ampersands(self):
        result = clarity_result()
        result.data["metadata"]["name"] = self.AMPERSAND_NAME
        result.data["metadata"]["sector"] = self.AMPERSAND_SECTOR
        return result

    def test_no_raw_ampersand_survives_into_the_html(self, tmp_path):
        import re as re_module

        html, _ = render_note(tmp_path, self.result_with_ampersands())

        assert self.AMPERSAND_NAME not in html
        assert self.AMPERSAND_SECTOR not in html
        assert "Catering &amp; Tourism" in html
        assert "Chemicals &amp; Petrochemicals" in html
        # Every `&` in the document opens a valid entity — a bare one is what
        # makes the markup invalid, and one slot missed is one bare ampersand.
        assert not re_module.findall(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[\da-fA-F]+);)",
                                     html)

    def test_the_reader_still_sees_the_name_they_typed(self, tmp_path):
        """Escaped, not mangled. R14 is a claim about content, and the content
        is the company's actual name."""
        html, _ = render_note(tmp_path, self.result_with_ampersands())

        assert self.AMPERSAND_NAME in visible_text(html)
        assert self.AMPERSAND_SECTOR in visible_text(html)

    def test_the_markdown_note_prints_the_ampersand_literally(self, tmp_path):
        """The asymmetry, and why the Markdown twin does not get `|e`: `&` is
        an ordinary character there, and `&amp;` would be an HTML entity shown
        to a reader of a plain-text file."""
        _, md = render_note(tmp_path, self.result_with_ampersands())

        assert self.AMPERSAND_NAME in md
        assert "&amp;" not in md

    def test_a_pipe_in_a_scraped_label_cannot_break_a_markdown_table(
        self, tmp_path
    ):
        """What Markdown actually breaks on. A `|` inside a table cell ends the
        cell and shifts every column after it, so the shareholding row would
        silently gain a column and misalign every figure in it.
        """
        result = clarity_result()
        result.data["shareholding_bse"] = None
        result.data["shareholding"] = pd.DataFrame([
            {"quarter": "Mar | 2026", "promoter_pct": 51.0, "fii_pct": 9.0,
             "dii_pct": 8.0, "public_pct": 32.0},
        ])

        html, md = render_note(tmp_path, result)

        row = next(line for line in md.splitlines() if "Mar" in line and "2026" in line)
        assert r"Mar \| 2026" in row
        # Seven declared columns means eight pipes; an unescaped one would make
        # nine and push every reading one column right.
        assert row.count("|") - row.count(r"\|") == 8
        assert "Mar | 2026" in visible_text(html)

    def test_the_html_note_escapes_the_scraped_quarter_label(self, tmp_path):
        result = clarity_result()
        result.data["shareholding_bse"] = None
        result.data["shareholding"] = pd.DataFrame([
            {"quarter": "Mar 2026 <b>", "promoter_pct": 51.0},
        ])

        html, _ = render_note(tmp_path, result)

        assert "Mar 2026 &lt;b&gt;" in html
        assert "Mar 2026 <b>" not in html

    @pytest.mark.parametrize("template_name,filter_token", [
        ("clarity_report.html.j2", "|e"),
        ("clarity_report.md.j2", "|md_text"),
    ])
    def test_every_non_surface_slot_carries_its_surfaces_filter(
        self, template_name, filter_token
    ):
        """The rule stated as a rule, so a slot added later is caught here
        rather than by whoever first analyses a company with an ampersand.

        `autoescape` stays off — the component renderers return pre-escaped
        fragments and `_paragraphize` returns Markup, so a global flip would
        double-escape this note *and* move the legacy dashboard's bytes, which
        the goldens forbid. So the rule is per slot, and it is "every
        non-`surface.` string slot" rather than "the ones that are scraped
        today": the second is a judgement to re-make every time a value's
        provenance changes, and the first is one this test can make for us.

        Both templates, because they are twins and a rule enforced on one of
        them is a rule that drifts on the other.
        """
        import re as re_module

        template = (PACKAGE_ROOT / "output" / "templates"
                    / template_name).read_text()

        unguarded = [
            slot for slot in re_module.findall(r"\{\{(.*?)\}\}", template)
            if "surface." not in slot        # already escaped by its renderer
            and filter_token not in slot
            and "format(" not in slot        # a float, not a string
            and "if " not in slot            # an inline literal choice
            and "section_block" not in slot  # a macro call, not a value
        ]

        assert unguarded == [], unguarded


class TestTheNoteNeverCostsTheRunItsOtherReports:
    """The fence around the new block. R16 is a promise about every run."""

    def test_a_failing_note_leaves_the_legacy_reports_on_disk(self, tmp_path,
                                                              monkeypatch):
        monkeypatch.setattr(
            ReportGenerator, "_clarity_context",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = clarity_result()

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            result, formats=["html", "md", "clarity"],
        )

        assert (report_dir / "TEST_dashboard.html").exists()
        assert (report_dir / "TEST_report.md").exists()
        assert not (report_dir / "TEST_note.md").exists()
        assert any("Research note" in e for e in result.errors)


class TestTheCompositeLineAgreesWithItself:
    """The note's opening figure and its reading must come from one number.

    `section_reading` rounds a score before banding it, because banding the
    raw figure while the headline rounds it produced `7.0 / 10 — Reads
    middling` against a band boundary at exactly seven. `_clarity_lead` builds
    the composite line by hand — the composite is not an element, so no
    element-shaped builder covered it — and did not carry that rule across.

    The console fixed it independently in `cli._composite_reading` and quoted
    the same reasoning, which left the two surfaces disagreeing about the same
    company: at a composite of 6.97 the note read "Reads middling" and the
    console read "Reads strong", both above a headline of 7.0. R14 says the
    surfaces render the same content, so one builder now produces both.
    """

    @staticmethod
    def lead_line(composite: float):
        from tests.conftest import make_result

        result = make_result()
        result.scores = {**(result.scores or {}), "composite": composite}
        lead = ReportGenerator()._clarity_lead(result, [])
        return lead.reading

    @pytest.mark.parametrize("composite", [6.97, 6.96, 3.98, 3.96])
    def test_the_headline_and_the_band_read_the_same_figure(self, composite):
        """A figure that rounds up across a boundary must band where it lands."""
        from boundless100x.output.report_components import score_band

        reading = self.lead_line(composite)

        assert reading.headline.startswith(f"{round(composite, 1):.1f}")
        assert score_band(round(composite, 1)) in reading.text, (
            f"composite {composite} renders headline {reading.headline!r} "
            f"beside {reading.text!r} — the number and its reading disagree"
        )

    def test_the_note_and_the_console_read_a_composite_identically(self):
        """R14 on the one line every report opens with."""
        from boundless100x.cli import _composite_reading

        for composite in (6.97, 8.4, 5.32, 2.1):
            note = self.lead_line(composite)
            console = _composite_reading(composite)

            assert note.text == console.text, composite
            assert note.headline == console.headline, composite

    def test_an_unscored_composite_is_unknown_with_the_same_reason(self):
        from boundless100x.cli import _composite_reading
        from boundless100x.output.report_vocabulary import COMPOSITE_UNKNOWN_REASON

        note = self.lead_line(None)
        console = _composite_reading(None)

        assert note.unknown is not None
        assert note.unknown.reason == COMPOSITE_UNKNOWN_REASON
        assert note.unknown.reason == console.unknown.reason
