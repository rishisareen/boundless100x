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
        "pe_vs_historical": MetricResult(
            value=82.0, flags=["pe_above_historical_75th"]
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
        "| RoCE 5yr Avg | 22.0 | 82% | 12% |",
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
