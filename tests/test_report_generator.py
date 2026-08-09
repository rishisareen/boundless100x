"""Report generation regression tests.

The peer-comparison removal left `_build_sector_context` reading a deleted
`AnalysisResult.comparison` field, which raised AttributeError for every
ticker. These tests pin the generator against that class of regression.

`TestTheLegacyReportIsFrozen` in the middle of this file is the other kind of
pin, and it is a freeze rather than an assertion: **the HTML and Markdown the
generator produces today, byte for byte after normalisation.** A section that
quietly stopped rendering, a number that gained a digit, a label that lost a
word all pass every test in this file that names something specific. Only a
whole-document comparison sees them.

The two goldens are now asymmetric, on purpose. **The Markdown golden is still
frozen**: the reading layer landed in the dashboard and nowhere else, so a
change to `pre_report_clarity_report.md` means something leaked. **The HTML
golden moves whenever the reading layer's own presentation changes** — once
when it was folded in, again when its first pass turned out to need polish
(one number shown twice per section in two colours, three identical headlines
stacked, a table clipped at its own edge) — and each re-baseline is what
proves nothing was *lost* doing it: the section, chart and chip counts are
checked against the previous golden before the new one is written, every time.

Regenerating the goldens is a deliberate act, never a way to make a red test
green. When a change to either report *is* intended:

    venv/bin/python -m pytest tests/test_report_generator.py -q  # read the diff
    venv/bin/python -c "from tests.test_report_generator import write_goldens; write_goldens()"
"""

from pathlib import Path

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import friction as friction_module
from boundless100x.output.report_generator import ELEMENT_CONFIG, ReportGenerator
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
# The reading layer, inside the dashboard (U10)
#
# It shipped first as a fourth format — a `clarity` note written beside the
# dashboard — and that was wrong in a way no test in this file could catch:
# every assertion below passed while the note carried six headings, none of the
# dashboard's six figures, no thesis, no snapshot and no DCF. A reading layer
# in its own document is a second document to open.
#
# So the acceptance criteria are unchanged and only the document they land in
# moved. Each class below still asserts what it always asserted — AE1's sector
# findings, AE3's readings without a model, AE4's unknown-with-reason, AE5's
# collapsed-versus-expanded contrast, AE7's coverage clause — against the
# dashboard.
#
# The goldens above are the other half: they freeze what the dashboard already
# had, so "the reading layer was folded in" and "nothing was lost doing it" are
# two separate failing tests rather than one judgement call.
# ──────────────────────────────────────────────────────────────────────────


def lender_result():
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


def render_dashboard(output_dir, result=None) -> str:
    """The dashboard, from the full `generate` path.

    Through `generate` for the reason `render_legacy` goes through it: the
    format gating, the shared builders and the order they run in are part of
    what is being asserted — the reading layer's corpus is read *after* this
    run's own `scores.json` is written, and only the real path does that.
    """
    subject = result if result is not None else lender_result()
    report_dir = ReportGenerator(output_dir=str(output_dir)).generate(
        subject, formats=["html", "json"],
    )
    return (report_dir / f"{subject.ticker}_dashboard.html").read_text()


def visible_text(markup: str) -> str:
    """What a reader of the HTML actually sees, with the markup taken off.

    The assertions below are about *content*, so matching raw markup would be
    asserting the wrong thing — `<strong>` and `class="headline"` are not what
    a reader meets. Tags come off, entities are unescaped, and whitespace
    collapses, so what is left is the words.
    """
    import html as html_module
    import re as re_module

    text = re_module.sub(r"<script.*?</script>|<style.*?</style>", " ", markup,
                         flags=re_module.S)
    text = re_module.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    return re_module.sub(r"\s+", " ", text)


def section_markup(html: str, title: str) -> str:
    """One element section's markup, from its `<h2>` to the next one.

    Sliced rather than searched, because a title legitimately appears in
    running text — the opening line names the sections that expanded — and a
    `find()` on the whole document would report that mention as the section.
    """
    start = html.index(f"<h2>{title}</h2>")
    rest = html[start + 4:]
    end = rest.find("<h2>")
    return rest[:end] if end >= 0 else rest


def is_open(section_html: str) -> bool:
    """Whether this section's rows render open (R5's expanded shape)."""
    return '<details class="reading-rows" open>' in section_html


def content_of(context: dict) -> list[str]:
    """Every string the reading layer handed the template, off the model itself.

    This is what makes the assertion below a content comparison rather than a
    string comparison: the expectations come from the `Section` objects the
    renderer received, not from the rendering.
    """
    from boundless100x.output import report_surfaces as rs

    pieces: list[str] = []
    for section in [context["lead"], *context["sections"].values()]:
        pieces.append(section.title)
        headline, body, qualifier = rs.reading_line(section.reading)
        # The headline is excluded on purpose, not an oversight this test
        # missed: it is the score, and the dashboard already shows the score
        # once per section via its own numeric badge (the composite's
        # `.composite-score`, each element's `.element-score-badge`) — the
        # reading line used to repeat it, in a different colour, inches below.
        # `TestOneRunProducesOneReadableReport::test_the_two_reports_agree_on_
        # every_element_score` is what checks the number itself still reaches
        # the page; this test is about everything else the model said.
        pieces.extend(p for p in (body, qualifier) if p)
        for finding in section.findings:
            pieces.extend(p for p in rs.finding_line(finding) if p)
        for row in section.rows:
            pieces.extend(rs.metric_cells(row))
        for unknown in section.unknowns:
            pieces.extend((unknown.subject, unknown.reason))
        for caveat in section.caveats:
            pieces.append(rs.caveat_line(caveat))

    for disclosure in context["disclosures"]:
        pieces.extend((disclosure.title, disclosure.body))

    import re as re_module

    return [re_module.sub(r"\s+", " ", p).strip() for p in pieces if str(p).strip()]


class TestOneRunProducesOneReadableReport:
    """The complaint that collapsed the note, as tests.

    "I will have to open dashboard and this together to understand this." So:
    no second document is written, the token that produced one is gone, and
    every format list that named it names it no longer.
    """

    def test_no_separate_note_is_written(self, tmp_path):
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            lender_result(),
        )

        written = sorted(p.name for p in report_dir.iterdir())
        assert "TEST_dashboard.html" in written
        assert "TEST_report.md" in written
        assert not [name for name in written if "_note." in name], written

    def test_the_note_templates_are_gone(self):
        """A template left on disk is a template somebody re-wires."""
        templates = PACKAGE_ROOT / "output" / "templates"

        assert sorted(p.name for p in templates.glob("*.j2")) == [
            "sqglp_report.html.j2", "sqglp_report.md.j2",
        ]

    def test_the_default_format_list_names_three_documents(self, tmp_path):
        from boundless100x.output.report_generator import DEFAULT_FORMATS

        assert DEFAULT_FORMATS == ["html", "md", "json"]

    def test_the_cli_default_agrees_with_the_generators(self):
        """`cli.py` passes `formats=` and never sees `DEFAULT_FORMATS`, so the
        two lists agree by hand or not at all."""
        source = (PACKAGE_ROOT / "cli.py").read_text()

        assert '"html,md,json"' in source
        assert "clarity,json" not in source

    def test_the_reading_layer_rides_on_the_html_token(self, tmp_path):
        """No token of its own: asking for the dashboard is asking for it."""
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            lender_result(), formats=["html"],
        )
        html = (report_dir / "TEST_dashboard.html").read_text()

        assert '<details class="reading-rows"' in html
        assert not (report_dir / "TEST_report.md").exists()

    def test_the_two_reports_agree_on_every_element_score(self, tmp_path):
        """One of the plan's three success criteria, asserted rather than hoped.

        The reading layer reads its figures off the same `result` the rest of
        the dashboard does, so a disagreement would mean one of them recomputed
        something.

        Matched with an optional space around the slash rather than a literal
        `" / "`: the dashboard's score badge and its reading line used to both
        print the figure, in two formats inches apart, and the badge's own
        `X.X/10` carries no space around the slash by design (it is a
        superscript-style suffix, not a fraction). One number is now shown
        once per section, so the exact spacing is a rendering detail, not the
        property this test is about.
        """
        import re

        result = lender_result()
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            result, formats=["html", "md"],
        )
        legacy = (report_dir / "TEST_report.md").read_text()
        # The badge splits "7.4" and "/10" across a `<span>` for the smaller
        # suffix — real markup a plain substring or a tag-blind regex both
        # miss. `visible_text` is what a reader sees once that span
        # disappears into ordinary text, which is what this assertion is
        # actually about.
        dashboard = visible_text((report_dir / "TEST_dashboard.html").read_text())

        for score in result.scores["elements"].values():
            assert f"{score:.1f}/10" in legacy
            assert re.search(rf"{score:.1f}\s*/\s*10", dashboard)
        assert str(result.scores["composite"]) in legacy
        assert re.search(
            rf"{result.scores['composite']:.1f}\s*/\s*10", dashboard
        )


class TestNothingTheModelCarriedIsDropped:
    """Every string the components were built from reaches the page.

    This was R14's cross-surface comparison when there were two documents. One
    surface does not make the claim vacuous — it makes it the claim that
    matters: a `Section` assembled and then not rendered is the silent omission
    the whole component set exists to prevent, and a heading with nothing under
    it is exactly what the note turned out to be.
    """

    def test_every_string_the_model_carried_reaches_the_dashboard(self, tmp_path):
        generator = ReportGenerator(output_dir=str(tmp_path))
        result = lender_result()
        context = generator._reading_context(result)
        html = render_dashboard(tmp_path, result)

        seen = visible_text(html)
        missing = [piece for piece in content_of(context) if piece not in seen]

        assert missing == []

    def test_every_element_section_carries_its_reading(self, tmp_path):
        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        assert set(context["sections"]) == set(ELEMENT_CONFIG)
        for element, section in context["sections"].items():
            markup = section_markup(html, f"{ELEMENT_CONFIG[element]['label']} "
                                          f"({ELEMENT_CONFIG[element]['short']})")
            assert '<details class="reading-rows"' in markup, element
            assert section.reading.text in visible_text(markup), element

    def test_the_reading_layers_fragments_reach_the_page_escaped(self, tmp_path):
        """The dashboard renders with `autoescape=False`, so escaping is the
        component renderer's job and nothing downstream checks it.

        The note carried its own `|e` on every non-`surface.` slot; those slots
        are gone with it, and the reading layer's own strings pass through
        `HtmlComponents`, which escapes. Pinned on the coverage clause because
        it is the one reading-layer sentence guaranteed to contain a character
        that must not reach the page raw — a run where this regressed would put
        an apostrophe or an ampersand straight into the markup.
        """
        result = lender_result()
        result.scores["coverage"]["elements"]["quality_business"] = 0.32
        html = render_dashboard(tmp_path, result)

        assert "this element&#39;s declared weight" in html
        assert "this element's declared weight" not in html

    def test_both_surfaces_render_every_member_of_the_closed_set(self):
        """R13's mechanism, asserted where it can be read as the requirement.

        `MarkdownComponents` has no consumer today — the Markdown report still
        renders from its own template and its turn is deferred, not cancelled.
        It stays registered and stays complete anyway: two renderers built from
        the same three composition helpers is what makes "the content is
        decided in one place" checkable by reading `report_surfaces.py`, and a
        single surface would let a phrase drift into markup unnoticed.
        """
        from boundless100x.output import report_components as rc
        from boundless100x.output import report_surfaces as rs

        assert rc.SURFACES["html"] is rs.HtmlComponents
        assert rc.SURFACES["markdown"] is rs.MarkdownComponents
        assert rc.missing_members(rs.HtmlComponents) == ()
        assert rc.missing_members(rs.MarkdownComponents) == ()


class TestASectionIsAsLongAsItHasSomethingToSay:
    """R5, R6, R7 as the reader meets them — AE5."""

    def test_a_collapsed_section_shows_its_score_and_one_line(self, tmp_path):
        """Closed, not deleted. The rows are one click away, which is the whole
        difference between a shape that guides a reader and one that hides
        evidence from them."""
        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        collapsed = [s for s in context["sections"].values() if not s.expanded]
        assert collapsed, "the fixture must leave something collapsed"

        for section in collapsed:
            markup = section_markup(html, f"{section.title} "
                                          f"({ELEMENT_CONFIG[section.key]['short']})")
            assert not is_open(markup), section.title
            assert 'class="finding' not in markup, section.title
            assert section.reading.text in visible_text(markup), section.title
            # The rows are in the document, behind the disclosure.
            assert '<table class="reading-table">' in markup, section.title

    def test_an_expanded_section_names_every_reason_it_expanded(self, tmp_path):
        """AE1: the three lender readings, stated before the table that scores
        all three at zero."""
        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        quality = context["sections"]["quality_business"]
        markup = section_markup(html, "Quality — Business (QB)")
        text = visible_text(markup)

        mismatches = [
            finding.text for finding in quality.findings
            if finding.source == "sector_mismatch"
        ]

        assert quality.expanded
        assert is_open(markup)
        assert len(mismatches) == 3, mismatches
        for reason in mismatches:
            assert reason in text
        assert markup.index("does not measure anything") < markup.index(
            '<table class="reading-table">'
        )

    def test_the_opening_states_how_many_sections_expanded(self, tmp_path):
        """AE5's other half: the contrast has to be legible without reading."""
        html = render_dashboard(tmp_path)

        assert "sections have something to explain" in visible_text(html)

    def test_a_company_that_fires_nothing_collapses_everywhere(self, tmp_path):
        """AE5. `rich_result()` is that company — no zero score, and a sector
        nobody has declared applicability for."""
        generator = ReportGenerator(output_dir=str(tmp_path))
        context = generator._reading_context(rich_result())
        html = render_dashboard(tmp_path, rich_result())

        assert not any(s.expanded for s in context["sections"].values())
        assert "No section needed more than its score and one line" in visible_text(html)
        assert '<details class="reading-rows" open>' not in html

    def test_the_expanded_and_collapsed_shapes_are_distinguishable(self, tmp_path):
        """The contrast measured on the page. A reader scanning two companies
        should see which one has problems before reading either."""
        clean = render_dashboard(tmp_path / "clean", rich_result())
        lender = render_dashboard(tmp_path / "lender", lender_result())

        assert clean.count('<details class="reading-rows" open>') == 0
        assert lender.count('<details class="reading-rows" open>') >= 1


class TestEveryReadingIsPresentWithoutAModel:
    """AE3, KD2: the reading layer is pure, so `--no-llm` loses nothing."""

    def test_a_run_with_no_llm_still_carries_every_section_reading(self, tmp_path):
        from boundless100x.output.report_surfaces import reading_line

        result = make_result()
        assert result.llm_analysis is None

        generator = ReportGenerator(output_dir=str(tmp_path))
        context = generator._reading_context(result)
        html = render_dashboard(tmp_path, result)

        seen = visible_text(html)
        for section in [context["lead"], *context["sections"].values()]:
            _headline, body, _qualifier = reading_line(section.reading)
            assert body, section.title
            assert body in seen, section.title
            assert section.title in seen, section.title

    def test_no_element_section_opens_on_a_blank(self, tmp_path):
        """The failure AE3 names: a heading with nothing under it."""
        html = render_dashboard(tmp_path, make_result())

        for element, cfg in ELEMENT_CONFIG.items():
            markup = section_markup(html, f"{cfg['label']} ({cfg['short']})")
            assert '<p class="reading' in markup, element
            assert visible_text(markup).strip(), element

    def test_the_dashboard_names_no_reading_layer_action_when_no_model_ran(
        self, tmp_path
    ):
        html = render_dashboard(tmp_path, make_result())

        assert "Action:" not in visible_text(html)


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
        context = generator._reading_context(lender_result())

        quality = context["sections"]["quality_business"]
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

        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        rows = [row for section in context["sections"].values()
                for row in section.rows]
        assert any(not row.value for row in rows), "the fixture must have one"
        assert NO_FIGURE_LABEL in visible_text(html)

    def test_every_rendered_row_carries_all_three_of_its_cells(self, tmp_path):
        from boundless100x.output.report_surfaces import metric_cells

        context = ReportGenerator(
            output_dir=str(tmp_path)
        )._reading_context(lender_result())

        for section in context["sections"].values():
            for row in section.rows:
                assert all(cell.strip() for cell in metric_cells(row)), row.label

    def test_no_bare_number_reaches_a_reading_cell(self, tmp_path):
        """R12, which is the defect the four-column drill-down embodied: `0.09`
        (a ratio), `25.7` (a percentage), `2.0` (a count of years) and `0.84` (a
        coefficient of variation) in one column with nothing to tell them
        apart. Every reading cell now carries an interpretation beside its
        figure, so none of them is only a number."""
        import re as re_module

        html = render_dashboard(tmp_path, lender_result())

        cells = re_module.findall(r"<tr class=\"(?:known|unknown)\">.*?</th>"
                                  r"<td>(.*?)</td>", html)
        assert cells
        for cell in cells:
            assert " — " in cell, cell


class TestCoverageIsStatedWhereItMatters:
    """R18 — AE7."""

    def test_a_thin_element_states_its_coverage_in_the_one_line_reading(
        self, tmp_path
    ):
        result = lender_result()
        # The number AE7 names, set on the fixture so the assertion is about
        # the rendering rather than about whatever the fixture happened to hold.
        result.scores["coverage"]["elements"]["quality_business"] = 0.32

        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        quality = context["sections"]["quality_business"]
        markup = section_markup(html, "Quality — Business (QB)")

        assert "32%" in quality.reading.qualifier
        assert quality.reading.qualifier in visible_text(markup)

    def test_an_element_above_the_bar_recites_no_number(self, tmp_path):
        result = lender_result()
        result.scores["coverage"]["elements"]["growth"] = 1.0

        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)

        assert context["sections"]["growth"].reading.qualifier == ""


class TestTheExplanationIsReachableAndNeverInline:
    """R3's two halves, and the second is what makes the first safe."""

    def test_every_row_reference_resolves_to_a_body_in_the_appendix(self, tmp_path):
        from boundless100x.output.report_surfaces import anchor_id

        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        bodies = {d.anchor for d in context["disclosures"]}
        # Every row, including a collapsed section's: the dashboard renders the
        # rows either way and only the `open` attribute differs, so a reference
        # that did not resolve would be a dead link one click from the surface.
        referenced = {
            row.disclosure.anchor
            for section in context["sections"].values()
            for row in section.rows
            if row.disclosure
        }

        assert referenced, "the fixture must reference at least one explanation"
        assert referenced <= bodies
        for anchor in referenced:
            assert f'id="{anchor_id(anchor)}"' in html
            assert f'href="#{anchor_id(anchor)}"' in html

    def test_no_explanation_body_appears_inside_an_element_section(self, tmp_path):
        """The half a type system enforces: `Section.flow` excludes the bodies,
        so this asserts the template did not go around it.

        Scoped to the six element sections rather than to everything above the
        Appendix, because one section above it states a meaning inline **by
        design**: Forward Signals renders each zero-weight metric's `meaning`
        beside its figure, and R8 requires exactly that — those metrics never
        receive a score, so the number is all a reader gets and a bare number
        is not signal. `rerating_headroom` is declared in the Price element, so
        its explanation legitimately appears twice in one document: once inline
        where nothing else interprets it, once in the Appendix where the Price
        row links to it. R3 is a rule about the reading flow, not about the
        page.
        """
        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        for element, cfg in ELEMENT_CONFIG.items():
            markup = section_markup(html, f"{cfg['label']} ({cfg['short']})")
            for disclosure in context["disclosures"]:
                assert disclosure.body not in markup, (element, disclosure.title)

    def test_each_explanation_body_appears_once(self, tmp_path):
        """Two sections can reference one metric's explanation. Two bodies
        would give the page two elements with the same id, and an anchor link
        would land on whichever the browser found first."""
        from boundless100x.output.report_surfaces import anchor_id

        result = lender_result()
        context = ReportGenerator(output_dir=str(tmp_path))._reading_context(result)
        html = render_dashboard(tmp_path, result)

        for disclosure in context["disclosures"]:
            assert html.count(f'id="{anchor_id(disclosure.anchor)}"') == 1, (
                disclosure.title
            )


class TestNothingTheDashboardAlreadySaidIsSaidTwice:
    """The reading layer adds; it does not restate.

    The note rebuilt the ten-year snapshot, the cash-flow history, the
    shareholding table and the whole-flag Signals list because it was a
    separate document and had to. Inside the dashboard those are the
    dashboard's own sections, and rebuilding them would put each on the page
    twice.
    """

    def test_the_appendix_holds_one_ten_year_snapshot(self, tmp_path):
        html = render_dashboard(tmp_path)

        assert html.count("10-Year Financial Snapshot") == 1
        assert html.count("Ten-year snapshot") == 0

    def test_the_flag_chips_are_not_also_rendered_as_findings(self, tmp_path):
        """`finding_from_flag` builds a headline and no body, so a flag as a
        finding says exactly what its chip says at four times the height. The
        reading layer is handed no flags at all, which is what makes this
        structural rather than a template that happens not to loop."""
        result = lender_result()
        generator = ReportGenerator(output_dir=str(tmp_path))
        context = generator._reading_context(result)
        flags = {f["label"] for f in generator._collect_flags(result.metrics)}
        html = render_dashboard(tmp_path, result)

        finding_headlines = {
            finding.headline
            for section in context["sections"].values()
            for finding in section.findings
        }
        assert not finding_headlines & flags

        # And each chip's label still reaches the page — once under its
        # element, once in the Appendix's All Signals, exactly as before.
        for label in flags:
            assert label in visible_text(html), label


class TestTheSubjectIsNotOneOfItsOwnComparables:
    """R8 asks how *other* companies read this metric.

    `generate()` writes the run's own `scores.json` before the reading layer is
    built and into the same directory `load_scored_corpus` then scans, so
    without an exclusion the ticker being rendered arrives as one of the votes
    on whether its own zero is corpus-wide.
    """

    def test_the_reading_layer_passes_its_own_ticker_as_the_exclusion(
        self, tmp_path, monkeypatch
    ):
        from boundless100x.output import report_generator as module

        seen: dict = {}
        real = module.load_scored_corpus

        def spy(reports_dir=None, **kwargs):
            seen.update(kwargs)
            return real(reports_dir, **kwargs)

        monkeypatch.setattr(module, "load_scored_corpus", spy)

        ReportGenerator(output_dir=str(tmp_path)).generate(
            lender_result(), formats=["json", "html"],
        )

        assert seen.get("exclude") == "TEST"

    def test_the_self_written_scores_would_otherwise_be_in_the_corpus(
        self, tmp_path
    ):
        """The precondition, asserted rather than assumed: the vote really is
        on disk by the time the reading layer is built, so the exclusion is
        doing work rather than guarding against something that cannot happen."""
        from boundless100x.output.report_expansion import load_scored_corpus

        ReportGenerator(output_dir=str(tmp_path)).generate(
            lender_result(), formats=["json", "html"],
        )

        assert load_scored_corpus(tmp_path).tickers == ("TEST",)
        assert load_scored_corpus(tmp_path, exclude="TEST").tickers == ()


class TestTheReadingLayerNeverCostsTheRunItsDashboard:
    """The fence, and the degrade path behind it.

    The note was an extra file: losing it cost a run nothing already on disk.
    The reading layer is inside the one document the run exists to produce, so
    a failure has to degrade rather than propagate — and degrading must not
    take the section's metrics with it, or a typo in a hand-maintained display
    table costs the dashboard its substance.
    """

    @staticmethod
    def _break_it(monkeypatch):
        monkeypatch.setattr(
            ReportGenerator, "_reading_context",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    def test_a_failing_reading_layer_still_writes_both_reports(self, tmp_path,
                                                               monkeypatch):
        self._break_it(monkeypatch)
        result = lender_result()

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            result, formats=["html", "md"],
        )

        assert (report_dir / "TEST_dashboard.html").exists()
        assert (report_dir / "TEST_report.md").exists()
        assert any("Reading layer" in e for e in result.errors)

    def test_the_drilldown_table_renders_in_its_place(self, tmp_path,
                                                      monkeypatch):
        """Not a leftover — the reason the fence is safe. Without it, a bad
        edit to `sector_applicability.yaml` would empty every element
        section."""
        self._break_it(monkeypatch)

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            lender_result(), formats=["html"],
        )
        html = (report_dir / "TEST_dashboard.html").read_text()

        assert '<details class="reading-rows"' not in html
        assert '<table class="drilldown-table">' in html
        assert "RoCE (5yr Avg)" in html

    def test_the_reason_reaches_the_reader_and_not_only_the_log(self, tmp_path,
                                                                monkeypatch):
        self._break_it(monkeypatch)
        result = lender_result()

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            result, formats=["html"],
        )
        html = (report_dir / "TEST_dashboard.html").read_text()

        assert "Warnings &amp; Errors" in html
        assert "Reading layer unavailable" in html


class TestTheCompositeLineAgreesWithItself:
    """The opening figure and its reading must come from one number.

    `section_reading` rounds a score before banding it, because banding the
    raw figure while the headline rounds it produced `7.0 / 10 — Reads
    middling` against a band boundary at exactly seven. `_reading_lead` builds
    the composite line by hand — the composite is not an element, so no
    element-shaped builder covered it — and did not carry that rule across.

    The console fixed it independently in `cli._composite_reading` and quoted
    the same reasoning, which left the two surfaces disagreeing about the same
    company: at a composite of 6.97 the report read "Reads middling" and the
    console read "Reads strong", both above a headline of 7.0. One builder now
    produces both.
    """

    @staticmethod
    def lead_line(composite: float):
        from tests.conftest import make_result

        result = make_result()
        result.scores = {**(result.scores or {}), "composite": composite}
        lead = ReportGenerator()._reading_lead(result, [])
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

    def test_the_report_and_the_console_read_a_composite_identically(self):
        """One builder, on the one line every surface opens with."""
        from boundless100x.cli import _composite_reading

        for composite in (6.97, 8.4, 5.32, 2.1):
            note = self.lead_line(composite)
            console = _composite_reading(composite)

            assert note.text == console.text, composite


class TestNoIdentifierReachesTheRenderedPage:
    """R15, asserted against the rendered document rather than a component.

    Every component is guarded at construction, and that was taken as proof
    the page was clean. It was not: the eligibility gates interpolated
    `{{ gate.reason }}` straight from the evaluator, which writes for a log —
    "Size headroom not met: market_cap 138,604.00 lt 30,000". Three metric
    ids, a flag id and a comparator reached the reader on the one surface this
    work exists to clean up, and every component-level test still passed,
    because none of that text was ever a component.

    So this reads the finished HTML. It is the only check positioned where the
    defect actually was.
    """

    # `claude_cli` is the one identifier that is also the right word: it is what
    # the owner types (`--llm-provider claude_cli`) and what `config.yaml` holds,
    # so rendering it as prose would name something the reader cannot act on.
    # Allowlisted explicitly rather than by a pattern, so the next addition is a
    # decision somebody makes rather than one a regex makes for them.
    ALLOWED = frozenset({"claude_cli"})

    def visible_text(self, html: str) -> str:
        """The words a reader actually sees.

        Entities are unescaped last and deliberately: `&gt;` survives tag
        stripping as a bare `gt`, which reads as the comparator key and made
        this test fail on labels like "RoCE > 15% Count (10yr)" that were never
        wrong. Scanning the escaped form finds the markup, not the text.
        """
        import html as html_mod
        import re

        for block in ("script", "style"):
            html = re.sub(rf"<{block}.*?</{block}>", " ", html, flags=re.S)
        return re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

    def test_the_dashboard_shows_no_snake_case_identifier(self, tmp_path):
        import re

        html, _ = render_legacy(tmp_path)
        found = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b",
                               self.visible_text(html))) - self.ALLOWED

        assert not found, f"raw identifiers on the page: {sorted(found)}"

    def test_the_dashboard_shows_no_registered_identifier(self, tmp_path):
        """The shape-blind half. `roiic` has no underscore, so the snake_case
        rule above could never see it, and it reached the page while every id
        beside it resolved. A registry id is a known string — match it exactly
        rather than by shape."""
        import re

        from boundless100x.compute_engine.engine import ComputeEngine
        from boundless100x.output.report_vocabulary import FLAG_LABELS

        text = self.visible_text(render_legacy(tmp_path)[0])
        known = (set(ComputeEngine().metrics) | set(FLAG_LABELS)) - self.ALLOWED
        found = {k for k in known if re.search(rf"\b{re.escape(k)}\b", text)}

        assert not found, f"registered identifiers on the page: {sorted(found)}"

    def test_the_dashboard_shows_no_bare_comparator(self, tmp_path):
        import re

        from boundless100x.output.report_components import COMPARATOR_SYMBOLS

        text = self.visible_text(render_legacy(tmp_path)[0])
        found = {c for c in COMPARATOR_SYMBOLS
                 if re.search(rf"\b{c}\b", text)}

        assert not found, f"raw comparators on the page: {sorted(found)}"

    def test_the_comparator_vocabulary_covers_every_declared_one(self):
        """A fifth comparator must not ship without wording."""
        from boundless100x.compute_engine.eligibility import COMPARATORS
        from boundless100x.output.report_components import COMPARATOR_SYMBOLS

        assert set(COMPARATOR_SYMBOLS) == set(COMPARATORS)


class TestTheBadgeColourAgreesWithTheWords:
    """The score badge used to colour itself by its own 7/4 rule while the
    reading line beside it worded the same figure against `SCORE_BANDS`' five
    boundaries — a 6.8 could be painted "strong" green on the same line its
    own sentence called "fair". `_score_color` is the one rule both now share.
    """

    def test_every_band_the_report_can_say_has_a_css_colour(self):
        from boundless100x.output.report_components import SCORE_BANDS, SCORE_LOW_LABEL
        from boundless100x.output.report_vocabulary import SCORE_BAND_CSS_COLOR

        words = {label for _, label in SCORE_BANDS} | {SCORE_LOW_LABEL}

        assert set(SCORE_BAND_CSS_COLOR) == words

    @pytest.mark.parametrize(
        "score",
        # One value inside each of the five bands, plus 6.97 — the exact
        # figure that motivated `composite_reading`: it rounds up across the
        # strong/fair edge, so a colour driven by the raw value would still
        # disagree with a headline driven by the rounded one.
        [1.2, 3.9, 5.4, 6.97, 7.0, 8.5, 10.0],
    )
    def test_the_badge_colour_is_read_off_the_same_band_the_words_use(self, score):
        from boundless100x.output.report_components import score_band
        from boundless100x.output.report_generator import _score_color
        from boundless100x.output.report_vocabulary import SCORE_BAND_CSS_COLOR

        shown = round(float(score), 1)
        assert _score_color(score) == SCORE_BAND_CSS_COLOR[score_band(shown)]

    def test_a_missing_score_is_grey_rather_than_a_guess(self):
        from boundless100x.output.report_generator import _score_color

        assert _score_color(None) == "gray"

    def test_a_non_numeric_score_is_grey_rather_than_an_exception(self):
        from boundless100x.output.report_generator import _score_color

        assert _score_color("N/A") == "gray"


class TestGroupedFindings:
    """Three metrics sharing one trigger used to print the same headline three
    times running, with the one thing that differed between them — which
    metric — buried in the first words of each body. `grouped_findings` is
    the render-time fix; `Section.findings` itself stays untouched.
    """

    @staticmethod
    def finding(headline, text="", sentiment="bad"):
        from boundless100x.output.report_components import Finding

        return Finding(headline=headline, text=text, sentiment=sentiment)

    def test_findings_sharing_a_headline_group_into_one_entry(self):
        from boundless100x.output.report_surfaces import grouped_findings

        findings = [
            self.finding("Measures the wrong thing", "About asset turnover."),
            self.finding("Measures the wrong thing", "About the equity multiplier."),
            self.finding("Measures the wrong thing", "About free cash flow."),
        ]

        groups = grouped_findings(findings)

        assert len(groups) == 1
        headline, sentiment, bodies = groups[0]
        assert headline == "Measures the wrong thing"
        assert sentiment == "bad"
        assert bodies == [
            "About asset turnover.",
            "About the equity multiplier.",
            "About free cash flow.",
        ]

    def test_distinct_headlines_stay_distinct_groups_in_first_seen_order(self):
        """F1's trigger order (sector mismatch, then contradiction, then
        zero-score) must survive grouping — this is why the function preserves
        first-seen order rather than sorting alphabetically, the way a
        template-level `groupby` filter would have."""
        from boundless100x.output.report_surfaces import grouped_findings

        findings = [
            self.finding("Two readings here disagree", "The contradiction."),
            self.finding("Measures the wrong thing", "Metric A."),
            self.finding("Measures the wrong thing", "Metric B."),
        ]

        groups = grouped_findings(findings)

        assert [g[0] for g in groups] == [
            "Two readings here disagree",
            "Measures the wrong thing",
        ]
        assert groups[1][2] == ["Metric A.", "Metric B."]

    def test_a_headline_only_finding_groups_with_an_empty_body_list(self):
        """A flag-derived finding ("Capex Dominated by Acquisitions") has no
        body at all — `render_finding` already treats that as a legitimate
        badge, and grouping must not invent one."""
        from boundless100x.output.report_surfaces import grouped_findings

        groups = grouped_findings([self.finding("Capex Dominated by Acquisitions")])

        assert groups == [("Capex Dominated by Acquisitions", "bad", [])]

    def test_an_empty_findings_list_produces_no_groups(self):
        from boundless100x.output.report_surfaces import grouped_findings

        assert grouped_findings([]) == []

    def test_html_renders_one_heading_and_one_paragraph_per_body(self):
        from boundless100x.output.report_surfaces import HtmlComponents

        html = HtmlComponents().render_finding_group(
            "Measures the wrong thing", "bad", ["About A.", "About B."]
        )

        assert html.count('<p class="headline">') == 1
        assert html.count('<p class="detail">') == 2
        assert "Measures the wrong thing" in html
        assert "About A." in html and "About B." in html

    def test_html_renders_no_detail_paragraph_for_an_empty_body_list(self):
        from boundless100x.output.report_surfaces import HtmlComponents

        html = HtmlComponents().render_finding_group("Capex Dominated", "bad", [])

        assert html.count('<p class="detail">') == 0


class TestOneNumberOncePerSection:
    """The score used to appear twice on one card — the badge in its band
    colour, then the reading line repeating it in blue, inches below.
    `show_headline=False` is how the reading line stops carrying the number
    where a badge already does.
    """

    def test_the_composite_reading_line_does_not_repeat_the_badge_number(
        self, tmp_path
    ):
        html = render_dashboard(tmp_path)
        text = visible_text(html)

        section = text[text.index("What this says"):text.index("What this says") + 400]

        assert "/ 10" not in section and "/10" not in section

    def test_an_element_reading_line_does_not_repeat_the_badge_number(self, tmp_path):
        """Scoped to the `p.reading` paragraph itself, not the whole card —
        the card's auto-generated summary ("Average at 5.0/10.") legitimately
        contains this shape for an unrelated figure, and a broader scan would
        flag that sentence as if it were the defect."""
        import re

        html = render_dashboard(tmp_path)
        card = section_markup(html, "Size (S)")

        match = re.search(r'<p class="reading[^"]*">.*?</p>', card, re.S)
        assert match, "no reading paragraph found in the Size section"
        text = visible_text(match.group(0))

        assert "/ 10" not in text and "/10" not in text
