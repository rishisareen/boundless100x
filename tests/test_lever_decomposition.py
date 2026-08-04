"""The 4-lever table must use the same assumptions and inputs as everything else.

Two divergences lived here. The table called the price lever without the macro
config, so a changed inflation assumption reached the scored metric but not the
report narrative. And it looked for a `pe_ratio` column that Screener never
produces, so its valuation check always read "cannot be computed" — except in
the report, which patched a P/E into its own separate copy. The model and the
reader were shown different verdicts on the same company.
"""

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.growth import (
    compute_lever_decomposition_table,
)
from tests.conftest import make_data, make_financials


class TestMacroReachesTheTable:
    def test_inflation_assumption_is_the_configured_one(self):
        table = compute_lever_decomposition_table(make_data(), macro={"inflation_pct": 9.0})

        price_row = next(r for r in table["lever_table"] if r["lever"] == "Price Lever")
        assert "9.0%" in price_row["analysis"] or "9.0" in price_row["analysis"]

    def test_changing_inflation_changes_the_lever_verdict(self):
        data = make_data()
        data["financials"] = make_financials(10, revenue_growth=0.10)

        low = compute_lever_decomposition_table(data, macro={"inflation_pct": 2.0})
        high = compute_lever_decomposition_table(data, macro={"inflation_pct": 14.0})

        low_row = next(r for r in low["lever_table"] if r["lever"] == "Price Lever")
        high_row = next(r for r in high["lever_table"] if r["lever"] == "Price Lever")
        assert low_row["status"] != high_row["status"]

    def test_default_is_unchanged_when_no_macro_passed(self):
        assert compute_lever_decomposition_table(make_data())["lever_table"]


class TestValuationCheckResolves:
    def test_current_pe_comes_from_metadata(self):
        """Screener has no pe_ratio column; the P/E lives in metadata."""
        data = make_data()
        data["metadata"]["Stock P/E"] = 42.0

        check = compute_lever_decomposition_table(data)["valuation_check"]

        assert check["current_pe"] == 42.0

    def test_trailing_peg_is_computed_rather_than_abandoned(self):
        data = make_data()
        data["metadata"]["Stock P/E"] = 40.0

        check = compute_lever_decomposition_table(data)["valuation_check"]

        assert check["trailing_peg"] is not None
        assert check["trailing_peg"] == pytest.approx(
            40.0 / check["pat_cagr_5yr"], rel=1e-6
        )

    def test_verdict_is_a_real_verdict(self):
        data = make_data()
        data["metadata"]["Stock P/E"] = 40.0

        verdict = compute_lever_decomposition_table(data)["valuation_check"]["verdict"]

        assert verdict
        assert "cannot" not in verdict.lower()

    def test_absent_pe_still_degrades_gracefully(self):
        data = make_data()
        data["metadata"].pop("Stock P/E", None)

        check = compute_lever_decomposition_table(data)["valuation_check"]

        assert check["current_pe"] is None
        assert check["trailing_peg"] is None


class TestMetricAndTableAgree:
    """The scored metric and the report table grade through one function."""

    def test_one_grade_per_company(self):
        from boundless100x.compute_engine.metrics.builtin.growth import (
            compute_growth_quality,
        )

        data = make_data()
        metric = compute_growth_quality(data, {})
        table = compute_lever_decomposition_table(
            data, macro={"inflation_pct": 5.0, "strong_real_growth_pct": 10.0}
        )

        assert metric.ok
        assert metric.value == table["growth_synthesis"]["quality_flag"]
        assert metric.metadata["primary_drivers"] == table["growth_synthesis"]["primary_drivers"]

    def test_leverage_driven_company_is_risky_in_both(self):
        from boundless100x.compute_engine.metrics.builtin.growth import (
            compute_growth_quality,
        )

        # Revenue below inflation (no volume/price drivers), EPS outrunning
        # EBIT several-fold (financial leverage the only driver).
        data = make_data(financials={"revenue_growth": 0.03, "pat_growth": 0.30})
        metric = compute_growth_quality(data, {})
        table = compute_lever_decomposition_table(
            data, macro={"inflation_pct": 5.0, "strong_real_growth_pct": 10.0}
        )

        assert metric.value == "risky"
        assert table["growth_synthesis"]["quality_flag"] == "risky"

    def test_grades_cover_the_declared_categories(self):
        """Every grade the shared grader emits must be scoreable by the YAML categories."""
        from boundless100x.compute_engine.engine import ComputeEngine

        categories = (
            ComputeEngine().metrics["growth_quality_grade"]["scoring"]["categories"]
        )
        for grade in ("high_quality", "moderate", "low_quality", "risky"):
            assert grade in categories


class TestReportAndModelSeeTheSameTable:
    def test_report_reuses_the_analysis_result_decomposition(self, tmp_path):
        """No second, silently different computation for the reader."""
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        result = make_result()
        sentinel = {
            "earnings_profile": {"pat_cagr_3yr": 11.0, "pat_cagr_5yr": 22.0},
            "lever_table": [],
            "growth_synthesis": {"quality_flag": "high_quality", "narrative": "from service"},
            "valuation_check": {"current_pe": 33.0, "pat_cagr_5yr": 22.0,
                                "trailing_peg": 1.5, "verdict": "fair"},
        }
        result.growth_decomposition = sentinel

        rendered = ReportGenerator(output_dir=str(tmp_path))._compute_growth_decomposition(result)

        assert rendered["growth_synthesis"]["narrative"] == "from service"
        assert rendered["valuation_check"]["current_pe"] == 33.0

    def test_report_still_computes_one_when_none_was_supplied(self, tmp_path):
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        result = make_result()
        assert result.growth_decomposition is None

        rendered = ReportGenerator(output_dir=str(tmp_path))._compute_growth_decomposition(result)

        assert rendered is not None
        assert "lever_table" in rendered
