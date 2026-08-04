"""100x eligibility gates.

The composite is additive, so strong quality can outvote a disqualifying size
or price — CDSL scored 6.31 with Size 2.96 and Price 3.35. The 100x evidence
base describes jointly necessary conditions, so eligibility is evaluated as
hard gates alongside the composite rather than folded into it.
"""

import pytest

from boundless100x.compute_engine.eligibility import (
    DEFAULT_GATES,
    EligibilityEvaluator,
)
from boundless100x.compute_engine.metrics.base import MetricResult


def shipped_gates() -> dict:
    """The YAML gates production actually runs, not the Python constant."""
    from boundless100x.compute_engine.engine import ComputeEngine

    return ComputeEngine().gates


def evaluator(gates: dict | None = None) -> EligibilityEvaluator:
    return EligibilityEvaluator(gates if gates is not None else shipped_gates())


def passing_metrics(**overrides) -> dict:
    metrics = {
        "market_cap": MetricResult(value=8_000.0),
        "trailing_peg": MetricResult(value=1.2),
        "peg_ratio": MetricResult(value=1.1),
        "roiic": MetricResult(value=28.0),
        # Veto source for the price gate: ran fine, emitted no flag.
        "reverse_dcf_growth": MetricResult(value=12.0),
    }
    metrics.update(overrides)
    return metrics


class TestVerdicts:
    def test_all_gates_passing_is_eligible(self):
        verdict = evaluator().evaluate(passing_metrics())

        assert verdict["eligible"] is True
        assert verdict["verdict"] == "eligible"
        assert verdict["failed"] == []

    def test_every_gate_reports_its_own_detail(self):
        gates = evaluator().evaluate(passing_metrics())["gates"]

        assert set(gates) == set(shipped_gates())
        for detail in gates.values():
            assert "label" in detail and "passed" in detail and "reason" in detail

    def test_large_cap_fails_only_the_size_gate(self):
        """The CDSL shape: excellent business, disqualifying size."""
        verdict = evaluator().evaluate(passing_metrics(market_cap=MetricResult(value=95_000.0)))

        assert verdict["eligible"] is False
        assert verdict["failed"] == ["size"]
        assert "size" in verdict["gates"]
        assert verdict["gates"]["size"]["passed"] is False

    def test_failure_reason_names_the_threshold(self):
        verdict = evaluator().evaluate(passing_metrics(market_cap=MetricResult(value=95_000.0)))

        assert "30,000" in verdict["gates"]["size"]["reason"] or "30000" in verdict["gates"]["size"]["reason"]

    def test_expensive_entry_fails_the_price_gate(self):
        verdict = evaluator().evaluate(passing_metrics(
            trailing_peg=MetricResult(value=4.0),
            peg_ratio=MetricResult(value=3.5),
        ))

        assert verdict["eligible"] is False
        assert "price" in verdict["failed"]

    def test_weak_incremental_returns_fail_the_reinvestment_gate(self):
        verdict = evaluator().evaluate(passing_metrics(roiic=MetricResult(value=3.0)))

        assert "reinvestment" in verdict["failed"]

    def test_multiple_failures_are_all_reported(self):
        verdict = evaluator().evaluate(passing_metrics(
            market_cap=MetricResult(value=95_000.0),
            roiic=MetricResult(value=1.0),
        ))

        assert set(verdict["failed"]) == {"size", "reinvestment"}


class TestAnyOfConditions:
    def test_either_peg_measure_can_satisfy_the_price_gate(self):
        verdict = evaluator().evaluate(passing_metrics(
            trailing_peg=MetricResult(value=2.6),   # fails on its own
            peg_ratio=MetricResult(value=1.2),      # but the 5yr-EPS trailing PEG is fine
        ))

        assert verdict["gates"]["price"]["passed"] is True


class TestVetoFlags:
    def test_overpriced_flag_fails_the_price_gate_despite_a_passing_ratio(self):
        verdict = evaluator().evaluate(passing_metrics(
            trailing_peg=MetricResult(value=1.0, flags=["reverse_dcf_overpriced"]),
        ))

        assert verdict["gates"]["price"]["passed"] is False
        assert "reverse_dcf_overpriced" in verdict["gates"]["price"]["reason"]

    def test_veto_flag_on_any_contributing_metric_counts(self):
        verdict = evaluator().evaluate(passing_metrics(
            peg_ratio=MetricResult(value=1.0, flags=["reverse_dcf_overpriced"]),
        ))

        assert verdict["gates"]["price"]["passed"] is False


class TestVetoSourceAvailability:
    """A veto whose source metric never ran is not evidence of affordability."""

    def test_errored_veto_source_makes_the_gate_indeterminate(self):
        verdict = evaluator().evaluate(passing_metrics(
            reverse_dcf_growth=MetricResult(error="Negative average FCF"),
        ))

        assert verdict["gates"]["price"]["passed"] is None
        assert verdict["verdict"] == "indeterminate"
        assert "price" in verdict["indeterminate"]
        assert "reverse_dcf_growth" in verdict["gates"]["price"]["reason"]

    def test_missing_veto_source_makes_the_gate_indeterminate(self):
        metrics = passing_metrics()
        del metrics["reverse_dcf_growth"]

        verdict = evaluator().evaluate(metrics)

        assert verdict["gates"]["price"]["passed"] is None
        assert verdict["verdict"] == "indeterminate"

    def test_available_veto_source_without_flag_lets_conditions_decide(self):
        """The source ran and did not flag: the PEG conditions rule."""
        verdict = evaluator().evaluate(passing_metrics())

        assert verdict["gates"]["price"]["passed"] is True

    def test_gates_without_veto_sources_keep_flag_only_behavior(self):
        gates = {"g": {
            "label": "G",
            "veto_flags": ["some_flag"],
            "conditions": [{"metric": "market_cap", "comparator": "lt", "threshold": 30000}],
        }}

        verdict = EligibilityEvaluator(gates).evaluate(passing_metrics())

        assert verdict["gates"]["g"]["passed"] is True


class TestIndeterminate:
    def test_missing_metric_is_indeterminate_not_a_pass(self):
        metrics = passing_metrics()
        del metrics["roiic"]

        verdict = evaluator().evaluate(metrics)

        assert verdict["eligible"] is not True
        assert verdict["verdict"] == "indeterminate"
        assert "reinvestment" in verdict["indeterminate"]

    def test_errored_metric_is_indeterminate(self):
        verdict = evaluator().evaluate(passing_metrics(
            roiic=MetricResult(error="Capital base flat"),
        ))

        assert verdict["verdict"] == "indeterminate"

    def test_a_real_failure_outranks_an_indeterminate_gate(self):
        """A known disqualification is decisive even if another gate is unknown."""
        metrics = passing_metrics(market_cap=MetricResult(value=95_000.0))
        del metrics["roiic"]

        verdict = evaluator().evaluate(metrics)

        assert verdict["eligible"] is False
        assert verdict["verdict"] == "not_eligible"


class TestYamlDriven:
    def test_thresholds_come_from_config_not_code(self):
        loose = {"size": {"label": "Size", "conditions": [
            {"metric": "market_cap", "comparator": "lt", "threshold": 200_000}]}}

        verdict = EligibilityEvaluator(loose).evaluate(passing_metrics(
            market_cap=MetricResult(value=95_000.0)))

        assert verdict["eligible"] is True

    def test_shipped_yaml_matches_the_python_defaults(self):
        """Two representations of the same contract must not drift apart."""
        assert shipped_gates() == DEFAULT_GATES

    def test_unknown_comparator_is_indeterminate_not_a_silent_pass(self):
        broken = {"g": {"label": "G", "conditions": [
            {"metric": "market_cap", "comparator": "wat", "threshold": 1}]}}

        verdict = EligibilityEvaluator(broken).evaluate(passing_metrics())

        assert verdict["eligible"] is not True


class TestReportIntegration:
    def test_badge_and_gate_detail_render_in_html(self, tmp_path):
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        result = make_result()
        result.eligibility = evaluator().evaluate(passing_metrics(
            market_cap=MetricResult(value=95_000.0)))

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(result, formats=["html", "md", "json"])
        html = (report_dir / f"{result.ticker}_dashboard.html").read_text()

        assert "Not a 100x Candidate" in html
        assert "Size headroom" in html
        assert (report_dir / "eligibility.json").exists()

    def test_report_without_eligibility_still_renders(self, tmp_path):
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            make_result(), formats=["html"])

        assert (report_dir / "TEST_dashboard.html").exists()
