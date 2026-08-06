"""The core-lane kill-switches, and the series trap they had to avoid.

A kill-switch that never fires is indistinguishable from a thesis that never
broke, so these tests check both directions for every switch: it fires on a
real breach, stays silent on a healthy company, and reads indeterminate when
its inputs are missing.
"""

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.evaluator import (
    SERIES_SAFE_METRICS,
    TriggerEvaluator,
    load_triggers,
    validate_triggers,
)

KILL_SWITCHES = (
    "capital_efficiency_break",
    "growth_quality_degradation",
    "incremental_return_break",
    "valuation_saturation",
    "governance_event",
    "checkpoints_failed",
)


def metric(value=None, *, flags=None, series=None, error=None) -> MetricResult:
    return MetricResult(
        value=value, flags=flags or [], raw_series=series or [], error=error
    )


def healthy() -> dict:
    return {
        "roce_5yr_avg": metric(24.0, series=[23.0, 24.0, 25.0, 24.0, 24.0]),
        "roiic": metric(28.0, series=[4000.0, 4500.0, 5000.0]),
        "growth_quality_grade": metric("high_quality", flags=["growth_quality_high_quality"]),
        "pe_vs_historical": metric(42.0, series=[20.0, 30.0, 40.0]),
        "reverse_dcf_growth": metric(14.0),
        "promoter_pledge": metric(0.0),
    }


@pytest.fixture
def evaluator():
    return TriggerEvaluator(load_triggers())


def fired(evaluator, metrics, state="scale", checkpoints=None) -> list[str]:
    return evaluator.evaluate(
        state, metrics=metrics, checkpoint_results=checkpoints
    )["fired"]


def detail(evaluator, trigger_id, metrics, state="scale", checkpoints=None) -> dict:
    return evaluator.evaluate(
        state, metrics=metrics, checkpoint_results=checkpoints
    )["triggers"][trigger_id]


class TestHealthyCompany:
    def test_no_kill_switch_fires_on_a_healthy_holding(self, evaluator):
        checkpoints = {"met": 3, "missed": 0, "due": 3, "total": 3}
        assert fired(evaluator, healthy(), checkpoints=checkpoints) == []

    def test_every_declared_switch_is_evaluated_from_scale(self, evaluator):
        evaluated = evaluator.evaluate("scale", metrics=healthy())["triggers"]
        assert set(KILL_SWITCHES) <= set(evaluated)

    def test_none_are_evaluated_before_a_position_exists(self, evaluator):
        """Nothing is held in `watch`, so there is no position to review."""
        evaluated = evaluator.evaluate("watch", metrics=healthy())["triggers"]
        assert not set(KILL_SWITCHES) & set(evaluated)


class TestCapitalEfficiencyBreak:
    def test_fires_on_two_consecutive_sub_threshold_years(self, evaluator):
        metrics = healthy()
        metrics["roce_5yr_avg"] = metric(19.0, series=[25.0, 24.0, 23.0, 12.0, 11.0])
        assert "capital_efficiency_break" in fired(evaluator, metrics)

    def test_one_weak_year_is_noise_not_a_break(self, evaluator):
        metrics = healthy()
        metrics["roce_5yr_avg"] = metric(21.0, series=[25.0, 24.0, 23.0, 22.0, 11.0])
        assert "capital_efficiency_break" not in fired(evaluator, metrics)

    def test_a_healthy_mean_cannot_hide_two_bad_recent_years(self, evaluator):
        """The reason the switch reads the series and not the five-year mean."""
        metrics = healthy()
        metrics["roce_5yr_avg"] = metric(30.0, series=[55.0, 50.0, 45.0, 12.0, 11.0])
        assert "capital_efficiency_break" in fired(evaluator, metrics)

    def test_missing_series_is_indeterminate_not_silent(self, evaluator):
        metrics = healthy()
        metrics["roce_5yr_avg"] = metric(11.0, series=[])
        assert detail(evaluator, "capital_efficiency_break", metrics)["fired"] is None


class TestOtherSwitches:
    def test_growth_quality_degradation_fires_on_the_risky_grade(self, evaluator):
        metrics = healthy()
        metrics["growth_quality_grade"] = metric("risky", flags=["growth_quality_risky"])
        assert "growth_quality_degradation" in fired(evaluator, metrics)

    def test_incremental_return_break_fires_below_cost_of_capital(self, evaluator):
        metrics = healthy()
        metrics["roiic"] = metric(6.0)
        assert "incremental_return_break" in fired(evaluator, metrics)

    def test_incremental_return_break_is_indeterminate_when_roiic_errored(self, evaluator):
        """A flat capital base makes ROIIC undefined — not a pass, not a fail."""
        metrics = healthy()
        metrics["roiic"] = metric(error="Capital base flat")
        assert detail(evaluator, "incremental_return_break", metrics)["fired"] is None

    def test_valuation_saturation_needs_both_limbs(self, evaluator):
        metrics = healthy()
        metrics["pe_vs_historical"] = metric(98.0)
        assert "valuation_saturation" not in fired(evaluator, metrics)

        metrics["reverse_dcf_growth"] = metric(45.0, flags=["reverse_dcf_overpriced"])
        assert "valuation_saturation" in fired(evaluator, metrics)

    def test_governance_event_fires_on_the_pledge_flag(self, evaluator):
        metrics = healthy()
        metrics["promoter_pledge"] = metric(18.0, flags=["promoter_pledge_red_flag"])
        assert "governance_event" in fired(evaluator, metrics)

    def test_governance_is_indeterminate_when_pledge_is_unknown(self, evaluator):
        """An unpledged promoter and an unknown one must not look identical."""
        metrics = healthy()
        metrics["promoter_pledge"] = metric(error="BSE did not supply pledge")
        assert detail(evaluator, "governance_event", metrics)["fired"] is None

    def test_checkpoints_failed_fires_on_two_misses(self, evaluator):
        assert "checkpoints_failed" in fired(
            evaluator, healthy(), checkpoints={"missed": 2, "due": 3, "total": 3}
        )

    def test_one_miss_is_not_enough(self, evaluator):
        assert "checkpoints_failed" not in fired(
            evaluator, healthy(), checkpoints={"missed": 1, "due": 3, "total": 3}
        )

    def test_no_checkpoints_recorded_is_indeterminate(self, evaluator):
        assert detail(evaluator, "checkpoints_failed", healthy())["fired"] is None

    def test_an_unmonitored_position_is_unknown_not_clear(self, evaluator):
        """Zero misses out of zero due means nobody checked, not all is well."""
        outcome = detail(
            evaluator, "checkpoints_failed", healthy(),
            checkpoints={"missed": 0, "due": 0, "total": 0},
        )
        assert outcome["fired"] is None


class TestPrePositionDeterioration:
    def test_a_watched_company_is_dropped_not_exit_reviewed(self, evaluator):
        metrics = healthy()
        metrics["roiic"] = metric(3.0)
        result = evaluator.evaluate("watch", metrics=metrics)

        assert "fundamentals_deteriorated" in result["fired"]
        assert result["triggers"]["fundamentals_deteriorated"]["to"] == "dropped"

    def test_any_single_breach_is_enough_before_entry(self, evaluator):
        metrics = healthy()
        metrics["promoter_pledge"] = metric(20.0, flags=["promoter_pledge_red_flag"])
        assert "fundamentals_deteriorated" in evaluator.evaluate(
            "watch", metrics=metrics
        )["fired"]

    def test_a_healthy_watched_company_is_not_dropped(self, evaluator):
        assert "fundamentals_deteriorated" not in evaluator.evaluate(
            "watch", metrics=healthy()
        )["fired"]


class TestSeriesSafety:
    """persist_years may only read a metric whose series is its own values."""

    def test_roiic_is_rejected_for_persist_years(self):
        """Its raw_series is capital employed (INR Cr), not yearly ROIIC (%)."""
        errors = validate_triggers({"t": {
            "label": "T", "from": ["scale"], "to": "exit_review",
            "conditions": [{"metric": "roiic", "comparator": "lt",
                            "threshold": 12, "persist_years": 2}],
        }})
        assert any("persist_years is not available" in e for e in errors)

    def test_pe_vs_historical_is_rejected_for_persist_years(self):
        """Its series is P/E multiples; its value is a 0-100 percentile."""
        errors = validate_triggers({"t": {
            "label": "T", "from": ["scale"], "to": "exit_review",
            "conditions": [{"metric": "pe_vs_historical", "comparator": "gt",
                            "threshold": 95, "persist_years": 2}],
        }})
        assert any("persist_years is not available" in e for e in errors)

    def test_the_allowlist_holds_only_verified_metrics(self):
        assert "roce_5yr_avg" in SERIES_SAFE_METRICS
        assert "roiic" not in SERIES_SAFE_METRICS
        assert "pe_vs_historical" not in SERIES_SAFE_METRICS


class TestShippedRegistry:
    def test_the_registry_including_kill_switches_is_valid(self):
        from boundless100x.compute_engine.engine import ComputeEngine

        assert validate_triggers(load_triggers(), set(ComputeEngine().metrics)) == []

    def test_every_kill_switch_routes_to_exit_review_not_a_sale(self):
        """The system proposes; the owner disposes. No switch sells."""
        triggers = load_triggers()
        for switch in KILL_SWITCHES:
            assert triggers[switch]["to"] == "exit_review"

    def test_kill_switches_only_apply_to_held_positions(self):
        triggers = load_triggers()
        for switch in KILL_SWITCHES:
            assert set(triggers[switch]["from"]) == {"probe", "scale"}
