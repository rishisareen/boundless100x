"""Trigger evaluation, and the indeterminate rule that keeps it honest.

The property under test throughout: a trigger whose inputs are missing is
indeterminate, never fired and never quietly false. A kill-switch that cannot
be evaluated must read as unknown, because "we could not check" is not the
same as "the thesis is fine" — and silence is how a broken thesis survives.
"""

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import states
from boundless100x.lifecycle.evaluator import (
    TriggerEvaluator,
    load_triggers,
    validate_triggers,
)


def metric(value=None, *, flags=None, series=None, error=None) -> MetricResult:
    return MetricResult(
        value=value, flags=flags or [], raw_series=series or [], error=error
    )


def trigger(**overrides) -> dict:
    spec = {
        "label": "Test trigger",
        "from": ["watch"],
        "to": "probe",
        "mode": "all",
        "conditions": [{"metric": "trailing_peg", "comparator": "lte", "threshold": 2.0}],
    }
    spec.update(overrides)
    return {"t": spec}


def evaluate(spec: dict, state="watch", **context) -> dict:
    return TriggerEvaluator(spec).evaluate(state, **{"metrics": {}, **context})["triggers"]["t"]


class TestMetricConditions:
    def test_a_met_condition_fires(self):
        detail = evaluate(trigger(), metrics={"trailing_peg": metric(1.4)})
        assert detail["fired"] is True

    def test_an_unmet_condition_does_not_fire(self):
        detail = evaluate(trigger(), metrics={"trailing_peg": metric(3.0)})
        assert detail["fired"] is False

    def test_a_missing_metric_is_indeterminate_not_false(self):
        detail = evaluate(trigger(), metrics={})
        assert detail["fired"] is None
        assert "not computed" in detail["conditions"][0]["detail"]

    def test_an_errored_metric_is_indeterminate(self):
        detail = evaluate(trigger(), metrics={"trailing_peg": metric(error="no earnings")})
        assert detail["fired"] is None

    def test_a_non_numeric_metric_is_indeterminate(self):
        detail = evaluate(trigger(), metrics={"trailing_peg": metric("cheap")})
        assert detail["fired"] is None

    def test_the_reason_names_the_number_that_decided_it(self):
        detail = evaluate(trigger(), metrics={"trailing_peg": metric(1.4)})
        assert "1.40" in detail["reason"] and "trailing_peg" in detail["reason"]


class TestModes:
    def test_all_requires_every_condition(self):
        spec = trigger(conditions=[
            {"metric": "trailing_peg", "comparator": "lte", "threshold": 2.0},
            {"metric": "pe_vs_historical", "comparator": "lte", "threshold": 60},
        ])
        detail = evaluate(spec, metrics={
            "trailing_peg": metric(1.4), "pe_vs_historical": metric(80)
        })
        assert detail["fired"] is False

    def test_any_needs_only_one(self):
        spec = trigger(mode="any", conditions=[
            {"metric": "trailing_peg", "comparator": "lte", "threshold": 2.0},
            {"metric": "pe_vs_historical", "comparator": "lte", "threshold": 60},
        ])
        detail = evaluate(spec, metrics={
            "trailing_peg": metric(1.4), "pe_vs_historical": metric(80)
        })
        assert detail["fired"] is True

    def test_a_known_failure_outranks_an_unknown_under_all(self):
        spec = trigger(conditions=[
            {"metric": "trailing_peg", "comparator": "lte", "threshold": 2.0},
            {"metric": "absent_metric", "comparator": "lte", "threshold": 1},
        ])
        detail = evaluate(spec, metrics={"trailing_peg": metric(9.0)})
        assert detail["fired"] is False


class TestPersistYears:
    """Consecutive-period rules read raw_series — there is no roce_latest."""

    def spec(self):
        return trigger(conditions=[{
            "metric": "roce_5yr_avg", "comparator": "lt",
            "threshold": 15, "persist_years": 2,
        }])

    def test_fires_when_every_recent_period_breaches(self):
        detail = evaluate(self.spec(), metrics={
            "roce_5yr_avg": metric(14.0, series=[22.0, 19.0, 12.0, 11.0])
        })
        assert detail["fired"] is True

    def test_does_not_fire_when_only_the_latest_period_breaches(self):
        """One bad year is not a broken compounding engine."""
        detail = evaluate(self.spec(), metrics={
            "roce_5yr_avg": metric(18.0, series=[22.0, 19.0, 18.0, 11.0])
        })
        assert detail["fired"] is False

    def test_a_series_shorter_than_the_window_is_indeterminate(self):
        detail = evaluate(self.spec(), metrics={
            "roce_5yr_avg": metric(11.0, series=[11.0])
        })
        assert detail["fired"] is None
        assert "needs 2" in detail["conditions"][0]["detail"]

    def test_the_mean_alone_never_decides_a_persistence_rule(self):
        """A 5yr mean above the threshold can still hide two bad recent years."""
        detail = evaluate(self.spec(), metrics={
            "roce_5yr_avg": metric(25.0, series=[40.0, 38.0, 12.0, 11.0])
        })
        assert detail["fired"] is True

    def test_the_evidence_shows_the_periods(self):
        detail = evaluate(self.spec(), metrics={
            "roce_5yr_avg": metric(14.0, series=[12.0, 11.0])
        })
        assert detail["conditions"][0]["series"] == [12.0, 11.0]


class TestFlagConditions:
    def spec(self, **kw):
        condition = {"flag_absent": "reverse_dcf_overpriced",
                     "sources": ["reverse_dcf_growth"]}
        condition.update(kw)
        return trigger(conditions=[condition])

    def test_absent_flag_with_a_working_source_passes(self):
        detail = evaluate(self.spec(), metrics={"reverse_dcf_growth": metric(18.0)})
        assert detail["fired"] is True

    def test_present_flag_blocks(self):
        detail = evaluate(self.spec(), metrics={
            "reverse_dcf_growth": metric(45.0, flags=["reverse_dcf_overpriced"])
        })
        assert detail["fired"] is False

    def test_absence_is_unconfirmed_when_the_source_did_not_run(self):
        """The price gate's rule: a flag that could not be emitted proves nothing."""
        detail = evaluate(self.spec(), metrics={
            "reverse_dcf_growth": metric(error="no FCF")
        })
        assert detail["fired"] is None
        assert "unconfirmed" in detail["conditions"][0]["detail"]

    def test_flag_present_fires_on_the_flag(self):
        spec = trigger(conditions=[{"flag_present": "growth_quality_risky"}])
        detail = evaluate(spec, metrics={
            "growth_quality_grade": metric("risky", flags=["growth_quality_risky"])
        })
        assert detail["fired"] is True

    def test_the_carrier_metric_is_named_as_evidence(self):
        spec = trigger(conditions=[{"flag_present": "growth_quality_risky"}])
        detail = evaluate(spec, metrics={
            "growth_quality_grade": metric("risky", flags=["growth_quality_risky"])
        })
        assert detail["conditions"][0]["carriers"] == ["growth_quality_grade"]


class TestScoreAndVerdictConditions:
    def test_composite_threshold(self):
        spec = trigger(conditions=[
            {"score": "composite", "comparator": "gte", "threshold": 5.5}
        ])
        assert evaluate(spec, scores={"composite": 6.4})["fired"] is True
        assert evaluate(spec, scores={"composite": 4.0})["fired"] is False

    def test_missing_scores_are_indeterminate(self):
        spec = trigger(conditions=[
            {"score": "composite", "comparator": "gte", "threshold": 5.5}
        ])
        assert evaluate(spec, scores=None)["fired"] is None

    def test_element_scores_are_addressable(self):
        spec = trigger(conditions=[
            {"score": "growth", "comparator": "gte", "threshold": 7.0}
        ])
        detail = evaluate(spec, scores={"composite": 6.0, "elements": {"growth": 7.4}})
        assert detail["fired"] is True

    def test_verdict_equality(self):
        spec = trigger(conditions=[{"verdict": "eligible"}])
        assert evaluate(spec, eligibility={"verdict": "eligible"})["fired"] is True
        assert evaluate(spec, eligibility={"verdict": "not_eligible"})["fired"] is False

    def test_an_unevaluated_verdict_is_indeterminate_not_a_pass(self):
        spec = trigger(conditions=[{"verdict": "eligible"}])
        assert evaluate(spec, eligibility=None)["fired"] is None


class TestCheckpointConditions:
    def spec(self):
        return trigger(conditions=[
            {"checkpoint": "missed", "comparator": "gte", "threshold": 2}
        ])

    def test_fires_on_enough_misses(self):
        summary = {"missed": 3, "due": 3, "total": 3}
        assert evaluate(self.spec(), checkpoint_results=summary)["fired"] is True

    def test_does_not_fire_below_the_threshold(self):
        summary = {"missed": 1, "due": 3, "total": 3}
        assert evaluate(self.spec(), checkpoint_results=summary)["fired"] is False

    def test_no_checkpoints_recorded_is_indeterminate(self):
        assert evaluate(self.spec(), checkpoint_results=None)["fired"] is None

    def test_zero_misses_out_of_zero_due_is_not_a_clean_bill_of_health(self):
        """An unchecked thesis must not read like a verified one."""
        summary = {"missed": 0, "due": 0, "total": 0}
        detail = evaluate(self.spec(), checkpoint_results=summary)
        assert detail["fired"] is None
        assert "no checkpoints recorded" in detail["conditions"][0]["detail"]

    def test_recorded_but_not_yet_due_is_also_indeterminate(self):
        summary = {"missed": 0, "due": 0, "total": 3, "pending": 3}
        detail = evaluate(self.spec(), checkpoint_results=summary)
        assert detail["fired"] is None
        assert "come due" in detail["conditions"][0]["detail"]


class TestStateFiltering:
    def test_only_triggers_declared_from_this_state_are_evaluated(self):
        evaluator = TriggerEvaluator(trigger(**{"from": ["watch"]}))
        assert evaluator.evaluate("watch", metrics={})["triggers"]
        assert evaluator.evaluate("scale", metrics={})["triggers"] == {}

    def test_from_any_applies_everywhere(self):
        evaluator = TriggerEvaluator(trigger(**{"from": "any"}))
        assert all(evaluator.evaluate(s, metrics={})["triggers"] for s in states.STATES)

    def test_fired_and_indeterminate_are_reported_separately(self):
        spec = {
            "hit": {**trigger()["t"], "conditions": [
                {"score": "composite", "comparator": "gte", "threshold": 1}]},
            "unknown": {**trigger()["t"], "conditions": [
                {"metric": "absent", "comparator": "lt", "threshold": 1}]},
        }
        result = TriggerEvaluator(spec).evaluate(
            "watch", metrics={}, scores={"composite": 6.0}
        )
        assert result["fired"] == ["hit"]
        assert result["indeterminate"] == ["unknown"]


class TestRegistryValidation:
    def test_unknown_destination_state_is_rejected(self):
        assert validate_triggers(trigger(to="nowhere"))

    def test_unknown_origin_state_is_rejected(self):
        assert validate_triggers(trigger(**{"from": ["nowhere"]}))

    def test_unknown_comparator_is_rejected(self):
        assert validate_triggers(trigger(conditions=[
            {"metric": "trailing_peg", "comparator": "approximately", "threshold": 1}
        ]))

    def test_unknown_metric_id_is_rejected_when_the_registry_is_known(self):
        """A trigger naming a nonexistent metric would be indeterminate forever."""
        errors = validate_triggers(trigger(), known_metric_ids={"market_cap"})
        assert any("unknown metric id" in e for e in errors)

    def test_a_condition_with_no_recognised_kind_is_rejected(self):
        assert validate_triggers(trigger(conditions=[{"threshold": 1}]))

    def test_a_condition_with_two_kinds_is_rejected(self):
        assert validate_triggers(trigger(conditions=[
            {"metric": "trailing_peg", "score": "composite",
             "comparator": "lt", "threshold": 1}
        ]))

    def test_a_trigger_with_no_conditions_is_rejected(self):
        assert validate_triggers(trigger(conditions=[]))

    def test_persist_years_must_be_at_least_two(self):
        assert validate_triggers(trigger(conditions=[
            {"metric": "roce_5yr_avg", "comparator": "lt",
             "threshold": 15, "persist_years": 1}
        ]))

    def test_construction_raises_on_an_invalid_registry(self):
        with pytest.raises(ValueError, match="validation failed"):
            TriggerEvaluator(trigger(to="nowhere"))


class TestShippedRegistry:
    def setup_method(self):
        self.triggers = load_triggers()

    def test_the_shipped_registry_is_valid(self):
        assert self.triggers
        assert validate_triggers(self.triggers) == []

    def test_it_validates_against_the_real_metric_registry(self):
        """Guards the pe_percentile_10y class of error at startup."""
        from boundless100x.compute_engine.engine import ComputeEngine

        assert validate_triggers(self.triggers, set(ComputeEngine().metrics)) == []

    def test_every_destination_is_a_known_state(self):
        assert all(states.is_state(s["to"]) for s in self.triggers.values())

    def test_position_opening_transitions_are_not_auto_applicable(self):
        """Entering probe deploys capital — it must never apply itself."""
        for spec in self.triggers.values():
            if spec["to"] in states.POSITIONED:
                assert states.moves_money(spec["to"])
