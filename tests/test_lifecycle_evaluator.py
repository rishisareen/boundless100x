"""Trigger evaluation, and the indeterminate rule that keeps it honest.

The property under test throughout: a trigger whose inputs are missing is
indeterminate, never fired and never quietly false. A kill-switch that cannot
be evaluated must read as unknown, because "we could not check" is not the
same as "the thesis is fine" — and silence is how a broken thesis survives.

Phase 3 adds three condition kinds and one new axis of applicability, and each
brings its own version of the same rule. `lane_verdict` reads the fast lane's
gate result; `catalyst_status` reads owner judgement no metric can compute;
`since_state_entry` reads the clock against the append-only state history. Lane
filtering is the second axis — `from: [any]` already says "every origin state",
and an absent `lane` key now says "every lane" the same way.
"""

from datetime import date, timedelta

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import states
from boundless100x.lifecycle.evaluator import (
    LANE_VERDICTS,
    TriggerEvaluator,
    load_triggers,
    validate_triggers,
)
from boundless100x.watchlist import LANES

# A fixed run date. Every `since_state_entry` fixture is dated relative to this
# and never to `date.today()`, so the suite reads the same on any day it runs —
# which is also the property one of these tests exists to prove.
AS_OF = date(2026, 8, 7)


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


class TestLaneVerdictConditions:
    """The fast lane's gate result, read as one condition.

    This is what lets a trigger say "all lane gates pass" without copying the
    gate list into the trigger registry, where the two would drift apart and
    nobody would find out.
    """

    def spec(self, expected="qualifies"):
        return trigger(conditions=[{"lane_verdict": expected}])

    def test_fires_when_the_lane_gates_qualify(self):
        detail = evaluate(self.spec(), lane_gate_result={"verdict": "qualifies"})
        assert detail["fired"] is True

    def test_does_not_fire_when_the_lane_gates_did_not_qualify(self):
        detail = evaluate(self.spec(), lane_gate_result={"verdict": "not_qualified"})
        assert detail["fired"] is False

    def test_an_unevaluated_lane_verdict_is_indeterminate_not_a_pass(self):
        """Mirrors the 100x verdict exactly: absent inputs never clear a gate."""
        detail = evaluate(self.spec(), lane_gate_result=None)
        assert detail["fired"] is None
        assert "not evaluated" in detail["conditions"][0]["detail"]

    def test_a_result_carrying_no_verdict_is_also_indeterminate(self):
        detail = evaluate(self.spec(), lane_gate_result={"gates": {}})
        assert detail["fired"] is None

    def test_an_indeterminate_lane_verdict_is_itself_addressable(self):
        """A trigger may legitimately ask "were the lane gates unreadable?"."""
        detail = evaluate(
            self.spec("indeterminate"), lane_gate_result={"verdict": "indeterminate"}
        )
        assert detail["fired"] is True

    def test_the_verdict_is_named_in_the_evidence(self):
        detail = evaluate(self.spec(), lane_gate_result={"verdict": "not_qualified"})
        assert "not_qualified" in detail["reason"]


class TestCatalystStatusConditions:
    """Owner judgement, and the two empty cases that must not collapse."""

    def spec(self, expected="spent"):
        return trigger(conditions=[{"catalyst_status": expected}])

    def test_fires_on_the_exact_status(self):
        assert evaluate(self.spec(), catalyst={"status": "spent"})["fired"] is True

    def test_a_different_status_does_not_fire(self):
        assert evaluate(self.spec(), catalyst={"status": "active"})["fired"] is False

    def test_an_entry_with_no_catalyst_recorded_is_a_plain_false(self):
        """Somebody looked at this company and recorded no catalyst.

        "Not yet identified" is a known fact about the entry, not a gap in the
        data — so it is False, and a trigger waiting on a spent catalyst
        correctly stays quiet rather than reading unknown forever.
        """
        detail = evaluate(self.spec(), catalyst={})
        assert detail["fired"] is False
        assert "no catalyst recorded" in detail["conditions"][0]["detail"]

    def test_no_watchlist_context_at_all_is_indeterminate(self):
        """`{}` is falsy too — only `is None` keeps these two apart."""
        detail = evaluate(self.spec(), catalyst=None)
        assert detail["fired"] is None
        assert "unknown" in detail["conditions"][0]["detail"]

    def test_the_two_empty_cases_do_not_read_the_same(self):
        """The whole point, stated as one assertion."""
        assert evaluate(self.spec(), catalyst={})["fired"] is False
        assert evaluate(self.spec(), catalyst=None)["fired"] is None


class TestSinceStateEntry:
    """The 18-month time stop: how long has this company sat where it is?"""

    def spec(self, target="probe", comparator="gte", threshold=545):
        return trigger(conditions=[{
            "since_state_entry": target,
            "comparator": comparator,
            "threshold": threshold,
        }])

    def history(self, days_ago, to="probe", as_of=AS_OF):
        return [{
            "at": (as_of - timedelta(days=days_ago)).isoformat(),
            "from": "watch",
            "to": to,
            "trigger_id": "buy_zone",
            "evidence": "",
            "applied_by": "owner",
        }]

    def test_a_long_stalled_probe_fires_the_time_stop(self):
        detail = evaluate(self.spec(), state_history=self.history(550), as_of=AS_OF)
        assert detail["fired"] is True

    def test_a_recent_probe_does_not(self):
        detail = evaluate(self.spec(), state_history=self.history(100), as_of=AS_OF)
        assert detail["fired"] is False

    def test_never_having_reached_the_state_is_indeterminate(self):
        """Not zero days. A company that never entered probe has no clock."""
        detail = evaluate(
            self.spec(), state_history=self.history(550, to="watch"), as_of=AS_OF
        )
        assert detail["fired"] is None
        assert "never reached probe" in detail["conditions"][0]["detail"]

    def test_an_empty_history_is_indeterminate(self):
        assert evaluate(self.spec(), state_history=[], as_of=AS_OF)["fired"] is None

    def test_no_history_supplied_is_indeterminate(self):
        assert evaluate(self.spec(), state_history=None, as_of=AS_OF)["fired"] is None

    def test_the_most_recent_entry_into_the_state_wins(self):
        """Re-entering probe restarts the clock — an old visit is not the stop."""
        history = self.history(900) + self.history(30)
        assert evaluate(self.spec(), state_history=history, as_of=AS_OF)["fired"] is False

    def test_other_transitions_in_between_do_not_reset_it(self):
        history = (
            self.history(550)
            + self.history(400, to="exit_review")
            + self.history(300, to="watch")
        )
        assert evaluate(self.spec(), state_history=history, as_of=AS_OF)["fired"] is True

    def test_it_reads_the_as_of_it_was_given_not_the_wall_clock(self):
        """A replay must be deterministic on any day the suite happens to run.

        The same fixture fires or does not fire depending solely on `as_of`; a
        wall-clock reading would fire on both, since AS_OF itself is in the past
        by the time anyone reads this.
        """
        history = self.history(550)
        assert evaluate(self.spec(), state_history=history, as_of=AS_OF)["fired"] is True
        assert evaluate(
            self.spec(), state_history=history, as_of=AS_OF - timedelta(days=100)
        )["fired"] is False

    def test_an_unreadable_timestamp_is_indeterminate(self):
        history = [{"at": "sometime last year", "from": "watch", "to": "probe"}]
        assert evaluate(self.spec(), state_history=history, as_of=AS_OF)["fired"] is None

    def test_the_evidence_states_the_day_count(self):
        detail = evaluate(self.spec(), state_history=self.history(550), as_of=AS_OF)
        assert "550" in detail["conditions"][0]["detail"]
        assert detail["conditions"][0]["value"] == 550.0


class TestLaneFiltering:
    """`lane` is `from`'s second axis: absent means universal."""

    def test_a_lane_scoped_trigger_is_absent_from_another_lane(self):
        evaluator = TriggerEvaluator(trigger(lane=["rerating"]))
        assert "t" not in evaluator.applicable("watch", lane="core")
        assert "t" in evaluator.applicable("watch", lane="rerating")

    def test_a_trigger_with_no_lane_key_applies_to_every_lane(self):
        evaluator = TriggerEvaluator(trigger())
        assert all("t" in evaluator.applicable("watch", lane=lane) for lane in LANES)

    def test_evaluate_forwards_the_lane_to_applicable(self):
        """`applicable` gaining the parameter alone is not the feature.

        `evaluate` is the only caller and the only thing the orchestrator
        invokes, so a lane that stops at `applicable`'s signature is filtering
        that silently never happens. Asserting on the returned `triggers` keys
        rather than on `applicable` is what makes the forward observable.
        """
        evaluator = TriggerEvaluator(trigger(lane=["rerating"]))
        assert evaluator.evaluate("watch", metrics={}, lane="core")["triggers"] == {}
        assert "t" in evaluator.evaluate("watch", metrics={}, lane="rerating")["triggers"]

    def test_a_universal_trigger_is_evaluated_in_both_lanes(self):
        evaluator = TriggerEvaluator(trigger())
        for lane in LANES:
            assert "t" in evaluator.evaluate("watch", metrics={}, lane=lane)["triggers"]

    def test_no_lane_context_evaluates_every_trigger(self):
        """An unknown lane must not silence a kill-switch.

        Filtering lane-scoped triggers out when the caller supplied no lane
        would make them unevaluable rather than unknown — and a kill-switch
        that never fires looks exactly like a thesis that never broke.
        """
        evaluator = TriggerEvaluator(trigger(lane=["rerating"]))
        assert "t" in evaluator.evaluate("watch", metrics={})["triggers"]

    def test_a_string_lane_is_read_like_a_single_item_list(self):
        evaluator = TriggerEvaluator(trigger(lane="rerating"))
        assert evaluator.evaluate("watch", metrics={}, lane="core")["triggers"] == {}
        assert "t" in evaluator.evaluate("watch", metrics={}, lane="rerating")["triggers"]

    def test_lane_filtering_does_not_loosen_state_filtering(self):
        evaluator = TriggerEvaluator(trigger(lane=["rerating"], **{"from": ["watch"]}))
        assert evaluator.evaluate("scale", metrics={}, lane="rerating")["triggers"] == {}


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


class TestPhase3RegistryValidation:
    """The new vocabularies are closed at startup, for the same reason.

    A trigger naming a catalyst status, lane verdict, lane or state that does
    not exist would read indeterminate forever, and a fast-lane kill-switch
    that never fires is indistinguishable from a re-rating thesis that held.
    """

    def test_an_unknown_catalyst_status_is_rejected(self):
        errors = validate_triggers(trigger(conditions=[{"catalyst_status": "pending"}]))
        assert any("catalyst status" in e for e in errors)

    def test_the_recorded_catalyst_statuses_are_accepted(self):
        for status in ("active", "spent"):
            assert validate_triggers(
                trigger(conditions=[{"catalyst_status": status}])
            ) == []

    def test_an_unknown_lane_verdict_is_rejected(self):
        """`eligible` belongs to the 100x question; the lane has its own words."""
        errors = validate_triggers(trigger(conditions=[{"lane_verdict": "eligible"}]))
        assert any("lane verdict" in e for e in errors)

    def test_the_lane_verdict_vocabulary_is_accepted(self):
        for verdict in LANE_VERDICTS:
            assert validate_triggers(
                trigger(conditions=[{"lane_verdict": verdict}])
            ) == []

    def test_an_unknown_lane_is_rejected(self):
        errors = validate_triggers(trigger(lane=["momentum"]))
        assert any("lane" in e for e in errors)

    def test_the_declared_lanes_are_accepted(self):
        assert validate_triggers(trigger(lane=list(LANES))) == []
        assert validate_triggers(trigger(lane="rerating")) == []

    def test_since_state_entry_needs_a_known_comparator(self):
        errors = validate_triggers(trigger(conditions=[
            {"since_state_entry": "probe", "comparator": "eventually",
             "threshold": 545}
        ]))
        assert any("comparator" in e for e in errors)

    def test_since_state_entry_needs_a_threshold(self):
        errors = validate_triggers(trigger(conditions=[
            {"since_state_entry": "probe", "comparator": "gte"}
        ]))
        assert any("threshold" in e for e in errors)

    def test_since_state_entry_must_name_a_known_state(self):
        errors = validate_triggers(trigger(conditions=[
            {"since_state_entry": "nowhere", "comparator": "gte", "threshold": 545}
        ]))
        assert any("state" in e for e in errors)

    def test_a_well_formed_since_state_entry_is_accepted(self):
        assert validate_triggers(trigger(conditions=[
            {"since_state_entry": "probe", "comparator": "gte", "threshold": 545}
        ])) == []

    def test_the_new_kinds_join_the_exactly_one_rule(self):
        assert validate_triggers(trigger(conditions=[
            {"catalyst_status": "spent", "lane_verdict": "qualifies"}
        ]))


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
