"""The core-lane kill-switches, and the series trap they had to avoid.

A kill-switch that never fires is indistinguishable from a thesis that never
broke, so these tests check both directions for every switch: it fires on a
real breach, stays silent on a healthy company, and reads indeterminate when
its inputs are missing.

Phase 3 adds a second axis to that question — *which lane* a trigger applies
to — and the split is asserted here in both directions because leaking either
way is a distinct bug. The four core entry rules become core-only, so a
re-rating candidate is never judged by the 100x gate set its own lane exists to
replace; the six kill-switches and `fundamentals_deteriorated` stay universal,
because §6.2 gives the fast lane its own way *in* and no way out of a
fundamentals break.
"""

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.evaluator import (
    SERIES_SAFE_METRICS,
    TriggerEvaluator,
    load_triggers,
    validate_triggers,
)
from boundless100x.watchlist import LANES

KILL_SWITCHES = (
    "capital_efficiency_break",
    "growth_quality_degradation",
    "incremental_return_break",
    "valuation_saturation",
    "governance_event",
    "checkpoints_failed",
)

# Universal by design, not by omission: no `lane` key, so both lanes are held
# to every one of them. `fundamentals_deteriorated` joins the six because it is
# the same rule pointed at a candidate rather than a position.
UNIVERSAL_TRIGGERS = KILL_SWITCHES + ("fundamentals_deteriorated",)

# The four the core lane keeps to itself. Each gates on the **100x** verdict or
# the core valuation rule, and a lane with its own declared gate set must not
# also be gated by another lane's.
CORE_ONLY_TRIGGERS = (
    "qualification_passed",
    "qualification_failed",
    "awaiting_entry_price",
    "valuation_buy_zone",
)

# Everything that existed before lane filtering shipped. The core lane must see
# exactly this set and nothing else — losing one would break the lane that has
# live entries, gaining one would mean a fast-lane rule leaked across.
PRE_PHASE3_TRIGGERS = UNIVERSAL_TRIGGERS + CORE_ONLY_TRIGGERS

FAST_LANE_TRIGGERS = (
    "fast_lane_qualification_passed",
    "fast_lane_qualification_failed",
    "fast_lane_awaiting_entry",
    "fast_lane_buy_zone",
    "fast_lane_target_reached",
    "fast_lane_time_stop",
    "fast_lane_catalyst_spent",
)

STATES_WITH_TRIGGERS = ("screen", "qualify", "watch", "probe", "scale")


def origins(trigger_id: str, triggers: dict | None = None) -> list[str]:
    """The origin states a shipped trigger declares."""
    spec = (triggers if triggers is not None else load_triggers())[trigger_id]
    declared = spec.get("from") or ["any"]
    return [declared] if isinstance(declared, str) else list(declared)


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


class TestUniversalTriggersStayUniversal:
    """Half one of the split: no fundamentals rule may be lane-scoped.

    §6.2 is explicit that the fast lane never trades through a fundamentals
    break, so these seven carry no `lane` key at all. Tested separately from
    the core-only half because a leak in this direction — a kill-switch that
    quietly stopped applying to one lane — is the failure mode this whole file
    exists to catch, and it looks exactly like a thesis that never broke.
    """

    def test_no_universal_trigger_declares_a_lane(self):
        triggers = load_triggers()
        for trigger_id in UNIVERSAL_TRIGGERS:
            assert "lane" not in triggers[trigger_id], (
                f"{trigger_id} became lane-scoped — §6.2 requires the fast lane "
                f"to have no way out of a fundamentals break"
            )

    def test_every_universal_trigger_is_evaluated_in_both_lanes(self, evaluator):
        triggers = load_triggers()
        for trigger_id in UNIVERSAL_TRIGGERS:
            for origin in origins(trigger_id, triggers):
                for lane in LANES:
                    assert trigger_id in evaluator.applicable(origin, lane=lane)

    def test_a_fundamentals_break_still_fires_on_a_fast_lane_position(self, evaluator):
        """The rule stated as behaviour, not just as registry shape."""
        metrics = healthy()
        metrics["roiic"] = metric(6.0)
        result = evaluator.evaluate("scale", metrics=metrics, lane="rerating")

        assert "incremental_return_break" in result["fired"]
        assert result["triggers"]["incremental_return_break"]["to"] == "exit_review"

    def test_a_fast_lane_candidate_is_still_dropped_on_deterioration(self, evaluator):
        metrics = healthy()
        metrics["promoter_pledge"] = metric(20.0, flags=["promoter_pledge_red_flag"])
        result = evaluator.evaluate("watch", metrics=metrics, lane="rerating")

        assert "fundamentals_deteriorated" in result["fired"]


class TestCoreEntryRulesAreCoreOnly:
    """Half two: the four rules that must not reach the re-rating lane.

    `qualification_failed` drops on the **100x** verdict, so left universal it
    would drop a fast-lane candidate before its own gates were ever consulted;
    `awaiting_entry_price` gates qualify→watch on the same verdict, stranding
    it; `valuation_buy_zone` would open a position bypassing all six lane
    gates. Each is checked at every origin state it declares.
    """

    def test_they_are_absent_from_the_fast_lane(self, evaluator):
        triggers = load_triggers()
        for trigger_id in CORE_ONLY_TRIGGERS:
            for origin in origins(trigger_id, triggers):
                assert trigger_id not in evaluator.applicable(origin, lane="rerating")

    def test_they_still_apply_to_the_core_lane(self, evaluator):
        triggers = load_triggers()
        for trigger_id in CORE_ONLY_TRIGGERS:
            for origin in origins(trigger_id, triggers):
                assert trigger_id in evaluator.applicable(origin, lane="core")

    def test_each_declares_the_core_lane_explicitly(self):
        triggers = load_triggers()
        for trigger_id in CORE_ONLY_TRIGGERS:
            assert triggers[trigger_id]["lane"] == ["core"]


class TestFastLanePathIsFastLaneOnly:
    """The mirror image: the new path may not leak onto a core compounder."""

    def test_every_fast_lane_trigger_is_declared(self):
        triggers = load_triggers()
        assert set(FAST_LANE_TRIGGERS) <= set(triggers)

    def test_each_declares_the_rerating_lane(self):
        triggers = load_triggers()
        for trigger_id in FAST_LANE_TRIGGERS:
            assert triggers[trigger_id]["lane"] == ["rerating"]

    def test_none_are_applicable_to_a_core_entry(self, evaluator):
        triggers = load_triggers()
        for trigger_id in FAST_LANE_TRIGGERS:
            for origin in origins(trigger_id, triggers):
                assert trigger_id not in evaluator.applicable(origin, lane="core")

    def test_the_lane_has_a_complete_pre_position_path(self, evaluator):
        """screen → qualify → watch → probe, with a drop rule of its own.

        A lane missing any one of these is worse than a lane with none: a
        candidate would advance to a state nothing can move it out of, and sit
        there looking like a considered decision.
        """
        for origin in ("screen", "qualify", "watch"):
            applicable = evaluator.applicable(origin, lane="rerating")
            assert any(t.startswith("fast_lane_") for t in applicable), origin

    def test_no_fast_lane_trigger_sells_on_its_own(self):
        """Same rule the core switches follow: exit_review, never exited."""
        triggers = load_triggers()
        for trigger_id in ("fast_lane_target_reached", "fast_lane_time_stop",
                           "fast_lane_catalyst_spent"):
            assert triggers[trigger_id]["to"] == "exit_review"
            assert set(triggers[trigger_id]["from"]) == {"probe", "scale"}


class TestCoreLaneUnchanged:
    """The stop condition, asserted: the core lane loses nothing.

    Lane filtering is the kind of change that can only be proved by what it
    does *not* do. Two properties together are the proof: the core lane sees
    exactly the pre-Phase-3 trigger set at every state, and every one of those
    triggers reads the same with a core lane in hand as with no lane context at
    all.
    """

    @pytest.mark.parametrize("state", STATES_WITH_TRIGGERS)
    def test_the_core_lane_sees_exactly_the_pre_phase3_triggers(self, evaluator, state):
        triggers = load_triggers()
        expected = {
            trigger_id
            for trigger_id in PRE_PHASE3_TRIGGERS
            if state in origins(trigger_id, triggers)
        }
        assert set(evaluator.applicable(state, lane="core")) == expected

    @pytest.mark.parametrize("state", STATES_WITH_TRIGGERS)
    @pytest.mark.parametrize("breach", [None, "roiic", "roce", "pledge", "pe"])
    def test_every_pre_existing_trigger_reads_identically_for_a_core_entry(
        self, evaluator, state, breach
    ):
        metrics = healthy()
        if breach == "roiic":
            metrics["roiic"] = metric(3.0)
        elif breach == "roce":
            metrics["roce_5yr_avg"] = metric(19.0, series=[25.0, 24.0, 12.0, 11.0])
        elif breach == "pledge":
            metrics["promoter_pledge"] = metric(18.0, flags=["promoter_pledge_red_flag"])
        elif breach == "pe":
            metrics["pe_vs_historical"] = metric(98.0)
            metrics["reverse_dcf_growth"] = metric(45.0, flags=["reverse_dcf_overpriced"])

        context = {
            "metrics": metrics,
            "scores": {"composite": 6.4, "elements": {}},
            "eligibility": {"verdict": "eligible"},
            "checkpoint_results": {"met": 1, "missed": 2, "due": 3, "total": 3},
        }
        unscoped = evaluator.evaluate(state, **context)["triggers"]
        core = evaluator.evaluate(state, lane="core", **context)["triggers"]

        for trigger_id in PRE_PHASE3_TRIGGERS:
            if trigger_id not in unscoped:
                continue
            assert trigger_id in core, f"{trigger_id} vanished from the core lane"
            assert core[trigger_id]["fired"] == unscoped[trigger_id]["fired"]
            assert core[trigger_id]["reason"] == unscoped[trigger_id]["reason"]
