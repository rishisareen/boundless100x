"""`simulator.owner` — the simulated-owner policy block (KTD3, KTD6).

Per the plan's U3 and the architecture note in its own module docstring,
`owner.py` is pure policy layered *on top of* an already-produced
`advance.decide()` proposal or an already-ranked routing candidate list. It
must never import `TriggerEvaluator`, `LaneGateEvaluator`, or
`lifecycle.advance.decide` — this file tests it accordingly, by hand-building
proposal/gate-result fixtures rather than running the real evaluators.
"""

import datetime
import json

import pytest

from boundless100x.lifecycle.evaluator import load_triggers
from boundless100x.lifecycle.states import DROPPED, EXIT_REVIEW, PROBE, QUALIFY, SCALE, WATCH
from boundless100x.simulator import owner
from boundless100x.simulator.replay import FAST_LANE_ENTRY_GATES


# ── shared fixtures/helpers ──────────────────────────────────────────────


def make_proposal(to: str, trigger_id: str = "valuation_buy_zone", **overrides) -> dict:
    proposal = {
        "ticker": "ASTRAL",
        "from": "watch",
        "to": to,
        "trigger_id": trigger_id,
        "label": "Test trigger",
        "evidence": "test evidence — condition met",
    }
    proposal.update(overrides)
    return proposal


def gate_result_for(**passed_by_gate) -> dict:
    """A `LaneGateEvaluator.evaluate()`-shaped result, hand-built per U3's
    Approach ("build the gate_result fixture directly") rather than run
    through the real evaluator — `owner.py` must never import it.

    `passed_by_gate` maps a gate id to its `passed` value (`True`/`False`/
    `None`); any of `FAST_LANE_ENTRY_GATES` not named defaults to `True`, so
    a caller need only name the gate(s) it wants to differ from "all clear".
    """
    gates = {gate_id: {"passed": True} for gate_id in FAST_LANE_ENTRY_GATES}
    for gate_id, passed in passed_by_gate.items():
        gates[gate_id] = {"passed": passed}
    return {"gates": gates}


def _count_business_days_between(start: datetime.date, end: datetime.date) -> int:
    """Independent of `owner._advance_trading_days` — walks day by day and
    counts Mon-Fri strictly after `start` up to and including `end`, so the
    lag tests do not simply re-assert the implementation against itself.
    """
    assert end >= start
    count = 0
    day = start
    while day < end:
        day += datetime.timedelta(days=1)
        if day.weekday() < 5:  # Monday=0 .. Friday=4
            count += 1
    return count


# ── _advance_trading_days: the trading-day helper itself ────────────────


class TestAdvanceTradingDays:
    def test_a_friday_start_skips_the_intervening_weekend(self):
        # 2026-08-07 is a Friday.
        start = datetime.date(2026, 8, 7)
        result = owner._advance_trading_days(start, 5)
        assert result == datetime.date(2026, 8, 14)  # the following Friday
        assert _count_business_days_between(start, result) == 5

    def test_a_weekend_start_is_normalized_forward_first(self):
        # 2026-08-08 is a Saturday, 2026-08-09 a Sunday.
        start = datetime.date(2026, 8, 8)
        result = owner._advance_trading_days(start, 5)
        # Normalizes to Monday 2026-08-10, then 5 trading days beyond that.
        assert result == datetime.date(2026, 8, 17)
        assert result.weekday() == 0  # Monday
        assert result > start

    def test_zero_lag_returns_the_normalized_start_unchanged(self):
        start = datetime.date(2026, 8, 7)  # already a Friday
        assert owner._advance_trading_days(start, 0) == start

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            owner._advance_trading_days(datetime.date(2026, 8, 7), -1)


# ── config_from ──────────────────────────────────────────────────────────


class TestConfigFrom:
    def test_defaults_with_no_config(self):
        settings = owner.config_from(None)
        assert settings["starting_pool"] == 100
        assert settings["confirmation_lag_days"] == {"entry": 5, "exit": 2, "route": 5}
        assert settings["catalyst_window_months"] == 6
        assert settings["cap_posture"] == "enforced"
        assert settings["reduce_fraction"] is None

    def test_accepts_the_simulator_block_alone(self):
        settings = owner.config_from({"starting_pool": 250, "cap_posture": "advisory"})
        assert settings["starting_pool"] == 250
        assert settings["cap_posture"] == "advisory"

    def test_accepts_the_whole_pipeline_config(self):
        whole_config = {
            "macro": {"inflation": 5.0},
            "simulator": {"starting_pool": 500},
        }
        settings = owner.config_from(whole_config)
        assert settings["starting_pool"] == 500

    def test_partial_lag_override_keeps_the_other_defaults(self):
        settings = owner.config_from({"confirmation_lag_days": {"entry": 3}})
        assert settings["confirmation_lag_days"] == {"entry": 3, "exit": 2, "route": 5}

    def test_reduce_fraction_stays_none_unless_configured(self):
        assert owner.config_from({}).get("reduce_fraction") is None
        assert owner.config_from({"reduce_fraction": 0.5})["reduce_fraction"] == 0.5


# ── override_caps_for (decision 5) ───────────────────────────────────────


class TestOverrideCapsFor:
    def test_enforced_withholds(self):
        assert owner.override_caps_for("enforced") is False

    def test_advisory_proceeds(self):
        assert owner.override_caps_for("advisory") is True

    def test_override_proceeds(self):
        assert owner.override_caps_for("override") is True

    def test_all_three_postures_are_reachable(self):
        assert owner.CAP_POSTURES == ("enforced", "advisory", "override")

    def test_garbage_input_fails_closed_to_enforced(self, caplog):
        with caplog.at_level("WARNING"):
            result = owner.override_caps_for("yolo")
        assert result is False
        assert any("yolo" in record.message for record in caplog.records)

    def test_none_fails_closed(self):
        assert owner.override_caps_for(None) is False


# ── severity_for / sell_fraction_for (§14.3, R6, decision 4) ────────────


def _exit_review_trigger_ids() -> set[str]:
    """Derived mechanically off the shipped registry, not hand-listed —
    the same "mechanical rather than remembered" discipline CLAUDE.md
    documents for `FLAG_ELEMENT_MAP`'s own test: a trigger added to
    `triggers.yaml` that reaches `exit_review` must be caught here without
    anyone remembering to update a parallel list.
    """
    return {tid for tid, spec in load_triggers().items() if spec.get("to") == EXIT_REVIEW}


class TestSeverityMapping:
    def test_severity_map_covers_exactly_the_exit_review_triggers(self):
        assert set(owner.SEVERITY_MAP) == _exit_review_trigger_ids()

    def test_governance_event_is_full_exit(self):
        assert owner.severity_for("governance_event") == "full_exit"
        assert owner.sell_fraction_for("governance_event") == 1.0

    def test_valuation_saturation_is_reduce_but_sells_in_full_with_no_fraction_configured(self):
        assert owner.severity_for("valuation_saturation") == "reduce"
        assert owner.sell_fraction_for("valuation_saturation") == 1.0

    def test_valuation_saturation_honours_a_configured_fraction(self):
        config = {"reduce_fraction": 0.4}
        assert owner.severity_for("valuation_saturation", config) == "reduce"
        assert owner.sell_fraction_for("valuation_saturation", config) == 0.4

    def test_reduce_event_is_distinguishable_from_a_full_exit_even_at_the_same_fraction(self):
        # With no fraction configured both resolve to sell_fraction 1.0, but
        # the severity tag is what "counted separately" (R6) needs
        # downstream — the two must never collapse into one label.
        assert owner.severity_for("governance_event") != owner.severity_for(
            "valuation_saturation"
        )
        assert (
            owner.sell_fraction_for("governance_event")
            == owner.sell_fraction_for("valuation_saturation")
            == 1.0
        )

    @pytest.mark.parametrize(
        "trigger_id",
        [
            "capital_efficiency_break",
            "growth_quality_degradation",
            "incremental_return_break",
            "checkpoints_failed",
            "fast_lane_target_reached",
            "fast_lane_time_stop",
            "fast_lane_catalyst_spent",
        ],
    )
    def test_every_other_exit_review_trigger_is_review(self, trigger_id):
        assert owner.severity_for(trigger_id) == "review"
        assert owner.sell_fraction_for(trigger_id) == 1.0

    def test_an_unrecognised_trigger_id_defaults_to_review_rather_than_raising(self):
        assert owner.severity_for("some_future_kill_switch_nobody_wrote_yet") == "review"
        assert owner.sell_fraction_for("some_future_kill_switch_nobody_wrote_yet") == 1.0

    def test_severity_overrides_take_precedence_over_the_shipped_map(self):
        config = {"severity_overrides": {"governance_event": "review"}}
        assert owner.severity_for("governance_event", config) == "review"
        # Unaffected trigger ids are untouched by the override.
        assert owner.severity_for("valuation_saturation", config) == "reduce"


# ── decide() ──────────────────────────────────────────────────────────


class TestDecide:
    def test_an_entry_proposal_is_never_confirmed_before_its_lag_elapses(self):
        as_of = datetime.date(2026, 8, 7)  # Friday
        proposal = make_proposal(PROBE, trigger_id="valuation_buy_zone")
        result = owner.decide(proposal, as_of, config=None)

        assert result["action"] == "confirm"
        confirm_at = datetime.date.fromisoformat(result["confirm_at"])
        assert confirm_at > as_of
        assert _count_business_days_between(as_of, confirm_at) == 5  # entry lag
        assert result["severity"] is None
        assert result["sell_fraction"] is None

    def test_entry_lag_spans_a_weekend_correctly(self):
        as_of = datetime.date(2026, 8, 8)  # Saturday
        proposal = make_proposal(SCALE, trigger_id="valuation_buy_zone")
        result = owner.decide(proposal, as_of, config=None)
        confirm_at = datetime.date.fromisoformat(result["confirm_at"])
        assert confirm_at == datetime.date(2026, 8, 17)
        assert confirm_at.weekday() < 5

    def test_an_exit_review_proposal_uses_the_exit_lag_and_carries_severity(self):
        as_of = datetime.date(2026, 8, 7)  # Friday
        proposal = make_proposal(EXIT_REVIEW, trigger_id="governance_event")
        result = owner.decide(proposal, as_of, config=None)

        assert result["action"] == "confirm"
        confirm_at = datetime.date.fromisoformat(result["confirm_at"])
        assert _count_business_days_between(as_of, confirm_at) == 2  # exit lag
        assert result["severity"] == "full_exit"
        assert result["sell_fraction"] == 1.0

    def test_a_valuation_saturation_exit_carries_the_reduce_tag(self):
        as_of = datetime.date(2026, 8, 7)
        proposal = make_proposal(EXIT_REVIEW, trigger_id="valuation_saturation")
        result = owner.decide(proposal, as_of, config={"reduce_fraction": 0.3})
        assert result["severity"] == "reduce"
        assert result["sell_fraction"] == 0.3

    def test_confirmation_lags_are_config_overridable(self):
        as_of = datetime.date(2026, 8, 7)
        proposal = make_proposal(PROBE)
        result = owner.decide(
            proposal, as_of, config={"confirmation_lag_days": {"entry": 1}}
        )
        confirm_at = datetime.date.fromisoformat(result["confirm_at"])
        assert _count_business_days_between(as_of, confirm_at) == 1

    def test_a_pre_position_destination_is_skipped_not_confirmed(self):
        as_of = datetime.date(2026, 8, 7)
        for to_state in (QUALIFY, WATCH, DROPPED):
            proposal = make_proposal(to_state, trigger_id="qualification_passed")
            result = owner.decide(proposal, as_of, config=None)
            assert result["action"] == "skip"
            assert result["confirm_at"] is None

    def test_an_empty_proposal_raises(self):
        with pytest.raises(ValueError):
            owner.decide({}, datetime.date(2026, 8, 7), config=None)
        with pytest.raises(ValueError):
            owner.decide(None, datetime.date(2026, 8, 7), config=None)

    def test_an_unparseable_as_of_raises(self):
        with pytest.raises(ValueError):
            owner.decide(make_proposal(PROBE), "not-a-date", config=None)

    # ── the U4 seam ──

    def test_portfolio_state_none_accepts_by_default(self):
        as_of = datetime.date(2026, 8, 7)
        result = owner.decide(
            make_proposal(PROBE), as_of, config=None, portfolio_state=None
        )
        assert result["action"] == "confirm"

    def test_portfolio_state_can_price_true_accepts(self):
        as_of = datetime.date(2026, 8, 7)
        result = owner.decide(
            make_proposal(PROBE), as_of, config=None,
            portfolio_state={"can_price": True},
        )
        assert result["action"] == "confirm"

    def test_portfolio_state_can_price_false_rejects(self):
        as_of = datetime.date(2026, 8, 7)
        result = owner.decide(
            make_proposal(PROBE), as_of, config=None,
            portfolio_state={"can_price": False},
        )
        assert result["action"] == "skip"
        assert "cannot be priced" in result["reason"]

    def test_decide_output_round_trips_through_json(self):
        as_of = datetime.date(2026, 8, 7)
        for to_state, trigger_id in (
            (PROBE, "valuation_buy_zone"),
            (EXIT_REVIEW, "governance_event"),
        ):
            result = owner.decide(make_proposal(to_state, trigger_id), as_of, config=None)
            round_tripped = json.loads(json.dumps(result))
            assert round_tripped == result


# ── catalyst_for (KTD6) ──────────────────────────────────────────────────


class TestCatalystFor:
    def test_fabricates_when_all_five_gates_clear(self):
        as_of = datetime.date(2026, 8, 7)
        result = owner.catalyst_for(
            "ASTRAL", gate_result_for(), config=None, as_of=as_of
        )
        assert result is not None
        description, expected_by = result
        assert "[simulated]" in description
        assert "ASTRAL" in description
        assert expected_by == "2027-02-07"  # +6 months, the settled window

    def test_returns_none_when_one_gate_is_false(self):
        result = owner.catalyst_for(
            "ASTRAL",
            gate_result_for(liquidity_floor=False),
            config=None,
            as_of=datetime.date(2026, 8, 7),
        )
        assert result is None

    def test_returns_none_when_one_gate_is_indeterminate(self):
        result = owner.catalyst_for(
            "ASTRAL",
            gate_result_for(institutional_accumulation=None),
            config=None,
            as_of=datetime.date(2026, 8, 7),
        )
        assert result is None

    def test_returns_none_on_an_empty_or_missing_gate_result(self):
        assert owner.catalyst_for(
            "ASTRAL", {}, config=None, as_of=datetime.date(2026, 8, 7)
        ) is None
        assert owner.catalyst_for(
            "ASTRAL", None, config=None, as_of=datetime.date(2026, 8, 7)
        ) is None

    def test_window_is_config_overridable(self):
        result = owner.catalyst_for(
            "ASTRAL",
            gate_result_for(),
            config={"catalyst_window_months": 12},
            as_of=datetime.date(2026, 8, 7),
        )
        _, expected_by = result
        assert expected_by == "2027-08-07"

    def test_an_unparseable_as_of_raises(self):
        with pytest.raises(ValueError):
            owner.catalyst_for("ASTRAL", gate_result_for(), config=None, as_of="nope")

    def test_output_round_trips_through_json(self):
        result = owner.catalyst_for(
            "ASTRAL", gate_result_for(), config=None, as_of=datetime.date(2026, 8, 7)
        )
        round_tripped = json.loads(json.dumps(result))
        assert list(round_tripped) == list(result)


# ── route() (KTD3) ────────────────────────────────────────────────────


class TestRoute:
    def test_holds_when_no_candidates(self):
        result = owner.route({"exit_id": "x1"}, [], datetime.date(2026, 8, 7), config=None)
        assert result["action"] == "hold"
        assert result["confirm_at"] is None
        assert result["candidate"] is None

    def test_schedules_the_top_ranked_candidate_at_as_of_plus_route_lag(self):
        as_of = datetime.date(2026, 8, 7)  # Friday
        candidates = [
            {"ticker": "BAJFINANCE", "state": "watch"},
            {"ticker": "ASTRAL", "state": "qualify"},
        ]
        result = owner.route({"exit_id": "x1"}, candidates, as_of, config=None)

        assert result["action"] == "confirm"
        assert result["candidate"]["ticker"] == "BAJFINANCE"  # top-ranked, not re-ranked
        confirm_at = datetime.date.fromisoformat(result["confirm_at"])
        assert _count_business_days_between(as_of, confirm_at) == 5  # route lag

    def test_does_not_re_rank(self):
        # Even a candidate that "looks" better by some field ordering must
        # not be promoted — route() trusts the caller's own ordering.
        as_of = datetime.date(2026, 8, 7)
        candidates = [
            {"ticker": "ZZZLOW", "composite": 1.0},
            {"ticker": "AAATOP", "composite": 9.9},
        ]
        result = owner.route({"exit_id": "x1"}, candidates, as_of, config=None)
        assert result["candidate"]["ticker"] == "ZZZLOW"

    def test_route_lag_is_config_overridable(self):
        as_of = datetime.date(2026, 8, 7)
        candidates = [{"ticker": "ASTRAL"}]
        result = owner.route(
            {"exit_id": "x1"}, candidates, as_of,
            config={"confirmation_lag_days": {"route": 1}},
        )
        confirm_at = datetime.date.fromisoformat(result["confirm_at"])
        assert _count_business_days_between(as_of, confirm_at) == 1

    def test_an_unparseable_as_of_raises(self):
        with pytest.raises(ValueError):
            owner.route({"exit_id": "x1"}, [{"ticker": "A"}], "nope", config=None)

    def test_output_round_trips_through_json(self):
        as_of = datetime.date(2026, 8, 7)
        for candidates in ([], [{"ticker": "ASTRAL"}]):
            result = owner.route({"exit_id": "x1"}, candidates, as_of, config=None)
            round_tripped = json.loads(json.dumps(result))
            assert round_tripped == result
