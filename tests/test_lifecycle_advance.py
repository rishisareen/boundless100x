"""`advance`: re-score, evaluate, propose.

The rule under test everywhere here is R7 — transitions that move money are
proposed and wait for the owner; transitions before a position exists apply
themselves. Plus the precedence rule that stops a company being bought into on
the same quarter its thesis broke.

Phase 3 adds a second lane through the same loop. Three properties get their
own coverage here: the fast lane has a complete path of its own that the 100x
verdict cannot strand or drop; a core kill-switch outranks a fast-lane exit
even when both propose the same destination, because the *displayed reason* is
what the owner reads to decide; and every outcome carries a fail-closed
`routing_safety` reading whose eligibility question follows the lane.
"""

from datetime import date, timedelta

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.advance import (
    advance,
    advance_ticker,
    record_checkpoints,
    routing_safety,
)
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.service import AnalysisResult
from boundless100x.watchlist import WatchlistManager

# A fixed run date, so a time stop reads the same on any day the suite runs.
AS_OF = date(2026, 8, 7)


def metric(value=None, *, flags=None, series=None, error=None) -> MetricResult:
    return MetricResult(
        value=value, flags=flags or [], raw_series=series or [], error=error
    )


def healthy_metrics() -> dict:
    return {
        "roce_5yr_avg": metric(24.0, series=[23.0, 24.0, 25.0, 24.0, 24.0]),
        "roiic": metric(28.0),
        "growth_quality_grade": metric("high_quality"),
        "pe_vs_historical": metric(42.0),
        "trailing_peg": metric(1.4),
        "reverse_dcf_growth": metric(14.0),
        "promoter_pledge": metric(0.0),
    }


def fast_lane_metrics(**overrides) -> dict:
    """A healthy holding that also clears all six declared lane gates.

    Built on `healthy_metrics` rather than beside it: a fast-lane candidate is
    still held to every universal kill-switch, so a fixture that cleared the
    lane gates while tripping a fundamentals rule would be testing the tie-break
    rather than the path.
    """
    metrics = healthy_metrics()
    metrics.update({
        "pe_vs_historical": metric(35.0),          # inside the discount band
        "rerating_headroom": metric(40.0),
        "ttm_growth_vs_cagr": metric(6.0),
        "institutional_accumulation_streak": metric(
            3.0, flags=["institutional_accumulation_rising"]
        ),
        "daily_turnover_ratio": metric(0.05),
    })
    metrics.update(overrides)
    return metrics


class StubService:
    """Stands in for the pipeline; `advance` only needs analyze() and engine."""

    def __init__(self, metrics=None, composite=6.4, verdict="eligible", data=None,
                 flags=None):
        self._metrics = metrics if metrics is not None else healthy_metrics()
        self._composite = composite
        self._verdict = verdict
        self._data = data or {}
        self._flags = list(flags or [])
        self.engine = type("E", (), {"registry_hash": "abc123", "metrics": {}})()
        self.calls: list[str] = []
        # `advance` builds no report, so it must not pay for a momentum read.
        self.momentum_requested: bool | None = None

    def analyze(self, ticker, use_llm=True, include_momentum=True, **kw):
        self.calls.append(ticker)
        self.momentum_requested = include_momentum
        return AnalysisResult(
            ticker=ticker,
            data=self._data,
            metrics=self._metrics,
            scores={
                "composite": self._composite, "elements": {}, "flags": self._flags
            },
            eligibility={"verdict": self._verdict},
        )


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


@pytest.fixture
def evaluator():
    return TriggerEvaluator(load_triggers())


def run(service, wm, evaluator, ticker="ASTRAL", apply=False, as_of=None):
    return advance_ticker(service, wm, ticker, evaluator, apply=apply, as_of=as_of)


def fast_lane_entry(wm, ticker="ZENSAR", state=None, catalyst="active") -> str:
    """A tracked re-rating candidate, optionally moved into a later state."""
    wm.add(ticker, lane="rerating")
    if catalyst:
        wm.record_catalyst(ticker, "Demerger of the services arm", "2026-12-31")
        if catalyst == "spent":
            wm.mark_catalyst_spent(ticker)
    if state:
        wm.transition(ticker, state, "seed")
    return ticker


class TestAutoApplied:
    def test_a_qualifying_screen_entry_advances_without_confirmation(self, wm, evaluator):
        wm.add("ASTRAL")
        outcome = run(StubService(), wm, evaluator)

        assert outcome["proposal"]["to"] == "qualify"
        assert outcome["proposal"]["applied"] is True
        assert outcome["proposal"]["needs_confirmation"] is False
        assert wm.get("ASTRAL")["state"] == "qualify"

    def test_a_failing_screen_entry_is_dropped(self, wm, evaluator):
        wm.add("ASTRAL")
        outcome = run(StubService(verdict="not_eligible"), wm, evaluator)

        assert outcome["proposal"]["to"] == "dropped"
        assert wm.get("ASTRAL")["state"] == "dropped"

    def test_the_transition_records_evidence(self, wm, evaluator):
        wm.add("ASTRAL")
        run(StubService(), wm, evaluator)

        record = wm.get("ASTRAL")["state_history"][0]
        assert record["trigger_id"] == "qualification_passed"
        assert "composite" in record["evidence"]
        assert record["applied_by"] == "auto"


class TestMoneyMovingTransitions:
    def setup_entry(self, wm):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")

    def test_entering_probe_is_proposed_not_applied(self, wm, evaluator):
        """R7: deploying capital waits for the owner."""
        self.setup_entry(wm)
        outcome = run(StubService(), wm, evaluator)

        assert outcome["proposal"]["to"] == "probe"
        assert outcome["proposal"]["needs_confirmation"] is True
        assert outcome["proposal"]["applied"] is False
        assert wm.get("ASTRAL")["state"] == "watch"

    def test_apply_confirms_it_and_marks_the_owner(self, wm, evaluator):
        self.setup_entry(wm)
        outcome = run(StubService(), wm, evaluator, apply=True)

        assert outcome["proposal"]["applied"] is True
        assert wm.get("ASTRAL")["state"] == "probe"
        assert wm.get("ASTRAL")["state_history"][-1]["applied_by"] == "owner"

    def test_a_kill_switch_is_proposed_not_applied(self, wm, evaluator):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "scale", "seed")
        metrics = healthy_metrics()
        metrics["roce_5yr_avg"] = metric(19.0, series=[25.0, 24.0, 12.0, 11.0])

        outcome = run(StubService(metrics=metrics), wm, evaluator)

        assert outcome["proposal"]["to"] == "exit_review"
        assert outcome["proposal"]["needs_confirmation"] is True
        assert wm.get("ASTRAL")["state"] == "scale"


class TestPrecedence:
    def test_a_kill_switch_outranks_an_entry_proposal(self, wm, evaluator):
        """A buy zone reached the quarter the thesis broke is not an entry."""
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")
        metrics = healthy_metrics()
        metrics["roiic"] = metric(3.0)  # trips fundamentals_deteriorated

        outcome = run(StubService(metrics=metrics), wm, evaluator)

        assert outcome["proposal"]["to"] == "dropped"
        assert "valuation_buy_zone" in outcome["proposal"]["superseded"]

    def test_the_superseded_triggers_are_still_reported(self, wm, evaluator):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")
        metrics = healthy_metrics()
        metrics["roiic"] = metric(3.0)

        outcome = run(StubService(metrics=metrics), wm, evaluator)
        assert outcome["proposal"]["superseded"]


class TestNoChange:
    def test_nothing_fires_when_nothing_changed(self, wm, evaluator):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")
        metrics = healthy_metrics()
        metrics["trailing_peg"] = metric(9.0)  # priced out of the buy zone

        outcome = run(StubService(metrics=metrics), wm, evaluator)
        assert outcome["proposal"] is None
        assert wm.get("ASTRAL")["state"] == "watch"

    def test_indeterminate_triggers_are_reported_not_acted_on(self, wm, evaluator):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")
        metrics = healthy_metrics()
        del metrics["trailing_peg"]

        outcome = run(StubService(metrics=metrics), wm, evaluator)
        assert outcome["proposal"] is None
        assert "valuation_buy_zone" in outcome["indeterminate"]


class TestBookkeeping:
    def test_the_snapshot_is_recorded_every_run(self, wm, evaluator):
        wm.add("ASTRAL")
        run(StubService(), wm, evaluator)

        snapshot = wm.get("ASTRAL")["last_score_snapshot"]
        assert snapshot["composite"] == 6.4
        assert snapshot["config_hash"] == "abc123"

    def test_kill_switch_status_is_stored_per_trigger(self, wm, evaluator):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "scale", "seed")
        run(StubService(), wm, evaluator)

        status = wm.get("ASTRAL")["kill_switch_status"]
        assert status["capital_efficiency_break"] == "clear"
        assert status["checkpoints_failed"] == "unknown"  # none recorded yet

    def test_advance_scores_without_the_llm(self, wm, evaluator):
        """Re-scoring the whole watchlist must not cost an LLM call per name."""
        wm.add("ASTRAL")
        service = StubService()

        class Recorder(StubService):
            def analyze(self, ticker, use_llm=True, **kw):
                assert use_llm is False
                return super().analyze(ticker, use_llm=use_llm, **kw)

        run(Recorder(), wm, evaluator)


class TestCheckpointRecording:
    def test_pass2_monitorables_become_checkpoints(self, wm):
        wm.add("ASTRAL")
        result = AnalysisResult(ticker="ASTRAL", llm_analysis={"pass2": {
            "structured_monitorables": [{
                "metric_id": "quarterly_opm_pct", "comparator": "gte",
                "threshold": 20.0, "due_date": "2026-11-15",
            }]
        }})

        record_checkpoints(wm, "ASTRAL", result)
        assert len(wm.get("ASTRAL")["checkpoints"]) == 1

    def test_a_run_without_the_llm_records_nothing(self, wm):
        wm.add("ASTRAL")
        record_checkpoints(wm, "ASTRAL", AnalysisResult(ticker="ASTRAL"))
        assert wm.get("ASTRAL")["checkpoints"] == []

    def test_checkpoints_are_evaluated_on_the_next_advance(self, wm, evaluator):
        import pandas as pd

        wm.add("ASTRAL")
        wm.transition("ASTRAL", "scale", "seed")
        wm.set_checkpoints("ASTRAL", [{
            "metric_id": "quarterly_opm_pct", "comparator": "gte",
            "threshold": 30.0, "due_date": "2026-01-01",
        }])
        quarterly = pd.DataFrame({"quarter": ["Q1", "Q2"], "opm_pct": [22.0, 21.0]})

        outcome = run(StubService(data={"quarterly": quarterly}), wm, evaluator)
        assert outcome["checkpoints"]["missed"] == 1


class TestBatch:
    def test_one_failure_does_not_stop_the_others(self, wm, evaluator):
        """A stale fetch for one holding is no reason to skip checking another."""
        wm.add("GOOD")
        wm.add("BAD")

        class Flaky(StubService):
            def analyze(self, ticker, use_llm=True, **kw):
                if ticker == "BAD":
                    raise RuntimeError("fetch failed")
                return super().analyze(ticker, use_llm=use_llm, **kw)

        result = advance(Flaky(), wm, evaluator=evaluator)

        assert [o["ticker"] for o in result["outcomes"]] == ["GOOD"]
        assert result["errors"] == [("BAD", "fetch failed")]

    def test_an_empty_watchlist_advances_cleanly(self, wm, evaluator):
        result = advance(StubService(), wm, evaluator=evaluator)

        assert result["outcomes"] == []
        assert result["errors"] == []
        # An injected evaluator is used exactly as supplied, so the pace
        # modulator records that it did not evaluate rather than claiming a
        # modulation that never happened.
        assert result["pace"]["applied"] is False


class TestFastLanePath:
    """The re-rating lane's own way in, walked one state at a time."""

    def test_a_not_eligible_candidate_is_not_dropped(self, wm, evaluator):
        """The P0 the lane scoping fixes.

        Under the pre-fix registry `qualification_failed` fired on
        `verdict: not_eligible` in every lane, so a re-rating candidate was
        dropped before a single lane gate was consulted — the fast lane
        explicitly does not require 100x candidacy.
        """
        fast_lane_entry(wm)
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.0, verdict="not_eligible"
        )
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["proposal"]["trigger_id"] == "fast_lane_qualification_passed"
        assert outcome["proposal"]["to"] == "qualify"
        assert wm.get("ZENSAR")["state"] == "qualify"

    def test_it_reaches_watch_on_its_own_quality_floor(self, wm, evaluator):
        """`awaiting_entry_price` would have stranded it in `qualify`."""
        fast_lane_entry(wm, state="qualify")
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.0, verdict="not_eligible"
        )
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["proposal"]["trigger_id"] == "fast_lane_awaiting_entry"
        assert wm.get("ZENSAR")["state"] == "watch"

    def test_all_six_lane_gates_passing_proposes_probe(self, wm, evaluator):
        fast_lane_entry(wm, state="watch")
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.5, verdict="not_eligible"
        )
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["proposal"]["trigger_id"] == "fast_lane_buy_zone"
        assert outcome["proposal"]["to"] == "probe"
        # R7 is unchanged in the second lane: deploying capital waits.
        assert outcome["proposal"]["needs_confirmation"] is True
        assert wm.get("ZENSAR")["state"] == "watch"

    def test_a_failing_lane_gate_proposes_nothing(self, wm, evaluator):
        """No catalyst recorded is a known fact, so the gate fails outright."""
        fast_lane_entry(wm, state="watch", catalyst=None)
        service = StubService(metrics=fast_lane_metrics(), composite=6.5)
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["proposal"] is None
        assert wm.get("ZENSAR")["state"] == "watch"

    def test_a_candidate_below_the_quality_floor_is_dropped(self, wm, evaluator):
        """Without its own drop rule it would sit in `screen` forever."""
        fast_lane_entry(wm)
        service = StubService(metrics=fast_lane_metrics(), composite=4.2)
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["proposal"]["trigger_id"] == "fast_lane_qualification_failed"
        assert wm.get("ZENSAR")["state"] == "dropped"

    def test_a_core_entry_in_watch_never_takes_the_fast_lane_path(self, wm, evaluator):
        """Same readings, other lane: the core valuation rule is what fires."""
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")
        outcome = run(StubService(metrics=fast_lane_metrics()), wm, evaluator)

        assert outcome["proposal"]["trigger_id"] == "valuation_buy_zone"

    def test_the_whole_path_walks_screen_to_probe(self, wm, evaluator):
        """End to end, with lane-appropriate evidence at every step."""
        fast_lane_entry(wm)
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.5, verdict="not_eligible"
        )

        expected = [
            ("fast_lane_qualification_passed", "qualify"),
            ("fast_lane_awaiting_entry", "watch"),
            ("fast_lane_buy_zone", "probe"),
        ]
        for trigger_id, destination in expected:
            outcome = run(service, wm, evaluator, ticker="ZENSAR", apply=True)
            assert outcome["proposal"]["trigger_id"] == trigger_id
            assert outcome["proposal"]["to"] == destination
            assert wm.get("ZENSAR")["state"] == destination

        history = [r["trigger_id"] for r in wm.get("ZENSAR")["state_history"]]
        assert history == [t for t, _ in expected]


class TestFastLaneExits:
    def test_the_target_reached_switch_fires_on_the_stretched_flag(self, wm, evaluator):
        fast_lane_entry(wm, state="scale")
        metrics = fast_lane_metrics(
            rerating_headroom=metric(2.0, flags=["rerating_headroom_stretched"])
        )
        outcome = run(StubService(metrics=metrics), wm, evaluator, ticker="ZENSAR")

        assert outcome["proposal"]["trigger_id"] == "fast_lane_target_reached"
        assert outcome["proposal"]["to"] == "exit_review"
        assert outcome["proposal"]["needs_confirmation"] is True

    def test_the_time_stop_reads_the_run_date_not_the_wall_clock(self, wm, evaluator):
        ticker = fast_lane_entry(wm, state="probe")
        service = StubService(metrics=fast_lane_metrics())

        # The `probe` record was written now, so the stop is far from due at
        # today's date and long overdue two years on. Nothing but `as_of`
        # differs between the two runs.
        entered = date.fromisoformat(
            wm.get(ticker)["state_history"][-1]["at"][:10]
        )
        assert run(service, wm, evaluator, ticker=ticker,
                   as_of=entered + timedelta(days=100))["proposal"] is None

        outcome = run(service, wm, evaluator, ticker=ticker,
                      as_of=entered + timedelta(days=550))
        assert outcome["proposal"]["trigger_id"] == "fast_lane_time_stop"

    def test_a_spent_catalyst_proposes_an_exit_review(self, wm, evaluator):
        fast_lane_entry(wm, state="scale", catalyst="spent")
        outcome = run(
            StubService(metrics=fast_lane_metrics()), wm, evaluator, ticker="ZENSAR"
        )

        assert outcome["proposal"]["trigger_id"] == "fast_lane_catalyst_spent"

    def test_an_active_catalyst_does_not(self, wm, evaluator):
        fast_lane_entry(wm, state="scale")
        outcome = run(
            StubService(metrics=fast_lane_metrics()), wm, evaluator, ticker="ZENSAR"
        )
        assert outcome["proposal"] is None


class TestLaneTieBreak:
    """A shared destination is not a shared reason.

    Every `exit_review`-bound trigger ranks identically by destination, so the
    fast-lane exits and the core kill-switches would otherwise tie-break on
    YAML declaration order. The destination is safe either way; what would be
    wrong is the *displayed rationale* — the owner reading "target reached" on
    a position being exited because its incremental returns fell below the cost
    of capital.
    """

    def broken_and_stretched(self) -> dict:
        return fast_lane_metrics(
            roiic=metric(3.0),  # trips incremental_return_break
            rerating_headroom=metric(2.0, flags=["rerating_headroom_stretched"]),
        )

    def test_a_core_kill_switch_outranks_a_fast_lane_exit(self, wm, evaluator):
        fast_lane_entry(wm, state="scale")
        outcome = run(
            StubService(metrics=self.broken_and_stretched()), wm, evaluator,
            ticker="ZENSAR",
        )

        assert outcome["proposal"]["to"] == "exit_review"
        assert outcome["proposal"]["trigger_id"] == "incremental_return_break"
        assert "fast_lane_target_reached" in outcome["proposal"]["superseded"]

    def test_the_kill_switchs_evidence_is_what_the_proposal_shows(self, wm, evaluator):
        fast_lane_entry(wm, state="scale")
        outcome = run(
            StubService(metrics=self.broken_and_stretched()), wm, evaluator,
            ticker="ZENSAR",
        )

        assert "roiic" in outcome["proposal"]["evidence"]
        assert "rerating_headroom_stretched" not in outcome["proposal"]["evidence"]

    def test_the_core_lanes_own_drop_collision_is_untouched(self, wm, evaluator):
        """The stop condition, guarded: the core lane loses nothing.

        A core entry in `qualify` can trip `qualification_failed` and
        `fundamentals_deteriorated` at once, and both propose `dropped`. That
        collision predates the fast lane and resolves by declaration order.
        Extending universal-before-scoped to every destination would flip the
        recorded reason here — a core-lane change this phase must not make as a
        side effect of opening a second lane.
        """
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "qualify", "seed")
        metrics = healthy_metrics()
        metrics["roiic"] = metric(3.0)

        outcome = run(
            StubService(metrics=metrics, verdict="not_eligible"), wm, evaluator
        )

        assert outcome["proposal"]["to"] == "dropped"
        assert outcome["proposal"]["trigger_id"] == "qualification_failed"
        assert "fundamentals_deteriorated" in outcome["proposal"]["superseded"]

    def test_the_tie_break_is_by_lane_scope_not_declaration_order(self, wm):
        """Reordering the registry must not change which reason is shown.

        Declaring the fast-lane exits first is exactly what a stable sort on
        destination alone would surface, so this is the case that separates a
        real rule from an accident of file layout.
        """
        triggers = load_triggers()
        reordered = {
            trigger_id: triggers[trigger_id]
            for trigger_id in sorted(
                triggers, key=lambda t: not t.startswith("fast_lane_")
            )
        }
        assert list(reordered)[0].startswith("fast_lane_")

        fast_lane_entry(wm, state="scale")
        outcome = run(
            StubService(metrics=self.broken_and_stretched()), wm,
            TriggerEvaluator(reordered), ticker="ZENSAR",
        )

        assert outcome["proposal"]["trigger_id"] == "incremental_return_break"


class TestRoutingSafety:
    """Fail-closed, and the eligibility question follows the lane.

    Consumed by a later unit's router. `_eligibility_constraints` cannot answer
    for the fast lane: it speaks the 100x vocabulary, so handed a
    `not_qualified` it emits no constraint at all — a fail-*open* that would
    route capital into a candidate that just failed its own gates.
    """

    def test_a_core_candidate_failing_the_100x_gates_is_blocked(self, wm, evaluator):
        wm.add("ASTRAL")
        outcome = run(StubService(verdict="not_eligible"), wm, evaluator)

        assert outcome["routing_safety"]["clear"] is False
        assert outcome["routing_safety"]["reasons"]

    def test_a_clean_core_candidate_clears(self, wm, evaluator):
        wm.add("ASTRAL")
        outcome = run(StubService(), wm, evaluator)

        assert outcome["routing_safety"]["clear"] is True
        assert outcome["routing_safety"]["reasons"] == []

    def test_the_fast_lane_clears_on_its_own_gates_despite_not_eligible(
        self, wm, evaluator
    ):
        """The lane asymmetry: the fast lane must be able to receive capital.

        Applying the 100x verdict here would reimpose the exact gate set §9.2
        exists to replace, and a lane that can never be routed into cannot be
        funded from its own exits.
        """
        fast_lane_entry(wm, state="watch")
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.5, verdict="not_eligible"
        )
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["routing_safety"]["lane"] == "rerating"
        assert outcome["routing_safety"]["clear"] is True

    def test_a_fast_lane_candidate_failing_its_gates_is_blocked(self, wm, evaluator):
        fast_lane_entry(wm, state="watch", catalyst=None)
        service = StubService(metrics=fast_lane_metrics(), composite=6.5)
        outcome = run(service, wm, evaluator, ticker="ZENSAR")

        assert outcome["routing_safety"]["clear"] is False
        assert any(
            "catalyst" in reason.lower()
            for reason in outcome["routing_safety"]["reasons"]
        )

    def test_indeterminate_lane_gates_block(self, wm, evaluator):
        fast_lane_entry(wm, state="watch")
        metrics = fast_lane_metrics()
        del metrics["daily_turnover_ratio"]  # liquidity unreadable
        outcome = run(
            StubService(metrics=metrics, composite=6.5), wm, evaluator, ticker="ZENSAR"
        )

        assert outcome["routing_safety"]["clear"] is False

    @pytest.mark.parametrize("lane", ["core", "rerating"])
    def test_thin_evidence_blocks_both_lanes(self, wm, evaluator, lane):
        """A score resting on incomplete data is no basis for capital."""
        if lane == "core":
            wm.add("ASTRAL")
            ticker = "ASTRAL"
        else:
            ticker = fast_lane_entry(wm, state="watch")
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.5,
            flags=["low_data_coverage"],
        )
        outcome = run(service, wm, evaluator, ticker=ticker)

        assert outcome["routing_safety"]["clear"] is False

    def test_a_missing_lane_gate_result_blocks(self):
        safety = routing_safety("rerating", {"verdict": "eligible"}, {}, None)

        assert safety["clear"] is False
        assert safety["reasons"]

    def test_an_unrecognised_lane_verdict_blocks(self):
        """The vocabulary trap, stated directly: unknown never routes."""
        safety = routing_safety(
            "rerating", {"verdict": "eligible"}, {}, {"verdict": "probably_fine"}
        )

        assert safety["clear"] is False
        assert any("probably_fine" in reason for reason in safety["reasons"])

    def test_an_unrecognised_lane_blocks(self):
        """A lane nobody declared cannot be shown to have cleared anything."""
        safety = routing_safety("momentum", {"verdict": "eligible"}, {}, None)

        assert safety["clear"] is False
