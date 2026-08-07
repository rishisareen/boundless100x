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
from functools import lru_cache

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import lane_gates as lane_gates_module
from boundless100x.lifecycle.advance import (
    advance,
    advance_ticker,
    decide,
    record_checkpoints,
    routing_safety,
)
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.service import AnalysisResult
from boundless100x.watchlist import WatchlistManager

# A fixed run date, so a time stop reads the same on any day the suite runs.
AS_OF = date(2026, 8, 7)


@lru_cache(maxsize=1)
def engine_metric_ids() -> frozenset:
    """The real registry's metric ids, built once for the whole session.

    `advance()` validates both the trigger registry and the lane-gate registry
    against `service.engine.metrics` at startup, and a stub carrying an empty
    mapping would make every declared metric id read as unknown — so the stub
    has to answer that question the way production does. Cached because the
    answer is a property of the shipped registry, and every StubService in the
    suite would otherwise re-parse it.
    """
    from boundless100x.compute_engine.engine import ComputeEngine

    return frozenset(ComputeEngine().metrics)


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


class StubPriceFetcher:
    """`suite.price_volume` — the one source a confirmed exit consults.

    Records its calls, because "how many sources did this touch?" is the whole
    question on the exit path: `confirm_exit` needs one column and used to run
    the entire pipeline to read it.
    """

    def __init__(self, price):
        self._price = price
        self.calls: list[tuple] = []

    def fetch(self, ticker, years=10, output_dir=None):
        self.calls.append((ticker, years, output_dir))
        return pd.DataFrame() if self._price is None else self._price


class StubSuite:
    """The fetcher suite, in the shape `DataFetcherSuite` exposes it."""

    def __init__(self, price):
        self.price_volume = StubPriceFetcher(price)
        self.price_years = 10
        self.raw_data_dir = "/nonexistent"


class StubService:
    """Stands in for the pipeline; `advance` only needs analyze() and engine."""

    def __init__(self, metrics=None, composite=6.4, verdict="eligible", data=None,
                 flags=None, config=None):
        self._metrics = metrics if metrics is not None else healthy_metrics()
        self._composite = composite
        self._verdict = verdict
        self._data = data or {}
        self._flags = list(flags or [])
        # The owner-policy blocks a run reads: `portfolio:` for the caps,
        # `friction:` and `deployment_pace:` for their own. Empty by default, so
        # every existing caller still gets the shipped defaults.
        self.config = config or {}
        self.engine = type(
            "E", (),
            {"registry_hash": "abc123", "metrics": engine_metric_ids()},
        )()
        # The same series `data["price"]` carries, reachable the way a caller
        # that wants only a price series reaches it — without running analyze().
        self.suite = StubSuite(self._data.get("price"))
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

    def test_a_ticker_whose_advance_raised_is_not_marked_freshly_scored(
        self, wm, evaluator
    ):
        """`get_stale(90)` reads the snapshot's timestamp to decide what a
        `--quarterly` run looks at, and the question it is asking is "was this
        company successfully evaluated recently?" — not "did an evaluation of
        it begin?".

        Written the moment `analyze()` returned, a ticker that raised anywhere
        downstream was reported in the run's errors *and* stamped fresh, so
        every quarterly run for the next three months skipped the one company
        whose last evaluation had failed. A thesis that broke on exactly that
        day went unseen until the quarter was up.
        """
        wm.add("ASTRAL")

        def evaluation_that_raises(*args, **kwargs):
            raise RuntimeError("the trigger registry blew up mid-run")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(evaluator, "evaluate", evaluation_that_raises)
            result = advance(StubService(), wm, evaluator=evaluator)

        assert [ticker for ticker, _ in result["errors"]] == ["ASTRAL"]
        assert wm.get("ASTRAL")["last_score_snapshot"] is None
        assert wm.get_stale(90) == ["ASTRAL"]

    def test_a_ticker_that_advanced_cleanly_is_marked_scored(self, wm, evaluator):
        """The other half — the reorder must not stop a successful run being
        recorded, or every run would re-score everything forever."""
        wm.add("ASTRAL")

        advance(StubService(), wm, evaluator=evaluator)

        assert wm.get("ASTRAL")["last_score_snapshot"] is not None
        assert wm.get_stale(90) == []


class TestTheLaneView:
    """The assembled view every surface renders, carried out of the loop.

    Built from the same `lane_view.build_lane_context` a report calls, so the
    terminal and the report cannot describe one position two ways — and handed
    the lane-gate result the run has already paid for rather than evaluating a
    second time.
    """

    def test_every_outcome_carries_one(self, wm, evaluator):
        wm.add("ASTRAL")
        outcome = run(StubService(), wm, evaluator)

        assert outcome["lane_context"]["lane"] == "core"

    def test_it_describes_the_state_the_run_left_the_company_in(self, wm, evaluator):
        """Assembled after the transition, or it would report yesterday's state."""
        wm.add("ASTRAL")
        outcome = run(StubService(), wm, evaluator)

        assert outcome["state"] == "screen"
        assert outcome["proposal"]["to"] == "qualify"
        assert outcome["lane_context"]["state"] == "qualify"

    def test_the_gate_result_is_the_one_already_computed(self, wm, evaluator):
        ticker = fast_lane_entry(wm, state="watch")
        outcome = run(StubService(fast_lane_metrics()), wm, evaluator, ticker=ticker)

        assert outcome["lane_context"]["lane_gates"] is outcome["lane_gates"]

    def test_a_core_outcome_carries_no_lane_gates(self, wm, evaluator):
        wm.add("ASTRAL")
        outcome = run(StubService(), wm, evaluator)

        assert outcome["lane_context"]["lane_gates"] is None


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

    def monitorable(self, due: str) -> AnalysisResult:
        return AnalysisResult(ticker="ASTRAL", llm_analysis={"pass2": {
            "structured_monitorables": [{
                "metric_id": "quarterly_opm_pct", "comparator": "gte",
                "threshold": 20.0, "due_date": due,
            }]
        }})

    def test_the_past_dating_check_follows_the_run_clock(self, wm):
        """The one seam where "the same clock the rest of the run reads" did
        not reach the recorder.

        `record_from_pass2` refuses a checkpoint already due when recorded —
        one due on the day it is written was never monitored. Left to default,
        that refusal read `date.today()` while the evaluator, the friction
        reading and the time stops all read the supplied date, so a backdated
        replay disagreed with itself about what "today" meant.

        Asserted in the direction the wall clock gets *wrong*: this due date is
        comfortably in the future today, so a recorder still reading the wall
        clock would keep it.
        """
        wm.add("ASTRAL")

        record_checkpoints(wm, "ASTRAL", self.monitorable("2026-11-15"),
                           as_of=date(2026, 12, 1))

        assert wm.get("ASTRAL")["checkpoints"] == []

    def test_a_backdated_replay_keeps_what_was_future_at_the_time(self, wm):
        """The other direction, and the case a replay actually meets: a date
        long past today was the future when the thesis was written, and the
        run replaying that day must record it as the monitorable it was."""
        wm.add("ASTRAL")

        recorded = record_checkpoints(wm, "ASTRAL", self.monitorable("2025-06-01"),
                                      as_of=date(2025, 1, 1))

        assert len(recorded["checkpoints"]) == 1
        assert recorded["demoted"] == []

    def test_no_clock_supplied_still_reads_today(self, wm):
        """The default is unchanged, so every existing caller behaves exactly
        as it did — the fix completes a seam rather than moving one."""
        wm.add("ASTRAL")

        record_checkpoints(wm, "ASTRAL", self.monitorable("2020-01-01"))

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


class TestTheLaneGateRegistryIsValidatedAtStartup:
    """A lane no company can enter looks exactly like a lane with no candidates.

    That sentence is `lane_gates.py`'s own statement of why its startup check
    exists, and on the production path the check never ran: `advance_ticker`
    built a `LaneGateEvaluator()` per re-rating ticker with no
    `known_metric_ids`, and `validate_lane_gates` guards the unknown-metric-id
    check behind `is not None`. Renaming a metric in `size.yaml` would have sent
    the fast lane permanently indeterminate with a green suite and no error.

    `TriggerEvaluator` is handed `set(service.engine.metrics)` at both of its
    production call sites; the sibling evaluator must be too.
    """

    def registry(self, tmp_path, metric_id: str):
        path = tmp_path / "lane_gates.yaml"
        path.write_text(
            "lane_gates:\n"
            "  institutional_accumulation:\n"
            "    label: Institutional accumulation\n"
            "    conditions:\n"
            f"      - metric: {metric_id}\n"
            "        comparator: gte\n"
            "        threshold: 2\n"
        )
        return path

    def test_a_gate_naming_an_unknown_metric_raises(
        self, wm, evaluator, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            lane_gates_module, "DEFAULT_LANE_GATES_PATH",
            self.registry(tmp_path, "institutional_accumulation_streek"),
        )
        fast_lane_entry(wm, state="watch")

        with pytest.raises(ValueError, match="institutional_accumulation_streek"):
            advance(StubService(fast_lane_metrics()), wm, evaluator=evaluator)

    def test_it_raises_before_a_single_company_is_advanced(
        self, wm, evaluator, tmp_path, monkeypatch
    ):
        """Startup, not lazily on the first re-rating ticker.

        An empty watchlist is the sharpest form of the claim: with nothing to
        advance there is no per-ticker path to reach the registry from at all,
        so a run that still raises can only have validated it up front.
        """
        monkeypatch.setattr(
            lane_gates_module, "DEFAULT_LANE_GATES_PATH",
            self.registry(tmp_path, "no_such_metric"),
        )

        with pytest.raises(ValueError, match="no_such_metric"):
            advance(StubService(), wm, evaluator=evaluator)

    def test_the_shipped_registry_advances_cleanly(self, wm, evaluator):
        """The other half: validation that fires on everything is not validation."""
        fast_lane_entry(wm, state="watch")
        out = advance(
            StubService(fast_lane_metrics(), composite=6.5), wm, evaluator=evaluator
        )

        assert out["errors"] == []
        assert out["outcomes"][0]["lane_gates"]["verdict"] == "qualifies"

    def test_the_registry_is_read_once_a_run_not_once_a_ticker(
        self, wm, evaluator, monkeypatch
    ):
        """One run, one reading — which is also what keeps an edited
        `lane_gates.yaml` taking effect at a run boundary rather than partway
        through a loop."""
        reads = []
        real = lane_gates_module.load_lane_gates
        monkeypatch.setattr(
            lane_gates_module, "load_lane_gates",
            lambda path=None: (reads.append(path), real(path))[1],
        )
        for ticker in ("ZENSAR", "COFORGE"):
            fast_lane_entry(wm, ticker=ticker, state="watch")

        advance(StubService(fast_lane_metrics(), composite=6.5), wm,
                evaluator=evaluator)

        assert len(reads) == 1


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

    def test_the_fail_closed_reason_points_at_what_has_to_change(self):
        """The branch catches two different situations and must not assert one
        of them.

        A lane can reach it by not being in `LANES` at all, or by being a
        perfectly declared lane that this function has no safety question for
        yet — a third lane added to `watchlist.LANES` lands here on the day it
        is declared. "Not a declared lane" is false in that case, and it sends
        whoever reads it to the wrong file.
        """
        reason = routing_safety("momentum", {"verdict": "eligible"}, {}, None)["reasons"][0]

        assert "lifecycle/advance.py" in reason
        assert "not a declared lane" not in reason


class TestConcentrationGatesTheMoneyMovingPath:
    """A cap that is checked before the transition, not counted after it.

    The reading existed before this and had exactly two consumers, both
    advisory: the routing proposal and a display line. Nothing in the
    transition path read it, and it was computed *after* the ticker loop — so
    the first time an owner learned a lane was over its cap, the transitions
    that broke it were already in an append-only history. A cap could be
    reported as breached and never prevented from being breached.

    Two properties carry the fix. It is asked **per candidate, live**, because
    an applying run changes the occupancy it is checking. And it is asked only
    when a transition would **add a name**, because that is what the caps
    count — a `probe → scale` moves the same company deeper into a position it
    already holds.

    Every figure is a count of positioned names, never a share of capital.
    """

    # A cap of one makes the boundary reachable without eight stub companies,
    # and the sector cap is left at the shipped default: this class is about
    # the lane axis, and `test_portfolio_concentration.py` owns the counting.
    CAPPED = {"portfolio": {"max_positioned_per_lane": {"core": 1, "rerating": 5}}}

    def full_lane(self, wm):
        """One positioned core name against a cap of one — the lane is full."""
        wm.add("HELD")
        wm.transition("HELD", "probe", "seed", applied_by="owner")

    def waiting(self, wm, ticker="ASTRAL"):
        """A candidate whose buy-zone trigger will fire this run."""
        wm.add(ticker)
        wm.transition(ticker, "watch", "seed")
        return ticker

    def advanced(self, wm, evaluator, **kwargs):
        return advance(
            StubService(config=self.CAPPED), wm, evaluator=evaluator, **kwargs
        )

    def proposal_for(self, result, ticker):
        return next(o["proposal"] for o in result["outcomes"] if o["ticker"] == ticker)

    def test_a_full_lane_withholds_the_transition_even_under_apply(
        self, wm, evaluator
    ):
        """The behaviour change, stated at its narrowest: `--apply` is no
        longer sufficient on its own to take a position."""
        self.full_lane(wm)
        self.waiting(wm)

        result = self.advanced(wm, evaluator, apply=True)
        proposal = self.proposal_for(result, "ASTRAL")

        assert proposal["to"] == "probe"
        assert proposal["concentration_withheld"] is True
        assert proposal["applied"] is False
        assert wm.get("ASTRAL")["state"] == "watch"

    def test_the_refusal_names_the_cap_and_the_basis(self, wm, evaluator):
        """A guardrail that refuses without saying which limit, at what count,
        and in what unit is a system the owner works around rather than with."""
        self.full_lane(wm)
        self.waiting(wm)

        result = self.advanced(wm, evaluator, apply=True)
        reasons = self.proposal_for(result, "ASTRAL")["concentration_reasons"]

        assert len(reasons) == 1
        assert "core lane already holds 1 of a maximum 1" in reasons[0]
        assert "counts of names, not a share of capital" in reasons[0]

    def test_a_lane_with_room_still_applies(self, wm, evaluator):
        """The guardrail must not be a blanket refusal — headroom is the
        ordinary case and has to stay ordinary."""
        self.waiting(wm)

        result = self.advanced(wm, evaluator, apply=True)
        proposal = self.proposal_for(result, "ASTRAL")

        assert proposal["concentration_withheld"] is False
        assert proposal["applied"] is True
        assert wm.get("ASTRAL")["state"] == "probe"

    def test_the_count_is_live_within_the_run(self, wm, evaluator):
        """Why the reading is recomputed per candidate rather than taken once.

        Both candidates pass a reading taken before the loop starts — the lane
        is empty then. The first transition is what fills it, and a pre-computed
        reading would let the second through on the strength of a count that
        stopped being true the moment the first was applied.
        """
        first = self.waiting(wm, "AAA")
        second = self.waiting(wm, "ZZZ")

        result = self.advanced(wm, evaluator, apply=True)

        assert self.proposal_for(result, first)["applied"] is True
        assert self.proposal_for(result, second)["concentration_withheld"] is True
        assert [wm.get(first)["state"], wm.get(second)["state"]] == ["probe", "watch"]

    def scale_registry(self):
        """The shipped registry plus a `probe → scale` trigger that fires.

        Synthetic, because the shipped registry declares none — `scale` appears
        only as a `from:` state today, so nothing currently proposes entering
        it. The rule is worth pinning anyway: it is a property of the gate
        rather than of any trigger, and the day a scale-up transition is
        declared, the guardrail must already know that adding to a position
        adds no name.
        """
        triggers = dict(load_triggers())
        triggers["scale_up"] = {
            "label": "Scale up",
            "rationale": "Synthetic — exercises the gate's add-a-name rule.",
            "from": ["probe"],
            "to": "scale",
            "lane": ["core"],
            "mode": "all",
            "conditions": [
                {"metric": "roce_5yr_avg", "comparator": "gte", "threshold": 20.0}
            ],
        }
        return triggers

    def test_scaling_an_existing_position_is_not_gated(self, wm):
        """The cap counts *names*. A company already in `probe` is already
        counted, so `probe → scale` adds nothing to any count — gating it would
        refuse to let an owner build a position on the grounds that they hold
        it.

        The lane is at its cap here *because of this very company*, which is
        the shape of the mistake: a naive `held + 1 > cap` would count it twice
        and refuse.
        """
        wm.add("HELD")
        wm.transition("HELD", "probe", "seed", applied_by="owner")
        evaluator = TriggerEvaluator(self.scale_registry())

        result = self.advanced(wm, evaluator, apply=True)
        proposal = self.proposal_for(result, "HELD")

        assert proposal["to"] == "scale"
        assert proposal["concentration_withheld"] is False
        assert wm.get("HELD")["state"] == "scale"

    def test_an_unreadable_count_blocks_rather_than_passes(self, wm, evaluator):
        """Absence must not read as headroom — the same rule every other gap in
        this layer follows. A reading that could not be built is not a lane with
        room, it is a lane whose occupancy nobody knows."""
        self.waiting(wm)
        service = StubService(config=self.CAPPED)

        def unreadable(*args, **kwargs):
            raise RuntimeError("the watchlist could not be counted")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "boundless100x.lifecycle.advance._concentration", unreadable
            )
            result = advance(service, wm, evaluator=evaluator, apply=True)

        proposal = self.proposal_for(result, "ASTRAL")
        assert proposal["concentration_withheld"] is True
        assert "could not be built" in proposal["concentration_reasons"][0]
        assert wm.get("ASTRAL")["state"] == "watch"

    def test_a_pre_position_transition_is_never_gated(self, wm, evaluator):
        """Qualifying and watching move no money, so a full lane has no opinion
        about them. Gating them would stop the pipeline that feeds the lane the
        moment the lane filled up."""
        self.full_lane(wm)
        wm.add("EARLY")

        result = self.advanced(wm, evaluator, apply=True)
        proposal = self.proposal_for(result, "EARLY")

        assert proposal["to"] not in ("probe", "scale")
        assert proposal["applied"] is True
        assert proposal["concentration_withheld"] is False


class TestOverridingAConcentrationCap:
    """The escape hatch, and why it is explicit rather than absent.

    A guardrail with no way past it can trap an owner out of their own
    decision — and the way past it that needs no flag is editing the cap in
    `config.yaml`, which leaves no record of the breach at all. So the override
    is a per-run flag, and taking it writes the breach into the append-only
    evidence beside the reason the transition fired.
    """

    CAPPED = TestConcentrationGatesTheMoneyMovingPath.CAPPED

    def setup_full_lane(self, wm):
        wm.add("HELD")
        wm.transition("HELD", "probe", "seed", applied_by="owner")
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")

    def advanced(self, wm, evaluator, **kwargs):
        return advance(
            StubService(config=self.CAPPED), wm, evaluator=evaluator, **kwargs
        )

    def proposal_for(self, result, ticker="ASTRAL"):
        return next(o["proposal"] for o in result["outcomes"] if o["ticker"] == ticker)

    def test_the_override_applies_the_transition(self, wm, evaluator):
        self.setup_full_lane(wm)

        result = self.advanced(wm, evaluator, apply=True, override_caps=True)

        assert self.proposal_for(result)["concentration_withheld"] is False
        assert self.proposal_for(result)["applied"] is True
        assert wm.get("ASTRAL")["state"] == "probe"

    def test_the_breach_is_written_into_the_history(self, wm, evaluator):
        """The point of the flag being a flag. A cap knowingly breached is a
        decision, and a decision this system records is one that can be
        reviewed later — which is what the whole append-only history is for."""
        self.setup_full_lane(wm)

        self.advanced(wm, evaluator, apply=True, override_caps=True)

        evidence = wm.get("ASTRAL")["state_history"][-1]["evidence"]
        assert "concentration:" in evidence
        assert "core lane already holds 1 of a maximum 1" in evidence
        assert "overridden by the owner" in evidence

    def test_the_reasons_still_travel_on_the_proposal(self, wm, evaluator):
        """Overriding suppresses the refusal, never the reading. A surface that
        stopped showing the breach once it was allowed would make the override
        the last time anyone saw it."""
        self.setup_full_lane(wm)

        result = self.advanced(wm, evaluator, apply=True, override_caps=True)

        assert self.proposal_for(result)["concentration_reasons"]

    def test_the_override_alone_still_does_not_apply(self, wm, evaluator):
        """It overrides the cap, not the owner's confirmation. Without
        `--apply` a money-moving transition is still a proposal, and a flag
        about concentration must not quietly become a second way to buy."""
        self.setup_full_lane(wm)

        result = self.advanced(wm, evaluator, override_caps=True)

        assert self.proposal_for(result)["applied"] is False
        assert self.proposal_for(result)["needs_confirmation"] is True
        assert wm.get("ASTRAL")["state"] == "watch"


class TestDecayOutOfWatch:
    """A candidate whose quality decays while it waits for an entry price.

    Both lanes' drop rules stopped at `qualify`, so a company that qualified,
    moved to `watch`, and then fell below its lane's floor was unreachable by
    anything except a fundamentals kill-switch — and those protect capital,
    which a `watch` entry has none of. It sat there indefinitely, which is the
    same criticism the core drop rule's own rationale levels at `screen`: a
    stalled entry is indistinguishable from a considered one.

    The precedence check matters as much as the drop. From `watch`, a buy-zone
    trigger can fire in the same run, and the protective rule has to win — the
    alternative is opening a position on the quarter a company stopped
    qualifying for the lane it is being bought into.
    """

    def test_a_core_watch_entry_that_fails_its_gates_is_dropped(self, wm, evaluator):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")

        outcome = run(StubService(verdict="not_eligible"), wm, evaluator)

        assert outcome["proposal"]["to"] == "dropped"
        assert outcome["proposal"]["trigger_id"] == "qualification_failed"
        assert wm.get("ASTRAL")["state"] == "dropped"

    def test_a_fast_lane_watch_entry_below_the_floor_is_dropped(self, wm, evaluator):
        fast_lane_entry(wm, state="watch")

        outcome = run(
            StubService(metrics=fast_lane_metrics(), composite=4.4),
            wm, evaluator, ticker="ZENSAR",
        )

        assert outcome["proposal"]["to"] == "dropped"
        assert outcome["proposal"]["trigger_id"] == "fast_lane_qualification_failed"
        assert wm.get("ZENSAR")["state"] == "dropped"

    def test_the_drop_outranks_a_buy_zone_firing_in_the_same_run(
        self, wm, evaluator
    ):
        """The company is priced to buy and no longer qualifies. Buying it
        because both rules fired is precisely what precedence exists to stop."""
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")

        outcome = run(StubService(verdict="not_eligible"), wm, evaluator, apply=True)

        assert outcome["proposal"]["to"] == "dropped"
        assert "valuation_buy_zone" in outcome["proposal"]["superseded"]
        assert wm.get("ASTRAL")["state"] == "dropped"

    def test_the_fast_lane_drop_and_its_buy_zone_cannot_both_fire(
        self, wm, evaluator
    ):
        """No precedence question arises on this side, and the reason is worth
        pinning: the lane's drop floor (composite < 5.0) sits *below* its
        `quality_floor` gate (>= 5.5), so a candidate low enough to drop can
        never hold a `qualifies` verdict. The gap between the two thresholds is
        deliberate — a candidate between 5.0 and 5.5 is neither dropped nor
        buyable — and if either number ever moved past the other, this is where
        the overlap would show up as a drop competing with a buy.
        """
        fast_lane_entry(wm, state="watch")

        outcome = run(
            StubService(metrics=fast_lane_metrics(), composite=4.4),
            wm, evaluator, ticker="ZENSAR", apply=True,
        )

        assert outcome["proposal"]["to"] == "dropped"
        assert outcome["lane_gates"]["verdict"] == "not_qualified"
        assert outcome["proposal"]["superseded"] == []

    def test_a_candidate_between_the_two_thresholds_is_left_alone(
        self, wm, evaluator
    ):
        """The other side of that gap, stated so it is a decision rather than
        an accident: above the drop floor and below the entry gate, a fast-lane
        candidate keeps waiting."""
        fast_lane_entry(wm, state="watch")

        outcome = run(
            StubService(metrics=fast_lane_metrics(), composite=5.2),
            wm, evaluator, ticker="ZENSAR", apply=True,
        )

        assert outcome["proposal"] is None
        assert wm.get("ZENSAR")["state"] == "watch"

    def test_a_healthy_watch_entry_is_still_bought_not_dropped(self, wm, evaluator):
        """The floor is a floor, not a new obstacle — a qualifying candidate
        reaches `probe` exactly as it did."""
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")

        outcome = run(StubService(), wm, evaluator, apply=True)

        assert outcome["proposal"]["to"] == "probe"
        assert wm.get("ASTRAL")["state"] == "probe"

    def test_the_drop_auto_applies_because_it_moves_no_money(self, wm, evaluator):
        """A `watch` entry holds no capital, so removing it is not a decision
        that waits for the owner — and a re-qualifying company is one
        `watchlist add` away."""
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "watch", "seed")

        outcome = run(StubService(verdict="not_eligible"), wm, evaluator)

        assert outcome["proposal"]["applied"] is True
        assert outcome["proposal"]["needs_confirmation"] is False
        assert wm.get("ASTRAL")["state_history"][-1]["applied_by"] == "auto"


class TestTheRunsResolutionsAreAllLiveAtOnce:
    """Three interactions that `advance()` composes and no test exercised.

    Every routing test injects an evaluator, which short-circuits pace
    resolution outright; every pace test passes no queue. So the two run-level
    resolutions were each covered alone and never together — and they are not
    independent: a tightened pace threshold withholds a `→ probe` proposal,
    and whether a candidate's entry trigger fired is the *first* key the
    routing ranking sorts on.

    Nothing here injects an evaluator, so the real pace path runs.
    """

    def spread(self, median, contributors=12) -> dict:
        """A corpus reading, in the shape `pace.corpus_spread` returns.

        Defined here rather than imported from `test_pace_modulator`, which
        already imports from this module — the dependency would be a cycle
        between two test files, which is a worse problem than five duplicated
        lines.
        """
        return {
            "median_pp": median,
            "contributors": contributors,
            "tickers": [f"T{i}" for i in range(contributors)],
        }

    def buyable(self) -> dict:
        """Metrics that clear the shipped entry bar but not a tightened one."""
        metrics = healthy_metrics()
        metrics["pe_vs_historical"] = metric(55.0)
        metrics["trailing_peg"] = metric(1.8)
        return metrics

    def watching(self, wm, *tickers):
        for ticker in tickers:
            wm.add(ticker)
            wm.transition(ticker, "watch", "seed")

    def queue_with_proceeds(self, tmp_path, wm):
        """A queue holding one completed, unrouted exit — so routing has
        capital to place and produces a proposal rather than `NO_PROCEEDS`."""
        from boundless100x.lifecycle.reinvestment import ReinvestmentQueue

        queue = ReinvestmentQueue(path=str(tmp_path / "queue.json"))
        wm.add("SOLD")
        wm.transition("SOLD", "probe", "seed", applied_by="owner")
        wm.transition("SOLD", "exit_review", "roiic_below_cost_of_capital")
        transition = wm.transition("SOLD", "exited",
                                   "roiic_below_cost_of_capital",
                                   applied_by="owner")
        event = queue.record_exit(
            ticker="SOLD", lane="core", trigger_id="roiic_below_cost_of_capital",
            friction={"available": False, "reason": "no probe"},
            at="2026-08-01", exit_id="SOLD:2026-08-01",
        )
        queue.record_confirmation(event["exit_id"], at=transition["at"])
        return queue

    def routing_of(self, result) -> dict:
        return result["routing"]

    # ── pace × routing ──

    def mixed_service(self):
        """Two candidates that respond differently to a tightened bar.

        Per ticker, because the interaction is invisible otherwise: with one
        metric set both candidates fire together, the ranking falls to its
        alphabetical last resort, and a test asserting the winner would pass
        whatever the pace did.

        `ASTRAL` clears the shipped entry bar (P/E ≤ 60, PEG ≤ 2.0) and not a
        tightened one. `ZENSAR` clears both. Alphabetically ASTRAL leads, so a
        flip to ZENSAR can only have come from the pace.
        """
        cheap, dear = healthy_metrics(), healthy_metrics()
        dear["pe_vs_historical"], dear["trailing_peg"] = metric(55.0), metric(1.8)
        cheap["pe_vs_historical"], cheap["trailing_peg"] = metric(30.0), metric(1.0)
        by_ticker = {"ASTRAL": dear, "ZENSAR": cheap}

        class PerTicker(StubService):
            def analyze(self, ticker, **kwargs):
                self._metrics = by_ticker.get(ticker, healthy_metrics())
                return super().analyze(ticker, **kwargs)

        return PerTicker()

    def test_a_wide_spread_ranks_the_alphabetically_first_of_two_that_both_fire(
        self, wm, tmp_path
    ):
        """The baseline the next test flips. Both entry triggers fire, both
        composites are equal, so the ranking reaches its last key — the
        ticker — and ASTRAL leads."""
        self.watching(wm, "ASTRAL", "ZENSAR")
        queue = self.queue_with_proceeds(tmp_path, wm)

        wide = advance(self.mixed_service(), wm, queue=queue,
                       pace_reading=self.spread(median=3.0))

        assert wide["pace"]["applied"] is False
        assert self.routing_of(wide)["ranked"] == ["ASTRAL", "ZENSAR"]
        assert self.routing_of(wide)["proposal"]["entry_trigger_fired"] is True

    def test_a_tightened_pace_changes_which_candidate_routing_ranks_first(
        self, wm, tmp_path
    ):
        """The interaction neither suite could see.

        Compressed, ASTRAL no longer clears the entry bar and its proposal is
        withheld; ZENSAR still clears it. Whether an entry trigger fired is the
        *first* key the routing ranking sorts on, so the leader flips — against
        the alphabetical order that decided it a moment ago, which is what says
        the pace caused it.
        """
        self.watching(wm, "ASTRAL", "ZENSAR")
        queue = self.queue_with_proceeds(tmp_path, wm)

        tight = advance(self.mixed_service(), wm, queue=queue,
                        pace_reading=self.spread(median=-4.0))

        assert tight["pace"]["applied"] is True
        assert self.routing_of(tight)["ranked"] == ["ZENSAR", "ASTRAL"]
        assert self.routing_of(tight)["proposal"]["ticker"] == "ZENSAR"

    def test_the_withheld_candidate_is_still_ranked_just_lower(
        self, wm, tmp_path
    ):
        """A tightened entry bar is not a disqualification. ASTRAL is still a
        `watch` entry that could receive proceeds — it simply has no fired
        trigger arguing for it — so it stays in the ranking rather than
        vanishing from it or being reported as blocked."""
        self.watching(wm, "ASTRAL", "ZENSAR")
        queue = self.queue_with_proceeds(tmp_path, wm)

        tight = advance(self.mixed_service(), wm, queue=queue,
                        pace_reading=self.spread(median=-4.0))

        assert "ASTRAL" in self.routing_of(tight)["ranked"]
        assert [b["ticker"] for b in self.routing_of(tight)["blocked"]] == []

    # ── apply × routing ──

    def test_a_candidate_bought_during_the_run_leaves_the_routing_pool(
        self, wm, tmp_path
    ):
        """`advance(apply=True)` with a queue, which nothing drove before.

        The run buys the very candidate the router would have proposed. By the
        time `propose_routing` reads the live watchlist the company sits in
        `probe`, which is outside `CANDIDATE_STATES` — so it must not be
        offered as somewhere to put more capital, and it is not reported as
        *blocked* either, since nothing about it needs unblocking.
        """
        self.watching(wm, "ASTRAL")
        queue = self.queue_with_proceeds(tmp_path, wm)

        result = advance(StubService(), wm, apply=True, queue=queue,
                         pace_reading=self.spread(median=3.0))

        assert wm.get("ASTRAL")["state"] == "probe"
        routing = self.routing_of(result)
        assert routing["ranked"] == []
        assert [b["ticker"] for b in routing["blocked"]] == []

    def test_the_snapshot_still_persists_on_an_applying_run(self, wm, tmp_path):
        """An applying run is a full run, so its view is the canonical one."""
        self.watching(wm, "ASTRAL")
        queue = self.queue_with_proceeds(tmp_path, wm)

        result = advance(StubService(), wm, apply=True, queue=queue,
                         pace_reading=self.spread(median=3.0))

        assert self.routing_of(result)["persisted"] is True
        assert queue.latest_proposal() is not None

    def test_an_applying_run_that_buys_the_last_candidate_says_why_it_names_none(
        self, wm, tmp_path
    ):
        """Not silence: an empty pipeline and a pipeline the run just emptied
        read identically otherwise."""
        self.watching(wm, "ASTRAL")
        queue = self.queue_with_proceeds(tmp_path, wm)

        result = advance(StubService(), wm, apply=True, queue=queue,
                         pace_reading=self.spread(median=3.0))

        assert self.routing_of(result)["reason"]

    # ── the degraded result the real service returns ──

    def test_advance_ticker_survives_the_empty_result_a_failed_fetch_produces(
        self, wm, evaluator
    ):
        """`service.analyze` catches its own exceptions and returns a result
        with empty scores; every stub in this file raises instead. So the shape
        production actually produces on a fetch failure had never reached
        `advance_ticker` — and it is the shape most likely to break it, because
        every downstream reader is handed `{}` rather than an exception to
        propagate.
        """
        wm.add("ASTRAL")

        class Degraded(StubService):
            def analyze(self, ticker, **kwargs):
                return AnalysisResult(ticker=ticker)

        outcome = run(Degraded(), wm, evaluator)

        assert outcome["composite"] is None
        assert outcome["verdict"] == "indeterminate"
        assert outcome["proposal"] is None

    def test_the_degraded_result_blocks_routing_rather_than_passing_it(
        self, wm, evaluator
    ):
        """Fail-closed on the path that matters: no eligibility reading means
        no capital, and an empty result must not read as a clear one."""
        wm.add("ASTRAL")

        class Degraded(StubService):
            def analyze(self, ticker, **kwargs):
                return AnalysisResult(ticker=ticker)

        outcome = run(Degraded(), wm, evaluator)

        assert outcome["routing_safety"]["clear"] is False
        assert outcome["routing_safety"]["reasons"]

    def test_a_degraded_ticker_does_not_stop_the_rest_of_the_run(self, wm):
        """It is not an error — `analyze` returned — so it must flow through
        the loop as an ordinary outcome rather than land in `errors`."""
        wm.add("ASTRAL")
        wm.add("ZENSAR")

        class Degraded(StubService):
            def analyze(self, ticker, **kwargs):
                if ticker == "ASTRAL":
                    return AnalysisResult(ticker=ticker)
                return super().analyze(ticker, **kwargs)

        result = advance(Degraded(), wm, pace_reading=self.spread(median=3.0))

        assert result["errors"] == []
        assert {o["ticker"] for o in result["outcomes"]} == {"ASTRAL", "ZENSAR"}
        assert wm.get("ZENSAR")["state"] == "qualify"


class TestTheDecisionCoreIsPure:
    """`decide` — every rule between the readings and the writes, and no I/O.

    Extracted from `advance_ticker` because a point-in-time replay needs the
    rules without `service.analyze`, and a second statement of them would drift
    from this one with nothing to say which one the money followed. The
    property that makes the extraction worth anything is that it performs no
    writes, so that is asserted directly against a store that raises on any —
    not inferred from `advance_ticker` still passing.
    """

    class ExplodingWatchlist:
        """Any write is a bug. Reads are not offered, because `decide` takes
        its entry as an argument and must never reach for a store at all."""

        def __getattr__(self, name):
            raise AssertionError(
                f"decide() must perform no I/O — it called watchlist.{name}"
            )

    @staticmethod
    def entry(state="watch", lane="core", history=None) -> dict:
        return {
            "state": state,
            "lane": lane,
            "state_history": history or [{"to": state, "at": "2026-01-01"}],
            "catalyst": {},
            "checkpoints": [],
        }

    def core(self, metrics=None, *, state="watch", evaluator=None, **kwargs):
        return decide(
            "ASTRAL",
            self.entry(state=state),
            state,
            "core",
            metrics=metrics if metrics is not None else healthy_metrics(),
            scores={"composite": 6.4, "elements": {}, "flags": []},
            eligibility={"verdict": "eligible"},
            as_of=AS_OF,
            evaluator=evaluator or TriggerEvaluator(load_triggers()),
            **kwargs,
        )

    def test_it_decides_without_touching_a_store(self):
        """The whole point of the split. A store handed in would be a seam the
        replay has to satisfy; `decide` is not given one."""
        decision = self.core()

        assert decision["proposal"]["to"] == "probe"
        # And the belt: nothing in the module reaches for a watchlist by any
        # other route either.
        assert "watchlist" not in decide.__code__.co_varnames

    def test_a_write_attempt_would_be_caught(self):
        """Guards the guard: if a future edit reintroduces a store argument,
        this fixture is what turns it into a failure rather than a silent
        dependency."""
        with pytest.raises(AssertionError, match="no I/O"):
            self.ExplodingWatchlist().set_kill_switch_status("ASTRAL", {})

    def test_applied_is_a_decision_not_a_claim_that_a_write_happened(self):
        """`decide` performs nothing, so `applied` says the caller *should*
        write — the one place the extracted vocabulary could mislead."""
        decision = self.core(state="watch", apply=False)

        assert decision["proposal"]["applied"] is False
        assert decision["proposal"]["needs_confirmation"] is True
        assert decision["moves_money"] is True

    def test_precedence_survives_the_extraction(self):
        """A kill-switch still outranks a buy zone. Restated here rather than
        left to `advance_ticker`'s coverage because this is now where the rule
        lives, and it is the one a replay would most flatter by getting wrong."""
        metrics = healthy_metrics()
        metrics["roiic"] = metric(3.0)

        decision = self.core(metrics)

        assert decision["proposal"]["to"] == "dropped"
        assert "valuation_buy_zone" in decision["proposal"]["superseded"]

    def test_kill_switch_status_is_derived_and_returned_not_written(self):
        status = self.core()["kill_switch_status"]

        assert status  # every declared trigger, per the evaluation
        assert set(status.values()) <= {"fired", "clear", "unknown"}

    def test_nothing_fired_still_reports_moves_money_false(self):
        """The quiet path. `moves_money` is read by the caller to choose
        `applied_by`, so it must exist even when no proposal does."""
        decision = self.core(state="qualify", metrics=healthy_metrics())

        assert decision["moves_money"] is False

    def test_a_concentration_breach_withholds_the_transition(self):
        """The gate is consulted inside the core, so the replay gets it by
        calling rather than by restating it."""
        decision = self.core(
            apply=True,
            concentration_gate=lambda lane, sector: ["the core lane is full"],
        )

        assert decision["proposal"]["concentration_withheld"] is True
        assert decision["proposal"]["applied"] is False
        assert "the core lane is full" in decision["proposal"]["evidence"]

    def test_an_override_lets_it_through_and_records_the_breach(self):
        decision = self.core(
            apply=True,
            concentration_gate=lambda lane, sector: ["the core lane is full"],
            override_caps=True,
        )

        assert decision["proposal"]["applied"] is True
        assert "overridden by the owner" in decision["proposal"]["evidence"]

    def test_the_sector_is_read_off_the_data_dict(self):
        """Takes the raw frames rather than an AnalysisResult, which is what
        lets a truncated replay view answer the same question."""
        decision = decide(
            "ASTRAL",
            self.entry(),
            "watch",
            "core",
            metrics=healthy_metrics(),
            scores={"composite": 6.4, "elements": {}, "flags": []},
            eligibility={"verdict": "eligible"},
            data={"metadata": {"sector": "Chemicals"}},
            as_of=AS_OF,
            evaluator=TriggerEvaluator(load_triggers()),
        )

        assert decision["sector"] == "Chemicals"
