"""`advance`: re-score, evaluate, propose.

The rule under test everywhere here is R7 — transitions that move money are
proposed and wait for the owner; transitions before a position exists apply
themselves. Plus the precedence rule that stops a company being bought into on
the same quarter its thesis broke.
"""

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.advance import advance, advance_ticker, record_checkpoints
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.service import AnalysisResult
from boundless100x.watchlist import WatchlistManager


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


class StubService:
    """Stands in for the pipeline; `advance` only needs analyze() and engine."""

    def __init__(self, metrics=None, composite=6.4, verdict="eligible", data=None):
        self._metrics = metrics if metrics is not None else healthy_metrics()
        self._composite = composite
        self._verdict = verdict
        self._data = data or {}
        self.engine = type("E", (), {"registry_hash": "abc123", "metrics": {}})()
        self.calls: list[str] = []

    def analyze(self, ticker, use_llm=True, **kw):
        self.calls.append(ticker)
        return AnalysisResult(
            ticker=ticker,
            data=self._data,
            metrics=self._metrics,
            scores={"composite": self._composite, "elements": {}},
            eligibility={"verdict": self._verdict},
        )


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


@pytest.fixture
def evaluator():
    return TriggerEvaluator(load_triggers())


def run(service, wm, evaluator, ticker="ASTRAL", apply=False):
    return advance_ticker(service, wm, ticker, evaluator, apply=apply)


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
