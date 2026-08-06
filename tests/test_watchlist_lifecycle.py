"""The watchlist as lifecycle store.

Two properties are load-bearing. A state is **earned, never granted** — `add`
creates at `screen` and only `transition()` can move a company, always
recording the trigger and evidence. And history is **append-only**, so a
decision that later looks wrong can still be traced to the evidence available
when it was taken.
"""

import json

import pytest

from boundless100x.lifecycle import states
from boundless100x.service import AnalysisResult
from boundless100x.watchlist import (
    APPLIED_AUTO,
    APPLIED_OWNER,
    WatchlistError,
    WatchlistManager,
)


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


def scored(composite=6.4, verdict="eligible") -> AnalysisResult:
    return AnalysisResult(
        ticker="ASTRAL",
        scores={"composite": composite, "elements": {"growth": 7.1}},
        eligibility={"verdict": verdict},
    )


class TestMembership:
    def test_add_creates_at_the_initial_state(self, wm):
        """Qualification is earned by evaluation, not by being added."""
        assert wm.add("astral", notes="Pipe leader") is True
        entry = wm.get("ASTRAL")
        assert entry["state"] == states.INITIAL == "screen"
        assert entry["lane"] == "core"
        assert entry["notes"] == "Pipe leader"

    def test_tickers_are_normalised_to_upper_case(self, wm):
        wm.add("astral")
        assert wm.get("astral") is not None
        assert wm.tickers() == ["ASTRAL"]

    def test_adding_twice_is_refused(self, wm):
        wm.add("ASTRAL")
        assert wm.add("ASTRAL") is False

    def test_remove(self, wm):
        wm.add("ASTRAL")
        assert wm.remove("ASTRAL") is True
        assert wm.remove("ASTRAL") is False

    def test_an_unknown_lane_is_refused(self, wm):
        """The re-rating lane does not exist until Phase 3."""
        with pytest.raises(WatchlistError, match="unknown lane"):
            wm.add("ASTRAL", lane="rerating")

    def test_an_empty_store_loads_cleanly(self, tmp_path):
        assert WatchlistManager(path=str(tmp_path / "absent.json")).list() == []


class TestTransitions:
    def test_a_transition_records_trigger_and_evidence(self, wm):
        wm.add("ASTRAL")
        record = wm.transition(
            "ASTRAL", "qualify", "qualification_passed", evidence="composite 6.4 gte 5.5"
        )

        assert record["from"] == "screen"
        assert record["to"] == "qualify"
        assert record["trigger_id"] == "qualification_passed"
        assert "6.4" in record["evidence"]
        assert wm.get("ASTRAL")["state"] == "qualify"

    def test_history_is_append_only_across_transitions(self, wm):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "qualify", "qualification_passed")
        first = json.dumps(wm.get("ASTRAL")["state_history"][0], sort_keys=True)

        wm.transition("ASTRAL", "watch", "awaiting_entry_price")
        history = wm.get("ASTRAL")["state_history"]

        assert len(history) == 2
        assert json.dumps(history[0], sort_keys=True) == first

    def test_history_survives_a_reload(self, tmp_path):
        path = str(tmp_path / "watchlist.json")
        first = WatchlistManager(path=path)
        first.add("ASTRAL")
        first.transition("ASTRAL", "qualify", "qualification_passed")

        assert len(WatchlistManager(path=path).get("ASTRAL")["state_history"]) == 1

    def test_who_applied_it_is_recorded(self, wm):
        """Auto-applied and owner-confirmed transitions must be distinguishable."""
        wm.add("ASTRAL")
        auto = wm.transition("ASTRAL", "qualify", "t")
        owner = wm.transition("ASTRAL", "probe", "t", applied_by=APPLIED_OWNER)

        assert auto["applied_by"] == APPLIED_AUTO
        assert owner["applied_by"] == APPLIED_OWNER

    def test_an_unknown_state_is_refused(self, wm):
        wm.add("ASTRAL")
        with pytest.raises(WatchlistError, match="unknown state"):
            wm.transition("ASTRAL", "moon", "t")

    def test_transitioning_an_untracked_company_is_refused(self, wm):
        with pytest.raises(WatchlistError, match="not on the watchlist"):
            wm.transition("NOPE", "qualify", "t")

    def test_state_cannot_be_set_except_through_a_transition(self, wm):
        """No setter exists — the trail cannot be bypassed."""
        assert not hasattr(wm, "set_state")


class TestSnapshots:
    def test_a_snapshot_records_score_verdict_and_regime(self, wm):
        wm.add("ASTRAL")
        wm.record_snapshot("ASTRAL", scored(), config_hash="715479102494")

        snapshot = wm.get("ASTRAL")["last_score_snapshot"]
        assert snapshot["composite"] == 6.4
        assert snapshot["verdict"] == "eligible"
        assert snapshot["config_hash"] == "715479102494"
        assert snapshot["elements"]["growth"] == 7.1

    def test_an_unevaluated_verdict_records_indeterminate(self, wm):
        wm.add("ASTRAL")
        wm.record_snapshot("ASTRAL", AnalysisResult(ticker="ASTRAL", scores={"composite": 5.0}))
        assert wm.get("ASTRAL")["last_score_snapshot"]["verdict"] == "indeterminate"

    def test_checkpoints_are_stored_against_the_entry(self, wm):
        wm.add("ASTRAL")
        wm.set_checkpoints("ASTRAL", [{"metric_id": "quarterly_opm_pct"}])
        assert len(wm.get("ASTRAL")["checkpoints"]) == 1

    def test_kill_switch_status_is_stored(self, wm):
        wm.add("ASTRAL")
        wm.set_kill_switch_status("ASTRAL", {"capital_efficiency_break": "clear"})
        assert wm.get("ASTRAL")["kill_switch_status"]["capital_efficiency_break"] == "clear"


class TestSchemaEnforcement:
    """One schema, no migration — an odd entry is a loud error."""

    def write(self, tmp_path, companies) -> str:
        path = tmp_path / "watchlist.json"
        path.write_text(json.dumps({"companies": companies}))
        return str(path)

    def test_a_pre_phase1_entry_raises_naming_the_ticker(self, tmp_path):
        legacy = {"ASTRAL": {
            "added": "2026-02-17T10:51:15", "last_run": None,
            "last_composite": 5.6, "notes": "Pipe sector leader",
        }}
        with pytest.raises(WatchlistError, match="ASTRAL"):
            WatchlistManager(path=self.write(tmp_path, legacy))

    def test_the_error_says_not_to_repair_it_silently(self, tmp_path):
        with pytest.raises(WatchlistError, match="single schema"):
            WatchlistManager(path=self.write(tmp_path, {"ASTRAL": {"added": "x"}}))

    def test_an_unknown_state_on_disk_raises(self, tmp_path):
        entry = {
            "added": "x", "notes": "", "lane": "core", "state": "moon",
            "checkpoints": [], "kill_switch_status": {},
            "last_score_snapshot": None, "state_history": [],
        }
        with pytest.raises(WatchlistError, match="unknown state"):
            WatchlistManager(path=self.write(tmp_path, {"ASTRAL": entry}))

    def test_a_non_object_entry_raises(self, tmp_path):
        with pytest.raises(WatchlistError, match="must be an object"):
            WatchlistManager(path=self.write(tmp_path, {"ASTRAL": "watching"}))


class TestStaleness:
    def test_a_never_scored_entry_is_stale(self, wm):
        wm.add("ASTRAL")
        assert wm.get_stale(90) == ["ASTRAL"]

    def test_a_freshly_scored_entry_is_not_stale(self, wm):
        wm.add("ASTRAL")
        wm.record_snapshot("ASTRAL", scored())
        assert wm.get_stale(90) == []

    def test_an_unparseable_timestamp_counts_as_stale(self, wm):
        """Fail toward re-scoring, never toward assuming freshness."""
        wm.add("ASTRAL")
        wm.get("ASTRAL")["last_score_snapshot"] = {"at": "some time last spring"}
        assert wm.get_stale(90) == ["ASTRAL"]


class TestListing:
    def test_rows_carry_lane_state_and_checkpoint_count(self, wm):
        wm.add("ASTRAL", notes="Pipe leader")
        wm.record_snapshot("ASTRAL", scored(), config_hash="h")
        wm.set_checkpoints("ASTRAL", [{"metric_id": "quarterly_opm_pct"}])

        row = wm.list()[0]
        assert row["ticker"] == "ASTRAL"
        assert row["lane"] == "core"
        assert row["state"] == "screen"
        assert row["last_composite"] == 6.4
        assert row["verdict"] == "eligible"
        assert row["checkpoints"] == 1
