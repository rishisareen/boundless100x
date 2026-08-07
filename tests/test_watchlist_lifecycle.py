"""The watchlist as lifecycle store.

Three properties are load-bearing. A state is **earned, never granted** — `add`
creates at `screen` and only `transition()` can move a company, always
recording the trigger and evidence. History is **append-only**, so a decision
that later looks wrong can still be traced to the evidence available when it
was taken. And persistence is **copy-on-write onto a temp file**, so a save
that fails costs the change rather than the store, and never leaves memory
describing a state that was never durable.
"""

import json
from pathlib import Path

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


def entry_on_disk(lane: str = "core", **extra) -> dict:
    """A stored entry exactly as a pre-Phase-3 watchlist holds one."""
    entry = {
        "added": "2026-02-17T10:51:15", "notes": "", "lane": lane,
        "state": "screen", "checkpoints": [], "kill_switch_status": {},
        "last_score_snapshot": None, "state_history": [],
    }
    entry.update(extra)
    return entry


def dump_that_dies_partway(obj, fp, **kwargs):
    """A write that fails after the file already holds bytes.

    The half-written JSON is what makes this the interesting failure: a store
    written in place would now be unloadable, so the assertion that the
    previous file survives is testing the temp-file hop and nothing else.
    """
    fp.write('{"companies": {"ASTR')
    raise RuntimeError("disk full")


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

    def test_a_company_can_be_added_into_the_rerating_lane(self, wm):
        """The fast lane is a lane an owner chooses, not a state earned."""
        assert wm.add("ASTRAL", lane="rerating") is True
        assert wm.get("ASTRAL")["lane"] == "rerating"
        assert wm.get("ASTRAL")["state"] == states.INITIAL

    def test_the_lane_survives_a_reload(self, tmp_path):
        path = str(tmp_path / "watchlist.json")
        WatchlistManager(path=path).add("ASTRAL", lane="rerating")
        assert WatchlistManager(path=path).get("ASTRAL")["lane"] == "rerating"

    def test_an_unknown_lane_is_refused(self, wm):
        with pytest.raises(WatchlistError, match="unknown lane"):
            wm.add("ASTRAL", lane="bogus")

    def test_a_refused_lane_stores_nothing(self, wm):
        with pytest.raises(WatchlistError):
            wm.add("ASTRAL", lane="bogus")
        assert wm.tickers() == []

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


class TestCatalyst:
    """The one fast-lane input the system cannot compute for itself.

    A catalyst is owner-supplied judgement about what will cause a re-rating
    and when. Both halves are required: a named event with no window cannot
    ever come due, and a window with no event cannot be checked.
    """

    def test_a_recorded_catalyst_is_active_and_named(self, wm):
        wm.add("ASTRAL", lane="rerating")
        recorded = wm.record_catalyst("ASTRAL", "Plant commissioning", "FY2027 Q2")

        stored = wm.get("ASTRAL")["catalyst"]
        assert stored["description"] == "Plant commissioning"
        assert stored["expected_by"] == "FY2027 Q2"
        assert stored["status"] == "active"
        assert stored["recorded_at"]
        assert stored == recorded

    def test_a_catalyst_survives_a_reload(self, tmp_path):
        path = str(tmp_path / "watchlist.json")
        first = WatchlistManager(path=path)
        first.add("ASTRAL", lane="rerating")
        first.record_catalyst("ASTRAL", "Plant commissioning", "FY2027 Q2")

        reloaded = WatchlistManager(path=path).get("ASTRAL")["catalyst"]
        assert reloaded["description"] == "Plant commissioning"
        assert reloaded["status"] == "active"

    def test_recording_again_replaces_it_with_a_fresh_timestamp(self, wm):
        wm.add("ASTRAL", lane="rerating")
        wm.record_catalyst("ASTRAL", "Plant commissioning", "FY2027 Q2")
        first = wm.get("ASTRAL")["catalyst"]["recorded_at"]

        wm.record_catalyst("ASTRAL", "Demerger approval", "FY2028 Q1")
        second = wm.get("ASTRAL")["catalyst"]

        assert second["description"] == "Demerger approval"
        assert second["status"] == "active"
        assert second["recorded_at"] != first

    @pytest.mark.parametrize("description,expected_by", [
        ("", "FY2027 Q2"),
        ("Plant commissioning", ""),
        ("", ""),
    ])
    def test_a_catalyst_missing_either_half_is_refused(self, wm, description, expected_by):
        wm.add("ASTRAL", lane="rerating")
        with pytest.raises(WatchlistError):
            wm.record_catalyst("ASTRAL", description, expected_by)
        assert wm.get("ASTRAL").get("catalyst") is None

    def test_the_refusal_names_the_missing_half(self, wm):
        wm.add("ASTRAL", lane="rerating")
        with pytest.raises(WatchlistError, match="expected_by"):
            wm.record_catalyst("ASTRAL", "Plant commissioning", "")

    def test_a_replacement_that_is_refused_leaves_the_old_one_alone(self, wm):
        wm.add("ASTRAL", lane="rerating")
        wm.record_catalyst("ASTRAL", "Plant commissioning", "FY2027 Q2")
        with pytest.raises(WatchlistError):
            wm.record_catalyst("ASTRAL", "Demerger approval", "")
        assert wm.get("ASTRAL")["catalyst"]["description"] == "Plant commissioning"

    def test_marking_it_spent_flips_the_status(self, wm):
        wm.add("ASTRAL", lane="rerating")
        wm.record_catalyst("ASTRAL", "Plant commissioning", "FY2027 Q2")
        wm.mark_catalyst_spent("ASTRAL")

        stored = wm.get("ASTRAL")["catalyst"]
        assert stored["status"] == "spent"
        assert stored["description"] == "Plant commissioning"

    def test_spending_a_catalyst_that_was_never_recorded_is_refused(self, wm):
        """There is nothing to spend, and inventing one would fabricate a thesis."""
        wm.add("ASTRAL", lane="rerating")
        with pytest.raises(WatchlistError, match="no catalyst"):
            wm.mark_catalyst_spent("ASTRAL")

    def test_a_catalyst_on_an_untracked_company_is_refused(self, wm):
        with pytest.raises(WatchlistError, match="not on the watchlist"):
            wm.record_catalyst("NOPE", "Plant commissioning", "FY2027 Q2")
        with pytest.raises(WatchlistError, match="not on the watchlist"):
            wm.mark_catalyst_spent("NOPE")

    def test_a_core_lane_entry_never_grows_a_catalyst_by_itself(self, wm):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "qualify", "t")
        wm.record_snapshot("ASTRAL", scored())
        assert wm.get("ASTRAL").get("catalyst") is None


class TestDurability:
    """A save that fails must cost the change, never the store.

    Both halves matter. The file on disk keeps its previous contents because
    the write lands on a temp file first; and `self.data` keeps describing
    exactly what is on disk because mutators stage on a copy and adopt it only
    after the write returns. Memory ahead of disk is the worse of the two —
    a same-process retry would then build on a change that was never durable.
    """

    def test_a_save_that_dies_partway_leaves_the_previous_file_loadable(self, wm, monkeypatch):
        wm.add("ASTRAL")
        before = Path(wm.path).read_text()

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            wm.add("BAJFINANCE")
        monkeypatch.undo()

        assert Path(wm.path).read_text() == before
        assert WatchlistManager(path=str(wm.path)).tickers() == ["ASTRAL"]

    def test_a_failed_save_leaves_no_temp_file_behind(self, wm, monkeypatch):
        wm.add("ASTRAL")
        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            wm.add("BAJFINANCE")
        monkeypatch.undo()

        assert list(Path(wm.path).parent.glob("*.tmp")) == []

    def test_a_failed_transition_leaves_memory_equal_to_disk(self, wm, monkeypatch):
        wm.add("ASTRAL")

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            wm.transition("ASTRAL", "qualify", "qualification_passed")
        monkeypatch.undo()

        assert wm.get("ASTRAL")["state"] == "screen"
        assert wm.get("ASTRAL")["state_history"] == []
        assert wm.data == WatchlistManager(path=str(wm.path)).data

    def test_a_failed_catalyst_write_leaves_memory_equal_to_disk(self, wm, monkeypatch):
        wm.add("ASTRAL", lane="rerating")

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            wm.record_catalyst("ASTRAL", "Plant commissioning", "FY2027 Q2")
        monkeypatch.undo()

        assert wm.get("ASTRAL").get("catalyst") is None
        assert wm.data == WatchlistManager(path=str(wm.path)).data

    def test_every_commit_bumps_the_revision(self, wm):
        wm.add("ASTRAL")
        first = wm.data["revision"]
        wm.transition("ASTRAL", "qualify", "t")
        wm.set_checkpoints("ASTRAL", [{"metric_id": "quarterly_opm_pct"}])

        assert wm.data["revision"] == first + 2

    def test_a_failed_save_does_not_bump_the_revision(self, wm, monkeypatch):
        wm.add("ASTRAL")
        before = wm.data["revision"]

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            wm.transition("ASTRAL", "qualify", "t")
        monkeypatch.undo()

        assert wm.data["revision"] == before

    def test_the_revision_survives_a_reload(self, tmp_path):
        path = str(tmp_path / "watchlist.json")
        first = WatchlistManager(path=path)
        first.add("ASTRAL")
        first.transition("ASTRAL", "qualify", "t")

        reloaded = WatchlistManager(path=path)
        assert reloaded.data["revision"] == first.data["revision"]
        reloaded.add("BAJFINANCE")
        assert reloaded.data["revision"] == first.data["revision"] + 1

    def test_a_store_written_before_revisions_existed_starts_at_zero(self, tmp_path):
        path = tmp_path / "watchlist.json"
        path.write_text(json.dumps({"companies": {"ASTRAL": entry_on_disk()}}))

        wm = WatchlistManager(path=str(path))
        assert wm.data["revision"] == 0
        wm.transition("ASTRAL", "qualify", "t")
        assert wm.data["revision"] == 1


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

    def test_an_entry_written_before_catalysts_existed_still_validates(self):
        """No migration and no new required key — Phase 3 must not orphan a store."""
        entry = entry_on_disk()
        WatchlistManager._validate_entry("ASTRAL", entry)
        assert entry.get("catalyst") is None

    def test_a_pre_phase3_store_loads_and_keeps_its_entries(self, tmp_path):
        path = self.write(tmp_path, {"ASTRAL": entry_on_disk()})
        assert WatchlistManager(path=path).get("ASTRAL")["lane"] == "core"

    def test_a_rerating_entry_carrying_a_catalyst_validates(self):
        entry = entry_on_disk(lane="rerating", catalyst={
            "description": "Plant commissioning", "expected_by": "FY2027 Q2",
            "status": "active", "recorded_at": "2026-08-07T09:00:00",
        })
        WatchlistManager._validate_entry("ASTRAL", entry)

    def test_a_catalyst_with_an_unknown_status_raises(self, tmp_path):
        entry = entry_on_disk(lane="rerating", catalyst={
            "description": "Plant commissioning", "expected_by": "FY2027 Q2",
            "status": "maybe", "recorded_at": "2026-08-07T09:00:00",
        })
        with pytest.raises(WatchlistError, match="catalyst"):
            WatchlistManager(path=self.write(tmp_path, {"ASTRAL": entry}))


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


class TestCommands:
    """The CLI surface, driven end to end.

    Every case here runs against a redirected store: the real
    `boundless100x/watchlist.json` holds live positions and no test may write it.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        import boundless100x.watchlist as watchlist_module

        path = tmp_path / "watchlist.json"
        monkeypatch.setattr(watchlist_module, "DEFAULT_WATCHLIST_PATH", path)
        return path

    @pytest.fixture
    def run(self, store):
        from typer.testing import CliRunner

        from boundless100x.cli import app

        runner = CliRunner()
        return lambda *args: runner.invoke(app, list(args))

    def test_add_accepts_the_rerating_lane(self, run, store):
        result = run("watchlist", "add", "astral", "--lane", "rerating")
        assert result.exit_code == 0
        assert WatchlistManager(path=str(store)).get("ASTRAL")["lane"] == "rerating"

    def test_add_still_defaults_to_the_core_lane(self, run, store):
        assert run("watchlist", "add", "astral").exit_code == 0
        assert WatchlistManager(path=str(store)).get("ASTRAL")["lane"] == "core"

    def test_add_refuses_an_unknown_lane(self, run, store):
        result = run("watchlist", "add", "astral", "--lane", "moonshot")
        assert result.exit_code != 0
        assert WatchlistManager(path=str(store)).tickers() == []

    def test_a_rerating_entry_round_trips_through_show(self, run):
        run("watchlist", "add", "astral", "--lane", "rerating")
        result = run("watchlist", "show")
        assert result.exit_code == 0
        assert "rerating" in result.output

    def test_catalyst_records_description_and_window(self, run, store):
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        result = run(
            "watchlist", "catalyst", "astral",
            "--description", "Plant commissioning", "--expected-by", "FY2027 Q2",
        )
        assert result.exit_code == 0

        catalyst = WatchlistManager(path=str(store)).get("ASTRAL")["catalyst"]
        assert catalyst["description"] == "Plant commissioning"
        assert catalyst["status"] == "active"

    def test_spent_flips_a_previously_active_catalyst(self, run, store):
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        run("watchlist", "catalyst", "ASTRAL",
            "--description", "Plant commissioning", "--expected-by", "FY2027 Q2")

        result = run("watchlist", "catalyst", "ASTRAL", "--spent")
        assert result.exit_code == 0
        assert WatchlistManager(path=str(store)).get("ASTRAL")["catalyst"]["status"] == "spent"

    def test_spent_cannot_be_combined_with_a_description(self, run, store):
        """A flip that also rewrote the catalyst would change what it refers to."""
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        run("watchlist", "catalyst", "ASTRAL",
            "--description", "Plant commissioning", "--expected-by", "FY2027 Q2")

        result = run("watchlist", "catalyst", "ASTRAL", "--spent",
                     "--description", "Demerger approval")
        assert result.exit_code == 2

        catalyst = WatchlistManager(path=str(store)).get("ASTRAL")["catalyst"]
        assert catalyst["status"] == "active"
        assert catalyst["description"] == "Plant commissioning"

    def test_a_description_without_a_window_names_the_missing_option(self, run):
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        result = run("watchlist", "catalyst", "ASTRAL",
                     "--description", "Plant commissioning")
        assert result.exit_code == 2
        assert "--expected-by" in result.output

    def test_a_window_without_a_description_names_the_missing_option(self, run):
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        result = run("watchlist", "catalyst", "ASTRAL", "--expected-by", "FY2027 Q2")
        assert result.exit_code == 2
        assert "--description" in result.output

    def test_catalyst_with_no_mode_at_all_is_a_usage_error(self, run):
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        assert run("watchlist", "catalyst", "ASTRAL").exit_code == 2

    def test_spending_a_catalyst_that_does_not_exist_exits_nonzero(self, run):
        run("watchlist", "add", "ASTRAL", "--lane", "rerating")
        assert run("watchlist", "catalyst", "ASTRAL", "--spent").exit_code == 1


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
