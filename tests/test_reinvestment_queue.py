"""The reinvestment queue: its durable store, and the routing view it derives.

The first half covers the *store* — appending events and surviving a failed
write. The second half covers what the queue is asked once events exist: which
non-positioned candidate should receive the proceeds, which candidates were
skipped and why, and how long each unrouted exit has been idle. What the
*owner reads* is a third concern, tested against the rendered surfaces in
`tests/test_routing_cli.py`.

Three properties are load-bearing in the store, and all three are here because
the exit protocol rests on them rather than because a store ought to be tidy.

**An append is idempotent, keyed by `exit_id`.** `confirm_exit` writes the
queue event before it writes the transition, precisely so that a crash between
the two leaves a recoverable state; that recovery is "run the command again",
which only works if the second append refuses rather than duplicating.

**The friction payload is stored whole.** A report reads it back later, and a
bare net figure — or worse, a sentence of evidence — cannot be parsed back
apart into gross, holding period, tax regime and basis.

**A failed write costs the change, never the store**, and never leaves memory
ahead of disk. The second half is the dangerous one here: a phantom event
surviving in the live object would let a same-process retry skip the append it
thinks already happened, and the exit would end up recorded in one store only.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from boundless100x.lifecycle import portfolio
from boundless100x.lifecycle.advance import advance, routing_safety
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.lifecycle.reinvestment import (
    NO_PROCEEDS,
    SNAPSHOT_CURRENT,
    SNAPSHOT_PARTIAL,
    SNAPSHOT_STALE,
    SNAPSHOT_UNAVAILABLE,
    ReinvestmentError,
    ReinvestmentQueue,
    snapshot_state,
)
from boundless100x.watchlist import WatchlistManager
from tests.test_lifecycle_advance import StubService, fast_lane_metrics


@pytest.fixture
def queue(tmp_path):
    return ReinvestmentQueue(path=str(tmp_path / "reinvestment_queue.json"))


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


def dump_that_dies_partway(obj, fp, **kwargs):
    """A write that fails once the file already holds bytes.

    The half-written JSON is the interesting failure: written in place the
    store would now be unloadable, so the surviving-file assertion is testing
    the temp-file hop and nothing else.
    """
    fp.write('{"events": [{"kind": "ex')
    raise RuntimeError("disk full")


def friction_payload(**overrides) -> dict:
    """A complete recorded reading, in the shape `friction.model_exit` returns."""
    payload = {
        "available": True,
        "gross_return_pct": 50.0,
        "holding_days": 400,
        "tax_regime": "ltcg",
        "tax_pct": 12.5,
        "taxed": True,
        "slippage_bps": 100,
        "after_slippage_pct": 49.0,
        "net_return_pct": 42.875,
        "basis": "recorded",
    }
    payload.update(overrides)
    return payload


def record(queue, ticker="ASTRAL", exit_id=None, friction=None, **kwargs) -> dict:
    return queue.record_exit(
        ticker=ticker,
        lane=kwargs.get("lane", "core"),
        trigger_id=kwargs.get("trigger_id", "roiic_below_cost_of_capital"),
        friction=friction if friction is not None else friction_payload(),
        at=kwargs.get("at", "2026-08-07"),
        exit_id=exit_id or f"{ticker}:2026-08-01T10:00:00",
    )


class TestExitEvents:
    def test_an_exit_event_carries_the_full_friction_payload(self, queue):
        """Not a bare net figure: a report reads this back, and gross, holding
        period, tax regime and basis cannot be recovered from one number."""
        event = record(queue, "ASTRAL")

        assert event["kind"] == "exit"
        assert event["ticker"] == "ASTRAL"
        assert event["lane"] == "core"
        assert event["trigger_id"] == "roiic_below_cost_of_capital"
        assert event["at"] == "2026-08-07"
        assert event["friction"] == friction_payload()

    def test_the_event_survives_a_reload_intact(self, queue):
        record(queue, "ASTRAL")

        reloaded = ReinvestmentQueue(path=str(queue.path))
        assert reloaded.events() == queue.events()
        assert reloaded.exits()[0]["friction"]["net_return_pct"] == 42.875

    def test_a_duplicate_exit_id_is_refused(self, queue):
        """What makes the append idempotent, and therefore the crash window
        recoverable by re-running the command."""
        record(queue, "ASTRAL", exit_id="ASTRAL:2026-08-01T10:00:00")

        with pytest.raises(ReinvestmentError, match="ASTRAL:2026-08-01T10:00:00"):
            record(queue, "ASTRAL", exit_id="ASTRAL:2026-08-01T10:00:00")

        assert len(queue.exits()) == 1

    def test_a_refused_duplicate_leaves_the_stored_payload_untouched(self, queue):
        """A retry must not overwrite the original reading with a re-priced one."""
        record(queue, "ASTRAL", friction=friction_payload(net_return_pct=42.875))

        with pytest.raises(ReinvestmentError):
            record(queue, "ASTRAL", friction=friction_payload(net_return_pct=-9.0))

        assert queue.exits()[0]["friction"]["net_return_pct"] == 42.875

    def test_an_unavailable_reading_is_stored_as_it_stands(self, queue):
        """A data gap must not stop reality being recorded — the sale happened
        whether or not it could be priced."""
        unavailable = {
            "available": False,
            "reason": "no price series is available for this position",
            "basis": "recorded",
        }
        event = record(queue, "SPLPETRO", friction=unavailable)

        assert event["friction"] == unavailable
        assert queue.find_exit(event["exit_id"])["friction"]["available"] is False

    def test_find_exit_returns_none_for_an_id_nobody_recorded(self, queue):
        assert queue.find_exit("NOPE:2026-01-01T00:00:00") is None

    def test_the_log_is_append_only(self, queue):
        """An earlier event is never rewritten by a later one."""
        first = json.dumps(record(queue, "ASTRAL"), sort_keys=True)
        record(queue, "ZENSAR")

        assert json.dumps(queue.events()[0], sort_keys=True) == first
        assert len(queue.events()) == 2


class TestRoutingEvents:
    def test_a_routing_event_stores_deployed_at_and_recorded_at_separately(self, queue):
        """The idle reading closes at `deployed_at` — when capital actually
        moved — so recording a deployment late does not inflate the window it
        closes."""
        exit_event = record(queue, "ASTRAL")

        routing = queue.record_routing(
            exit_id=exit_event["exit_id"],
            candidate="ZENSAR",
            deployed_at="2026-08-10T09:15:00",
            recorded_at="2026-09-01T18:40:00",
        )

        assert routing["kind"] == "routing"
        assert routing["exit_id"] == exit_event["exit_id"]
        assert routing["candidate"] == "ZENSAR"
        assert routing["deployed_at"] == "2026-08-10T09:15:00"
        assert routing["recorded_at"] == "2026-09-01T18:40:00"
        assert routing["deployed_at"] != routing["recorded_at"]

    def test_routing_is_an_append_not_a_mutation_of_the_exit(self, queue):
        """Marking an exit routed leaves the exit event byte-identical: the log
        is append-only, and the exit's own record is what a report reads."""
        exit_event = record(queue, "ASTRAL")
        before = json.dumps(exit_event, sort_keys=True)

        queue.record_routing(
            exit_id=exit_event["exit_id"],
            candidate="ZENSAR",
            deployed_at="2026-08-10T09:15:00",
        )

        assert json.dumps(queue.exits()[0], sort_keys=True) == before
        assert len(queue.events()) == 2

    def test_an_unrouted_exit_is_listed_until_a_routing_event_closes_it(self, queue):
        first = record(queue, "ASTRAL", exit_id="ASTRAL:a")
        second = record(queue, "SPLPETRO", exit_id="SPLPETRO:b")

        assert [e["exit_id"] for e in queue.unrouted_exits()] == ["ASTRAL:a", "SPLPETRO:b"]

        queue.record_routing(
            exit_id=first["exit_id"], candidate="ZENSAR",
            deployed_at="2026-08-10T09:15:00",
        )

        assert [e["exit_id"] for e in queue.unrouted_exits()] == [second["exit_id"]]
        assert queue.routing_for("ASTRAL:a")["candidate"] == "ZENSAR"

    def test_routing_an_exit_nobody_recorded_is_refused(self, queue):
        """A routing event that closes nothing is unreadable — the idle
        reading it claims to end has no beginning."""
        with pytest.raises(ReinvestmentError, match="NOPE:2026-01-01"):
            queue.record_routing(
                exit_id="NOPE:2026-01-01", candidate="ZENSAR",
                deployed_at="2026-08-10T09:15:00",
            )
        assert queue.events() == []

    def test_routing_an_already_routed_exit_is_refused(self, queue):
        exit_event = record(queue, "ASTRAL")
        queue.record_routing(
            exit_id=exit_event["exit_id"], candidate="ZENSAR",
            deployed_at="2026-08-10T09:15:00",
        )

        with pytest.raises(ReinvestmentError):
            queue.record_routing(
                exit_id=exit_event["exit_id"], candidate="BAJFINANCE",
                deployed_at="2026-08-20T09:15:00",
            )

        assert len(queue.events()) == 2


class TestDurability:
    """A save that fails must cost the change, never the store.

    Mirrors `WatchlistManager`'s own durability contract, because it is the
    same argument about the same kind of file — and because `confirm_exit`
    writes both stores in sequence, so a difference between them would be a
    hole in the exit protocol rather than an inconsistency of style.
    """

    def test_a_save_that_dies_partway_leaves_the_previous_file_loadable(
        self, queue, monkeypatch
    ):
        record(queue, "ASTRAL")
        before = Path(queue.path).read_text()

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            record(queue, "ZENSAR", exit_id="ZENSAR:x")
        monkeypatch.undo()

        assert Path(queue.path).read_text() == before
        assert len(ReinvestmentQueue(path=str(queue.path)).exits()) == 1

    def test_a_failed_save_leaves_no_temp_file_behind(self, queue, monkeypatch):
        record(queue, "ASTRAL")

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            record(queue, "ZENSAR", exit_id="ZENSAR:x")
        monkeypatch.undo()

        assert list(Path(queue.path).parent.glob("*.tmp")) == []

    def test_a_failed_append_leaves_memory_equal_to_disk(self, queue, monkeypatch):
        """No phantom event for a same-process retry to build on: a live object
        holding an event that was never durable would let the retry skip the
        append it believes already landed."""
        record(queue, "ASTRAL")

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            record(queue, "ZENSAR", exit_id="ZENSAR:x")
        monkeypatch.undo()

        assert [e["ticker"] for e in queue.exits()] == ["ASTRAL"]
        assert queue.data == ReinvestmentQueue(path=str(queue.path)).data

    def test_every_commit_bumps_the_revision(self, queue):
        """A sibling reader compares revisions to decide whether its view is
        current, so the counter counts durable commits."""
        record(queue, "ASTRAL", exit_id="ASTRAL:a")
        first = queue.data["revision"]
        record(queue, "SPLPETRO", exit_id="SPLPETRO:b")
        queue.record_routing(
            exit_id="ASTRAL:a", candidate="ZENSAR", deployed_at="2026-08-10T09:15:00"
        )

        assert queue.data["revision"] == first + 2

    def test_a_failed_save_does_not_bump_the_revision(self, queue, monkeypatch):
        record(queue, "ASTRAL")
        before = queue.data["revision"]

        monkeypatch.setattr(json, "dump", dump_that_dies_partway)
        with pytest.raises(RuntimeError):
            record(queue, "ZENSAR", exit_id="ZENSAR:x")
        monkeypatch.undo()

        assert queue.data["revision"] == before

    def test_the_revision_survives_a_reload(self, queue):
        record(queue, "ASTRAL")

        reloaded = ReinvestmentQueue(path=str(queue.path))
        assert reloaded.data["revision"] == queue.data["revision"]

    def test_a_store_written_before_revisions_existed_starts_at_zero(self, tmp_path):
        path = tmp_path / "reinvestment_queue.json"
        path.write_text(json.dumps({"events": []}))

        queue = ReinvestmentQueue(path=str(path))
        assert queue.data["revision"] == 0
        record(queue, "ASTRAL")
        assert queue.data["revision"] == 1


class TestStoreShape:
    def test_a_missing_file_loads_as_an_empty_queue(self, tmp_path):
        queue = ReinvestmentQueue(path=str(tmp_path / "nothing-here.json"))

        assert queue.events() == []
        assert queue.unrouted_exits() == []
        assert queue.data["latest_proposal"] is None

    def test_the_latest_proposal_slot_round_trips_through_an_append(self, tmp_path):
        """The store must be able to *hold* the replaceable proposal slot even
        though appending an event never touches it — a routing snapshot written
        by one command must survive the next exit."""
        path = tmp_path / "reinvestment_queue.json"
        proposal = {"as_of": "2026-08-07", "status": "current", "proposal": {"ticker": "ZENSAR"}}
        path.write_text(json.dumps({"events": [], "latest_proposal": proposal, "revision": 3}))

        queue = ReinvestmentQueue(path=str(path))
        record(queue, "ASTRAL")

        assert queue.data["latest_proposal"] == proposal
        assert ReinvestmentQueue(path=str(path)).data["latest_proposal"] == proposal

    def test_an_event_missing_required_keys_is_a_loud_error(self, tmp_path):
        path = tmp_path / "reinvestment_queue.json"
        path.write_text(json.dumps({"events": [{"kind": "exit", "ticker": "ASTRAL"}]}))

        with pytest.raises(ReinvestmentError, match="exit_id"):
            ReinvestmentQueue(path=str(path))

    def test_an_unknown_event_kind_is_a_loud_error(self, tmp_path):
        path = tmp_path / "reinvestment_queue.json"
        path.write_text(json.dumps({"events": [{"kind": "teleport", "exit_id": "x"}]}))

        with pytest.raises(ReinvestmentError, match="teleport"):
            ReinvestmentQueue(path=str(path))


# ── The routing view ────────────────────────────────────────────────────────
#
# `propose_routing` answers three questions at once, and the second is the one
# that is easy to leave out: which candidates were *skipped*, and why. A view
# that reported only its winner would render an all-blocked run identically to
# an empty watchlist, and those two mean opposite things to an owner deciding
# whether the system is working.

AS_OF = date(2026, 8, 7)

# Caps small enough that a two-name lane is already full — the arithmetic is
# the same at eight, and a fixture that needs nine entries to test one rule is
# a fixture nobody reads.
CONFIG = {
    "portfolio": {
        "max_positioned_per_lane": {"core": 2, "rerating": 2},
        "max_positioned_per_sector": 3,
    }
}


def outcome_for(wm, ticker, *, composite=6.5, safety=None, buy_zone=False,
                sector=None, verdict="eligible") -> dict:
    """One `advance_ticker` outcome, in the shape the real loop returns.

    Hand-built rather than produced by a run, so a ranking test can state the
    exact trigger state it means to rank. The end-to-end fixture at the bottom
    of this file runs the real `advance()` against the same code path, which is
    what keeps this shape honest.
    """
    entry = wm.get(ticker)
    lane, state = entry["lane"], entry["state"]
    return {
        "ticker": ticker,
        "state": state,
        "lane": lane,
        "sector": sector,
        "composite": composite,
        "verdict": verdict,
        "lane_gates": None,
        "proposal": {
            "ticker": ticker,
            "from": state,
            "to": "probe",
            "trigger_id": "valuation_buy_zone",
            "evidence": "P/E at the 22nd percentile of its own history",
            "applied": False,
            "needs_confirmation": True,
            "superseded": [],
        } if buy_zone else None,
        "indeterminate": [],
        "checkpoints": {},
        "checkpoint_outcomes": [],
        "routing_safety": safety if safety is not None else {
            "lane": lane, "clear": True, "reasons": []
        },
    }


def candidate(wm, ticker, *, state="watch", lane="core", **kwargs) -> dict:
    """Track a company at a given state and return its advance outcome."""
    wm.add(ticker, lane=lane)
    if state != "screen":
        wm.transition(ticker, state, "seed")
    return outcome_for(wm, ticker, **kwargs)


def positioned(wm, ticker, lane="core", state="probe") -> None:
    """A name already holding capital — counted by the caps, never a candidate."""
    wm.add(ticker, lane=lane)
    wm.transition(ticker, state, "seed", applied_by="owner")


def concentration_for(wm, config=None, sectors=None) -> dict:
    """The reading `advance()` builds, from the same watchlist the test wrote."""
    sectors = sectors or {}
    entries = [
        {
            "ticker": ticker,
            "lane": wm.get(ticker)["lane"],
            "state": wm.get(ticker)["state"],
            "sector": sectors.get(ticker),
        }
        for ticker in wm.tickers()
    ]
    return portfolio.check_concentration(entries, config or CONFIG)


def completed_exit(wm, queue, ticker="SOLD", lane="core", at="2026-08-01") -> dict:
    """A sale recorded in *both* stores — the only kind that is routable proceeds."""
    wm.add(ticker, lane=lane)
    wm.transition(ticker, "probe", "seed", applied_by="owner")
    wm.transition(ticker, "exit_review", "roiic_below_cost_of_capital")
    wm.transition(ticker, "exited", "roiic_below_cost_of_capital",
                  applied_by="owner")
    return queue.record_exit(
        ticker=ticker, lane=lane, trigger_id="roiic_below_cost_of_capital",
        friction=friction_payload(), at=at, exit_id=f"{ticker}:{at}",
    )


def stranded_exit(wm, queue, ticker="HALFSOLD", lane="core", at="2026-08-01") -> dict:
    """KTD10's crash window: the queue event landed, the transition did not."""
    wm.add(ticker, lane=lane)
    wm.transition(ticker, "probe", "seed", applied_by="owner")
    wm.transition(ticker, "exit_review", "roiic_below_cost_of_capital")
    return queue.record_exit(
        ticker=ticker, lane=lane, trigger_id="roiic_below_cost_of_capital",
        friction=friction_payload(), at=at, exit_id=f"{ticker}:{at}",
    )


class TestRanking:
    def test_a_fired_buy_zone_outranks_a_quiet_qualify(self, wm, queue):
        """Trigger state, not lifecycle state alone, and not the composite.

        The quiet candidate is given the *better* score on purpose: if the
        ranking fell back to the composite, it would win, and the run would
        route proceeds into a company whose entry condition has not been met.
        """
        completed_exit(wm, queue)
        outcomes = [
            candidate(wm, "QUIET", state="qualify", composite=8.0),
            candidate(wm, "READY", state="watch", buy_zone=True, composite=5.5),
        ]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"]["ticker"] == "READY"
        assert view["proposal"]["trigger_id"] == "valuation_buy_zone"
        assert "22nd percentile" in view["proposal"]["evidence"]

    def test_a_positioned_name_is_never_a_candidate(self, wm, queue):
        """Proceeds go to something not already holding them."""
        completed_exit(wm, queue)
        held = candidate(wm, "HELD", state="probe", buy_zone=True)
        quiet = candidate(wm, "QUIET", state="qualify")

        view = queue.propose_routing(wm, [held, quiet], concentration_for(wm),
                                     as_of=AS_OF)

        assert view["proposal"]["ticker"] == "QUIET"
        assert "HELD" not in [b["ticker"] for b in view["blocked"]]

    def test_a_lane_at_its_cap_skips_to_the_next_candidate(self, wm, queue):
        """U5's guardrail, consulted before a proposal rather than after it."""
        completed_exit(wm, queue)
        positioned(wm, "HOLD1")
        positioned(wm, "HOLD2")
        outcomes = [
            candidate(wm, "COREONE", state="watch", buy_zone=True, composite=7.0),
            candidate(wm, "FASTONE", state="watch", lane="rerating",
                      buy_zone=True, composite=6.0),
        ]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"]["ticker"] == "FASTONE"
        blocked = {b["ticker"]: b for b in view["blocked"]}
        assert "COREONE" in blocked
        assert any("core" in reason for reason in blocked["COREONE"]["reasons"])

    def test_a_sector_already_at_its_cap_blocks_a_candidate_joining_it(
        self, wm, queue
    ):
        """The other half of U5's guardrail — a cap on correlated names, in
        counts of names exactly like the lane cap."""
        completed_exit(wm, queue)
        for ticker in ("HOLD1", "HOLD2", "HOLD3"):
            positioned(wm, ticker)
        sectors = dict.fromkeys(("HOLD1", "HOLD2", "HOLD3", "CHEMONE"), "Chemicals")
        config = {"portfolio": {
            "max_positioned_per_lane": {"core": 9, "rerating": 9},
            "max_positioned_per_sector": 3,
        }}
        outcomes = [candidate(wm, "CHEMONE", state="watch", buy_zone=True,
                              sector="Chemicals")]

        view = queue.propose_routing(
            wm, outcomes, concentration_for(wm, config, sectors), as_of=AS_OF
        )

        assert view["proposal"] is None
        assert any(
            "Chemicals" in reason for reason in view["blocked"][0]["reasons"]
        )

    def test_a_pre_position_proposal_is_not_an_entry_trigger(self, wm, queue):
        """A run that just moved a company `screen → qualify` has not found a
        buy zone, and must not outrank one that has."""
        completed_exit(wm, queue)
        promoted = candidate(wm, "PROMOTED", state="qualify", composite=9.0)
        promoted["proposal"] = {
            "ticker": "PROMOTED", "to": "watch",
            "trigger_id": "awaiting_entry_price", "evidence": "quality floor met",
        }
        ready = candidate(wm, "READY", state="watch", buy_zone=True, composite=5.0)

        view = queue.propose_routing(wm, [promoted, ready], concentration_for(wm),
                                     as_of=AS_OF)

        assert view["proposal"]["ticker"] == "READY"


class TestRoutingSafetyIsConsulted:
    """KTD11's posture, and the lane asymmetry it exists for.

    Built through the real `routing_safety()` rather than by hand-setting
    `clear`, because the property under test is that the *lane* decides which
    eligibility question is asked — a hand-written payload would assert only
    that a boolean is read.
    """

    def test_a_core_candidate_failing_the_100x_gates_is_skipped(self, wm, queue):
        completed_exit(wm, queue)
        blocked_safety = routing_safety("core", {"verdict": "not_eligible"}, {})
        outcomes = [
            candidate(wm, "COREONE", state="watch", buy_zone=True, composite=8.0,
                      safety=blocked_safety),
            candidate(wm, "CORETWO", state="qualify", composite=5.0),
        ]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"]["ticker"] == "CORETWO"
        assert view["blocked"][0]["ticker"] == "COREONE"
        assert view["blocked"][0]["reasons"]

    def test_a_rerating_candidate_with_the_same_verdict_is_routed(self, wm, queue):
        """The fast lane must be able to receive capital from its own exits.

        Same `not_eligible` 100x verdict as the core candidate above; the lane
        gates say `qualifies`, and that is the question its lane asks.
        """
        completed_exit(wm, queue, lane="rerating")
        safety = routing_safety(
            "rerating", {"verdict": "not_eligible"}, {}, {"verdict": "qualifies"}
        )
        outcomes = [
            candidate(wm, "FASTONE", state="watch", lane="rerating",
                      buy_zone=True, safety=safety, verdict="not_eligible"),
        ]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"]["ticker"] == "FASTONE"
        assert view["blocked"] == []

    @pytest.mark.parametrize("lane", ["core", "rerating"])
    def test_low_data_coverage_blocks_both_lanes(self, wm, queue, lane):
        """A score resting on incomplete data is no basis for deploying capital,
        whichever lane asked."""
        completed_exit(wm, queue)
        scores = {"flags": ["low_data_coverage"]}
        safety = routing_safety(
            lane, {"verdict": "eligible"}, scores, {"verdict": "qualifies"}
        )
        outcomes = [
            candidate(wm, "THIN", state="watch", lane=lane, buy_zone=True,
                      safety=safety)
        ]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"] is None
        assert view["blocked"][0]["ticker"] == "THIN"
        # The safety reading's own words, carried through rather than
        # paraphrased: the router explains the block in the language the check
        # that made it used.
        assert view["blocked"][0]["reasons"] == safety["reasons"]
        assert "incomplete evidence" in view["blocked"][0]["reasons"][0]

    @pytest.mark.parametrize("lane_gate_result", [
        None,
        {"verdict": "indeterminate"},
        {"verdict": "probably_fine"},
    ])
    def test_a_fast_lane_verdict_that_is_not_qualifies_blocks(
        self, wm, queue, lane_gate_result
    ):
        """Fail-closed vocabulary: `not_qualified` is not `not_eligible`, and a
        word nobody declared is not a clearance."""
        completed_exit(wm, queue, lane="rerating")
        safety = routing_safety(
            "rerating", {"verdict": "eligible"}, {}, lane_gate_result
        )
        outcomes = [
            candidate(wm, "FASTONE", state="watch", lane="rerating",
                      buy_zone=True, safety=safety)
        ]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"] is None
        assert view["blocked"][0]["reasons"]


class TestNothingToPropose:
    def test_an_all_blocked_run_is_not_an_empty_one(self, wm, queue):
        """The distinction the `blocked` field exists for: "everything was
        blocked" and "nothing exists" are opposite diagnoses."""
        completed_exit(wm, queue)
        safety = routing_safety("core", {"verdict": "not_eligible"}, {})
        outcomes = [candidate(wm, "COREONE", state="watch", buy_zone=True,
                              safety=safety)]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"] is None
        assert [b["ticker"] for b in view["blocked"]] == ["COREONE"]
        assert view["reason"]

    def test_an_empty_watchlist_reports_nothing_to_propose_not_an_error(
        self, wm, queue
    ):
        view = queue.propose_routing(wm, [], concentration_for(wm), as_of=AS_OF)

        assert view["proposal"] is None
        assert view["blocked"] == []
        assert view["idle"] == []
        assert view["reason"]

    def test_a_clear_candidate_with_no_proceeds_is_still_not_a_proposal(
        self, wm, queue
    ):
        """Capital that does not exist cannot be routed toward a candidate."""
        outcomes = [candidate(wm, "READY", state="watch", buy_zone=True)]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"] is None
        assert view["reason"] == NO_PROCEEDS

    def test_propose_routing_never_transitions(self, wm, queue, monkeypatch):
        """The proposal is inert data (R10) — a caller decides what to do with
        it. Asserted by making a transition impossible rather than by comparing
        states, so the claim is about the call and not about its result."""
        completed_exit(wm, queue)
        outcomes = [candidate(wm, "READY", state="watch", buy_zone=True)]

        def no_transitions(*args, **kwargs):
            raise AssertionError("propose_routing moved a company")

        monkeypatch.setattr(wm, "transition", no_transitions)
        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"]["ticker"] == "READY"
        assert wm.get("READY")["state"] == "watch"


class TestIdleReadings:
    def test_an_unrouted_exit_reports_the_days_since_the_sale(self, wm, queue):
        completed_exit(wm, queue, at="2026-08-01")

        view = queue.propose_routing(wm, [], concentration_for(wm), as_of=AS_OF)

        assert [r["exit_id"] for r in view["idle"]] == ["SOLD:2026-08-01"]
        assert view["idle"][0]["idle_days"] == 6
        assert view["idle"][0]["closed"] is False

    def test_a_routed_exit_closes_its_reading_at_deployed_at(self, wm, queue):
        """Not at `recorded_at`: recording a deployment late must not inflate
        the window it closes."""
        event = completed_exit(wm, queue, at="2026-08-01")
        queue.record_routing(
            exit_id=event["exit_id"], candidate="READY",
            deployed_at="2026-08-04T10:00:00", recorded_at="2026-09-30T18:00:00",
        )

        view = queue.propose_routing(wm, [], concentration_for(wm), as_of=AS_OF)
        reading = queue.exit_views(wm, as_of=AS_OF)[0]

        assert view["idle"] == []
        assert reading["closed"] is True
        assert reading["idle_days"] == 3
        assert reading["routed_into"] == "READY"

    def test_an_exit_still_in_exit_review_is_excluded_from_routing(self, wm, queue):
        """Proceeds from an unfinished exit record are not routable — the
        KTD10 crash window, named rather than silently counted."""
        stranded_exit(wm, queue)
        outcomes = [candidate(wm, "READY", state="watch", buy_zone=True)]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"] is None
        assert view["reason"] == NO_PROCEEDS
        assert [e["ticker"] for e in view["incomplete"]] == ["HALFSOLD"]

    def test_a_completed_exit_beside_a_stranded_one_still_routes(self, wm, queue):
        completed_exit(wm, queue)
        stranded_exit(wm, queue)
        outcomes = [candidate(wm, "READY", state="watch", buy_zone=True)]

        view = queue.propose_routing(wm, outcomes, concentration_for(wm), as_of=AS_OF)

        assert view["proposal"]["ticker"] == "READY"
        assert [e["exit_id"] for e in view["idle"]] == ["SOLD:2026-08-01"]


class TestSnapshotState:
    """Precedence, and why it is that order.

    `Partial` outranks `Stale` because an incomplete run and a superseded one
    need different fixes — re-running helps the first, and knowing *which*
    ticker failed is the whole point. Both outrank `Current`, and only
    `Current` may render a proposal.
    """

    def snapshot(self, **overrides) -> dict:
        payload = {
            "as_of": "2026-08-07",
            "generated_at": "2026-08-07T10:00:00",
            "status": SNAPSHOT_CURRENT,
            "watchlist_revision": 4,
            "queue_revision": 2,
            "proposal": {"ticker": "READY"},
            "blocked": [],
            "idle": [],
            "errors": [],
        }
        payload.update(overrides)
        return payload

    def test_no_snapshot_reads_unavailable(self):
        reading = snapshot_state(None, 4, 2)

        assert reading["state"] == SNAPSHOT_UNAVAILABLE
        assert reading["renders_proposal"] is False
        assert reading["reason"]

    def test_errored_tickers_read_partial_even_when_revisions_advanced(self):
        """Both conditions hold; the more actionable one wins, and it names the
        ticker that failed."""
        reading = snapshot_state(
            self.snapshot(status=SNAPSHOT_PARTIAL, errors=["SPLPETRO"]), 9, 7
        )

        assert reading["state"] == SNAPSHOT_PARTIAL
        assert reading["errors"] == ["SPLPETRO"]
        assert reading["renders_proposal"] is False

    @pytest.mark.parametrize("watchlist_revision,queue_revision", [
        (5, 2), (4, 3), (5, 3),
    ])
    def test_either_store_advancing_reads_stale(
        self, watchlist_revision, queue_revision
    ):
        reading = snapshot_state(self.snapshot(), watchlist_revision, queue_revision)

        assert reading["state"] == SNAPSHOT_STALE
        assert reading["renders_proposal"] is False

    def test_a_snapshot_with_no_revisions_recorded_reads_stale(self):
        """Fail-closed: a snapshot that cannot prove it is current is not."""
        reading = snapshot_state(
            self.snapshot(watchlist_revision=None, queue_revision=None), 4, 2
        )

        assert reading["state"] == SNAPSHOT_STALE

    def test_matching_revisions_and_no_errors_read_current(self):
        reading = snapshot_state(self.snapshot(), 4, 2)

        assert reading["state"] == SNAPSHOT_CURRENT
        assert reading["renders_proposal"] is True


class TestSnapshotWriting:
    """Written once, at the end of a full run, through the store's atomic path."""

    def evaluator(self):
        return TriggerEvaluator(load_triggers())

    def test_a_full_run_writes_a_current_snapshot(self, wm, queue):
        completed_exit(wm, queue)
        wm.add("READY")
        wm.transition("READY", "watch", "seed")

        result = advance(StubService(), wm, evaluator=self.evaluator(),
                         as_of=AS_OF, queue=queue)

        snapshot = ReinvestmentQueue(path=str(queue.path)).latest_proposal()
        assert result["routing"]["available"] is True
        assert snapshot["status"] == SNAPSHOT_CURRENT
        assert snapshot["errors"] == []
        assert snapshot["as_of"] == str(AS_OF)

    def test_the_written_snapshot_reads_current_immediately_afterwards(self, wm, queue):
        """The write bumps the queue's own revision, so a snapshot recording the
        revision it was *generated* against would be stale the moment it
        landed."""
        completed_exit(wm, queue)
        wm.add("READY")
        wm.transition("READY", "watch", "seed")

        advance(StubService(), wm, evaluator=self.evaluator(), as_of=AS_OF,
                queue=queue)

        reloaded_queue = ReinvestmentQueue(path=str(queue.path))
        reloaded_wm = WatchlistManager(path=str(wm.path))
        reading = snapshot_state(
            reloaded_queue.latest_proposal(),
            reloaded_wm.data["revision"], reloaded_queue.data["revision"],
        )
        assert reading["state"] == SNAPSHOT_CURRENT

    def test_a_quarterly_run_never_overwrites_the_canonical_snapshot(self, wm, queue):
        """A stale subset must not promote a lower-ranked candidate merely
        because the better one was not re-scored that day."""
        completed_exit(wm, queue)
        wm.add("READY")
        wm.transition("READY", "watch", "seed")
        advance(StubService(), wm, evaluator=self.evaluator(), as_of=AS_OF,
                queue=queue)
        canonical = json.dumps(queue.latest_proposal(), sort_keys=True)

        advance(StubService(), wm, evaluator=self.evaluator(), as_of=AS_OF,
                queue=queue, quarterly=True)

        assert json.dumps(queue.latest_proposal(), sort_keys=True) == canonical
        assert ReinvestmentQueue(path=str(queue.path)).latest_proposal() == \
            queue.latest_proposal()

    def test_a_quarterly_run_still_reports_its_routing_view(self, wm, queue):
        """Not persisted is not the same as not computed — and the run says
        which of the two happened."""
        completed_exit(wm, queue)
        result = advance(StubService(), wm, evaluator=self.evaluator(),
                         as_of=AS_OF, queue=queue, quarterly=True)

        assert result["routing"]["available"] is True
        assert result["routing"]["persisted"] is False
        assert "subset" in result["routing"]["persist_reason"]

    def test_a_run_with_an_errored_ticker_writes_partial_naming_it(self, wm, queue):
        completed_exit(wm, queue)
        wm.add("GOOD")
        wm.add("BAD")

        class Flaky(StubService):
            def analyze(self, ticker, use_llm=True, **kw):
                if ticker == "BAD":
                    raise RuntimeError("fetch failed")
                return super().analyze(ticker, use_llm=use_llm, **kw)

        advance(Flaky(), wm, evaluator=self.evaluator(), as_of=AS_OF, queue=queue)

        snapshot = queue.latest_proposal()
        assert snapshot["status"] == SNAPSHOT_PARTIAL
        assert snapshot["errors"] == ["BAD"]

    def test_no_queue_means_routing_is_unavailable_and_nothing_persists(self, wm):
        """Idle readings and route state live in the event log; without it there
        is no partial view to fake."""
        wm.add("READY")
        wm.transition("READY", "watch", "seed")

        result = advance(StubService(), wm, evaluator=self.evaluator(), as_of=AS_OF)

        assert result["routing"]["available"] is False
        assert result["routing"]["reason"]
        assert "proposal" not in result["routing"]

    def test_a_failure_building_the_view_never_costs_the_run(self, wm, queue,
                                                             monkeypatch):
        completed_exit(wm, queue)
        wm.add("READY")

        def broken(*args, **kwargs):
            raise RuntimeError("the router fell over")

        monkeypatch.setattr(queue, "propose_routing", broken)
        result = advance(StubService(), wm, evaluator=self.evaluator(),
                         as_of=AS_OF, queue=queue)

        assert [o["ticker"] for o in result["outcomes"]] == ["SOLD", "READY"]
        assert result["routing"]["available"] is False
        assert "fell over" in result["routing"]["reason"]


class TestEndToEnd:
    """The verification fixture: two lanes, one exit, one outstanding candidate.

    Everything here runs through the real `advance()` — real triggers, real
    lane gates, real concentration reading — so a change that breaks the seam
    between them fails here rather than in a hand-built outcome dict.
    """

    def two_lane_run(self, wm, queue):
        completed_exit(wm, queue, ticker="SOLD", lane="core")
        wm.add("ZENSAR", lane="rerating")
        wm.record_catalyst("ZENSAR", "Demerger of the services arm", "2026-12-31")
        wm.transition("ZENSAR", "watch", "seed")
        service = StubService(
            metrics=fast_lane_metrics(), composite=6.5, verdict="not_eligible"
        )
        return advance(service, wm, evaluator=TriggerEvaluator(load_triggers()),
                       as_of=AS_OF, queue=queue)

    def test_the_run_proposes_the_outstanding_candidate_with_its_evidence(
        self, wm, queue
    ):
        result = self.two_lane_run(wm, queue)

        proposal = result["routing"]["proposal"]
        assert proposal["ticker"] == "ZENSAR"
        assert proposal["lane"] == "rerating"
        assert proposal["trigger_id"] == "fast_lane_buy_zone"
        assert proposal["evidence"]

    def test_the_exit_is_listed_idle_until_it_is_routed(self, wm, queue):
        self.two_lane_run(wm, queue)
        assert [r["exit_id"] for r in queue.latest_proposal()["idle"]] == \
            ["SOLD:2026-08-01"]

        queue.record_routing(
            exit_id="SOLD:2026-08-01", candidate="ZENSAR",
            deployed_at="2026-08-05T10:00:00",
        )

        views = queue.exit_views(wm, as_of=AS_OF)
        assert views[0]["closed"] is True
        assert views[0]["routed_into"] == "ZENSAR"
        assert queue.unrouted_exits() == []

    def test_a_core_lane_exit_can_fund_the_fast_lane(self, wm, queue):
        """The asymmetry stated as a portfolio fact rather than a gate reading:
        a `not_eligible` re-rating candidate receives core proceeds."""
        result = self.two_lane_run(wm, queue)

        assert result["outcomes"][-1]["verdict"] == "not_eligible"
        assert result["routing"]["proposal"]["ticker"] == "ZENSAR"
