"""The three surfaces an owner actually touches: `exit`, `queue`, `queue route`.

Everything here runs against redirected stores. The real `watchlist.json` holds
live positions and the real `reinvestment_queue.json` records real sales; no
test may write either.

Three properties are worth stating before the cases, because each is a rule
about *display* that carries a decision behind it.

**Only `Current` renders the proposal.** `Partial` and `Stale` keep every
diagnostic they have — the blocked list, the idle readings, the errored tickers
— and print the refresh instruction where the candidate would go. A ranking
built on a run that did not finish, or on inputs that have since moved, is a
recommendation its own evidence no longer backs, and rendering it anyway is how
an owner acts on a name the system would no longer choose.

**`watchlist queue` is a pure read.** It never calls `advance()` and never
constructs a service. A display command that re-scored the corpus would mutate
lifecycle state as a side effect of being looked at, and would cost minutes to
answer a question about a stored snapshot.

**`queue route` records a deployment, not an intention.** It validates against
the *live* watchlist rather than the snapshot, refuses a candidate that never
actually received capital after the exit, and refuses to guess between two
deployment dates. The idle reading closes at `deployed_at`, so entering a route
late does not inflate the window it closes.
"""

import json
from datetime import date, datetime, timedelta

import pytest
from typer.testing import CliRunner

from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
from boundless100x.watchlist import WatchlistManager
from tests.test_confirm_exit import REVIEW_REASON, REVIEW_TRIGGER, priced_service
from tests.test_reinvestment_queue import friction_payload

# The exit is dated well before every deployment fixture below, so "on or after
# the exit" is a property of the dates in the test rather than of the day it
# runs.
EXIT_AT = "2026-08-01"
EXIT_ID = f"SOLD:{EXIT_AT}"
PROBE_AT = "2026-08-04T09:15:00"
SCALE_AT = "2026-08-06T11:00:00"


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Both durable stores, redirected, plus a console wide enough to read.

    The width matters: rich wraps to the terminal width, and a wrapped line
    would make a substring assertion fail for a reason that has nothing to do
    with what was printed.
    """
    from rich.console import Console

    from boundless100x import cli
    from boundless100x.lifecycle import reinvestment as reinvestment_module
    from boundless100x import watchlist as watchlist_module

    watchlist_path = tmp_path / "watchlist.json"
    queue_path = tmp_path / "reinvestment_queue.json"
    monkeypatch.setattr(watchlist_module, "DEFAULT_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(reinvestment_module, "DEFAULT_QUEUE_PATH", queue_path)
    monkeypatch.setattr(cli, "console", Console(width=200))

    class Stores:
        watchlist = watchlist_path
        queue = queue_path

        def wm(self):
            return WatchlistManager(path=str(watchlist_path))

        def q(self):
            return ReinvestmentQueue(path=str(queue_path))

    return Stores()


@pytest.fixture
def run():
    from boundless100x.cli import app

    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


@pytest.fixture
def no_pipeline(monkeypatch):
    """Make re-scoring impossible, so a pure-read surface proves it is one."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a display command re-scored the corpus")

    monkeypatch.setattr("boundless100x.service.Boundless100xService", forbidden)
    monkeypatch.setattr("boundless100x.lifecycle.advance.advance", forbidden)


# ── fixtures on disk ────────────────────────────────────────────────────────


def sold(stores, ticker="SOLD", lane="core", at=EXIT_AT, complete=True) -> dict:
    """A recorded exit, optionally left in KTD10's crash window.

    A complete one carries the `confirmed` stamp as well as the transition,
    which is what `confirm_exit` writes and what makes the proceeds routable.
    The incomplete one stops after the queue event, exactly where a crash
    between the first two writes leaves it.
    """
    wm = stores.wm()
    wm.add(ticker, lane=lane)
    wm.transition(ticker, "probe", "seed", applied_by="owner")
    wm.transition(ticker, "exit_review", REVIEW_TRIGGER, evidence=REVIEW_REASON)
    transition = (
        wm.transition(ticker, "exited", REVIEW_TRIGGER, applied_by="owner")
        if complete else None
    )

    queue = stores.q()
    event = queue.record_exit(
        ticker=ticker, lane=lane, trigger_id=REVIEW_TRIGGER,
        friction=friction_payload(), at=at, exit_id=f"{ticker}:{at}",
    )
    if transition is not None:
        queue.record_confirmation(event["exit_id"], at=transition["at"])
    return event


def deployed(stores, ticker="ZENSAR", at=PROBE_AT, to="probe",
             applied_by="owner", lane="rerating") -> None:
    """A candidate holding a deployment transition at an exact timestamp.

    Staged onto the store rather than written through `transition`, which
    stamps the wall clock: the whole subject here is *which date* closes an
    idle reading, and a test that could not choose the date could not say.
    """
    wm = stores.wm()
    if wm.get(ticker) is None:
        wm.add(ticker, lane=lane)
    staged = wm._stage()
    entry = staged["companies"][ticker]
    entry["state_history"].append({
        "at": at,
        "from": entry["state"],
        "to": to,
        "trigger_id": "fast_lane_buy_zone",
        "evidence": "all six lane gates pass",
        "applied_by": applied_by,
    })
    entry["state"] = to
    wm._commit(staged)


def snapshot(stores, **overrides) -> dict:
    """Write a routing snapshot describing the stores as they currently stand."""
    wm, queue = stores.wm(), stores.q()
    payload = {
        "as_of": "2026-08-07",
        "generated_at": datetime.now().isoformat(),
        "status": "current",
        "watchlist_revision": wm.data["revision"],
        "queue_revision": queue.data["revision"],
        "proposal": {
            "ticker": "ZENSAR", "lane": "rerating", "state": "watch",
            "composite": 6.5, "sector": None, "entry_trigger_fired": True,
            "trigger_id": "fast_lane_buy_zone",
            "evidence": "all six lane gates pass",
        },
        "reason": "",
        "blocked": [],
        "idle": [],
        "incomplete": [],
        "errors": [],
    }
    payload.update(overrides)
    return queue.write_proposal(payload)


# ── watchlist exit ──────────────────────────────────────────────────────────


class TestExitCommand:
    """It moves money, so it is a command of its own and never a flag."""

    @pytest.fixture
    def reviewed(self, stores, monkeypatch):
        wm = stores.wm()
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "probe", "valuation_buy_zone", applied_by="owner")
        wm.transition("ASTRAL", "exit_review", REVIEW_TRIGGER, evidence=REVIEW_REASON)
        entered = date.fromisoformat(wm.get("ASTRAL")["state_history"][0]["at"][:10])

        monkeypatch.setattr(
            "boundless100x.service.Boundless100xService",
            lambda *a, **k: priced_service(entered),
        )
        return entered

    def test_it_records_the_exit_and_states_the_transition(self, run, stores, reviewed):
        result = run("watchlist", "exit", "ASTRAL",
                     "--as-of", str(reviewed + timedelta(days=400)))

        assert result.exit_code == 0
        assert "exit_review" in result.output and "exited" in result.output
        assert stores.wm().get("ASTRAL")["state"] == "exited"
        assert len(stores.q().exits()) == 1

    def test_the_output_carries_the_date_trigger_friction_and_exit_id(
        self, run, stores, reviewed
    ):
        """Everything the owner needs to reconcile the sale against a broker
        statement, including the id a retry would recompute."""
        as_of = reviewed + timedelta(days=400)
        result = run("watchlist", "exit", "ASTRAL", "--as-of", str(as_of))

        exit_id = stores.q().exits()[0]["exit_id"]
        assert str(as_of) in result.output
        assert REVIEW_TRIGGER in result.output
        assert "gross" in result.output and "net" in result.output
        assert "LTCG" in result.output
        assert exit_id in result.output

    def test_an_unavailable_friction_reading_says_why(self, run, stores, monkeypatch):
        from tests.test_lifecycle_advance import StubService

        wm = stores.wm()
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "exit_review", REVIEW_TRIGGER, evidence=REVIEW_REASON)
        monkeypatch.setattr(
            "boundless100x.service.Boundless100xService",
            lambda *a, **k: StubService(data={}),
        )

        result = run("watchlist", "exit", "ASTRAL", "--as-of", "2026-08-07")

        assert result.exit_code == 0
        assert "unavailable" in result.output.lower()
        assert stores.wm().get("ASTRAL")["state"] == "exited"

    def test_a_store_write_failure_is_a_message_and_a_recovery_instruction(
        self, run, stores, reviewed, monkeypatch
    ):
        """`confirm_exit` lets an exception escape step 3 on purpose — the queue
        event is already durable, so re-running reconciles.

        That design only helps if the owner is told to re-run. A traceback at
        the moment two stores disagree is the one moment they need a single
        fact and a single command, so the surface catches what the operation
        deliberately does not.
        """
        from boundless100x.watchlist import WatchlistManager

        def disk_full(*args, **kwargs):
            raise OSError("[Errno 28] No space left on device")

        monkeypatch.setattr(WatchlistManager, "transition", disk_full)

        result = run("watchlist", "exit", "ASTRAL",
                     "--as-of", str(reviewed + timedelta(days=400)))

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "No space left on device" in result.output
        assert "watchlist exit ASTRAL" in result.output
        assert "watchlist queue" in result.output
        # The queue event landed before the transition was attempted; that is
        # the state the instruction reconciles from.
        assert len(stores.q().exits()) == 1

    def test_a_refusal_names_the_state_and_says_nothing_was_recorded(
        self, run, stores, monkeypatch
    ):
        from tests.test_lifecycle_advance import StubService

        wm = stores.wm()
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "scale", "seed", applied_by="owner")
        monkeypatch.setattr(
            "boundless100x.service.Boundless100xService",
            lambda *a, **k: StubService(),
        )

        result = run("watchlist", "exit", "ASTRAL")

        assert result.exit_code == 1
        assert "scale" in result.output
        assert "nothing was recorded" in result.output.lower()
        assert stores.q().events() == []
        assert stores.wm().get("ASTRAL")["state"] == "scale"


# ── watchlist queue ─────────────────────────────────────────────────────────


class TestQueueDisplay:
    def test_it_never_re_scores_or_advances(self, run, stores, no_pipeline):
        sold(stores)
        snapshot(stores)

        result = run("watchlist", "queue")

        assert result.exit_code == 0

    def test_no_snapshot_reads_unavailable(self, run, stores, no_pipeline):
        result = run("watchlist", "queue")

        assert result.exit_code == 0
        assert "Unavailable" in result.output
        assert "ZENSAR" not in result.output

    def test_a_current_snapshot_renders_the_proposal_with_its_evidence(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        snapshot(stores, idle=[{"exit_id": EXIT_ID, "ticker": "SOLD",
                                "idle_days": 6, "closed": False}])

        result = run("watchlist", "queue")

        assert "Current" in result.output
        assert "ZENSAR" in result.output
        assert "all six lane gates pass" in result.output

    def test_a_partial_snapshot_names_its_errored_tickers_and_withholds_the_proposal(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        snapshot(stores, status="partial", errors=["SPLPETRO"])

        result = run("watchlist", "queue")

        assert "Partial" in result.output
        assert "SPLPETRO" in result.output
        assert "watchlist advance" in result.output
        assert "ZENSAR" not in result.output

    def test_partial_wins_even_when_the_revisions_have_also_moved(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        snapshot(stores, status="partial", errors=["SPLPETRO"])
        stores.wm().add("LATER")

        result = run("watchlist", "queue")

        assert "Partial" in result.output
        assert "Stale" not in result.output

    @pytest.mark.parametrize("mutate", [
        lambda stores: stores.wm().add("LATER"),
        lambda stores: stores.wm().record_catalyst(
            "ZENSAR", "Demerger of the services arm", "2026-12-31"
        ),
        lambda stores: stores.q().record_routing(
            exit_id=EXIT_ID, candidate="ZENSAR", deployed_at=PROBE_AT
        ),
    ], ids=["add", "catalyst-edit", "recorded-route"])
    def test_any_post_snapshot_mutation_reads_stale(
        self, run, stores, no_pipeline, mutate
    ):
        """Freshness is revisions, not clocks — so a catalyst edit invalidates a
        snapshot exactly as a re-score would."""
        sold(stores)
        stores.wm().add("ZENSAR", lane="rerating")
        snapshot(stores)

        mutate(stores)
        result = run("watchlist", "queue")

        assert "Stale" in result.output
        assert "watchlist advance" in result.output
        # The candidate name may still appear as a diagnostic elsewhere; what
        # must not appear is the proposal block that recommends it.
        assert "Proposed destination" not in result.output

    def test_stale_keeps_its_diagnostics(self, run, stores, no_pipeline):
        sold(stores)
        snapshot(stores, blocked=[{
            "ticker": "ASTRAL", "lane": "core", "state": "watch",
            "reasons": ["the 100x eligibility gates were not passed"],
        }])
        stores.wm().add("LATER")

        result = run("watchlist", "queue")

        assert "Stale" in result.output
        assert "ASTRAL" in result.output
        assert "100x eligibility gates" in result.output

    def test_blocked_candidates_render_with_their_reasons(
        self, run, stores, no_pipeline
    ):
        """An all-blocked run must not read like an empty pipeline."""
        sold(stores)
        snapshot(stores, proposal=None, reason="every candidate was blocked (1)",
                 blocked=[{
                     "ticker": "ASTRAL", "lane": "core", "state": "watch",
                     "reasons": ["the 100x eligibility gates were not passed"],
                 }])

        result = run("watchlist", "queue")

        assert "ASTRAL" in result.output
        assert "100x eligibility gates" in result.output
        assert "blocked" in result.output.lower()

    def test_a_genuinely_empty_queue_says_so_instead(self, run, stores, no_pipeline):
        snapshot(stores, proposal=None, reason="No exit proceeds awaiting routing")

        result = run("watchlist", "queue")

        assert "No exit proceeds awaiting routing" in result.output
        assert "blocked" not in result.output.lower()

    def test_an_unrouted_exit_shows_its_idle_days(self, run, stores, no_pipeline):
        sold(stores)
        snapshot(stores)

        result = run("watchlist", "queue")

        assert "SOLD" in result.output
        assert EXIT_ID in result.output
        assert "idle" in result.output.lower()

    def test_an_exit_stranded_in_exit_review_shows_the_recovery_command(
        self, run, stores, no_pipeline
    ):
        sold(stores, ticker="HALFSOLD", complete=False)
        snapshot(stores)

        result = run("watchlist", "queue")

        assert "Exit recording incomplete" in result.output
        assert "watchlist exit HALFSOLD" in result.output
        # Nothing here is routable, and the display says so — but it must not
        # say the queue is *empty*. An unfinished record is capital the owner
        # has and cannot yet reach, and "No exit proceeds awaiting routing"
        # over the top of it is a false all-clear: it reads as nothing to do,
        # on the one screen whose job is to say what is outstanding.
        assert "No exit proceeds awaiting routing" not in result.output
        assert "not confirmed" in result.output


class TestAdvanceRoutingLine:
    """What `watchlist advance` says about routing, captured the way
    `_print_concentration` is tested.

    The rule under test is that a candidate is named only when the view was
    durably stored: a `--quarterly` run ranked a subset, and a full run whose
    write failed has nothing behind the name either. Naming one anyway would
    have the advance output and tomorrow's `watchlist queue` disagree.
    """

    def render(self, routing) -> str:
        from boundless100x import cli

        with cli.console.capture() as captured:
            cli._print_routing_result(routing)
        return captured.get()

    def stored(self, **overrides) -> dict:
        """A routing view exactly as `advance._routing` returns one.

        `status` and `errors` are carried because the real payload always
        carries them — the snapshot dict is spread into every branch of
        `_routing` — and because they are what decides whether a candidate may
        be named at all.
        """
        payload = {
            "available": True, "persisted": True, "persist_reason": "",
            "status": "current", "errors": [],
            "proposal": {"ticker": "ZENSAR", "lane": "rerating"},
            "reason": "", "blocked": [],
        }
        payload.update(overrides)
        return payload

    def test_a_stored_view_names_its_candidate(self, stores):
        assert "ZENSAR" in self.render(self.stored())

    def test_an_unpersisted_view_names_nobody_and_says_why(self, stores):
        text = self.render(self.stored(
            persisted=False, persist_reason="a --quarterly run advances a subset"
        ))

        assert "ZENSAR" not in text
        assert "quarterly" in text

    def test_an_unavailable_view_states_its_reason(self, stores):
        text = self.render({"available": False, "persisted": False,
                            "reason": "no reinvestment queue was supplied"})

        assert "unavailable" in text.lower()
        assert "no reinvestment queue was supplied" in text

    def test_a_stored_view_with_no_candidate_states_the_reason(self, stores):
        text = self.render(self.stored(
            proposal=None, reason="No exit proceeds awaiting routing"
        ))

        assert "No exit proceeds awaiting routing" in text

    def test_blocked_candidates_are_counted_and_pointed_at(self, stores):
        text = self.render(self.stored(blocked=[{"ticker": "ASTRAL"}]))

        assert "1 candidate(s) blocked" in text
        assert "ASTRAL" in text
        assert "watchlist queue" in text

    def test_a_partial_run_names_nobody_and_says_which_ticker_failed(self, stores):
        """The withheld side is the fail-closed one, and both surfaces take it.

        `snapshot_state` refuses to render a `partial` snapshot's proposal, so
        a line printed seconds earlier from the same stored bytes must refuse
        it too — otherwise `advance` names a candidate that `watchlist queue`
        will not, which is the disagreement this whole helper exists to avoid.
        """
        text = self.render(self.stored(status="partial", errors=["SPLPETRO"]))

        assert "ZENSAR" not in text
        assert "SPLPETRO" in text
        assert "watchlist advance" in text

    def test_errored_tickers_withhold_the_candidate_whatever_the_status_says(
        self, stores
    ):
        """`snapshot_state` reads `partial` on a non-empty `errors` list alone;
        so does this, or the two disagree on a hand-edited snapshot."""
        text = self.render(self.stored(errors=["SPLPETRO"]))

        assert "ZENSAR" not in text
        assert "SPLPETRO" in text

    def test_a_view_carrying_no_status_names_nobody(self, stores):
        """Fail closed: a payload that cannot prove the run finished is not one
        to name a destination for capital from."""
        payload = self.stored()
        payload.pop("status")

        assert "ZENSAR" not in self.render(payload)

    def test_a_partial_run_still_counts_its_blocked_candidates(self, stores):
        """The blocked list is a true statement about what the run saw, and
        stays true when the ranking built on it may not be shown."""
        text = self.render(self.stored(
            status="partial", errors=["SPLPETRO"], blocked=[{"ticker": "ASTRAL"}]
        ))

        assert "1 candidate(s) blocked" in text
        assert "ASTRAL" in text


class TestAdvanceWhenTheQueueCannotBeRead:
    """A fault in the *routing* store must not stop companies being re-scored.

    `advance._routing` is built to degrade — `queue=None` reports routing
    unavailable-with-reason and the run continues. Constructing the queue
    inline as an argument defeated that: `_load` raises on unreadable JSON
    before `advance()` is ever entered, so a corrupt routing file took the
    kill-switches down with it.
    """

    def test_the_run_still_advances_and_reports_routing_unavailable(
        self, run, stores, monkeypatch
    ):
        from tests.test_lifecycle_advance import StubService

        stores.wm().add("ASTRAL")
        stores.queue.write_text('{"events": [ this is not json')
        monkeypatch.setattr(
            "boundless100x.service.Boundless100xService",
            lambda *a, **k: StubService(),
        )

        result = run("watchlist", "advance")

        assert result.exit_code == 0
        assert "ASTRAL" in result.output
        assert "routing" in result.output.lower()
        assert "unavailable" in result.output.lower()

    def test_the_watchlist_still_records_what_the_run_decided(
        self, run, stores, monkeypatch
    ):
        """The point of degrading rather than aborting: the lifecycle work of
        the run survives a fault in a store it only reports from."""
        from tests.test_lifecycle_advance import StubService

        wm = stores.wm()
        wm.add("ASTRAL")
        stores.queue.write_text("{")
        monkeypatch.setattr(
            "boundless100x.service.Boundless100xService",
            lambda *a, **k: StubService(),
        )

        run("watchlist", "advance")

        assert stores.wm().get("ASTRAL")["last_score_snapshot"] is not None


# ── watchlist queue route ───────────────────────────────────────────────────


class TestQueueRoute:
    def test_it_records_the_deployment_and_closes_the_idle_reading(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        deployed(stores)

        result = run("watchlist", "queue", "route", EXIT_ID, "zensar")

        assert result.exit_code == 0
        queue = stores.q()
        routing = queue.routing_for(EXIT_ID)
        assert routing["candidate"] == "ZENSAR"
        assert routing["deployed_at"] == PROBE_AT
        assert queue.unrouted_exits() == []

    def test_the_idle_reading_closes_at_deployed_at_not_recorded_at(
        self, run, stores, no_pipeline
    ):
        """A route entered weeks late must not read as weeks of idle capital."""
        sold(stores)
        deployed(stores)

        run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        queue = stores.q()
        routing = queue.routing_for(EXIT_ID)
        view = queue.exit_views(stores.wm(), as_of=date(2026, 9, 30))[0]

        assert routing["recorded_at"] != routing["deployed_at"]
        assert view["idle_days"] == 3      # 2026-08-01 → 2026-08-04
        assert view["closed"] is True

    def test_zero_proceeds_is_reported_before_any_argument_is_judged(
        self, run, stores, no_pipeline
    ):
        """The emptiness is the answer; validating a nonexistent exit's id first
        would report the wrong problem."""
        result = run("watchlist", "queue", "route", "NOPE:2020-01-01", "ZENSAR")

        assert result.exit_code == 1
        assert "No exit proceeds awaiting routing" in result.output
        assert "NOPE" not in result.output

    def test_an_unknown_exit_id_is_refused(self, run, stores, no_pipeline):
        sold(stores)
        deployed(stores)

        result = run("watchlist", "queue", "route", "GHOST:2026-01-01", "ZENSAR")

        assert result.exit_code == 1
        assert "GHOST:2026-01-01" in result.output
        assert stores.q().routings() == []

    def test_an_already_routed_exit_is_refused(self, run, stores, no_pipeline):
        sold(stores)
        sold(stores, ticker="ALSOSOLD", at="2026-07-01")
        deployed(stores)
        run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert result.exit_code == 1
        assert "already routed" in result.output.lower()
        assert len(stores.q().routings()) == 1

    def test_an_exit_still_in_exit_review_is_refused_by_the_command(
        self, run, stores, no_pipeline
    ):
        """The display excludes it; the direct command must not be a way round
        that exclusion."""
        sold(stores, ticker="HALFSOLD", complete=False)
        sold(stores, ticker="SOLD")
        deployed(stores)

        result = run("watchlist", "queue", "route", f"HALFSOLD:{EXIT_AT}", "ZENSAR")

        assert result.exit_code == 1
        assert "exit_review" in result.output
        assert "watchlist exit HALFSOLD" in result.output
        assert stores.q().routings() == []

    def test_a_candidate_that_never_deployed_is_refused(
        self, run, stores, no_pipeline
    ):
        """A plan that never executed must not close an idle reading measuring
        exit-to-deployed-capital."""
        sold(stores)
        stores.wm().add("ZENSAR", lane="rerating")
        stores.wm().transition("ZENSAR", "watch", "seed")

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert result.exit_code == 1
        assert "ZENSAR" in result.output
        assert stores.q().routings() == []

    def test_a_deployment_predating_the_exit_does_not_count(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        deployed(stores, at="2026-07-20T09:15:00")

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert result.exit_code == 1
        assert stores.q().routings() == []

    def test_an_auto_applied_transition_does_not_count(self, run, stores, no_pipeline):
        sold(stores)
        deployed(stores, applied_by="auto")

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert result.exit_code == 1
        assert stores.q().routings() == []

    def test_two_eligible_transitions_are_refused_with_both_timestamps(
        self, run, stores, no_pipeline
    ):
        """`deployed_at` is a recorded fact, and a guess between two dates
        fabricates it."""
        sold(stores)
        deployed(stores, at=PROBE_AT, to="probe")
        deployed(stores, at=SCALE_AT, to="scale")

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert result.exit_code == 1
        assert PROBE_AT in result.output
        assert SCALE_AT in result.output
        assert "--transition-at" in result.output
        assert stores.q().routings() == []

    def test_transition_at_selects_one_of_them(self, run, stores, no_pipeline):
        sold(stores)
        deployed(stores, at=PROBE_AT, to="probe")
        deployed(stores, at=SCALE_AT, to="scale")

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR",
                     "--transition-at", SCALE_AT)

        assert result.exit_code == 0
        assert stores.q().routing_for(EXIT_ID)["deployed_at"] == SCALE_AT

    def test_transition_at_naming_no_eligible_transition_is_refused(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        deployed(stores, at=PROBE_AT)

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR",
                     "--transition-at", "2026-08-05T00:00:00")

        assert result.exit_code == 1
        assert stores.q().routings() == []

    def test_the_selector_is_optional_when_exactly_one_is_eligible(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        deployed(stores, at=PROBE_AT)

        result = run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert result.exit_code == 0
        assert stores.q().routing_for(EXIT_ID)["deployed_at"] == PROBE_AT

    def test_the_candidate_need_not_be_the_proposed_one(
        self, run, stores, no_pipeline
    ):
        """The proposal advises; the owner may deploy elsewhere, and the event
        records what actually happened."""
        sold(stores)
        snapshot(stores)          # proposes ZENSAR
        deployed(stores, ticker="ASTRAL", lane="core", at=PROBE_AT)

        result = run("watchlist", "queue", "route", EXIT_ID, "ASTRAL")

        assert result.exit_code == 0
        assert stores.q().routing_for(EXIT_ID)["candidate"] == "ASTRAL"

    def test_the_route_is_visible_in_the_queue_display(
        self, run, stores, no_pipeline
    ):
        sold(stores)
        deployed(stores)
        run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        result = run("watchlist", "queue")

        assert "ZENSAR" in result.output
        assert "No exit proceeds awaiting routing" in result.output

    def test_the_event_log_is_the_record_not_the_snapshot(
        self, run, stores, no_pipeline
    ):
        """Routing writes an event; the stale snapshot beside it is untouched."""
        sold(stores)
        stored = snapshot(stores)
        deployed(stores)

        run("watchlist", "queue", "route", EXIT_ID, "ZENSAR")

        assert json.dumps(stores.q().latest_proposal(), sort_keys=True) == \
            json.dumps(stored, sort_keys=True)


class TestRemoveGuardsRecordedProceeds:
    """`watchlist remove`, refused while the company holds an unconfirmed exit.

    The one state whose repair genuinely needs the lifecycle record:
    `watchlist exit` keys on the entry's `exit_review` transition and completes
    from its history, so deleting the entry underneath an unconfirmed exit
    strands the proceeds with no command able to reach them — and the queue
    then reports nothing outstanding, which is the point at which the money
    stops being looked for.

    A *confirmed* exit is deliberately not guarded. Its completion is stamped
    on the queue event, so it survives the removal and stays routable; guarding
    it too would mean a company could never leave the watchlist until its
    proceeds were redeployed, which is a rule about bookkeeping masquerading as
    a rule about capital.
    """

    def test_an_unconfirmed_exit_refuses_the_removal(self, run, stores):
        sold(stores, ticker="HALFSOLD", complete=False)

        result = run("watchlist", "remove", "HALFSOLD")

        assert result.exit_code == 1
        assert "not yet confirmed" in result.output
        assert "watchlist exit HALFSOLD" in result.output
        assert "Nothing was removed" in result.output

    def test_the_refusal_leaves_the_entry_exactly_where_it_was(self, run, stores):
        sold(stores, ticker="HALFSOLD", complete=False)

        run("watchlist", "remove", "HALFSOLD")

        entry = stores.wm().get("HALFSOLD")
        assert entry is not None
        assert entry["state"] == "exit_review"

    def test_completing_the_exit_then_allows_the_removal(self, run, stores):
        """The refusal has to be a step, not a wall — and this is the step."""
        sold(stores, ticker="SOLD", complete=False)
        wm = stores.wm()
        transition = wm.transition("SOLD", "exited", REVIEW_TRIGGER,
                                   applied_by="owner")
        stores.q().record_confirmation(EXIT_ID, at=transition["at"])

        result = run("watchlist", "remove", "SOLD")

        assert result.exit_code == 0
        assert stores.wm().get("SOLD") is None

    def test_the_removed_company_s_confirmed_proceeds_stay_routable(
        self, run, stores, no_pipeline
    ):
        """The whole reason the guard can stop at *unconfirmed*."""
        sold(stores, ticker="SOLD")
        run("watchlist", "remove", "SOLD")

        result = run("watchlist", "queue")

        assert EXIT_ID in result.output
        assert "No exit proceeds awaiting routing" not in result.output
        assert [v["exit_id"] for v in stores.q().routable_exits(stores.wm())] \
            == [EXIT_ID]

    def test_a_company_with_no_exit_at_all_is_removed_normally(self, run, stores):
        stores.wm().add("QUIET")

        result = run("watchlist", "remove", "QUIET")

        assert result.exit_code == 0
        assert stores.wm().get("QUIET") is None

    def test_an_unreadable_queue_refuses_rather_than_assuming_safety(
        self, run, stores
    ):
        """Fail closed on the one path that cannot be undone: "could not check"
        must not resolve to "go ahead"."""
        stores.wm().add("QUIET")
        stores.queue.write_text('{"events": [{"kind": "nonsense"}]}')

        result = run("watchlist", "remove", "QUIET")

        assert result.exit_code == 1
        assert "could not be read" in result.output
        assert "Nothing was removed" in result.output
        assert stores.wm().get("QUIET") is not None


class TestCappedTransitionDisplay:
    """What an owner sees when a concentration cap holds a transition back.

    Its own block rather than a cell in the table, for `_print_exit_friction`'s
    reason: the cap has to travel with the count it breaches and the basis that
    count is in, and the evidence column truncates at 54 characters — "the core
    lane already holds 1 of a maxi…" reads as a system that refused without
    saying why.

    The escape hatch is printed too. A guardrail whose only visible face is a
    refusal invites being worked around by quietly raising the cap in the
    config, which is the version of the override that leaves no record.
    """

    def render(self, outcomes) -> str:
        from boundless100x import cli

        with cli.console.capture() as captured:
            cli._print_capped_transitions(outcomes)
        return captured.get()

    def withheld(self, ticker="ASTRAL", **overrides) -> dict:
        proposal = {
            "to": "probe",
            "concentration_withheld": True,
            "concentration_reasons": [
                "the core lane already holds 8 of a maximum 8 positioned "
                "name(s) — one more would breach the cap (counts of names, "
                "not a share of capital)"
            ],
        }
        proposal.update(overrides)
        return {"ticker": ticker, "state": "watch", "proposal": proposal}

    def test_it_names_the_company_the_move_and_the_cap(self, stores):
        text = self.render([self.withheld()])

        assert "ASTRAL" in text
        assert "watch → probe" in text
        assert "8 of a maximum 8" in text

    def test_the_basis_survives_into_the_rendered_line(self, stores):
        """Counts of names, never a share of capital — the sentence this whole
        module exists to keep attached to its numbers."""
        assert "not a share of capital" in self.render([self.withheld()])

    def test_it_offers_the_three_ways_forward(self, stores):
        text = self.render([self.withheld()])

        assert "--override-caps" in text
        assert "config.yaml" in text
        assert "Exit or drop a name" in text

    def test_an_uncapped_run_prints_nothing_at_all(self, stores):
        """Silence is the ordinary case, and a heading over an empty list would
        make every clean run look like it had something to answer for."""
        clear = {"ticker": "ASTRAL", "state": "watch",
                 "proposal": {"to": "probe", "concentration_withheld": False}}

        assert self.render([clear, {"ticker": "X", "proposal": None}]) == ""
