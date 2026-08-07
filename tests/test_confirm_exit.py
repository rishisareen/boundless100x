"""`confirm_exit` — the only path to `exited` (KTD10, R3a).

No metric can observe that the owner sold. A trigger firing on price or
fundamentals would record a sale that may never have happened, which is exactly
the automated execution §13 forbids — so `exited` is reached by an explicit
owner command and by nothing else. The last test in this file is that
invariant, asserted from both ends: no declared trigger names `exited`, and a
full `advance()` over a fixture in every state proposes `exit_review` at most.

The rest is KTD10's validate-then-write protocol. Two JSON files cannot be
written atomically, so the failure window between them is made *recoverable*
rather than pretended away: the queue event lands first, keyed by an `exit_id`
a retry recomputes identically, and re-running the command finds that event,
skips the duplicate append, and completes the transition. Reconciliation is
"run it again".

The reverse order would be unrecoverable by construction — transition first
plus a crash leaves an exited position with no queue event, and the state check
would then refuse the very retry that could repair it. The crash-recovery test
here interrupts the real ordering rather than hand-writing the intermediate
state, so it fails if the two writes are ever swapped.

Two further rules are tested because both are easy to get backwards. A friction
reading that cannot be computed does **not** abort the exit: the owner's sale
is a fact, and refusing to record reality over a data gap leaves the books
wrong, which is worse. And the payload written to the two stores must be the
same object of record — asserted together, because the agreement *is* the
contract.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from boundless100x import score_history
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.advance import advance
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.lifecycle.exit import confirm_exit
from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
from boundless100x.watchlist import WatchlistManager
from tests.test_friction import price_frame
from tests.test_lifecycle_advance import StubService

# The reason the exit review was recorded under — carried into the exit's own
# evidence, so the record says why the position was being reviewed at all.
REVIEW_TRIGGER = "roiic_below_cost_of_capital"
REVIEW_REASON = "RoIIC 3.0% has been below the 12% cost of capital for two years"


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


@pytest.fixture
def queue(tmp_path):
    return ReinvestmentQueue(path=str(tmp_path / "reinvestment_queue.json"))


def reviewed_position(wm, ticker="ASTRAL", lane="core", probed=True) -> date:
    """A holding that entered `probe` and is now under `exit_review`.

    Returns the date the probe was confirmed. Built through `transition` rather
    than written to disk, because `watchlist._now()` stamps the wall clock and
    a hand-written history would date the holding period from a day the price
    fixture has never seen.
    """
    wm.add(ticker, lane=lane)
    if probed:
        wm.transition(ticker, "probe", "valuation_buy_zone",
                      evidence="P/E at the 22nd percentile of its own history")
    wm.transition(ticker, "exit_review", REVIEW_TRIGGER, evidence=REVIEW_REASON)
    return date.fromisoformat(wm.get(ticker)["state_history"][0]["at"][:10])


def stepped_price(entered: date, days: int = 900) -> pd.DataFrame:
    """100.00 for a week, then 150.00 to day 500, then 300.00.

    The second step is what gives the crash-recovery test teeth: a re-price on
    the later run would read a different *gross* return, not merely a longer
    holding period, so adopting the stored payload is visibly distinguishable
    from recomputing it.
    """
    dates = pd.date_range(entered, periods=days, freq="D")
    closes = [100.0] * 7 + [150.0] * (500 - 7) + [300.0] * (days - 500)
    return price_frame(dates, closes)


def priced_service(entered: date) -> StubService:
    return StubService(data={"price": stepped_price(entered)})


class TestTheRecordedExit:
    def test_it_records_the_transition_and_exactly_one_queue_event(self, wm, queue):
        entered = reviewed_position(wm)
        as_of = entered + timedelta(days=400)

        outcome = confirm_exit(wm, queue, "ASTRAL", priced_service(entered), as_of)

        assert outcome["ok"] is True
        assert wm.get("ASTRAL")["state"] == "exited"
        record = wm.get("ASTRAL")["state_history"][-1]
        assert (record["from"], record["to"]) == ("exit_review", "exited")
        assert len(queue.exits()) == 1

    def test_both_stores_carry_the_same_full_friction_payload(self, wm, queue):
        """Asserted together, because the agreement is the contract — one
        store's figure disagreeing with the other's about the same sale is the
        failure this whole protocol exists to prevent."""
        entered = reviewed_position(wm)
        as_of = entered + timedelta(days=400)

        confirm_exit(wm, queue, "ASTRAL", priced_service(entered), as_of)

        details = wm.get("ASTRAL")["state_history"][-1]["details"]
        stored = queue.exits()[0]["friction"]

        assert details == stored
        # The full structured payload, not a bare net figure: gross, holding
        # period, tax regime, net and basis all read back apart.
        #   gross    = (150.00 / 100.00 - 1) * 100                    = 50.0%
        #   slippage = 100bps round trip                              = 1.00pp
        #   400 days >= the 365-day line, so LTCG at 12.5%:
        #   net      = 49.0 * (1 - 0.125)                             = 42.875%
        assert details["gross_return_pct"] == pytest.approx(50.0)
        assert details["holding_days"] == 400
        assert details["tax_regime"] == "ltcg"
        assert details["net_return_pct"] == pytest.approx(42.875)
        assert details["basis"] == "recorded"

    def test_the_transition_cites_the_exit_review_trigger_and_the_owner(self, wm, queue):
        entered = reviewed_position(wm)

        confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                     entered + timedelta(days=400))

        record = wm.get("ASTRAL")["state_history"][-1]
        assert record["trigger_id"] == REVIEW_TRIGGER
        assert record["applied_by"] == "owner"
        assert queue.exits()[0]["trigger_id"] == REVIEW_TRIGGER

    def test_the_evidence_carries_the_recorded_reason_and_the_net_figure(self, wm, queue):
        entered = reviewed_position(wm)

        confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                     entered + timedelta(days=400))

        evidence = wm.get("ASTRAL")["state_history"][-1]["evidence"]
        assert REVIEW_REASON in evidence
        assert "42.9" in evidence          # the net figure, as `describe` renders it
        assert "realiz" not in evidence.lower()   # KTD7: never a realized return

    def test_the_exit_id_names_the_ticker_and_the_review_timestamp(self, wm, queue):
        entered = reviewed_position(wm)
        review_at = wm.get("ASTRAL")["state_history"][-1]["at"]

        outcome = confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                               entered + timedelta(days=400))

        assert outcome["exit_id"] == f"ASTRAL:{review_at}"
        assert queue.exits()[0]["exit_id"] == outcome["exit_id"]

    def test_the_exit_event_records_the_lane(self, wm, queue):
        entered = reviewed_position(wm, ticker="ZENSAR", lane="rerating")

        confirm_exit(wm, queue, "ZENSAR", priced_service(entered),
                     entered + timedelta(days=400))

        assert queue.exits()[0]["lane"] == "rerating"

    def test_the_exit_event_dates_the_sale_not_the_bar(self, wm, queue):
        """`at` is the owner's exit date. The bar that priced it travels inside
        the friction payload, where it can be inspected without being mistaken
        for the date of the sale."""
        entered = reviewed_position(wm)
        as_of = entered + timedelta(days=400)

        confirm_exit(wm, queue, "ASTRAL", priced_service(entered), as_of)

        assert queue.exits()[0]["at"] == str(as_of)


class TestRefusals:
    @pytest.mark.parametrize("state", ["scale", "watch", "exited"])
    def test_any_state_other_than_exit_review_is_refused(self, wm, queue, state):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", state, "seed")

        outcome = confirm_exit(wm, queue, "ASTRAL", StubService(), date(2026, 8, 7))

        assert outcome["ok"] is False
        assert state in outcome["reason"]
        assert "nothing was recorded" in outcome["reason"].lower()

    @pytest.mark.parametrize("state", ["scale", "watch", "exited"])
    def test_a_refusal_records_nothing_in_either_store(self, wm, queue, state):
        wm.add("ASTRAL")
        wm.transition("ASTRAL", state, "seed")
        before = len(wm.get("ASTRAL")["state_history"])

        confirm_exit(wm, queue, "ASTRAL", StubService(), date(2026, 8, 7))

        assert len(wm.get("ASTRAL")["state_history"]) == before
        assert wm.get("ASTRAL")["state"] == state
        assert queue.events() == []

    def test_a_completed_exit_is_refused_and_appends_nothing(self, wm, queue):
        """The second run of a *successful* exit, as opposed to a retry of a
        crashed one: the state is now `exited`, so there is nothing to
        complete."""
        entered = reviewed_position(wm)
        service = priced_service(entered)
        confirm_exit(wm, queue, "ASTRAL", service, entered + timedelta(days=400))

        outcome = confirm_exit(wm, queue, "ASTRAL", service,
                               entered + timedelta(days=500))

        assert outcome["ok"] is False
        assert "exited" in outcome["reason"]
        assert len(queue.events()) == 1
        assert len(wm.get("ASTRAL")["state_history"]) == 3

    def test_an_untracked_ticker_is_refused(self, wm, queue):
        outcome = confirm_exit(wm, queue, "NOPE", StubService(), date(2026, 8, 7))

        assert outcome["ok"] is False
        assert "NOPE" in outcome["reason"]
        assert queue.events() == []

    def test_an_exit_review_with_no_transition_behind_it_is_refused(self, wm, queue):
        """A state written straight onto disk has no timestamp to key the
        `exit_id` from, so a retry could not recompute the same id — and
        without that, the append cannot be made idempotent. Refuse rather than
        write something a retry would duplicate."""
        wm.add("ASTRAL")
        staged = wm._stage()
        staged["companies"]["ASTRAL"]["state"] = "exit_review"
        wm._commit(staged)

        outcome = confirm_exit(wm, queue, "ASTRAL", StubService(), date(2026, 8, 7))

        assert outcome["ok"] is False
        assert "exit_review" in outcome["reason"]
        assert queue.events() == []


class TestCrashRecovery:
    """The window between the two writes, reproduced by interrupting it.

    The interruption is applied to `watchlist.transition` rather than to a
    hand-written store, so the test also asserts the *ordering*: if the
    transition were written first, nothing would reach the queue and the retry
    would meet a state check it could never pass.
    """

    def interrupted(self, wm, queue, entered, monkeypatch) -> str:
        def transition_that_never_lands(*args, **kwargs):
            raise RuntimeError("power cut between the two writes")

        monkeypatch.setattr(wm, "transition", transition_that_never_lands)
        with pytest.raises(RuntimeError):
            confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                         entered + timedelta(days=400))
        monkeypatch.undo()

        assert len(queue.exits()) == 1        # the queue event landed first
        assert wm.get("ASTRAL")["state"] == "exit_review"
        return queue.exits()[0]["exit_id"]

    def test_the_retry_recomputes_the_same_exit_id(self, wm, queue, monkeypatch):
        entered = reviewed_position(wm)
        exit_id = self.interrupted(wm, queue, entered, monkeypatch)

        outcome = confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                               entered + timedelta(days=700))

        assert outcome["ok"] is True
        assert outcome["exit_id"] == exit_id
        assert outcome["adopted"] is True

    def test_the_retry_adopts_the_stored_date_and_payload_rather_than_re_pricing(
        self, wm, queue, monkeypatch
    ):
        """A retry on a later day must complete the original exit, not re-price
        it — the two stores would otherwise disagree about the same sale. The
        price fixture steps again after day 500, so a recomputed reading would
        report a 200% gross return against the stored 50%."""
        entered = reviewed_position(wm)
        self.interrupted(wm, queue, entered, monkeypatch)
        stored = dict(queue.exits()[0])

        confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                     entered + timedelta(days=700))

        assert queue.exits()[0] == stored
        assert queue.exits()[0]["friction"]["gross_return_pct"] == pytest.approx(50.0)
        assert queue.exits()[0]["at"] == str(entered + timedelta(days=400))

    def test_two_runs_leave_one_event_one_transition_and_identical_payloads(
        self, wm, queue, monkeypatch
    ):
        entered = reviewed_position(wm)
        self.interrupted(wm, queue, entered, monkeypatch)

        confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                     entered + timedelta(days=700))

        assert len(queue.exits()) == 1
        exits = [r for r in wm.get("ASTRAL")["state_history"] if r["to"] == "exited"]
        assert len(exits) == 1
        assert exits[0]["details"] == queue.exits()[0]["friction"]
        assert wm.get("ASTRAL")["state"] == "exited"


class TestPricingReadsOneSource:
    """Confirming a sale is not a scoring run, and must not look like one.

    The friction reading needs exactly one thing — the price series — and used
    to obtain it by calling `service.analyze()`: the whole fetch suite (six
    sources, each a network hit past its TTL at a 2s rate limit), all 51
    metrics, scoring, eligibility, and, at Stage 4.6, **an append to the
    git-tracked, append-only score history**. That last one is the real damage.
    `score_history.jsonl` is a record of scoring runs somebody asked for; a row
    written because a position was sold is a run nobody performed, and it lands
    on the one code path whose entire design goal is that exactly two stores are
    touched.
    """

    def scoring_service(self, entered: date) -> StubService:
        """A stub whose `analyze` logs a history row, exactly as Stage 4.6 does.

        The point of logging from the stub is that the assertion then has teeth
        from the store's side rather than only from a call counter: an empty
        history file is proof the pipeline was never entered, whatever route a
        future refactor takes to the price series.
        """

        class ScoringService(StubService):
            def analyze(self, ticker, use_llm=True, **kw):
                result = super().analyze(ticker, use_llm=use_llm, **kw)
                score_history.append_run(result, "abc123")
                return result

        return ScoringService(data={"price": stepped_price(entered)})

    def test_confirming_an_exit_appends_nothing_to_the_score_history(self, wm, queue):
        entered = reviewed_position(wm)
        service = self.scoring_service(entered)

        confirm_exit(wm, queue, "ASTRAL", service, entered + timedelta(days=400))

        # Redirected to a tmp path by the autouse `isolate_score_history`
        # fixture, so this asserts on the log the run would actually have
        # written rather than on the repo's own.
        assert not score_history.DEFAULT_HISTORY_PATH.exists()

    def test_the_pipeline_is_never_run(self, wm, queue):
        entered = reviewed_position(wm)
        service = self.scoring_service(entered)

        confirm_exit(wm, queue, "ASTRAL", service, entered + timedelta(days=400))

        assert service.calls == []

    def test_the_price_series_is_fetched_once_and_from_one_source(self, wm, queue):
        entered = reviewed_position(wm)
        service = priced_service(entered)

        confirm_exit(wm, queue, "ASTRAL", service, entered + timedelta(days=400))

        assert [call[0] for call in service.suite.price_volume.calls] == ["ASTRAL"]

    def test_the_reading_is_the_same_one_the_pipeline_would_have_produced(
        self, wm, queue
    ):
        """The fetch is the same TTL-cached DataFrame `fetch_all` would have
        put in `data["price"]`, so the figure must not move."""
        entered = reviewed_position(wm)

        outcome = confirm_exit(wm, queue, "ASTRAL", priced_service(entered),
                               entered + timedelta(days=400))

        assert outcome["friction"]["gross_return_pct"] == pytest.approx(50.0)
        assert outcome["friction"]["net_return_pct"] == pytest.approx(42.875)
        assert outcome["friction"]["basis"] == "recorded"

    def test_a_fetch_that_raises_costs_the_reading_and_not_the_exit(self, wm, queue):
        """The unavailable-with-reason behaviour is unchanged: the sale is a
        fact, and a broken source must not stop it being recorded."""
        entered = reviewed_position(wm)
        service = priced_service(entered)

        def refuses(*args, **kwargs):
            raise RuntimeError("the price source is down")

        service.suite.price_volume.fetch = refuses

        outcome = confirm_exit(wm, queue, "ASTRAL", service,
                               entered + timedelta(days=400))

        assert outcome["ok"] is True
        assert wm.get("ASTRAL")["state"] == "exited"
        assert outcome["friction"]["available"] is False
        assert "price source is down" in outcome["friction"]["reason"]


class TestUnpriceableExits:
    """A data gap must not stop reality from being recorded.

    The sale happened. Refusing to write it because the position could not be
    priced would leave the books wrong, which is worse than a reading that
    says, in the house style, unknown *with its reason*.
    """

    def test_an_exit_with_no_probe_in_its_history_still_records(self, wm, queue):
        reviewed_position(wm, probed=False)

        outcome = confirm_exit(wm, queue, "ASTRAL", StubService(), date(2026, 8, 7))

        assert outcome["ok"] is True
        assert wm.get("ASTRAL")["state"] == "exited"
        assert outcome["friction"]["available"] is False
        assert "probe" in outcome["friction"]["reason"]

    def test_an_exit_with_no_usable_price_bars_still_records(self, wm, queue):
        entered = reviewed_position(wm)

        outcome = confirm_exit(wm, queue, "ASTRAL", StubService(data={}),
                               entered + timedelta(days=400))

        assert outcome["ok"] is True
        assert wm.get("ASTRAL")["state"] == "exited"
        assert outcome["friction"]["available"] is False
        assert outcome["friction"]["reason"]

    def test_the_unavailable_reading_lands_on_both_stores(self, wm, queue):
        entered = reviewed_position(wm)

        confirm_exit(wm, queue, "ASTRAL", StubService(data={}),
                     entered + timedelta(days=400))

        details = wm.get("ASTRAL")["state_history"][-1]["details"]
        assert details == queue.exits()[0]["friction"]
        assert details["available"] is False
        assert details["reason"]

    def test_the_evidence_states_why_no_figure_could_be_computed(self, wm, queue):
        """Not silence: a line that never mentions friction is indistinguishable
        from one where the model simply was not run."""
        entered = reviewed_position(wm)

        confirm_exit(wm, queue, "ASTRAL", StubService(data={}),
                     entered + timedelta(days=400))

        evidence = wm.get("ASTRAL")["state_history"][-1]["evidence"]
        assert REVIEW_REASON in evidence
        assert "unavailable" in evidence.lower()


class TestTransitionDetails:
    """`transition`'s new structured field, from the caller's side.

    It exists because a friction payload reports must read back cannot ride
    prose. Two properties keep it from costing anything: a transition that
    carries no payload writes exactly the record it always wrote, and a payload
    handed in is copied rather than wired into an append-only store.
    """

    def test_a_transition_with_no_details_records_no_details_key(self, wm):
        wm.add("ASTRAL")
        record = wm.transition("ASTRAL", "qualify", "qualification_passed")

        assert "details" not in record
        assert "details" not in wm.get("ASTRAL")["state_history"][-1]

    def test_a_stored_payload_is_a_copy_not_the_caller_s_object(self, wm):
        """An append-only record that a caller can still edit after the fact is
        not append-only."""
        wm.add("ASTRAL")
        payload = {"net_return_pct": 42.875, "basis": "recorded"}

        wm.transition("ASTRAL", "qualify", "t", details=payload)
        payload["net_return_pct"] = -99.0

        assert wm.get("ASTRAL")["state_history"][-1]["details"]["net_return_pct"] == 42.875


class TestExitedIsUnreachableByAnyOtherPath:
    """The invariant KTD10 rests on, asserted from both ends."""

    def test_no_declared_trigger_names_exited_as_its_destination(self):
        offenders = [
            trigger_id
            for trigger_id, spec in load_triggers().items()
            if spec.get("to") == lifecycle_states.EXITED
        ]
        assert offenders == []

    def test_a_full_advance_over_every_state_never_proposes_exited(self, wm):
        """Run with `apply=True`, the most permissive setting there is: even
        then the furthest any company moves is `exit_review`."""
        for state in lifecycle_states.STATES:
            ticker = f"T{state.upper()}"
            wm.add(ticker)
            if state != lifecycle_states.SCREEN:
                wm.transition(ticker, state, "seed")

        result = advance(
            StubService(), wm, apply=True,
            evaluator=TriggerEvaluator(load_triggers()), as_of=date(2026, 8, 7),
        )

        proposed = [
            outcome["proposal"]["to"]
            for outcome in result["outcomes"] if outcome["proposal"]
        ]
        assert proposed, "the fixture proposed nothing at all — the test is vacuous"
        assert lifecycle_states.EXITED not in proposed
        # `TEXITED` was seeded there directly and stays there; nothing else may
        # have arrived.
        arrived = [t for t in wm.tickers() if wm.get(t)["state"] == "exited"]
        assert arrived == ["TEXITED"]
