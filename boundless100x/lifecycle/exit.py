"""Confirming an exit — the only path to `exited` (KTD10, R3a).

**No metric can observe that the owner sold.** Every other lifecycle transition
is proposed by a trigger reading the company's own numbers, but a trigger
firing on price or fundamentals would record a *sale*, and a sale is an act,
not a reading. That would be the automated execution v05 §13 forbids. So the
registry produces `exit_review` and nothing produces `exited`; `confirm_exit`
is the operation, invoked by an explicit owner command, and it is deliberately
not wired into `advance()` at all.

Confirming an exit writes **two stores** — a queue event and a watchlist
transition — and two JSON files cannot be written atomically. Rather than
pretend otherwise with a transaction that does not exist, the failure window is
made *recoverable*, in three ordered steps:

  1. **Validate before any durable write.** The state must be `exit_review`;
     anything else is a clean refusal that names the actual state and says
     nothing was recorded. The `exit_id` is derived from the timestamp of the
     `exit_review` transition in the entry's own history, so a retry on any
     later day computes exactly the same id.
  2. **Queue event first**, keyed by that id. `record_exit` refuses a duplicate,
     which is what makes the append idempotent.
  3. **Transition second**, carrying the same friction payload as structured
     `details` beside its prose evidence.

**The ordering is the crash-safety argument.** A crash between steps 2 and 3
leaves the entry still in `exit_review` with a queue event present. Re-running
the command recomputes the same `exit_id`, finds that event, **adopts its exit
date and friction payload verbatim**, skips the duplicate append, and completes
the transition. Reconciliation is "run it again" — no new tooling, no repair
mode. Adopting rather than recomputing matters on its own: a retry days later
would otherwise re-price the same sale against newer bars, and the two stores
would disagree about a single event.

The reverse order would be unrecoverable by construction. Transition first plus
a crash leaves an exited position with no queue event — and the state check
would then refuse the very retry that could repair it, because the entry is no
longer in `exit_review`. Atomic replace inside each store (`atomic_write_json`)
covers a crash *within* one write; this ordering covers the window *between*
them.

**A friction reading that cannot be computed does not abort the exit.** The
owner's sale is a fact. Refusing to record it because there is no `probe`
transition to date the holding period from, or no usable price bars to price it
against, would leave the books wrong — which is worse than a reading that says,
in the house style used everywhere else, unknown *with its reason*. So an
unpriceable exit records `{available: false, reason}` on both stores.

That draws the line for what *is* a refusal. A failure in the **pricing** half
costs the reading and nothing else. A failure in the **identifying** half —
the entry cannot be read, the history holds no `exit_review` transition to key
on, the queue cannot be consulted — is a refusal with nothing mutated, because
without a stable `exit_id` the append cannot be made idempotent and a retry
would record the same sale twice.
"""

import logging
from datetime import date

from boundless100x.lifecycle import friction as friction_module
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.watchlist import APPLIED_OWNER

logger = logging.getLogger(__name__)


def _refused(ticker: str, reason: str, state: str | None = None) -> dict:
    """A refusal that says what was wrong and that nothing was written.

    Returned rather than raised: every refusal here is a statement about the
    owner's command, not a fault in the system, and the surface invoking it
    needs a line to print rather than a traceback to interpret. The
    "nothing was recorded" half is not decoration — after a command that writes
    two stores, the one thing the owner needs to know is whether either of them
    moved.
    """
    logger.info(f"{ticker}: exit refused — {reason}")
    return {"ok": False, "ticker": ticker, "state": state, "reason": reason}


def _friction_for_confirmed_exit(service, ticker: str, entry: dict, exit_date) -> dict:
    """What the position is modeled to have kept, at `basis: recorded`.

    `recorded` means the two dates stopped moving — the entry is the confirmed
    `probe` and the exit is the sale the owner is confirming now — **not** that
    the figure stopped being a model. It rests on the same proxies every other
    reading does: a confirmation date rather than a fill, market bars rather
    than trade prices, no cost basis anywhere. `lifecycle/friction.py` carries
    that language, and this function only chooses the two dates.

    Never raises. Every gap it can meet comes back as unavailable-with-reason,
    because the exit records either way and the alternative — an exception here
    aborting a sale that already happened — is the failure this design refuses.
    """
    probe = lifecycle_states.last_transition_into(entry, lifecycle_states.PROBE)
    if probe is None:
        return {
            **friction_module.unavailable(
                "no probe transition in this company's history — there is no "
                "recorded entry date to measure a holding period from"
            ),
            "basis": friction_module.BASIS_RECORDED,
        }

    try:
        result = service.analyze(ticker, use_llm=False, include_momentum=False)
        return friction_module.model_exit(
            (result.data or {}).get("price"),
            probe.get("at"),
            exit_date,
            config=getattr(service, "config", {}),
            basis=friction_module.BASIS_RECORDED,
        )
    except Exception as e:
        # The sale is being recorded regardless, so a broken fetch costs the
        # reading and says so. Logged at warning because an exit with no
        # figure beside it is worth noticing, even though it is not an error.
        logger.warning(f"{ticker}: the exit could not be priced: {e}")
        return {
            **friction_module.unavailable(
                f"the position could not be priced ({e})"
            ),
            "basis": friction_module.BASIS_RECORDED,
        }


def confirm_exit(watchlist, queue, ticker: str, service, as_of=None) -> dict:
    """Record an owner-confirmed exit across both stores. The only path to `exited`.

    Returns `{ok: True, ...}` with the exit id, date, friction payload, queue
    event and transition record, or `{ok: False, reason, state}` on a refusal
    that wrote nothing. See the module docstring for why the two writes are
    ordered as they are, and why an unpriceable exit still records.

    An exception escaping step 3 is left to propagate rather than converted:
    the queue event is already durable, so the situation is exactly the
    recoverable crash window, and re-running the command completes it.
    """
    ticker = ticker.upper()

    # ── step 1: validate, and identify the exit, before anything is written ──
    try:
        entry = watchlist.get(ticker)
        if entry is None:
            return _refused(ticker, f"{ticker} is not on the watchlist — nothing was recorded")

        state = entry["state"]
        if state != lifecycle_states.EXIT_REVIEW:
            return _refused(
                ticker,
                f"{ticker} is in {state!r}, not {lifecycle_states.EXIT_REVIEW!r} — an "
                f"exit is confirmed only from an exit review, and nothing was recorded",
                state,
            )

        review = lifecycle_states.last_transition_into(
            entry, lifecycle_states.EXIT_REVIEW
        )
        if review is None:
            return _refused(
                ticker,
                f"{ticker} is in exit_review but its history holds no transition "
                f"into exit_review, so the exit has no timestamp to key a stable "
                f"id on — a retry could not recognise its own earlier attempt, and "
                f"nothing was recorded",
                state,
            )

        # The id a retry recomputes identically, which is the whole basis of
        # the idempotent append below.
        exit_id = f"{ticker}:{review['at']}"
        existing = queue.find_exit(exit_id)
    except Exception as e:
        # Nothing has been written at this point, and without an `exit_id` the
        # append could not be made safe anyway — so this refuses rather than
        # guessing its way forward.
        logger.error(f"{ticker}: the exit could not be identified: {e}")
        return _refused(
            ticker, f"the exit could not be identified ({e}) — nothing was recorded"
        )

    trigger_id = review.get("trigger_id", "")
    lane = entry["lane"]

    if existing is not None:
        # A previous attempt got through step 2 and stopped. Complete *that*
        # exit: its date and payload are adopted verbatim rather than
        # recomputed, or a retry on a later day would price the same sale
        # against newer bars and leave the two stores disagreeing about it.
        adopted = True
        exit_date = existing["at"]
        friction = existing["friction"]
        logger.info(f"{ticker}: completing the exit already recorded as {exit_id}")
    else:
        adopted = False
        exit_date = str(as_of or date.today())
        friction = _friction_for_confirmed_exit(service, ticker, entry, as_of or date.today())

    reason = review.get("evidence") or f"exit review recorded under {trigger_id}"
    # `describe` states the net figure beside the gross one when the reading is
    # available, and the reason it could not be computed when it is not. The
    # unavailable branch is deliberately *in* the evidence here, unlike an exit
    # *proposal*: a proposal that mentions a missing estimate is noise, while a
    # recorded sale with no figure needs the record to say why there is none.
    evidence = f"{reason} [{friction_module.describe(friction)}]"

    # ── step 2: the queue event, first and idempotently ──
    if adopted:
        event = existing
    else:
        event = queue.record_exit(
            ticker=ticker, lane=lane, trigger_id=trigger_id,
            friction=friction, at=exit_date, exit_id=exit_id,
        )

    # ── step 3: the transition, second ──
    # The same payload reaches both stores, and `details` is what makes that
    # possible: the prose says it, the structured field is what a report reads
    # back apart.
    record = watchlist.transition(
        ticker,
        lifecycle_states.EXITED,
        trigger_id=trigger_id,
        evidence=evidence,
        details=friction,
        applied_by=APPLIED_OWNER,
    )

    return {
        "ok": True,
        "ticker": ticker,
        "lane": lane,
        "exit_id": exit_id,
        "exit_date": exit_date,
        "trigger_id": trigger_id,
        "friction": friction,
        # True when this run completed an exit a previous run had already
        # queued — the caller renders it, because "recorded" and "reconciled"
        # are different things to have just done.
        "adopted": adopted,
        "evidence": evidence,
        "event": event,
        "transition": record,
    }
