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
made *recoverable*, in four ordered steps:

  1. **Validate before any durable write.** The state must be `exit_review`;
     anything else is a clean refusal that names the actual state and says
     nothing was recorded. The `exit_id` is derived from the timestamp of the
     `exit_review` transition in the entry's own history, so a retry on any
     later day computes exactly the same id.
  2. **Queue event first**, keyed by that id. `record_exit` refuses a duplicate,
     which is what makes the append idempotent.
  3. **Transition second**, carrying the same friction payload as structured
     `details` beside its prose evidence.
  4. **The completion stamp last** — a `confirmed` event, keyed by the same id,
     recording that the watchlist agreed. It exists because completeness has to
     outlive the entry that proves it: read from live lifecycle state instead,
     an exit whose company later left the watchlist became permanently
     unroutable, and the queue then reported no proceeds outstanding.
     `lifecycle/reinvestment.py` argues the reading; this is where it is
     written.

**The ordering is the crash-safety argument.** A crash between steps 2 and 3
leaves the entry still in `exit_review` with a queue event present. Re-running
the command recomputes the same `exit_id`, finds that event, **adopts its exit
date and friction payload verbatim**, skips the duplicate append, and completes
the transition. Reconciliation is "run it again" — no new tooling, no repair
mode. Adopting rather than recomputing matters on its own: a retry days later
would otherwise re-price the same sale against newer bars, and the two stores
would disagree about a single event.

A crash between steps 3 and 4 is the shallowest window and repairs the same
way. The sale is in both stores and only the stamp is missing, so the retry
meets an entry already in `exited` — a state the command used to refuse
outright, which would have made the one command that could finish the record
refuse on the very state it was being asked to finish. It now appends the stamp
and nothing else: no re-pricing, no second transition, no new queue event.

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

**Two stores are written and exactly one source is read.** The friction reading
needs the price series and nothing else, so it fetches the price series and
nothing else. Running the pipeline for it — which is how this was first written
— also meant appending to the append-only score history at Stage 4.6, logging a
scoring run that never happened every time a sale was confirmed.

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
from boundless100x.lifecycle import reinvestment
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.states import APPLIED_OWNER

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

    **One source, and no compute engine.** The reading needs exactly one thing
    — the price series — and reaching it through `service.analyze()` bought
    that column at the price of the whole pipeline: six fetchers, each a
    network hit past its TTL at a two-second rate limit, then 51 metrics,
    scoring and eligibility. Worse, `analyze` appends a row to the git-tracked,
    append-only score history at Stage 4.6, so confirming a sale logged a
    scoring run nobody performed — on the one path whose entire design is that
    exactly two stores are written. `suite.price_volume.fetch` returns the same
    TTL-cached DataFrame `fetch_all` would have put in `data["price"]`, so the
    figure is unchanged and only the cost is.

    Never raises. Every gap it can meet comes back as unavailable-with-reason,
    because the exit records either way and the alternative — an exception here
    aborting a sale that already happened — is the failure this design refuses.
    An empty frame from a source that returned nothing is one of those gaps:
    `friction.model_exit` answers it with its own reason rather than a figure.

    The probe lookup, the pricing and the failure conversion are
    `friction.reading_for_exit`'s, shared with the two estimate paths. What this
    path states for itself is what makes it the *recorded* one: a missing
    `probe` is unavailable-with-reason rather than silence, because a sale going
    into the books needs the record to say why there is no figure beside it;
    and the price series is supplied as a **callable**, so the fetch happens
    inside the same try that catches a pricing failure, and does not happen at
    all when there is no `probe` to price from.
    """
    def fetch_price():
        suite = service.suite
        return suite.price_volume.fetch(
            ticker, years=suite.price_years, output_dir=suite.raw_data_dir
        )

    return friction_module.reading_for_exit(
        entry,
        fetch_price,
        exit_date,
        config=getattr(service, "config", {}),
        basis=friction_module.BASIS_RECORDED,
        no_probe_reason=(
            "no probe transition in this company's history — there is no "
            "recorded entry date to measure a holding period from"
        ),
        failure_reason="the position could not be priced ({error})",
        label=ticker,
    )


def _stamp_only(watchlist, queue, ticker: str, entry: dict, exit_id: str,
                event: dict) -> dict:
    """Complete an exit whose transition landed but whose stamp did not.

    Step 4's own crash window, and the shallowest of the three: both stores
    already hold the sale, and what is missing is only the queue's record that
    they agree. So nothing is re-priced, no transition is written, and no
    second queue event is appended — the stamp adopts the timestamp of the
    `exited` transition that is already there, and the proceeds become
    routable.

    Reached rather than refused because the alternative is the failure the
    whole protocol is built against: an entry sitting in `exited` whose
    proceeds no surface will offer to route, with `watchlist exit` — the one
    command that could repair it — refusing on the very state it is being asked
    to repair.

    The friction payload comes back from the stored event rather than being
    recomputed, for the reason the adopting retry does the same: a figure
    re-priced days later would make the two stores disagree about one sale.
    """
    exited = lifecycle_states.last_record_into(
        entry.get("state_history"), lifecycle_states.EXITED
    )
    if not (exited or {}).get("at"):
        return _refused(
            ticker,
            f"{ticker} is in {lifecycle_states.EXITED!r} but its history holds no "
            f"transition into it, so there is no timestamp to record as the "
            f"moment the exit completed — the stores cannot be reconciled "
            f"automatically, and nothing was recorded",
            entry.get("state"),
        )

    confirmation = queue.record_confirmation(exit_id, at=exited["at"])
    logger.info(f"{ticker}: stamped the exit already transitioned as {exit_id}")
    return {
        "ok": True,
        "ticker": ticker,
        "lane": entry.get("lane"),
        "exit_id": exit_id,
        "exit_date": event.get("at"),
        "trigger_id": event.get("trigger_id", ""),
        "friction": event.get("friction") or {},
        "adopted": True,
        # Distinct from `adopted`, because the two runs did different amounts of
        # work and the surface says so: an adopting retry completed the
        # transition, this one found the transition already there and completed
        # only the record of it.
        "stamp_only": True,
        "evidence": (exited or {}).get("evidence", ""),
        "event": event,
        "transition": exited,
        "confirmation": confirmation,
    }


def confirm_exit(watchlist, queue, ticker: str, service, as_of=None) -> dict:
    """Record an owner-confirmed exit across both stores. The only path to `exited`.

    Returns `{ok: True, ...}` with the exit id, date, friction payload, queue
    event, transition record and completion stamp, or `{ok: False, reason,
    state}` on a refusal that wrote nothing. See the module docstring for why
    the writes are ordered as they are, and why an unpriceable exit still
    records.

    An exception escaping step 3 or step 4 is left to propagate rather than
    converted: the queue event is already durable, so the situation is exactly
    a recoverable crash window, and re-running the command completes it.
    """
    ticker = ticker.upper()

    # ── step 1: validate, and identify the exit, before anything is written ──
    stamp_only = False
    try:
        entry = watchlist.get(ticker)
        if entry is None:
            return _refused(ticker, f"{ticker} is not on the watchlist — nothing was recorded")

        state = entry["state"]
        review = lifecycle_states.last_transition_into(
            entry, lifecycle_states.EXIT_REVIEW
        )
        # The id a retry recomputes identically, which is the whole basis of the
        # idempotent appends below. The format is `reinvestment`'s, because the
        # store that persists the id also parses it back apart to find the
        # review it keys on — two copies would eventually disagree about which
        # sale an event describes.
        exit_id = (
            reinvestment.exit_id_for(ticker, review["at"]) if review else ""
        )
        existing = queue.find_exit(exit_id) if exit_id else None
        stamped = queue.find_confirmation(exit_id) if exit_id else None

        if state == lifecycle_states.EXITED and existing is not None and stamped is None:
            # Step 4's window: the sale is recorded in both stores and only the
            # stamp is missing. Handled below rather than here, so the append
            # sits outside this try — a commit failure there is the same
            # recoverable situation as one in step 3, and converting it to a
            # refusal would tell the owner nothing was written when the point is
            # that something already was.
            stamp_only = True
        elif state != lifecycle_states.EXIT_REVIEW:
            return _refused(
                ticker,
                f"{ticker} is in {state!r}, not {lifecycle_states.EXIT_REVIEW!r} — an "
                f"exit is confirmed only from an exit review, and nothing was recorded",
                state,
            )
        elif review is None:
            return _refused(
                ticker,
                f"{ticker} is in exit_review but its history holds no transition "
                f"into exit_review, so the exit has no timestamp to key a stable "
                f"id on — a retry could not recognise its own earlier attempt, and "
                f"nothing was recorded",
                state,
            )
    except Exception as e:
        # Nothing has been written at this point, and without an `exit_id` the
        # append could not be made safe anyway — so this refuses rather than
        # guessing its way forward.
        logger.error(f"{ticker}: the exit could not be identified: {e}")
        return _refused(
            ticker, f"the exit could not be identified ({e}) — nothing was recorded"
        )

    if stamp_only:
        return _stamp_only(watchlist, queue, ticker, entry, exit_id, existing)

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
    # back apart. `at=exit_date` for the same reason `record_exit` above takes
    # it explicitly: `exit_date` is the sale's own date (the owner's `as_of`,
    # or the adopted event's original date on a retry), not the moment this
    # command happens to run. In real-time use the two are the same instant
    # (`as_of` defaults to `date.today()`), so this changes nothing for a
    # live `watchlist exit` — but a caller passing a genuinely historical
    # `as_of` (Phase 4's replay) would otherwise have every `EXITED`
    # transition stamped with wall-clock time instead of the date being
    # replayed, silently breaking `since_state_entry` and every holding-period
    # reading downstream of it (`lifecycle/friction.py`).
    record = watchlist.transition(
        ticker,
        lifecycle_states.EXITED,
        trigger_id=trigger_id,
        evidence=evidence,
        details=friction,
        applied_by=APPLIED_OWNER,
        at=exit_date,
    )

    # ── step 4: the completion stamp, last ──
    # Carrying the transition's own timestamp, not this moment's: the stamp
    # records when the watchlist agreed the sale completed, and a run
    # reconciling an older crash must not restate that as having happened when
    # it caught up.
    confirmation = queue.record_confirmation(exit_id, at=record["at"])

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
        "stamp_only": False,
        "evidence": evidence,
        "event": event,
        "transition": record,
        "confirmation": confirmation,
    }
