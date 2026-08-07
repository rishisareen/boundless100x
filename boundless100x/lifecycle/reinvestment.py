"""The reinvestment queue — where exited capital went, and where it has not.

v05 §8.1 asks what happens to the proceeds of an exit. This module holds the
durable half of the answer: an **append-only log of events** recording that a
position was exited, and, later, that its proceeds were deployed into something
else. From those two kinds of event everything else is derived — which exits
are still unrouted, and how long each has been idle.

It is a sibling store to `watchlist.json`, deliberately: a tracked JSON file
with the same load / validate / commit shape, not generated state. What was
sold and where the money went is a record of decisions, exactly like the
watchlist's own `state_history`, and it outlives any cache.

Four properties are load-bearing.

**An exit append is idempotent, keyed by `exit_id`.** KTD10's exit protocol
writes this store *before* it writes the transition, so that a crash between
the two leaves a state that re-running the command repairs. That repair only
works because a second append with the same `exit_id` is refused rather than
duplicated — the refusal is what turns "run it again" into reconciliation.

**Completeness is stamped, not inferred.** An exit is complete when a
`confirmed` event says so — appended after the watchlist transition lands, and
keyed by the same `exit_id`. It was once derived from live lifecycle state
instead ("does this ticker hold an `exited` transition?"), which answered a
per-*exit* question with a per-*ticker* fact and was wrong in both directions.
Removing a company from the watchlist made its already-recorded proceeds
permanently unroutable, and the queue then reported "No exit proceeds awaiting
routing" — the exact false all-clear this module exists to prevent — while the
recovery the display offered (`watchlist exit <ticker>`) could never succeed,
the ticker being gone. In the other direction, a ticker re-added and exited a
second time made the *older* event read complete on the strength of the newer
sale. A stamped event survives the entry it describes, which is the property
the reading needed and live state cannot have.

The live fallback is kept for exactly one case: the window between the
transition and the stamp. There the entry's own history is the evidence that
the stamp has not caught up yet — and it is read **matched to this exit's own
review**, never to any `exited` record the ticker happens to hold, or the
per-ticker confusion comes straight back through the fallback.

**Routing is an append, never a mutation.** Marking an exit routed adds a
`routing` event referencing the `exit_id` it deploys. The exit event itself is
never touched, so what was recorded at the moment of the sale stays exactly as
it was recorded, and the log can be read in order and believed.

**A routing event carries two timestamps.** `deployed_at` is when capital
actually moved, taken from the deployment transition; `recorded_at` is when the
command was run. The idle reading closes at `deployed_at`, so recording a
deployment late does not inflate the window it closes. Collapsing them into one
field would make lateness in the *recording* look like lateness in the
*deployment*.

**Every commit is copy-on-write, and the mechanics come from `watchlist.py`.**
`ReinvestmentQueue` extends `watchlist._JsonStore` rather than restating the
staging, the atomic write and the revision counter — the durability argument is
per-file and identical wherever it applies, and two copies would be two things
to keep in step. The counter makes that concrete: `snapshot_state` compares
**both** stores' revisions, so a clamping rule that disagreed between the files
would render a routing proposal current against one store and stale against the
other. A mutator stages onto a deep copy, writes it, and adopts it only once the
write returns. A crash mid-write leaves the previous store rather than truncated
JSON; a failed write leaves `self.data` describing exactly what is on disk. The
second is the more dangerous: a phantom event surviving in memory would let a
same-process retry skip an append it believes already landed, and the exit would
end up recorded in one store only — the precise disagreement the protocol exists
to prevent. Only the mechanics are shared: the file, the schema and the
validation are this module's own.

The store also holds a **replaceable `latest_proposal` slot** beside the log:
the whole-run routing view, which is a snapshot rather than a record and is
therefore overwritten rather than appended. Appending an event never touches
it, and loading preserves it untouched.

There is one schema and no migration path, following the watchlist's own rule:
an event that does not match is a loud error, because with one schema in
existence an odd event means something is wrong, and repairing it silently is
how proceeds end up attributed to a sale nobody made.

── The routing view ────────────────────────────────────────────────────────

`propose_routing` is the derived half: given the run's advance outcomes and its
concentration reading, it names the one candidate best placed to receive
proceeds, the candidates it skipped **with their blocking reasons**, and how
long each unrouted exit has been idle. It is inert data. It never transitions
anything, because a proposal that could move a company would be the automated
execution v05 §13 forbids, and because the owner may deploy somewhere else
entirely — the routing *event* records what actually happened, not what was
advised.

Four rules shape it, and each exists because its opposite is a plausible
mistake:

**A proposal requires proceeds.** With no completed unrouted exit the view
carries no candidate. Capital that does not exist cannot be routed toward one,
and a standing recommendation with nothing to fund it reads as an instruction
to buy.

**An unfinished exit record is not proceeds — and is not emptiness either.** An
exit event whose ticker still sits in `exit_review` is KTD10's crash window:
the queue event landed and the transition did not. It is excluded from routing
until it is completed, both here and in `queue route`, so the direct command
cannot bypass the display's exclusion. But excluded is not absent. `NO_PROCEEDS`
is reserved for a queue with genuinely nothing outstanding; where the only thing
standing between the owner and their capital is a half-written record,
`unroutable_reason` says so and names the command that finishes it. The two
sentences look alike and mean opposite things — one says there is nothing to do,
the other says there is something only the owner can do.

**Safety is read, never re-derived.** Each outcome already carries a
`routing_safety` payload built by `advance_ticker`, whose eligibility question
follows the lane. Calling `action_policy.resolve_for_result` here instead would
return `None` on this path — `advance` analyses with `use_llm=False` — and pass
every candidate silently.

**Blocked candidates are part of the answer.** A view reporting only its winner
renders an all-blocked run identically to an empty watchlist, and those two
mean opposite things to whoever is deciding whether the system still works.

── Freshness is revisions, not clocks ──────────────────────────────────────

The snapshot captures both stores' `revision` at generation, and
`snapshot_state` compares them against the live counters. Any later mutation —
an add, a removal, a catalyst edit, an exit, a route — advances one and renders
the snapshot `Stale`. A clock comparison was the earlier design and is wrong
twice over: it misses every mutation that does not re-score, and it breaks on a
backdated run, since `as_of` may be a historical business date.
`generated_at` is therefore display-only.

**Only `Current` may render the proposal.** `Partial` (a full run with errored
tickers) and `Stale` (superseded inputs) keep their diagnostics and say to
re-run instead, because a candidate named by incomplete or superseded inputs is
a recommendation the inputs no longer back. `Partial` outranks `Stale`: both
can hold at once, and only one of them names a ticker to go and look at.
"""

import copy
import json
import logging
from datetime import date, datetime
from pathlib import Path

from boundless100x.lifecycle import portfolio
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.watchlist import APPLIED_OWNER, _JsonStore, _revision_of

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path(__file__).parent / "reinvestment_queue.json"

EXIT_EVENT = "exit"
ROUTING_EVENT = "routing"
CONFIRMED_EVENT = "confirmed"
EVENT_KINDS = (EXIT_EVENT, ROUTING_EVENT, CONFIRMED_EVENT)

# The one sentence for "there is nothing to route". Shared between the view and
# every surface that renders it, so the display and the `queue route` refusal
# cannot drift into saying different things about the same emptiness.
#
# **Reserved for genuine emptiness.** Reaching for it whenever no exit is
# routable is the false all-clear the module docstring argues against — go
# through `unroutable_reason`, which picks this sentence only when there is in
# fact nothing outstanding.
NO_PROCEEDS = "No exit proceeds awaiting routing"

# KTD10's crash window, stated with the command that closes it. An owner
# meeting this line needs the recovery step in the same breath, not a
# description of a state they did not know existed.
INCOMPLETE_EXIT_NOTICE = (
    "Exit recording incomplete — run `watchlist exit {ticker}` to complete it"
)

# The two shapes of disagreement no command can repair, because the lifecycle
# record the queue would reconcile against is gone or was never written. Both
# are reachable only by editing a store by hand or restoring one from a copy —
# `watchlist remove` refuses while an exit is unconfirmed, and `confirm_exit`
# stamps before anything can remove the entry. They are still rendered rather
# than swallowed: an unroutable event nobody can see is how proceeds go missing.
ORPHANED_EXIT_NOTICE = (
    "{ticker} is no longer on the watchlist and this exit was never confirmed, "
    "so no command can complete it — the stores disagree and one of them was "
    "edited outside this system"
)
UNMATCHED_EXIT_NOTICE = (
    "{ticker} is on the watchlist in {state!r} but its history holds no "
    "`exited` transition for this exit's review — the sale is recorded in the "
    "queue alone"
)

# An `exit_id` is `TICKER:<the exit_review transition's timestamp>`. The format
# lives here, beside the store that persists it, rather than in `exit.py` which
# computes it: a retry recomputing the id is the whole basis of the idempotent
# append, and the per-exit completeness reading below parses the same id back
# apart to find the review it keys on. Two copies of the format and those two
# would eventually disagree about which sale an event describes.
#
# A ticker never contains a colon, so one split on the first separator is exact.
EXIT_ID_SEPARATOR = ":"

# Snapshot states, in precedence order. `Unavailable` and `Partial` are facts
# about the run that produced the snapshot; `Stale` is a fact about everything
# that happened since.
SNAPSHOT_UNAVAILABLE = "unavailable"
SNAPSHOT_PARTIAL = "partial"
SNAPSHOT_STALE = "stale"
SNAPSHOT_CURRENT = "current"

# States a candidate can be in and still receive capital. Deliberately not
# "everything that is not positioned": `dropped`, `exited` and `exit_review`
# are all non-positioned, and proposing that proceeds be deployed into a
# company the same run just dropped is not a proposal anyone should be shown.
CANDIDATE_STATES = (
    lifecycle_states.WATCH,
    lifecycle_states.QUALIFY,
    lifecycle_states.SCREEN,
)

# Nearest to a position first. This is the *readiness* ordering, and therefore
# the mirror image of `advance._PRECEDENCE`, which ranks by protectiveness —
# the two answer opposite questions and must not share a table.
_STATE_PRIORITY = {state: rank for rank, state in enumerate(CANDIDATE_STATES)}
_UNRANKED_STATE = len(CANDIDATE_STATES)

# Per kind, because the two events answer different questions and a missing
# field in either is a different kind of hole. An exit with no `friction` key
# is a sale nobody can price after the fact; a routing event with no
# `deployed_at` is an idle reading that can never be closed at the right date.
REQUIRED_KEYS = {
    EXIT_EVENT: ("kind", "exit_id", "ticker", "lane", "trigger_id", "at", "friction"),
    ROUTING_EVENT: ("kind", "exit_id", "candidate", "deployed_at", "recorded_at"),
    # `at` is the *transition's* timestamp, not this append's: the stamp records
    # when the watchlist agreed the sale completed, and a reconciling run days
    # later must not restate that as having happened when it caught up.
    CONFIRMED_EVENT: ("kind", "exit_id", "at", "recorded_at"),
}


class ReinvestmentError(ValueError):
    """A stored event does not match the schema, or an append was refused."""


def _now() -> str:
    return datetime.now().isoformat()


def _days_between(start, end) -> int | None:
    """Whole days from one stored timestamp to another, or None if unreadable.

    None rather than zero, for the reason the whole codebase treats gaps this
    way: a zero-day idle reading means the proceeds were redeployed the same
    day, and an unreadable one means nobody can say. In a table those look
    identical.

    Both ends go through `states.as_date`, which is the lifecycle layer's one
    timestamp parser. Exit dates are written as `str(as_of)` and transition
    timestamps as full ISO datetimes, so an idle reading spans two differently
    shaped strings — the same two shapes a time stop reads, which is exactly why
    one parser rather than a local copy: this store and the evaluator must not
    disagree about whether a stored `at` is readable.
    """
    first = lifecycle_states.as_date(start)
    second = lifecycle_states.as_date(end)
    if first is None or second is None:
        return None
    return (second - first).days


def exit_id_for(ticker: str, review_at: str) -> str:
    """The id both stores key this sale on: ticker and its `exit_review` stamp.

    Derived rather than generated, which is what lets a retry on any later day
    compute the same one and recognise its own earlier attempt. The review's
    timestamp is the identifying half — a position exited, re-entered and
    exited again produces two reviews and therefore two ids, so the second sale
    can never be mistaken for the first.
    """
    return f"{ticker.upper()}{EXIT_ID_SEPARATOR}{review_at}"


def review_at_of(exit_id) -> str:
    """The `exit_review` timestamp an id keys on, or `""` if it holds none.

    The inverse of `exit_id_for`, and the reason the format is stated once: the
    completeness fallback below has to find *this* exit's review in a history
    that may hold several, and it has nothing else to match on.
    """
    if not isinstance(exit_id, str) or EXIT_ID_SEPARATOR not in exit_id:
        return ""
    return exit_id.split(EXIT_ID_SEPARATOR, 1)[1]


def _exited_after_review(entry: dict, review_at: str) -> bool:
    """Whether the entry left *this* exit's review for `exited`.

    Matched to the review the `exit_id` keys on, never to any `exited` record
    the ticker happens to hold. The looser reading is what made completeness a
    per-ticker fact: a company exited, re-added, and taken to a second exit
    would answer "yes" for the *first* sale on the strength of the second, and
    proceeds recorded months apart would collapse into one routable event.

    Position in the history is the test, because that is what "after" means in
    an append-only log — comparing timestamps instead would have to reconcile a
    transition's wall clock against a review's, which are the same clock only
    while nobody replays a backdated run.
    """
    if not review_at:
        return False
    history = [r for r in entry.get("state_history") or [] if isinstance(r, dict)]
    opened = [
        index
        for index, record in enumerate(history)
        if record.get("to") == lifecycle_states.EXIT_REVIEW
        and record.get("at") == review_at
    ]
    if not opened:
        return False
    return any(
        record.get("to") == lifecycle_states.EXITED
        for record in history[opened[-1] + 1:]
    )


def exit_is_complete(event: dict | None, entry: dict | None,
                     confirmation: dict | None = None) -> bool:
    """Whether this sale is fully recorded — a question about the exit, not the ticker.

    **The stamp answers it.** A `confirmed` event is the watchlist's agreement,
    written down at the moment it was given, and it stays true afterwards
    however the entry it describes changes or whether the entry survives at all.
    That is the whole point: proceeds recorded and confirmed remain routable
    after the company leaves the watchlist, where the previous live-state
    reading made them permanently unroutable and then reported the queue as
    empty.

    **Live state is the fallback, for one window only.** Between step 3 (the
    transition) and step 4 (the stamp) the sale is complete and the queue does
    not yet say so, so the entry's own history stands in — read strictly, and
    matched to this exit's review. Everything else reads incomplete: an entry
    still in `exit_review` is KTD10's earlier crash window, and an entry that is
    simply gone is a disagreement `exit_views` reports rather than resolves.
    """
    if isinstance(confirmation, dict):
        return True
    if not isinstance(entry, dict):
        return False
    if entry.get("state") == lifecycle_states.EXIT_REVIEW:
        return False
    return _exited_after_review(entry, review_at_of((event or {}).get("exit_id")))


def unroutable_reason(incomplete: list[dict] | None) -> str:
    """Why nothing can be routed — emptiness, or a record only the owner can finish.

    One function because the two sentences are one decision, and every surface
    that reports "nothing to route" has to make it the same way. Reaching for
    `NO_PROCEEDS` directly is how a half-written exit came to render as an empty
    queue: identical wording for a queue holding nothing and a queue holding
    capital nobody can reach.
    """
    if not incomplete:
        return NO_PROCEEDS
    tickers = ", ".join(
        sorted({str(view.get("ticker")) for view in incomplete})
    )
    return (
        f"{len(incomplete)} exit(s) are recorded but not confirmed ({tickers}) "
        f"— their proceeds cannot be routed until the recording is completed, "
        f"and this is not an empty queue"
    )


def eligible_deployments(entry: dict | None, exit_at) -> list[dict]:
    """Owner-applied `probe`/`scale` transitions dated on or after an exit.

    The idle reading measures exit-to-*deployed-capital*, so only a transition
    that actually executed may close it. Three conditions, each excluding a
    different near-miss:

      * `probe`/`scale` only — the states where capital is committed;
      * `applied_by == owner` — an auto-applied transition moves no money, and
        this system never applies a money-moving one on its own anyway;
      * dated on or after the exit — a position entered *before* the sale was
        not funded by its proceeds, however well it would have been.

    Compared at day granularity because the two timestamps come from different
    clocks: the exit carries the owner's stated sale date, the transition a
    wall-clock stamp.
    """
    sold_on = lifecycle_states.as_date(exit_at)
    matches = []
    for record in (entry or {}).get("state_history") or []:
        if not isinstance(record, dict):
            continue
        if record.get("to") not in lifecycle_states.POSITIONED:
            continue
        if record.get("applied_by") != APPLIED_OWNER:
            continue
        moved_on = lifecycle_states.as_date(record.get("at"))
        if sold_on is not None and (moved_on is None or moved_on < sold_on):
            continue
        matches.append(record)
    return matches


def snapshot_state(snapshot: dict | None, watchlist_revision: int,
                   queue_revision: int) -> dict:
    """Which of the four states the stored routing snapshot is in, and why.

    Resolved in precedence order — `Unavailable`, `Partial`, `Stale`,
    `Current` — and the order is load-bearing twice. `Partial` before `Stale`
    because both can hold at once and only one of them names a ticker whose
    analysis failed. And **only `Current` sets `renders_proposal`**: a
    candidate named by a run that did not finish, or by inputs that have since
    moved, is a recommendation its own evidence no longer backs. The
    diagnostics survive in every state, because the blocked list and the idle
    readings are still true statements about what the run saw.

    Freshness is the revision comparison and nothing else. `generated_at` is
    carried for display; comparing it against anything would miss every
    mutation that does not re-score, and `as_of` may be a historical business
    date in any case.
    """
    if not snapshot:
        return {
            "state": SNAPSHOT_UNAVAILABLE,
            "reason": (
                "no routing snapshot has been generated yet — run "
                "`watchlist advance` to produce one"
            ),
            "renders_proposal": False,
            "errors": [],
            "generated_at": None,
            "as_of": None,
        }

    errors = list(snapshot.get("errors") or [])
    common = {
        "errors": errors,
        "generated_at": snapshot.get("generated_at"),
        "as_of": snapshot.get("as_of"),
    }

    if snapshot.get("status") == SNAPSHOT_PARTIAL or errors:
        return {
            "state": SNAPSHOT_PARTIAL,
            "reason": (
                f"the run that produced this snapshot could not evaluate "
                f"{', '.join(errors) or 'every tracked company'} — the ranking "
                f"was built on an incomplete field"
            ),
            "renders_proposal": False,
            **common,
        }

    stale = [
        label
        for label, current, captured in (
            ("watchlist", watchlist_revision, snapshot.get("watchlist_revision")),
            ("reinvestment queue", queue_revision, snapshot.get("queue_revision")),
        )
        # `!=` rather than `>`: a counter that went backwards (a store restored
        # from a copy) also means the snapshot does not describe this store, and
        # a snapshot with no counter at all cannot prove it is current. Both
        # fail closed.
        if not isinstance(captured, int) or current != captured
    ]
    if stale:
        return {
            "state": SNAPSHOT_STALE,
            "reason": (
                f"the {' and the '.join(stale)} changed after this snapshot was "
                f"written — its ranking was built on inputs that have since moved"
            ),
            "renders_proposal": False,
            **common,
        }

    return {
        "state": SNAPSHOT_CURRENT,
        "reason": "",
        "renders_proposal": True,
        **common,
    }


# ── Ranking a candidate for proceeds ──
#
# Read by `propose_routing` and by nothing else. Kept as module functions so
# that each rule can be read — and argued with — on its own, rather than
# buried inside a loop that is also sorting.


def _safety_reasons(outcome: dict) -> list[str]:
    """Why this candidate is not safe to deploy into, read off its own outcome.

    `advance_ticker` has already answered the question with the lane's own
    eligibility test (KTD11); this only reads the answer. An outcome carrying
    no reading at all blocks rather than passes — the payload is built on every
    successful advance, so its absence means something about this candidate is
    not what the router thinks it is.
    """
    safety = outcome.get("routing_safety")
    if not isinstance(safety, dict):
        return [
            "no routing-safety reading was produced for this candidate — "
            "routing is refused rather than assumed"
        ]
    if safety.get("clear") is True:
        return []
    return list(safety.get("reasons") or [
        "the routing-safety reading did not clear this candidate"
    ])


def _candidate_payload(outcome: dict, entry: dict, fired: dict | None) -> dict:
    """One ranked candidate, carrying the evidence that ranked it.

    A proposal naming a company without saying why is an instruction. When an
    entry trigger fired, its own evidence travels; when none has, the payload
    says so in as many words rather than leaving the field empty, because an
    empty evidence cell reads as evidence nobody bothered to render.
    """
    state = entry.get("state")
    return {
        "ticker": outcome.get("ticker"),
        "lane": entry.get("lane"),
        "state": state,
        "composite": outcome.get("composite"),
        "sector": outcome.get("sector"),
        "entry_trigger_fired": fired is not None,
        "trigger_id": (fired or {}).get("trigger_id"),
        "evidence": (fired or {}).get("evidence") or (
            f"no entry trigger has fired; ranked on lifecycle state ({state})"
        ),
    }


class ReinvestmentQueue(_JsonStore):
    """Reads and writes the exit / routing event log.

    A sibling store, not a second watchlist: its own file, its own schema, its
    own validation, its own question. What it inherits is the commit mechanics
    and nothing else — copy-on-write staging, the atomic write, and the revision
    counter whose clamping rule `snapshot_state` compares *across* the two
    stores and which therefore cannot be allowed to mean two things.
    """

    def __init__(self, path: str | None = None):
        super().__init__(path, DEFAULT_QUEUE_PATH)

    # ── persistence ──

    def _load(self) -> dict:
        if not self.path.exists():
            return {"events": [], "latest_proposal": None, "revision": 0}
        with open(self.path) as f:
            data = json.load(f)

        events = data.get("events", [])
        if not isinstance(events, list):
            raise ReinvestmentError("the queue's `events` must be a list")
        for index, event in enumerate(events):
            self._validate_event(index, event)

        return {
            "events": events,
            "latest_proposal": data.get("latest_proposal"),
            "revision": _revision_of(data),
        }

    @staticmethod
    def _validate_event(index: int, event: object) -> None:
        if not isinstance(event, dict):
            raise ReinvestmentError(f"event {index}: must be an object")

        kind = event.get("kind")
        if kind not in EVENT_KINDS:
            raise ReinvestmentError(
                f"event {index}: unknown kind {kind!r} — the queue records "
                f"{' and '.join(EVENT_KINDS)} events and nothing else"
            )

        missing = [key for key in REQUIRED_KEYS[kind] if key not in event]
        if missing:
            raise ReinvestmentError(
                f"event {index}: {kind} event is missing {', '.join(missing)}. The "
                f"queue has a single schema and no migration path — fix or remove "
                f"the event rather than letting it be repaired silently."
            )

    # ── events ──

    def record_exit(
        self,
        ticker: str,
        lane: str,
        trigger_id: str,
        friction: dict,
        at: str,
        exit_id: str,
    ) -> dict:
        """Append the record of a confirmed exit, refusing a duplicate `exit_id`.

        **The refusal is the point.** KTD10's protocol writes this event before
        the watchlist transition, so a crash between the two leaves an event
        here and an entry still in `exit_review`; re-running the exit command
        recomputes the same `exit_id` and must not append a second event for a
        sale that happened once. Refusing loudly rather than returning the
        existing event keeps a caller from mistaking "already recorded" for
        "recorded just now" — the one caller that legitimately meets this case
        (`lifecycle/exit.py`) asks `find_exit` first and adopts what it finds.

        `friction` is stored **whole**, in either of its two shapes: the full
        reading (gross, holding days, tax regime, net, basis) or the
        unavailable-with-reason form. Never a bare net figure and never prose —
        a report reads this back later, and an evidence string cannot be parsed
        apart into the fields it mentions.

        `at` is the date of the sale as the owner stated it, not the date of
        the bar that priced it. The bar travels inside the payload, where it
        can be inspected without being mistaken for the sale.
        """
        if not exit_id:
            raise ReinvestmentError(
                "an exit event needs an exit_id — without one the append cannot "
                "be made idempotent, and a retry would record the sale twice"
            )
        existing = self.find_exit(exit_id)
        if existing is not None:
            raise ReinvestmentError(
                f"exit {exit_id} is already recorded (on {existing.get('at')}) — "
                f"the log is append-only and an exit is recorded once"
            )

        event = {
            "kind": EXIT_EVENT,
            "exit_id": exit_id,
            "ticker": ticker.upper(),
            "lane": lane,
            "trigger_id": trigger_id,
            "at": at,
            # Separate from `at` for the same reason a routing event separates
            # its two: when the sale happened and when it was written down are
            # different facts, and a late recording must not move the sale.
            "recorded_at": _now(),
            "friction": copy.deepcopy(friction),
        }

        staged = self._stage()
        staged["events"].append(event)
        self._commit(staged)
        logger.info(f"{event['ticker']}: exit recorded ({exit_id})")
        return event

    def record_confirmation(self, exit_id: str, at: str) -> dict:
        """Append the stamp that the watchlist agreed this sale completed.

        KTD10's step 4, and the last of the three writes. It goes **after** the
        transition for the same reason the exit event goes before it: each write
        must leave a state its own retry can recognise. Stamping first would
        assert a completed sale the watchlist had not yet recorded — the one
        claim this store must never make on its own — while stamping last means
        a crash here leaves an exit the retry finds already transitioned and
        finishes by appending nothing but this event.

        `at` is the transition's timestamp, carried in rather than taken from
        the clock: a run reconciling a week-old crash records when the sale was
        agreed, not when someone got round to noticing. `recorded_at` is what
        holds the latter, exactly as it does on a routing event.

        Two refusals, both structural. A stamp for an unrecorded exit would
        assert completeness for a sale this store never saw, and a second stamp
        for one exit is a duplicate of a fact that is already true.
        """
        if self.find_exit(exit_id) is None:
            raise ReinvestmentError(
                f"no exit {exit_id} is recorded — a confirmation must reference "
                f"the exit whose completion it stamps"
            )
        existing = self.find_confirmation(exit_id)
        if existing is not None:
            raise ReinvestmentError(
                f"exit {exit_id} was already confirmed (on {existing.get('at')}) "
                f"— the log is append-only and a sale completes once"
            )
        if not at:
            raise ReinvestmentError(
                f"the confirmation of exit {exit_id} needs the timestamp of the "
                f"transition that completed it — without one the stamp cannot "
                f"say when the watchlist agreed, only when it was written down"
            )

        event = {
            "kind": CONFIRMED_EVENT,
            "exit_id": exit_id,
            "at": at,
            "recorded_at": _now(),
        }

        staged = self._stage()
        staged["events"].append(event)
        self._commit(staged)
        logger.info(f"{exit_id}: exit confirmed complete")
        return event

    def record_routing(
        self,
        exit_id: str,
        candidate: str,
        deployed_at: str,
        recorded_at: str | None = None,
    ) -> dict:
        """Append the record that an exit's proceeds were deployed.

        An append rather than a flag on the exit event: what was recorded at
        the moment of the sale stays exactly as recorded, and "routed" becomes
        a fact derived from the log rather than a field that can be edited.

        **Two timestamps, and they are not interchangeable.** `deployed_at`
        comes from the transition where capital actually moved, and is what an
        idle reading closes at; `recorded_at` is when this command ran. A route
        entered days after the deployment must not read as days of extra idle
        capital.

        Two refusals, both about referential sense rather than policy: an
        `exit_id` nobody recorded would close an idle reading that never
        opened, and an exit already routed cannot be routed again. The command
        that issues these events validates far more than this (that the ticker
        actually holds a completed `exited` transition, that the candidate was
        genuinely positioned after the exit) and refuses first with a better
        message; these two are the store's own last line.
        """
        if self.find_exit(exit_id) is None:
            raise ReinvestmentError(
                f"no exit {exit_id} is recorded — a routing event must reference "
                f"the exit whose proceeds it deploys"
            )
        routed = self.routing_for(exit_id)
        if routed is not None:
            raise ReinvestmentError(
                f"exit {exit_id} was already routed into "
                f"{routed.get('candidate')} on {routed.get('deployed_at')}"
            )

        event = {
            "kind": ROUTING_EVENT,
            "exit_id": exit_id,
            "candidate": candidate.upper(),
            "deployed_at": deployed_at,
            "recorded_at": recorded_at or _now(),
        }

        staged = self._stage()
        staged["events"].append(event)
        self._commit(staged)
        logger.info(f"{exit_id}: routed into {event['candidate']}")
        return event

    # ── reading ──

    def events(self) -> list[dict]:
        """The whole log, oldest first.

        Not a write path, for `WatchlistManager.get`'s reason: the next commit
        replaces `self.data` wholesale, so anything written into a returned
        event goes nowhere. Every change belongs in a recorder, which is also
        the only way it reaches disk.
        """
        return list(self.data["events"])

    def exits(self) -> list[dict]:
        return [e for e in self.data["events"] if e.get("kind") == EXIT_EVENT]

    def routings(self) -> list[dict]:
        return [e for e in self.data["events"] if e.get("kind") == ROUTING_EVENT]

    def find_exit(self, exit_id: str) -> dict | None:
        """The exit event with this id, or None.

        The lookup `confirm_exit` makes before it appends: finding an event
        here means a previous attempt got through step 2 and stopped, and its
        date and payload are adopted rather than recomputed.
        """
        for event in self.data["events"]:
            if event.get("kind") == EXIT_EVENT and event.get("exit_id") == exit_id:
                return event
        return None

    def find_confirmation(self, exit_id: str) -> dict | None:
        """The stamp saying this exit's sale completed, or None if it has none."""
        for event in self.data["events"]:
            if event.get("kind") == CONFIRMED_EVENT and event.get("exit_id") == exit_id:
                return event
        return None

    def unconfirmed_exits(self, ticker: str | None = None) -> list[dict]:
        """Exit events carrying no completion stamp, oldest first.

        The question `watchlist remove` asks before it deletes an entry. An
        unconfirmed exit is the one state whose repair genuinely needs the
        lifecycle record — `confirm_exit` reads the entry's `exit_review`
        transition to key on and its history to complete — so removing the
        entry underneath it strands the proceeds with no command able to reach
        them. A confirmed exit needs no such thing and survives the removal,
        which is the difference the stamp buys.
        """
        events = [
            event for event in self.exits()
            if self.find_confirmation(event.get("exit_id")) is None
        ]
        if ticker:
            events = [e for e in events if e.get("ticker") == ticker.upper()]
        return events

    def routing_for(self, exit_id: str) -> dict | None:
        """The routing event that closed this exit, or None if it is still open."""
        for event in self.data["events"]:
            if event.get("kind") == ROUTING_EVENT and event.get("exit_id") == exit_id:
                return event
        return None

    def unrouted_exits(self) -> list[dict]:
        """Exits whose proceeds have not been deployed, oldest first.

        Derived from the log rather than stored, so there is no routed flag to
        fall out of step with the routing events themselves.
        """
        routed = {e.get("exit_id") for e in self.routings()}
        return [e for e in self.exits() if e.get("exit_id") not in routed]

    def latest_proposal(self) -> dict | None:
        """The stored whole-run routing snapshot, or None if none was written."""
        return self.data.get("latest_proposal")

    # ── the routing view ──

    def exit_views(self, watchlist, as_of=None) -> list[dict]:
        """Every exit event with its route state and idle reading, oldest first.

        The idle reading **closes at `deployed_at`**, not at the moment the
        route was recorded: a deployment entered a month late did not leave the
        proceeds idle for that month, and a reading that said so would make
        bookkeeping lateness look like indecision.

        Each view also states whether the sale completed, read from this exit's
        own stamp and falling back to lifecycle state only inside the window
        before the stamp lands. An incomplete event carries the reason it is
        incomplete in its own `note` — with the recovery command when there is
        one, and saying plainly that there is none when there is not, because
        whoever meets the line is the person who has to act on it.
        """
        as_of = lifecycle_states.as_date(as_of) or date.today()
        views = []
        for event in self.exits():
            ticker = event.get("ticker")
            exit_id = event.get("exit_id")
            routing = self.routing_for(exit_id)
            entry = watchlist.get(ticker) if watchlist is not None else None
            complete = exit_is_complete(
                event, entry, self.find_confirmation(exit_id)
            )

            if complete:
                note = ""
            elif entry is None:
                note = ORPHANED_EXIT_NOTICE.format(ticker=ticker)
            elif entry.get("state") == lifecycle_states.EXIT_REVIEW:
                note = INCOMPLETE_EXIT_NOTICE.format(ticker=ticker)
            else:
                note = UNMATCHED_EXIT_NOTICE.format(
                    ticker=ticker, state=entry.get("state")
                )

            views.append({
                "exit_id": event.get("exit_id"),
                "ticker": ticker,
                "lane": event.get("lane"),
                "at": event.get("at"),
                "trigger_id": event.get("trigger_id"),
                "friction": event.get("friction"),
                "closed": routing is not None,
                "routed_into": (routing or {}).get("candidate"),
                "deployed_at": (routing or {}).get("deployed_at"),
                "recorded_at": (routing or {}).get("recorded_at"),
                "idle_days": _days_between(
                    event.get("at"),
                    routing["deployed_at"] if routing else as_of,
                ),
                "complete": complete,
                "note": note,
            })
        return views

    def unrouted_views(self, watchlist, as_of=None) -> tuple[list[dict], list[dict]]:
        """Unrouted exits split into the routable and the merely incomplete.

        Returned together because every caller needs both halves to say
        anything honest: the first is what can be deployed, and the second is
        the difference between "nothing to route" and "something only you can
        unblock". Splitting it here rather than at each surface is what stops
        one of them reporting an empty queue while another shows the events in
        it.
        """
        unrouted = [
            view for view in self.exit_views(watchlist, as_of) if not view["closed"]
        ]
        return (
            [view for view in unrouted if view["complete"]],
            [view for view in unrouted if not view["complete"]],
        )

    def routable_exits(self, watchlist, as_of=None) -> list[dict]:
        """Unrouted exits whose sale is complete — the only proceeds to deploy.

        Both filters matter. A routed exit's capital is already somewhere, and
        an unconfirmed one is a sale only this store believes in.
        """
        return self.unrouted_views(watchlist, as_of)[0]

    def propose_routing(self, watchlist, advance_outcomes, concentration,
                        as_of=None) -> dict:
        """Where this run's proceeds should go, what was skipped, and what is idle.

        Inert data (R10). Nothing here transitions a company, and the returned
        candidate is advice: `queue route` records where capital *actually*
        went, which may be somewhere else entirely.

        Ranking is by trigger state first — a `watch` entry whose buy-zone
        trigger just fired outranks a `qualify` with nothing pending, and
        outranks it however good the quiet candidate's composite is, because a
        higher score is not an entry condition being met. Lifecycle state
        breaks ties between candidates in the same trigger state, the composite
        breaks what is left, and the ticker breaks the rest so two runs over
        unchanged inputs agree.

        Skipping is fail-closed on both axes: an unclear `routing_safety`
        reading and a cap that one more name would breach each remove a
        candidate, and **the reasons travel with it**. See the module docstring
        for why the safety reading is read rather than re-derived here.

        The state each candidate is ranked in comes from the **live watchlist**
        rather than the outcome, because a run auto-applies its pre-position
        transitions: a company the loop just moved from `qualify` to `watch`
        should be ranked where it now is, not where it started the run.
        """
        as_of_date = lifecycle_states.as_date(as_of) or date.today()
        idle, incomplete = self.unrouted_views(watchlist, as_of_date)

        ranked, blocked = self._rank_candidates(
            watchlist, advance_outcomes, concentration
        )

        proposal = ranked[0] if ranked else None
        reason = ""
        if not idle:
            # Checked before the ranking is consulted, and reported in place of
            # a candidate: a recommendation with nothing to fund it reads as an
            # instruction to buy. Which sentence goes here is
            # `unroutable_reason`'s decision, never a bare `NO_PROCEEDS` — a
            # half-recorded exit is capital the owner has, and reporting it as
            # an empty queue is the failure this view exists to prevent.
            proposal, reason = None, unroutable_reason(incomplete)
        elif proposal is None and blocked:
            reason = (
                f"every candidate was blocked ({len(blocked)}) — see the "
                f"reasons below rather than reading this as an empty pipeline"
            )
        elif proposal is None:
            reason = (
                "no tracked company is in a state that could receive proceeds "
                "— nothing is ranked, and nothing was blocked"
            )

        return {
            "as_of": str(as_of_date),
            "proposal": proposal,
            "reason": reason,
            "blocked": blocked,
            "idle": idle,
            "incomplete": incomplete,
            "ranked": [candidate["ticker"] for candidate in ranked],
        }

    def _rank_candidates(self, watchlist, advance_outcomes,
                         concentration) -> tuple[list[dict], list[dict]]:
        """Surviving candidates best-first, and the skipped ones with reasons."""
        ranked, blocked = [], []

        for outcome in advance_outcomes or []:
            if not isinstance(outcome, dict):
                continue
            ticker = outcome.get("ticker")
            entry = watchlist.get(ticker) if ticker else None
            # A ticker removed during the run has no lane and no state to rank
            # or to count against a cap. It is not blocked — it is no longer a
            # tracked company at all, and reporting it as skipped would invite
            # someone to go and unblock it.
            if entry is None or entry.get("state") not in CANDIDATE_STATES:
                continue

            fired = outcome.get("proposal") or None
            if fired and fired.get("to") not in lifecycle_states.POSITIONED:
                # A pre-position proposal (qualify, watch, dropped) is not an
                # entry signal, so it must not lift a candidate into the tier
                # reserved for a buy-zone trigger that actually fired.
                fired = None

            candidate = _candidate_payload(outcome, entry, fired)
            # The same question `advance` asks before it applies a transition
            # into a position, asked through the same function. A router that
            # skipped a candidate the transition path was happy to buy would
            # read as a ranking quirk rather than as a guardrail with two minds.
            reasons = _safety_reasons(outcome) + portfolio.would_breach(
                entry.get("lane"), outcome.get("sector"), concentration
            )
            if reasons:
                blocked.append({**candidate, "reasons": reasons})
                continue

            ranked.append(candidate)

        ranked.sort(key=lambda c: (
            0 if c["entry_trigger_fired"] else 1,
            _STATE_PRIORITY.get(c["state"], _UNRANKED_STATE),
            # Negated so a higher composite sorts first; a candidate with no
            # composite ranks below every scored one rather than above them.
            -(c["composite"] if isinstance(c["composite"], (int, float)) else -1.0),
            c["ticker"] or "",
        ))
        return ranked, blocked

    def write_proposal(self, snapshot: dict) -> dict:
        """Replace the stored whole-run routing snapshot, atomically.

        The one write that overwrites rather than appends, because a snapshot
        is a view and not a record. It goes through the same copy-on-write
        commit as every event, so a crashed run leaves the **previous complete
        snapshot** intact rather than a half-written one.

        `queue_revision` is stamped here as the revision this very commit
        produces, not the one the view was generated against. Recording the
        latter would make the snapshot stale the instant it landed — its own
        write having advanced the counter it was being compared to — and every
        `watchlist queue` would report a staleness that nothing caused.
        """
        staged = self._stage()
        stored = copy.deepcopy(snapshot)
        stored["queue_revision"] = _revision_of(self.data) + 1
        staged["latest_proposal"] = stored
        self._commit(staged)
        logger.info(
            f"routing snapshot written ({stored.get('status')}, "
            f"proposal: {(stored.get('proposal') or {}).get('ticker', 'none')})"
        )
        return stored
