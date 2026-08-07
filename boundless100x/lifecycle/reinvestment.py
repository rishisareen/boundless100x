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

**Every commit is copy-on-write, through `atomic_write_json`.** Imported from
`watchlist.py` rather than written a second time — the durability argument is
per-file and identical wherever it applies, and two copies would be two things
to keep in step. A mutator stages onto a deep copy, writes it, and adopts it
only once the write returns. A crash mid-write leaves the previous store rather
than truncated JSON; a failed write leaves `self.data` describing exactly what
is on disk. The second is the more dangerous: a phantom event surviving in
memory would let a same-process retry skip an append it believes already
landed, and the exit would end up recorded in one store only — the precise
disagreement the protocol exists to prevent.

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
carries `NO_PROCEEDS` in place of a candidate. Capital that does not exist
cannot be routed toward one, and a standing recommendation with nothing to fund
it reads as an instruction to buy.

**An unfinished exit record is not proceeds.** An exit event whose ticker still
sits in `exit_review` is KTD10's crash window — the queue event landed and the
transition did not. It is reported with the command that completes it and
excluded from routing until it is, both here and in `queue route`, so the
direct command cannot bypass the display's exclusion.

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
from boundless100x.watchlist import APPLIED_OWNER, atomic_write_json

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path(__file__).parent / "reinvestment_queue.json"

EXIT_EVENT = "exit"
ROUTING_EVENT = "routing"
EVENT_KINDS = (EXIT_EVENT, ROUTING_EVENT)

# The one sentence for "there is nothing to route". Shared between the view and
# every surface that renders it, so the display and the `queue route` refusal
# cannot drift into saying different things about the same emptiness.
NO_PROCEEDS = "No exit proceeds awaiting routing"

# KTD10's crash window, stated with the command that closes it. An owner
# meeting this line needs the recovery step in the same breath, not a
# description of a state they did not know existed.
INCOMPLETE_EXIT_NOTICE = (
    "Exit recording incomplete — run `watchlist exit {ticker}` to complete it"
)

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
}


class ReinvestmentError(ValueError):
    """A stored event does not match the schema, or an append was refused."""


def _now() -> str:
    return datetime.now().isoformat()


def _as_date(value) -> date | None:
    """A calendar date from whatever the stores happen to hold, or None.

    Exit dates are written as `str(as_of)` and transition timestamps as full
    ISO datetimes, so an idle reading spans two differently shaped strings. Day
    granularity is the honest resolution for both: the exit date is the day the
    owner says they sold, not a fill time.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _days_between(start, end) -> int | None:
    """Whole days from one stored timestamp to another, or None if unreadable.

    None rather than zero, for the reason the whole codebase treats gaps this
    way: a zero-day idle reading means the proceeds were redeployed the same
    day, and an unreadable one means nobody can say. In a table those look
    identical.
    """
    first, second = _as_date(start), _as_date(end)
    if first is None or second is None:
        return None
    return (second - first).days


def exit_is_complete(entry: dict | None) -> bool:
    """Whether the watchlist confirms the sale this exit event records.

    Two conditions, and the second is KTD10's crash window: the entry must hold
    an `exited` transition, and it must not still be sitting in `exit_review`.
    An event whose entry never left the review is a sale the watchlist has not
    agreed to — `confirm_exit` writes the queue first precisely so that this
    disagreement is visible rather than lost, and routing proceeds that only
    one store believes in is what the visibility is for.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("state") == lifecycle_states.EXIT_REVIEW:
        return False
    return any(
        isinstance(record, dict) and record.get("to") == lifecycle_states.EXITED
        for record in entry.get("state_history") or []
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
    sold_on = _as_date(exit_at)
    matches = []
    for record in (entry or {}).get("state_history") or []:
        if not isinstance(record, dict):
            continue
        if record.get("to") not in lifecycle_states.POSITIONED:
            continue
        if record.get("applied_by") != APPLIED_OWNER:
            continue
        moved_on = _as_date(record.get("at"))
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


def _concentration_reasons(lane: str, sector, reading: dict | None) -> list[str]:
    """Why adding one more name to this lane or sector would breach a cap.

    Every figure consulted here is a **count of positioned names**, never a
    share of capital — `lifecycle/portfolio.py` argues why that is the only
    honest guardrail this system can compute.

    A reading that could not be built blocks everything. The alternative is
    proposing capital into a lane whose occupancy is unknown, which is the
    failure mode `portfolio.unavailable` exists to make visible: absence reads
    as headroom.

    **A lane with no configured cap blocks for the same reason.**
    `portfolio._lane_counts` reports it honestly — `max: None`, "counted, not
    checked" — and that honesty is precisely what the router must not read as
    room. Treated as a pass, the one lane nobody had got round to configuring
    became the one lane capital could always flow into, which inverts the
    guardrail. Zero is a cap, not a gap: `portfolio._cap` allows it because
    "hold nothing in this lane" is a real instruction, and it blocks on the cap
    it breaches rather than on missing configuration.

    The sector half is deliberately partial and says so. `check_concentration`
    reports groups of two or more, so a candidate joining a sector that
    currently holds one positioned name is invisible here. That is the group
    size the cap is nowhere near, and reconstructing the full sector census
    would mean the router keeping its own copy of a count the reading already
    owns.
    """
    if not isinstance(reading, dict) or not reading.get("available"):
        detail = (reading or {}).get("reason", "no reading was produced")
        return [
            f"the concentration reading is unavailable ({detail}) — routing "
            f"cannot confirm the {lane} lane has room"
        ]

    reasons = []
    lane_row = (reading.get("lanes") or {}).get(lane)
    if not isinstance(lane_row, dict):
        reasons.append(
            f"the concentration reading describes no {lane!r} lane, so its "
            f"occupancy is unknown — routing is refused rather than assumed"
        )
    else:
        cap = lane_row.get("max")
        held = lane_row.get("positioned", 0)
        if cap is None:
            reasons.append(
                f"the {lane} lane holds {held} positioned name(s) and has no cap "
                f"configured (portfolio.max_positioned_per_lane[{lane}]) — there "
                f"is no limit to check one more against, so routing is refused "
                f"rather than assumed"
            )
        elif held + 1 > cap:
            reasons.append(
                f"the {lane} lane already holds {held} of a maximum {cap} "
                f"positioned name(s) — one more would breach the cap "
                f"(counts of names, not a share of capital)"
            )

    key = _sector_key(sector)
    if key:
        for group in reading.get("sectors") or []:
            cap = group.get("max")
            if _sector_key(group.get("sector")) != key or cap is None:
                continue
            if group.get("count", 0) + 1 > cap:
                reasons.append(
                    f"the {group['sector']} sector already holds "
                    f"{group['count']} positioned name(s) against a cap of "
                    f"{cap} ({', '.join(group.get('tickers') or [])}) — counts "
                    f"of names, not a share of capital"
                )
    return reasons


def _sector_key(sector) -> str:
    """The same folding `check_concentration` grouped by, borrowed not rewritten.

    The router matches a candidate's sector against an already-reported group,
    so both sides of the comparison must fold identically. A second
    implementation here would eventually differ, and "Chemicals" would read as
    two sectors on one side and one on the other — a cap check that silently
    stops matching.
    """
    return portfolio._sector_key(sector)


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


def _revision_of(data: dict) -> int:
    """The store's commit counter, defaulting to zero for a store without one.

    Absent on a file written before the counter existed, and hand-editable into
    nonsense like anything else on disk — either way it restarts from zero
    rather than raising. A missing revision is a staleness signal nobody can
    read yet, not a corrupt queue. Mirrors `watchlist._revision_of` for the
    same reason the write helper is shared: one argument, one behaviour.
    """
    revision = data.get("revision", 0)
    if not isinstance(revision, int) or revision < 0:
        return 0
    return revision


class ReinvestmentQueue:
    """Reads and writes the exit / routing event log."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else DEFAULT_QUEUE_PATH
        self.data = self._load()

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

    def _stage(self) -> dict:
        """A deep copy of the store, safe to mutate before anything is committed."""
        return copy.deepcopy(self.data)

    def _commit(self, staged: dict) -> None:
        """Persist a staged store, then adopt it — never the other way round.

        The revision bumps here and nowhere else, so it counts durable commits
        rather than attempts. A reader comparing revisions to decide whether
        its view is current would otherwise be told a change happened that the
        store never took.
        """
        staged["revision"] = _revision_of(self.data) + 1
        atomic_write_json(self.path, staged)
        self.data = staged

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

        Each view also states whether the watchlist agrees the sale completed.
        An event stranded in KTD10's crash window carries the recovery command
        in its own `note`, because whoever meets the line is the person who has
        to run it.
        """
        as_of = _as_date(as_of) or date.today()
        views = []
        for event in self.exits():
            ticker = event.get("ticker")
            routing = self.routing_for(event.get("exit_id"))
            entry = watchlist.get(ticker) if watchlist is not None else None
            complete = exit_is_complete(entry)

            if complete:
                note = ""
            elif entry is None:
                note = (
                    f"{ticker} is no longer on the watchlist, so this exit "
                    f"cannot be confirmed against a lifecycle record"
                )
            else:
                note = INCOMPLETE_EXIT_NOTICE.format(ticker=ticker)

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

    def routable_exits(self, watchlist, as_of=None) -> list[dict]:
        """Unrouted exits the watchlist confirms — the only proceeds to deploy.

        Both filters matter. A routed exit's capital is already somewhere, and
        an unconfirmed one is a sale only this store believes in.
        """
        return [
            view for view in self.exit_views(watchlist, as_of)
            if not view["closed"] and view["complete"]
        ]

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
        as_of_date = _as_date(as_of) or date.today()
        views = self.exit_views(watchlist, as_of_date)
        unrouted = [view for view in views if not view["closed"]]
        idle = [view for view in unrouted if view["complete"]]
        incomplete = [view for view in unrouted if not view["complete"]]

        ranked, blocked = self._rank_candidates(
            watchlist, advance_outcomes, concentration
        )

        proposal = ranked[0] if ranked else None
        reason = ""
        if not idle:
            # Checked before the ranking is consulted, and reported in place of
            # a candidate: a recommendation with nothing to fund it reads as an
            # instruction to buy.
            proposal, reason = None, NO_PROCEEDS
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
            reasons = _safety_reasons(outcome) + _concentration_reasons(
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
