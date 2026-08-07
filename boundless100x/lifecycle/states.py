"""The lifecycle state machine.

States are ordered by commitment, not by time: a company moves rightward as
the evidence for holding it strengthens, and leaves for `exit_review` from
anywhere once a kill-switch fires.

    screen → qualify → watch → probe → scale
                 │        │       │       │
                 ▼        ▼       ▼       ▼
              dropped   dropped  exit_review → exited

The split that matters is not which state is "further along" but which
transitions **move money**. Entering `probe` or `scale` deploys capital;
entering `exit_review` or `exited` withdraws it. Those are proposed with
evidence and confirmed by the owner (v05 §14.4). Everything before a position
exists — qualifying, watching, dropping a candidate — moves no money and may
apply automatically.

This module also owns the two readings of `state_history` that every consumer
of it needs — *which record entered a state* and *what day a stored timestamp
means*. Both live here for the same reason: `state_history` is the append-only
record the whole lifecycle argues from, and a helper that reads it must agree
with itself across every surface that reads it. Neither needs anything outside
the standard library, which is what lets `evaluator`, `friction`,
`reinvestment`, `advance`, `exit` and `lane_view` all reach them without any of
those modules importing each other.
"""

from datetime import date, datetime

SCREEN = "screen"
QUALIFY = "qualify"
WATCH = "watch"
PROBE = "probe"
SCALE = "scale"
EXIT_REVIEW = "exit_review"
EXITED = "exited"
DROPPED = "dropped"

STATES = (
    SCREEN,
    QUALIFY,
    WATCH,
    PROBE,
    SCALE,
    EXIT_REVIEW,
    EXITED,
    DROPPED,
)

# States a company can be in while capital is committed. Reaching or leaving
# one of these is the owner's decision, never the system's.
POSITIONED = frozenset({PROBE, SCALE})

# Destination states an `advance` run may apply on its own. Everything else is
# proposed and waits for confirmation.
AUTO_APPLICABLE = frozenset({QUALIFY, WATCH, DROPPED})

# The state a newly tracked company starts in. Nothing is granted on entry —
# qualification is earned by evaluation, not by being added to the watchlist.
INITIAL = SCREEN


# ── The rest of the lifecycle's vocabulary ──
#
# Lanes, who applied a transition, and the status of an owner-recorded
# catalyst. All three were defined in `watchlist.py`, which is where the store
# is rather than where the meaning is, and six lifecycle modules reached back
# into it for them — making `boundless100x.watchlist` and this package mutually
# dependent, latent only because `lifecycle/__init__.py` is a bare docstring.
#
# They are vocabulary, not storage. A lane is how a company is judged; an
# `applied_by` is what a transition record means; a catalyst status is a
# condition the evaluator reads. Every one of them is a fact about the
# lifecycle that the watchlist merely happens to persist, so they belong beside
# the states in the module every layer can already reach. `watchlist.py`
# re-exports them, so the name it published still resolves.

# §4.4's two lanes: the same state machine, two parameter sets.
CORE_LANE = "core"
RERATING_LANE = "rerating"
LANES = (CORE_LANE, RERATING_LANE)

# Who applied a transition. The distinction is load-bearing wherever money is
# involved: `reinvestment.eligible_deployments` counts only owner-applied
# `probe`/`scale` transitions, because an auto-applied one moves no capital.
APPLIED_AUTO = "auto"
APPLIED_OWNER = "owner"

# The owner-recorded catalyst the fast lane gates entry on. `spent` rather than
# deleted: a position whose catalyst was spent without the re-rating following
# is exactly the case worth being able to see.
CATALYST_ACTIVE = "active"
CATALYST_SPENT = "spent"
CATALYST_STATUSES = (CATALYST_ACTIVE, CATALYST_SPENT)


def is_state(value: object) -> bool:
    return isinstance(value, str) and value in STATES


def as_date(value) -> date | None:
    """A calendar date from whatever a caller or a store happened to hold.

    **The one parser for lifecycle timestamps**, and it is here because the
    thing being parsed is `state_history`. Three modules used to carry a copy
    of this — `evaluator` for the time stop, `friction` for a holding period,
    `reinvestment` for an idle-day count — each documenting the reconciliation
    as though it were the only one, and they did not agree. The same stored
    `at` could therefore make a time stop read indeterminate while an idle
    reading beside it printed a confident number, which is the one kind of
    disagreement nobody looking at either surface could see.

    Two shapes genuinely arrive. `as_of` is a `date`, as it is throughout
    `lifecycle.checkpoints`; a `state_history` record's `at` is a full ISO
    datetime, because `watchlist._now()` writes `datetime.now()`. Exit dates
    are written as `str(as_of)`, so they are the first shape spelled as the
    second. Timestamps normalize to days because a market bar has no time of
    day, and pretending otherwise would put a spurious few hours into a holding
    period that decides a tax bracket.

    **A string must parse whole.** The lenient variant this replaces read the
    first ten characters and discarded the rest, so `"2026-08-07 chaos"` came
    back as a date — that is the divergence, and it is resolved toward the
    strict reading rather than the lenient one, because an unreadable timestamp
    is a gap and the whole layer's rule is that a gap says so. Unreadable comes
    back None so the caller can name *what* it could not read, rather than
    quietly becoming a date nobody supplied.

    No pandas branch is needed and none is written: `pandas.Timestamp`
    subclasses `datetime.datetime`, so the first check already covers a value
    lifted straight out of a price frame. Keeping the standard library the only
    import is what lets every layer read this one.
    """
    if isinstance(value, datetime):  # checked first — datetime subclasses date
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def last_record_into(records, to_state: str) -> dict | None:
    """The most recent record entering a state, from a bare history list.

    The **last** match, not the first, and the rule is load-bearing enough to
    live in one place: history is append-only and in order, so a position
    re-entered after an earlier stint restarts the clock. Dating a holding
    period from a stint that already ended would put it in the wrong tax
    bracket, keying an `exit_id` on an old review would collide with an exit
    already recorded, and timing a stop from a previous visit could end a
    position months before its clock actually ran out.

    Stated here rather than in any one caller because `exit.py`, `advance.py`,
    `lane_view.py` and the evaluator's time stop all depend on it agreeing with
    itself — a report showing a different holding period than the transition
    that recorded it would be a disagreement nobody could see.

    Takes the record list rather than the entry so the evaluator, which is
    handed `state_history` on its own and never sees an entry, reads the same
    rule unwrapped instead of fabricating a one-key dict to get at it.
    """
    matches = [
        record
        for record in records or []
        if isinstance(record, dict) and record.get("to") == to_state
    ]
    return matches[-1] if matches else None


def last_transition_into(entry: dict, to_state: str) -> dict | None:
    """`last_record_into`, for the callers that hold a whole watchlist entry."""
    return last_record_into(entry.get("state_history"), to_state)


def moves_money(to_state: str) -> bool:
    """Whether entering this state commits or withdraws capital."""
    return to_state not in AUTO_APPLICABLE
