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
"""

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


def is_state(value: object) -> bool:
    return isinstance(value, str) and value in STATES


def last_transition_into(entry: dict, to_state: str) -> dict | None:
    """The most recent record entering a state, or None.

    The **last** match, not the first, and the rule is load-bearing enough to
    live in one place: history is append-only and in order, so a position
    re-entered after an earlier stint restarts the clock. Dating a holding
    period from a stint that already ended would put it in the wrong tax
    bracket, and keying an `exit_id` on an old review would collide with an
    exit already recorded.

    Stated here rather than in any one caller because `exit.py`, `advance.py`
    and `lane_view.py` all depend on it agreeing with itself — a report showing
    a different holding period than the transition that recorded it would be a
    disagreement nobody could see.
    """
    records = [
        record
        for record in entry.get("state_history") or []
        if isinstance(record, dict) and record.get("to") == to_state
    ]
    return records[-1] if records else None


def moves_money(to_state: str) -> bool:
    """Whether entering this state commits or withdraws capital."""
    return to_state not in AUTO_APPLICABLE
