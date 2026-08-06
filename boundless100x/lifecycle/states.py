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


def moves_money(to_state: str) -> bool:
    """Whether entering this state commits or withdraws capital."""
    return to_state not in AUTO_APPLICABLE
