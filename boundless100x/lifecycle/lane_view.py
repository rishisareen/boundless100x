"""What a surface needs to say about a tracked company's lane.

**Pure, and shared, and that combination is the point.** Two surfaces need the
same four facts — lane, state, catalyst, and the friction reading appropriate
to where the company stands — and only one of them ever runs `advance()`. The
report is built by `analyze`, which re-scores and renders without touching the
lifecycle loop at all; a figure that only an advance outcome could carry would
simply be missing there, and a report showing a position's lane but not its
modeled friction is the half that leaves an owner reading a gross return
somewhere else. So this takes a stored entry and a scored result, computes
nothing that needs a network or a store write, and hands back a dict.

`advance_ticker` passes the `lane_gate_result` it has already paid for; the CLI
calls this fresh and lets it evaluate. Same output either way, which is what
makes it safe for the two surfaces to disagree about who computed what.

Three rules are inherited rather than reinvented.

**An exited position reports what was recorded.** The `exited` transition
carries its friction payload as structured `details` (that is why the field
exists), and the recorded payload is returned verbatim — never a fresh model.
Re-pricing a sale that already happened against today's bars would make the
number drift every time the report was regenerated, and would leave the report
disagreeing with the queue event written at the same moment.

**A gap is unavailable with its reason, never a zero and never silence.** The
distinction this module has to keep is three-way: *absent* means there is no
modeled position at all (nothing was ever bought), *unavailable* means there is
one and nobody could price it, and a number means it was priced. Collapsing the
first two would make an untouched candidate read like a broken data feed, and
collapsing either into a zero would make both read like a position that went
nowhere.

**An overdue catalyst is a display flag.** §13 keeps the system advisory: the
clock feeds the time stop and nothing else. Noticing that a window has passed
must not propose a transition, and this function writes nothing regardless.
"""

import logging
from datetime import date, datetime

from boundless100x.lifecycle import friction as friction_module
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.lane_gates import LaneGateEvaluator
from boundless100x.lifecycle.states import CATALYST_ACTIVE, RERATING_LANE

logger = logging.getLogger(__name__)


def build_lane_context(
    entry: dict | None,
    result,
    as_of=None,
    lane_gate_result: dict | None = None,
    config: dict | None = None,
    friction_estimate: dict | None = None,
) -> dict | None:
    """Everything a surface renders about one tracked company's lane.

    Returns None for a company that is not tracked — the same answer the
    calling surface's own membership check gives, stated here so a caller that
    forgets to ask cannot render a lane section for a company with no lane.

    `config` is the pipeline config (or a `friction:` block); the tax and
    slippage rates it resolves travel in the context because the break-even
    line lists them, and a rendered assumption must be the one that was
    actually applied rather than a plausible default typed into a template.

    `friction_estimate` is the same already-computed seam `lane_gate_result` is,
    and for the same reason: on an exit-proposing ticker `advance_ticker` has
    already modeled this exact reading from these exact arguments, and modeling
    it twice means rebuilding a frame over the whole daily price series to
    reach an answer already in hand — or, worse, a *different* answer if
    anything underneath moved between the two passes. It is used only where a
    reading would otherwise be computed, so a caller that supplies one for a
    company with no modeled position still gets None.
    """
    if not entry:
        return None

    return {
        "lane": entry.get("lane"),
        "state": entry.get("state"),
        "as_of": str(as_of or date.today()),
        "catalyst": _catalyst_view(entry, as_of),
        "lane_gates": _lane_gates(entry, result, lane_gate_result),
        "friction": _friction(entry, result, as_of, config, friction_estimate),
        "friction_assumptions": friction_module.config_from(config),
    }


def _lane_gates(entry: dict, result, lane_gate_result: dict | None) -> dict | None:
    """The six fast-lane entry gates, for a fast-lane entry only.

    A caller-supplied result wins outright: `advance_ticker` has already
    evaluated it against the same readings, and evaluating again would spend a
    second YAML parse to produce the same answer — or, worse, a different one
    if the registry changed mid-run.

    A core entry gets None rather than an empty result. The gates are the fast
    lane's own question, and rendering six blanks against a core company would
    invite reading them as six unmet conditions.
    """
    if entry.get("lane") != RERATING_LANE:
        return None
    if lane_gate_result is not None:
        return lane_gate_result

    try:
        return LaneGateEvaluator().evaluate(
            getattr(result, "metrics", None) or {},
            getattr(result, "scores", None),
            # `{}` rather than a bare `.get("catalyst")`, for the reason
            # `lane_gates._evaluate_catalyst` spells out: an entry somebody has
            # looked at and which carries no catalyst is a plain failure, while
            # None means no watchlist context was supplied at all and reads
            # indeterminate. Both are falsy, so the default keeps them apart.
            entry.get("catalyst", {}),
        )
    except Exception as e:
        # A malformed registry must not cost a report that is otherwise fine.
        # The section simply carries no gates, which reads as "not shown"
        # rather than as a verdict nobody reached.
        logger.warning(f"The fast-lane gates could not be evaluated: {e}")
        return None


def _catalyst_view(entry: dict, as_of) -> dict | None:
    """The recorded catalyst plus whether its window has passed.

    A copy, not the stored dict: the entry belongs to the watchlist store, and
    a view that added a key to it would be editing an owner's record from a
    render path.
    """
    catalyst = entry.get("catalyst")
    if not catalyst:
        return None
    return {**catalyst, "overdue": _is_overdue(catalyst, as_of)}


def _is_overdue(catalyst: dict, as_of) -> bool:
    """Whether an *active* catalyst's expected window is behind us.

    Two things are deliberately not overdue. A **spent** catalyst happened, and
    the date it happened after is not a warning about anything. And a window
    nobody can parse — `expected_by` is owner free text, so "H2 FY27" is a
    perfectly ordinary value — is unknown, not passed: guessing that an
    unreadable window has expired would put a red flag on a thesis on the
    strength of a formatting choice.
    """
    if catalyst.get("status") != CATALYST_ACTIVE:
        return False

    raw = catalyst.get("expected_by")
    try:
        window = datetime.fromisoformat(str(raw)).date()
    except (TypeError, ValueError):
        return False

    return window < (as_of or date.today())


def _friction(entry: dict, result, as_of, config, estimate=None) -> dict | None:
    """The reading that fits where this company stands, or None if none does.

    Three states of the world, kept apart on purpose:

      * **exited** — the payload recorded on the transition, verbatim;
      * **positioned or under exit review** — a fresh estimate from the last
        `probe` confirmation to `as_of`, whose exit end is still moving, hence
        `basis: estimate`;
      * **anything earlier** — None. No capital was ever committed, so there is
        no modeled position to price, and that is a different fact from one
        that could not be priced.

    The state dispatch happens *before* any supplied `estimate` is consulted,
    which is what keeps the third case honest: a caller that modeled an exit for
    a company sitting at `watch` — possible, since a kill-switch can propose an
    exit review from anywhere its `from` list allows — still gets None here,
    because whether there is a position to report is this function's question
    and not the caller's.
    """
    state = entry.get("state")

    if state == lifecycle_states.EXITED:
        return _recorded_reading(entry)

    if state in lifecycle_states.POSITIONED or state == lifecycle_states.EXIT_REVIEW:
        return estimate if estimate is not None else _estimated_reading(
            entry, result, as_of, config
        )

    return None


def _recorded_reading(entry: dict) -> dict:
    """The friction payload the exit was recorded with — never a new one.

    `confirm_exit` writes the same payload to the queue event and to the
    transition's `details`, so this is the object of record for that sale.
    Recomputing it here would produce a number that moves with the market long
    after the position was closed, and the report would then disagree with the
    queue about a single event.

    An exit recorded without a payload says so rather than filling the gap: the
    figure that belongs there is the one from the day of the sale, and it no
    longer exists to be read.
    """
    record = lifecycle_states.last_transition_into(entry, lifecycle_states.EXITED)
    details = (record or {}).get("details")
    if isinstance(details, dict) and details:
        return dict(details)

    return {
        **friction_module.unavailable(
            "this exit was recorded without a friction payload — the figures "
            "belong to the day of the sale and re-pricing it now would report "
            "a different number than the one it was recorded at"
        ),
        "basis": friction_module.BASIS_RECORDED,
    }


def _estimated_reading(entry: dict, result, as_of, config) -> dict | None:
    """An in-flight reading: last `probe` confirmation → `as_of`.

    `friction.reading_for_exit` does the work, which is the whole point of that
    helper existing: this and `advance._friction_for_exit` are the same reading
    of the same position, and they used to be two copies of it. A position with
    no recorded `probe` is None either way — no modeled holding period, and
    inventing one from the day the company was added to the watchlist would
    date a tax bracket off an administrative act — and any failure becomes
    unavailable-with-reason, because a report is worth more than the reading it
    could not take.
    """
    return friction_module.reading_for_exit(
        entry,
        (getattr(result, "data", None) or {}).get("price"),
        as_of or date.today(),
        config=config,
        basis=friction_module.BASIS_ESTIMATE,
    )
