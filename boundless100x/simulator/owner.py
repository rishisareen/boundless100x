"""The simulated owner: every human input the production lifecycle expects,
stated as policy (KTD3, KTD6).

§14.4's production lifecycle proposes money-moving transitions and waits for
a person to confirm them, record a catalyst, or route an exit's proceeds.
The replay has no person, so it substitutes this module — a set of **pure**
policy functions, no I/O, reading only `boundless100x/config.yaml`'s
`simulator:` block (`config_from`) and the arguments each function is
handed.

**What this module is not.** `advance.decide()` (`lifecycle/advance.py`) is
production's single statement of "given these readings, what should happen
to this company next" — it runs `TriggerEvaluator`/`LaneGateEvaluator`,
resolves precedence between competing triggers, derives kill-switch status,
computes a friction estimate on an exit proposal, and asks the concentration
gate. A later unit (U7) calls `decide()` once per ticker per replay date and
hands **this module's** functions the `proposal` dict it already produced.
So nothing here imports `TriggerEvaluator`, `LaneGateEvaluator`, or
`lifecycle.advance.decide`/`advance_ticker`, and nothing here re-derives a
kill-switch, a precedence ranking, or a concentration breach — restating any
of that would be exactly the second statement of the trigger rules KTD1
exists to forbid ("a simulator with its own copy of the trigger rules would
prove something about *those* rules"). This module's job is strictly the
layer *on top* of an already-produced proposal: given it, decide **when**
(or whether) it becomes a confirmed action.

One consequence: the cap-posture decision (Session-settled decision 5) is
not something this module re-checks. `advance.decide()` already asks
`concentration_gate(lane, sector)` and withholds a cap-breaching transition
unless `override_caps=True` is passed in. `override_caps_for` is a pure
config-to-boolean mapping a caller (U7) applies **before** invoking
`advance.decide()` — it is not a second cap check inside this module's own
`decide()`.

Four seams, one per human input production expects:

  * `decide(proposal, as_of, config)` — confirm-after for a money-moving
    proposal (`probe`/`scale`/`exit_review`), with the severity/sell-fraction
    an `exit_review` needs attached (§14.3, R6).
  * `catalyst_for(candidate, gate_result, config, as_of=...)` — fabricates a
    synthetic fast-lane catalyst once the other five lane gates have
    cleared (KTD6), self-describing as simulated so a downstream limitations
    block can name it.
  * `route(exit_event, ranked_candidates, as_of, config)` — the routing lag
    over production's own already-ranked candidate list
    (`reinvestment.propose_routing`/`_rank_candidates`); this module never
    re-ranks, only decides when the top-ranked candidate is accepted.
  * `override_caps_for(posture)` — decision 5's config-to-boolean mapping.

Every returned decision is a plain, JSON-serialisable dict — Phase 5's
sweeps and U6's output artifact both read these programmatically, and KTD3
requires every simulated-owner policy to be recorded into the run's own
artifact verbatim.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from boundless100x.lifecycle.states import EXIT_REVIEW, PROBE, SCALE, as_date
from boundless100x.simulator.replay import FAST_LANE_ENTRY_GATES

logger = logging.getLogger(__name__)


# ── shipped defaults, mirroring config.yaml's `simulator:` block ──────────
#
# The `friction.config_from`/`portfolio.config_from` idiom: a module-level
# constant per setting, read by `config_from` so a caller supplying no
# config (a test, a direct call) sees the same numbers the CLI would.

DEFAULT_STARTING_POOL = 100
DEFAULT_CONFIRMATION_LAG_DAYS = {"entry": 5, "exit": 2, "route": 5}
DEFAULT_CATALYST_WINDOW_MONTHS = 6

POSTURE_ENFORCED = "enforced"
POSTURE_ADVISORY = "advisory"
POSTURE_OVERRIDE = "override"
CAP_POSTURES = (POSTURE_ENFORCED, POSTURE_ADVISORY, POSTURE_OVERRIDE)
DEFAULT_CAP_POSTURE = POSTURE_ENFORCED

# §14.3's severity vocabulary.
SEVERITY_FULL_EXIT = "full_exit"
SEVERITY_REDUCE = "reduce"
SEVERITY_REVIEW = "review"
SEVERITIES = (SEVERITY_FULL_EXIT, SEVERITY_REDUCE, SEVERITY_REVIEW)

# The trigger ids that can propose `exit_review` (`triggers.yaml`): the six
# universal fundamentals kill-switches plus the three fast-lane thesis
# exits. Governance is the one true full exit — a pledge crossing its
# red-flag threshold is thesis-level invalidation, not a metric that merely
# worsens. Valuation saturation is tagged `reduce` per §14.3's third value,
# even though decision 4 leaves the fraction unsettled (see
# `sell_fraction_for`) — the tag alone is what lets the affected sample be
# counted separately once the owner does settle one. Every other kill-switch
# and every fast-lane thesis exit resolves to `review`, R6's placeholder
# ("everything else = exit review followed by simulated confirmation") — a
# scheduled exit confirmation after the exit lag, with no partial-sale
# destination.
#
# `severity_for` falls back to `SEVERITY_REVIEW` for any trigger id not
# listed here, so a `triggers.yaml` that gains a new kill-switch later does
# not crash the simulator on its next replay.
SEVERITY_MAP = {
    "governance_event": SEVERITY_FULL_EXIT,
    "valuation_saturation": SEVERITY_REDUCE,
    "capital_efficiency_break": SEVERITY_REVIEW,
    "growth_quality_degradation": SEVERITY_REVIEW,
    "incremental_return_break": SEVERITY_REVIEW,
    "checkpoints_failed": SEVERITY_REVIEW,
    "fast_lane_target_reached": SEVERITY_REVIEW,
    "fast_lane_time_stop": SEVERITY_REVIEW,
    "fast_lane_catalyst_spent": SEVERITY_REVIEW,
}

# Destinations `decide()` treats as money-moving proposals worth scheduling.
# `EXIT_REVIEW` is the only exit-bound one; `exited` itself is never an
# `advance()`/`decide()` proposal target (see `lifecycle/advance.py`), so it
# is deliberately absent here too.
_ENTRY_STATES = (PROBE, SCALE)


# ── config resolution ──────────────────────────────────────────────────


def config_from(config: dict | None) -> dict:
    """Owner settings for the simulator's policy block, with shipped defaults.

    Accepts either the whole pipeline config (`config_from(service.config)`)
    or the `simulator:` block alone — the `friction.config_from`/
    `portfolio.config_from` idiom, for the same reason both call sites are
    natural and a caller passing the wrong one must not silently get shipped
    defaults presented as the owner's own settings.

    `reduce_fraction` is never defaulted (decision 4, deferred): the
    returned value is whatever the owner configured, or `None` when they
    have not — never `DEFAULT_...`-style substitution. A config that
    invented a fraction here would read as settled when it is not.

    `severity_overrides` is an optional, undocumented-in-config-yaml escape
    hatch for Phase 5: `simulator.severity_overrides: {trigger_id:
    severity}` lets a sweep test a different §14.3 mapping without editing
    `SEVERITY_MAP`. It is not one of the four settled values (Session-settled
    decisions 1-5) and the shipped `config.yaml` carries none — absent, it
    resolves to `{}` and `severity_for` reads the module constant unchanged.
    """
    config = config or {}
    section = config.get("simulator") if "simulator" in config else config
    section = section or {}

    configured_lags = section.get("confirmation_lag_days") or {}
    lags = {
        **DEFAULT_CONFIRMATION_LAG_DAYS,
        **{k: v for k, v in configured_lags.items() if k in DEFAULT_CONFIRMATION_LAG_DAYS},
    }

    return {
        "starting_pool": section.get("starting_pool", DEFAULT_STARTING_POOL),
        "confirmation_lag_days": lags,
        "catalyst_window_months": section.get(
            "catalyst_window_months", DEFAULT_CATALYST_WINDOW_MONTHS
        ),
        "cap_posture": section.get("cap_posture", DEFAULT_CAP_POSTURE),
        # Deliberately not defaulted — see the docstring above.
        "reduce_fraction": section.get("reduce_fraction"),
        "severity_overrides": section.get("severity_overrides") or {},
    }


# ── decision 5: cap posture -> advance.decide()'s override_caps ───────────


def override_caps_for(posture: str) -> bool:
    """The pure mapping from a cap posture (decision 5) to
    `advance.decide()`'s own `override_caps: bool`.

    `enforced` -> `False` (a cap-breaching transition is withheld, exactly
    what `advance --apply` runs under today). `advisory`/`override` ->
    `True` (the transition proceeds and `advance.decide()` writes the
    breach into the proposal's own evidence — it already does this
    unconditionally when `override_caps=True`, so there is nothing further
    for this module to record).

    This function does not itself withhold or apply anything, and it is not
    a second cap check: `advance.decide()`'s `concentration_gate` +
    `override_caps` parameters are what act on the boolean returned here. A
    caller (U7) resolves the posture once per run and passes the result
    straight into `advance.decide(..., override_caps=override_caps_for(...))`.

    An unrecognised posture fails **closed** — `False`, i.e. `enforced` —
    with a logged warning, matching this layer's "absence must not read as
    headroom" rule (`lifecycle/portfolio.py`'s docstring states the same
    rule for a concentration reading that could not be built): a typo in
    config must tighten the guardrail, never loosen it.
    """
    if posture == POSTURE_ENFORCED:
        return False
    if posture in (POSTURE_ADVISORY, POSTURE_OVERRIDE):
        return True
    logger.warning(
        f"owner.override_caps_for: unrecognised cap posture {posture!r} "
        f"(expected one of {CAP_POSTURES}) — failing closed to "
        f"{POSTURE_ENFORCED!r} (override_caps=False) rather than reading an "
        f"unknown value as headroom"
    )
    return False


# ── §14.3 severity mapping ─────────────────────────────────────────────


def severity_for(trigger_id: str, config: dict | None = None) -> str:
    """Which kind of exit this `exit_review`-bound trigger resolves to (R6).

    `SEVERITY_MAP` is the base statement — a Python constant rather than a
    config default, because assigning a kill-switch's severity is a
    modelling choice about the *rule*, not a per-run preference, and none of
    the four settled simulator config values (Session-settled decisions 1-5)
    is a severity map. `config`'s `simulator.severity_overrides` (see
    `config_from`) may override per trigger id for a Phase 5 sweep.

    A trigger id neither the override nor `SEVERITY_MAP` recognises resolves
    to `"review"` — the safe, documented default — rather than raising: a
    `triggers.yaml` that gains a new kill-switch must not crash the next
    replay on account of it.
    """
    overrides = config_from(config).get("severity_overrides") or {}
    if trigger_id in overrides:
        return overrides[trigger_id]
    return SEVERITY_MAP.get(trigger_id, SEVERITY_REVIEW)


def sell_fraction_for(trigger_id: str, config: dict | None = None) -> float:
    """The fraction of a position this trigger's exit sells, in the baseline.

    `"full_exit"` and `"review"` both resolve to `1.0`: R6 states plainly
    that `"review"` is "exit review followed by simulated confirmation" —
    mechanically a full exit today, since there is no partial-sale
    destination for it in this system.

    `"reduce"` resolves to the configured `reduce_fraction` **only if one is
    configured** (decision 4 is owner-deferred, so the baseline has none),
    and to `1.0` otherwise — the "ships built but inactive" behaviour R6 and
    decision 4 both describe: a `valuation_saturation` exit in the baseline
    still sells everything, but stays tagged `"reduce"` so it can be counted
    separately once the owner settles a fraction ("the affected sample is
    visible").
    """
    severity = severity_for(trigger_id, config)
    if severity == SEVERITY_REDUCE:
        fraction = config_from(config).get("reduce_fraction")
        if fraction is not None:
            return float(fraction)
    return 1.0


# ── trading-day arithmetic ─────────────────────────────────────────────


def _advance_trading_days(start: date, n: int) -> date:
    """`start` plus `n` trading days (Mon-Fri; no holiday calendar — the
    corpus's own bars are the finer-grained source of holidays, and this
    helper only ever schedules a *later* confirmation date against them).

    A `start` that does not itself fall on a trading day is normalized
    forward to the next one first (`pandas.bdate_range`'s own behaviour),
    and `n` trading days are then added from there — so a proposal or exit
    event dated on a weekend still yields one unambiguous, strictly-later
    confirmation date rather than raising or silently collapsing `n` days of
    lag into fewer calendar days than intended.
    """
    if n < 0:
        raise ValueError(f"_advance_trading_days: n must be >= 0, got {n}")
    return pd.bdate_range(start=pd.Timestamp(start), periods=n + 1)[-1].date()


# ── R3/KTD3: confirm-after for a money-moving proposal ─────────────────


def decide(
    proposal: dict,
    as_of,
    config: dict | None = None,
    *,
    portfolio_state: dict | None = None,
) -> dict:
    """Whether, and when, a money-moving proposal from `advance.decide()`
    becomes a confirmed action.

    `proposal` is shaped like `advance.decide()`'s own `proposal` — it has
    at least `to`, `trigger_id`, `ticker`, `evidence` — and this function is
    meant to be called only for the money-moving ones (`decide()`'s own
    `needs_confirmation=True`, i.e. `proposal["to"]` is `probe`/`scale`/
    `exit_review`): pre-position transitions (`qualify`/`watch`/`dropped`)
    auto-apply inside `advance.decide()` itself and never reach here.

    **Accept-when.** `advance.decide()` only returns a proposal when a
    trigger genuinely fired, with its evidence already assembled — so
    acceptance at this layer is close to unconditional. The one thing this
    layer additionally requires is the `portfolio_state` seam below; nothing
    about the proposal's own evidence is re-checked, because KTD3's
    acceptance test ("proposal evidence complete") is already satisfied by
    the time a proposal reaches here.

    **Confirm-after.** The lag depends on the proposal's destination:
    `to in (probe, scale)` uses the entry lag, `to == exit_review` uses the
    exit lag (`config_from(config)["confirmation_lag_days"]`) — imported
    state constants, never hardcoded strings. The confirmation date is
    `as_of` plus that many trading days (`_advance_trading_days`).

    **Reject-when.** This layer's own reject case is genuinely
    simulator-only: `portfolio_state` is a seam for **U4's ledger, which
    does not exist yet** — an optional `dict | None` (`None` means "no
    ledger reading supplied, accept") that may say `{"can_price": False}` to
    signal the position cannot be priced or sized at all. This is
    deliberately the only shape read from it; no ledger-shaped logic beyond
    that one key is invented here, since U4/U7's authors have not defined
    the interface yet. `advance.decide()`'s own reject cases — a cap breach,
    an unreadable eligibility reading — are not re-checked; they are why the
    proposal was withheld from `applied` in the first place, upstream of
    this call.

    Returns `{"action": "confirm"|"skip", "confirm_at": iso-date|None,
    "reason": str, "severity": str|None, "sell_fraction": float|None,
    "proposal": proposal}` — JSON-serialisable as-is, since the run's
    policy artifact and a Phase 5 sweep both need it round-tripped through
    JSON unchanged. `severity`/`sell_fraction` are populated only for an
    `exit_review` proposal (`severity_for`/`sell_fraction_for`); the ledger
    needs them at settlement.
    """
    if not proposal:
        raise ValueError("owner.decide: proposal must be a non-empty dict")

    as_of_date = as_date(as_of)
    if as_of_date is None:
        raise ValueError(f"owner.decide: as_of {as_of!r} could not be parsed to a date")

    if portfolio_state is not None and portfolio_state.get("can_price", True) is False:
        return {
            "action": "skip",
            "confirm_at": None,
            "reason": (
                "portfolio_state reports this position cannot be priced or "
                "sized — a simulator-only reject case advance.decide() has "
                "no way to know about"
            ),
            "severity": None,
            "sell_fraction": None,
            "proposal": proposal,
        }

    to_state = proposal.get("to")
    trigger_id = proposal.get("trigger_id")
    settings = config_from(config)

    if to_state in _ENTRY_STATES:
        lag_days = settings["confirmation_lag_days"]["entry"]
        severity, sell_fraction = None, None
    elif to_state == EXIT_REVIEW:
        lag_days = settings["confirmation_lag_days"]["exit"]
        severity = severity_for(trigger_id, config)
        sell_fraction = sell_fraction_for(trigger_id, config)
    else:
        # Defensive: a caller handing this function a pre-position proposal
        # (or an unrecognised destination) gets a documented skip rather
        # than a silently-wrong lag. `advance.decide()` never routes
        # `qualify`/`watch`/`dropped` through this function in practice.
        return {
            "action": "skip",
            "confirm_at": None,
            "reason": (
                f"proposal destination {to_state!r} is not a money-moving "
                f"transition this policy layer confirms — only "
                f"{PROBE!r}/{SCALE!r}/{EXIT_REVIEW!r} reach the simulated "
                f"owner; pre-position transitions auto-apply inside "
                f"advance.decide() and never reach here"
            ),
            "severity": None,
            "sell_fraction": None,
            "proposal": proposal,
        }

    confirm_at = _advance_trading_days(as_of_date, lag_days)

    return {
        "action": "confirm",
        "confirm_at": confirm_at.isoformat(),
        "reason": (
            f"accepted — {to_state} proposal evidence already complete "
            f"(advance.decide() only returns a proposal when its trigger "
            f"fired); scheduled {lag_days} trading day(s) after {as_of_date.isoformat()}"
        ),
        "severity": severity,
        "sell_fraction": sell_fraction,
        "proposal": proposal,
    }


# ── KTD6: fabricated fast-lane catalyst ────────────────────────────────


def catalyst_for(
    candidate: str,
    gate_result: dict,
    config: dict | None = None,
    *,
    as_of,
) -> tuple[str, str] | None:
    """KTD6's fabrication: a synthetic `(description, expected_by)` catalyst
    for a fast-lane candidate that has cleared the other five gates.

    Catalysts are owner judgement the `catalyst_identified` lane gate
    requires (§9.2), and no metric can derive one — no LLM runs in the
    replay to read an annual report for a pending demerger or plant
    commissioning. So the simulated-owner policy fabricates one: the gate
    machinery stays live (a real `LaneGateEvaluator` still evaluates
    `catalyst_identified` against whatever this function records), and the
    input itself is named as fabricated so a downstream limitations block
    can say so.

    `gate_result` is shaped like `LaneGateEvaluator.evaluate()`'s own
    return (the same shape `simulator.replay.assign_lane` produces) — this
    function reads `gate_result["gates"][gate_id]["passed"]` for every
    `gate_id` in `FAST_LANE_ENTRY_GATES` (imported from `simulator.replay`
    rather than re-listed, so the two cannot drift), and fabricates a
    catalyst only when **all five** read `passed is True`. A `False` or an
    indeterminate (`None`) reading on any one of them returns `None` — a
    gate that has not cleared, or could not be evaluated, earns no
    fabricated catalyst either way (KTD6's "indeterminate is not a pass"
    discipline, applied here to the fabrication trigger itself).

    `expected_by` is `as_of` plus `catalyst_window_months` (config, default
    6 — decision 3's settled value), a calendar-months offset via
    `pandas.DateOffset`. The description is prefixed `"[simulated]"` and
    names the candidate, the gates it cleared, and the date, so a caller
    surfacing it later (U6) does not have to reconstruct why it exists.

    Returns `None` when fabrication does not apply; otherwise a
    `(description, expected_by)` pair shaped for a caller to pass straight
    into `watchlist.record_catalyst(candidate, description, expected_by)` —
    this function never touches a watchlist itself, per U3's own "pure
    policy functions" approach.
    """
    gates = (gate_result or {}).get("gates") or {}
    cleared = all(
        gates.get(gate_id, {}).get("passed") is True for gate_id in FAST_LANE_ENTRY_GATES
    )
    if not cleared:
        return None

    as_of_date = as_date(as_of)
    if as_of_date is None:
        raise ValueError(f"owner.catalyst_for: as_of {as_of!r} could not be parsed to a date")

    settings = config_from(config)
    window_months = settings["catalyst_window_months"]
    expected_by = (
        (pd.Timestamp(as_of_date) + pd.DateOffset(months=window_months)).date().isoformat()
    )

    description = (
        f"[simulated] fast-lane candidacy catalyst — {candidate} cleared "
        f"{', '.join(FAST_LANE_ENTRY_GATES)} as of {as_of_date.isoformat()}; "
        f"fabricated per KTD6, not an owner-observed event"
    )
    return description, expected_by


# ── KTD3: the routing lag over production's own ranking ────────────────


def route(
    exit_event: dict,
    ranked_candidates: list[dict],
    as_of,
    config: dict | None = None,
) -> dict:
    """KTD3's routing lag, layered over production's already-ranked
    candidate list.

    `ranked_candidates` is whatever a caller (U7) hands in from
    `ReinvestmentQueue.propose_routing(...)`'s own ranking
    (`_rank_candidates`'s `ranked` list — candidate payload dicts each
    carrying at least a `"ticker"` key, best-ranked first) — this function
    never calls `propose_routing`/`_rank_candidates` itself and never
    re-ranks; the ranking is entirely production's question ("which
    candidate deserves this capital"), and this function answers a
    different one ("when does the simulated owner act on that answer").

    Holds when `ranked_candidates` is empty — there is nothing to accept.
    Otherwise the top-ranked candidate (`ranked_candidates[0]`) is accepted
    and scheduled `confirmation_lag_days["route"]` trading days after
    `as_of` (`_advance_trading_days`) — decision 2's settled route lag.

    Returns `{"action": "confirm"|"hold", "confirm_at": iso-date|None,
    "candidate": dict|None, "reason": str, "exit_event": exit_event}` —
    JSON-serialisable as-is, mirroring `decide()`'s shape so a caller can
    treat both the same way when assembling the run's policy record.
    """
    as_of_date = as_date(as_of)
    if as_of_date is None:
        raise ValueError(f"owner.route: as_of {as_of!r} could not be parsed to a date")

    if not ranked_candidates:
        return {
            "action": "hold",
            "confirm_at": None,
            "candidate": None,
            "reason": (
                "no ranked candidate was supplied — nothing to route this "
                "exit's proceeds into"
            ),
            "exit_event": exit_event,
        }

    top = ranked_candidates[0]
    settings = config_from(config)
    lag_days = settings["confirmation_lag_days"]["route"]
    confirm_at = _advance_trading_days(as_of_date, lag_days)

    return {
        "action": "confirm",
        "confirm_at": confirm_at.isoformat(),
        "candidate": top,
        "reason": (
            f"top-ranked candidate {top.get('ticker')!r} accepted as "
            f"production's own router named it; scheduled {lag_days} "
            f"trading day(s) after {as_of_date.isoformat()}"
        ),
        "exit_event": exit_event,
    }
