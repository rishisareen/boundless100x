"""Advancing the lifecycle: re-score, evaluate, propose.

This is the loop that turns a pile of declared rules into a research process.
For each tracked company it re-scores, checks the thesis's own checkpoints,
evaluates the transitions declared for its current state, and reports what
should happen next together with the evidence that says so.

What it deliberately does not do is move money on its own. Transitions that
commit or withdraw capital are proposed and wait for the owner; only
transitions before a position exists — qualifying, watching, dropping a
candidate — apply themselves. The system advises, the owner decides.

When several triggers fire at once, the most protective wins: an exit review
outranks a drop, and both outrank any proposal to buy more. A company whose
valuation entered the buy zone in the same quarter its RoCE broke is not a
buying opportunity.

Both lanes run through this one loop. A re-rating entry additionally has its
six lane gates evaluated, and its catalyst, state history and lane are handed
to the evaluator so the fast lane's own conditions can be read at all; a
core-lane advance never builds a lane-gate evaluator and reaches the same
decision, on the same evidence, as it did before the second lane existed.

Every outcome also carries a `routing_safety` reading — a deterministic,
fail-closed answer to "may capital be deployed into this?", whose eligibility
question follows the lane. A later unit's reinvestment router is its only
consumer.

One run-level input reaches this loop: the deployment-pace modulator reads the
cached corpus's median earnings-yield spread once, ahead of the evaluator's
construction, and hands in a trigger set whose *entry* thresholds are tighter
when the corpus is expensive. Nothing else moves — see `lifecycle/pace.py` for
why macro is allowed to slow buying and nothing else.
"""

import logging

from boundless100x.action_policy import (
    _coverage_constraints,
    _eligibility_constraints,
)
from boundless100x.lifecycle import pace as pace_module
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.checkpoints import (
    evaluate_all,
    record_from_pass2,
    summarise,
)
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.lifecycle.lane_gates import LaneGateEvaluator
from boundless100x.watchlist import (
    APPLIED_AUTO,
    APPLIED_OWNER,
    CORE_LANE,
    RERATING_LANE,
)

logger = logging.getLogger(__name__)

# Most protective first. A kill-switch outranks an entry proposal in the same
# run — the alternative is buying into a company on the quarter its thesis
# broke, because both rules happened to fire.
_PRECEDENCE = {
    lifecycle_states.EXIT_REVIEW: 0,
    lifecycle_states.EXITED: 1,
    lifecycle_states.DROPPED: 2,
}
_DEFAULT_PRECEDENCE = 10

# The lane verdict that clears a re-rating candidate for capital. Stated as a
# constant so the fail-closed check below reads as "only this one word", which
# is the whole rule.
_LANE_QUALIFIES = "qualifies"


def _rank(to_state: str) -> int:
    return _PRECEDENCE.get(to_state, _DEFAULT_PRECEDENCE)


def _exit_rank(to_state: str, spec: dict) -> int:
    """At an exit review, a universal switch before a lane-scoped one.

    `_PRECEDENCE` alone cannot express this. Every `exit_review`-bound trigger —
    a fundamentals kill-switch and a fast-lane target alike — ranks identically
    by destination, so the winner would fall out of `triggers.yaml` declaration
    order. The *destination* is safe either way; the **displayed rationale** is
    not, and that is what the owner reads to decide. A position being exited
    because its incremental returns fell below the cost of capital must not be
    presented as one exiting on a re-rating target it happened to hit in the
    same quarter.

    "Carries no `lane` key" is the test rather than a hand-listed set of
    kill-switch ids: the ids would need maintaining alongside the registry, and
    lane-lessness is exactly the property that makes a trigger a statement about
    the business rather than about one lane's thesis.

    **Confined to `exit_review` on purpose.** The core lane has one other
    same-destination collision that predates this phase — at `qualify`,
    `qualification_failed` and `fundamentals_deteriorated` both propose
    `dropped`, and declaration order gives the reason to the first. Applying
    universal-before-scoped there would flip that recorded reason on a
    **core**-lane entry, and this phase's contract is that the fast lane gains a
    path while the core lane loses nothing. Reordering it may well be an
    improvement — a broken business is a better reason to drop a candidate than
    a failed 100x gate — but it is a separate change to make deliberately, not a
    side effect of opening a second lane.
    """
    if to_state != lifecycle_states.EXIT_REVIEW:
        return 0
    return 1 if spec.get("lane") is not None else 0


def _lane_gate_constraints(lane_gate_result: dict | None) -> list[str]:
    """Reasons the fast lane's own gates do not clear a candidate for capital.

    Deliberately **not** `action_policy._eligibility_constraints`, though the
    shape is the same. That helper speaks the 100x vocabulary — it recognises
    `not_eligible` and `indeterminate` and returns an empty list for anything
    else — while a lane-gate result answers `not_qualified`. Handed a word it
    does not know it would emit no constraint at all: a fail-*open* that routes
    capital into a candidate which has just failed its own entry gates.

    So this clears on one exact word and nothing else. `not_qualified`,
    `indeterminate`, a result that was never produced, and any value nobody
    anticipated all block, each with its own reason. Unknown never routes.
    """
    if not lane_gate_result:
        return ["fast-lane entry gates were not evaluated"]

    verdict = lane_gate_result.get("verdict")
    if verdict == _LANE_QUALIFIES:
        return []

    gates = lane_gate_result.get("gates") or {}

    def reasons_for(gate_ids) -> list[str]:
        return [
            gates[gate_id]["reason"]
            for gate_id in gate_ids
            if gate_id in gates and gates[gate_id].get("reason")
        ]

    if verdict == "not_qualified":
        return (
            reasons_for(lane_gate_result.get("failed", []))
            or ["fails at least one fast-lane entry gate"]
        )
    if verdict == "indeterminate":
        return (
            reasons_for(lane_gate_result.get("indeterminate", []))
            or ["a fast-lane entry gate could not be evaluated"]
        )
    return [
        f"fast-lane gate verdict {verdict!r} is not recognised — "
        f"only {_LANE_QUALIFIES!r} clears a candidate for capital"
    ]


def routing_safety(
    lane: str,
    eligibility: dict | None,
    scores: dict | None,
    lane_gate_result: dict | None = None,
) -> dict:
    """Whether a candidate is safe to deploy capital into, and why not.

    A deterministic payload built during `advance`, in the spirit of
    `action_policy` but not by borrowing it: `resolve_for_result` returns None
    whenever Pass 2 is absent, and `advance_ticker` analyses with
    `use_llm=False` by construction — so reusing it would have made every
    candidate read "no cap known" and passed everything, which is worse than
    having no check at all.

    Two halves. **Evidence** is lane-independent: a score resting on incomplete
    data is not a basis for deploying capital whichever lane asked. **Which
    eligibility question is asked follows the lane** — the core lane's whole
    thesis is hundred-bagger candidacy, while applying that verdict to a
    re-rating candidate would reimpose the exact gate set §9.2 exists to
    replace, and would leave the fast lane unable to receive capital even from
    its own exits.

    A lane nobody declared blocks outright. There is no default question to
    fall back on, and answering an unrecognised lane with the core lane's test
    would be a guess presented as a clearance.

    A later unit's reinvestment router is the only consumer.
    """
    reasons = list(_coverage_constraints(scores))

    if lane == RERATING_LANE:
        reasons += _lane_gate_constraints(lane_gate_result)
    elif lane == CORE_LANE:
        reasons += _eligibility_constraints(eligibility)
    else:
        reasons.append(
            f"lane {lane!r} is not a declared lane — no safety question applies "
            f"to it, so routing is blocked"
        )

    return {"lane": lane, "clear": not reasons, "reasons": reasons}


def record_checkpoints(watchlist, ticker: str, result) -> dict:
    """Store the checkpoints Pass 2 proposed, if it produced any.

    Called after a full analysis; `advance` itself runs without the LLM, so
    checkpoints are written when a thesis is generated and checked on every
    run thereafter.
    """
    llm = result.llm_analysis or {}
    pass2 = llm.get("pass2") if isinstance(llm, dict) else None
    recorded = record_from_pass2(pass2)

    if recorded["checkpoints"]:
        watchlist.set_checkpoints(ticker, recorded["checkpoints"])
        logger.info(
            f"{ticker}: recorded {len(recorded['checkpoints'])} checkpoint(s) "
            f"({len(recorded['demoted'])} demoted to prose)"
        )
    return recorded


def advance_ticker(
    service,
    watchlist,
    ticker: str,
    evaluator: TriggerEvaluator,
    apply: bool = False,
    as_of=None,
    pace: dict | None = None,
) -> dict:
    """Re-score one company and decide what its state should be next."""
    state = watchlist.get(ticker)["state"]

    # No report is built on this path, and momentum is only ever rendered into
    # one — so asking for it would re-read and re-parse the whole append-only
    # score-history log once per tracked ticker for a value nobody reads.
    result = service.analyze(ticker, use_llm=False, include_momentum=False)
    watchlist.record_snapshot(ticker, result, service.engine.registry_hash)

    # Re-read after the commit, not before it. `record_snapshot` stages onto a
    # deep copy and **replaces** `watchlist.data` when the write succeeds, so
    # any entry held across that call is a detached pre-commit object. The two
    # happen to agree today — `record_snapshot` touches neither the catalyst
    # nor the state history — but reading a detached copy is a bug waiting for
    # the first mutator that does, and it would be invisible when it arrived.
    entry = watchlist.get(ticker)
    lane = entry["lane"]

    outcomes = evaluate_all(entry.get("checkpoints"), result.data, as_of)
    checkpoint_summary = summarise(outcomes)

    # Only the fast lane asks this question, so only the fast lane pays for it:
    # a core-lane advance never constructs the evaluator at all. Built per
    # re-rating entry rather than once per run because the cost is one small
    # YAML parse and the alternative — a module-level cache — would make an
    # edited `lane_gates.yaml` take effect at an unpredictable moment.
    lane_gate_result = (
        LaneGateEvaluator().evaluate(
            result.metrics, result.scores, entry.get("catalyst", {})
        )
        if lane == RERATING_LANE
        else None
    )

    evaluation = evaluator.evaluate(
        state,
        metrics=result.metrics,
        scores=result.scores,
        eligibility=result.eligibility,
        checkpoint_results=checkpoint_summary,
        lane_gate_result=lane_gate_result,
        # `{}` rather than a bare `.get("catalyst")`: an entry somebody has
        # looked at and which carries no catalyst is a known fact that reads
        # False, while `None` means no watchlist context was supplied at all
        # and reads indeterminate. Both are falsy, so the default is what keeps
        # them apart.
        catalyst=entry.get("catalyst", {}),
        state_history=entry["state_history"],
        lane=lane,
        # The same clock the checkpoints above were read against. A time stop
        # measured against a different `as_of` than the rest of the run would
        # make a replay disagree with itself.
        as_of=as_of,
    )

    watchlist.set_kill_switch_status(ticker, {
        trigger_id: (
            "fired" if detail["fired"] is True
            else "unknown" if detail["fired"] is None
            else "clear"
        )
        for trigger_id, detail in evaluation["triggers"].items()
    })

    candidates = sorted(
        (
            {
                "ticker": ticker,
                "from": state,
                "to": evaluation["triggers"][trigger_id]["to"],
                "trigger_id": trigger_id,
                "label": evaluation["triggers"][trigger_id]["label"],
                "evidence": evaluation["triggers"][trigger_id]["reason"],
            }
            for trigger_id in evaluation["fired"]
        ),
        # Destination first, then — at an exit review only — universal before
        # lane-scoped. The second key decides only ties the first cannot: which
        # *reason* is shown when two triggers propose the same move. The lane
        # scope is read off the evaluator's own trigger set rather than carried
        # on the proposal, so the payload stays what it was and the
        # pace-modulated copy is read exactly as the shipped registry would be.
        key=lambda p: (
            _rank(p["to"]),
            _exit_rank(p["to"], evaluator.triggers.get(p["trigger_id"], {})),
        ),
    )

    proposal = candidates[0] if candidates else None
    if proposal:
        proposal["superseded"] = [c["trigger_id"] for c in candidates[1:]]
        moves_money = lifecycle_states.moves_money(proposal["to"])

        # A tightened entry threshold must never be invisible in the record
        # that justified the buy. Attached only to the transitions it could
        # have affected — saying "pace applied" beside an exit review would
        # imply macro reached a kill-switch, which is exactly what it may not do.
        if pace and pace.get("applied") and proposal["to"] in pace.get("adjusted_states", ()):
            proposal["pace"] = pace
            proposal["evidence"] = (
                f"{proposal['evidence']} [deployment pace: {pace['evidence']}]"
            )
        should_apply = apply if moves_money else True

        if should_apply:
            watchlist.transition(
                ticker,
                proposal["to"],
                proposal["trigger_id"],
                evidence=proposal["evidence"],
                applied_by=APPLIED_OWNER if moves_money else APPLIED_AUTO,
            )
        proposal["applied"] = should_apply
        proposal["needs_confirmation"] = moves_money and not should_apply

    return {
        "ticker": ticker,
        "state": state,
        "lane": lane,
        "composite": (result.scores or {}).get("composite"),
        "verdict": (result.eligibility or {}).get("verdict", "indeterminate"),
        "lane_gates": lane_gate_result,
        "proposal": proposal,
        "indeterminate": evaluation["indeterminate"],
        "checkpoints": checkpoint_summary,
        "checkpoint_outcomes": outcomes,
        "routing_safety": routing_safety(
            lane, result.eligibility, result.scores, lane_gate_result
        ),
    }


def _resolve_pace(service, evaluator, pace_reading) -> tuple[TriggerEvaluator, dict]:
    """The evaluator this run will use, and why the pace reading did or did not apply.

    An injected evaluator wins outright. It is the seam callers already use to
    supply an exact trigger set, and silently re-deriving one from it would
    make the injection a suggestion rather than a contract — so the decision
    records that the caller supplied it rather than claiming a modulation that
    never happened.
    """
    if evaluator is not None:
        return evaluator, {
            "applied": False,
            "reason": "an evaluator was supplied by the caller — pace not evaluated",
            "median_pp": None,
            "contributors": 0,
            "adjusted": {},
            "adjusted_states": (),
            "evidence": "",
        }

    reading = (
        pace_reading
        if pace_reading is not None
        else pace_module.corpus_spread(
            service.suite.raw_data_dir, macro=getattr(service.engine, "macro", {})
        )
    )
    triggers, decision = pace_module.modulate(
        load_triggers(), reading, **pace_module.config_from(getattr(service, "config", {}))
    )
    return (
        TriggerEvaluator(triggers, known_metric_ids=set(service.engine.metrics)),
        decision,
    )


def advance(
    service,
    watchlist,
    apply: bool = False,
    quarterly: bool = False,
    evaluator: TriggerEvaluator | None = None,
    as_of=None,
    pace_reading: dict | None = None,
) -> dict:
    """Advance every tracked company. Returns per-ticker outcomes and errors.

    A failure on one company must not stop the rest: a stale fetch for one
    holding is no reason to skip checking whether another one's thesis broke.

    The deployment-pace reading is resolved **once, before the ticker loop**,
    and reaches the loop only as the evaluator's trigger set. That ordering is
    the point: a per-company reading could not have supplied a single shared
    evaluator, and computing it per ticker would have made the market's
    valuation a per-name test — which is precisely what §11 forbids.

    `pace_reading` lets a caller supply the corpus reading directly (tests, a
    future simulator) without touching `raw_data/`.
    """
    # Defence in depth for the same guarantee: this resolves once, before the
    # per-ticker loop's own isolation, so anything it raises would end the run
    # for every tracked company. An unresolvable pace reading must cost the
    # modulation, never the advance — and unmodulated is the safe direction,
    # since an unknown macro reading may not tighten entry either.
    try:
        evaluator, pace = _resolve_pace(service, evaluator, pace_reading)
    except Exception as e:
        logger.error(f"Deployment pace could not be resolved: {e}")
        evaluator = evaluator or TriggerEvaluator(
            known_metric_ids=set(service.engine.metrics)
        )
        pace = {
            "applied": False,
            "reason": f"pace could not be resolved ({e}) — entry unmodulated",
            "median_pp": None, "contributors": 0,
            "adjusted": {}, "adjusted_states": (), "evidence": "",
        }

    tickers = watchlist.get_stale(90) if quarterly else watchlist.tickers()

    outcomes: list[dict] = []
    errors: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            outcomes.append(
                advance_ticker(
                    service, watchlist, ticker, evaluator, apply, as_of, pace
                )
            )
        except Exception as e:
            logger.error(f"Advance failed for {ticker}: {e}")
            errors.append((ticker, str(e)))

    return {"outcomes": outcomes, "errors": errors, "pace": pace}
