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
core-lane advance never consults the lane gates and reaches the same decision,
on the same evidence, as it did before the second lane existed. The lane-gate
evaluator is built **once per run and validated against the engine's metric
ids**, beside the trigger evaluator and for the same reason: a gate naming a
metric nobody computes reads indeterminate forever, and from inside the loop
that is indistinguishable from a lane with no qualifying candidates.

Every outcome also carries a `routing_safety` reading — a deterministic,
fail-closed answer to "may capital be deployed into this?", whose eligibility
question follows the lane. A later unit's reinvestment router is its only
consumer.

A proposal to review an exit additionally carries a `friction` reading: what
the position is *modeled* to keep after capital-gains tax and round-trip
slippage, stated beside its gross figure so neither is ever read alone. It is
an estimate in the strict sense — the holding period runs from a `probe`
confirmation date rather than a fill, and market bars stand in for trade
prices — and `lifecycle/friction.py` is where that language is enforced.

Two run-level readings come *out* of the loop the same way. After every company
has been advanced, the portfolio's concentration is counted once — positioned
names per lane and per sector, seeded from the watchlist so a failed fetch
cannot make a position disappear from a cap check. It is a count of names
rather than a share of capital, because no capital is recorded anywhere in this
system; `lifecycle/portfolio.py` is where that decision is argued.

Then, last and consulting that count, the reinvestment router asks where the
proceeds of past exits should go. This is the one moment current trigger state
exists, which is why it runs here rather than in the display command that reads
it back. It writes a whole-run snapshot — only on a full run, never on
`--quarterly` — and proposes without ever applying: `lifecycle/reinvestment.py`
argues why a proposal must stay inert, and why a missing queue reads as
*unavailable* rather than as an empty one.

One run-level input reaches this loop: the deployment-pace modulator reads the
cached corpus's median earnings-yield spread once, ahead of the evaluator's
construction, and hands in a trigger set whose *entry* thresholds are tighter
when the corpus is expensive. Nothing else moves — see `lifecycle/pace.py` for
why macro is allowed to slow buying and nothing else.
"""

import logging
from datetime import date, datetime

from boundless100x.action_policy import (
    _coverage_constraints,
    _eligibility_constraints,
)
from boundless100x.lifecycle import friction as friction_module
from boundless100x.lifecycle import lane_view
from boundless100x.lifecycle import pace as pace_module
from boundless100x.lifecycle import portfolio
from boundless100x.lifecycle import reinvestment
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.checkpoints import (
    evaluate_all,
    record_from_pass2,
    summarise,
)
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.lifecycle.lane_gates import (
    INDETERMINATE,
    NOT_QUALIFIED,
    QUALIFIES,
    LaneGateEvaluator,
)
from boundless100x.lifecycle.states import (
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

    So this clears on one exact word and nothing else. `NOT_QUALIFIED`,
    `INDETERMINATE`, a result that was never produced, and any value nobody
    anticipated all block, each with its own reason. Unknown never routes.

    The words are imported from `lane_gates` rather than spelled here, because
    a rename that missed this file would not raise: the recognised branches
    would simply stop matching and every fast-lane candidate would fall through
    to the unrecognised-verdict block — routing blocked with a reason that
    reads like a bug in the gates. A silent capital freeze is the quietest of
    the three failure modes this vocabulary has, and the hardest to attribute.
    """
    if not lane_gate_result:
        return ["fast-lane entry gates were not evaluated"]

    verdict = lane_gate_result.get("verdict")
    if verdict == QUALIFIES:
        return []

    gates = lane_gate_result.get("gates") or {}

    def reasons_for(gate_ids) -> list[str]:
        return [
            gates[gate_id]["reason"]
            for gate_id in gate_ids
            if gate_id in gates and gates[gate_id].get("reason")
        ]

    if verdict == NOT_QUALIFIED:
        return (
            reasons_for(lane_gate_result.get("failed", []))
            or ["fails at least one fast-lane entry gate"]
        )
    if verdict == INDETERMINATE:
        return (
            reasons_for(lane_gate_result.get("indeterminate", []))
            or ["a fast-lane entry gate could not be evaluated"]
        )
    return [
        f"fast-lane gate verdict {verdict!r} is not recognised — "
        f"only {QUALIFIES!r} clears a candidate for capital"
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

    A lane with no question here blocks outright. There is no default to fall
    back on, and answering with the core lane's test would be a guess presented
    as a clearance. Two different situations land there — a lane nobody
    declared, and a lane declared in `watchlist.LANES` that this function has
    no branch for yet — so the reason names the file that has to change rather
    than asserting the first, which would send whoever reads it to the wrong
    place on the day a third lane is added.

    A later unit's reinvestment router is the only consumer.
    """
    reasons = list(_coverage_constraints(scores))

    if lane == RERATING_LANE:
        reasons += _lane_gate_constraints(lane_gate_result)
    elif lane == CORE_LANE:
        reasons += _eligibility_constraints(eligibility)
    else:
        reasons.append(
            f"lane {lane!r} has no routing-safety question declared for it in "
            f"lifecycle/advance.py — routing is blocked rather than cleared by "
            f"a test that was never written for this lane"
        )

    return {"lane": lane, "clear": not reasons, "reasons": reasons}


def _friction_for_exit(service, ticker: str, entry: dict, result, as_of) -> dict | None:
    """What a proposed exit is modeled to keep after tax and slippage.

    Attached to an `exit_review` proposal because that is the one transition
    whose headline number is a *return*, and a gross return overstates what a
    position keeps by enough to change the decision — §8.2's whole point. Gross
    and net travel together from here on (R5).

    **The entry date is the most recent transition into `probe`**, which is a
    confirmation date rather than a fill; the exit date is the run's `as_of`,
    which is still moving while the exit is only proposed. Hence
    `basis: "estimate"` — every figure downstream of those two proxies is a
    model, and `lifecycle/friction.py` says so in every string it renders.

    A company with no recorded `probe` transition returns **None**, and the
    proposal carries no `friction` key at all: there is no modeled position to
    price, which is a different fact from a position that could not be priced,
    and the two must not render alike. `exited` is deliberately not handled
    here — it is never an `advance()` proposal target, and its `recorded`
    reading is written where an exit is actually confirmed.

    A failure computing the reading costs the reading, never the exit proposal.
    A kill-switch that fired must reach the owner whether or not a price series
    could be read beside it.

    The probe lookup and the failure conversion are `friction.reading_for_exit`'s
    — the same call the lane view and the confirmed-exit path make. What is left
    here is the choice of the two dates, which is the only thing an exit
    *proposal* decides differently from the other two.
    """
    return friction_module.reading_for_exit(
        entry,
        (result.data or {}).get("price"),
        # The same clock the rest of the run reads, so a replay of an old
        # decision reproduces the figure it was decided on.
        as_of or date.today(),
        config=getattr(service, "config", {}),
        basis=friction_module.BASIS_ESTIMATE,
        label=ticker,
    )


def _lane_context(
    service,
    watchlist,
    ticker: str,
    result,
    as_of,
    lane_gate_result: dict | None,
    friction_estimate: dict | None = None,
) -> dict | None:
    """The lane view for this company, as it stands after the run's own transition.

    Assembled by `lifecycle/lane_view.py` — the same function the report calls —
    so a lane, a catalyst window and a modeled friction figure read identically
    wherever they are shown. **Two already-paid-for readings are handed in
    rather than recomputed.** The lane-gate result was evaluated above against
    these exact readings, and a second evaluation could disagree with the first
    if the registry changed underneath the run. The friction estimate, when this
    run proposed an exit, was modeled above from the same entry, the same price
    series and the same `as_of` — so recomputing it would rebuild a frame over
    the whole daily series to reach the number already attached to the proposal,
    and any disagreement between the two would be an owner reading one net
    return in the terminal and another in the report.

    Never raises. A view is a nice-to-have beside a proposal that may be moving
    money, and an advance run that failed because a display field could not be
    assembled would be the expensive half of that trade.
    """
    try:
        return lane_view.build_lane_context(
            watchlist.get(ticker),
            result,
            as_of,
            lane_gate_result,
            config=getattr(service, "config", {}),
            friction_estimate=friction_estimate,
        )
    except Exception as e:
        logger.warning(f"{ticker}: the lane view could not be assembled: {e}")
        return None


def _sector_of(result) -> str | None:
    """The sector this ticker's last fetch recorded, or None.

    Only tickers fetched after the breadcrumb fix carry `metadata.sector`, so
    None is the expected reading for part of the corpus rather than a fault.
    Every layer of the lookup is guarded because `metadata.json` is scraped: a
    file holding valid JSON of the wrong shape parses fine and then fails on
    attribute access, and a concentration reading is not worth an advance run.
    """
    data = getattr(result, "data", None)
    metadata = data.get("metadata") if isinstance(data, dict) else None
    sector = metadata.get("sector") if isinstance(metadata, dict) else None
    return sector if isinstance(sector, str) and sector.strip() else None


def record_checkpoints(watchlist, ticker: str, result, as_of=None) -> dict:
    """Store the checkpoints Pass 2 proposed, if it produced any.

    Called after a full analysis; `advance` itself runs without the LLM, so
    checkpoints are written when a thesis is generated and checked on every
    run thereafter.

    **`as_of` is the run's clock, and it has to reach the recorder.**
    `record_from_pass2` refuses a checkpoint already due when it is recorded —
    a checkpoint due on the day it is written was never monitored, and pending
    versus due is the whole value of the mechanism. Left to default, that
    refusal reads `date.today()` while everything else in the run reads the
    supplied date, so a backdated replay would validate a `due_date` against a
    "today" no other part of the run agrees with. This is the one seam where
    the layer's "same clock throughout" discipline did not reach; it now does.
    """
    llm = result.llm_analysis or {}
    pass2 = llm.get("pass2") if isinstance(llm, dict) else None
    recorded = record_from_pass2(pass2, as_of=lifecycle_states.as_date(as_of))

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
    lane_gates: LaneGateEvaluator | None = None,
    concentration_gate=None,
    override_caps: bool = False,
) -> dict:
    """Re-score one company and decide what its state should be next.

    `lane_gates` is the run's validated lane-gate evaluator, injected the way
    `evaluator` already is — `advance()` builds one per run against the
    engine's metric ids. It is optional so a direct caller (a test, a future
    single-ticker surface) still works; that caller gets an unvalidated
    evaluator built here, which is the seam `lane_view` has always used.

    `concentration_gate` is `(lane, sector) -> [reasons]`, supplied by
    `advance()` and consulted **before** a transition that would take a
    position. Optional for the same seam reason: a direct caller with no gate
    gets the pre-guardrail behaviour, which is what every non-`advance()`
    caller wants.
    """
    state = watchlist.get(ticker)["state"]

    # No report is built on this path, and momentum is only ever rendered into
    # one — so asking for it would re-read and re-parse the whole append-only
    # score-history log once per tracked ticker for a value nobody reads.
    result = service.analyze(ticker, use_llm=False, include_momentum=False)

    # The snapshot is **not** written here. It stamps `last_score_snapshot.at`,
    # which is the exact field `get_stale(90)` reads, so writing it the moment
    # the analysis returned meant a ticker whose advance raised anywhere below
    # was recorded in the run's `errors` *and* marked freshly scored — and
    # `advance --quarterly` would then not look at it again for three months. A
    # thesis that broke on the one day a ticker errored went unevaluated until
    # the quarter was up. It is written at the end instead, so "scored" means
    # the run got through, not that it started. See the commit below the
    # transition.

    # Every mutator stages onto a deep copy and **replaces** `watchlist.data`
    # on a successful write, so an entry held across one is a detached
    # pre-commit object. Read here, before anything in this function commits.
    entry = watchlist.get(ticker)
    lane = entry["lane"]

    outcomes = evaluate_all(entry.get("checkpoints"), result.data, as_of)
    checkpoint_summary = summarise(outcomes)

    # Only the fast lane asks this question, so only the fast lane is evaluated
    # — a core-lane advance produces None and reaches the same decision it did
    # before the second lane existed. The evaluator itself now comes from the
    # run rather than being built here: see `advance()` for why the validation
    # that requires has to happen once, at startup, against the engine's metric
    # ids. A caller that supplied none gets the old unvalidated construction, so
    # the direct-call seam keeps working.
    lane_gate_result = (
        (lane_gates or LaneGateEvaluator()).evaluate(
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

    # Modeled at most once per ticker, and only if something asks. The lane view
    # below reads whatever this holds rather than modeling its own, so the
    # proposal's evidence line and the report's lane section are the same
    # reading and not two passes over the same price series.
    friction_estimate = None

    # Read once and used twice — by the concentration gate below and by the
    # outcome this returns. The run's only path from a per-ticker analysis to
    # the sector census: `result` is local to this function and never reaches
    # the caller. Read rather than stored, per `lifecycle/portfolio.py`.
    sector = _sector_of(result)

    proposal = candidates[0] if candidates else None
    if proposal:
        proposal["superseded"] = [c["trigger_id"] for c in candidates[1:]]
        moves_money = lifecycle_states.moves_money(proposal["to"])

        # A tightened entry threshold must never be invisible in the record
        # that justified the buy — and, just as strictly, must never appear in
        # a record it did not tighten. Attached **by trigger id**, which is how
        # `pace["adjusted"]` is keyed.
        #
        # Keyed by destination state, as this was while exactly one trigger
        # targeted `probe`, the clause reached both lanes' entries. Only one of
        # them is tightenable: `valuation_buy_zone` carries `metric` conditions
        # with thresholds a factor can move, while `fast_lane_buy_zone`'s single
        # condition is `lane_verdict: qualifies` and holds no threshold
        # anywhere. So a fast-lane buy was recorded — permanently, since
        # `transition` writes evidence into an append-only history — claiming a
        # discipline never applied to it. `pace.py` renders its own line from
        # the values it actually wrote for exactly this reason; the rule has to
        # survive the layer that records the line.
        #
        # `adjusted_states` stays what it is: the run-level display aggregate
        # the CLI prints before the table, and no longer an attachment key.
        changes = ((pace or {}).get("adjusted") or {}).get(proposal["trigger_id"])
        if pace and pace.get("applied") and changes:
            clause = pace_module.evidence_for(pace, proposal["trigger_id"])
            # The record carried on the proposal states only this trigger's own
            # changes, so whatever reads it back agrees with the string that
            # went into the history rather than with the whole-run line.
            proposal["pace"] = {
                **pace,
                "adjusted": {proposal["trigger_id"]: changes},
                "evidence": clause,
            }
            proposal["evidence"] = f"{proposal['evidence']} [deployment pace: {clause}]"

        # Appended to the evidence *before* the transition below writes it, so
        # the append-only history records the net figure beside the gross one
        # rather than only the reason the exit was proposed. An unavailable
        # reading is still attached — the CLI says why it could not be
        # computed — but nothing goes into the evidence, because a recorded
        # line claiming a friction estimate that does not exist is worse than
        # a line that never mentioned one.
        if proposal["to"] == lifecycle_states.EXIT_REVIEW:
            reading = friction_estimate = _friction_for_exit(
                service, ticker, entry, result, as_of
            )
            if reading is not None:
                proposal["friction"] = reading
                if reading.get("available"):
                    proposal["evidence"] = (
                        f"{proposal['evidence']} "
                        f"[{friction_module.describe(reading)}]"
                    )

        # ── the concentration guardrail, asked before the money moves ──
        #
        # It used to be counted only after the whole loop, which meant a cap
        # could be *reported* as breached and never *prevented* from being
        # breached: by the time the reading existed, the transitions that broke
        # it were already in an append-only history. A guardrail an owner only
        # ever meets in the past tense is a report, not a guardrail.
        #
        # Asked only when this transition would **add a name**, because that is
        # what the caps count. A `probe → scale` moves the same company deeper
        # into a position it already holds and changes no count, so gating it
        # would refuse to let an owner build a position they are already in on
        # the grounds that they are already in it.
        cap_reasons = []
        if (
            concentration_gate is not None
            and proposal["to"] in lifecycle_states.POSITIONED
            and state not in lifecycle_states.POSITIONED
        ):
            cap_reasons = list(concentration_gate(lane, sector) or [])

        if cap_reasons:
            # Into the evidence as well as onto the proposal, following the
            # deployment-pace clause: if the owner overrides and the transition
            # lands, the append-only history has to record that a cap was
            # knowingly breached rather than silently.
            proposal["concentration_reasons"] = cap_reasons
            proposal["evidence"] = (
                f"{proposal['evidence']} "
                f"[concentration: {'; '.join(cap_reasons)}"
                f"{'; overridden by the owner' if override_caps else ''}]"
            )

        # Withheld even under `--apply`, and that is the whole behaviour change.
        # An override exists because a guardrail with no way past it can trap
        # the owner out of their own decision — but it is explicit, per-run, and
        # recorded in the evidence above.
        withheld = bool(cap_reasons) and not override_caps
        proposal["concentration_withheld"] = withheld

        should_apply = (apply and not withheld) if moves_money else True

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

    # ── the scoring snapshot, last of this function's writes ──
    #
    # Last because `get_stale(90)` reads its timestamp to decide what a
    # `--quarterly` run looks at, and that question is "was this company
    # successfully evaluated recently?", not "did an evaluation of it begin?".
    # Written up front, a ticker that raised below was reported as an error and
    # simultaneously marked fresh, so the next three months of quarterly runs
    # skipped the one company whose last run had failed.
    #
    # After the transition rather than before it, and the ordering is the safe
    # one of the two: a snapshot write that fails leaves a recorded transition
    # with a stale score, which the next run corrects. The reverse leaves a
    # company that looks scored and never moved.
    watchlist.record_snapshot(ticker, result, service.engine.registry_hash)

    return {
        "ticker": ticker,
        "state": state,
        "lane": lane,
        # The same assembled view a report renders, so the terminal and the
        # report cannot describe one position two ways. Built **after** the
        # transition above and from a re-read entry, so it describes where the
        # company now stands rather than where it stood when the run began, and
        # handed both the lane-gate result and the friction estimate already
        # computed above rather than paying for a second of either. A failure
        # costs the view, never the advance.
        "lane_context": _lane_context(
            service, watchlist, ticker, result, as_of, lane_gate_result,
            friction_estimate,
        ),
        "sector": sector,
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


def _concentration(service, watchlist, outcomes: list[dict]) -> dict:
    """How crowded the portfolio is, counted once after the whole run.

    **Seeded from the watchlist, not from the run's outcomes**, and the
    direction is the whole guarantee. A positioned name whose analysis failed
    still holds its capital; built from successful outcomes, it would drop out
    of its lane's total and a full lane would read as having room on the one day
    its fetch broke — a guardrail that stops seeing a position is worse than no
    guardrail, because absence reads as headroom.

    The sector comes the other way, overlaid from each successful outcome, and
    a name without one is excluded from sector *grouping* only. It is also why
    this runs after the loop rather than inside it: the states it counts include
    any transition the run just applied, and one reading per run mirrors how
    `pace.py` resolves its corpus median.

    A failure costs the reading, never the run — every ticker has already been
    advanced by the time this is called, and throwing that away to report a
    count would be the expensive half of the trade.
    """
    sectors = {
        outcome["ticker"]: outcome.get("sector")
        for outcome in outcomes
        if outcome.get("ticker")
    }
    entries = []
    for ticker in watchlist.tickers():
        entry = watchlist.get(ticker) or {}
        entries.append({
            "ticker": ticker,
            "lane": entry.get("lane"),
            "state": entry.get("state"),
            "sector": sectors.get(ticker),
        })
    return portfolio.check_concentration(
        entries, getattr(service, "config", {})
    )


def _routing(
    queue, watchlist, outcomes, concentration, errors, quarterly, as_of
) -> dict:
    """Where this run says the proceeds of past exits should go.

    **A missing queue means routing is unavailable, never a partial view.** The
    idle readings and the route state live in the event log; without it there
    is nothing to compute them from, and a view claiming "no proceeds awaiting
    routing" when the queue simply was not supplied would be a false all-clear.
    So it says so and persists nothing.

    **Only a full run writes the snapshot.** A `--quarterly` run advances a
    stale subset, and its ranking is drawn from whichever companies happened to
    be 90 days old — overwriting the canonical view with it would promote a
    lower-ranked candidate merely because the better one was not re-scored that
    day. The view is still returned, and `persisted: False` with its reason is
    what keeps "not written" from reading as "not computed".

    A failure here costs the routing view and nothing else. Every company has
    already been advanced by the time this runs, and throwing that away to
    report a proposal would be the expensive half of the trade.

    The snapshot is the router's own view (`as_of`, `proposal`, `reason`,
    `blocked`, `idle`, `incomplete`, `ranked`) plus what only the run knows:
    when it was generated, whether every ticker evaluated, and the revision of
    each store at that moment.
    """
    if queue is None:
        return {
            "available": False,
            "persisted": False,
            "reason": (
                "no reinvestment queue was supplied — idle readings and route "
                "state live in its event log and cannot be computed without it"
            ),
        }

    try:
        view = queue.propose_routing(watchlist, outcomes, concentration, as_of=as_of)
    except Exception as e:
        logger.error(f"The routing view could not be built: {e}")
        return {
            "available": False,
            "persisted": False,
            "reason": f"the routing view could not be built ({e})",
        }

    errored = [ticker for ticker, _ in errors]
    snapshot = {
        **view,
        "generated_at": datetime.now().isoformat(),
        # `partial` names the tickers whose analysis failed, because a ranking
        # built on an incomplete field is exactly as good as knowing which
        # company is missing from it.
        "status": (
            reinvestment.SNAPSHOT_PARTIAL if errored
            else reinvestment.SNAPSHOT_CURRENT
        ),
        "errors": errored,
        "watchlist_revision": watchlist.data.get("revision"),
        # Overwritten by `write_proposal` with the revision its own commit
        # produces; correct as it stands for a run that does not persist.
        "queue_revision": queue.data.get("revision"),
    }

    if quarterly:
        return {
            **snapshot,
            "available": True,
            "persisted": False,
            "persist_reason": (
                "a --quarterly run advances only stale entries, so its ranking "
                "is drawn from a subset — the stored snapshot is left alone "
                "rather than overwritten by an incomplete field"
            ),
        }

    try:
        snapshot = queue.write_proposal(snapshot)
        persisted, persist_reason = True, ""
    except Exception as e:
        # The previous complete snapshot survives (atomic replace), and the
        # run's own view is still returned — a failed write must not cost the
        # caller the reading it already has in hand.
        logger.error(f"The routing snapshot could not be written: {e}")
        persisted, persist_reason = False, f"the snapshot could not be written ({e})"

    return {
        **snapshot,
        "available": True,
        "persisted": persisted,
        "persist_reason": persist_reason,
    }


def advance(
    service,
    watchlist,
    apply: bool = False,
    quarterly: bool = False,
    evaluator: TriggerEvaluator | None = None,
    as_of=None,
    pace_reading: dict | None = None,
    queue=None,
    override_caps: bool = False,
) -> dict:
    """Advance every tracked company. Returns outcomes, errors, and three run-level readings.

    The concentration reading is counted after the loop and from the watchlist,
    so it describes the portfolio as it stands once the run's own transitions
    have been applied — including the positions of any company whose analysis
    failed. See `_concentration`.

    It is *also* consulted inside the loop, before any transition that would
    take a position, which is the one place a cap can be honoured rather than
    merely reported. Recomputed per candidate rather than taken once up front,
    because an applying run changes the very occupancy it is checking: two
    probes into a lane with room for one would both pass a reading taken before
    either landed. `override_caps` lets the owner proceed anyway, and the
    breach is written into the evidence when they do — see `advance_ticker`.

    A failure on one company must not stop the rest: a stale fetch for one
    holding is no reason to skip checking whether another one's thesis broke.

    The deployment-pace reading is resolved **once, before the ticker loop**,
    and reaches the loop only as the evaluator's trigger set. That ordering is
    the point: a per-company reading could not have supplied a single shared
    evaluator, and computing it per ticker would have made the market's
    valuation a per-name test — which is precisely what §11 forbids.

    `pace_reading` lets a caller supply the corpus reading directly (tests, a
    future simulator) without touching `raw_data/`.

    The lane-gate evaluator is resolved once here for the same shape of reason
    and one further one: constructing it per ticker meant it was never handed
    the engine's metric ids, so its startup validation — the check that turns a
    gate naming a nonexistent metric into an error rather than into a lane
    nobody can enter — never ran outside the tests. It raises rather than
    degrading, because unreadable entry rules are not a reading to carry on
    without.

    `queue` is the reinvestment store the routing view is derived from and
    written to. It is optional and defaults to None — see `_routing` for why
    that reads as *unavailable* rather than as an empty queue.
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

    # Once per run, and **validated**, which is the whole point of building it
    # here rather than inside the loop. `validate_lane_gates` guards its
    # unknown-metric-id check behind `known_metric_ids is not None`, so a
    # per-ticker `LaneGateEvaluator()` with no ids meant that check never ran on
    # the production path: rename `institutional_accumulation_streak` in
    # `size.yaml` and the fast lane goes permanently indeterminate with a green
    # suite and no startup error — and, as `lane_gates.py` puts it, a lane no
    # company can ever enter looks exactly like a lane with no qualifying
    # candidates. `TriggerEvaluator` is handed the same set at both of its
    # production call sites; the sibling has to be too.
    #
    # Deliberately **outside** the pace block's try/except. An unresolvable
    # macro reading is a degraded reading to carry on without; an unreadable
    # gate registry is the fast lane's entry rules being unreadable, and
    # carrying on would mean admitting nobody while saying nothing. Building it
    # per run rather than caching at module level also keeps the existing
    # semantics: an edited `lane_gates.yaml` takes effect at the next run
    # boundary, never partway through a loop.
    lane_gates = LaneGateEvaluator(known_metric_ids=set(service.engine.metrics))

    tickers = watchlist.get_stale(90) if quarterly else watchlist.tickers()

    outcomes: list[dict] = []
    errors: list[tuple[str, str]] = []

    def concentration_gate(lane, sector) -> list[str]:
        """Whether one more positioned name fits, counted as the loop stands.

        Live rather than pre-computed: the loop applies the transitions it is
        checking, so a reading taken before it started would let a second probe
        into a lane that had room for one. Cheap enough to repeat — the count
        reads the already-loaded watchlist and the sectors gathered so far, and
        touches no source.

        A reading that could not be built blocks rather than passes, and says
        so. Absence must not read as headroom, which is the same rule
        `portfolio.would_breach` applies to every other gap it meets.
        """
        try:
            reading = _concentration(service, watchlist, outcomes)
        except Exception as e:
            logger.error(f"Concentration could not be counted before applying: {e}")
            reading = portfolio.unavailable(
                f"the concentration reading could not be built ({e})"
            )
        return portfolio.would_breach(lane, sector, reading)

    for ticker in tickers:
        try:
            outcomes.append(
                advance_ticker(
                    service, watchlist, ticker, evaluator, apply, as_of, pace,
                    lane_gates, concentration_gate, override_caps,
                )
            )
        except Exception as e:
            logger.error(f"Advance failed for {ticker}: {e}")
            errors.append((ticker, str(e)))

    try:
        concentration = _concentration(service, watchlist, outcomes)
    except Exception as e:
        logger.error(f"Concentration reading could not be built: {e}")
        concentration = portfolio.unavailable(
            f"the concentration reading could not be built ({e})"
        )

    return {
        "outcomes": outcomes,
        "errors": errors,
        "pace": pace,
        "concentration": concentration,
        # Last, and after the concentration reading it consults: the router
        # must see the portfolio as it stands once this run's own transitions
        # have been applied.
        "routing": _routing(
            queue, watchlist, outcomes, concentration, errors, quarterly, as_of
        ),
    }
