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
"""

import logging

from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.checkpoints import (
    evaluate_all,
    record_from_pass2,
    summarise,
)
from boundless100x.lifecycle.evaluator import TriggerEvaluator
from boundless100x.watchlist import APPLIED_AUTO, APPLIED_OWNER

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
) -> dict:
    """Re-score one company and decide what its state should be next."""
    entry = watchlist.get(ticker)
    state = entry["state"]

    result = service.analyze(ticker, use_llm=False)
    watchlist.record_snapshot(ticker, result, service.engine.registry_hash)

    outcomes = evaluate_all(entry.get("checkpoints"), result.data, as_of)
    checkpoint_summary = summarise(outcomes)

    evaluation = evaluator.evaluate(
        state,
        metrics=result.metrics,
        scores=result.scores,
        eligibility=result.eligibility,
        checkpoint_results=checkpoint_summary,
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
        key=lambda p: _rank(p["to"]),
    )

    proposal = candidates[0] if candidates else None
    if proposal:
        proposal["superseded"] = [c["trigger_id"] for c in candidates[1:]]
        moves_money = lifecycle_states.moves_money(proposal["to"])
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
        "composite": (result.scores or {}).get("composite"),
        "verdict": (result.eligibility or {}).get("verdict", "indeterminate"),
        "proposal": proposal,
        "indeterminate": evaluation["indeterminate"],
        "checkpoints": checkpoint_summary,
        "checkpoint_outcomes": outcomes,
    }


def advance(
    service,
    watchlist,
    apply: bool = False,
    quarterly: bool = False,
    evaluator: TriggerEvaluator | None = None,
    as_of=None,
) -> dict:
    """Advance every tracked company. Returns per-ticker outcomes and errors.

    A failure on one company must not stop the rest: a stale fetch for one
    holding is no reason to skip checking whether another one's thesis broke.
    """
    evaluator = evaluator or TriggerEvaluator(
        known_metric_ids=set(service.engine.metrics)
    )
    tickers = watchlist.get_stale(90) if quarterly else watchlist.tickers()

    outcomes: list[dict] = []
    errors: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            outcomes.append(
                advance_ticker(service, watchlist, ticker, evaluator, apply, as_of)
            )
        except Exception as e:
            logger.error(f"Advance failed for {ticker}: {e}")
            errors.append((ticker, str(e)))

    return {"outcomes": outcomes, "errors": errors}
