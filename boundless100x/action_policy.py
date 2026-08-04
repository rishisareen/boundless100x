"""The final investment action, decided in code rather than by the model.

The composite and the eligibility gates answer different questions, and the
action a reader acts on has to respect both. Pass 2 sees metrics and scores
and returns a `suggested_action`; nothing in a prompt can be the guard that
stops that action contradicting a deterministic verdict already computed
upstream. A report showing "Not a 100x Candidate" beside "STRONG BUY" is the
failure this module exists to prevent.

Capping is deliberately not the same as overriding. Failing a gate does not
make a company a bad investment — only an unlikely hundred-bagger; a large,
excellent compounder fails the size gate by construction. So a constrained
action is lowered to `watchlist` (quality noted, entry not endorsed) rather
than flipped to `avoid`, and the model's original action is preserved beside
it rather than erased.
"""

# Worst to best. Ordering is what makes "cap" meaningful — an action already
# at or below the ceiling is left alone rather than raised to it.
ACTION_ORDER = ("avoid", "watchlist", "hold", "buy", "strong_buy")

# Quality may still be real; the entry is what is not endorsed.
CAP_CEILING = "watchlist"


def _rank(action) -> int | None:
    """Position in ACTION_ORDER, or None for anything unrecognised."""
    if not isinstance(action, str):
        return None
    try:
        return ACTION_ORDER.index(action.strip().lower())
    except ValueError:
        return None


def _eligibility_constraints(eligibility: dict | None) -> list[str]:
    """Reasons the 100x verdict does not clear an unqualified buy.

    A missing or errored evaluation is itself a constraint: Stage 3.6 catches
    its own exceptions, so `None` here means the verdict never ran, not that
    it passed. Treating that as clean would fail open at exactly the point
    this module exists to hold.
    """
    if not eligibility or not eligibility.get("verdict"):
        return ["100x eligibility was not evaluated"]

    verdict = eligibility["verdict"]
    gates = eligibility.get("gates", {})

    def reasons_for(gate_ids) -> list[str]:
        return [
            gates[g]["reason"] for g in gate_ids
            if g in gates and gates[g].get("reason")
        ]

    if verdict == "not_eligible":
        detail = reasons_for(eligibility.get("failed", []))
        return detail or ["fails at least one 100x eligibility gate"]
    if verdict == "indeterminate":
        detail = reasons_for(eligibility.get("indeterminate", []))
        return detail or ["a 100x eligibility gate could not be evaluated"]
    return []


def _coverage_constraints(scores: dict | None) -> list[str]:
    """Reasons the evidence behind the score is too thin for an unqualified buy."""
    flags = (scores or {}).get("flags") or []
    if "low_data_coverage" not in flags:
        return []

    coverage = (scores or {}).get("coverage", {}).get("composite")
    if isinstance(coverage, (int, float)):
        return [f"only {coverage:.0%} of the scoring evidence was available"]
    return ["the score rests on incomplete evidence"]


def resolve_final_action(
    llm_action, eligibility: dict | None, scores: dict | None = None
) -> dict:
    """Resolve the action a report may display.

    Returns the decision plus everything needed to explain it:
        action      — what the report shows
        llm_action   — what Pass 2 actually said, always preserved
        capped       — whether `action` differs from `llm_action`
        ceiling      — the cap applied, if any
        constraints  — why, in reader-facing language

    `constraints` is populated whenever the verdict or the evidence is not
    clean, even if no cap was needed — a `hold` on a failed gate is already
    below the ceiling, but the reader should still see the reason.
    """
    constraints = _eligibility_constraints(eligibility) + _coverage_constraints(scores)

    decision = {
        "action": llm_action,
        "llm_action": llm_action,
        "capped": False,
        "ceiling": None,
        "constraints": constraints,
    }

    if llm_action is None or not constraints:
        return decision

    # An unrecognised action cannot be shown to be within the ceiling, so it
    # is capped rather than trusted.
    rank = _rank(llm_action)
    if rank is None or rank > _rank(CAP_CEILING):
        decision["action"] = CAP_CEILING
        decision["capped"] = True
        decision["ceiling"] = CAP_CEILING

    return decision


def resolve_for_result(result) -> dict | None:
    """The decision for a whole AnalysisResult, or None when there is no LLM view.

    The single derivation of "which action does this result get", so the
    service and the render boundary cannot drift apart on the answer. Reads
    only `llm_analysis`, `eligibility` and `scores` — never a stored
    `final_action`, which is an output of this function and must never
    become an input to it.

    Recomputing rather than trusting a stored decision matters because
    `final_action` is a mutable field that is also serialised into reports:
    anything that re-evaluates eligibility or rescores after Stage 4.5 leaves
    it stale, and a stale decision is exactly as dangerous as an absent one.
    The function is pure over plain dicts, so recomputing costs nothing.
    """
    llm = getattr(result, "llm_analysis", None)
    if not llm or llm.get("skipped"):
        return None

    p2 = llm.get("pass2") or {}
    if p2.get("error") or p2.get("skipped"):
        return None

    return resolve_final_action(
        p2.get("suggested_action"),
        getattr(result, "eligibility", None),
        getattr(result, "scores", None),
    )
