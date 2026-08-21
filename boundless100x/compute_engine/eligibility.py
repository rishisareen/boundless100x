"""100x eligibility gates — the conjunctive companion to the additive composite.

The SQGLP composite is a weighted mean, so strong quality can outvote a
disqualifying size or entry price. The 100x evidence base describes conditions
that are jointly necessary: a company too large to multiply, priced for
perfection, or unable to earn on new capital will not compound a hundredfold
however good the rest of its profile looks. These gates answer that question
separately, and never fold back into the composite.

Gates are declared in `metrics/registry.yaml` so tuning a threshold stays a
config edit, matching the metric registry's own pattern.
"""

import logging

logger = logging.getLogger(__name__)

COMPARATORS = {
    "lt": lambda value, threshold: value < threshold,
    "lte": lambda value, threshold: value <= threshold,
    "gt": lambda value, threshold: value > threshold,
    "gte": lambda value, threshold: value >= threshold,
}

# Shipped defaults, mirrored in registry.yaml. Starting points, expected to be
# tuned once the backtest provides evidence.
DEFAULT_GATES = {
    "size": {
        "label": "Size headroom",
        "rationale": "A hundredfold move needs room to grow into.",
        "conditions": [
            {"metric": "market_cap", "comparator": "lt", "threshold": 30000}
        ],
    },
    "price": {
        "label": "Entry price sanity",
        "rationale": "Growth already paid for cannot be earned twice.",
        "mode": "any",
        "conditions": [
            {"metric": "trailing_peg", "comparator": "lt", "threshold": 2.0},
            {"metric": "peg_ratio", "comparator": "lt", "threshold": 1.5},
        ],
        "veto_flags": ["reverse_dcf_overpriced"],
        # Metrics expected to emit the veto flags. If they are unavailable, the
        # absence of the flag proves nothing and the gate is indeterminate.
        "veto_sources": ["reverse_dcf_growth"],
    },
    "reinvestment": {
        "label": "Incremental returns",
        "rationale": "Compounding requires new capital to earn, not just sit.",
        "conditions": [
            {"metric": "roiic", "comparator": "gte", "threshold": 15.0}
        ],
        # Consulted ONLY when every primary condition above reads
        # indeterminate. See `_evaluate_gate`.
        "fallback_conditions": [
            {"metric": "roe_5yr_avg", "comparator": "gte", "threshold": 15.0}
        ],
    },
}


def effective_gates(gates: dict | None) -> dict:
    """The gates that will actually be applied for a given registry section.

    A registry with no `eligibility_gates` section falls back to the shipped
    `DEFAULT_GATES`, so "no gates declared" never means "no gates enforced".
    Callers that need to describe the gate regime — the registry hash stamped
    on every score-history row — must resolve it through here rather than
    reading the raw section, or a run governed by the code-level defaults
    would be recorded as if it had been governed by an empty config.
    """
    return gates or DEFAULT_GATES


def _format_threshold(value) -> str:
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


class EligibilityEvaluator:
    """Evaluates declared gates against computed metrics."""

    def __init__(self, gates: dict | None = None):
        self.gates = gates if gates is not None else DEFAULT_GATES

    def evaluate(self, metrics: dict) -> dict:
        """Return the verdict plus per-gate detail.

        A gate whose inputs are missing or errored is `indeterminate` — never a
        silent pass, since an unknown is not evidence of eligibility.
        """
        results: dict[str, dict] = {}
        failed: list[str] = []
        indeterminate: list[str] = []

        for gate_id, spec in self.gates.items():
            detail = self._evaluate_gate(spec, metrics)
            results[gate_id] = detail
            if detail["passed"] is False:
                failed.append(gate_id)
            elif detail["passed"] is None:
                indeterminate.append(gate_id)

        if failed:
            verdict, eligible = "not_eligible", False
        elif indeterminate:
            verdict, eligible = "indeterminate", None
        else:
            verdict, eligible = "eligible", True

        return {
            "eligible": eligible,
            "verdict": verdict,
            "gates": results,
            "failed": failed,
            "indeterminate": indeterminate,
        }

    def _evaluate_gate(self, spec: dict, metrics: dict) -> dict:
        label = spec.get("label", "Gate")
        mode = spec.get("mode", "all")
        conditions = spec.get("conditions", []) or []

        detail = {
            "label": label,
            "rationale": spec.get("rationale", ""),
            "passed": None,
            "reason": "",
            "conditions": [],
        }

        # A veto flag disqualifies regardless of how the ratios read.
        veto_flags = spec.get("veto_flags", []) or []
        for flag in veto_flags:
            carriers = [
                mid for mid, result in metrics.items()
                if getattr(result, "flags", None) and flag in result.flags
            ]
            if carriers:
                detail["passed"] = False
                detail["reason"] = f"{label} vetoed by {flag} on {', '.join(sorted(carriers))}"
                return detail

        # No metric carried the veto — but that is only reassuring if the metric
        # that would have emitted it actually ran. A veto whose source errored
        # reads indeterminate, matching how missing conditions are handled.
        veto_sources = spec.get("veto_sources", []) or []
        if veto_flags and veto_sources:
            unavailable = []
            for mid in veto_sources:
                result = metrics.get(mid)
                if result is None or not getattr(result, "ok", False) or result.value is None:
                    unavailable.append(mid)
            if unavailable:
                detail["passed"] = None
                detail["reason"] = (
                    f"{label} indeterminate: veto source(s) "
                    f"{', '.join(sorted(unavailable))} unavailable — absence of "
                    f"{', '.join(veto_flags)} cannot be confirmed"
                )
                return detail

        outcomes = []
        for condition in conditions:
            outcomes.append(self._evaluate_condition(condition, metrics))
        detail["conditions"] = outcomes

        verdicts = [o["passed"] for o in outcomes]
        if not verdicts:
            detail["reason"] = f"{label} has no conditions declared"
            return detail

        # A second-best measure, consulted ONLY when the primary one could not
        # be read at all — never when it read and disappointed.
        #
        # The case this exists for: `roiic` is undefined whenever the capital
        # base shrinks, which for a lender in run-off is not a data gap but a
        # permanent state. EDELWEISS therefore read `indeterminate` on this
        # gate in every run it will ever have, and an indeterminate verdict
        # caps the displayed action forever — a company can be refused for
        # good on a question the pipeline is structurally unable to ask. A
        # determinate "does not earn enough on equity to multiply a
        # hundredfold" is both truer and actionable.
        #
        # The narrowness is the safety. Firing only on all-indeterminate means
        # this can never overturn a primary condition that actually failed,
        # and can never let a company through on the softer of two tests when
        # the harder one was available: for every company whose ROIIC computes,
        # this branch is dead code.
        if all(v is None for v in verdicts):
            fallback_spec = spec.get("fallback_conditions", []) or []
            fallback = [
                self._evaluate_condition(condition, metrics)
                for condition in fallback_spec
            ]
            fallback_verdicts = [o["passed"] for o in fallback]
            if fallback_verdicts and not all(v is None for v in fallback_verdicts):
                detail["conditions"] = outcomes + fallback
                detail["fallback_used"] = True
                passed = self._combine(fallback_verdicts, mode)
                detail["passed"] = passed
                primary = ", ".join(
                    o.get("detail", "") for o in outcomes if o.get("detail")
                )
                detail["reason"] = (
                    f"{self._summarise(label, passed, fallback, mode)} "
                    f"(fallback measure — {primary})"
                )
                return detail

        detail["passed"] = self._combine(verdicts, mode)
        detail["reason"] = self._summarise(label, detail["passed"], outcomes, mode)
        return detail

    @staticmethod
    def _combine(verdicts: list, mode: str):
        """Fold per-condition verdicts into one, three-valued throughout."""
        if mode == "any":
            if any(v is True for v in verdicts):
                return True
            if any(v is None for v in verdicts):
                return None
            return False
        if any(v is False for v in verdicts):
            return False
        if any(v is None for v in verdicts):
            return None
        return True

    def _evaluate_condition(self, condition: dict, metrics: dict) -> dict:
        metric_id = condition.get("metric")
        comparator = condition.get("comparator", "lt")
        threshold = condition.get("threshold")

        outcome = {
            "metric": metric_id,
            "comparator": comparator,
            "threshold": threshold,
            "value": None,
            "passed": None,
        }

        result = metrics.get(metric_id)
        if result is None:
            outcome["detail"] = f"{metric_id} not computed"
            return outcome
        if not getattr(result, "ok", False) or result.value is None:
            outcome["detail"] = f"{metric_id} unavailable: {getattr(result, 'error', 'no value')}"
            return outcome
        if not isinstance(result.value, (int, float)):
            outcome["detail"] = f"{metric_id} is not numeric"
            return outcome

        compare = COMPARATORS.get(comparator)
        if compare is None:
            logger.warning(f"Unknown comparator '{comparator}' for {metric_id}")
            outcome["detail"] = f"unknown comparator '{comparator}'"
            return outcome

        outcome["value"] = float(result.value)
        outcome["passed"] = bool(compare(outcome["value"], threshold))
        outcome["detail"] = (
            f"{metric_id} {outcome['value']:,.2f} {comparator} {_format_threshold(threshold)}"
        )
        return outcome

    @staticmethod
    def _summarise(label: str, passed, outcomes: list[dict], mode: str) -> str:
        joiner = " or " if mode == "any" else " and "
        rendered = joiner.join(o.get("detail", "") for o in outcomes if o.get("detail"))
        if passed is True:
            return f"{label} met: {rendered}"
        if passed is False:
            return f"{label} not met: {rendered}"
        return f"{label} indeterminate: {rendered}"
