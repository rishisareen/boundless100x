"""Evaluates the fast lane's entry gates against a scored analysis.

The third sibling. `compute_engine/eligibility.py` asks "could this plausibly
100x?" and `lifecycle/evaluator.py` asks "what transition is due?"; these gates
ask "does this qualify for the fast lane, right now?" — a question the design
doc keeps in the lifecycle section rather than folding into the compute
engine's eligibility registry, because conflating a permanent property of a
company with a present-tense entry decision would leave neither readable.

Mirroring the siblings is deliberate and load-bearing, not incidental reuse:
the same imported `COMPARATORS`, the same three-valued outcome, the same
per-condition `detail` strings feeding a human-readable reason. There is one
indeterminate semantic in this system rather than three, and the rule that
carries the most weight is inherited unchanged: **a gate whose inputs are
missing is indeterminate, never a silent pass.** On this side of the system a
cleared gate is what lets capital move, so "we could not check" must never
render as "checked and fine".

Two things are new here. The condition set is wider than `EligibilityEvaluator`
supports — a lane gate needs `score` and flag conditions, which
`TriggerEvaluator` already has — plus one kind neither sibling has:
`catalyst_status`, which reads the owner-recorded catalyst on the watchlist
entry rather than a computed metric. And that kind carries a distinction the
others do not have to make. An entry with no catalyst has been looked at and
has none, which is a plain failure; **no watchlist context at all is an
unknown**. Since an empty catalyst dict is falsy, only an explicit `is None`
check keeps those apart, and collapsing them would let a company nobody has
assessed read exactly like one assessed and found wanting.
"""

import logging
from pathlib import Path

import yaml

from boundless100x.compute_engine.eligibility import COMPARATORS, _format_threshold
from boundless100x.watchlist import CATALYST_STATUSES

logger = logging.getLogger(__name__)

DEFAULT_LANE_GATES_PATH = Path(__file__).parent / "lane_gates.yaml"

# Condition kinds, identified by which key the YAML entry carries.
_METRIC = "metric"
_SCORE = "score"
_FLAG_PRESENT = "flag_present"
_FLAG_ABSENT = "flag_absent"
_CATALYST = "catalyst_status"

CONDITION_KINDS = (_METRIC, _SCORE, _FLAG_PRESENT, _FLAG_ABSENT, _CATALYST)

# Shipped defaults, mirrored in lane_gates.yaml — the same arrangement
# `DEFAULT_GATES` has with registry.yaml, so a missing or unreadable registry
# file means the shipped regime rather than no gates at all. Every threshold is
# a STARTING POINT awaiting Phase 4/5 simulator evidence.
DEFAULT_LANE_GATES = {
    "quality_floor": {
        "label": "Quality floor",
        "rationale": "A discount is only worth acting on in a company worth owning.",
        "conditions": [
            {"score": "composite", "comparator": "gte", "threshold": 5.5}
        ],
    },
    "valuation_discount": {
        "label": "Valuation discount",
        "rationale": "A re-rating needs something to re-rate from.",
        "mode": "any",
        "conditions": [
            {"metric": "pe_vs_historical", "comparator": "lte", "threshold": 50},
            {
                "flag_present": "rerating_headroom_favourable",
                "sources": ["rerating_headroom"],
            },
        ],
    },
    "growth_intact": {
        "label": "Growth intact",
        "rationale": (
            "The discount must be the market's mistake, not the company's decline."
        ),
        "conditions": [
            {"metric": "ttm_growth_vs_cagr", "comparator": "gte", "threshold": 0},
            {
                "flag_absent": "growth_quality_risky",
                "sources": ["growth_quality_grade"],
            },
        ],
    },
    "institutional_accumulation": {
        "label": "Institutional accumulation",
        "rationale": "Somebody obliged to disclose is already buying.",
        "conditions": [
            {
                "metric": "institutional_accumulation_streak",
                "comparator": "gte",
                "threshold": 2,
            }
        ],
    },
    "catalyst_identified": {
        "label": "Catalyst identified",
        "rationale": (
            "A re-rating thesis with no named catalyst is a hope with a deadline."
        ),
        "conditions": [{"catalyst_status": "active"}],
    },
    "liquidity_floor": {
        "label": "Liquidity floor",
        "rationale": "A position that cannot be exited at size is not one to enter.",
        "conditions": [
            {"metric": "daily_turnover_ratio", "comparator": "gte", "threshold": 0.02}
        ],
    },
}


def load_lane_gates(path: str | Path | None = None) -> dict:
    """Read the declared lane gates. Returns {gate_id: spec}."""
    target = Path(path) if path else DEFAULT_LANE_GATES_PATH
    if not target.exists():
        logger.warning(f"No lane-gate registry at {target}")
        return {}
    loaded = yaml.safe_load(target.read_text()) or {}
    return loaded.get("lane_gates", {}) or {}


def effective_lane_gates(gates: dict | None) -> dict:
    """The gates that will actually be applied.

    An empty or absent registry falls back to the shipped defaults, so "no
    gates declared" never means "no gates enforced" — the same rule
    `eligibility.effective_gates` states for the 100x gates, and for the same
    reason: an entry decision taken under the code-level defaults must not be
    describable as one taken under an empty config.
    """
    return gates or DEFAULT_LANE_GATES


def validate_lane_gates(gates: dict, known_metric_ids: set[str] | None = None) -> list[str]:
    """Return a list of registry errors — empty when the registry is sound.

    Startup validation exists because the failure it prevents is silent: a gate
    naming a metric that does not exist would read indeterminate forever, and a
    lane no company can ever enter looks exactly like a lane with no qualifying
    candidates.
    """
    errors: list[str] = []

    for gate_id, spec in gates.items():
        if not isinstance(spec, dict):
            errors.append(f"{gate_id}: spec must be a mapping")
            continue

        mode = spec.get("mode", "all")
        if mode not in ("all", "any"):
            errors.append(f"{gate_id}: mode must be 'all' or 'any', got {mode!r}")

        conditions = spec.get("conditions") or []
        if not conditions:
            errors.append(f"{gate_id}: no conditions declared")

        for index, condition in enumerate(conditions):
            where = f"{gate_id}.conditions[{index}]"
            if not isinstance(condition, dict):
                errors.append(f"{where}: must be a mapping")
                continue

            kinds = [k for k in CONDITION_KINDS if k in condition]
            if len(kinds) != 1:
                errors.append(
                    f"{where}: expected exactly one of {', '.join(CONDITION_KINDS)}, "
                    f"found {kinds or 'none'}"
                )
                continue
            kind = kinds[0]

            if kind in (_METRIC, _SCORE):
                comparator = condition.get("comparator")
                if comparator not in COMPARATORS:
                    errors.append(f"{where}: unknown comparator {comparator!r}")
                if condition.get("threshold") is None:
                    errors.append(f"{where}: threshold is required")

            if kind == _METRIC and known_metric_ids is not None:
                if condition[_METRIC] not in known_metric_ids:
                    errors.append(f"{where}: unknown metric id {condition[_METRIC]!r}")

            if kind == _CATALYST and condition[_CATALYST] not in CATALYST_STATUSES:
                errors.append(
                    f"{where}: unknown catalyst status {condition[_CATALYST]!r}"
                )

            sources = condition.get("sources")
            if sources is not None and known_metric_ids is not None:
                for source in sources:
                    if source not in known_metric_ids:
                        errors.append(f"{where}: unknown source metric id {source!r}")

    return errors


class LaneGateEvaluator:
    """Evaluates the declared fast-lane gates against one company's readings."""

    def __init__(
        self,
        gates: dict | None = None,
        known_metric_ids: set[str] | None = None,
    ):
        self.gates = effective_lane_gates(
            gates if gates is not None else load_lane_gates()
        )
        errors = validate_lane_gates(self.gates, known_metric_ids)
        if errors:
            for error in errors:
                logger.error(f"  LANE GATE REGISTRY ERROR: {error}")
            # The errors travel in the message as well as the log, unlike the
            # sibling evaluators': those validate one shipped YAML file, while
            # this is also constructed from caller-supplied gates in tests and
            # by future callers, where a bare count leaves the reader with
            # nothing to act on.
            raise ValueError(
                f"Lane-gate registry validation failed: {len(errors)} errors — "
                f"{'; '.join(errors)}"
            )

    def evaluate(
        self,
        metrics: dict,
        scores: dict | None = None,
        catalyst: dict | None = None,
    ) -> dict:
        """Return the lane verdict plus per-gate detail.

        The verdict vocabulary is this context's own —
        `qualifies`/`not_qualified`/`indeterminate` — because "eligible" is
        already taken by the 100x question and a company can easily be one and
        not the other. The three-valued shape underneath is identical.

        `catalyst` defaults to None meaning *no watchlist context was supplied*,
        which is distinct from an entry carrying no catalyst.
        """
        results: dict[str, dict] = {}
        failed: list[str] = []
        indeterminate: list[str] = []

        for gate_id, spec in self.gates.items():
            detail = self._evaluate_gate(spec, metrics, scores, catalyst)
            results[gate_id] = detail
            if detail["passed"] is False:
                failed.append(gate_id)
            elif detail["passed"] is None:
                indeterminate.append(gate_id)

        # A failure settles the question even when another gate is unknown: one
        # gate is already known to be unmet, and no later reading can unmeet it.
        if failed:
            verdict, qualifies = "not_qualified", False
        elif indeterminate:
            verdict, qualifies = "indeterminate", None
        else:
            verdict, qualifies = "qualifies", True

        return {
            "qualifies": qualifies,
            "verdict": verdict,
            "gates": results,
            "failed": failed,
            "indeterminate": indeterminate,
        }

    def _evaluate_gate(
        self, spec: dict, metrics: dict, scores: dict | None, catalyst: dict | None
    ) -> dict:
        label = spec.get("label", "Gate")
        mode = spec.get("mode", "all")

        detail = {
            "label": label,
            "rationale": spec.get("rationale", ""),
            "passed": None,
            "reason": "",
            "conditions": [],
        }

        outcomes = [
            self._evaluate_condition(condition, metrics, scores, catalyst)
            for condition in (spec.get("conditions") or [])
        ]
        detail["conditions"] = outcomes

        verdicts = [o["passed"] for o in outcomes]
        if not verdicts:
            detail["reason"] = f"{label} has no conditions declared"
            return detail

        if mode == "any":
            if any(v is True for v in verdicts):
                detail["passed"] = True
            elif any(v is None for v in verdicts):
                detail["passed"] = None
            else:
                detail["passed"] = False
        else:
            if any(v is False for v in verdicts):
                detail["passed"] = False
            elif any(v is None for v in verdicts):
                detail["passed"] = None
            else:
                detail["passed"] = True

        detail["reason"] = self._summarise(label, detail["passed"], outcomes, mode)
        return detail

    def _evaluate_condition(
        self, condition: dict, metrics: dict, scores: dict | None, catalyst: dict | None
    ) -> dict:
        if _METRIC in condition:
            return self._evaluate_metric(condition, metrics)
        if _SCORE in condition:
            return self._evaluate_score(condition, scores)
        if _FLAG_PRESENT in condition or _FLAG_ABSENT in condition:
            return self._evaluate_flag(condition, metrics)
        if _CATALYST in condition:
            return self._evaluate_catalyst(condition, catalyst)

        return {
            "kind": "unknown",
            "passed": None,
            "detail": f"unrecognised condition {sorted(condition)}",
        }

    def _evaluate_metric(self, condition: dict, metrics: dict) -> dict:
        metric_id = condition[_METRIC]
        comparator = condition.get("comparator", "gte")
        threshold = condition.get("threshold")

        outcome = {
            "kind": _METRIC,
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
            outcome["detail"] = (
                f"{metric_id} unavailable: {getattr(result, 'error', 'no value')}"
            )
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
            f"{metric_id} {outcome['value']:,.2f} {comparator} "
            f"{_format_threshold(threshold)}"
        )
        return outcome

    def _evaluate_score(self, condition: dict, scores: dict | None) -> dict:
        """A condition on the composite or an element score."""
        field = condition[_SCORE]
        comparator = condition.get("comparator", "gte")
        threshold = condition.get("threshold")

        outcome = {
            "kind": _SCORE,
            "score": field,
            "comparator": comparator,
            "threshold": threshold,
            "value": None,
            "passed": None,
        }

        if not scores:
            outcome["detail"] = "scores unavailable"
            return outcome

        value = (
            scores.get("composite")
            if field == "composite"
            else (scores.get("elements") or {}).get(field)
        )
        if value is None:
            outcome["detail"] = f"score '{field}' unavailable"
            return outcome

        compare = COMPARATORS.get(comparator)
        if compare is None:
            outcome["detail"] = f"unknown comparator '{comparator}'"
            return outcome

        outcome["value"] = float(value)
        outcome["passed"] = bool(compare(outcome["value"], threshold))
        outcome["detail"] = (
            f"score {field} {outcome['value']:,.2f} {comparator} "
            f"{_format_threshold(threshold)}"
        )
        return outcome

    def _evaluate_flag(self, condition: dict, metrics: dict) -> dict:
        """Flag presence, with the absence caveat the 100x price gate established.

        A flag that is not present proves nothing unless the metric that would
        have emitted it actually ran — so `sources` names those metrics, and an
        unavailable source makes the condition indeterminate rather than a pass
        earned by silence.
        """
        want_present = _FLAG_PRESENT in condition
        flag = condition[_FLAG_PRESENT if want_present else _FLAG_ABSENT]
        sources = condition.get("sources") or []

        outcome = {
            "kind": _FLAG_PRESENT if want_present else _FLAG_ABSENT,
            "flag": flag,
            "passed": None,
        }

        carriers = sorted(
            mid
            for mid, result in metrics.items()
            if getattr(result, "flags", None) and flag in result.flags
        )
        if carriers:
            outcome["carriers"] = carriers
            outcome["passed"] = want_present
            outcome["detail"] = f"{flag} present on {', '.join(carriers)}"
            return outcome

        unavailable = [
            mid
            for mid in sources
            if metrics.get(mid) is None
            or not getattr(metrics.get(mid), "ok", False)
            or metrics[mid].value is None
        ]
        if unavailable:
            outcome["detail"] = (
                f"{flag} absence unconfirmed: source(s) "
                f"{', '.join(sorted(unavailable))} unavailable"
            )
            return outcome

        outcome["passed"] = not want_present
        outcome["detail"] = f"{flag} absent"
        return outcome

    def _evaluate_catalyst(self, condition: dict, catalyst: dict | None) -> dict:
        """The owner-recorded catalyst, which no metric can compute.

        §9.2 calls this one "recorded, not scored". The three cases must stay
        distinguishable: an active catalyst passes, an entry carrying none (or
        a spent one) fails because somebody looked and there is nothing to wait
        for, and no watchlist context at all is unknown. Only `is None`
        separates the last two — an empty dict is falsy and a truthiness check
        would quietly turn "nobody assessed this company" into "assessed and
        found wanting".
        """
        expected = condition[_CATALYST]
        outcome = {
            "kind": _CATALYST,
            "expected": expected,
            "value": None,
            "passed": None,
        }

        if catalyst is None:
            outcome["detail"] = (
                "no watchlist entry supplied — catalyst status unknown"
            )
            return outcome

        status = catalyst.get("status")
        outcome["value"] = status
        outcome["passed"] = status == expected
        if outcome["passed"]:
            outcome["detail"] = f"catalyst is {status}"
        elif status:
            outcome["detail"] = f"catalyst is {status}, wanted {expected}"
        else:
            outcome["detail"] = "no catalyst recorded on this entry"
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
