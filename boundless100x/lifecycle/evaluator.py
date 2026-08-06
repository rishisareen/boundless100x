"""Evaluates declared lifecycle transitions against a scored analysis.

This is the eligibility-gate evaluator's sibling and deliberately mirrors it:
the same comparator table, the same three-valued outcome, the same
per-condition `detail` strings feeding a human-readable reason. A reader who
understands why a gate failed understands why a transition fired, and there is
one indeterminate semantic in the system rather than two.

The rule that carries the most weight is inherited unchanged: **a trigger
whose inputs are missing is indeterminate, never fired and never quietly
false**. A kill-switch that cannot be evaluated must surface as unknown, since
"we could not check" is not the same as "the thesis is fine".
"""

import logging
from pathlib import Path

import yaml

from boundless100x.compute_engine.eligibility import COMPARATORS, _format_threshold
from boundless100x.lifecycle import states as lifecycle_states

logger = logging.getLogger(__name__)

DEFAULT_TRIGGERS_PATH = Path(__file__).parent / "triggers.yaml"

# Condition kinds, identified by which key the YAML entry carries.
_METRIC = "metric"
_SCORE = "score"
_VERDICT = "verdict"
_FLAG_PRESENT = "flag_present"
_FLAG_ABSENT = "flag_absent"
_CHECKPOINT = "checkpoint"

CONDITION_KINDS = (_METRIC, _SCORE, _VERDICT, _FLAG_PRESENT, _FLAG_ABSENT, _CHECKPOINT)


def load_triggers(path: str | Path | None = None) -> dict:
    """Read the declared triggers. Returns {trigger_id: spec}."""
    target = Path(path) if path else DEFAULT_TRIGGERS_PATH
    if not target.exists():
        logger.warning(f"No trigger registry at {target}")
        return {}
    loaded = yaml.safe_load(target.read_text()) or {}
    return loaded.get("triggers", {}) or {}


def validate_triggers(
    triggers: dict, known_metric_ids: set[str] | None = None
) -> list[str]:
    """Return a list of registry errors — empty when the registry is sound.

    Startup validation exists because the failure it prevents is silent: a
    trigger naming a metric that does not exist would evaluate indeterminate
    forever, and a kill-switch that never fires looks exactly like a thesis
    that never broke.
    """
    errors: list[str] = []

    for trigger_id, spec in triggers.items():
        if not isinstance(spec, dict):
            errors.append(f"{trigger_id}: spec must be a mapping")
            continue

        destination = spec.get("to")
        if not lifecycle_states.is_state(destination):
            errors.append(f"{trigger_id}: unknown destination state {destination!r}")

        origins = spec.get("from") or []
        if isinstance(origins, str):
            origins = [origins]
        for origin in origins:
            if origin != "any" and not lifecycle_states.is_state(origin):
                errors.append(f"{trigger_id}: unknown origin state {origin!r}")

        mode = spec.get("mode", "all")
        if mode not in ("all", "any"):
            errors.append(f"{trigger_id}: mode must be 'all' or 'any', got {mode!r}")

        conditions = spec.get("conditions") or []
        if not conditions:
            errors.append(f"{trigger_id}: no conditions declared")

        for index, condition in enumerate(conditions):
            where = f"{trigger_id}.conditions[{index}]"
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

            if kind in (_METRIC, _SCORE, _CHECKPOINT):
                comparator = condition.get("comparator")
                if comparator not in COMPARATORS:
                    errors.append(f"{where}: unknown comparator {comparator!r}")
                if condition.get("threshold") is None:
                    errors.append(f"{where}: threshold is required")

            if kind == _METRIC and known_metric_ids is not None:
                if condition[_METRIC] not in known_metric_ids:
                    errors.append(f"{where}: unknown metric id {condition[_METRIC]!r}")

            if kind == _VERDICT:
                expected = condition[_VERDICT]
                if expected not in ("eligible", "not_eligible", "indeterminate"):
                    errors.append(f"{where}: unknown verdict {expected!r}")

            persist = condition.get("persist_years")
            if persist is not None and (not isinstance(persist, int) or persist < 2):
                errors.append(f"{where}: persist_years must be an integer >= 2")

            sources = condition.get("sources")
            if sources is not None and known_metric_ids is not None:
                for source in sources:
                    if source not in known_metric_ids:
                        errors.append(f"{where}: unknown source metric id {source!r}")

    return errors


class TriggerEvaluator:
    """Evaluates the transitions declared for a company's current state."""

    def __init__(
        self,
        triggers: dict | None = None,
        known_metric_ids: set[str] | None = None,
    ):
        self.triggers = triggers if triggers is not None else load_triggers()
        errors = validate_triggers(self.triggers, known_metric_ids)
        if errors:
            for error in errors:
                logger.error(f"  TRIGGER REGISTRY ERROR: {error}")
            raise ValueError(f"Trigger registry validation failed: {len(errors)} errors")

    def applicable(self, state: str) -> dict:
        """Triggers declared to fire from this state."""
        applicable = {}
        for trigger_id, spec in self.triggers.items():
            origins = spec.get("from") or ["any"]
            if isinstance(origins, str):
                origins = [origins]
            if "any" in origins or state in origins:
                applicable[trigger_id] = spec
        return applicable

    def evaluate(
        self,
        state: str,
        metrics: dict,
        scores: dict | None = None,
        eligibility: dict | None = None,
        checkpoint_results: dict | None = None,
    ) -> dict:
        """Evaluate every trigger applicable to `state`.

        Returns the per-trigger detail plus the ids that fired and the ids
        that could not be decided. Both lists matter: the second is what stops
        an unevaluable kill-switch from reading as silence.
        """
        results: dict[str, dict] = {}
        fired: list[str] = []
        indeterminate: list[str] = []

        for trigger_id, spec in self.applicable(state).items():
            detail = self._evaluate_trigger(
                spec, metrics, scores, eligibility, checkpoint_results
            )
            results[trigger_id] = detail
            if detail["fired"] is True:
                fired.append(trigger_id)
            elif detail["fired"] is None:
                indeterminate.append(trigger_id)

        return {
            "state": state,
            "triggers": results,
            "fired": fired,
            "indeterminate": indeterminate,
        }

    def _evaluate_trigger(
        self,
        spec: dict,
        metrics: dict,
        scores: dict | None,
        eligibility: dict | None,
        checkpoint_results: dict | None,
    ) -> dict:
        label = spec.get("label", "Trigger")
        mode = spec.get("mode", "all")

        detail = {
            "label": label,
            "rationale": spec.get("rationale", ""),
            "to": spec.get("to"),
            "fired": None,
            "reason": "",
            "conditions": [],
        }

        outcomes = [
            self._evaluate_condition(
                condition, metrics, scores, eligibility, checkpoint_results
            )
            for condition in (spec.get("conditions") or [])
        ]
        detail["conditions"] = outcomes

        verdicts = [o["passed"] for o in outcomes]
        if not verdicts:
            detail["reason"] = f"{label} has no conditions declared"
            return detail

        if mode == "any":
            if any(v is True for v in verdicts):
                detail["fired"] = True
            elif any(v is None for v in verdicts):
                detail["fired"] = None
            else:
                detail["fired"] = False
        else:
            if any(v is False for v in verdicts):
                detail["fired"] = False
            elif any(v is None for v in verdicts):
                detail["fired"] = None
            else:
                detail["fired"] = True

        detail["reason"] = self._summarise(label, detail["fired"], outcomes, mode)
        return detail

    def _evaluate_condition(
        self,
        condition: dict,
        metrics: dict,
        scores: dict | None,
        eligibility: dict | None,
        checkpoint_results: dict | None,
    ) -> dict:
        if _METRIC in condition:
            if condition.get("persist_years"):
                return self._evaluate_series(condition, metrics)
            return self._evaluate_metric(condition, metrics)
        if _SCORE in condition:
            return self._evaluate_score(condition, scores)
        if _VERDICT in condition:
            return self._evaluate_verdict(condition, eligibility)
        if _FLAG_PRESENT in condition or _FLAG_ABSENT in condition:
            return self._evaluate_flag(condition, metrics)
        if _CHECKPOINT in condition:
            return self._evaluate_checkpoint(condition, checkpoint_results)

        return {
            "kind": "unknown",
            "passed": None,
            "detail": f"unrecognised condition {sorted(condition)}",
        }

    def _evaluate_metric(self, condition: dict, metrics: dict) -> dict:
        metric_id = condition[_METRIC]
        comparator = condition.get("comparator", "lt")
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

    def _evaluate_series(self, condition: dict, metrics: dict) -> dict:
        """A rule that must hold for N consecutive periods.

        The registry has no `roce_latest`; it has `roce_5yr_avg`, whose mean
        cannot express "below 15% for two consecutive years" — but whose
        `raw_series` carries the yearly values that can. A series shorter than
        the window is indeterminate: two bad years cannot be confirmed from
        one year of data.
        """
        metric_id = condition[_METRIC]
        comparator = condition.get("comparator", "lt")
        threshold = condition.get("threshold")
        window = condition["persist_years"]

        outcome = {
            "kind": _METRIC,
            "metric": metric_id,
            "comparator": comparator,
            "threshold": threshold,
            "persist_years": window,
            "value": None,
            "passed": None,
        }

        result = metrics.get(metric_id)
        if result is None:
            outcome["detail"] = f"{metric_id} not computed"
            return outcome

        series = [v for v in (getattr(result, "raw_series", None) or []) if v is not None]
        if len(series) < window:
            outcome["detail"] = (
                f"{metric_id} has {len(series)} periods, needs {window} to judge "
                f"persistence"
            )
            return outcome

        compare = COMPARATORS.get(comparator)
        if compare is None:
            logger.warning(f"Unknown comparator '{comparator}' for {metric_id}")
            outcome["detail"] = f"unknown comparator '{comparator}'"
            return outcome

        recent = [float(v) for v in series[-window:]]
        outcome["value"] = recent[-1]
        outcome["series"] = recent
        outcome["passed"] = all(compare(v, threshold) for v in recent)
        rendered = ", ".join(f"{v:,.2f}" for v in recent)
        outcome["detail"] = (
            f"{metric_id} last {window} periods [{rendered}] "
            f"{'all' if outcome['passed'] else 'not all'} {comparator} "
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

    def _evaluate_verdict(self, condition: dict, eligibility: dict | None) -> dict:
        expected = condition[_VERDICT]
        outcome = {"kind": _VERDICT, "expected": expected, "value": None, "passed": None}

        if not eligibility or not eligibility.get("verdict"):
            outcome["detail"] = "100x eligibility was not evaluated"
            return outcome

        actual = eligibility["verdict"]
        outcome["value"] = actual
        outcome["passed"] = actual == expected
        outcome["detail"] = f"eligibility verdict is {actual} (wanted {expected})"
        return outcome

    def _evaluate_flag(self, condition: dict, metrics: dict) -> dict:
        """Flag presence, with the absence caveat the price gate established.

        A flag that is not present proves nothing unless the metric that
        would have emitted it actually ran — so `sources` names those metrics
        and an unavailable source makes the condition indeterminate.
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

    def _evaluate_checkpoint(
        self, condition: dict, checkpoint_results: dict | None
    ) -> dict:
        """A condition on recorded checkpoint outcomes (see lifecycle.checkpoints).

        `missed` counts checkpoints that came due and were not met. Checkpoints
        that could not be evaluated are counted separately and never as
        misses — a data gap must not end a thesis.
        """
        field = condition[_CHECKPOINT]
        comparator = condition.get("comparator", "gte")
        threshold = condition.get("threshold")

        outcome = {
            "kind": _CHECKPOINT,
            "checkpoint": field,
            "comparator": comparator,
            "threshold": threshold,
            "value": None,
            "passed": None,
        }

        if not checkpoint_results:
            outcome["detail"] = "no checkpoints recorded"
            return outcome

        value = checkpoint_results.get(field)
        if value is None:
            outcome["detail"] = f"checkpoint summary '{field}' unavailable"
            return outcome

        compare = COMPARATORS.get(comparator)
        if compare is None:
            outcome["detail"] = f"unknown comparator '{comparator}'"
            return outcome

        outcome["value"] = float(value)
        outcome["passed"] = bool(compare(outcome["value"], threshold))
        outcome["detail"] = (
            f"checkpoints {field} {outcome['value']:,.0f} {comparator} "
            f"{_format_threshold(threshold)}"
        )
        return outcome

    @staticmethod
    def _summarise(label: str, fired, outcomes: list[dict], mode: str) -> str:
        joiner = " or " if mode == "any" else " and "
        rendered = joiner.join(o.get("detail", "") for o in outcomes if o.get("detail"))
        if fired is True:
            return f"{label} fired: {rendered}"
        if fired is False:
            return f"{label} not fired: {rendered}"
        return f"{label} indeterminate: {rendered}"
