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

Phase 3's second lane widens this in two directions without disturbing either
rule. Three condition kinds join the set — `lane_verdict` reads the fast lane's
gate result, `catalyst_status` reads owner judgement no metric can compute, and
`since_state_entry` reads the clock against the append-only state history — and
each restates the indeterminate rule in its own terms rather than inventing a
new one. And `lane` becomes a second axis of applicability alongside `from`,
carrying the same idiom: an absent key means "every lane", exactly as
`from: [any]` means "every origin state".
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from boundless100x.compute_engine.eligibility import COMPARATORS, _format_threshold
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.watchlist import CATALYST_STATUSES, LANES

logger = logging.getLogger(__name__)

DEFAULT_TRIGGERS_PATH = Path(__file__).parent / "triggers.yaml"

# Condition kinds, identified by which key the YAML entry carries.
_METRIC = "metric"
_SCORE = "score"
_VERDICT = "verdict"
_FLAG_PRESENT = "flag_present"
_FLAG_ABSENT = "flag_absent"
_CHECKPOINT = "checkpoint"
_LANE_VERDICT = "lane_verdict"
_CATALYST = "catalyst_status"
_SINCE_STATE_ENTRY = "since_state_entry"

CONDITION_KINDS = (
    _METRIC,
    _SCORE,
    _VERDICT,
    _FLAG_PRESENT,
    _FLAG_ABSENT,
    _CHECKPOINT,
    _LANE_VERDICT,
    _CATALYST,
    _SINCE_STATE_ENTRY,
)

# The fast lane's verdict vocabulary, deliberately not the 100x one: a company
# can easily be `eligible` for a hundredfold and `not_qualified` for a
# re-rating today, and one word covering both would hide exactly that. Stated
# here rather than imported because `lane_gates.py` declares it only inside
# `LaneGateEvaluator.evaluate`'s docstring — if that module ever exports a
# constant, this should read it instead of restating it.
LANE_VERDICTS = ("qualifies", "not_qualified", "indeterminate")

# Metrics whose `raw_series` is that metric's own quantity over time, in the
# same units as its threshold — the only ones `persist_years` can read.
#
# `raw_series` has no declared contract, and two metrics prove why this must be
# an allowlist rather than an open door:
#   roiic             carries the *capital employed* series (INR Cr) beside an
#                     incremental-return value (%), so `roiic persist_years: 2`
#                     would compare rupees against a percentage and never fire
#   pe_vs_historical  carries the historical P/E values beside a *percentile*,
#                     so the same rule would test P/E multiples against a
#                     0–100 percentile threshold
# Both would validate, run, and silently never trigger — a kill-switch that
# never fires is indistinguishable from a thesis that never broke. Adding a
# metric here means reading its implementation first.
SERIES_SAFE_METRICS = frozenset({
    "roce_5yr_avg",          # yearly RoCE %
    "roe_5yr_avg",           # yearly RoE %
    "operating_margin_5yr",  # yearly OPM %
})


def _as_date(value) -> date | None:
    """A calendar date from whatever the caller or the store happened to hold.

    This is the one place two time formats meet. `as_of` is a `date`, as it is
    throughout `lifecycle.checkpoints`; a `state_history` record's `at` is a
    full ISO datetime, because `watchlist._now()` writes `datetime.now()`.
    Anything unreadable comes back None so its caller can say *what* it could
    not read, rather than quietly becoming a date nobody supplied.
    """
    if isinstance(value, datetime):  # checked first — datetime subclasses date
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class _Inputs:
    """Everything a condition may read, carried as one value.

    Nine condition kinds now draw on eight distinct inputs, and threading each
    through `_evaluate_trigger` into `_evaluate_condition` as its own
    positional argument had stopped being readable somewhere around the fifth.
    Bundling them keeps the "absent means indeterminate" rule uniform too: an
    input nobody supplied is None on this record, and the branch that needs it
    is the one that explains what was missing.
    """

    metrics: dict
    scores: dict | None = None
    eligibility: dict | None = None
    checkpoint_results: dict | None = None
    lane_gate_result: dict | None = None
    catalyst: dict | None = None
    state_history: list | None = None
    as_of: date | None = None


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

        # An absent `lane` key means "every lane", so only a *declared* lane is
        # checked. A trigger scoped to a lane nobody runs would never be
        # evaluated at all, which is the same silence a nonexistent metric id
        # produces and is caught here for the same reason.
        lanes = spec.get("lane")
        if lanes is not None:
            if isinstance(lanes, str):
                lanes = [lanes]
            if not isinstance(lanes, list) or not lanes:
                errors.append(
                    f"{trigger_id}: lane must be a lane name or a non-empty list "
                    f"of them, got {spec.get('lane')!r}"
                )
            else:
                for lane in lanes:
                    if lane not in LANES:
                        errors.append(f"{trigger_id}: unknown lane {lane!r}")

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

            if kind in (_METRIC, _SCORE, _CHECKPOINT, _SINCE_STATE_ENTRY):
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

            if kind == _LANE_VERDICT and condition[_LANE_VERDICT] not in LANE_VERDICTS:
                errors.append(
                    f"{where}: unknown lane verdict {condition[_LANE_VERDICT]!r} — "
                    f"one of {', '.join(LANE_VERDICTS)}"
                )

            if kind == _CATALYST and condition[_CATALYST] not in CATALYST_STATUSES:
                errors.append(
                    f"{where}: unknown catalyst status {condition[_CATALYST]!r}"
                )

            # A time stop naming a state that does not exist would read "never
            # reached it" forever — an 18-month stop that can never come due
            # looks precisely like a position that never stalled.
            if kind == _SINCE_STATE_ENTRY:
                target = condition[_SINCE_STATE_ENTRY]
                if not lifecycle_states.is_state(target):
                    errors.append(f"{where}: unknown state {target!r}")

            persist = condition.get("persist_years")
            if persist is not None:
                if not isinstance(persist, int) or persist < 2:
                    errors.append(f"{where}: persist_years must be an integer >= 2")
                elif condition.get(_METRIC) not in SERIES_SAFE_METRICS:
                    errors.append(
                        f"{where}: persist_years is not available for "
                        f"{condition.get(_METRIC)!r} — its raw_series is not a series "
                        f"of its own values in threshold units. See SERIES_SAFE_METRICS."
                    )

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

    def applicable(self, state: str, lane: str | None = None) -> dict:
        """Triggers declared to fire from this state, in this lane.

        Two axes, one idiom. `from: [any]` — or no `from` at all — means every
        origin state; an absent `lane` key means every lane, and a declared one
        narrows the trigger to the lanes it names. The fast lane's kill-switches
        have no business being evaluated against a core compounder, and the
        core lane's patience triggers have none against a re-rating thesis.

        A caller supplying **no lane** gets everything. That is not a shortcut:
        filtering lane-scoped triggers out when the lane is unknown would make
        them unevaluable rather than unknown, and a kill-switch that never
        fires looks exactly like a thesis that never broke. An unknown lane is
        left to the conditions, which say so in their own words.
        """
        applicable = {}
        for trigger_id, spec in self.triggers.items():
            origins = spec.get("from") or ["any"]
            if isinstance(origins, str):
                origins = [origins]
            if not ("any" in origins or state in origins):
                continue

            lanes = spec.get("lane")
            if lanes is not None and lane is not None:
                if isinstance(lanes, str):
                    lanes = [lanes]
                if lane not in lanes:
                    continue

            applicable[trigger_id] = spec
        return applicable

    def evaluate(
        self,
        state: str,
        metrics: dict,
        scores: dict | None = None,
        eligibility: dict | None = None,
        checkpoint_results: dict | None = None,
        lane_gate_result: dict | None = None,
        catalyst: dict | None = None,
        state_history: list | None = None,
        lane: str | None = None,
        as_of: date | None = None,
    ) -> dict:
        """Evaluate every trigger applicable to `state` in `lane`.

        Returns the per-trigger detail plus the ids that fired and the ids
        that could not be decided. Both lists matter: the second is what stops
        an unevaluable kill-switch from reading as silence.

        Every optional input is optional in the same way: absent, it makes the
        conditions that need it read indeterminate, never pass. `as_of` is the
        one exception and defaults to today, matching the same parameter in
        `lifecycle.checkpoints` — a replay hands one in and gets the same answer
        on any day it is run.

        **`lane` is forwarded to `applicable` here, and that forward is the
        feature.** This is `applicable`'s only caller and the only entry point
        the orchestrator invokes, so a lane parameter that stopped at
        `applicable`'s signature would be filtering that silently never
        happens — the worst kind, because every trigger would still appear to
        be evaluated correctly.
        """
        inputs = _Inputs(
            metrics=metrics,
            scores=scores,
            eligibility=eligibility,
            checkpoint_results=checkpoint_results,
            lane_gate_result=lane_gate_result,
            catalyst=catalyst,
            state_history=state_history,
            as_of=_as_date(as_of) or date.today(),
        )

        results: dict[str, dict] = {}
        fired: list[str] = []
        indeterminate: list[str] = []

        for trigger_id, spec in self.applicable(state, lane).items():
            detail = self._evaluate_trigger(spec, inputs)
            results[trigger_id] = detail
            if detail["fired"] is True:
                fired.append(trigger_id)
            elif detail["fired"] is None:
                indeterminate.append(trigger_id)

        return {
            "state": state,
            # Recorded because it decided which triggers were even considered;
            # a proposal that cannot be re-derived from its own record is not
            # reviewable later, which is the point of recording any of this.
            "lane": lane,
            "triggers": results,
            "fired": fired,
            "indeterminate": indeterminate,
        }

    def _evaluate_trigger(self, spec: dict, inputs: _Inputs) -> dict:
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
            self._evaluate_condition(condition, inputs)
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

    def _evaluate_condition(self, condition: dict, inputs: _Inputs) -> dict:
        if _METRIC in condition:
            if condition.get("persist_years"):
                return self._evaluate_series(condition, inputs.metrics)
            return self._evaluate_metric(condition, inputs.metrics)
        if _SCORE in condition:
            return self._evaluate_score(condition, inputs.scores)
        if _VERDICT in condition:
            return self._evaluate_verdict(condition, inputs.eligibility)
        if _FLAG_PRESENT in condition or _FLAG_ABSENT in condition:
            return self._evaluate_flag(condition, inputs.metrics)
        if _CHECKPOINT in condition:
            return self._evaluate_checkpoint(condition, inputs.checkpoint_results)
        if _LANE_VERDICT in condition:
            return self._evaluate_lane_verdict(condition, inputs.lane_gate_result)
        if _CATALYST in condition:
            return self._evaluate_catalyst(condition, inputs.catalyst)
        if _SINCE_STATE_ENTRY in condition:
            return self._evaluate_since_state_entry(
                condition, inputs.state_history, inputs.as_of
            )

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
        outcome["detail"] = (
            f"eligibility verdict is {actual}"
            if outcome["passed"]
            else f"eligibility verdict is {actual}, wanted {expected}"
        )
        return outcome

    def _evaluate_lane_verdict(
        self, condition: dict, lane_gate_result: dict | None
    ) -> dict:
        """The fast lane's verdict, read exactly as the 100x verdict above it.

        A trigger wanting "every lane gate passes" could name the gates itself,
        and would then be a second copy of the gate list — one that drifts from
        `lane_gates.yaml` the first time either side changes, with nothing to
        say which one the money followed. Reading `LaneGateEvaluator`'s verdict
        keeps a single statement of what the lane requires.

        The vocabulary is the lane's own (`qualifies` / `not_qualified` /
        `indeterminate`) because `eligible` already answers the hundredfold
        question, and a company is routinely one and not the other. A result
        that was never produced is indeterminate: on this side of the system a
        cleared gate is what lets capital move.
        """
        expected = condition[_LANE_VERDICT]
        outcome = {
            "kind": _LANE_VERDICT,
            "expected": expected,
            "value": None,
            "passed": None,
        }

        if not lane_gate_result or not lane_gate_result.get("verdict"):
            outcome["detail"] = "fast-lane gates were not evaluated"
            return outcome

        actual = lane_gate_result["verdict"]
        outcome["value"] = actual
        outcome["passed"] = actual == expected
        outcome["detail"] = (
            f"lane verdict is {actual}"
            if outcome["passed"]
            else f"lane verdict is {actual}, wanted {expected}"
        )
        return outcome

    def _evaluate_catalyst(self, condition: dict, catalyst: dict | None) -> dict:
        """The owner-recorded catalyst — a non-metric input, like a checkpoint.

        Two empty cases that must not collapse into one:

        * `catalyst={}` — an entry somebody has looked at that carries no
          catalyst. "Not yet identified" is a **known fact** about that entry,
          so the condition is plainly **False**.
        * `catalyst=None` — no watchlist context was supplied at all. Nothing
          was looked at, so the condition is **indeterminate**.

        Both are falsy, so only an explicit `is None` keeps them apart, and a
        truthiness check would quietly turn "nobody has assessed this company"
        into "assessed and found wanting" — the same silent-pass failure this
        whole file exists to prevent, pointed the other way. Mirrors
        `LaneGateEvaluator._evaluate_catalyst`, which reasons identically.
        """
        expected = condition[_CATALYST]
        outcome = {
            "kind": _CATALYST,
            "expected": expected,
            "value": None,
            "passed": None,
        }

        if catalyst is None:
            outcome["detail"] = "no watchlist entry supplied — catalyst status unknown"
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

    def _evaluate_since_state_entry(
        self, condition: dict, state_history: list | None, as_of: date
    ) -> dict:
        """How long a company has sat where it is — the time stop's one input.

        A re-rating thesis that has not re-rated in eighteen months is wrong
        about something, and the only way to know is to read the append-only
        `state_history` on the watchlist entry for the most recent transition
        *into* the state being timed.

        `_evaluate_series`'s discipline applies unchanged: **an unknown elapsed
        time is indeterminate, never assumed zero.** Zero would read as "just
        arrived", which is the single answer that keeps a time stop
        permanently quiet on the position it exists to end.

        Time is read against the `as_of` handed in rather than the wall clock,
        so a replay of an old decision reaches the same conclusion on any day
        it is run — the same reason `lifecycle.checkpoints` takes the parameter.
        """
        target = condition[_SINCE_STATE_ENTRY]
        comparator = condition.get("comparator", "gte")
        threshold = condition.get("threshold")

        outcome = {
            "kind": _SINCE_STATE_ENTRY,
            "state": target,
            "comparator": comparator,
            "threshold": threshold,
            "entered_at": None,
            "value": None,
            "passed": None,
        }

        if state_history is None:
            outcome["detail"] = "no state history supplied — time in state unknown"
            return outcome

        # History is append-only and written in order, so the *last* matching
        # record is the current visit. Re-entering a state restarts the clock:
        # a stint that ended is not what a time stop measures.
        entries = [
            record
            for record in state_history
            if isinstance(record, dict) and record.get("to") == target
        ]
        if not entries:
            outcome["detail"] = f"never reached {target}"
            return outcome

        entered = _as_date(entries[-1].get("at"))
        if entered is None:
            # Deliberately not falling back to an earlier matching record: that
            # would date the stop from a previous visit and could end a
            # position months before its clock actually ran out.
            outcome["detail"] = (
                f"the {target} transition carries an unreadable timestamp "
                f"{entries[-1].get('at')!r}"
            )
            return outcome

        compare = COMPARATORS.get(comparator)
        if compare is None:
            logger.warning(f"Unknown comparator '{comparator}' for since_state_entry")
            outcome["detail"] = f"unknown comparator '{comparator}'"
            return outcome

        days = (as_of - entered).days
        outcome["entered_at"] = entries[-1].get("at")
        outcome["value"] = float(days)
        outcome["passed"] = bool(compare(days, threshold))
        outcome["detail"] = (
            f"{days:,.0f} days in {target} since {entered.isoformat()} "
            f"{'is' if outcome['passed'] else 'is not'} {comparator} "
            f"{_format_threshold(threshold)} days"
        )
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

        # Zero misses out of zero due checkpoints is not a thesis holding up —
        # it is a thesis nobody has checked. Counting that as `clear` would let
        # an unmonitored position read exactly like a verified one, which is
        # the failure this whole layer exists to prevent.
        if not checkpoint_results.get("due"):
            total = checkpoint_results.get("total", 0)
            outcome["detail"] = (
                f"no checkpoints have come due yet ({total} recorded)"
                if total
                else "no checkpoints recorded for this thesis"
            )
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
