"""Which two readings in one report can disagree, and what it means when they do.

R6's second expansion trigger. The declaration lives in
`contradiction_pairs.yaml` beside this file; this module loads it, validates it
at construction, and answers the per-metric question U8's section-level OR is
built from: *does any declared pair naming this metric currently disagree?*

**Curated, not detected** (KTD4). Two of the three examples that motivated the
trigger turned out not to be contradictions, and a sentiment-diff detector
would have manufactured a false positive on every company in the corpus. So
coverage is bounded to pairs somebody wrote down, and the honest consequence is
that a metric in no declared pair reports so plainly rather than being folded
in with the ones that were checked and agreed.

**A pair is two readings, not two metric ids.** R6's wording is "two readings",
and the one surviving genuine instance pairs a metric against a 100x
eligibility gate — which has no metric id at all. So a side names a `kind` from
a closed vocabulary of two: `metric` and `eligibility_gate`. The vocabulary
being closed is what keeps this honest; an open string field would let a typo'd
kind sit in the file forever, never matching, indistinguishable from a
condition that is simply never met. See the YAML's header for the argument in
full.

The fourth declared-registry evaluator's shape, deliberately: same three-valued
outcome as `EligibilityEvaluator`, `TriggerEvaluator`, `LaneGateEvaluator` and
`SectorApplicability`, same per-side `detail` strings, same
indeterminate-never-a-silent-pass rule, and — following `SectorApplicability`
rather than `LaneGateEvaluator` — both vocabularies are **required positional**
arguments, because an optional one whose absence disables the check is a
validation that silently never runs.

This module declares and evaluates a pair. It does **not** decide whether a
section expands: the OR across a section's metrics and the corpus-relative
suppression of the zero-score trigger are U8's, and combining triggers here
would put two of R6's three clauses in a file named after one of them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import yaml

from boundless100x.output.report_reading import NOT_APPLICABLE, Reading

logger = logging.getLogger(__name__)

DEFAULT_PAIRS_PATH = Path(__file__).parent / "contradiction_pairs.yaml"

# ── The closed participant vocabulary ─────────────────────────────────────
#
# Two kinds, each with exactly one condition key. The pairing is one-to-one on
# purpose: a `metric` side is read through its declared interpretation bands
# and an `eligibility_gate` side through its three-valued verdict, and neither
# has a second sensible handle. Adding a kind means adding a condition key,
# a reader for it, and a test — which is the cost that keeps the vocabulary
# from drifting open one convenience at a time.

METRIC = "metric"
ELIGIBILITY_GATE = "eligibility_gate"
PARTICIPANT_KINDS = (METRIC, ELIGIBILITY_GATE)

CONDITION_KEY_BY_KIND = {METRIC: "band_in", ELIGIBILITY_GATE: "verdict_in"}

# A gate's tri-state, in the words a declaration writes. `eligibility.py`
# carries it as `passed: True | False | None`; the mapping is stated once here
# so a YAML author never has to write `null` and mean "we could not tell".
GATE_MET = "met"
GATE_NOT_MET = "not_met"
GATE_INDETERMINATE = "indeterminate"
GATE_STATES = (GATE_MET, GATE_NOT_MET, GATE_INDETERMINATE)
_GATE_STATE_BY_PASSED = {
    True: GATE_MET,
    False: GATE_NOT_MET,
    None: GATE_INDETERMINATE,
}
_GATE_STATE_PHRASES = {
    GATE_MET: "was met",
    GATE_NOT_MET: "was not met",
    GATE_INDETERMINATE: "could not be judged",
}

# ── The verdict vocabulary ────────────────────────────────────────────────
#
# Four values, and `NOT_DECLARED` is the one worth explaining. It reports
# `contradicts: False` — the trigger definitively does not fire — while saying
# *why* it does not: nothing was declared about this metric, rather than
# something was declared and the readings agreed. Both are False for R6's
# purposes and KTD4 accepts that bound explicitly, but they are different
# facts, and collapsing them would hide how thin the declared coverage is
# behind a page full of confident agreement.

CONTRADICTS = "contradicts"
AGREES = "agrees"
INDETERMINATE = "indeterminate"
NOT_DECLARED = "not_declared"
CONTRADICTION_VERDICTS = (CONTRADICTS, AGREES, INDETERMINATE, NOT_DECLARED)

# Keys an entry may carry. An allowlist for the same reason the sector table
# uses one: a typo'd `reasson:` would otherwise ship a pair that fires with
# nothing to say, and R7 makes the sentence the whole deliverable.
_PAIR_KEYS = frozenset({"label", "sides", "reason"})
_SIDE_KEYS = frozenset({"kind", "id", "when"})

NOT_EVALUATED_REASON = (
    "The 100x eligibility gates were not evaluated on this run, so whether "
    "they disagree with this reading is unknown"
)


@lru_cache(maxsize=4)
def load_contradiction_pairs(path: str | None = None) -> dict:
    """Read the declared pairs. Returns the `pairs` mapping, exactly as written.

    Unnormalised, so `validate_contradiction_pairs` can name a malformed entry
    rather than quietly smoothing it into something that loads.

    An unreadable file degrades to `{}`, and here the degradation is safe in
    the direction that matters: with no pairs declared, nothing fires. The
    trigger loses its signal — loudly, in the log — but it never asserts a
    disagreement nobody declared, and it never suppresses one either, since the
    other two triggers in R6 are evaluated independently.
    """
    target = Path(path) if path else DEFAULT_PAIRS_PATH

    try:
        raw = yaml.safe_load(target.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read contradiction pairs at {target}: {exc}")
        return {}

    return raw.get("pairs", {}) or {}


def declared_band_labels(config: Mapping | None) -> set[str]:
    """Every interpretation-band label a metric can actually produce.

    The set a `band_in` condition is checked against. Empty for the nine
    shipped metrics that declare no bands on purpose — and an empty set is what
    makes a `band_in` on one of them a startup error rather than a rule that
    quietly never matches.
    """
    presentation = (config or {}).get("presentation") or {}
    labels = {
        str(band[1])
        for band in (presentation.get("bands") or [])
        if isinstance(band, (list, tuple)) and len(band) == 2
    }
    low_label = presentation.get("low_label")
    if low_label:
        labels.add(str(low_label))
    return labels


def _is_scored(config: Mapping | None) -> bool:
    """Whether a metric carries weight — `ComputeEngine._scored`'s rule.

    Restated rather than imported so this module does not pull the engine into
    the output layer for one comparison. The rule is one line and it is the
    engine's own definition of the weight split that KTD5 turns on.
    """
    weight = ((config or {}).get("scoring") or {}).get("weight", 0) or 0
    return weight > 0


def _validate_side(
    where: str,
    side,
    metric_configs: Mapping,
    gate_specs: Mapping,
) -> list[str]:
    errors: list[str] = []

    if not isinstance(side, Mapping):
        return [f"{where}: a side must be a mapping, got {type(side).__name__}"]

    unknown_keys = sorted(set(side) - _SIDE_KEYS)
    if unknown_keys:
        errors.append(
            f"{where}: unknown key(s) {', '.join(unknown_keys)} — "
            f"expected {', '.join(sorted(_SIDE_KEYS))}"
        )

    kind = side.get("kind")
    if kind not in PARTICIPANT_KINDS:
        # Stop here. Everything below is kind-specific, and reporting an
        # unknown band label against a kind nobody recognises would bury the
        # error that actually matters.
        errors.append(
            f"{where}: unknown participant kind {kind!r} — expected one of "
            f"{', '.join(PARTICIPANT_KINDS)}"
        )
        return errors

    participant_id = side.get("id")
    if not isinstance(participant_id, str) or not participant_id.strip():
        errors.append(f"{where}: id must be a non-empty string, got {participant_id!r}")
        return errors

    condition_key = CONDITION_KEY_BY_KIND[kind]
    when = side.get("when")
    if not isinstance(when, Mapping):
        errors.append(
            f"{where}: `when` must be a mapping carrying {condition_key}, got "
            f"{type(when).__name__}"
        )
        when = {}
    else:
        unexpected = sorted(set(when) - {condition_key})
        if unexpected:
            errors.append(
                f"{where}.when: unknown condition key(s) "
                f"{', '.join(unexpected)} — a side of kind {kind!r} is read "
                f"through {condition_key} and nothing else"
            )

    expected = when.get(condition_key)
    if not isinstance(expected, (list, tuple)) or not expected:
        errors.append(
            f"{where}.when.{condition_key}: must be a non-empty list of states "
            f"this side counts as disagreeing, got {expected!r}"
        )
        expected = []

    if kind == METRIC:
        config = metric_configs.get(participant_id)
        if config is None:
            errors.append(f"{where}: unknown metric id {participant_id!r}")
            return errors
        if not _is_scored(config):
            # KTD5. Expansion is prominence, and a zero-weight signal must not
            # be able to buy it — that is the coupling the forward-signals
            # design exists to keep separate.
            errors.append(
                f"{where}: {participant_id!r} carries zero weight — forward "
                f"signals are ineligible for the contradiction pool (KTD5), "
                f"because expansion is prominence and a signal that cannot "
                f"move a score must not move the report's shape either"
            )
            return errors
        declared = declared_band_labels(config)
        if not declared:
            errors.append(
                f"{where}: {participant_id!r} declares no interpretation "
                f"bands, so it never produces a band a condition could match"
            )
            return errors
        for label in expected:
            if label not in declared:
                errors.append(
                    f"{where}.when.band_in: {participant_id!r} declares no band "
                    f"{label!r} — its bands are "
                    f"{', '.join(repr(b) for b in sorted(declared))}"
                )
    else:
        if participant_id not in gate_specs:
            errors.append(
                f"{where}: unknown eligibility gate {participant_id!r} — "
                f"declared gates are "
                f"{', '.join(repr(g) for g in sorted(gate_specs))}"
            )
        for state in expected:
            if state not in GATE_STATES:
                errors.append(
                    f"{where}.when.verdict_in: {state!r} is not a gate verdict "
                    f"— expected one of {', '.join(GATE_STATES)}"
                )

    return errors


def validate_contradiction_pairs(
    table: dict, metric_configs: Mapping, gate_specs: Mapping
) -> list[str]:
    """Return a list of declaration errors — empty when the file is sound.

    Startup validation, because every failure here is invisible at runtime. A
    pair naming a metric the registry does not define, or a band label the
    metric does not declare, would sit in the file forever without ever
    matching — and a rule that can never fire looks exactly like a rule whose
    condition is never met, which is the same argument the sector table and the
    trigger registry both make. A blank reason is caught here too: R7 makes the
    reconciling sentence the deliverable, so a pair that fires with nothing to
    say expands a section and then shrugs at the reader.
    """
    errors: list[str] = []

    if not isinstance(table, Mapping):
        return [f"pairs must be a mapping, got {type(table).__name__}"]

    for pair_id, entry in table.items():
        if not isinstance(pair_id, str) or not pair_id.strip():
            errors.append(f"pair key {pair_id!r} must be a non-empty string")
            continue
        if not isinstance(entry, Mapping):
            errors.append(f"{pair_id}: entry must be a mapping")
            continue

        unknown_keys = sorted(set(entry) - _PAIR_KEYS)
        if unknown_keys:
            errors.append(
                f"{pair_id}: unknown key(s) {', '.join(unknown_keys)} — "
                f"expected {', '.join(sorted(_PAIR_KEYS))}"
            )

        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(
                f"{pair_id}: needs a label naming the disagreement, not {label!r}"
            )

        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(
                f"{pair_id}: needs a reason that reconciles the two readings "
                f"for a reader, not {reason!r}"
            )

        sides = entry.get("sides")
        if not isinstance(sides, (list, tuple)) or len(sides) != 2:
            errors.append(
                f"{pair_id}: a pair is exactly two sides, got "
                f"{len(sides) if isinstance(sides, (list, tuple)) else type(sides).__name__}"
            )
            continue

        for index, side in enumerate(sides):
            errors.extend(
                _validate_side(
                    f"{pair_id}.sides[{index}]", side, metric_configs, gate_specs
                )
            )

        identities = [
            (s.get("kind"), s.get("id")) for s in sides if isinstance(s, Mapping)
        ]
        if len(identities) == 2 and identities[0] == identities[1]:
            errors.append(
                f"{pair_id}: both sides name the same reading "
                f"({identities[0][1]!r}), which cannot disagree with itself"
            )

        if not any(
            isinstance(s, Mapping) and s.get("kind") == METRIC for s in sides
        ):
            # R6's trigger is per metric. A pair of two non-metric readings has
            # no section to attach to, so it would be dead on arrival rather
            # than merely rare.
            errors.append(
                f"{pair_id}: at least one side must be a metric — R6 fires this "
                f"trigger on a metric, so a pair naming none can never reach a "
                f"section"
            )

    return errors


class ContradictionPairs:
    """Answers "do two declared readings currently disagree about this company?".

    Both vocabularies are **required and positional**. `metric_configs` is the
    engine's own `engine.metrics` mapping — ids, weights and `presentation:`
    blocks all come from it, so the weight rule KTD5 turns on and the band
    labels a condition is checked against are read from the same registry the
    report renders. `gate_specs` must be resolved through
    `eligibility.effective_gates(engine.gates)`, so the gate regime a pair is
    validated against is the regime that will actually be enforced — reading
    the raw registry section instead would validate against an empty mapping on
    any install running the shipped defaults.

    Its sibling `LaneGateEvaluator` takes its metric ids optionally and carries
    a warning in CLAUDE.md for it: construct it without and the unknown-metric
    check silently never runs. `SectorApplicability` fixed that by making the
    argument required, and this follows it.
    """

    def __init__(
        self,
        metric_configs: Mapping,
        gate_specs: Mapping,
        table: dict | None = None,
    ):
        self.metric_configs = dict(metric_configs or {})
        self.gate_specs = dict(gate_specs or {})
        self.pairs = load_contradiction_pairs() if table is None else table

        errors = validate_contradiction_pairs(
            self.pairs, self.metric_configs, self.gate_specs
        )
        if errors:
            for error in errors:
                logger.error(f"  CONTRADICTION PAIR ERROR: {error}")
            # The errors travel in the message as well as the log, matching
            # `SectorApplicability`: this is constructed from caller-supplied
            # tables in tests, where a bare count leaves nothing to act on.
            raise ValueError(
                f"Contradiction pair validation failed: {len(errors)} errors — "
                f"{'; '.join(errors)}"
            )

    # ── Names, never ids (R15) ────────────────────────────────────────────

    def _metric_name(self, metric_id: str) -> str:
        return str((self.metric_configs.get(metric_id) or {}).get("name") or metric_id)

    def _gate_name(self, gate_id: str) -> str:
        return str((self.gate_specs.get(gate_id) or {}).get("label") or gate_id)

    # ── Lookup ────────────────────────────────────────────────────────────

    def pairs_for(self, metric_id: str) -> list[str]:
        """Every declared pair with a `metric` side naming this metric."""
        return sorted(
            pair_id
            for pair_id, entry in self.pairs.items()
            if any(
                side.get("kind") == METRIC and side.get("id") == metric_id
                for side in entry.get("sides", [])
            )
        )

    # ── Reading one side ──────────────────────────────────────────────────

    def _read_metric_side(
        self, side: Mapping, readings: Mapping[str, Reading]
    ) -> dict:
        metric_id = side["id"]
        name = self._metric_name(metric_id)
        expected = list(side.get("when", {}).get("band_in", []))
        outcome = {
            "kind": METRIC,
            "id": metric_id,
            "name": name,
            "expected": expected,
            "state": None,
            "matched": None,
            "detail": "",
            # The upstream sentence, kept for a caller that wants it and
            # separated from `detail` because it can carry a raw exception
            # string on the error path — which R15 forbids putting in front of
            # a reader without laundering it first.
            "source_reason": "",
        }

        reading = readings.get(metric_id) if isinstance(readings, Mapping) else None
        if reading is None:
            outcome["detail"] = f"{name} was not read for this company"
            return outcome

        if getattr(reading, "status", None) == NOT_APPLICABLE:
            # The sector trigger already owns this company/metric pair and the
            # reading layer deliberately withheld the band, so there is no
            # reading here to disagree with anything — and firing a second
            # trigger off a number the report has just said means nothing here
            # would be the misreading twice over.
            outcome["detail"] = (
                f"{name} does not apply to a company of this kind, so it has no "
                f"reading to compare"
            )
            outcome["source_reason"] = getattr(reading, "reason", "")
            return outcome

        if not getattr(reading, "known", False):
            outcome["detail"] = f"{name} could not be read for this company"
            outcome["source_reason"] = getattr(reading, "reason", "")
            return outcome

        band = getattr(reading, "band", "")
        outcome["state"] = band
        outcome["matched"] = band in expected
        quantity = getattr(reading, "quantity", None)
        figure = f"{quantity.text} — " if quantity is not None else ""
        outcome["detail"] = f"{name}: {figure}{band}"
        return outcome

    def _read_gate_side(self, side: Mapping, eligibility) -> dict:
        gate_id = side["id"]
        name = self._gate_name(gate_id)
        expected = list(side.get("when", {}).get("verdict_in", []))
        outcome = {
            "kind": ELIGIBILITY_GATE,
            "id": gate_id,
            "name": name,
            "expected": expected,
            "state": None,
            "matched": None,
            "detail": "",
            "source_reason": "",
        }

        gates = (eligibility or {}).get("gates") or {}
        gate = gates.get(gate_id)
        if not isinstance(gate, Mapping):
            outcome["detail"] = f"{name} was not evaluated on this run"
            outcome["source_reason"] = NOT_EVALUATED_REASON
            return outcome

        state = _GATE_STATE_BY_PASSED.get(gate.get("passed"), GATE_INDETERMINATE)
        outcome["state"] = state
        outcome["matched"] = state in expected
        outcome["detail"] = f"{name} {_GATE_STATE_PHRASES[state]}"
        # Carries raw metric ids from `eligibility.py`'s condition summaries;
        # see `source_reason` above.
        outcome["source_reason"] = str(gate.get("reason") or "")
        return outcome

    def _read_side(
        self, side: Mapping, readings: Mapping[str, Reading], eligibility
    ) -> dict:
        if side.get("kind") == METRIC:
            return self._read_metric_side(side, readings)
        return self._read_gate_side(side, eligibility)

    # ── Evaluating ────────────────────────────────────────────────────────

    def evaluate_pair(
        self, pair_id: str, readings: Mapping[str, Reading], eligibility=None
    ) -> dict:
        """One declared pair, against one company's readings.

        `readings` is the `{metric_id: Reading}` mapping `read_metrics`
        produces; `eligibility` is `result.eligibility`, the dict
        `EligibilityEvaluator.evaluate` returns. `eligibility` of `None` means
        the gates were not evaluated, which reads indeterminate rather than
        agreeing — a run that never asked the question has not been told the
        answer is no.
        """
        entry = self.pairs[pair_id]
        sides = [
            self._read_side(side, readings, eligibility)
            for side in entry.get("sides", [])
        ]

        outcome = {
            "pair": pair_id,
            "label": str(entry.get("label") or pair_id),
            "contradicts": None,
            "verdict": INDETERMINATE,
            "reason": "",
            "sides": sides,
        }

        unread = [side for side in sides if side["state"] is None]
        if unread:
            outcome["reason"] = (
                "These two readings could not be compared: "
                + "; ".join(side["detail"] for side in unread)
            )
            return outcome

        if all(side["matched"] for side in sides):
            outcome["contradicts"] = True
            outcome["verdict"] = CONTRADICTS
            # The declared sentence, verbatim (R7). The per-side details sit
            # beside it so a surface can show what the two readings actually
            # said without the declaration having to guess at them.
            outcome["reason"] = str(entry.get("reason") or "")
            return outcome

        outcome["contradicts"] = False
        outcome["verdict"] = AGREES
        outcome["reason"] = (
            "These two readings do not disagree: "
            + "; ".join(side["detail"] for side in sides)
        )
        return outcome

    def evaluate(
        self, metric_id: str, readings: Mapping[str, Reading], eligibility=None
    ) -> dict:
        """R6's contradiction trigger for one metric. U8's entry point.

        Returns the three-valued answer plus `reasons`, which is always
        populated and always matches the verdict: the declared sentences when
        something fired (R7 renders these), the explanations of what could not
        be compared when indeterminate, and the agreement statements otherwise.
        U8 ORs `contradicts` across a section's metrics and renders `reasons`.

        A metric no declared pair names reports `not_declared` with
        `contradicts: False`. That False is honest rather than a silent pass —
        nothing was declared, so nothing could disagree, and KTD4 accepts that
        bound out loud. It is kept distinct from `agrees` because "checked, and
        they agree" and "nobody wrote a pair for this" are different facts
        about how much this trigger has actually covered.
        """
        pair_ids = self.pairs_for(metric_id)
        if not pair_ids:
            return {
                "metric": metric_id,
                "verdict": NOT_DECLARED,
                "contradicts": False,
                "reasons": [
                    f"No declared contradiction pair names "
                    f"{self._metric_name(metric_id)}, so there is nothing here "
                    f"for its reading to disagree with"
                ],
                "pairs": [],
            }

        outcomes = [
            self.evaluate_pair(pair_id, readings, eligibility)
            for pair_id in pair_ids
        ]

        fired = [o for o in outcomes if o["contradicts"] is True]
        if fired:
            return {
                "metric": metric_id,
                "verdict": CONTRADICTS,
                "contradicts": True,
                "reasons": [o["reason"] for o in fired],
                "pairs": outcomes,
            }

        unknown = [o for o in outcomes if o["contradicts"] is None]
        if unknown:
            return {
                "metric": metric_id,
                "verdict": INDETERMINATE,
                "contradicts": None,
                "reasons": [o["reason"] for o in unknown],
                "pairs": outcomes,
            }

        return {
            "metric": metric_id,
            "verdict": AGREES,
            "contradicts": False,
            "reasons": [o["reason"] for o in outcomes],
            "pairs": outcomes,
        }


__all__ = [
    "AGREES",
    "CONTRADICTION_VERDICTS",
    "CONTRADICTS",
    "ContradictionPairs",
    "DEFAULT_PAIRS_PATH",
    "ELIGIBILITY_GATE",
    "GATE_INDETERMINATE",
    "GATE_MET",
    "GATE_NOT_MET",
    "GATE_STATES",
    "INDETERMINATE",
    "METRIC",
    "NOT_DECLARED",
    "PARTICIPANT_KINDS",
    "declared_band_labels",
    "load_contradiction_pairs",
    "validate_contradiction_pairs",
]
