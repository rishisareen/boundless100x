"""Checkpoints: the written promises a thesis is held to.

Pass 2 has always emitted `key_monitorables` — "what to track quarterly" — as
free-text strings that no code has ever read. They are rendered as bullets and
then abandoned, which means the thesis is written down and never checked
again. A checkpoint is the same intent expressed so that code can check it:
a metric from a closed vocabulary, a comparator, a threshold, and a date by
which it should be true.

Two rules do the real work here.

**The vocabulary is closed.** Only series the pipeline can read quarterly are
admissible, so a checkpoint can always actually come due. An id outside the
vocabulary is refused at recording time and demoted to prose, never stored as
something the evaluator will silently fail to find.

**A data gap is not a miss.** A checkpoint that cannot be evaluated — no
quarterly file, too short a series, a column the fetch never produced — is
`indeterminate`, and the kill-switch that counts missed checkpoints does not
count it. Dropping a company because its data was stale would be the
lifecycle's version of a gate passing on missing evidence.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from boundless100x.compute_engine.eligibility import COMPARATORS, _format_threshold

logger = logging.getLogger(__name__)

DEFAULT_VOCABULARY_PATH = Path(__file__).parent / "checkpoint_vocabulary.yaml"

# Periods back for a year-over-year comparison on a quarterly series.
_QUARTERS_PER_YEAR = 4

MET = "met"
MISSED = "missed"
PENDING = "pending"
INDETERMINATE = "indeterminate"


def load_vocabulary(path: str | Path | None = None) -> dict:
    """Read the checkpoint vocabulary. Returns {metric_id: spec}."""
    target = Path(path) if path else DEFAULT_VOCABULARY_PATH
    if not target.exists():
        logger.warning(f"No checkpoint vocabulary at {target}")
        return {}
    loaded = yaml.safe_load(target.read_text()) or {}
    return loaded.get("checkpoints", {}) or {}


def vocabulary_prompt_block(vocabulary: dict | None = None) -> str:
    """The vocabulary rendered for the Pass 2 prompt.

    Pass 2 is told these are the only valid `metric_id` values. Sending the
    list is what makes structured monitorables possible at all — asked for an
    id without a menu, a model invents plausible ones.
    """
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
    lines = [
        f"  {metric_id} — {spec.get('label', metric_id)}"
        for metric_id, spec in sorted(vocabulary.items())
    ]
    return "\n".join(lines)


def validate(checkpoint: dict, vocabulary: dict | None = None) -> list[str]:
    """Return the reasons this checkpoint is not machine-evaluable.

    An empty list means it can be stored as structured; anything else means it
    is demoted to prose-only with the reasons logged.
    """
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
    errors: list[str] = []

    if not isinstance(checkpoint, dict):
        return ["checkpoint must be a mapping"]

    metric_id = checkpoint.get("metric_id")
    if metric_id not in vocabulary:
        errors.append(f"metric_id {metric_id!r} is not in the checkpoint vocabulary")

    if checkpoint.get("comparator") not in COMPARATORS:
        errors.append(f"unknown comparator {checkpoint.get('comparator')!r}")

    threshold = checkpoint.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        errors.append(f"threshold must be numeric, got {threshold!r}")

    due = checkpoint.get("due_date")
    if not isinstance(due, str) or not _parse_date(due):
        errors.append(f"due_date must be an ISO date, got {due!r}")

    return errors


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _series_value(spec: dict, data: dict) -> tuple[float | None, str, str | None]:
    """Resolve a vocabulary entry against fetched data.

    Returns (value, explanation, period_label). A None value always carries an
    explanation of what was missing.
    """
    frame = data.get(spec.get("source"))
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None, f"no {spec.get('source')} data — refetch required", None

    columns = spec.get("columns") or []
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return None, f"column(s) {', '.join(missing)} absent from {spec['source']}", None

    period_col = "quarter" if "quarter" in frame.columns else None
    period = str(frame[period_col].iloc[-1]) if period_col else None
    transform = spec.get("transform", "latest")

    if transform == "sum":
        total = 0.0
        for column in columns:
            value = pd.to_numeric(frame[column], errors="coerce").dropna()
            if value.empty:
                return None, f"{column} has no numeric values", period
            total += float(value.iloc[-1])
        return total, f"{' + '.join(columns)} = {total:,.2f}", period

    column = columns[0]
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None, f"{column} has no numeric values", period

    if transform == "yoy_pct":
        if len(values) <= _QUARTERS_PER_YEAR:
            return (
                None,
                f"{column} has {len(values)} periods, needs "
                f"{_QUARTERS_PER_YEAR + 1} for a year-over-year comparison",
                period,
            )
        latest = float(values.iloc[-1])
        year_ago = float(values.iloc[-1 - _QUARTERS_PER_YEAR])
        if year_ago == 0:
            return None, f"{column} was zero a year earlier — no YoY basis", period
        change = (latest - year_ago) / abs(year_ago) * 100
        return change, f"{column} {latest:,.2f} vs {year_ago:,.2f} a year earlier", period

    return float(values.iloc[-1]), f"{column} = {float(values.iloc[-1]):,.2f}", period


def evaluate(
    checkpoint: dict,
    data: dict,
    as_of: date | None = None,
    vocabulary: dict | None = None,
) -> dict:
    """Evaluate one checkpoint against fetched data.

    `status` is the honest four-way answer: met, missed, pending (not yet
    due), or indeterminate (could not be read). Only `missed` counts against
    a thesis.
    """
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
    as_of = as_of or date.today()

    metric_id = checkpoint.get("metric_id")
    comparator = checkpoint.get("comparator")
    threshold = checkpoint.get("threshold")

    outcome = {
        "metric_id": metric_id,
        "comparator": comparator,
        "threshold": threshold,
        "due_date": checkpoint.get("due_date"),
        "value": None,
        "period": None,
        "status": INDETERMINATE,
        "detail": "",
    }

    errors = validate(checkpoint, vocabulary)
    if errors:
        outcome["detail"] = "; ".join(errors)
        return outcome

    due = _parse_date(checkpoint["due_date"])
    if due > as_of:
        outcome["status"] = PENDING
        outcome["detail"] = f"not due until {due.isoformat()}"
        return outcome

    value, explanation, period = _series_value(vocabulary[metric_id], data)
    outcome["period"] = period
    if value is None:
        outcome["detail"] = explanation
        return outcome

    compare = COMPARATORS[comparator]
    met = bool(compare(value, threshold))
    outcome["value"] = value
    outcome["status"] = MET if met else MISSED
    outcome["detail"] = (
        f"{vocabulary[metric_id].get('label', metric_id)}"
        f"{f' ({period})' if period else ''}: {explanation} — "
        f"{'met' if met else 'missed'} {comparator} {_format_threshold(threshold)}"
    )
    return outcome


def evaluate_all(
    checkpoints: list[dict],
    data: dict,
    as_of: date | None = None,
    vocabulary: dict | None = None,
) -> list[dict]:
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
    return [evaluate(c, data, as_of, vocabulary) for c in (checkpoints or [])]


def summarise(outcomes: list[dict]) -> dict:
    """Counts by status, for the checkpoint conditions in triggers.yaml.

    `missed` deliberately excludes indeterminate outcomes: a thesis is ended
    by evidence that it is not happening, never by an absence of evidence.
    """
    counts = {MET: 0, MISSED: 0, PENDING: 0, INDETERMINATE: 0}
    for outcome in outcomes or []:
        counts[outcome.get("status", INDETERMINATE)] += 1
    counts["total"] = sum(
        counts[k] for k in (MET, MISSED, PENDING, INDETERMINATE)
    )
    counts["due"] = counts[MET] + counts[MISSED]
    return counts
