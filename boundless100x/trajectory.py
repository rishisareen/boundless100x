"""Score momentum: what the append-only history says about direction.

A single SQGLP composite is a photograph. `score_history` has been recording
one row per scored run since Phase 0, and nothing read it — so a company whose
fundamentals are improving right now scores identically to one that has been
flat at the same level for a decade. This module turns those rows into deltas.

Three rules do the work, and each exists because its absence would produce a
number that reads confident and means nothing.

**A diff never crosses a regime boundary.** Every row carries the registry hash
that produced it. Two rows under different hashes were scored by different
rulers, so their difference measures the ruler, not the company. Rows are
partitioned by hash and each partition is walked independently.

**A backfilled row is never diffed against an organic one.** `synthetic: true`
marks a score produced by re-scoring truncated history rather than by a run
that actually happened. Mixing the two in one figure would let a reconstruction
supply the baseline for a real observation, so synthetic rows form their own
partition and never supply the headline reading.

**Every figure states the span it covers.** The interval comes from the actual
row dates. Without it a 400-day gap and a 90-day gap render identically, and a
reader comparing two companies would be comparing an annual drift against a
quarterly one.

And the outcome that matters most on the day this lands: **insufficient history
is a distinct answer, never a zero.** With fewer than two organic rows in a
regime there is no delta to report — which is the common case at first, not an
edge case. A zero delta means flat; no delta means unknown; they look identical
in a table and mean opposite things.

This is deliberately not part of `score_history`. That module's contract is
append and read; interpretation belongs to a consumer, or an append-only store
starts carrying opinions about what its rows mean.
"""

import logging
from datetime import date

from boundless100x import score_history

logger = logging.getLogger(__name__)

OK = "ok"
INSUFFICIENT_HISTORY = "insufficient_history"

# Boundaries for the human span label only. Nothing is computed from these —
# `interval_days` is always the exact figure and travels beside the label.
_SHORT_GAP_DAYS = 60
_YEAR_SCALE_DAYS = 330
_DAYS_PER_MONTH = 30.44


def _span_label(days: int) -> str:
    """How far apart two observations were, said plainly.

    A label, never a rounding: `interval_days` travels beside it so a reader
    who needs the exact gap has it. The label exists so a table cannot present
    a year of drift with the same weight as a quarter of momentum — which
    means the approximation has to be honest at every scale, not just at the
    ends. Months rather than quarters, because a "quarter" bucket wide enough
    to cover 90 to 200 days calls half a year one quarter.
    """
    if days < _SHORT_GAP_DAYS:
        return f"{days} days"
    if days < _YEAR_SCALE_DAYS:
        return f"{days} days (~{round(days / _DAYS_PER_MONTH)} months)"
    return f"{days} days (~{days / 365.25:.1f} years)"


def _parse(value) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _numeric(value) -> float | None:
    """A score, or None. Booleans are not scores."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _element_deltas(earlier: dict, later: dict) -> dict:
    """Per-element change, omitting any element either side cannot supply.

    An element absent from one row, or scored None because no metric in it
    computed, is unknown rather than zero. Treating the absence as zero would
    report an element as perfectly stable precisely when nothing is known about
    it — the same failure the eligibility gates and lifecycle triggers refuse.
    """
    before = earlier if isinstance(earlier, dict) else {}
    after = later if isinstance(later, dict) else {}

    deltas = {}
    for element, raw_after in after.items():
        value_after = _numeric(raw_after)
        value_before = _numeric(before.get(element))
        if value_after is None or value_before is None:
            continue
        deltas[element] = round(value_after - value_before, 3)
    return deltas


def _usable(rows: list[dict]) -> list[tuple[date, dict]]:
    """Rows that carry both a readable date and a composite, oldest first.

    A row missing either cannot anchor a diff. It is dropped rather than
    interpolated: the step then spans the two rows that *do* carry values, and
    says so through its own dates and interval.
    """
    dated = []
    for row in rows:
        when = _parse(row.get("date"))
        if when is None:
            logger.warning(
                f"Skipping score-history row with unreadable date {row.get('date')!r}"
            )
            continue
        if _numeric(row.get("composite")) is None:
            continue
        dated.append((when, row))
    return sorted(dated, key=lambda pair: pair[0])


def _steps(partition: list[tuple[date, dict]], config_hash, synthetic: bool) -> list[dict]:
    """Consecutive diffs within one regime."""
    steps = []
    for (from_date, earlier), (to_date, later) in zip(partition, partition[1:]):
        composite_from = _numeric(earlier.get("composite"))
        composite_to = _numeric(later.get("composite"))
        steps.append({
            "config_hash": config_hash,
            "synthetic": synthetic,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "interval_days": (to_date - from_date).days,
            "span": _span_label((to_date - from_date).days),
            "composite_from": composite_from,
            "composite_to": composite_to,
            "composite_delta": round(composite_to - composite_from, 3),
            "element_deltas": _element_deltas(
                earlier.get("elements"), later.get("elements")
            ),
        })
    return steps


def compute_momentum(
    ticker: str,
    path=None,
    rows: list[dict] | None = None,
) -> dict:
    """Per-regime momentum for one ticker.

    `rows` is for callers that already hold history (tests, a future
    simulator); otherwise rows are read through `score_history.load_history`,
    which resolves same-day re-runs before anything is diffed.

    Returns `{ticker, status, reason, latest, regimes}`. `latest` is the
    freshest *organic* step and is what a report headline should show; it is
    None whenever `status` is `insufficient_history`.
    """
    if rows is None:
        rows = score_history.load_history(ticker, path=path)

    partitions: dict[tuple, list[tuple[date, dict]]] = {}
    for when, row in _usable(rows):
        key = (row.get("config_hash"), bool(row.get("synthetic")))
        partitions.setdefault(key, []).append((when, row))

    regimes = []
    for (config_hash, synthetic), partition in partitions.items():
        steps = _steps(partition, config_hash, synthetic)
        regimes.append({
            "config_hash": config_hash,
            "synthetic": synthetic,
            "rows": len(partition),
            "status": OK if steps else INSUFFICIENT_HISTORY,
            "reason": (
                ""
                if steps
                else f"only {len(partition)} scored run(s) under this regime — "
                     f"two are needed to measure a change"
            ),
            "steps": steps,
            "from_date": partition[0][0].isoformat(),
            "to_date": partition[-1][0].isoformat(),
        })
    regimes.sort(key=lambda r: (r["synthetic"], r["to_date"]))

    # The headline is organic only. A backfilled pair may well be the most
    # recent thing in the log, but it describes a reconstruction rather than
    # something that was observed.
    organic_steps = [
        step
        for regime in regimes
        if not regime["synthetic"]
        for step in regime["steps"]
    ]
    latest = max(organic_steps, key=lambda s: s["to_date"]) if organic_steps else None

    return {
        "ticker": ticker,
        "status": OK if latest else INSUFFICIENT_HISTORY,
        "reason": "" if latest else _no_reading_reason(regimes),
        "latest": latest,
        "regimes": regimes,
    }


def _no_reading_reason(regimes: list[dict]) -> str:
    """Why there is no momentum figure — never left to the reader to guess."""
    if not regimes:
        return (
            "no scored runs recorded yet — momentum accumulates from here "
            "and cannot be backfilled"
        )
    if all(regime["synthetic"] for regime in regimes):
        return (
            "only synthetic (backfilled) rows are recorded; a reconstruction "
            "is not an observation, so it never supplies a momentum reading"
        )
    organic = [r for r in regimes if not r["synthetic"]]
    return (
        f"not enough history yet: {sum(r['rows'] for r in organic)} scored run(s) "
        f"across {len(organic)} scoring regime(s), and two runs under one regime "
        f"are needed to measure a change"
    )
