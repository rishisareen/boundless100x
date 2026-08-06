"""Append-only score history — the raw material for score momentum.

A single SQGLP score is a photograph. Returns come from deltas: improving
fundamentals precede re-rating, so a company moving 5.8 → 6.6 → 7.2 is often a
better forward signal than a static 8.0. Nothing in the pipeline recorded that
sequence — `scores.json` is overwritten per report directory and the watchlist
keeps only the latest composite — so this module writes one row per scored run
and never rewrites one.

History cannot be backfilled from organic runs: a score not written when the
run happened is gone. That is why persistence ships ahead of the Phase 2 diff
computation that consumes it.

**Every row carries the registry hash** that produced it. Without it a weight
change or threshold edit would read as fundamental momentum — the company
would appear to improve because the ruler moved. Readers compare within a
regime, never across one.

The file is append-only by contract. Same-day re-runs append duplicate-dated
rows rather than overwriting; `load_history` resolves them at read time by
keeping the last row for a given (date, config_hash). Writing never rewrites
existing bytes, so a corrupted or half-written line can never destroy history
that was already recorded.
"""

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path(__file__).parent / "score_history.jsonl"

SCHEMA_VERSION = 1


def _row_from(result, config_hash: str, synthetic: bool) -> dict:
    """Build the history row for a completed run.

    `coverage` is the composite coverage share alone, not the full coverage
    dict: a momentum reader needs to know whether a score rested on thin
    evidence, and the per-metric `unscored` list would bloat a git-tracked log
    that grows by a row per run forever. The full breakdown stays in that
    run's `scores.json`.
    """
    scores = result.scores or {}
    eligibility = result.eligibility or {}
    coverage = scores.get("coverage") or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": result.ticker,
        "date": date.today().isoformat(),
        "composite": scores.get("composite"),
        "elements": scores.get("elements", {}),
        # An eligibility evaluation that could not run is unknown, never a
        # pass — the same rule the gates themselves follow.
        "verdict": eligibility.get("verdict", "indeterminate"),
        "coverage": coverage.get("composite"),
        "flags": scores.get("flags", []),
        "config_hash": config_hash,
        # True only for rows synthesised by re-scoring truncated history
        # (v05 §7.1). Organic runs are never mixed with those in a momentum
        # read without the marker being visible.
        "synthetic": synthetic,
    }


def append_run(
    result,
    config_hash: str,
    path: str | Path | None = None,
    synthetic: bool = False,
) -> dict | None:
    """Append one row for a completed run. Returns the row, or None if skipped.

    A run whose scoring failed carries no composite to remember, so it is
    skipped rather than recorded as a null-scored point that a later diff
    would have to special-case.
    """
    if not result.scores:
        logger.warning(
            f"No score history written for {result.ticker}: scoring produced nothing"
        )
        return None

    row = _row_from(result, config_hash, synthetic)
    target = Path(path) if path else DEFAULT_HISTORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")

    logger.info(
        f"Score history: {row['ticker']} {row['date']} "
        f"composite={row['composite']} registry={config_hash}"
    )
    return row


def read_rows(path: str | Path | None = None) -> list[dict]:
    """Every row in the log, in file order, unfiltered and undeduplicated.

    Malformed lines are skipped with a warning rather than raising: a
    truncated final line from an interrupted write must not make the whole
    history unreadable.
    """
    target = Path(path) if path else DEFAULT_HISTORY_PATH
    if not target.exists():
        return []

    rows = []
    for number, line in enumerate(target.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Skipping unparseable score-history line {number}")
    return rows


def load_history(
    ticker: str | None = None, path: str | Path | None = None
) -> list[dict]:
    """Rows for a ticker, oldest first, with same-day re-runs resolved.

    Two runs on the same date under the same registry describe one
    observation, so the later supersedes the earlier — the log keeps both,
    the reader sees one. Rows under different registry hashes are both kept:
    they are different scoring regimes and it is the caller's job (Phase 2)
    to refuse to diff across them.
    """
    rows = read_rows(path)
    if ticker is not None:
        rows = [r for r in rows if r.get("ticker") == ticker]

    resolved: dict[tuple, dict] = {}
    for row in rows:
        resolved[(row.get("ticker"), row.get("date"), row.get("config_hash"))] = row

    return sorted(resolved.values(), key=lambda r: (r.get("date") or "", r.get("ticker") or ""))
