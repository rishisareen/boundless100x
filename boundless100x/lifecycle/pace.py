"""Deployment pace: deploy more slowly when the whole corpus is expensive.

v05 §11 allows the market's valuation to modulate the *pace* of entry and
nothing else. That boundary is the whole design: a company must never be
blocked from a kill-switch, an exit review, or an eligibility verdict because
the market is dear. Macro slows buying; company evidence decides everything
else.

**Correcting the roadmap.** §11 names `earnings_yield_vs_gsec` as the pace
input, but that metric is per-*company*: `compute_earnings_yield_spread` reads
`data["metadata"]["Stock P/E"]` — that one ticker's multiple — against the
macro G-Sec yield. Wiring it in would tighten entry when the *company* is
expensive, which is the inverse of the modulator's purpose and a second
per-name valuation test on a buy-zone trigger that already tests valuation.

So the pace input is the **median `earnings_yield_vs_gsec` across the cached
corpus**: a breadth reading assembled from per-name metrics that already
exist, needing no new data source (§13 forbids one) and — decisively — no
number anyone has to remember to refresh. An owner-set spread would have
needed an `as_of` date and stale-as-unset handling precisely because it decays
in silence; a computed median cannot go stale.

Two limits are recorded rather than hidden. The "market" here is ~20
survivorship-selected names, so this reads *the corpus's* valuation rather
than the market's. And the median is taken over the cached corpus, not the
watchlist, so adding or dropping a tracked company does not move the signal
underneath a decision.

Mechanism is unchanged from what already exists: `TriggerEvaluator` accepts an
injected trigger dict and `advance()` an injected evaluator, so this derives a
threshold-tightened *copy* of the entry triggers and passes it in. Adding a
spread *condition* to the trigger instead would make macro a gate — a company
blocked from entry by the market's valuation rather than its own, which §11
forbids. A run-level input also keeps the single-evaluator seam valid:
`advance()` builds one evaluator before the ticker loop, which a per-company
reading could not have supplied.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from statistics import median

import pandas as pd

from boundless100x.compute_engine.metrics.builtin.valuation import (
    compute_earnings_yield_spread,
)
from boundless100x.lifecycle import states as lifecycle_states

logger = logging.getLogger(__name__)

# Below this median spread, the corpus is expensive enough to slow entry.
# Owner config (§14.1-.3 family), deliberately outside the hashed `macro:`
# block: it is a policy preference, not an assumption a metric computes with.
# A STARTING POINT awaiting Phase 4 simulator evidence.
DEFAULT_FLOOR_PP = -1.0

# Entry thresholds are multiplied (or divided) by this to become harder to
# satisfy. 0.85 is a tightening, not a freeze — the intent is to slow
# deployment, never to stop it.
DEFAULT_TIGHTEN_FACTOR = 0.85

# A median over three names is not a regime signal. Below this, the reading is
# unknown — and an unknown macro reading must not tighten entry any more than
# it may loosen it.
DEFAULT_MIN_CONTRIBUTORS = 8

# A directory with financials is a real ticker; BSE-code directories hold only
# annual report PDFs. Same marker the backtest uses to discover the corpus.
TICKER_MARKER = "financials.csv"

# Comparators whose threshold gets *harder* to satisfy by moving down.
_LOWER_IS_TIGHTER = ("lt", "lte")
_HIGHER_IS_TIGHTER = ("gt", "gte")


def corpus_spread(raw_data_dir, macro: dict | None = None) -> dict:
    """Median earnings-yield-over-G-Sec across every cached ticker.

    Reads `metadata.json` and `financials.csv` straight from `raw_data/`, which
    is what makes this a property of the corpus rather than of the watchlist:
    adding or dropping a tracked company cannot move it. Each ticker's figure
    is as fresh as its own last fetch — that is what "cached corpus" means, and
    the contributor list travels with the reading so it is inspectable.

    Returns `{median_pp, contributors, tickers, values}`. `median_pp` is None
    when nothing could be read; a caller must treat that as unknown.
    """
    root = Path(raw_data_dir)
    if not root.exists():
        logger.info(f"No cached corpus at {root} — deployment pace reads unknown")
        return {"median_pp": None, "contributors": 0, "tickers": [], "values": []}

    readings: list[tuple[str, float]] = []
    for directory in sorted(d for d in root.iterdir() if d.is_dir()):
        if not (directory / TICKER_MARKER).exists():
            continue
        meta_path = directory / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
            financials = pd.read_csv(directory / TICKER_MARKER)
        except Exception as e:
            logger.warning(f"Pace: could not read {directory.name}: {e}")
            continue

        result = compute_earnings_yield_spread(
            {"metadata": metadata, "financials": financials}, dict(macro or {})
        )
        if result.ok and isinstance(result.value, (int, float)):
            readings.append((directory.name, float(result.value)))

    readings.sort()
    values = [value for _, value in readings]
    return {
        "median_pp": round(median(values), 2) if values else None,
        "contributors": len(values),
        "tickers": [name for name, _ in readings],
        "values": [round(v, 2) for v in values],
    }


def _tighten(threshold, comparator: str, factor: float):
    """A threshold that is harder to satisfy, whichever way the comparator points.

    `lte`/`lt` conditions tighten by moving down; `gte`/`gt` by moving up.
    Doing this by comparator rather than assuming "lower is stricter" means a
    future entry trigger declared the other way round is still tightened rather
    than quietly loosened.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return threshold
    if comparator in _LOWER_IS_TIGHTER:
        return round(threshold * factor, 4)
    if comparator in _HIGHER_IS_TIGHTER and factor:
        return round(threshold / factor, 4)
    return threshold


def modulate(
    triggers: dict,
    reading: dict,
    floor_pp: float = DEFAULT_FLOOR_PP,
    factor: float = DEFAULT_TIGHTEN_FACTOR,
    min_contributors: int = DEFAULT_MIN_CONTRIBUTORS,
) -> tuple[dict, dict]:
    """A trigger set with entry thresholds tightened, plus the decision record.

    Only transitions into `probe` are touched — the one entry transition the
    lifecycle declares, and the only one §11 permits macro to influence. Every
    other trigger, kill-switches most of all, is returned unchanged and
    identical by value.

    Returns `(triggers, decision)`. `decision["applied"]` is False whenever the
    reading is missing, thin, or wide, and carries the reason: an unknown macro
    reading must never tighten entry any more than it may loosen it.
    """
    median_pp = reading.get("median_pp")
    contributors = reading.get("contributors", 0)

    decision = {
        "applied": False,
        "reason": "",
        "median_pp": median_pp,
        "contributors": contributors,
        "floor_pp": floor_pp,
        "factor": factor,
        "adjusted": {},
        "evidence": "",
    }

    if median_pp is None:
        decision["reason"] = (
            "no corpus spread could be read — deployment pace unmodulated"
        )
    elif contributors < min_contributors:
        decision["reason"] = (
            f"only {contributors} contributors to the corpus spread "
            f"(need {min_contributors}) — a median over that few names is not a "
            f"regime signal, so pace is unmodulated"
        )
    elif median_pp >= floor_pp:
        decision["reason"] = (
            f"corpus earnings-yield spread {median_pp:+.2f}pp is at or above the "
            f"{floor_pp:+.2f}pp floor across {contributors} names — "
            f"deployment pace unmodulated"
        )
    else:
        decision["applied"] = True
        decision["reason"] = (
            f"corpus earnings-yield spread {median_pp:+.2f}pp is below the "
            f"{floor_pp:+.2f}pp floor across {contributors} names — "
            f"entry thresholds tightened by x{factor}"
        )

    if not decision["applied"]:
        decision["evidence"] = decision["reason"]
        return triggers, decision

    modulated = deepcopy(triggers)
    for trigger_id, spec in modulated.items():
        if spec.get("to") != lifecycle_states.PROBE:
            continue
        changes = []
        for condition in spec.get("conditions") or []:
            if not isinstance(condition, dict) or "metric" not in condition:
                continue  # flag and checkpoint conditions carry no threshold
            comparator = condition.get("comparator", "lt")
            before = condition.get("threshold")
            after = _tighten(before, comparator, factor)
            if after != before:
                condition["threshold"] = after
                changes.append({
                    "metric": condition["metric"],
                    "comparator": comparator,
                    "from": before,
                    "to": after,
                })
        if changes:
            decision["adjusted"][trigger_id] = changes

    rendered = "; ".join(
        f"{c['metric']} {c['from']}->{c['to']}"
        for changes in decision["adjusted"].values()
        for c in changes
    )
    decision["evidence"] = f"{decision['reason']} ({rendered})"
    logger.info(f"Deployment pace: {decision['evidence']}")
    return modulated, decision


def config_from(config: dict) -> dict:
    """Owner settings for the modulator, with the shipped defaults.

    Only the floor, the tightening factor, and the contributor minimum are
    configurable. The spread itself is computed, so there is no value to keep
    current and no staleness to handle — which is the entire reason the input
    became a corpus median rather than an owner-set number.
    """
    section = (config or {}).get("deployment_pace", {}) or {}
    return {
        "floor_pp": section.get("floor_pp", DEFAULT_FLOOR_PP),
        "factor": section.get("tighten_factor", DEFAULT_TIGHTEN_FACTOR),
        "min_contributors": section.get("min_contributors", DEFAULT_MIN_CONTRIBUTORS),
    }
