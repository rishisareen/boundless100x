"""`raw_data/` discovery and per-ticker candidacy under KTD8.

There is no point-in-time universe and no simulated "owner adds a name"
input (plan §14.6, KTD8), so the candidacy rule is: every `raw_data/`
ticker joins the simulated watchlist at `screen` on the first replay date
whose truncated financials meet the engine's minimum-years bar
(`backtest.MIN_TOTAL_YEARS`). A ticker that never clears that bar across
the whole replay window is an **exclusion, named with its reason** — never
a silent omission — which is the whole point of KTD8: the survivorship
assumption is made visible as a count rather than hidden as one.

`discover_candidates` mirrors `WalkForwardBacktest.discover_candidates`'s
own three-line idiom (a `raw_data/` subdirectory is a real ticker iff it
carries `TICKER_MARKER`) rather than importing it — small enough that
duplicating it is not the "reimplement a rule" KTD1 forbids, and it keeps
this module free of a dependency on `backtest.py`'s instance-bound version.

`load_ticker_data` is `WalkForwardBacktest._load` plus two files backtest
never reads (`quarterly.csv`, `shareholding.csv`) — the frames the fast
lane's gates need and the backtest's published correlations were never
built on. Kept in **obvious lockstep** with `backtest._load`'s file list
and column handling (CLAUDE.md's own words for this exact risk): a drift
here would silently exclude or wrongly include tickers relative to what the
backtest itself would have loaded for the same ticker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from boundless100x.compute_engine.backtest import MIN_TOTAL_YEARS
from boundless100x.compute_engine.point_in_time import (
    ANNUAL_FRAMES,
    ANNUAL_REPORTING_LAG_MONTHS,
    truncate_to_date,
)
from boundless100x.data_fetcher.corpus_snapshot import TICKER_MARKER

logger = logging.getLogger(__name__)

# Mirrors `backtest.REQUIRED_FILES` exactly — a genuine ticker missing either
# file cannot be scored at all, so it is skipped with a reason rather than
# silently dropped from the discovery list.
REQUIRED_FILES = ("financials.csv", "price_volume.csv")

# The two frames the backtest's own `_load` never reads (its `data` never
# carries them, so `NON_TRUNCATABLE_INPUTS`'s strip of `shareholding` is moot
# for that caller — see `point_in_time.py`'s module docstring). The
# simulator's lane gates need both.
_EXTRA_FRAMES = ("quarterly", "shareholding")


# ── discovery ────────────────────────────────────────────────────────────


def discover_candidates(raw_data_dir: str | Path) -> list[Path]:
    """Directories holding a real ticker's fetched data.

    BSE-code directories carry only annual report PDFs and are not tickers
    at all — `TICKER_MARKER` (`financials.csv`) is what tells the two apart.
    """
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        return []
    return sorted(
        d for d in raw_data_dir.iterdir()
        if d.is_dir() and (d / TICKER_MARKER).exists()
    )


def load_ticker_data(ticker_dir: Path) -> dict:
    """One ticker's raw frames, in the shape `point_in_time.truncate_to_date`
    expects as `data`.

    Extends `WalkForwardBacktest._load` with `quarterly` and `shareholding` —
    everything else (the annual frames, the price-date parsing/tz handling,
    the metadata read) is copied verbatim so this loader answers "what does
    the backtest see for this ticker" identically wherever the two overlap.
    """
    data: dict = {}
    for name in ANNUAL_FRAMES:
        path = ticker_dir / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path)

    price = pd.read_csv(ticker_dir / "price_volume.csv")
    price["date"] = pd.to_datetime(
        price["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    data["price"] = price.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for name in _EXTRA_FRAMES:
        path = ticker_dir / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path)

    meta_path = ticker_dir / "metadata.json"
    data["_metadata_raw"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return data


# ── candidacy (KTD8) ────────────────────────────────────────────────────


def first_sufficient_history_date(
    data: dict,
    replay_dates: Sequence[pd.Timestamp],
    *,
    min_total_years: int = MIN_TOTAL_YEARS,
    annual_lag_months: int = ANNUAL_REPORTING_LAG_MONTHS,
) -> pd.Timestamp | None:
    """The first `replay_dates` entry whose truncated financials clear
    `min_total_years` rows — one ticker's KTD8 candidacy date.

    Truncates at every candidate date rather than reading the total history
    length once: a row only counts once its period-end plus the annual
    reporting lag falls on or before the cutoff, which is exactly what
    `truncate_to_date` already enforces for every other consumer. `None`
    when no candidate date clears the bar — the caller is responsible for
    recording that as a named exclusion rather than dropping the ticker
    silently.
    """
    for cutoff in sorted(replay_dates):
        truncated, _reason = truncate_to_date(
            data, cutoff, annual_lag_months=annual_lag_months, rebuild_valuation=False,
        )
        if truncated is None:
            continue
        financials = truncated.get("financials")
        if financials is not None and len(financials) >= min_total_years:
            return cutoff
    return None


@dataclass
class UniverseResult:
    """`build_universe`'s return: who is in, who is out, and why.

    `ticker_dirs` carries every discovered ticker (eligible or not) that
    could at least be read — `replay.py` uses it to re-load a specific
    ticker's data at its own first-eligible date without re-scanning
    `raw_data/` or re-running candidacy for the whole corpus.
    """
    eligible: dict[str, pd.Timestamp] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    ticker_dirs: dict[str, Path] = field(default_factory=dict)


def build_universe(
    raw_data_dir: str | Path,
    replay_dates: Sequence[pd.Timestamp],
    *,
    min_total_years: int = MIN_TOTAL_YEARS,
    annual_lag_months: int = ANNUAL_REPORTING_LAG_MONTHS,
) -> UniverseResult:
    """KTD8 in full: every `raw_data/` ticker, sorted into an eligible
    `{ticker: first_eligible_date}` map and an excluded `{ticker: reason}`
    map — nothing dropped without a reason attached.
    """
    result = UniverseResult()

    for ticker_dir in discover_candidates(raw_data_dir):
        ticker = ticker_dir.name
        missing = [f for f in REQUIRED_FILES if not (ticker_dir / f).exists()]
        if missing:
            result.excluded[ticker] = f"missing {', '.join(missing)}"
            continue

        try:
            data = load_ticker_data(ticker_dir)
        except Exception as exc:  # noqa: BLE001 — a corrupt file is an exclusion, not a crash
            result.excluded[ticker] = f"could not read data: {exc}"
            continue

        result.ticker_dirs[ticker] = ticker_dir

        if not replay_dates:
            result.excluded[ticker] = "no replay dates supplied to evaluate candidacy against"
            continue

        first_date = first_sufficient_history_date(
            data, replay_dates,
            min_total_years=min_total_years, annual_lag_months=annual_lag_months,
        )
        if first_date is None:
            window = f"{min(replay_dates).date()}–{max(replay_dates).date()}"
            result.excluded[ticker] = (
                f"never reaches {min_total_years} years of truncated annual "
                f"financials across the replay window ({window})"
            )
        else:
            result.eligible[ticker] = first_date

    return result
