"""Replay dates from the corpus's own fiscal calendar (KTD7), plus the
per-lane battery-complete reading U6's comparison needs.

Three questions, answered once each and returned together in
`ReplayCalendar` so `replay.py` (and later U7) never has to re-derive them:

1. **Which dates does the replay evaluate on?** Quarterly, on the corpus's
   own fiscal quarter grid (the dominant annual period-end month across the
   universe, almost certainly March — derived, not hardcoded, so a
   December-year-end ticker mixed into the corpus does not silently shift
   the grid) — each quarter-end pushed forward by a reporting lag, so every
   returned date is one a real point-in-time query could legitimately pass
   to `truncate_to_date` as `cutoff` (KTD7's own phrasing). The window runs
   from `REPLAY_START` (2023-01-01, owner decision — see below) to the last
   date the price corpus supports.

2. **When is each lane's gate battery structurally complete?** Not "when
   does scoring first succeed" (that depends on reconciliation, sector,
   coverage — a per-ticker, per-date outcome) but "when does the corpus
   first contain enough periods, by construction, for the deepest gate in
   that lane to be computable at all." KTD7's own measurement is that this
   date is later for `rerating` than for `core`, and by how much is exactly
   what §10's per-lane comparison needs to know before it can be read
   honestly — see `_battery_complete_core` / `_battery_complete_rerating`.

**The start is 2023-01-01 by owner decision, not the earliest scorable
date** (~2020-09 per U1's own measurement). Plan lines 889–912: the five
quarters the earlier start would buy are ones in which the fast lane is
*structurally* unable to qualify (`growth_intact` and
`institutional_accumulation` have no data before their frames' 2023
starts), so a window whose first fifth admits only core-lane entries would
make the per-lane comparison an artifact of the corpus rather than a
finding about the rules. Kept as a named, documented constant (not buried
in a function default) so a future config override can move it without
hunting for the number.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from boundless100x.compute_engine.backtest import MIN_TOTAL_YEARS
from boundless100x.compute_engine.metrics.builtin._helpers import period_end_date, quarter_index
from boundless100x.compute_engine.point_in_time import (
    ANNUAL_REPORTING_LAG_MONTHS,
    NON_TRUNCATABLE_INPUTS,
    QUARTERLY_REPORTING_LAG_MONTHS,
    SHAREHOLDING_REPORTING_LAG_MONTHS,
    truncate_to_date,
)
from boundless100x.simulator.universe import (
    discover_candidates,
    first_sufficient_history_date,
    load_ticker_data,
)

logger = logging.getLogger(__name__)

CORE_LANE = "core"
RERATING_LANE = "rerating"

# Owner decision, 2026-08-08 (see module docstring). A config override can
# move this once Phase 5 wants to sweep it — the name is what a future
# caller reaches for, not a bare literal buried in a function signature.
REPLAY_START = pd.Timestamp("2023-01-01")

# `growth_intact`'s `ttm_growth_vs_cagr` needs a trailing-twelve-month window
# *and* the twelve months before it, both fully contiguous by period label —
# `compute_engine/metrics/builtin/growth.py::_TTM_WINDOW = _TTM_QUARTERS * 2`.
# Read off that implementation rather than assumed: a plausible-sounding "a
# four-quarter TTM needs four quarters" undercounts it by half, because the
# gate compares this TTM against the *prior* TTM, not against the CAGR
# metric's own (differently-shaped) annual series.
GROWTH_INTACT_QUARTERS_NEEDED = 8

# `institutional_accumulation_streak`'s gate condition is `>= 2` rises, and a
# rise is a comparison between two *adjacent* quarters
# (`compute_engine/metrics/builtin/size.py::compute_institutional_accumulation_trend`);
# two rises is a walk across three consecutive points.
INSTITUTIONAL_ACCUMULATION_QUARTERS_NEEDED = 3


# ── fiscal calendar ─────────────────────────────────────────────────────


def _dominant_period_end_month(annual_frames: Sequence[pd.DataFrame | None]) -> int:
    """The most common period-end month across every ticker's annual `year`
    labels — the corpus's fiscal quarter grid.

    Falls back to March only when not a single label across the whole
    corpus was parseable, which is not a real corpus this module has ever
    seen (every cached ticker's `financials.csv` carries readable `Mar
    20XX`-style labels) — the fallback exists so an empty or corrupted
    corpus fails at "no replay dates" rather than at a `Counter` on an
    empty list, not because March is assumed to be correct.
    """
    months: list[int] = []
    for frame in annual_frames:
        if frame is None or "year" not in getattr(frame, "columns", []):
            continue
        for label in frame["year"]:
            end = period_end_date(label)
            if end is not None:
                months.append(end.month)
    if not months:
        return 3
    return Counter(months).most_common(1)[0][0]


def _quarter_end_months(dominant_month: int) -> list[int]:
    """The four calendar months a quarter ends in, given the corpus's fiscal
    year-end month — `dominant_month` itself and the three months three
    apart from it, wrapped mod 12."""
    return sorted(((dominant_month - 1 - 3 * i) % 12) + 1 for i in range(4))


def _quarter_end_dates(
    quarter_end_months: Sequence[int], from_year: int, to_year: int
) -> list[pd.Timestamp]:
    return sorted(
        pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        for year in range(from_year, to_year + 1)
        for month in quarter_end_months
    )


def _replay_dates(
    dominant_month: int, start: pd.Timestamp, end: pd.Timestamp, lag_months: int
) -> list[pd.Timestamp]:
    """Quarter-end dates on the corpus's fiscal grid, each pushed forward by
    `lag_months`, restricted to `[start, end]`.

    `lag_months` is the caller's `max(quarterly_lag_months,
    shareholding_lag_months)` — the longer of the two quarterly-grain lags,
    so that by the time a replay date arrives, both frames' most recent
    quarter is already legitimately public, not just one of them.
    """
    quarter_end_months = _quarter_end_months(dominant_month)
    candidates = _quarter_end_dates(quarter_end_months, start.year - 1, end.year + 1)
    lagged = (qe + pd.DateOffset(months=lag_months) for qe in candidates)
    return sorted({d for d in lagged if start <= d <= end})


# ── battery-complete readings ───────────────────────────────────────────


def _quarter_period_indices(frame: pd.DataFrame | None, period_column: str = "quarter") -> list[int]:
    if frame is None or period_column not in getattr(frame, "columns", []):
        return []
    indices = (quarter_index(label) for label in frame[period_column])
    return sorted(i for i in indices if i is not None)


def _longest_trailing_contiguous_run(indices: Sequence[int]) -> int:
    """How many of the largest period indices are mutually adjacent, walking
    backward from the latest.

    Examines frame *shape* only (distinct period indices, one apart) — it
    answers "does this frame have enough contiguous periods for the gate to
    even attempt a reading," not the gate's actual value, so this is not a
    second statement of either `ttm_growth_vs_cagr`'s or
    `institutional_accumulation_streak`'s own logic (KTD1's concern is
    reimplementing an evaluator's *decision*, not counting rows).
    """
    ordered = sorted(set(indices))
    if not ordered:
        return 0
    run = 1
    for i in range(len(ordered) - 1, 0, -1):
        if ordered[i] - ordered[i - 1] == 1:
            run += 1
        else:
            break
    return run


def _first_lane_gate_depth_date(
    data: dict,
    replay_dates: Sequence[pd.Timestamp],
    *,
    quarterly_lag_months: int,
    shareholding_lag_months: int,
) -> pd.Timestamp | None:
    """The first `replay_dates` entry at which one ticker's truncated
    `quarterly` and `shareholding` frames are both structurally deep enough
    for the fast lane's two quarterly-grain gates.

    `shareholding` is dropped from the strip list for this call only — the
    same opt-out `point_in_time.py`'s module docstring describes for "a
    later simulator caller" — since `NON_TRUNCATABLE_INPUTS`'s default
    would otherwise remove the very frame this function inspects.
    """
    non_truncatable = tuple(x for x in NON_TRUNCATABLE_INPUTS if x != "shareholding")
    for cutoff in sorted(replay_dates):
        truncated, _reason = truncate_to_date(
            data, cutoff,
            quarterly_lag_months=quarterly_lag_months,
            shareholding_lag_months=shareholding_lag_months,
            rebuild_valuation=False,
            non_truncatable_inputs=non_truncatable,
        )
        if truncated is None:
            continue
        quarterly_run = _longest_trailing_contiguous_run(
            _quarter_period_indices(truncated.get("quarterly"))
        )
        shareholding_run = _longest_trailing_contiguous_run(
            _quarter_period_indices(truncated.get("shareholding"))
        )
        if (
            quarterly_run >= GROWTH_INTACT_QUARTERS_NEEDED
            and shareholding_run >= INSTITUTIONAL_ACCUMULATION_QUARTERS_NEEDED
        ):
            return cutoff
    return None


def _battery_complete_core(
    loaded: dict[str, dict],
    replay_dates: Sequence[pd.Timestamp],
    *,
    min_total_years: int,
    annual_lag_months: int,
) -> tuple[pd.Timestamp | None, dict]:
    """The 100x eligibility gates (`size`, `price`, `reinvestment`) need only
    annual financials + price — the same structural requirement KTD8's
    candidacy date already answers per ticker. "Battery complete" for `core`
    is the earliest of those per-ticker dates: the first point at which the
    corpus, taken as a whole, can structurally support the gates for at
    least one company — not the point at which every company can.
    """
    per_ticker = {
        ticker: date
        for ticker, date in (
            (
                ticker,
                first_sufficient_history_date(
                    data, replay_dates,
                    min_total_years=min_total_years, annual_lag_months=annual_lag_months,
                ),
            )
            for ticker, data in loaded.items()
        )
        if date is not None
    }
    if not per_ticker:
        return None, {
            "reason": (
                f"no ticker in the corpus reaches {min_total_years} years of "
                "truncated annual financials within the replay window"
            )
        }
    binding = min(per_ticker, key=per_ticker.get)
    return per_ticker[binding], {
        "binding_ticker": binding,
        "tickers_ready": sorted(per_ticker),
        "basis": (
            "size/price/reinvestment need only annual financials + price; "
            "this is the first replay date at which any corpus ticker's "
            f"truncated financials reach {min_total_years} years"
        ),
    }


def _battery_complete_rerating(
    loaded: dict[str, dict],
    replay_dates: Sequence[pd.Timestamp],
    *,
    quarterly_lag_months: int,
    shareholding_lag_months: int,
) -> tuple[pd.Timestamp | None, dict]:
    """The fast lane's quarterly-grain gates (`growth_intact`,
    `institutional_accumulation`) need `quarterly`/`shareholding` depth no
    annual frame can supply. "Battery complete" for `rerating` is the
    earliest per-ticker date both frames clear their own gate's depth
    requirement simultaneously — see `_first_lane_gate_depth_date`.
    """
    per_ticker = {
        ticker: date
        for ticker, date in (
            (
                ticker,
                _first_lane_gate_depth_date(
                    data, replay_dates,
                    quarterly_lag_months=quarterly_lag_months,
                    shareholding_lag_months=shareholding_lag_months,
                ),
            )
            for ticker, data in loaded.items()
        )
        if date is not None
    }
    if not per_ticker:
        return None, {
            "reason": (
                "no ticker in the corpus ever supplies both a contiguous "
                f"{GROWTH_INTACT_QUARTERS_NEEDED}-quarter window in `quarterly` "
                f"and a contiguous {INSTITUTIONAL_ACCUMULATION_QUARTERS_NEEDED}-"
                "quarter window in `shareholding` within the replay window"
            )
        }
    binding = min(per_ticker, key=per_ticker.get)
    return per_ticker[binding], {
        "binding_ticker": binding,
        "tickers_ready": sorted(per_ticker),
        "basis": (
            f"growth_intact needs {GROWTH_INTACT_QUARTERS_NEEDED} contiguous "
            "`quarterly` rows and institutional_accumulation needs "
            f"{INSTITUTIONAL_ACCUMULATION_QUARTERS_NEEDED} contiguous "
            "`shareholding` rows; this is the first replay date at which any "
            "corpus ticker clears both"
        ),
    }


# ── the public entry point ──────────────────────────────────────────────


@dataclass
class ReplayCalendar:
    """`compute_calendar`'s return. `battery_complete[lane]` is `None` when
    the corpus never clears that lane's depth requirement inside the
    window — a real possibility this dataclass must be able to say plainly
    rather than crash on, since a corpus that never grows a fast-lane
    candidate is itself a finding.
    """
    dates: list[pd.Timestamp]
    start: pd.Timestamp
    end: pd.Timestamp
    end_basis: str
    dominant_fiscal_month: int
    lag_months: int
    battery_complete: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    battery_detail: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """A JSON-friendly rendering — dates as ISO strings — for the output
        artifact U6 eventually writes."""
        return {
            "dates": [d.date().isoformat() for d in self.dates],
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "end_basis": self.end_basis,
            "dominant_fiscal_month": self.dominant_fiscal_month,
            "lag_months": self.lag_months,
            "battery_complete": {
                lane: (date.date().isoformat() if date is not None else None)
                for lane, date in self.battery_complete.items()
            },
            "battery_detail": self.battery_detail,
        }


def compute_calendar(
    raw_data_dir: str | Path,
    *,
    start: pd.Timestamp = REPLAY_START,
    quarterly_lag_months: int = QUARTERLY_REPORTING_LAG_MONTHS,
    shareholding_lag_months: int = SHAREHOLDING_REPORTING_LAG_MONTHS,
    annual_lag_months: int = ANNUAL_REPORTING_LAG_MONTHS,
    min_total_years: int = MIN_TOTAL_YEARS,
) -> ReplayCalendar:
    """The replay calendar: quarterly dates on the corpus's own fiscal grid,
    from `start` to the last date the price corpus supports, plus each
    lane's battery-complete date.

    Loads every discovered ticker's data once (`universe.load_ticker_data`)
    to answer three questions from the same pass: the dominant fiscal
    month, the corpus's shared price end-date, and each lane's structural
    depth reading. A ticker that fails to load is skipped with a warning —
    it still shows up (or is excluded) in `universe.build_universe`'s own
    pass, which is the one whose exclusions are the load-bearing record.
    """
    raw_data_dir = Path(raw_data_dir)
    loaded: dict[str, dict] = {}
    for ticker_dir in discover_candidates(raw_data_dir):
        try:
            loaded[ticker_dir.name] = load_ticker_data(ticker_dir)
        except Exception as exc:  # noqa: BLE001 — logged, not fatal to the calendar
            logger.warning(f"calendar: could not read {ticker_dir.name}: {exc}")

    if not loaded:
        raise ValueError(f"No tickers discovered under {raw_data_dir}")

    dominant_month = _dominant_period_end_month(
        [d.get("financials") for d in loaded.values()]
    )

    last_priced = {
        ticker: d["price"]["date"].max()
        for ticker, d in loaded.items()
        if isinstance(d.get("price"), pd.DataFrame) and not d["price"].empty
    }
    if not last_priced:
        raise ValueError("No ticker in the corpus has a usable price series")
    binding_ticker = min(last_priced, key=last_priced.get)
    end = last_priced[binding_ticker]

    lag_months = max(quarterly_lag_months, shareholding_lag_months)
    dates = _replay_dates(dominant_month, start, end, lag_months)

    core_date, core_detail = _battery_complete_core(
        loaded, dates, min_total_years=min_total_years, annual_lag_months=annual_lag_months,
    )
    rerating_date, rerating_detail = _battery_complete_rerating(
        loaded, dates,
        quarterly_lag_months=quarterly_lag_months, shareholding_lag_months=shareholding_lag_months,
    )

    return ReplayCalendar(
        dates=dates,
        start=start,
        end=end,
        end_basis=(
            "minimum across discovered tickers' last priced date — the "
            "replay evaluates every active ticker on the same date, so a "
            "date past the shortest-lived series would leave that ticker "
            "unmarkable; binding ticker: " + binding_ticker
        ),
        dominant_fiscal_month=dominant_month,
        lag_months=lag_months,
        battery_complete={CORE_LANE: core_date, RERATING_LANE: rerating_date},
        battery_detail={CORE_LANE: core_detail, RERATING_LANE: rerating_detail},
    )
