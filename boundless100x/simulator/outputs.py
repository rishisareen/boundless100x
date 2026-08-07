"""Readings and the benchmark over already-produced replay data (U6, R7/R8/R9/R10).

**This module is a pure-function library, exactly like `owner.py` (U3) and
`friction_cash.py` (U5) before it.** It never runs a replay itself — U7's
six-step loop (truncate -> score -> evaluate -> propose -> confirm/settle ->
mark to market) does not exist yet, so every function here consumes data
shaped like what `ledger.py` (U4) and `calendar.py`/`universe.py`/`replay.py`
(U2) already produce. **U7's author must conform their loop's output shapes
to the contracts below** — this docstring is the single most important thing
this module produces, more than any individual formula.

## Input contracts

### 1. Equity curve — `list[dict]`, oldest first

Exactly the list of `Ledger.mark_to_market(...)`'s own return dicts, one per
replay date, in the order they were produced. **No new shape is invented**:
`{"date": iso-str, "cash": float, "positions_value": float, "total_value":
float, "marks": {ticker: float}, "stale_marks": [ticker, ...], "basis":
"modeled_capital"}`. This module trusts the ordering — it never re-sorts.

### 2. Trade log — `list[dict]`, in the order trades occurred, EXPLICITLY TAGGED

A list mixing `Ledger.buy()` result dicts (only the `filled: True` ones —
a refusal moved no notional and should not be appended) and `Ledger.sell()`
settlement dicts (one list element per lot touched). **Neither dict carries
a `"kind"` field on its own**, so this module requires the caller to add one
before handing the record in: `{**buy_result, "kind": "buy"}` or
`{**settlement, "kind": "sell"}`. A record missing `"kind"` (or carrying
anything other than `"buy"`/`"sell"`) raises `ValueError` naming the record's
index — a loud failure at the boundary rather than a guess at the shape
inferred from which keys happen to be present. `_split_trade_log` is the one
place this check lives.

### 3. Lane-gate coverage records — `list[dict]`, one per (replay date, ticker)

**The one genuinely new contract this module invents**, because nothing
upstream has produced a *history* of lane-gate readings yet (U2's
`replay.assign_lane` only produced one, at first-eligible-date). Each record:

    {"date": <anything boundless100x.lifecycle.states.as_date can parse —
              a pd.Timestamp, a datetime.date, or an ISO string>,
     "ticker": <str>,
     "lane_gate_result": <LaneGateEvaluator.evaluate()'s own return shape,
                          i.e. {"qualifies": bool|None, "verdict": str,
                          "gates": {gate_id: {"passed": bool|None, ...}},
                          "failed": [...], "indeterminate": [...]}
                          — OR None>}

`lane_gate_result` is `None` for a core-lane ticker at a date `decide()`
never lane-gate-evaluates (mirrors `replay.assign_lane`'s own catalyst=None
convention) — this is DIFFERENT from a gate reading indeterminate. A record
whose `lane_gate_result` is a dict but a particular gate id is absent from
its `"gates"` sub-dict (a custom registry that dropped a gate) is a third,
also-distinguished case. `gate_coverage_matrix` keeps all three apart:
"not_evaluated" (no reading attempted — `lane_gate_result is None`, or the
gate id is absent from a present result's `"gates"` dict) is never folded
into "indeterminate" (a reading was attempted and its own `passed` came back
`None`).

### 4. Exit views — `list[dict] | None`, `ReinvestmentQueue.exit_views()`'s own shape

**This module never calls `ReinvestmentQueue.exit_views()` itself and never
imports `lifecycle.reinvestment`** — a deliberate choice (documented at
`cash_drag`) to keep this module a pure-function library the way `owner.py`
and `friction_cash.py` are, free of a `WatchlistManager`/`ReinvestmentQueue`
dependency this module has no other reason to carry. The caller (U7) already
holds both a live `ReinvestmentQueue` and `WatchlistManager` at the end of a
replay and calls `queue.exit_views(watchlist, as_of=...)` itself, handing the
resulting list here. Each element's `"idle_days"` field (`int | None`,
already computed by `reinvestment._days_between`) is read verbatim — **never
recomputed** (the same KTD1 discipline the rest of this plan applies to
trigger/gate evaluation, applied here to idle-day arithmetic, which is a
production rule living in `reinvestment.py` and three other production
surfaces already).

### 5. Universe / calendar results — the U2 dataclasses, or a duck-typed dict

`describe_exclusions`/`build_limitations` read `UniverseResult.excluded` and
`ReplayCalendar.battery_complete`/`battery_detail` by attribute first, falling
back to `dict.get` — so a test may hand in either the real dataclass or a
plain dict shaped the same way.

### 6. Caller-supplied exclusion inputs (KTD6, KTD0's guard)

Nothing before U7 produces these, so this module defines the shape it
expects and accepts anything conforming:

  * `checkpoint_excluded_transitions`: `None`, an `int` (a bare count), or a
    `list` (items kept and counted).
  * `reconciliation_failures`: a `list[dict]`, each shaped like
    `point_in_time.py`'s own exclusion() — `{"ticker": str, "date": ...,
    "code": one of "withheld_to_prevent_leak"/"never_fetched"/
    "reconciliation_failed"/"non_positive_input", "detail": str}`.

### 7. Benchmark inputs (KTD9)

`build_benchmark_curve` takes `universe_eligible` (`UniverseResult.eligible`,
`{ticker: first_eligible_date}`), `replay_dates` (`ReplayCalendar.dates`),
and `price_frames` (`{ticker: DataFrame}` — the same per-ticker raw price
frame `universe.load_ticker_data(...)["price"]` produces, and the same shape
`Ledger.mark_to_market` itself already consumes). See "Why the benchmark
does not call `Ledger.buy()`" below for the one deliberate divergence from
the plan's literal "one `Ledger.buy()` each" phrasing.

## Output contract

`build_result(...)` returns one plain, `json.dumps`-round-trippable dict —
the artifact (KTD10) minus the simulated owner's own policy-decision log
(U7's to add, since this module never sees `owner.decide()`'s call history):

    {"schema_version": int,
     "equity_curve": [...],  "trade_log": [...],            # echoed back
     "benchmark": {"equity_curve": [...], "trade_log": [...],
                   "metrics": {"aggregate": {...}, "per_lane": {...}}} | None,
     "metrics": {"portfolio_cagr": {...}, "max_drawdown": {...},
                 "turnover": {...}, "per_lane_net_vs_gross": {...},
                 "fast_lane_break_even": {...}, "cash_drag": {...}},
     "gate_coverage": {...},
     "exclusions": [...],
     "limitations": {...}}

Every sub-function is also exported individually — `build_result` is a thin
assembler, not the only way to reach a reading, so U7 (or a test) may call
`portfolio_cagr(equity_curve)` directly without building the whole result.

## Why the benchmark does not call `Ledger.buy()`

The plan's Approach text says "one `Ledger.buy()` each." `Ledger.buy()`'s own
notional sizing (`_tranche_notional`) is the *strategy's* sleeve/tranche
judgment call (ledger.py's own docstring calls it out as one) — it sizes off
the ledger's **current** `total_value` at call time, which drifts as
already-bought tickers' prices move between one benchmark entry and the
next (KTD8 candidates can become eligible years apart). Forcing that formula
to emit a *fixed* per-ticker notional (`starting_pool / n_tickers`, KTD9's
own "equal split of the **starting** pool") would need a fabricated
per-ticker `sleeve_split` recomputed against a total_value nobody has
observed yet without an extra `mark_to_market` call — coupling accounting
steps that do not need to be coupled. KTD9 also states the benchmark exists
specifically to isolate what the strategy's *sizing* rule adds or subtracts
("timing, sizing, exits, and friction") — so the benchmark deliberately uses
a *different* sizing rule (fixed equal-weight) than the strategy's tranche
formula, which argues against reusing that formula at all. `_benchmark_buy`
instead reuses `friction_cash.cost_of_buy` for slippage (same mechanics, R7's
own words) and mutates the **same** `Ledger` instance's public `cash`/
`positions` state directly, mirroring `Ledger.buy()`'s own bookkeeping and
return shape field-for-field. `Ledger.mark_to_market` — the one method that
does the actual valuation work — is reused completely unchanged. This is a
judgment call, surfaced per CLAUDE.md's convention rather than silently
guessed.

One more wrinkle the fixed-notional reading exposes: sizing every ticker's
`notional` at exactly `starting_pool / n` leaves zero cash headroom for entry
slippage, and because `n` tickers' notionals alone already sum to the whole
pool, the *last* ticker(s) bought would always be refused whenever slippage
is nonzero — not an edge case, a structural certainty. `build_benchmark_curve`
therefore splits the pool's own **total cash outlay** (notional plus
slippage) equally instead, solving `notional * (1 + unit_slippage_rate) ==
starting_pool / n` via one extra call to `cost_of_buy(1.0, config)` (which
reads the effective per-unit-notional slippage rate for free) — so all `n`
tickers, once bought, exhaust the pool exactly rather than falling short by a
compounding slippage residue.
"""

from __future__ import annotations

import logging
import statistics
from typing import Sequence

import pandas as pd

from boundless100x import price_bars
from boundless100x.lifecycle.lane_gates import DEFAULT_LANE_GATES
from boundless100x.lifecycle.states import as_date
from boundless100x.simulator import friction_cash, owner
from boundless100x.simulator.ledger import BASIS_MODELED_CAPITAL, QTY_EPSILON, Ledger
from boundless100x.watchlist import CORE_LANE, RERATING_LANE

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# The full six lane-gate ids, read off the shipped registry rather than
# hand-listed a second time (the same "mechanical rather than remembered"
# discipline CLAUDE.md's "Forward signals" section names for FLAG_ELEMENT_MAP).
# `gate_coverage_matrix` unions this with whatever ids actually appear in its
# input records, so a caller running a custom `lane_gates.yaml` still gets a
# complete matrix.
LANE_GATE_IDS = tuple(sorted(DEFAULT_LANE_GATES))


# ── bar-selection hygiene (benchmark buys only) ───────────────────────────
#
# Bar cleaning/selection itself now lives in `boundless100x/price_bars.py`,
# a shared leaf module — this used to be a third local copy of the "on or
# before" hygiene `friction.py` and `ledger.py` each carried locally, before
# the two duplicates that appeared within the simulator (this module and
# `ledger.py`) were consolidated there (Phase 4 residual fix); see that
# module's own docstring for why `friction.py`'s copy stays separate. What
# remains here is `_resolved_bar`, a thin wrapper matching `Ledger.buy()`'s
# own `{date, price}` bar contract, plus the pre-cleaned-frame caches
# `build_benchmark_curve`/`lane_position_value_curve` build below so neither
# re-cleans a ticker's whole raw history once per (ticker, date) pair.


def _resolved_bar(price_df, as_of) -> dict | None:
    """The `{date, price}` bar contract `Ledger.buy()`-shaped calls expect:
    the last usable close on or before `as_of`, together with its OWN date
    (which may be earlier than `as_of` across a gap in the series).

    `price_df` may be raw (`close`/`adj_close`) or already cleaned (a
    `price_bars.clean_price_bars` frame, carrying `price`) — see
    `price_bars.bar_on_or_before`'s own docstring for how the two are told
    apart. Callers here that hold a pre-cleaned cache pass the cleaned frame
    straight through and skip the re-clean.
    """
    as_of_date = as_date(as_of)
    if as_of_date is None:
        return None
    return price_bars.bar_on_or_before(price_df, as_of_date)


# ── trade-log kind tagging (input contract #2) ────────────────────────────

_VALID_KINDS = ("buy", "sell")


def _split_trade_log(trade_log: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Split a tagged trade log into (buys, sells). See the module docstring's
    Input Contract #2 — every record must carry `"kind"`, added by the caller.
    """
    buys, sells = [], []
    for index, record in enumerate(trade_log or []):
        kind = record.get("kind")
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"outputs: trade_log[{index}] carries kind={kind!r} — every record "
                f"must be tagged 'kind': 'buy' or 'kind': 'sell' by the caller before "
                f"being handed to this module (module docstring, Input Contract #2); "
                f"neither Ledger.buy() nor Ledger.sell()'s own return dicts carry this "
                f"key on their own"
            )
        if kind == "buy":
            if not record.get("filled", True):
                continue  # a refused buy moved no notional — not a trade
            buys.append(record)
        else:
            sells.append(record)
    return buys, sells


# ── elapsed-time helper, shared by CAGR and turnover's annualization ──────


def _elapsed_days(equity_curve: Sequence[dict]) -> tuple[int | None, str | None, str | None]:
    if not equity_curve:
        return None, None, None
    start_date = as_date(equity_curve[0].get("date"))
    end_date = as_date(equity_curve[-1].get("date"))
    if start_date is None or end_date is None:
        return None, None, None
    return (end_date - start_date).days, str(start_date), str(end_date)


# ── 1a. Portfolio CAGR ──────────────────────────────────────────────────


def portfolio_cagr(equity_curve: list[dict], *, value_key: str = "total_value") -> dict:
    """CAGR from the curve's first/last `value_key`, annualized over the
    actual elapsed calendar days between the curve's first and last date
    (`365 / elapsed_days`) — the same annualization convention `turnover`
    below states explicitly and reuses.
    """
    if not equity_curve:
        return {"cagr_pct": None, "note": "empty equity curve — nothing to annualize"}

    elapsed_days, start_date, end_date = _elapsed_days(equity_curve)
    start_value = equity_curve[0].get(value_key)
    end_value = equity_curve[-1].get(value_key)

    if elapsed_days is None:
        return {
            "cagr_pct": None,
            "start_value": start_value, "end_value": end_value,
            "note": "unreadable date on the curve's first or last point",
        }
    if elapsed_days <= 0:
        return {
            "cagr_pct": None, "elapsed_days": elapsed_days,
            "start_date": start_date, "end_date": end_date,
            "start_value": start_value, "end_value": end_value,
            "note": "curve spans zero or negative elapsed days — cannot annualize",
        }
    if start_value is None or start_value <= 0:
        return {
            "cagr_pct": None, "elapsed_days": elapsed_days,
            "start_date": start_date, "end_date": end_date,
            "start_value": start_value, "end_value": end_value,
            "note": f"non-positive starting {value_key} — cannot annualize a ratio off it",
        }
    if end_value is None:
        return {
            "cagr_pct": None, "elapsed_days": elapsed_days,
            "start_date": start_date, "end_date": end_date,
            "start_value": start_value, "end_value": end_value,
            "note": f"missing ending {value_key}",
        }

    cagr = (end_value / start_value) ** (365.0 / elapsed_days) - 1.0
    return {
        "cagr_pct": cagr * 100.0,
        "elapsed_days": elapsed_days,
        "start_date": start_date, "end_date": end_date,
        "start_value": start_value, "end_value": end_value,
        "note": "",
    }


# ── 1b. Max drawdown ────────────────────────────────────────────────────


def max_drawdown(equity_curve: list[dict], *, value_key: str = "total_value") -> dict:
    """The largest peak-to-trough decline in `value_key` across the curve,
    walked forward once (running peak, no re-sorting — trusts input order).
    """
    if not equity_curve:
        return {"max_drawdown_pct": None, "note": "empty equity curve"}

    peak_value = None
    peak_date = None
    worst = 0.0
    worst_peak_value = worst_peak_date = worst_trough_value = worst_trough_date = None

    for point in equity_curve:
        value = point.get(value_key)
        point_date = point.get("date")
        if value is None:
            continue
        if peak_value is None or value > peak_value:
            peak_value = value
            peak_date = point_date
        if peak_value and peak_value > 0:
            drawdown = (peak_value - value) / peak_value
            if drawdown > worst:
                worst = drawdown
                worst_peak_value, worst_peak_date = peak_value, peak_date
                worst_trough_value, worst_trough_date = value, point_date

    if peak_value is None:
        return {"max_drawdown_pct": None, "note": f"no usable {value_key} readings on the curve"}

    return {
        "max_drawdown_pct": worst * 100.0,
        "peak_date": worst_peak_date, "peak_value": worst_peak_value,
        "trough_date": worst_trough_date, "trough_value": worst_trough_value,
        "note": "" if worst > 0 else "no drawdown observed — the curve never fell below a prior peak",
    }


# ── 1c. Turnover ─────────────────────────────────────────────────────────


def turnover(equity_curve: list[dict], trade_log: list[dict]) -> dict:
    """Traded notional over the curve's MEAN `total_value`, annualized by
    `365 / elapsed_days` — the same convention `portfolio_cagr` states.

    Traded notional: a buy's own `notional` field; a sell settlement's
    `qty * exit_price` (gross proceeds before slippage/tax — derivable from
    the settlement dict's own fields, per the plan's own words).
    """
    buys, sells = _split_trade_log(trade_log)
    traded_notional = sum(b["notional"] for b in buys) + sum(
        s["qty"] * s["exit_price"] for s in sells
    )

    values = [p.get("total_value") for p in equity_curve if p.get("total_value") is not None]
    if not values:
        return {
            "traded_notional": traded_notional,
            "turnover_ratio_annualized": None,
            "note": "empty equity curve — cannot annualize",
        }
    mean_value = sum(values) / len(values)

    elapsed_days, start_date, end_date = _elapsed_days(equity_curve)
    if not elapsed_days or elapsed_days <= 0 or mean_value <= 0:
        return {
            "traded_notional": traded_notional,
            "mean_total_value": mean_value,
            "turnover_ratio_annualized": None,
            "note": "cannot annualize — zero/negative elapsed days or mean total_value",
        }

    annualization_factor = 365.0 / elapsed_days
    raw_ratio = traded_notional / mean_value
    return {
        "traded_notional": traded_notional,
        "mean_total_value": mean_value,
        "elapsed_days": elapsed_days,
        "start_date": start_date, "end_date": end_date,
        "annualization_factor": annualization_factor,
        "turnover_ratio": raw_ratio,
        "turnover_ratio_annualized": raw_ratio * annualization_factor,
        "note": "annualized by 365/elapsed_days, matching portfolio_cagr's own convention",
    }


# ── 1d. Per-lane net-vs-gross ──────────────────────────────────────────


def per_lane_net_vs_gross(trade_log: list[dict]) -> dict:
    """R7's "per-lane net-vs-gross," grouped by each sell settlement's own
    `lane`. This module's chosen pair (documented so a reader never has to
    guess): **gross** = `qty * (exit_price - entry_price)`, no friction at
    all; **net** = `proceeds - (qty * entry_price)`, net of slippage AND
    tax. The settlement's own `gain` (net of slippage, pre-tax) is reported
    alongside as `net_of_slippage_pretax` for transparency but is NOT the
    official "net" half of the pair.
    """
    _, sells = _split_trade_log(trade_log)
    by_lane: dict[str, dict] = {}
    for settlement in sells:
        lane = settlement.get("lane", "unknown")
        bucket = by_lane.setdefault(lane, {
            "n_settlements": 0, "gross": 0.0, "net": 0.0, "net_of_slippage_pretax": 0.0,
        })
        bucket["n_settlements"] += 1
        bucket["gross"] += settlement["qty"] * (settlement["exit_price"] - settlement["entry_price"])
        bucket["net"] += settlement["proceeds"] - settlement["qty"] * settlement["entry_price"]
        bucket["net_of_slippage_pretax"] += settlement["gain"]

    return {
        "definition": (
            "gross = qty * (exit_price - entry_price), no friction at all. "
            "net = proceeds - (qty * entry_price), net of slippage AND tax — the pair "
            "this module calls 'net-vs-gross'. net_of_slippage_pretax (== the "
            "settlement's own 'gain') is reported for transparency only."
        ),
        "by_lane": by_lane,
    }


# ── 1e. Fast-lane break-even ────────────────────────────────────────────


def fast_lane_break_even(trade_log: list[dict], *, central_tendency: str = "median") -> dict:
    """§8.2's annualized-return gap a fast-lane round trip must clear —
    derived from MEASURED friction on actual rerating-lane sell settlements
    (the number Phase 3 declined to compute), never `friction_cash`'s
    theoretical per-leg rates.

    Each rerating-lane settlement already carries its own entry/exit price
    and `holding_days`; the gross and net returns are each annualized by
    that cycle's own holding period, and the gap between them is the
    reading. Zero usable cycles (very plausible on the real corpus — the
    rerating battery does not complete until ~2024-05-31) reports
    `status: "unmeasured"` with a reason, never a bare `0`/`None` that
    would read as "no gap."
    """
    _, sells = _split_trade_log(trade_log)
    cycles = [s for s in sells if s.get("lane") == RERATING_LANE]

    usable = []
    skipped = 0
    for settlement in cycles:
        holding_days = settlement.get("holding_days")
        entry_price = settlement.get("entry_price")
        qty = settlement.get("qty")
        if not holding_days or holding_days <= 0 or not entry_price or entry_price <= 0 or not qty:
            skipped += 1
            continue

        gross_return = (settlement["exit_price"] - entry_price) / entry_price
        net_return = (settlement["proceeds"] - qty * entry_price) / (qty * entry_price)
        factor = 365.0 / holding_days
        annualized_gross = (1.0 + gross_return) ** factor - 1.0
        annualized_net = (1.0 + net_return) ** factor - 1.0
        usable.append({
            "ticker": settlement.get("ticker"),
            "entry_bar_date": settlement.get("entry_bar_date"),
            "exit_bar_date": settlement.get("exit_bar_date"),
            "holding_days": holding_days,
            "annualized_gross_pct": annualized_gross * 100.0,
            "annualized_net_pct": annualized_net * 100.0,
            "annualized_gap_pct": (annualized_gross - annualized_net) * 100.0,
        })

    if not usable:
        return {
            "status": "unmeasured",
            "reason": (
                "no rerating-lane sell settlement with a usable holding period was "
                "observed in this run's trade log — either the rerating battery never "
                "completed within the replay window (see ReplayCalendar."
                "battery_complete['rerating']), every rerating-lane candidate this run "
                "entered was still open at the end of the replay, or none entered the "
                "lane at all"
            ),
            "n_cycles": 0,
            "n_skipped": skipped,
        }

    gaps = [c["annualized_gap_pct"] for c in usable]
    if central_tendency not in ("mean", "median"):
        central_tendency = "median"
    central = (sum(gaps) / len(gaps)) if central_tendency == "mean" else statistics.median(gaps)

    return {
        "status": "measured",
        "central_tendency": central_tendency,
        "break_even_gap_pct": central,
        "n_cycles": len(usable),
        "n_skipped": skipped,
        "cycles": usable,
        "note": (
            "the annualized-return gap (gross minus net) a fast-lane round trip must "
            "clear (§8.2), from measured friction on this run's own rerating-lane exits"
        ),
    }


# ── 1f. Cash drag ────────────────────────────────────────────────────────


def cash_drag(equity_curve: list[dict], exit_views: list[dict] | None = None) -> dict:
    """Mean/median idle days between exit and redeployment (from
    `exit_views`, see Input Contract #4 — never recomputed here) plus,
    separately, the pool's own idle share of time: at each equity-curve
    date, `cash / total_value`, averaged across the curve.
    """
    exit_views = exit_views or []
    idle_days = [v["idle_days"] for v in exit_views if v.get("idle_days") is not None]
    n_unreadable = sum(1 for v in exit_views if v.get("idle_days") is None)

    idle_reading = {
        "n_exits": len(exit_views),
        "n_with_readable_idle_days": len(idle_days),
        "n_unreadable_idle_days": n_unreadable,
        "mean_idle_days": (sum(idle_days) / len(idle_days)) if idle_days else None,
        "median_idle_days": statistics.median(idle_days) if idle_days else None,
        "note": (
            "idle_days read verbatim from ReinvestmentQueue.exit_views() — never "
            "recomputed here (module docstring, Input Contract #4)"
            if exit_views else
            "no exit views were supplied — either no exit occurred in this run, or "
            "the caller did not pass any"
        ),
    }

    shares = [
        point["cash"] / point["total_value"]
        for point in equity_curve
        if point.get("cash") is not None
        and point.get("total_value") is not None
        and point["total_value"] > 0
    ]
    pool_reading = {
        "n_points": len(shares),
        "mean_idle_share": (sum(shares) / len(shares)) if shares else None,
        "median_idle_share": statistics.median(shares) if shares else None,
        "note": (
            "cash / total_value at every equity-curve point, averaged across the curve"
            if shares else
            "no usable equity-curve points to compute an idle-cash share from"
        ),
    }
    return {"idle_days": idle_reading, "pool_idle_share": pool_reading}


# ── 2. The benchmark (KTD9) ──────────────────────────────────────────────


def _benchmark_buy(ledger: Ledger, ticker: str, notional: float, bar: dict, config: dict | None) -> dict:
    """Open one benchmark lot of exactly `notional` — the equal-weight share
    of the ORIGINAL starting pool computed once by the caller — reusing
    `friction_cash.cost_of_buy` for entry slippage and `ledger`'s own public
    `cash`/`positions` as the container. See the module docstring's "Why the
    benchmark does not call Ledger.buy()" for why this mirrors, rather than
    calls, `Ledger.buy()`.
    """
    price = bar["price"]
    entry_date = bar["date"]
    if price is None or price <= 0:
        raise ValueError(f"outputs._benchmark_buy: bar price {price!r} is not positive")
    if notional is None or notional <= 0:
        return {
            "filled": False, "ticker": ticker, "reason": "non-positive equal-weight notional",
            "basis": BASIS_MODELED_CAPITAL,
        }

    slippage = friction_cash.cost_of_buy(notional, config)
    total_cost = notional + slippage
    if total_cost > ledger.cash:
        return {
            "filled": False, "ticker": ticker,
            "reason": (
                f"insufficient modeled cash for the benchmark's equal-weight buy: needs "
                f"{total_cost!r} (notional {notional!r} + entry slippage {slippage!r}) "
                f"against {ledger.cash!r} cash on hand"
            ),
            "basis": BASIS_MODELED_CAPITAL,
        }

    qty = notional / price
    tranche_index = len(ledger.positions.get(ticker, []))
    lot = {
        "qty": qty, "entry_bar_date": entry_date, "entry_price": price,
        "lane": "benchmark", "tranche_index": tranche_index,
    }
    ledger.positions.setdefault(ticker, []).append(lot)
    ledger.cash -= total_cost

    return {
        "filled": True, "ticker": ticker, "tranche_index": tranche_index,
        "notional": notional, "slippage": slippage, "qty": qty, "price": price,
        "entry_bar_date": str(entry_date), "cash_after": ledger.cash,
        "basis": BASIS_MODELED_CAPITAL,
    }


def build_benchmark_curve(
    universe_eligible: dict[str, "pd.Timestamp"],
    replay_dates: Sequence,
    price_frames: dict[str, pd.DataFrame],
    config: dict | None = None,
    *,
    starting_pool: float | None = None,
    ticker_lanes: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """KTD9's benchmark: an equal-weight position per universe ticker as it
    becomes eligible, funded from the same starting pool, marked on the same
    bars, charged the same entry slippage, never sold, taxed nothing.

    `ticker_lanes` (optional, `{ticker: "core"|"rerating"}` — typically
    `{ticker: assignment.lane for ticker, assignment in assignments.items()}`
    off U2's `build_initial_watchlist` return) tags each buy record's own
    `"lane"` with the ticker's REAL strategy-assigned lane, which is what
    lets `lane_position_value_curve` build a per-lane benchmark reading
    (KTD9: "stated per lane as well as in aggregate"). Untagged tickers fall
    back to `"unknown"`.

    Returns `(equity_curve, trade_log)` — both already shaped per this
    module's own Input Contracts #1/#2 (the trade log is pre-tagged
    `"kind": "buy"`).
    """
    ledger = Ledger(config=config, starting_pool=starting_pool)
    equity_curve: list[dict] = []
    trade_log: list[dict] = []

    tickers_by_date = sorted(
        universe_eligible.items(), key=lambda kv: (as_date(kv[1]) or as_date("9999-12-31"), kv[0])
    )
    n = len(tickers_by_date)
    if n == 0:
        return equity_curve, trade_log

    # Cleaned once per ticker, up front, rather than re-parsed/re-cleaned by
    # `_resolved_bar` on every (ticker, replay_date) pair below — the raw
    # frames are static for the whole call. `Ledger.mark_to_market` (called
    # per date, below) does its own equivalent caching internally.
    cleaned_frames = {ticker: price_bars.clean_price_bars(df) for ticker, df in price_frames.items()}

    # An equal split of the starting pool's own TOTAL cash outlay (notional
    # PLUS entry slippage), not of notional alone — sizing notional to
    # `cash / n` would leave zero headroom for slippage, and since n
    # tickers' notionals alone already sum to the whole pool, the LAST
    # ticker(s) bought would always be refused for insufficient cash
    # whenever slippage is nonzero (a structural shortfall, not an edge
    # case). `cost_of_buy(1.0, config)` reads the effective per-unit-notional
    # slippage rate off the same `friction_cash.cost_of_buy` this module
    # already reuses, so solving `notional * (1 + rate) == target_cost` needs
    # no second read of `slippage_bps`.
    target_cost = ledger.cash / n
    unit_slippage_rate = friction_cash.cost_of_buy(1.0, config)
    target_notional = target_cost / (1.0 + unit_slippage_rate)
    ticker_lanes = ticker_lanes or {}

    sorted_dates = sorted(d for d in (as_date(x) for x in replay_dates) if d is not None)

    for replay_date in sorted_dates:
        for ticker, first_eligible in tickers_by_date:
            # A ticker enters `ledger.positions` exactly when (and only
            # when) `_benchmark_buy` below successfully adds it — the
            # benchmark never sells, so nothing is ever removed either.
            # Checking `ledger.positions` directly retires a redundant
            # local `bought` set this module used to track in parallel.
            if ticker in ledger.positions:
                continue
            first_eligible_date = as_date(first_eligible)
            if first_eligible_date is None or replay_date < first_eligible_date:
                continue
            resolved = _resolved_bar(cleaned_frames.get(ticker), replay_date)
            if resolved is None:
                continue  # no usable bar yet — retry on a later replay date
            result = _benchmark_buy(ledger, ticker, target_notional, resolved, config)
            lane = ticker_lanes.get(ticker, "unknown")
            trade_log.append({**result, "kind": "buy", "lane": lane})

        equity_curve.append(ledger.mark_to_market(replay_date, price_frames))

    return equity_curve, trade_log


def lane_position_value_curve(
    trade_log: list[dict], price_frames: dict[str, pd.DataFrame], dates: Sequence, lane: str,
) -> list[dict]:
    """A POSITION-VALUE-ONLY value series for one lane — deliberately NOT a
    cash-inclusive equity curve, because a `Ledger`'s cash is one shared,
    fungible pool with no per-lane split to attribute it from. This is the
    proxy KTD9's "stated per lane as well as in aggregate" resolves to for
    the benchmark (which has no lane-scoped settlements to read a net-vs-
    gross figure from, unlike the strategy). Reusable for the strategy's own
    trade log too, since it handles both buy and sell (FIFO) records.

    Returns rows shaped `{"date": iso-str, "positions_value": float,
    "tickers_held": [ticker, ...]}` — feed `value_key="positions_value"`
    into `portfolio_cagr`/`max_drawdown` to read a lane-scoped CAGR/drawdown
    off it.

    **This FIFO quantity-consumption walk is deliberately separate from,
    and simpler than, `Ledger.sell()`'s own FIFO** — open quantity only,
    reconstructed from a flat historical trade log, for a lane the ledger
    itself does not track natively. `Ledger.sell()` additionally handles
    cost basis, tax and slippage against live per-lot state, which this
    function has no need of (it only ever wants a quantity still held at a
    given date). The two are not extracted into one shared helper on
    purpose: they walk genuinely different data (a live `Ledger`'s lot
    dicts vs. a flat replay of past buy/sell records), and forcing a shared
    abstraction across that difference would be over-simplification, not
    the kind of consolidation `price_bars.py` above is (same hazard, same
    data shape, three copies of one another) — different problem, different
    answer.
    """
    parsed_dates = sorted(d for d in (as_date(x) for x in dates) if d is not None)

    # Cleaned once per ticker, up front — see `build_benchmark_curve`'s own
    # note; the same static `price_frames` would otherwise be re-parsed and
    # re-cleaned once per (ticker, date) pair in the loop below.
    cleaned_frames = {ticker: price_bars.clean_price_bars(df) for ticker, df in price_frames.items()}

    events: list[tuple] = []
    for record in trade_log:
        if record.get("lane") != lane:
            continue
        kind = record.get("kind")
        if kind == "buy":
            event_date = as_date(record.get("entry_bar_date"))
        elif kind == "sell":
            event_date = as_date(record.get("exit_bar_date"))
        else:
            continue
        if event_date is None:
            continue
        events.append((event_date, kind, record))
    events.sort(key=lambda e: e[0])

    open_qty: dict[str, list[float]] = {}
    curve: list[dict] = []
    idx = 0
    for point_date in parsed_dates:
        while idx < len(events) and events[idx][0] <= point_date:
            _, kind, record = events[idx]
            ticker = record["ticker"]
            if kind == "buy":
                open_qty.setdefault(ticker, []).append(record["qty"])
            else:
                remaining = record["qty"]
                lots = open_qty.get(ticker, [])
                survivors = []
                for lot_qty in lots:
                    if remaining <= 0:
                        survivors.append(lot_qty)
                        continue
                    consume = min(lot_qty, remaining)
                    remaining -= consume
                    leftover = lot_qty - consume
                    if leftover > QTY_EPSILON:
                        survivors.append(leftover)
                open_qty[ticker] = survivors
            idx += 1

        positions_value = 0.0
        tickers_held = []
        for ticker, lots in open_qty.items():
            total_qty = sum(lots)
            if total_qty <= 0:
                continue
            resolved = _resolved_bar(cleaned_frames.get(ticker), point_date)
            if resolved is None:
                continue  # no usable price yet — excluded from this point's value
            positions_value += total_qty * resolved["price"]
            tickers_held.append(ticker)

        curve.append({
            "date": str(point_date), "positions_value": positions_value,
            "tickers_held": sorted(tickers_held),
        })
    return curve


# ── 3. Gate coverage, per gate per window ─────────────────────────────


def gate_coverage_matrix(records: list[dict] | None) -> dict:
    """Per gate, per replay-date window — not per run (U6's most
    load-bearing single design point; see the plan's KTD7/U6.3). A window
    with zero lane-gate evaluations reports the lane as `"unmeasured"` with
    a reason, never as all-zero pass/fail counts that would read
    indistinguishably from "assessed and found wanting" (mirrors
    `lane_gates.py`'s own "no watchlist context" vs. "assessed" distinction).

    Every `qualifies`/`verdict` reading is traceable: each window's
    `"readings"` list carries one entry per (ticker, lane_gate_result not
    None) record, with `"deciding_gates"` echoing every gate id's own
    `passed` value — so a near-miss (most gates pass, one or two
    indeterminate, verdict NOT_QUALIFIED/INDETERMINATE) can be attributed to
    the exact gates that decided it, not just the aggregate verdict.
    """
    records = records or []
    if not records:
        return {
            "status": "unmeasured",
            "reason": "no gate-coverage records were supplied for this run",
            "gate_ids": list(LANE_GATE_IDS),
            "windows": {},
        }

    by_window: dict[str, list[dict]] = {}
    for index, record in enumerate(records):
        parsed = as_date(record.get("date"))
        if parsed is None:
            raise ValueError(
                f"outputs.gate_coverage_matrix: records[{index}] carries an unreadable "
                f"date {record.get('date')!r}"
            )
        by_window.setdefault(str(parsed), []).append(record)

    gate_ids = set(LANE_GATE_IDS)
    for window_records in by_window.values():
        for record in window_records:
            result = record.get("lane_gate_result")
            if result is not None:
                gate_ids.update((result.get("gates") or {}).keys())
    gate_ids = sorted(gate_ids)

    windows_out: dict[str, dict] = {}
    for window_key in sorted(by_window):
        window_records = by_window[window_key]
        measured_records = [r for r in window_records if r.get("lane_gate_result") is not None]

        if not measured_records:
            windows_out[window_key] = {
                "status": "unmeasured",
                "reason": (
                    "no ticker had lane gates evaluated on this replay date — every "
                    "supplied record's lane_gate_result was None (core-lane tickers, "
                    "which decide() never lane-gate-evaluates), or there were no "
                    "records at all"
                ),
                "n_records": len(window_records),
                "gates": {},
                "readings": [],
            }
            continue

        gate_counts = {
            gate_id: {"passed": 0, "failed": 0, "indeterminate": 0, "not_evaluated": 0}
            for gate_id in gate_ids
        }
        readings = []
        for record in window_records:
            result = record.get("lane_gate_result")
            deciding: dict[str, bool | None] = {}
            for gate_id in gate_ids:
                gate_detail = (result or {}).get("gates", {}).get(gate_id) if result is not None else None
                if result is None or gate_detail is None:
                    gate_counts[gate_id]["not_evaluated"] += 1
                    deciding[gate_id] = None
                    continue
                passed = gate_detail.get("passed")
                deciding[gate_id] = passed
                if passed is True:
                    gate_counts[gate_id]["passed"] += 1
                elif passed is False:
                    gate_counts[gate_id]["failed"] += 1
                else:
                    gate_counts[gate_id]["indeterminate"] += 1

            if result is not None:
                readings.append({
                    "ticker": record.get("ticker"),
                    "verdict": result.get("verdict"),
                    "qualifies": result.get("qualifies"),
                    "deciding_gates": deciding,
                })

        windows_out[window_key] = {
            "status": "measured",
            "reason": "",
            "n_records": len(window_records),
            "n_measured": len(measured_records),
            "gates": gate_counts,
            "readings": readings,
        }

    return {"status": "measured", "reason": "", "gate_ids": gate_ids, "windows": windows_out}


# ── 4. Exclusions and limitations (R8, backtest idiom) ────────────────


def describe_exclusions(
    *,
    universe_result=None,
    checkpoint_excluded_transitions=None,
    gate_coverage_result: dict | None = None,
    equity_curve: list[dict] | None = None,
    reconciliation_failures: list[dict] | None = None,
) -> list[dict]:
    """Every exclusion kind R8 names, following `backtest._describe_exclusions`'s
    own rendering idiom (a sorted list of `{"category", "count", ...}` dicts).
    """
    exclusions: list[dict] = []

    # 1. Never-eligible tickers (KTD8).
    excluded_map = getattr(universe_result, "excluded", None)
    if excluded_map is None and isinstance(universe_result, dict):
        excluded_map = universe_result.get("excluded")
    excluded_map = excluded_map or {}
    exclusions.append({
        "category": "never_eligible_tickers",
        "count": len(excluded_map),
        "detail": (
            "raw_data/ tickers whose truncated financials never clear the engine's "
            "minimum-years bar within the replay window (KTD8)"
        ),
        "items": dict(sorted(excluded_map.items())),
    })

    # 2. Checkpoint-driven transitions excluded (R9/KTD6).
    if checkpoint_excluded_transitions is None:
        checkpoint_items, checkpoint_count = [], 0
    elif isinstance(checkpoint_excluded_transitions, int):
        checkpoint_items, checkpoint_count = [], checkpoint_excluded_transitions
    else:
        checkpoint_items = list(checkpoint_excluded_transitions)
        checkpoint_count = len(checkpoint_items)
    exclusions.append({
        "category": "checkpoint_driven_transitions_excluded",
        "count": checkpoint_count,
        "detail": (
            "transitions whose conditions need LLM-produced checkpoint inputs (R9) — "
            "unreplayable; validated on the annual grain until organic quarterly "
            "history accumulates (§10 rev)"
        ),
        "items": checkpoint_items,
    })

    # 3. Gate-indeterminate readings, attributable per gate per window (point 3).
    indeterminate_total = 0
    per_gate: dict[str, int] = {}
    if gate_coverage_result and gate_coverage_result.get("status") == "measured":
        for window in gate_coverage_result.get("windows", {}).values():
            if window.get("status") != "measured":
                continue
            for gate_id, counts in window.get("gates", {}).items():
                n = counts.get("indeterminate", 0)
                indeterminate_total += n
                per_gate[gate_id] = per_gate.get(gate_id, 0) + n
    exclusions.append({
        "category": "gate_indeterminate_readings",
        "count": indeterminate_total,
        "detail": (
            "lane-gate readings that came back indeterminate (missing inputs), "
            "attributable per gate per replay-date window — see this result's own "
            "gate_coverage"
        ),
        "items": dict(sorted(per_gate.items())),
    })

    # 4. Stale-mark events.
    stale_total = 0
    stale_by_ticker: dict[str, int] = {}
    for point in equity_curve or []:
        for ticker in point.get("stale_marks") or []:
            stale_total += 1
            stale_by_ticker[ticker] = stale_by_ticker.get(ticker, 0) + 1
    exclusions.append({
        "category": "stale_mark_events",
        "count": stale_total,
        "detail": (
            "equity-curve points where a ticker had no usable bar on or before the "
            "mark date and was carried at its last known mark instead"
        ),
        "items": dict(sorted(stale_by_ticker.items())),
    })

    # 5. Reconciliation failures (KTD0's guard).
    reconciliation_failures = list(reconciliation_failures or [])
    by_code: dict[str, int] = {}
    for item in reconciliation_failures:
        code = item.get("code", "unknown")
        by_code[code] = by_code.get(code, 0) + 1
    exclusions.append({
        "category": "reconciliation_failures",
        "count": len(reconciliation_failures),
        "detail": (
            "point-in-time Market Cap / Stock P/E rebuilds (KTD0) that were withheld, "
            "never had the inputs, or failed the reconciliation-tolerance guard — coded "
            "withheld_to_prevent_leak / never_fetched / reconciliation_failed / "
            "non_positive_input, exactly as point_in_time.py's own exclusion() shape"
        ),
        "items_by_code": dict(sorted(by_code.items())),
        "items": reconciliation_failures,
    })

    return exclusions


def build_limitations(
    *, calendar_result=None, gate_coverage_result: dict | None = None, config: dict | None = None,
) -> dict:
    """R8's limitations block: survivorship/upper-bound (§14.6), quarterly
    depth (§10 rev), the fast-lane gate-coverage caveat (present from day
    one — point 3 — never derived after the fact), the rebuilt-multiple
    basis divergence (KTD0), every simulated-owner policy by name (pulled
    from `owner.config_from`, not just asserted "applied"), and the
    statistical-humility clause (§12 Phase 5 rev, quoted).
    """
    owner_settings = owner.config_from(config)

    battery_complete = getattr(calendar_result, "battery_complete", None)
    if battery_complete is None and isinstance(calendar_result, dict):
        battery_complete = calendar_result.get("battery_complete")
    battery_complete = battery_complete or {}

    battery_detail = getattr(calendar_result, "battery_detail", None)
    if battery_detail is None and isinstance(calendar_result, dict):
        battery_detail = calendar_result.get("battery_detail")
    battery_detail = battery_detail or {}

    def _fmt_battery(lane: str) -> str:
        value = battery_complete.get(lane)
        if value is None:
            reason = (battery_detail.get(lane) or {}).get("reason", "no reason recorded")
            return f"never completes within the replay window — {reason}"
        return value.date().isoformat() if hasattr(value, "date") else str(value)

    n_measured = n_unmeasured = 0
    if gate_coverage_result:
        for window in gate_coverage_result.get("windows", {}).values():
            if window.get("status") == "measured":
                n_measured += 1
            else:
                n_unmeasured += 1

    return {
        "survivorship_and_upper_bound": (
            "The universe is the survivorship-selected raw_data/ corpus (§14.6), not a "
            "point-in-time screen — companies that failed and were never fetched cannot "
            "appear. Every reading in this result is an UPPER BOUND until a "
            "point-in-time universe exists; building one is real work, deliberately not "
            "gated on for this phase."
        ),
        "quarterly_depth": (
            "Replay evaluation is quarterly-grain (KTD7) on the corpus's own fiscal "
            "calendar, over daily-grain pricing. Screener renders only ~11-13 quarters, "
            "so quarterly.csv begins Mar/Jun 2023 and shareholding.csv Sep 2023, and no "
            "refetch can deepen either. The fast lane's two quarterly-grain gates "
            "(growth_intact, institutional_accumulation) are structurally indeterminate "
            f"before the corpus's own battery-complete date — core: "
            f"{_fmt_battery(CORE_LANE)}, rerating: {_fmt_battery(RERATING_LANE)}."
        ),
        "fast_lane_gate_coverage": (
            "Named from day one, not derived after the fact: a fast-lane 'qualifies' "
            "verdict is only as strong as the gates actually computable at that "
            "reading — see this result's gate_coverage for exactly which gates decided "
            f"which verdict, at which replay date. {n_measured} window(s) in this run "
            f"had at least one lane-gate reading; {n_unmeasured} had none and are "
            "reported as unmeasured rather than as 'no qualifying candidates'."
        ),
        "rebuilt_multiple_basis": (
            "Market Cap and Stock P/E are rebuilt from truncatable inputs (KTD0), not "
            "read from Screener's stored figures. Stock P/E is raw close over the "
            "latest annual EPS (valuation._current_multiple's own basis, carried as "
            "_stock_pe_basis in metadata) — NOT Screener's stored TTM Stock P/E, a "
            "different multiple entirely (KTD0 measured -14% to +1169% divergence "
            "between the two). Any comparison against a production report's own Stock "
            "P/E for the same ticker/date compares two different multiples by "
            "construction."
        ),
        "simulated_owner_policies": {
            "starting_pool": owner_settings["starting_pool"],
            "confirmation_lag_days": owner_settings["confirmation_lag_days"],
            "catalyst_window_months": owner_settings["catalyst_window_months"],
            "cap_posture": owner_settings["cap_posture"],
            "reduce_fraction": owner_settings["reduce_fraction"],
            "severity_overrides": owner_settings["severity_overrides"],
            "note": (
                "every value the simulated owner (owner.py, KTD3/KTD6) actually used "
                "this run, read from owner.config_from(config) rather than asserted as "
                "'policies were applied'; the fabricated fast-lane catalyst (KTD6) is "
                "one of them — catalyst_window_months above is its window, not an "
                "owner-observed input"
            ),
        },
        "statistical_humility": (
            "Quoted from the design doc's own Phase 5 clause (rev 2026-08-06b), "
            "included here even though the sweeps themselves are Phase 5's: "
            '"sweep outcomes over the small survivorship-selected universe are '
            "directional only — a sweep may suggest a parameter change, but acting on "
            "one requires a minimum transition count and follows the same documented "
            "before/after evidence rule as any other Phase 5 retune. Sweeps that would "
            'fit noise are reported as noise, not as settings."'
        ),
    }


# ── 5. The assembled artifact + a thin renderer ────────────────────────


def build_result(
    *,
    equity_curve: list[dict],
    trade_log: list[dict],
    benchmark_equity_curve: list[dict] | None = None,
    benchmark_trade_log: list[dict] | None = None,
    benchmark_per_lane: dict[str, list[dict]] | None = None,
    lane_gate_records: list[dict] | None = None,
    exit_views: list[dict] | None = None,
    universe_result=None,
    calendar_result=None,
    checkpoint_excluded_transitions=None,
    reconciliation_failures=None,
    config: dict | None = None,
) -> dict:
    """Assemble every reading into one plain, JSON-round-trippable dict — the
    artifact (KTD10) minus the simulated owner's own decision log (U7 adds
    that at the top level, since this module never sees `owner.decide()`'s
    call history). See the module docstring's Output Contract for the shape.
    """
    gate_coverage = gate_coverage_matrix(lane_gate_records)

    metrics = {
        "portfolio_cagr": portfolio_cagr(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "turnover": turnover(equity_curve, trade_log),
        "per_lane_net_vs_gross": per_lane_net_vs_gross(trade_log),
        "fast_lane_break_even": fast_lane_break_even(trade_log),
        "cash_drag": cash_drag(equity_curve, exit_views),
    }

    benchmark_block = None
    if benchmark_equity_curve is not None:
        benchmark_block = {
            "equity_curve": benchmark_equity_curve,
            "trade_log": benchmark_trade_log or [],
            "metrics": {
                "aggregate": {
                    "portfolio_cagr": portfolio_cagr(benchmark_equity_curve),
                    "max_drawdown": max_drawdown(benchmark_equity_curve),
                },
                "per_lane": {
                    lane: {
                        "portfolio_cagr": portfolio_cagr(curve, value_key="positions_value"),
                        "max_drawdown": max_drawdown(curve, value_key="positions_value"),
                    }
                    for lane, curve in (benchmark_per_lane or {}).items()
                },
            },
            "n_buys_filled": sum(1 for r in (benchmark_trade_log or []) if r.get("filled")),
            "n_buys_refused": sum(1 for r in (benchmark_trade_log or []) if not r.get("filled")),
        }

    exclusions = describe_exclusions(
        universe_result=universe_result,
        checkpoint_excluded_transitions=checkpoint_excluded_transitions,
        gate_coverage_result=gate_coverage,
        equity_curve=equity_curve,
        reconciliation_failures=reconciliation_failures,
    )
    limitations = build_limitations(
        calendar_result=calendar_result, gate_coverage_result=gate_coverage, config=config,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "benchmark": benchmark_block,
        "metrics": metrics,
        "gate_coverage": gate_coverage,
        "exclusions": exclusions,
        "limitations": limitations,
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def render_summary(result: dict) -> str:
    """A thin, human-readable console rendering of `build_result`'s own
    dict — "the renderer for the CLI" U6's Approach names. Wiring this into
    an actual CLI command is U7's job; this function only needs to exist and
    be callable.
    """
    lines: list[str] = []
    metrics = result.get("metrics", {})
    cagr = metrics.get("portfolio_cagr", {})
    dd = metrics.get("max_drawdown", {})
    turn = metrics.get("turnover", {})

    lines.append("Strategy")
    lines.append(
        f"  CAGR: {_fmt_pct(cagr.get('cagr_pct'))} over {cagr.get('elapsed_days')} day(s) "
        f"({cagr.get('start_date')} -> {cagr.get('end_date')})"
    )
    lines.append(
        f"  Max drawdown: {_fmt_pct(dd.get('max_drawdown_pct'))} "
        f"(peak {dd.get('peak_date')} -> trough {dd.get('trough_date')})"
    )
    lines.append(f"  Turnover (annualized): {_fmt_pct(turn.get('turnover_ratio_annualized'))}")

    for lane, figs in sorted(metrics.get("per_lane_net_vs_gross", {}).get("by_lane", {}).items()):
        lines.append(
            f"  {lane}: gross {figs['gross']:.2f}, net {figs['net']:.2f} "
            f"over {figs['n_settlements']} settlement(s)"
        )

    breakeven = metrics.get("fast_lane_break_even", {})
    if breakeven.get("status") == "measured":
        lines.append(
            f"  Fast-lane break-even gap: {breakeven['break_even_gap_pct']:.2f}pp "
            f"({breakeven['central_tendency']} of {breakeven['n_cycles']} cycle(s))"
        )
    else:
        lines.append(f"  Fast-lane break-even: unmeasured — {breakeven.get('reason', '')}")

    idle = metrics.get("cash_drag", {}).get("idle_days", {})
    lines.append(
        f"  Cash drag (idle days): mean={idle.get('mean_idle_days')}, "
        f"median={idle.get('median_idle_days')} across {idle.get('n_exits', 0)} exit(s)"
    )

    benchmark = result.get("benchmark")
    lines.append("")
    if benchmark:
        b_cagr = benchmark["metrics"]["aggregate"].get("portfolio_cagr", {})
        b_dd = benchmark["metrics"]["aggregate"].get("max_drawdown", {})
        lines.append("Benchmark (equal-weight buy-and-hold, KTD9)")
        lines.append(f"  CAGR: {_fmt_pct(b_cagr.get('cagr_pct'))}")
        lines.append(f"  Max drawdown: {_fmt_pct(b_dd.get('max_drawdown_pct'))}")
    else:
        lines.append("Benchmark: not supplied to this result")

    gate_coverage = result.get("gate_coverage", {})
    lines.append("")
    lines.append(f"Gate coverage: {gate_coverage.get('status')}")
    if gate_coverage.get("status") != "measured":
        lines.append(f"  {gate_coverage.get('reason', '')}")
    else:
        windows = gate_coverage.get("windows", {})
        n_unmeasured = sum(1 for w in windows.values() if w.get("status") != "measured")
        lines.append(f"  {len(windows)} window(s), {n_unmeasured} unmeasured")

    lines.append("")
    lines.append("Exclusions")
    for item in result.get("exclusions", []):
        lines.append(f"  {item['category']}: {item['count']}")

    return "\n".join(lines)
