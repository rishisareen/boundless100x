"""Phase 4 strategy simulator — a replay of the production lifecycle over
truncated historical data (`docs/plans/2026-08-07-007-feat-phase4-strategy-simulator-plan.md`).

The central design constraint (per the plan's Requirements, R2/R10) is that
the replay calls the production evaluators — `ComputeEngine.run_all`,
`SQGLPScorer`, `EligibilityEvaluator`, `TriggerEvaluator`, `LaneGateEvaluator`
— on point-in-time-truncated data, rather than a second statement of the
same rules. A simulator that reimplemented the lifecycle would prove
something about *that* reimplementation, not about the shipped one.

Submodules, in the order the replay loop consumes them:

  * `calendar.py`  — replay dates from the corpus's own fiscal calendar
                      (KTD7), plus the per-lane battery-complete reading.
  * `universe.py`   — `raw_data/` discovery and per-ticker candidacy under
                      KTD8 (first replay date a ticker's truncated financials
                      clear the engine's minimum-years bar).
  * `replay.py`     — the loop skeleton: temp-dir production stores,
                      `add`-and-lane-assign every eligible ticker at
                      `screen`. The full six-step loop (truncate -> score ->
                      evaluate -> propose -> confirm -> settle -> mark to
                      market) is U7; this module stops at "a watchlist
                      populated with lane-assigned candidates at their
                      screen dates."
  * `owner.py`      — the simulated-owner policy (U3, KTD3/KTD6): pure
                      functions deciding when an already-produced
                      `advance.decide()` proposal is confirmed, whether a
                      fast-lane candidate earns a fabricated catalyst, and
                      when a routed exit's proceeds are accepted. Never
                      calls `TriggerEvaluator`/`LaneGateEvaluator`/
                      `advance.decide()` itself — see its own module
                      docstring for why that boundary is load-bearing.
  * `ledger.py`     — modeled cash, per-tranche lots, mark-to-market (U4,
                      KTD4). The one exception to production's "no rupees"
                      boundary — a real cash pool, entirely inside this
                      package.
  * `friction_cash.py` — the `friction:` regime (STCG/LTCG, slippage)
                      applied to traded notional rather than to a return
                      percentage (U5, KTD5).
  * `replay.py`     — `run_replay(...)`, the full six-step loop: truncate,
                      score, evaluate (lane gates + triggers), propose to
                      the simulated owner, settle due confirmations through
                      the ledger, mark to market (U7).
  * `outputs.py`    — the six §10 readings, the KTD9 benchmark, gate
                      coverage, exclusions and limitations, assembled from
                      already-produced replay data (U6).

`simulate(config, overrides) -> dict` below is R10's one importable entry
point — no subprocess, no CLI dependency — a Phase 5 sweep loops over it
directly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.suite import DataFetcherSuite
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import outputs as outputs_module
from boundless100x.simulator import replay as replay_module
from boundless100x.simulator import universe as universe_module
from boundless100x.watchlist import CORE_LANE, RERATING_LANE


def _apply_override(config: dict, dotted_key: str, value) -> None:
    """Set `config[a][b]...[z] = value` for a dot-path key, creating any
    missing intermediate dict along the way.

    The Phase 5 sweep seam (R10): a caller varies one setting —
    `"simulator.confirmation_lag_days.entry"` — without hand-editing
    `config.yaml` or touching the filesystem. A path segment that already
    exists but is not itself a dict (a config bug, or a caller's typo one
    level too shallow) is overwritten rather than raised through — the same
    "fail loud at the door, not three frames later" discipline the rest of
    this layer applies, chosen here as "the override wins" since the whole
    point of the seam is that the caller's value is the one that should
    stick.
    """
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def simulate(
    config: dict | None = None,
    overrides: dict | None = None,
    *,
    tickers: list[str] | None = None,
    start=None,
    end=None,
    raw_data_dir: str | Path | None = None,
) -> dict:
    """R10's callable seam: build the whole replay and run it, in-process,
    no subprocess and no CLI dependency — the entry point a Phase 5 sweep
    loops over directly.

    `config` is an already-loaded config dict, or `None` to load the
    shipped `boundless100x/config.yaml` (`service.load_config`). Either way
    it is deep-copied before anything is applied to it, so a caller's own
    dict (or the module-level default) is never mutated by a run.

    `overrides` is a flat `{"dotted.path.to.key": value}` mapping applied on
    top of the loaded config via `_apply_override` — e.g.
    `{"simulator.confirmation_lag_days.entry": 0,
    "portfolio.max_positioned_per_lane.core": 4}`. This is the one seam
    Phase 5 needs: every reading in the returned result is then a function
    of `overrides` alone, holding the rest of the config fixed.

    `tickers` restricts the simulated watchlist to a named subset — every
    other corpus ticker is excluded from `universe_result.eligible` with an
    explicit reason (the same exclusion vocabulary `universe.build_universe`
    already uses, not a new one) — but **not** from the deployment-pace
    corpus, which production's own §11 design reads across the *whole*
    cached corpus regardless of watchlist membership; narrowing that too
    would make `--tickers` quietly change a macro reading it has no business
    touching.

    `start`/`end` trim `calendar_result.dates` to a sub-window of the
    corpus's own fiscal grid (`calendar.compute_calendar` always computes
    the full grid first; this slices the list afterward rather than
    building a second one) — accepts anything `pandas.Timestamp` accepts.

    `raw_data_dir` defaults to the same resolution the CLI already uses
    (`DataFetcherSuite(config).raw_data_dir` — see `cli.py::_raw_data_dir`),
    overridable directly for a test pointed at a synthetic fixture directory.

    **The KTD9 benchmark is built here, after `run_replay` returns, never
    inside it** — `outputs.build_benchmark_curve`/`outputs.lane_position_value_
    curve` need the same per-ticker price frames `run_replay` already loaded
    (its own returned `price_frames`), the universe's own first-eligible
    dates, and the calendar's own dates; nothing about the benchmark depends
    on what the strategy actually did, so it is computed independently
    rather than threaded through the six-step loop. `ticker_lanes` is read
    off `assignments` so the benchmark's own trade log can be split "per
    lane as well as in aggregate" exactly as R7 requires.

    Returns `run_replay`'s own ingredients assembled through
    `outputs.build_result` (equity curve, trade log, benchmark, the six §10
    metrics, gate coverage, exclusions, limitations), plus what only this
    function and `run_replay` know and `build_result` carries no slot for:
    `owner_decisions` (KTD3 — every simulated-owner policy decision this run
    actually made), `errors` (per-ticker-per-date failures, isolated exactly
    as `advance()` isolates them), `unsettled_confirmations` (scheduled but
    never due before the window ended), and `config` (the fully-resolved
    config this run actually used, overrides applied, so the result can be
    re-derived from its own record).
    """
    from boundless100x.service import load_config

    config = copy.deepcopy(config) if config is not None else load_config()
    for dotted_key, value in (overrides or {}).items():
        _apply_override(config, dotted_key, value)

    if raw_data_dir is None:
        raw_data_dir = DataFetcherSuite(config).raw_data_dir
    raw_data_dir = str(raw_data_dir)

    engines = replay_module.build_engines(config)
    calendar_result = calendar_module.compute_calendar(raw_data_dir)

    if start is not None or end is not None:
        window_start = pd.Timestamp(start) if start is not None else calendar_result.start
        window_end = pd.Timestamp(end) if end is not None else calendar_result.end
        calendar_result.dates = [
            d for d in calendar_result.dates if window_start <= d <= window_end
        ]

    universe_result = universe_module.build_universe(raw_data_dir, calendar_result.dates)

    if tickers:
        wanted = {t.upper() for t in tickers}
        for ticker in list(universe_result.eligible):
            if ticker not in wanted:
                del universe_result.eligible[ticker]
                universe_result.excluded[ticker] = (
                    "excluded by simulate()'s own tickers= filter, not a KTD8 "
                    "candidacy finding"
                )

    assignments = replay_module.compute_lane_assignments(universe_result, engines)
    stores = replay_module.build_stores()
    try:
        raw = replay_module.run_replay(
            stores, calendar_result, universe_result, assignments, engines, config,
        )
    finally:
        stores.close()

    # KTD9's benchmark, independent of the strategy replay above — it needs
    # only the universe's own first-eligible dates, the replay calendar, and
    # the same price frames `run_replay` already loaded and returned rather
    # than a second read of every ticker's history off disk.
    benchmark_equity_curve, benchmark_trade_log = outputs_module.build_benchmark_curve(
        universe_result.eligible,
        calendar_result.dates,
        raw["price_frames"],
        config,
        ticker_lanes={ticker: a.lane for ticker, a in assignments.items()},
    )
    benchmark_per_lane = {
        lane: outputs_module.lane_position_value_curve(
            benchmark_trade_log, raw["price_frames"], calendar_result.dates, lane,
        )
        for lane in (CORE_LANE, RERATING_LANE)
    }

    result = outputs_module.build_result(
        equity_curve=raw["equity_curve"],
        trade_log=raw["trade_log"],
        benchmark_equity_curve=benchmark_equity_curve,
        benchmark_trade_log=benchmark_trade_log,
        benchmark_per_lane=benchmark_per_lane,
        lane_gate_records=raw["lane_gate_records"],
        exit_views=raw["exit_views"],
        universe_result=universe_result,
        calendar_result=calendar_result,
        checkpoint_excluded_transitions=raw["checkpoint_excluded_transitions"],
        reconciliation_failures=raw["reconciliation_failures"],
        config=config,
    )
    result["owner_decisions"] = raw["owner_decisions"]
    result["errors"] = raw["errors"]
    result["unsettled_confirmations"] = raw["unsettled_confirmations"]
    result["config"] = config
    return result
