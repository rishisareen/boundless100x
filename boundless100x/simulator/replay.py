"""The replay loop's skeleton: raw `raw_data/` -> one simulated watchlist,
every KTD8-eligible ticker `add`ed at `screen` in the lane its own
first-eligible-date reading earns it.

**This is explicitly a skeleton (U2), not the loop (U7).** The plan's
High-Level Technical Design states the full six steps a replay date goes
through: truncate -> score -> evaluate lane gates and triggers -> hand
money-moving proposals to the simulated owner -> settle due confirmations
through the ledger -> mark to market. Only the first half of step 1
through the start of step 3 exists here — enough to get every eligible
ticker onto one watchlist, in a lane, at its own screen date — because U3
(the simulated owner), U4 (the ledger), U5 (cash-level friction) and U6
(the outputs) do not exist yet and this module must not guess their
shape. `build_initial_watchlist` is the seam U7 is expected to extend from.

**R2/KTD1 — scoring calls the production machinery, nothing reimplemented.**
`build_engines` wires `ComputeEngine`, `SQGLPScorer`, `EligibilityEvaluator`
and `LaneGateEvaluator` the way `Boundless100xService.__init__` does, minus
the fetch suite and the LLM orchestrator — the two pieces `service.analyze`
carries that a replay must never call (R9: no LLM stages; and a fetch would
reach the network per ticker per replay date, the exact look-ahead risk the
whole exercise exists to avoid). Scoring itself is `engine.run_all` on
truncated data, the backtest's own idiom.

**R10 — production stores are provably untouched.** `build_stores`
constructs `WatchlistManager` and `ReinvestmentQueue` on an explicit
`tempfile.TemporaryDirectory()` path every time; neither is ever
constructed with `path=None` anywhere in this module, which is what keeps
production's `boundless100x/watchlist.json`,
`boundless100x/lifecycle/reinvestment_queue.json` and
`boundless100x/score_history.jsonl` (score history is never even touched —
this module never calls `service.analyze` or `score_history.append_run`)
untouched by construction rather than by the test suite's autouse
monkeypatch, which a production caller would not have.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.eligibility import EligibilityEvaluator, effective_gates
from boundless100x.compute_engine.point_in_time import NON_TRUNCATABLE_INPUTS, truncate_to_date
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.lifecycle.lane_gates import LaneGateEvaluator
from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import universe as universe_module
from boundless100x.watchlist import CORE_LANE, RERATING_LANE, WatchlistManager

logger = logging.getLogger(__name__)

# The fast lane's five gates a candidate can clear on its own, at a date
# with no watchlist entry and therefore no catalyst. The sixth,
# `catalyst_identified`, is deliberately excluded: KTD6 fabricates a
# synthetic catalyst as a simulated-owner policy once a candidate has
# cleared these five — that is U3's job, not this module's. Named
# explicitly (rather than "every gate but one") so a future edit to
# `lane_gates.yaml` cannot silently widen or narrow what this skeleton
# checks without also touching this list.
FAST_LANE_ENTRY_GATES = (
    "quality_floor",
    "valuation_discount",
    "growth_intact",
    "institutional_accumulation",
    "liquidity_floor",
)

# The simulator needs `Market Cap` and `Stock P/E` to evaluate the
# eligibility gates and lane gates it replays (KTD0) — unlike the backtest,
# which deliberately omits both. `shareholding` is dropped from the strip
# list for the same reason `calendar.py`'s battery-complete search drops
# it: the lane gates need a truncated shareholding view and
# `NON_TRUNCATABLE_INPUTS`'s default would otherwise remove it.
_SIMULATOR_NON_TRUNCATABLE_INPUTS = tuple(
    x for x in NON_TRUNCATABLE_INPUTS if x != "shareholding"
)


# ── production machinery, wired without the fetch/LLM stages (R2, R9) ───


@dataclass
class SimulatorEngines:
    """The production compute objects the replay scores through. Never
    `service.analyze()` (R2) — that fetches, calls the LLM path (R9), and
    appends to the real `score_history.jsonl`, none of which belong in a
    replay of already-fetched, truncated history.
    """
    engine: ComputeEngine
    scorer: SQGLPScorer
    eligibility: EligibilityEvaluator
    lane_gates: LaneGateEvaluator


def build_engines(config: dict | None = None) -> SimulatorEngines:
    """Wire `ComputeEngine`/`SQGLPScorer`/`EligibilityEvaluator`/
    `LaneGateEvaluator` the way `Boundless100xService.__init__` wires the
    first three, plus the lane-gate evaluator `advance()` also constructs.
    `LaneGateEvaluator(known_metric_ids=...)` is required, not optional —
    without it, a lane gate naming a metric the loaded registry does not
    know about would read indeterminate forever instead of failing at
    construction (CLAUDE.md, "Two lanes").
    """
    config = config or {}
    engine = ComputeEngine(macro=config.get("macro", {}))
    eligibility = EligibilityEvaluator(effective_gates(engine.gates))
    scorer = SQGLPScorer(
        engine.metrics,
        engine.element_weights,
        history_waiver_mcap=engine.master.get("history_waiver_mcap"),
    )
    lane_gates = LaneGateEvaluator(known_metric_ids=set(engine.metrics))
    return SimulatorEngines(
        engine=engine, scorer=scorer, eligibility=eligibility, lane_gates=lane_gates,
    )


# ── temp-dir production stores (R10) ─────────────────────────────────────


@dataclass
class SimulatedStores:
    """Owns the `tempfile.TemporaryDirectory` the simulated watchlist and
    reinvestment queue live in.

    The directory object is kept on the instance deliberately: letting it
    fall out of scope (or go uncollected in a way the GC does not run
    promptly) would delete the backing files out from under a still-live
    `WatchlistManager`. Call `close()` once the caller is done with both
    stores.
    """
    tmpdir: tempfile.TemporaryDirectory
    watchlist: WatchlistManager
    queue: ReinvestmentQueue

    def close(self) -> None:
        self.tmpdir.cleanup()


def build_stores() -> SimulatedStores:
    """`WatchlistManager` and `ReinvestmentQueue`, each pointed at an
    explicit path inside a fresh `tempfile.TemporaryDirectory()` — never
    constructed with `path=None`, which is what would reach
    `DEFAULT_WATCHLIST_PATH` / `DEFAULT_QUEUE_PATH` and therefore
    production's real files.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix="boundless100x-simulator-")
    root = Path(tmpdir.name)
    watchlist = WatchlistManager(str(root / "watchlist.json"))
    queue = ReinvestmentQueue(str(root / "reinvestment_queue.json"))
    return SimulatedStores(tmpdir=tmpdir, watchlist=watchlist, queue=queue)


# ── scoring + lane assignment at a ticker's own first-eligible date ─────


def score_ticker_at(
    data: dict, cutoff: pd.Timestamp, engines: SimulatorEngines,
) -> tuple[dict, dict]:
    """Truncate at `cutoff` and score exactly as the backtest's own idiom
    does (R2): `engine.run_all` on truncated data, never `service.analyze`.

    `rebuild_valuation=True` — unlike the backtest, the simulator needs
    `Market Cap` and `Stock P/E` to evaluate the eligibility gates and lane
    gates it replays (KTD0). Raises if `cutoff` cannot be truncated at all
    (the price series starts after it); a candidate's own KTD8
    first-eligible date is derived from a cutoff `truncate_to_date` already
    accepted, so this should not happen for a ticker `universe.py` found
    eligible — if it does, that is a real inconsistency worth surfacing
    rather than swallowing.
    """
    truncated, reason = truncate_to_date(
        data, cutoff,
        rebuild_valuation=True,
        non_truncatable_inputs=_SIMULATOR_NON_TRUNCATABLE_INPUTS,
    )
    if truncated is None:
        raise ValueError(f"cannot truncate at {cutoff.date()}: {reason}")
    metrics = engines.engine.run_all(truncated)
    scores = engines.scorer.score(metrics)
    return metrics, scores


@dataclass
class LaneAssignment:
    """Which lane a candidate entered, and the evidence — kept per ticker
    so U6's coverage matrix can attribute a lane to the gates that decided
    it rather than re-deriving the same read from a bare lane label.
    """
    ticker: str
    lane: str
    entry_date: pd.Timestamp
    gate_result: dict | None
    deciding_gates: dict[str, bool | None]
    error: str | None = None


def assign_lane(
    metrics: dict, scores: dict, engines: SimulatorEngines,
) -> tuple[str, dict, dict[str, bool | None]]:
    """KTD6/U2's lane rule: the fast lane's five computable gates, catalyst
    excluded.

    `catalyst=None` makes `LaneGateEvaluator._evaluate_catalyst` read
    `catalyst_identified` indeterminate rather than failed — correct for a
    ticker with no watchlist entry yet — but that also pins the evaluator's
    own aggregate `verdict` at `INDETERMINATE` forever in this call, never
    `QUALIFIES`. So this function reads each of the other five gates' own
    `passed` value directly out of `result["gates"]` rather than trusting
    the aggregate, exactly as the module doc for `LaneGateEvaluator.evaluate`
    says a caller in this position should.
    """
    result = engines.lane_gates.evaluate(metrics, scores, catalyst=None)
    deciding = {
        gate_id: result["gates"][gate_id]["passed"] for gate_id in FAST_LANE_ENTRY_GATES
    }
    lane = RERATING_LANE if all(v is True for v in deciding.values()) else CORE_LANE
    return lane, result, deciding


# ── the skeleton's one entry point ───────────────────────────────────────


def build_initial_watchlist(
    raw_data_dir: str | Path,
    config: dict | None = None,
    *,
    calendar_result: calendar_module.ReplayCalendar | None = None,
    engines: SimulatorEngines | None = None,
) -> tuple[
    SimulatedStores,
    calendar_module.ReplayCalendar,
    universe_module.UniverseResult,
    dict[str, LaneAssignment],
]:
    """Raw `raw_data/` -> one simulated watchlist, every KTD8-eligible
    ticker `add`ed at `screen` in its first-eligible-date lane.

    This is where U2 stops. **U7** takes the returned stores, calendar and
    universe and drives the six-step loop from here forward (`decide()`,
    `TriggerEvaluator`, `LaneGateEvaluator` again per subsequent replay
    date, the simulated owner, the ledger, mark-to-market). Nothing in this
    function proposes or applies a lifecycle *transition* — `add()` is the
    only watchlist write it performs, and every entry lands at `screen`,
    which is earned by nothing but existing.

    Args:
        raw_data_dir: the fetched corpus root (`data_fetcher/raw_data/` in
            production; a synthetic fixture directory in tests).
        config: forwarded to `build_engines` for `macro:` only, unless
            `engines` is supplied directly.
        calendar_result: reuse an already-computed calendar (e.g. across a
            `--tickers` subset iteration) instead of recomputing it.
        engines: reuse already-built production objects instead of
            constructing new ones — the same object across many calls is
            safe, since none of `ComputeEngine`/`SQGLPScorer`/
            `EligibilityEvaluator`/`LaneGateEvaluator` hold per-ticker state.

    Returns `(stores, calendar_result, universe_result, assignments)`:
        stores: the two temp-dir production stores, holding exactly the
            `add()`s below. Call `stores.close()` when done.
        calendar_result: `calendar.compute_calendar`'s return (or the
            caller's own, echoed back).
        universe_result: `universe.build_universe`'s return — every
            eligible ticker's first-eligible date and every exclusion's
            reason, named per KTD8.
        assignments: `{ticker: LaneAssignment}` for every entered ticker.
    """
    engines = engines or build_engines(config)
    calendar_result = calendar_result or calendar_module.compute_calendar(raw_data_dir)
    universe_result = universe_module.build_universe(raw_data_dir, calendar_result.dates)

    stores = build_stores()
    assignments: dict[str, LaneAssignment] = {}

    for ticker, first_date in sorted(universe_result.eligible.items()):
        ticker_dir = universe_result.ticker_dirs[ticker]
        lane, gate_result, deciding, error = CORE_LANE, None, {}, None
        try:
            data = universe_module.load_ticker_data(ticker_dir)
            metrics, scores = score_ticker_at(data, first_date, engines)
            lane, gate_result, deciding = assign_lane(metrics, scores, engines)
        except Exception as exc:  # noqa: BLE001 — recorded on the assignment, not swallowed
            # A candidate whose lane gates could not be evaluated has not
            # proven it clears the fast lane's bar, so it defaults to
            # `core` — the same "indeterminate is not a pass" discipline
            # every evaluator in this system already applies, just applied
            # here to the lane decision itself rather than to one gate.
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"{ticker}: could not evaluate lane gates at {first_date.date()}, "
                f"defaulting to {CORE_LANE}: {error}"
            )

        stores.watchlist.add(ticker, lane=lane)
        assignments[ticker] = LaneAssignment(
            ticker=ticker, lane=lane, entry_date=first_date,
            gate_result=gate_result, deciding_gates=deciding, error=error,
        )

    return stores, calendar_result, universe_result, assignments
