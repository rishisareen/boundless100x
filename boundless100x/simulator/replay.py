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
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd

from boundless100x import price_bars
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.eligibility import EligibilityEvaluator, effective_gates
from boundless100x.compute_engine.metrics.builtin.valuation import compute_earnings_yield_spread
from boundless100x.compute_engine.point_in_time import NON_TRUNCATABLE_INPUTS, truncate_to_date
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.lifecycle import advance as advance_module
from boundless100x.lifecycle import exit as exit_module
from boundless100x.lifecycle import pace as pace_module
from boundless100x.lifecycle import portfolio as portfolio_module
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.lifecycle.lane_gates import LaneGateEvaluator
from boundless100x.lifecycle.reinvestment import CANDIDATE_STATES, ReinvestmentQueue
from boundless100x.lifecycle.states import (
    APPLIED_AUTO,
    APPLIED_OWNER,
    EXIT_REVIEW,
    PROBE,
    as_date,
)
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import universe as universe_module
from boundless100x.watchlist import CORE_LANE, RERATING_LANE, WatchlistManager

logger = logging.getLogger(__name__)

# `owner.py` and `ledger.py` both import FROM this module (`owner.py` reaches
# back for `FAST_LANE_ENTRY_GATES`; `ledger.py` imports `owner.py`), so a
# top-level `import` of either here would be circular — Python would be
# mid-way through initializing this module (before `FAST_LANE_ENTRY_GATES`
# below is even defined) when `owner.py`'s own top-level import tried to read
# it back. `run_replay` imports both lazily, inside its own body, exactly the
# way `compute_engine/point_in_time.py::_rebuild_stock_pe` lazily imports
# `valuation.py` for the identical shape of reason (see that function's own
# docstring). Safe as a function-local import: by the time `run_replay` is
# ever *called*, both modules have already finished loading.

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


def truncate_ticker_at(data: dict, cutoff: pd.Timestamp) -> dict:
    """The truncated, point-in-time view `score_ticker_at` scores from —
    pulled out on its own because **U7's replay loop needs the truncated
    view itself, not just what `engine.run_all` derives from it.**
    `lifecycle.advance.decide()`'s own `data` argument (the price series and
    sector `_sector_of`/`_friction_for_exit` read out of it), the
    deployment-pace modulator's per-ticker metadata (`_corpus_spread_at`
    below), and the KTD0 reconciliation-exclusion bookkeeping all need this
    same truncated frame — re-truncating separately for each would be three
    silently-driftable copies of one another rather than one.

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
    return truncated


def score_ticker_at(
    data: dict, cutoff: pd.Timestamp, engines: SimulatorEngines,
) -> tuple[dict, dict]:
    """Truncate at `cutoff` and score exactly as the backtest's own idiom
    does (R2): `engine.run_all` on truncated data, never `service.analyze`.
    Delegates the truncation itself to `truncate_ticker_at` (see its own
    docstring for why that is now a function of its own).
    """
    truncated = truncate_ticker_at(data, cutoff)
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
    assignments = compute_lane_assignments(universe_result, engines)
    for ticker in sorted(assignments):
        stores.watchlist.add(ticker, lane=assignments[ticker].lane)

    return stores, calendar_result, universe_result, assignments


def _assign_lane_for_ticker(
    ticker: str, ticker_dir: Path, first_date: pd.Timestamp, engines: SimulatorEngines,
) -> LaneAssignment:
    """One ticker's `LaneAssignment`, scored at `first_date`. Factored out of
    `build_initial_watchlist` so it and `compute_lane_assignments` (U7) share
    the identical scoring logic rather than becoming two statements of "how
    is a candidate's lane decided."
    """
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
    return LaneAssignment(
        ticker=ticker, lane=lane, entry_date=first_date,
        gate_result=gate_result, deciding_gates=deciding, error=error,
    )


def compute_lane_assignments(
    universe_result: universe_module.UniverseResult, engines: SimulatorEngines,
) -> dict[str, LaneAssignment]:
    """Every eligible ticker's `LaneAssignment`, scored at its own
    first-eligible date — the same computation `build_initial_watchlist`
    performs, but **without writing to a watchlist**.

    **U7's per-date loop needs this split.** `build_initial_watchlist` adds
    every eligible ticker to a watchlist in one eager pass at construction
    time, which is exactly what the plan's own per-date requirement forbids:
    a ticker must not become visible to the concentration gate (or be
    evaluated by `decide()` at all) before the replay loop actually reaches
    its own first-eligible date. `run_replay` therefore does not call
    `build_initial_watchlist` — it calls this function once, up front, to
    get every candidate's lane (reusing the identical scoring this always
    did), and calls `watchlist.add()` itself only once its own date loop
    reaches each ticker's `entry_date`. `build_initial_watchlist` is kept
    exactly as it was — its own existing callers and tests still get the
    eager, single-pass watchlist it always built — this function only lifts
    the *scoring* half out so the two no longer duplicate it.
    """
    return {
        ticker: _assign_lane_for_ticker(
            ticker, universe_result.ticker_dirs[ticker], first_date, engines,
        )
        for ticker, first_date in sorted(universe_result.eligible.items())
    }


# ── U7: the deployment-pace modulator, read point-in-time ───────────────
#
# `lifecycle/pace.py::corpus_spread` reads `metadata.json`/`financials.csv`
# straight off disk — exactly the live, unrebuilt, un-truncated read a
# replay must never make (a pace reading built off *today's* fetch would
# leak into a replay date years earlier, the Goal Capsule's stop condition
# (d)). This reproduces its shape (`{median_pp, contributors, tickers,
# values}`) by calling the SAME production metric function
# (`compute_earnings_yield_spread`, `pace.py`'s own import) against each
# corpus ticker's own point-in-time-truncated metadata instead — KTD1
# forbids reimplementing a *rule*, not calling a metric function with
# different, correctly point-in-time inputs. This is the "new work" the
# dispatch calls out: production reads its corpus median once per *run*;
# the replay must read it once per *date*, off that date's own truncated
# corpus, which nothing before U7 needed to do.


def _corpus_spread_at(tickers, truncate_fn, cutoff: pd.Timestamp, macro: dict) -> dict:
    """`pace.corpus_spread`'s own reading, at one replay date.

    `truncate_fn(ticker, cutoff) -> dict` is the caller's own (cached)
    truncation. A ticker that cannot be truncated at `cutoff` (its price
    series starts later, or it is simply unreadable) contributes nothing —
    mirroring `corpus_spread`'s own per-directory try/except, so one bad
    corpus member never costs the whole date's pace reading.
    """
    readings: list[tuple[str, float]] = []
    for ticker in sorted(tickers):
        try:
            truncated = truncate_fn(ticker, cutoff)
        except Exception as exc:  # noqa: BLE001 — one contributor lost, not the reading
            logger.debug(f"pace: {ticker} could not be truncated at {cutoff.date()}: {exc}")
            continue
        try:
            result = compute_earnings_yield_spread(truncated, dict(macro or {}))
        except Exception as exc:  # noqa: BLE001 — same as above
            logger.debug(f"pace: {ticker} earnings-yield spread failed: {exc}")
            continue
        if result.ok and isinstance(result.value, (int, float)) and math.isfinite(result.value):
            readings.append((ticker, float(result.value)))

    readings.sort()
    values = [value for _, value in readings]
    return {
        "median_pp": round(median(values), 2) if values else None,
        "contributors": len(values),
        "tickers": [name for name, _ in readings],
        "values": [round(v, 2) for v in values],
    }


# ── U7: the two-attribute stand-in `confirm_exit` needs ─────────────────
#
# `lifecycle/exit.py::confirm_exit`'s one dependency on a live system is
# `_friction_for_confirmed_exit`, which reads `service.suite.price_volume.
# fetch(ticker, years, output_dir)` and `service.config`. Read against the
# real function before building this (per the dispatch's own instruction):
# it touches exactly `suite.price_volume` (called), `suite.price_years` and
# `suite.raw_data_dir` (read but never used by anything this shim's `fetch`
# does — see `_ReplaySuite` below) and `service.config`. Nothing else of
# `Boundless100xService` is ever reached, so a two-class stand-in is
# genuinely sufficient rather than a guess at the shape.


class _ReplayPriceFetch:
    """A `.fetch(ticker, years, output_dir)`-shaped stand-in for
    `PriceVolumeFetcher`. Always returns the frame it was built with —
    already truncated to the confirmation date by the caller — and ignores
    every argument, because a live fetch here would be exactly the
    look-ahead leak R2/R9 forbid: `confirm_exit`'s own friction reading must
    never be able to see a bar past the moment being replayed.
    """

    def __init__(self, price_df: "pd.DataFrame | None"):
        self._price_df = price_df

    def fetch(self, ticker: str, years, output_dir):
        return self._price_df


class _ReplaySuite:
    def __init__(self, price_df: "pd.DataFrame | None"):
        self.price_volume = _ReplayPriceFetch(price_df)
        # `_friction_for_confirmed_exit` reads these two attributes off
        # `service.suite` *before* calling `.fetch(...)` — never consumed by
        # `_ReplayPriceFetch.fetch` itself (which ignores its own `years`/
        # `output_dir` arguments), but the attributes must exist or the
        # lookup raises before `fetch` is ever reached.
        self.price_years = 10
        self.raw_data_dir = ""


class _ReplayService:
    """The minimal `service`-shaped object `lifecycle.exit.confirm_exit`
    needs: `.suite.price_volume.fetch(...)` and `.config`. Built fresh per
    settlement with that settlement's own already-truncated price frame —
    never the live `Boundless100xService`.
    """

    def __init__(self, price_df: "pd.DataFrame | None", config: dict | None):
        self.suite = _ReplaySuite(price_df)
        self.config = config or {}


# ── U7: the replay loop ───────────────────────────────────────────────────


def run_replay(
    stores: SimulatedStores,
    calendar_result: calendar_module.ReplayCalendar,
    universe_result: universe_module.UniverseResult,
    assignments: dict[str, LaneAssignment],
    engines: SimulatorEngines,
    config: dict | None = None,
) -> dict:
    """The six-step loop (High-Level Technical Design), once per
    `calendar_result.dates` entry, over every ticker on `stores.watchlist` —
    including tickers `universe_result.eligible` names but that have not yet
    joined it, which this function `add()`s itself the moment the loop
    reaches their own `entry_date` (see `compute_lane_assignments`'s own
    docstring for why the eager, U2-era `build_initial_watchlist` is not
    reused here).

    Per date: truncate + score every active ticker (`truncate_ticker_at` +
    `engine.run_all` + `scorer.score`, cached by `(ticker, cutoff,
    registry_hash)`), resolve the deployment-pace modulator off that date's
    own truncated corpus (`_corpus_spread_at`), call
    `lifecycle.advance.decide()` once per ticker (always `apply=False`),
    capture the lane-gate coverage record, hand any money-moving proposal to
    `simulator.owner.decide()`, settle whatever is now due (entries via
    `Ledger.buy` + a `probe`/`scale` transition; exits via
    `lifecycle.exit.confirm_exit` followed by an independent `Ledger.sell`,
    per R5/KTD5 — the two are deliberately separate models, never
    reconciled; reinvestment routing via `Ledger.buy` into the router's
    top-ranked candidate plus `ReinvestmentQueue.record_routing`), fabricate
    a fast-lane catalyst where KTD6 earns one, then mark to market.

    A per-ticker failure (truncation, scoring, `decide()` raising) is caught,
    logged and recorded in the returned `errors` list — mirroring
    `advance()`'s own per-ticker isolation (CLAUDE.md, "Lifecycle"): a stale
    or broken ticker is no reason to stop advancing every other one. A
    settlement failure is isolated the same way.

    Returns a plain dict:

        {"equity_curve": [...], "trade_log": [...],
         "lane_gate_records": [...], "exit_views": [...],
         "reconciliation_failures": [...],
         "checkpoint_excluded_transitions": int,
         "owner_decisions": [...], "errors": [...],
         "unsettled_confirmations": [...], "price_frames": {...}}

    The first five are exactly `outputs.build_result`'s own input contracts
    (equity curve, trade log, lane-gate coverage records, exit views,
    reconciliation failures, checkpoint exclusions — see `outputs.py`'s
    module docstring). `owner_decisions` and `errors` are this loop's own —
    `build_result` carries no slot for either, since nothing before this
    function ever produced a decision log or a per-date error list.
    `unsettled_confirmations` is whatever was scheduled but never became due
    before the replay window ended (diagnostic only). `price_frames` is the
    `{ticker: DataFrame}` map this loop already built and used for every
    settlement bar and mark-to-market call — returned rather than silently
    kept local so `simulate()` can hand the identical frames to
    `outputs.build_benchmark_curve`/`outputs.lane_position_value_curve`
    without loading every ticker's price history from disk a second time
    (KTD9's benchmark needs the same bars the strategy was marked on).

    `outputs.build_result` is deliberately **not** called here.
    `outputs.py` (like `ledger.py`) imports `simulator.owner`, which imports
    *this* module for `FAST_LANE_ENTRY_GATES` — assembling the final
    artifact is `simulate()`'s job (`simulator/__init__.py`), a module this
    one is never imported by, so it can safely import everything.
    """
    # Lazy — see the module-level comment above `logger = ...` for why.
    from boundless100x.simulator import owner as owner_module
    from boundless100x.simulator.ledger import Ledger

    config = config or {}
    watchlist = stores.watchlist
    queue = stores.queue

    owner_settings = owner_module.config_from(config)
    override_caps = owner_module.override_caps_for(owner_settings["cap_posture"])

    # Preloaded once — reused for every truncation (scoring, pace) and as
    # the raw price frames `Ledger.mark_to_market`/the settlement bar
    # lookups need. `universe_result.ticker_dirs` is every corpus ticker
    # that could be read at all (KTD8), not just the eligible ones: the
    # deployment-pace median is read across the whole cached corpus (§11),
    # independent of watchlist membership, exactly as production reads it.
    raw_by_ticker = {
        ticker: universe_module.load_ticker_data(ticker_dir)
        for ticker, ticker_dir in universe_result.ticker_dirs.items()
    }
    price_frames = {ticker: data.get("price") for ticker, data in raw_by_ticker.items()}

    ledger = Ledger(config=config)
    trigger_registry = load_triggers()
    known_metric_ids = set(engines.engine.metrics)

    equity_curve: list[dict] = []
    trade_log: list[dict] = []
    lane_gate_records: list[dict] = []
    reconciliation_failures: list[dict] = []
    owner_decisions: list[dict] = []
    errors: list[dict] = []
    checkpoints_evaluated = 0
    # Sector persists across dates once read — a ticker not scored on a
    # given date (nothing forces re-scoring here, but a settlement can touch
    # a ticker on a later date than it was last advanced) still needs the
    # last sector reading the concentration gate saw for it.
    ticker_sectors: dict[str, str | None] = {}
    scheduled: list[dict] = []

    # (ticker, cutoff-iso, registry-hash) -> truncated data. The registry
    # hash rides along defensively — it cannot change mid-run — per the
    # plan's own Approach for U7 ("per-date score caching keyed on (ticker,
    # cutoff, registry hash)"). Shared between scoring and the pace reading:
    # nearly every watchlist ticker is also a pace-corpus ticker, so this
    # cache is what keeps the two from truncating the same (ticker, date)
    # pair twice.
    truncated_cache: dict[tuple, dict] = {}

    def truncate_cached(ticker: str, cutoff: pd.Timestamp) -> dict:
        key = (ticker, cutoff.isoformat(), engines.engine.registry_hash)
        if key not in truncated_cache:
            truncated_cache[key] = truncate_ticker_at(raw_by_ticker[ticker], cutoff)
        return truncated_cache[key]

    def collect_reconciliation(ticker: str, cutoff: pd.Timestamp, truncated: dict) -> None:
        meta = truncated.get("metadata") or {}
        for key in ("_market_cap_exclusion", "_stock_pe_exclusion"):
            exclusion = meta.get(key)
            if exclusion:
                reconciliation_failures.append({
                    "ticker": ticker, "date": str(cutoff.date()),
                    "code": exclusion.get("code"), "detail": exclusion.get("detail"),
                })

    def watchlist_rows() -> list[dict]:
        return [
            {
                "ticker": t,
                "lane": (watchlist.get(t) or {}).get("lane"),
                "state": (watchlist.get(t) or {}).get("state"),
                "sector": ticker_sectors.get(t),
            }
            for t in watchlist.tickers()
        ]

    def concentration_reading() -> dict:
        return portfolio_module.check_concentration(watchlist_rows(), config)

    def concentration_gate(lane: str, sector) -> list[str]:
        return portfolio_module.would_breach(lane, sector, concentration_reading())

    def resolved_bar(ticker: str, at_date) -> dict | None:
        return price_bars.bar_on_or_before(price_frames.get(ticker), at_date)

    # ── settlement: probe/scale entry ──

    def settle_entry(item: dict, at_date) -> None:
        ticker, proposal, lane = item["ticker"], item["proposal"], item["lane"]
        entry = watchlist.get(ticker)
        if entry is None or entry["state"] != item["from_state"]:
            # Overtaken by a different transition since the proposal was
            # scheduled — a pre-position trigger such as
            # `fundamentals_deteriorated` auto-applies on any later date, so
            # the `watch` this buy zone was read against may no longer be
            # the company's state. Nothing is bought against a stale read.
            return

        # The cap is checked HERE, at settlement, not at proposal time —
        # `decide()`'s own `concentration_gate` call (inside the per-ticker
        # loop, days before this runs) reads occupancy as it stood then, and
        # a second proposal scheduled the same date would have read the
        # identical stale, pre-either-settlement count. Re-checked live,
        # immediately before the write, so a sibling settlement processed
        # earlier in this same `due` batch is already visible here — the
        # production rule CLAUDE.md states ("a cap is checked before the
        # transition, not counted after it... an applying run changes the
        # occupancy it is checking") applied at the moment this loop actually
        # changes it, not only at the moment a proposal was merely accepted.
        # Only when this transition would ADD a name, exactly `decide()`'s
        # own test: a `scale` follow-on tranche into an already-positioned
        # ticker changes no count.
        if (
            proposal["to"] in lifecycle_states.POSITIONED
            and item["from_state"] not in lifecycle_states.POSITIONED
        ):
            cap_reasons = concentration_gate(lane, ticker_sectors.get(ticker))
            if cap_reasons and not override_caps:
                owner_decisions.append({
                    "date": str(at_date), "kind": "cap_withheld",
                    "ticker": ticker, "lane": lane, "to": proposal["to"],
                    "reasons": cap_reasons,
                })
                return

        bar = resolved_bar(ticker, at_date)
        if bar is None:
            errors.append({
                "ticker": ticker, "date": str(at_date),
                "error": f"entry confirmed for {at_date} but no usable price bar exists on or before it",
            })
            return
        result = ledger.buy(ticker, lane, bar, config)
        trade_log.append({**result, "kind": "buy"})
        if result.get("filled"):
            watchlist.transition(
                ticker, proposal["to"], proposal["trigger_id"],
                evidence=proposal["evidence"], applied_by=APPLIED_OWNER,
                at=at_date.isoformat(),
            )

    # ── settlement: the kill-switch's exit_review -> exited, then the sale ──

    def settle_exit(item: dict, at_date, outcomes_this_date: list[dict]) -> None:
        ticker, proposal = item["ticker"], item["proposal"]
        entry = watchlist.get(ticker)
        if entry is None:
            return
        if entry["state"] != item["from_state"] and entry["state"] != EXIT_REVIEW:
            # Neither still positioned as proposed, nor already in review —
            # overtaken by something else since scheduling. Nothing to do.
            return

        # Resolved and validated BEFORE any durable write. `confirm_exit`
        # below performs KTD10's three-store write (queue event, the
        # EXIT_REVIEW->EXITED transition, the confirmation stamp) — once it
        # succeeds, the watchlist and queue agree the sale happened, and
        # nothing here can undo that. `price_bars.bar_on_or_before` filters
        # NaN/estimated-alias rows but not a non-positive close (a real,
        # if rare, corpus data artifact), and `Ledger.sell`'s own
        # `_bar_price` guard raises on exactly that — checked too late if it
        # runs after `confirm_exit`, leaving the watchlist/queue saying
        # "exited" while the ledger still holds the position for the rest of
        # the run. Validating here first, matching `settle_entry`'s own
        # order, means a bad bar refuses the WHOLE settlement — no
        # transition, no confirm_exit, nothing written — rather than leaving
        # the two stores disagreed.
        bar = resolved_bar(ticker, at_date)
        if bar is None or bar["price"] <= 0:
            errors.append({
                "ticker": ticker, "date": str(at_date),
                "error": (
                    f"exit due for {at_date} but no usable positive price bar "
                    f"exists on or before it — settlement refused, nothing written"
                ),
            })
            return

        # `decide()` always runs with `apply=False`, so `exit_review` itself
        # was never written when the proposal fired — this settlement writes
        # it now, mirroring what `advance_ticker` would have written under
        # `apply=True` at proposal time, just later, once the owner's lag
        # has elapsed. Skipped when already in `exit_review` (the `elif`
        # above already refused every other stale state).
        if entry["state"] == item["from_state"]:
            watchlist.transition(
                ticker, EXIT_REVIEW, proposal["trigger_id"],
                evidence=proposal["evidence"], applied_by=APPLIED_OWNER,
                at=at_date.isoformat(),
            )

        raw_price = price_frames.get(ticker)
        truncated_price = (
            raw_price[raw_price["date"] <= pd.Timestamp(at_date)]
            if raw_price is not None else None
        )
        shim = _ReplayService(truncated_price, config)
        confirmation = exit_module.confirm_exit(watchlist, queue, ticker, shim, as_of=at_date)
        if not confirmation.get("ok"):
            errors.append({
                "ticker": ticker, "date": str(at_date),
                "error": confirmation.get("reason", "confirm_exit refused"),
            })
            return

        # Separately and independently (R5/KTD5): `confirm_exit`'s own
        # friction reading and the ledger's cash settlement are two
        # intentionally distinct models, never reconciled against each
        # other. The bar is the confirmed date's own, already validated above.
        fraction = item.get("sell_fraction") or 1.0
        reason = f"{proposal.get('trigger_id', '')}: {proposal.get('evidence', '')}"
        for settlement in ledger.sell(ticker, fraction, bar, reason, config):
            trade_log.append({**settlement, "kind": "sell"})

        # KTD3: schedule the routing of this exit's proceeds, over
        # production's own ranking (`ReinvestmentQueue.propose_routing`).
        exit_event = confirmation["event"]
        view = queue.propose_routing(watchlist, outcomes_this_date, concentration_reading(), as_of=at_date)
        ranked_candidates = [view["proposal"]] if view.get("proposal") else []
        route_decision = owner_module.route(exit_event, ranked_candidates, at_date, config)
        owner_decisions.append({"date": str(at_date), "kind": "route", **route_decision})
        if route_decision["action"] == "confirm":
            candidate = route_decision["candidate"] or {}
            scheduled.append({
                "kind": "route", "confirm_at": route_decision["confirm_at"],
                "exit_id": exit_event["exit_id"],
                "ticker": candidate.get("ticker"),
                "lane": candidate.get("lane") or CORE_LANE,
            })

    # ── settlement: reinvestment routing — a buy funded by an exit's proceeds ──

    def settle_route(item: dict, at_date) -> None:
        candidate, lane = item["ticker"], item["lane"]
        entry = watchlist.get(candidate)
        if entry is None or entry["state"] in (
            lifecycle_states.DROPPED, EXIT_REVIEW, lifecycle_states.EXITED,
        ):
            # No longer a live destination for capital by the time the
            # routing lag elapsed — proceeds stay in cash (a longer idle
            # reading), never a buy into a candidate that has since left.
            return

        # Same live re-check as `settle_entry`, and for the identical
        # reason: only when this routing would move the candidate INTO a
        # positioned state (from CANDIDATE_STATES) does it add a name the
        # cap counts. A routed buy into an already-positioned candidate
        # (funding a further tranche) changes no count and is not gated.
        if entry["state"] in CANDIDATE_STATES:
            cap_reasons = concentration_gate(lane, ticker_sectors.get(candidate))
            if cap_reasons and not override_caps:
                owner_decisions.append({
                    "date": str(at_date), "kind": "cap_withheld",
                    "ticker": candidate, "lane": lane, "to": PROBE,
                    "reasons": cap_reasons,
                })
                return  # proceeds stay in cash; idle_days keeps counting

        bar = resolved_bar(candidate, at_date)
        if bar is None:
            errors.append({
                "ticker": candidate, "date": str(at_date),
                "error": f"routing confirmed for {at_date} but no usable price bar exists on or before it",
            })
            return
        result = ledger.buy(candidate, lane, bar, config)
        trade_log.append({**result, "kind": "buy"})
        if not result.get("filled"):
            return  # the exit's proceeds stay unrouted; idle_days keeps counting
        if entry["state"] in CANDIDATE_STATES:
            watchlist.transition(
                candidate, PROBE, "reinvestment_routing",
                evidence=(
                    f"funded by exit {item['exit_id']}'s proceeds via the simulated "
                    f"reinvestment router (KTD3) — not a triggers.yaml-declared entry"
                ),
                applied_by=APPLIED_OWNER,
                at=at_date.isoformat(),
            )
        # else: already positioned (an earlier tranche settled separately) —
        # this buy just adds a tranche; no transition to write.
        queue.record_routing(item["exit_id"], candidate, deployed_at=result["entry_bar_date"])

    # ── the date loop ──

    for raw_date in calendar_result.dates:
        cutoff = pd.Timestamp(raw_date)

        # Step 0 (KTD8): a ticker joins the watchlist exactly when the loop
        # reaches its own first-eligible date — never earlier, or it would
        # be visible to the concentration gate and evaluated on dates
        # before it was truly eligible.
        already_tracked = set(watchlist.tickers())
        for ticker, first_date in sorted(universe_result.eligible.items()):
            if ticker in already_tracked or first_date > cutoff:
                continue
            lane = assignments[ticker].lane if ticker in assignments else CORE_LANE
            watchlist.add(ticker, lane=lane)

        active_tickers = sorted(watchlist.tickers())

        # Step 2's pace half: resolved once per date, off this date's own
        # truncated corpus (not once per run, unlike `advance()` — see the
        # module-level note above `_corpus_spread_at`).
        pace_reading = _corpus_spread_at(
            raw_by_ticker.keys(), truncate_cached, cutoff, engines.engine.macro,
        )
        pace_triggers, pace_decision = pace_module.modulate(
            trigger_registry, pace_reading, **pace_module.config_from(config)
        )
        trigger_evaluator = TriggerEvaluator(pace_triggers, known_metric_ids=known_metric_ids)

        outcomes_this_date: list[dict] = []

        for ticker in active_tickers:
            try:
                entry = watchlist.get(ticker)
                state, lane = entry["state"], entry["lane"]

                truncated = truncate_cached(ticker, cutoff)
                collect_reconciliation(ticker, cutoff, truncated)
                metrics = engines.engine.run_all(truncated)
                scores = engines.scorer.score(metrics)
                eligibility = engines.eligibility.evaluate(metrics)
                sector = (truncated.get("metadata") or {}).get("sector")
                ticker_sectors[ticker] = sector

                # Step 3: `decide()` — production's single statement of
                # "what should happen next" (KTD1), always `apply=False`.
                decision = advance_module.decide(
                    ticker, entry, state, lane,
                    metrics=metrics, scores=scores, eligibility=eligibility,
                    data=truncated, as_of=cutoff,
                    evaluator=trigger_evaluator, lane_gates=engines.lane_gates,
                    pace=pace_decision, apply=False,
                    concentration_gate=concentration_gate,
                    override_caps=override_caps, config=config,
                )

                # R9's exclusion count: `checkpoints_failed` can never fire
                # in a replay (no LLM Pass 2 ever runs here, so no company
                # ever carries a recorded checkpoint) — counted whenever it
                # was at least *applicable*, so the exclusion is visible
                # rather than silently absent.
                if "checkpoints_failed" in decision["evaluation"]["triggers"]:
                    checkpoints_evaluated += 1

                lane_gate_records.append({
                    "date": cutoff, "ticker": ticker, "lane_gate_result": decision["lane_gates"],
                })

                # Written only when it actually changed — every write is a
                # full store commit (atomic_write_json: temp file, fsync,
                # rename, revision bump), and re-running it for every ticker
                # on every replay date when nothing moved is pure overhead
                # across a run that can reach hundreds of (ticker, date)
                # pairs. `entry` is re-read fresh (not the pre-loop `entry`
                # local, which a same-loop `watchlist.transition` above may
                # have already staled) so the comparison is against what is
                # actually on disk right now.
                if (watchlist.get(ticker) or {}).get("kill_switch_status") != decision["kill_switch_status"]:
                    watchlist.set_kill_switch_status(ticker, decision["kill_switch_status"])

                proposal = decision["proposal"]
                outcomes_this_date.append({
                    "ticker": ticker,
                    "composite": (scores or {}).get("composite"),
                    "sector": sector,
                    "proposal": proposal,
                    "routing_safety": decision["routing_safety"],
                })

                if proposal and proposal["applied"]:
                    # Pre-position transitions (`qualify`/`watch`/`dropped`)
                    # auto-apply inside `decide()` itself even under
                    # `apply=False` — this is the one write that mirrors.
                    watchlist.transition(
                        ticker, proposal["to"], proposal["trigger_id"],
                        evidence=proposal["evidence"],
                        applied_by=APPLIED_OWNER if decision["moves_money"] else APPLIED_AUTO,
                        at=cutoff.isoformat(),
                    )
                elif proposal and proposal["needs_confirmation"]:
                    # Step 4: hand the money-moving proposal to the
                    # simulated owner — unless one of the same kind for this
                    # ticker is already awaiting settlement. Settlement runs
                    # once per date, *after* this ticker loop, so a proposal
                    # confirmed today does not update the watchlist until
                    # its own confirm date arrives — on quarterly replay
                    # grain that is almost always a *later* replay date than
                    # today. Left unguarded, `decide()` would keep reading
                    # the same not-yet-applied `state` and re-propose the
                    # identical trigger every date until settlement finally
                    # catches up: harmless (the staleness guard in
                    # `settle_entry`/`settle_exit` refuses every stale
                    # duplicate but the first), but it would silently fill
                    # `owner_decisions`/`scheduled` with noise a real owner
                    # was never actually asked to repeat.
                    schedule_kind = "exit" if proposal["to"] == EXIT_REVIEW else "entry"
                    already_pending = any(
                        item["ticker"] == ticker and item["kind"] == schedule_kind
                        for item in scheduled
                    )
                    if not already_pending:
                        owner_decision = owner_module.decide(proposal, cutoff, config)
                        owner_decisions.append({
                            "date": str(cutoff.date()), "kind": "entry_or_exit", **owner_decision,
                        })
                        if owner_decision["action"] == "confirm":
                            scheduled.append({
                                "kind": schedule_kind,
                                "ticker": ticker, "lane": lane, "from_state": state,
                                "confirm_at": owner_decision["confirm_at"], "proposal": proposal,
                                "sell_fraction": owner_decision.get("sell_fraction"),
                            })

                # KTD6: a fast-lane candidate earns a fabricated catalyst
                # the moment it has cleared the five non-catalyst gates and
                # does not already carry an active one.
                if lane == RERATING_LANE and decision["lane_gates"] is not None:
                    current = watchlist.get(ticker) or {}
                    has_active = (
                        (current.get("catalyst") or {}).get("status")
                        == lifecycle_states.CATALYST_ACTIVE
                    )
                    if not has_active:
                        fabricated = owner_module.catalyst_for(
                            ticker, decision["lane_gates"], config, as_of=cutoff,
                        )
                        if fabricated is not None:
                            description, expected_by = fabricated
                            watchlist.record_catalyst(ticker, description, expected_by)

            except Exception as exc:  # noqa: BLE001 — one ticker's failure must not stop the rest
                logger.error(f"{ticker} at {cutoff.date()}: {exc}")
                errors.append({"ticker": ticker, "date": str(cutoff.date()), "error": str(exc)})
                continue

        # Steps 4-5: settle whatever the owner's lag has now cleared.
        due, still_pending = [], []
        for item in scheduled:
            confirm_date = as_date(item["confirm_at"])
            (due if confirm_date is not None and confirm_date <= cutoff.date() else still_pending).append(item)
        scheduled = still_pending

        for item in due:
            try:
                if item["kind"] == "exit":
                    settle_exit(item, as_date(item["confirm_at"]), outcomes_this_date)
                elif item["kind"] == "entry":
                    settle_entry(item, as_date(item["confirm_at"]))
                elif item["kind"] == "route":
                    settle_route(item, as_date(item["confirm_at"]))
            except Exception as exc:  # noqa: BLE001 — one settlement's failure must not stop the rest
                logger.error(f"settlement ({item['kind']}) for {item.get('ticker')} at {cutoff.date()}: {exc}")
                errors.append({
                    "ticker": item.get("ticker"), "date": str(cutoff.date()),
                    "error": f"settlement ({item['kind']}) failed: {exc}",
                })

        # Step 6: mark to market.
        equity_curve.append(ledger.mark_to_market(cutoff, price_frames))

    last_date = calendar_result.dates[-1] if calendar_result.dates else None
    exit_views = queue.exit_views(watchlist, as_of=last_date)

    return {
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "lane_gate_records": lane_gate_records,
        "exit_views": exit_views,
        "reconciliation_failures": reconciliation_failures,
        "checkpoint_excluded_transitions": checkpoints_evaluated,
        "owner_decisions": owner_decisions,
        "errors": errors,
        "unsettled_confirmations": scheduled,
        "price_frames": price_frames,
    }
