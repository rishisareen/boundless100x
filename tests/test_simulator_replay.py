"""`simulator.replay` — the skeleton (U2) and the full six-step loop (U7):
raw universe -> one simulated watchlist -> truncate/score/evaluate/propose/
confirm/settle/mark-to-market, replayed date by date.

U2's tests (above) prove the skeleton wires the pieces together correctly
rather than exhaustively covering lane-gate arithmetic (already covered by
`tests/test_lane_gates.py`) or scoring mechanics (covered by the
compute-engine and backtest test suites). U7's own tests (below) are
integration coverage for the loop itself — that it runs and reconciles, that
a kill switch fires and settles correctly on schedule, that a ticker never
becomes visible before its own first-eligible date, and that
`simulator.simulate`'s override seam is genuinely exercised end to end. This
is deliberately **not** the plan's mandated hand-computed two-name exact-match
fixture (`tests/test_simulator_fixture.py`) — that is a separate follow-up
once this plumbing is reviewed.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.lane_gates import DEFAULT_LANE_GATES
from boundless100x.lifecycle.states import as_date
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import replay as replay_module
from boundless100x.simulator import universe as universe_module
from boundless100x.simulator import simulate as simulate_entry_point
from boundless100x.watchlist import CORE_LANE, RERATING_LANE
from tests.conftest import make_scores, write_ticker_dir

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── FAST_LANE_ENTRY_GATES stays in sync with the shipped registry ───────


def test_fast_lane_entry_gates_is_every_gate_but_the_catalyst():
    assert set(replay_module.FAST_LANE_ENTRY_GATES) | {"catalyst_identified"} == set(
        DEFAULT_LANE_GATES
    )
    assert "catalyst_identified" not in replay_module.FAST_LANE_ENTRY_GATES


# ── production wiring ────────────────────────────────────────────────────


def test_build_engines_wires_real_production_objects():
    engines = replay_module.build_engines()

    assert engines.engine.metrics  # the real registry, auto-discovered
    assert engines.scorer.element_weights == engines.engine.element_weights
    assert engines.eligibility.gates
    # Constructed with known_metric_ids — an unknown metric id in the
    # shipped lane-gate registry would have raised at construction.
    assert engines.lane_gates.gates


def test_build_stores_never_touches_production_paths():
    stores = replay_module.build_stores()
    try:
        assert "boundless100x-simulator-" in str(stores.tmpdir.name)
        assert stores.watchlist.path.parent == Path(stores.tmpdir.name)
        assert stores.queue.path.parent == Path(stores.tmpdir.name)
        # Empty stores start with nothing tracked.
        assert stores.watchlist.tickers() == []
    finally:
        stores.close()
    assert not Path(stores.tmpdir.name).exists()


# ── assign_lane: the five-gate rule ──────────────────────────────────────


def _clearing_metrics(**overrides) -> dict:
    """A candidate clearing all five of `FAST_LANE_ENTRY_GATES` — mirrors
    `test_lane_gates.py`'s `passing_metrics()` fixture, scoped to just the
    metrics those five gates read (no catalyst-related metric exists)."""
    metrics = {
        "pe_vs_historical": MetricResult(value=35.0),
        "growth_quality_grade": MetricResult(
            value="high_quality", flags=["growth_quality_high_quality"]
        ),
        "rerating_headroom": MetricResult(value=40.0),
        "ttm_growth_vs_cagr": MetricResult(value=6.0),
        "institutional_accumulation_streak": MetricResult(
            value=3.0, flags=["institutional_accumulation_rising"]
        ),
        "daily_turnover_ratio": MetricResult(value=0.05),
    }
    metrics.update(overrides)
    return metrics


@pytest.fixture(scope="module")
def engines():
    return replay_module.build_engines()


def test_assign_lane_enters_rerating_when_all_five_gates_clear(engines):
    lane, result, deciding = replay_module.assign_lane(
        _clearing_metrics(), make_scores(composite=6.5), engines,
    )

    assert lane == RERATING_LANE
    assert all(v is True for v in deciding.values())
    assert set(deciding) == set(replay_module.FAST_LANE_ENTRY_GATES)
    # The aggregate verdict is pinned INDETERMINATE by the missing
    # catalyst — assign_lane must not have trusted it.
    assert result["verdict"] == "indeterminate"
    assert result["gates"]["catalyst_identified"]["passed"] is None


def test_assign_lane_enters_core_when_one_gate_fails(engines):
    failing = _clearing_metrics(
        institutional_accumulation_streak=MetricResult(value=1.0)
    )
    lane, result, deciding = replay_module.assign_lane(
        failing, make_scores(composite=6.5), engines,
    )

    assert lane == CORE_LANE
    assert deciding["institutional_accumulation"] is False
    assert all(v is True for k, v in deciding.items() if k != "institutional_accumulation")


def test_assign_lane_enters_core_when_a_gate_is_indeterminate(engines):
    """A gate that could not be evaluated must not read as a pass — the
    same "indeterminate is not a pass" rule the gate evaluator itself
    enforces, one level up at the lane-assignment decision."""
    incomplete = _clearing_metrics()
    del incomplete["daily_turnover_ratio"]

    lane, result, deciding = replay_module.assign_lane(
        incomplete, make_scores(composite=6.5), engines,
    )

    assert lane == CORE_LANE
    assert deciding["liquidity_floor"] is None


# ── score_ticker_at: truncate + score through the production engine ─────


def test_score_ticker_at_rebuilds_valuation_for_the_gates(tmp_path, engines):
    from boundless100x.simulator import universe as universe_module

    ticker_dir = write_ticker_dir(
        tmp_path / "raw_data", "AAA",
        years=10, quarters=13, shareholding_quarters=12, price_days=3200,
    )
    data = universe_module.load_ticker_data(ticker_dir)

    metrics, scores = replay_module.score_ticker_at(data, pd.Timestamp("2023-02-28"), engines)

    assert "market_cap" in metrics and metrics["market_cap"].ok
    assert "composite" in scores


# ── build_initial_watchlist: the end-to-end skeleton ─────────────────────


def test_build_initial_watchlist_adds_only_eligible_tickers_at_screen(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "GOOD", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    write_ticker_dir(root, "SHORT", years=3, quarters=13, shareholding_quarters=12, price_days=3200)

    stores, cal, uni, assignments = replay_module.build_initial_watchlist(str(root))
    try:
        assert stores.watchlist.tickers() == ["GOOD"]
        entry = stores.watchlist.get("GOOD")
        assert entry["state"] == lifecycle_states.SCREEN
        assert entry["lane"] in (CORE_LANE, RERATING_LANE)
        assert entry["lane"] == assignments["GOOD"].lane

        assert "SHORT" in uni.excluded
        assert "GOOD" in uni.eligible
        assert assignments["GOOD"].entry_date == uni.eligible["GOOD"]
        assert "SHORT" not in assignments
    finally:
        stores.close()


def test_build_initial_watchlist_records_deciding_gates_per_ticker(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    stores, cal, uni, assignments = replay_module.build_initial_watchlist(str(root))
    try:
        assignment = assignments["AAA"]
        assert assignment.gate_result is not None
        assert set(assignment.deciding_gates) == set(replay_module.FAST_LANE_ENTRY_GATES)
        assert assignment.error is None
    finally:
        stores.close()


def test_build_initial_watchlist_reuses_a_supplied_calendar(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    cal = calendar_module.compute_calendar(root)
    stores, cal_out, uni, assignments = replay_module.build_initial_watchlist(
        str(root), calendar_result=cal,
    )
    try:
        assert cal_out is cal
    finally:
        stores.close()


# ── R10: production stores are provably untouched ────────────────────────


def _hash_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_stores_are_byte_identical_before_and_after(tmp_path):
    """The stop condition from the plan's Goal Capsule: production's
    `watchlist.json`, `score_history.jsonl` and
    `lifecycle/reinvestment_queue.json` must be provably untouched by any
    simulator code path — checked against the real repo paths directly
    (not through the test suite's autouse `isolate_live_stores` redirect,
    which would make this assertion vacuous by construction)."""
    watched_paths = [
        REPO_ROOT / "boundless100x" / "watchlist.json",
        REPO_ROOT / "boundless100x" / "score_history.jsonl",
        REPO_ROOT / "boundless100x" / "lifecycle" / "reinvestment_queue.json",
    ]
    before = {p: _hash_or_none(p) for p in watched_paths}

    root = tmp_path / "raw_data"
    write_ticker_dir(root, "GOOD", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    write_ticker_dir(root, "SHORT", years=3, quarters=13, shareholding_quarters=12, price_days=3200)

    stores, cal, uni, assignments = replay_module.build_initial_watchlist(str(root))
    try:
        assert "GOOD" in stores.watchlist.tickers()
    finally:
        stores.close()

    after = {p: _hash_or_none(p) for p in watched_paths}
    assert before == after, "a simulator run touched a production store"


# ═══════════════════════════════════════════════════════════════════════
# U7: the full six-step loop (`run_replay`) and the `simulate()` seam
# ═══════════════════════════════════════════════════════════════════════


def _run(root, config=None):
    """The U7 wiring every scenario below needs: engines, calendar,
    universe, lane assignments, fresh temp-dir stores, one `run_replay`
    call. Returns `(result, stores, calendar_result, universe_result)` —
    the caller is responsible for `stores.close()`.
    """
    engines = replay_module.build_engines(config)
    calendar_result = calendar_module.compute_calendar(root)
    universe_result = universe_module.build_universe(root, calendar_result.dates)
    assignments = replay_module.compute_lane_assignments(universe_result, engines)
    stores = replay_module.build_stores()
    result = replay_module.run_replay(
        stores, calendar_result, universe_result, assignments, engines, config,
    )
    return result, stores, calendar_result, universe_result


# ── cash + marks reconciles exactly, at every point on the curve ────────


def test_run_replay_produces_a_nonempty_reconciling_equity_curve(tmp_path):
    """A full loop run over a small synthetic multi-year, multi-ticker
    fixture: a non-empty equity curve, and `cash + positions_value ==
    total_value` exactly at every single point — the same exact-
    reconciliation property U4's own `Ledger` tests assert in isolation,
    proven here to survive the whole loop (truncation, scoring, `decide()`,
    owner confirmation, settlement, mark-to-market) rather than just the
    ledger's own arithmetic.
    """
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    write_ticker_dir(
        root, "DDD", years=10, quarters=13, shareholding_quarters=12, price_days=3200,
        market_cap=8000.0, financials_kwargs={"revenue_growth": 0.15, "pat_growth": 0.18},
    )

    result, stores, cal, uni = _run(root)
    try:
        assert result["errors"] == []
        curve = result["equity_curve"]
        assert len(curve) > 0
        for point in curve:
            assert point["total_value"] == point["cash"] + point["positions_value"], point

        # Real trades happened — an empty trade log would make the
        # reconciliation trivially true rather than a real proof.
        assert any(t["kind"] == "buy" and t.get("filled") for t in result["trade_log"])
    finally:
        stores.close()


# ── a kill switch fires mid-window, confirms after its own lag, taxes by ──
# ── holding period, and completes KTD10's confirm_exit trio ──────────────


def test_run_replay_kill_switch_exits_after_lag_and_books_tax_by_holding_period(tmp_path):
    """A fixture whose fundamentals genuinely break partway through: RoCE
    holds at a healthy 22% for the first eight of ten annual rows (enough
    for the company to qualify, await an entry price, and clear
    `valuation_buy_zone` on nothing but the default builders — verified
    empirically before writing this test, not assumed from the formulas),
    then craters to 8% for the last two — below `capital_efficiency_break`'s
    15% floor for two consecutive years, and visible only once those two
    rows themselves become visible (their own annual reporting lag), well
    after the position was already opened.

    Asserts, in KTD10's own order:
      * the position is bought before the kill switch is even readable;
      * the kill switch proposes `exit_review` (not before its own
        condition is readable, and not "no watchlist entry ever
        registered it");
      * the scheduled confirmation lands strictly after the exit lag, never
        before (the `probe`/`scale` state survives at least one further
        replay date after the trigger first fires);
      * the ledger closes the lot with tax booked by its own bar-to-bar
        holding period;
      * the confirm_exit trio is complete: an `exited` transition, an
        `exit` queue event, and a `confirmed` stamp.
    """
    root = tmp_path / "raw_data"
    ticker_dir = write_ticker_dir(
        root, "BBB", years=10, quarters=13, shareholding_quarters=12, price_days=3200,
    )
    ratios_path = ticker_dir / "ratios.csv"
    ratios = pd.read_csv(ratios_path)
    ratios.loc[ratios.index[-2:], "roce"] = 8.0  # the trailing two years break
    ratios.to_csv(ratios_path, index=False)

    result, stores, cal, uni = _run(root)
    try:
        assert result["errors"] == []

        entry = stores.watchlist.get("BBB")
        assert entry["state"] == lifecycle_states.EXITED

        history = entry["state_history"]
        probe_record = lifecycle_states.last_record_into(history, lifecycle_states.PROBE)
        review_record = lifecycle_states.last_record_into(history, lifecycle_states.EXIT_REVIEW)
        exited_record = lifecycle_states.last_record_into(history, lifecycle_states.EXITED)
        assert probe_record is not None and probe_record["trigger_id"] == "valuation_buy_zone"
        assert review_record is not None and review_record["trigger_id"] == "capital_efficiency_break"
        assert exited_record is not None and exited_record["trigger_id"] == "capital_efficiency_break"

        # The kill switch's own evidence names the trailing two years that
        # broke it, and the transition happened after the trigger's own
        # evidence date — never before the trigger fired at all.
        probe_at = as_date(probe_record["at"])
        review_at = as_date(review_record["at"])
        assert probe_at < review_at

        # "Confirmed after the exit lag, not before": find the earliest
        # date the simulated owner was actually asked about this exit
        # (kind == "entry_or_exit", proposing exit_review) and confirm the
        # scheduled confirmation date it returned is strictly after that
        # date — and that the transition landed on or after that
        # confirmation date, never earlier.
        exit_asks = [
            d for d in result["owner_decisions"]
            if d.get("kind") == "entry_or_exit"
            and d.get("proposal", {}).get("ticker") == "BBB"
            and d.get("proposal", {}).get("to") == lifecycle_states.EXIT_REVIEW
        ]
        assert exit_asks, "the simulated owner was never asked about BBB's exit"
        first_ask = exit_asks[0]
        asked_on = as_date(first_ask["date"])
        confirmed_at = as_date(first_ask["confirm_at"])
        assert asked_on < confirmed_at, "the exit lag produced no lag at all"
        assert review_at >= confirmed_at, (
            "the exit_review transition landed before its own scheduled confirmation"
        )

        # Tax booked by holding period: a sell settlement in the trade log,
        # closing the whole position (this fixture never partially exits),
        # with a positive holding period measured bar-to-bar.
        sells = [t for t in result["trade_log"] if t["kind"] == "sell" and t["ticker"] == "BBB"]
        assert len(sells) == 1
        settlement = sells[0]
        assert settlement["holding_days"] > 0
        assert settlement["regime"] in ("ltcg", "stcg")
        assert settlement["tax"] >= 0.0
        # The holding period run this fixture produces clears the LTCG
        # line (the position is held well over a year before the kill
        # switch's own two-year-persistence condition can even become
        # readable) — a gain past that line is taxed.
        if settlement["gain"] > 0:
            assert settlement["taxed"] is True
            assert settlement["tax"] > 0.0

        # KTD10's confirm_exit trio, on the queue.
        exits = stores.queue.exits()
        assert len(exits) == 1
        exit_event = exits[0]
        assert exit_event["ticker"] == "BBB"
        confirmation = stores.queue.find_confirmation(exit_event["exit_id"])
        assert confirmation is not None
        # And the transition itself, again, named directly by id: the
        # queue's own `exit_is_complete` reading agrees the sale is done.
        from boundless100x.lifecycle.reinvestment import exit_is_complete
        assert exit_is_complete(exit_event, entry, confirmation) is True
    finally:
        stores.close()


# ── a ticker joins the watchlist only once the loop reaches its own date ──


def test_run_replay_adds_a_ticker_only_at_its_own_first_eligible_date(tmp_path):
    """Two tickers whose KTD8 candidacy dates are years apart (a shorter
    financials history clears the engine's minimum-years bar much later):
    `run_replay` must not evaluate — and therefore must not make visible to
    the concentration gate — the later ticker before the replay loop
    actually reaches its own first-eligible date, even though
    `universe.build_universe` already knows that date up front and
    `compute_lane_assignments` already scored it there.
    """
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "GOOD", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    write_ticker_dir(root, "LATE", years=8, quarters=13, shareholding_quarters=12, price_days=3200)

    result, stores, cal, uni = _run(root)
    try:
        assert result["errors"] == []
        assert uni.eligible["GOOD"] < uni.eligible["LATE"]

        for ticker in ("GOOD", "LATE"):
            seen_dates = sorted({
                as_date(record["date"])
                for record in result["lane_gate_records"]
                if record["ticker"] == ticker
            })
            assert seen_dates, f"{ticker} was never evaluated at all"
            # The very first date this ticker was ever advanced is exactly
            # its own KTD8 first-eligible date — never earlier.
            assert seen_dates[0] == uni.eligible[ticker].date()

        # And LATE genuinely was not on the watchlist while GOOD's own
        # early history was being advanced: no LATE record exists on any
        # date GOOD was already active on but LATE's own eligibility had
        # not yet arrived.
        early_date = uni.eligible["GOOD"]
        late_date = uni.eligible["LATE"]
        assert early_date < late_date
        late_records_before_own_date = [
            r for r in result["lane_gate_records"]
            if r["ticker"] == "LATE" and as_date(r["date"]) < late_date.date()
        ]
        assert late_records_before_own_date == []
    finally:
        stores.close()


# ── production non-mutation, for a FULL replay (not just the skeleton) ──


def test_run_replay_never_touches_production_stores(tmp_path):
    """The Goal Capsule's stop condition, proven for the loop itself
    (`test_production_stores_are_byte_identical_before_and_after` above
    proves it for U2's skeleton only) — a full `run_replay` including every
    settlement path (buys, a kill-switch exit, routing) must still leave
    `boundless100x/watchlist.json`, `boundless100x/score_history.jsonl` and
    `boundless100x/lifecycle/reinvestment_queue.json` byte-identical.
    """
    watched_paths = [
        REPO_ROOT / "boundless100x" / "watchlist.json",
        REPO_ROOT / "boundless100x" / "score_history.jsonl",
        REPO_ROOT / "boundless100x" / "lifecycle" / "reinvestment_queue.json",
    ]
    before = {p: _hash_or_none(p) for p in watched_paths}

    root = tmp_path / "raw_data"
    ticker_dir = write_ticker_dir(
        root, "BBB", years=10, quarters=13, shareholding_quarters=12, price_days=3200,
    )
    ratios_path = ticker_dir / "ratios.csv"
    ratios = pd.read_csv(ratios_path)
    ratios.loc[ratios.index[-2:], "roce"] = 8.0
    ratios.to_csv(ratios_path, index=False)

    result, stores, cal, uni = _run(root)
    try:
        assert result["errors"] == []
        assert stores.watchlist.get("BBB")["state"] == lifecycle_states.EXITED
    finally:
        stores.close()

    after = {p: _hash_or_none(p) for p in watched_paths}
    assert before == after, "a full replay run touched a production store"


# ── R10: `simulate()`'s override seam, actually exercised end to end ────


def test_simulate_runs_without_subprocesses_and_returns_a_result(tmp_path):
    """`simulate(config, overrides)` is a plain in-process function call —
    no subprocess, no CLI dependency — and its return is exactly what
    `outputs.build_result` documents plus the fields `run_replay`/
    `simulate` add on top (`owner_decisions`, `errors`,
    `unsettled_confirmations`, `config`).
    """
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    result = simulate_entry_point(None, None, raw_data_dir=root, tickers=["AAA"])

    for key in (
        "schema_version", "equity_curve", "trade_log", "benchmark", "metrics",
        "gate_coverage", "exclusions", "limitations",
        "owner_decisions", "errors", "unsettled_confirmations", "config",
    ):
        assert key in result
    assert result["errors"] == []
    assert len(result["equity_curve"]) > 0


def test_simulate_wires_the_ktd9_benchmark(tmp_path):
    """A genuine gap found in review: `simulate()` built `run_replay`'s
    result and handed `outputs.build_result` none of `benchmark_equity_curve`
    / `benchmark_trade_log` / `benchmark_per_lane`, so every simulation's
    `result["benchmark"]` was silently `None` — despite KTD9 requiring the
    benchmark on every run ("so the outputs carry both a strategy and its
    counterfactual") and despite `test_simulate_runs_without_subprocesses_
    and_returns_a_result` above already checking `"benchmark" in result`,
    which a `None` value satisfies just as well as a real curve. This test
    is the sharper assertion that catches the gap that one could not.

    `AAA` (10 years of history via `write_ticker_dir`'s defaults) clears
    KTD8 candidacy comfortably inside the default replay window, so the
    benchmark has at least one ticker to buy an equal-weight position in —
    the fixture `build_benchmark_curve`'s own module docstring calls for
    ("a real equity curve", not an empty one from an empty universe).
    """
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    result = simulate_entry_point(None, None, raw_data_dir=root, tickers=["AAA"])

    assert result["errors"] == []
    benchmark = result["benchmark"]
    assert benchmark is not None
    assert len(benchmark["equity_curve"]) > 0
    assert benchmark["n_buys_filled"] >= 1
    # A real position was opened — the aggregate benchmark metrics block is
    # not the "empty equity curve" degenerate case `portfolio_cagr`/
    # `max_drawdown` report when there is nothing to compute from.
    aggregate = benchmark["metrics"]["aggregate"]
    assert aggregate["portfolio_cagr"]["cagr_pct"] is not None
    assert aggregate["max_drawdown"]["max_drawdown_pct"] is not None
    # Stated per lane as well as in aggregate (KTD9's own words).
    assert set(benchmark["metrics"]["per_lane"]) == {"core", "rerating"}


def test_simulate_override_seam_changes_recorded_policy_and_observed_behavior(tmp_path):
    """The Phase 5 seam, exercised end to end rather than merely present:
    `{"simulator.confirmation_lag_days.entry": 0}` must both (a) be
    recorded verbatim into the returned config/limitations block, and (b)
    actually change what the run *did* — an entry confirmed with zero lag
    settles on the same date it was proposed rather than five trading days
    later, which is a different, observable trade.
    """
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    default_result = simulate_entry_point(None, None, raw_data_dir=root, tickers=["AAA"])
    fast_result = simulate_entry_point(
        None, {"simulator.confirmation_lag_days.entry": 0}, raw_data_dir=root, tickers=["AAA"],
    )

    assert default_result["config"]["simulator"]["confirmation_lag_days"]["entry"] == 5
    assert fast_result["config"]["simulator"]["confirmation_lag_days"]["entry"] == 0
    recorded = fast_result["limitations"]["simulated_owner_policies"]["confirmation_lag_days"]
    assert recorded["entry"] == 0

    def first_buy_date(result):
        for trade in result["trade_log"]:
            if trade["kind"] == "buy" and trade.get("filled"):
                return trade["entry_bar_date"]
        return None

    default_buy = first_buy_date(default_result)
    fast_buy = first_buy_date(fast_result)
    assert default_buy is not None and fast_buy is not None
    assert as_date(fast_buy) < as_date(default_buy), (
        "a zero entry lag must settle no later than the default 5-day lag"
    )
