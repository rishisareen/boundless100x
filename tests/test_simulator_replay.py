"""`simulator.replay` — the skeleton: raw universe -> one simulated
watchlist, every KTD8-eligible ticker `add`ed at `screen` in its lane.

U7 builds the actual six-step loop on top of this; per the plan and the
implementation note in `replay.py`, this file proves the skeleton wires the
pieces together correctly rather than exhaustively covering lane-gate
arithmetic (already covered by `tests/test_lane_gates.py`) or scoring
mechanics (covered by the compute-engine and backtest test suites).
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import states as lifecycle_states
from boundless100x.lifecycle.lane_gates import DEFAULT_LANE_GATES
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import replay as replay_module
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
