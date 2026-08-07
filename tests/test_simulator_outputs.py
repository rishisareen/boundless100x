"""`simulator.outputs` — the six §10 readings, the benchmark, gate coverage,
and the exclusions/limitations blocks (U6).

Per U6's own instructions, this module never runs a replay — U7 doesn't
exist yet. Every fixture here is either hand-built to the shapes
`outputs.py`'s own module docstring documents as its Input Contracts, or (for
the equity curve / trade log, where hand-typing every settlement field would
risk testing a fiction) produced by driving a REAL `Ledger` instance exactly
as U4's own test file does, then independently re-deriving the expected
numbers with plain arithmetic in this file — never by calling the function
under test on itself.
"""

import json
import math
import statistics
from datetime import date

import pandas as pd
import pytest

from boundless100x.simulator import outputs
from boundless100x.simulator.ledger import Ledger
from boundless100x.simulator.universe import UniverseResult
from boundless100x.watchlist import CORE_LANE, RERATING_LANE

CONFIG = {
    "simulator": {
        "starting_pool": 1000,
        "confirmation_lag_days": {"entry": 7, "exit": 3, "route": 4},
        "catalyst_window_months": 9,
        "cap_posture": "advisory",
    },
    "portfolio": {
        "sleeve_split": {"core": 1.0, "rerating": 1.0},
        "tranche_size_pct": {"core": 0.5, "rerating": 0.5},
    },
    "friction": {
        "stcg_pct": 20.0,
        "ltcg_pct": 12.5,
        "ltcg_holding_days": 365,
        "slippage_bps": 100,
    },
}


def bar(bar_date, price: float) -> dict:
    return {"date": bar_date, "price": price}


def tag(record: dict, kind: str) -> dict:
    return {**record, "kind": kind}


# ── the micro run: a real Ledger, driven by hand, re-derived independently ─


def build_micro_run():
    """One core-lane position: bought, marked through a dip and a recovery,
    then sold in full at a gain (STCG regime, 180-day hold).

    Prices: buy @100 (2023-01-02) -> mark @80 (2023-02-01, a dip) -> mark
    @120 (2023-04-02, a recovery past the entry price) -> sell @150
    (2023-07-01).
    """
    price_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-02-01", "2023-04-02", "2023-07-01"]),
        "close": [100.0, 80.0, 120.0, 150.0],
        "adj_close": [100.0, 80.0, 120.0, 150.0],
    })
    price_frames = {"TICK": price_df}
    ledger = Ledger(config=CONFIG)

    buy = ledger.buy("TICK", "core", bar(date(2023, 1, 2), 100.0), CONFIG)
    equity_curve = [ledger.mark_to_market(date(2023, 1, 2), price_frames)]
    equity_curve.append(ledger.mark_to_market(date(2023, 2, 1), price_frames))
    equity_curve.append(ledger.mark_to_market(date(2023, 4, 2), price_frames))

    settlements = ledger.sell("TICK", 1.0, bar(date(2023, 7, 1), 150.0), "test-exit", CONFIG)
    equity_curve.append(ledger.mark_to_market(date(2023, 7, 1), price_frames))

    trade_log = [tag(buy, "buy")] + [tag(s, "sell") for s in settlements]
    return equity_curve, trade_log, buy, settlements[0]


MICRO_EQUITY_CURVE, MICRO_TRADE_LOG, MICRO_BUY, MICRO_SETTLEMENT = build_micro_run()


class TestPortfolioCagr:
    def test_matches_hand_computation(self):
        start = MICRO_EQUITY_CURVE[0]["total_value"]
        end = MICRO_EQUITY_CURVE[-1]["total_value"]
        elapsed = (date(2023, 7, 1) - date(2023, 1, 2)).days
        expected = (end / start) ** (365.0 / elapsed) - 1.0

        result = outputs.portfolio_cagr(MICRO_EQUITY_CURVE)

        assert result["elapsed_days"] == elapsed == 180
        assert result["cagr_pct"] == pytest.approx(expected * 100.0)
        assert result["start_date"] == "2023-01-02"
        assert result["end_date"] == "2023-07-01"

    def test_empty_curve_reports_a_note_not_a_crash(self):
        result = outputs.portfolio_cagr([])
        assert result["cagr_pct"] is None
        assert "empty" in result["note"]

    def test_non_positive_start_value_is_not_annualized(self):
        curve = [
            {"date": "2023-01-01", "total_value": 0.0},
            {"date": "2023-06-01", "total_value": 100.0},
        ]
        result = outputs.portfolio_cagr(curve)
        assert result["cagr_pct"] is None
        assert "non-positive" in result["note"]


class TestMaxDrawdown:
    def test_matches_hand_computation(self):
        values = [p["total_value"] for p in MICRO_EQUITY_CURVE]
        assert values == pytest.approx([997.5, 897.5, 1097.5, 1194.5])
        expected_dd = (997.5 - 897.5) / 997.5 * 100.0  # peak 997.5 -> trough 897.5

        result = outputs.max_drawdown(MICRO_EQUITY_CURVE)

        assert result["max_drawdown_pct"] == pytest.approx(expected_dd)
        assert result["peak_value"] == pytest.approx(997.5)
        assert result["trough_value"] == pytest.approx(897.5)
        assert result["peak_date"] == "2023-01-02"
        assert result["trough_date"] == "2023-02-01"

    def test_monotonically_rising_curve_has_zero_drawdown(self):
        curve = [
            {"date": "2023-01-01", "total_value": 100.0},
            {"date": "2023-02-01", "total_value": 110.0},
            {"date": "2023-03-01", "total_value": 130.0},
        ]
        result = outputs.max_drawdown(curve)
        assert result["max_drawdown_pct"] == 0.0
        assert "no drawdown observed" in result["note"]


class TestTurnover:
    def test_matches_hand_computation(self):
        traded_notional = MICRO_BUY["notional"] + (
            MICRO_SETTLEMENT["qty"] * MICRO_SETTLEMENT["exit_price"]
        )
        values = [p["total_value"] for p in MICRO_EQUITY_CURVE]
        mean_value = sum(values) / len(values)
        elapsed = 180
        expected_annualized = (traded_notional / mean_value) * (365.0 / elapsed)

        result = outputs.turnover(MICRO_EQUITY_CURVE, MICRO_TRADE_LOG)

        assert result["traded_notional"] == pytest.approx(1250.0)
        assert result["mean_total_value"] == pytest.approx(1046.75)
        assert result["turnover_ratio_annualized"] == pytest.approx(expected_annualized)

    def test_states_its_annualization_convention(self):
        result = outputs.turnover(MICRO_EQUITY_CURVE, MICRO_TRADE_LOG)
        assert "365/elapsed_days" in result["note"]


class TestPerLaneNetVsGross:
    def test_matches_hand_computation_and_documents_the_pair(self):
        expected_gross = MICRO_SETTLEMENT["qty"] * (
            MICRO_SETTLEMENT["exit_price"] - MICRO_SETTLEMENT["entry_price"]
        )
        expected_net = MICRO_SETTLEMENT["proceeds"] - (
            MICRO_SETTLEMENT["qty"] * MICRO_SETTLEMENT["entry_price"]
        )
        assert expected_gross == pytest.approx(250.0)
        assert expected_net == pytest.approx(197.0)

        result = outputs.per_lane_net_vs_gross(MICRO_TRADE_LOG)

        core = result["by_lane"][CORE_LANE]
        assert core["gross"] == pytest.approx(expected_gross)
        assert core["net"] == pytest.approx(expected_net)
        assert core["net_of_slippage_pretax"] == pytest.approx(MICRO_SETTLEMENT["gain"])
        assert core["n_settlements"] == 1
        assert "gross" in result["definition"] and "net" in result["definition"]


class TestFastLaneBreakEven:
    def test_no_rerating_cycles_reports_unmeasured_not_zero_or_none(self):
        """The micro run is entirely core-lane — zero rerating cycles. §8.2's
        break-even must say so plainly rather than returning a 0/None that
        would read as "no gap."
        """
        result = outputs.fast_lane_break_even(MICRO_TRADE_LOG)
        assert result["status"] == "unmeasured"
        assert result["n_cycles"] == 0
        assert "reason" in result and result["reason"]
        assert "break_even_gap_pct" not in result

    def test_measured_case_matches_hand_computation(self):
        ledger = Ledger(config=CONFIG)
        buy = ledger.buy("FAST", "rerating", bar(date(2023, 1, 2), 50.0), CONFIG)
        assert buy["filled"]
        settlements = ledger.sell(
            "FAST", 1.0, bar(date(2023, 4, 2), 75.0), "fast_lane_target_reached", CONFIG
        )
        settlement = settlements[0]
        assert settlement["holding_days"] == 90

        gross_return = (settlement["exit_price"] - settlement["entry_price"]) / settlement["entry_price"]
        net_return = (
            settlement["proceeds"] - settlement["qty"] * settlement["entry_price"]
        ) / (settlement["qty"] * settlement["entry_price"])
        factor = 365.0 / 90
        annualized_gross = (1 + gross_return) ** factor - 1
        annualized_net = (1 + net_return) ** factor - 1
        expected_gap = (annualized_gross - annualized_net) * 100.0

        trade_log = [tag(buy, "buy"), tag(settlement, "sell")]
        result = outputs.fast_lane_break_even(trade_log)

        assert result["status"] == "measured"
        assert result["n_cycles"] == 1
        assert result["break_even_gap_pct"] == pytest.approx(expected_gap)
        assert result["cycles"][0]["ticker"] == "FAST"

    def test_central_tendency_is_configurable_and_documented(self):
        ledger = Ledger(config=CONFIG)
        trade_log = []
        # Two independent rerating round trips with different gaps.
        for i, (entry, exit_, days) in enumerate([(50.0, 75.0, 90), (50.0, 55.0, 400)]):
            ticker = f"FAST{i}"
            buy = ledger.buy(ticker, "rerating", bar(date(2023, 1, 2), entry), CONFIG)
            exit_date = date(2023, 1, 2) + pd.Timedelta(days=days)
            settlement = ledger.sell(ticker, 1.0, bar(exit_date, exit_), "fast_lane_target_reached", CONFIG)[0]
            trade_log += [tag(buy, "buy"), tag(settlement, "sell")]

        median_result = outputs.fast_lane_break_even(trade_log, central_tendency="median")
        mean_result = outputs.fast_lane_break_even(trade_log, central_tendency="mean")
        gaps = [c["annualized_gap_pct"] for c in median_result["cycles"]]

        assert median_result["central_tendency"] == "median"
        assert median_result["break_even_gap_pct"] == pytest.approx(statistics.median(gaps))
        assert mean_result["central_tendency"] == "mean"
        assert mean_result["break_even_gap_pct"] == pytest.approx(sum(gaps) / len(gaps))


class TestCashDrag:
    def test_idle_days_reads_exit_views_verbatim(self):
        exit_views = [
            {"exit_id": "A::1", "idle_days": 10},
            {"exit_id": "B::1", "idle_days": 20},
            {"exit_id": "C::1", "idle_days": None},  # unreadable — excluded from the mean
        ]
        result = outputs.cash_drag(MICRO_EQUITY_CURVE, exit_views)

        idle = result["idle_days"]
        assert idle["n_exits"] == 3
        assert idle["n_with_readable_idle_days"] == 2
        assert idle["n_unreadable_idle_days"] == 1
        assert idle["mean_idle_days"] == pytest.approx(15.0)
        assert idle["median_idle_days"] == pytest.approx(15.0)

    def test_no_exit_views_is_distinguishable_from_zero_idle_days(self):
        result = outputs.cash_drag(MICRO_EQUITY_CURVE, None)
        idle = result["idle_days"]
        assert idle["mean_idle_days"] is None
        assert "no exit views were supplied" in idle["note"]

    def test_pool_idle_share_matches_hand_computation(self):
        expected_shares = [p["cash"] / p["total_value"] for p in MICRO_EQUITY_CURVE]
        result = outputs.cash_drag(MICRO_EQUITY_CURVE, [])
        pool = result["pool_idle_share"]
        assert pool["mean_idle_share"] == pytest.approx(sum(expected_shares) / len(expected_shares))
        assert pool["n_points"] == len(MICRO_EQUITY_CURVE)


# ── trade-log tagging contract ─────────────────────────────────────────


class TestTradeLogTagging:
    def test_untagged_record_raises_with_the_index(self):
        with pytest.raises(ValueError, match=r"trade_log\[0\]"):
            outputs.turnover(MICRO_EQUITY_CURVE, [MICRO_BUY])  # no "kind" key

    def test_unrecognised_kind_raises(self):
        with pytest.raises(ValueError, match="kind="):
            outputs.turnover(MICRO_EQUITY_CURVE, [{**MICRO_BUY, "kind": "sideways"}])

    def test_a_refused_buy_is_skipped_not_counted(self):
        refused = {"filled": False, "ticker": "X", "reason": "no headroom", "kind": "buy"}
        buys, sells = outputs._split_trade_log([refused])
        assert buys == [] and sells == []


# ── gate coverage matrix ────────────────────────────────────────────────


ALL_GATE_IDS = outputs.LANE_GATE_IDS


def gate_result(**passed_by_gate) -> dict:
    """A `LaneGateEvaluator.evaluate()`-shaped result, hand-built exactly as
    `test_simulator_owner.py`'s own `gate_result_for` does — every gate
    defaults to `passed=True`; name the ones that should differ. Mirrors the
    evaluator's own precedence (a failure beats an unknown, an unknown beats
    a pass) so a test fixture cannot desync from what the real evaluator
    would compute for the same per-gate readings.
    """
    gates = {gid: {"passed": True, "label": gid, "reason": ""} for gid in ALL_GATE_IDS}
    for gid, passed in passed_by_gate.items():
        gates[gid] = {"passed": passed, "label": gid, "reason": ""}
    failed = sorted(g for g, d in gates.items() if d["passed"] is False)
    indeterminate = sorted(g for g, d in gates.items() if d["passed"] is None)
    if failed:
        verdict, qualifies = "not_qualified", False
    elif indeterminate:
        verdict, qualifies = "indeterminate", None
    else:
        verdict, qualifies = "qualifies", True
    return {
        "qualifies": qualifies, "verdict": verdict, "gates": gates,
        "failed": failed, "indeterminate": indeterminate,
    }


class TestGateCoverageMatrix:
    def test_no_records_at_all_is_unmeasured(self):
        for empty in (None, []):
            result = outputs.gate_coverage_matrix(empty)
            assert result["status"] == "unmeasured"
            assert result["reason"]
            assert result["windows"] == {}

    def test_near_miss_records_which_gates_decided_it(self):
        """Most gates pass; two (growth_intact, catalyst_identified) read
        indeterminate, so the aggregate verdict is INDETERMINATE (never
        QUALIFIES with any gate unknown) — the matrix must still show
        exactly which four gates decided this specific reading.
        """
        record = {
            "date": date(2023, 5, 1), "ticker": "FASTCO",
            "lane_gate_result": gate_result(growth_intact=None, catalyst_identified=None),
        }

        result = outputs.gate_coverage_matrix([record])

        window = result["windows"]["2023-05-01"]
        assert window["status"] == "measured"
        assert window["gates"]["growth_intact"] == {
            "passed": 0, "failed": 0, "indeterminate": 1, "not_evaluated": 0,
        }
        assert window["gates"]["catalyst_identified"]["indeterminate"] == 1
        for passing_gate in ("quality_floor", "valuation_discount", "institutional_accumulation", "liquidity_floor"):
            assert window["gates"][passing_gate]["passed"] == 1

        reading = window["readings"][0]
        assert reading["ticker"] == "FASTCO"
        assert reading["verdict"] == "indeterminate"
        assert reading["qualifies"] is None
        assert reading["deciding_gates"]["growth_intact"] is None
        assert reading["deciding_gates"]["catalyst_identified"] is None
        assert reading["deciding_gates"]["quality_floor"] is True
        assert reading["deciding_gates"]["valuation_discount"] is True
        assert reading["deciding_gates"]["institutional_accumulation"] is True
        assert reading["deciding_gates"]["liquidity_floor"] is True

    def test_a_window_with_only_core_lane_records_is_unmeasured_not_zero(self):
        """A core-lane ticker's `lane_gate_result` is None — decide() never
        lane-gate-evaluates it. A window holding only such records must read
        `unmeasured`, never as all-zero gate counts (which would look
        indistinguishable from "many candidates, none qualified").
        """
        record = {"date": date(2023, 2, 1), "ticker": "COREONLY", "lane_gate_result": None}
        result = outputs.gate_coverage_matrix([record])

        window = result["windows"]["2023-02-01"]
        assert window["status"] == "unmeasured"
        assert "core-lane" in window["reason"] or "None" in window["reason"]
        assert window["gates"] == {}
        assert window["readings"] == []

    def test_a_mixed_window_is_measured_and_the_none_record_counts_not_evaluated(self):
        records = [
            {"date": date(2023, 3, 1), "ticker": "COREONLY", "lane_gate_result": None},
            {
                "date": date(2023, 3, 1), "ticker": "FASTCO",
                "lane_gate_result": gate_result(),  # all six pass
            },
        ]
        result = outputs.gate_coverage_matrix(records)

        window = result["windows"]["2023-03-01"]
        assert window["status"] == "measured"
        assert window["n_records"] == 2
        assert window["n_measured"] == 1
        # Every gate's tally: 1 pass (FASTCO) + 1 not_evaluated (COREONLY).
        for gate_id in ALL_GATE_IDS:
            assert window["gates"][gate_id]["passed"] == 1
            assert window["gates"][gate_id]["not_evaluated"] == 1
        assert len(window["readings"]) == 1  # only the measured ticker

    def test_unreadable_date_raises(self):
        with pytest.raises(ValueError, match="unreadable date"):
            outputs.gate_coverage_matrix([{"date": "not a date", "ticker": "X", "lane_gate_result": None}])


# ── exclusions ────────────────────────────────────────────────────────


class TestDescribeExclusions:
    def _build(self):
        universe_result = UniverseResult(excluded={"NEVER1": "too short", "NEVER2": "too short"})
        gate_coverage_result = outputs.gate_coverage_matrix([
            {"date": date(2023, 5, 1), "ticker": "A", "lane_gate_result": gate_result(growth_intact=None)},
            {"date": date(2023, 6, 1), "ticker": "B", "lane_gate_result": gate_result(liquidity_floor=None, quality_floor=None)},
        ])
        equity_curve = [
            {"date": "2023-01-01", "cash": 10, "total_value": 100, "stale_marks": ["X"]},
            {"date": "2023-02-01", "cash": 10, "total_value": 110, "stale_marks": ["X", "Y"]},
        ]
        reconciliation_failures = [
            {"ticker": "X", "date": "2023-01-01", "code": "never_fetched", "detail": "no face value"},
            {"ticker": "Y", "date": "2023-01-01", "code": "reconciliation_failed", "detail": "diverged"},
            {"ticker": "Z", "date": "2023-01-01", "code": "never_fetched", "detail": "no raw close"},
        ]
        return outputs.describe_exclusions(
            universe_result=universe_result,
            checkpoint_excluded_transitions=["ASTRAL::checkpoints_failed", "TCS::checkpoints_failed"],
            gate_coverage_result=gate_coverage_result,
            equity_curve=equity_curve,
            reconciliation_failures=reconciliation_failures,
        )

    def test_every_exclusion_kind_is_present_and_counted(self):
        exclusions = self._build()
        by_category = {item["category"]: item for item in exclusions}

        assert set(by_category) == {
            "never_eligible_tickers", "checkpoint_driven_transitions_excluded",
            "gate_indeterminate_readings", "stale_mark_events", "reconciliation_failures",
        }
        assert by_category["never_eligible_tickers"]["count"] == 2
        assert by_category["never_eligible_tickers"]["items"] == {
            "NEVER1": "too short", "NEVER2": "too short",
        }
        assert by_category["checkpoint_driven_transitions_excluded"]["count"] == 2
        assert by_category["gate_indeterminate_readings"]["count"] == 3  # 1 + 2
        assert by_category["gate_indeterminate_readings"]["items"]["growth_intact"] == 1
        assert by_category["gate_indeterminate_readings"]["items"]["liquidity_floor"] == 1
        assert by_category["gate_indeterminate_readings"]["items"]["quality_floor"] == 1
        assert by_category["stale_mark_events"]["count"] == 3  # X, X+Y
        assert by_category["stale_mark_events"]["items"] == {"X": 2, "Y": 1}
        assert by_category["reconciliation_failures"]["count"] == 3
        assert by_category["reconciliation_failures"]["items_by_code"] == {
            "never_fetched": 2, "reconciliation_failed": 1,
        }

    def test_int_checkpoint_count_is_accepted(self):
        exclusions = outputs.describe_exclusions(checkpoint_excluded_transitions=5)
        by_category = {item["category"]: item for item in exclusions}
        assert by_category["checkpoint_driven_transitions_excluded"]["count"] == 5
        assert by_category["checkpoint_driven_transitions_excluded"]["items"] == []

    def test_absent_inputs_all_read_as_zero_not_crash(self):
        exclusions = outputs.describe_exclusions()
        for item in exclusions:
            assert item["count"] == 0


# ── limitations ──────────────────────────────────────────────────────


class TestBuildLimitations:
    def test_names_every_simulated_owner_policy_by_its_actual_value(self):
        limitations = outputs.build_limitations(config=CONFIG)
        policies = limitations["simulated_owner_policies"]

        assert policies["starting_pool"] == 1000
        assert policies["confirmation_lag_days"] == {"entry": 7, "exit": 3, "route": 4}
        assert policies["catalyst_window_months"] == 9
        assert policies["cap_posture"] == "advisory"

    def test_battery_complete_none_is_named_plainly(self):
        calendar_result = {
            "battery_complete": {CORE_LANE: pd.Timestamp("2023-03-31"), RERATING_LANE: None},
            "battery_detail": {
                RERATING_LANE: {"reason": "no ticker ever supplies both required windows"},
            },
        }
        limitations = outputs.build_limitations(calendar_result=calendar_result, config=CONFIG)
        text = limitations["quarterly_depth"]

        assert "2023-03-31" in text
        assert "never completes within the replay window" in text
        assert "no ticker ever supplies both required windows" in text

    def test_fast_lane_gate_coverage_caveat_is_always_present(self):
        """R8/point 3: this caveat is named FROM DAY ONE, not derived only
        when coverage happens to be incomplete."""
        empty_gate_coverage = outputs.gate_coverage_matrix([])
        limitations = outputs.build_limitations(gate_coverage_result=empty_gate_coverage, config=CONFIG)
        assert "fast_lane_gate_coverage" in limitations
        assert limitations["fast_lane_gate_coverage"]

    def test_statistical_humility_clause_is_quoted(self):
        limitations = outputs.build_limitations(config=CONFIG)
        assert "directional only" in limitations["statistical_humility"]
        assert "minimum transition count" in limitations["statistical_humility"]

    def test_rebuilt_multiple_basis_names_ktd0(self):
        limitations = outputs.build_limitations(config=CONFIG)
        assert "_stock_pe_basis" in limitations["rebuilt_multiple_basis"]
        assert "Screener" in limitations["rebuilt_multiple_basis"]


# ── benchmark ────────────────────────────────────────────────────────


class TestBenchmark:
    def _build(self):
        universe_eligible = {"AAA": pd.Timestamp("2023-01-02"), "BBB": pd.Timestamp("2023-04-02")}
        replay_dates = [
            pd.Timestamp("2023-01-02"), pd.Timestamp("2023-04-02"), pd.Timestamp("2023-07-01"),
        ]
        aaa = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-04-02", "2023-07-01"]),
            "close": [100.0, 110.0, 120.0], "adj_close": [100.0, 110.0, 120.0],
        })
        bbb = pd.DataFrame({
            "date": pd.to_datetime(["2023-04-02", "2023-07-01"]),
            "close": [50.0, 60.0], "adj_close": [50.0, 60.0],
        })
        price_frames = {"AAA": aaa, "BBB": bbb}
        curve, trades = outputs.build_benchmark_curve(
            universe_eligible, replay_dates, price_frames, CONFIG,
            ticker_lanes={"AAA": CORE_LANE, "BBB": RERATING_LANE},
        )
        return curve, trades, price_frames, replay_dates

    def test_equal_weight_entries_never_sell_and_never_overdraw_cash(self):
        curve, trades, _, _ = self._build()

        buys = [t for t in trades if t["kind"] == "buy"]
        assert all(t["filled"] for t in buys), trades
        assert len(buys) == 2
        # Equal split of the STARTING pool's total cash outlay: both tickers
        # cost the same total (notional + slippage) even though their entry
        # prices differ.
        aaa_total_cost = buys[0]["notional"] + buys[0]["slippage"]
        bbb_total_cost = buys[1]["notional"] + buys[1]["slippage"]
        assert aaa_total_cost == pytest.approx(bbb_total_cost)
        assert aaa_total_cost == pytest.approx(500.0)
        # No sells ever appear in a benchmark trade log.
        assert all(t["kind"] == "buy" for t in trades)
        # The pool is exactly exhausted once both tickers have entered.
        assert curve[-1]["cash"] == pytest.approx(0.0)

    def test_lane_tagging_flows_through_to_the_trade_log(self):
        _, trades, _, _ = self._build()
        by_ticker = {t["ticker"]: t for t in trades}
        assert by_ticker["AAA"]["lane"] == CORE_LANE
        assert by_ticker["BBB"]["lane"] == RERATING_LANE

    def test_untagged_ticker_falls_back_to_unknown_lane(self):
        universe_eligible = {"AAA": pd.Timestamp("2023-01-02")}
        price_frames = {"AAA": pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02"]), "close": [100.0], "adj_close": [100.0],
        })}
        _, trades = outputs.build_benchmark_curve(
            universe_eligible, [pd.Timestamp("2023-01-02")], price_frames, CONFIG,
        )
        assert trades[0]["lane"] == "unknown"

    def test_per_lane_position_value_curve_isolates_each_lane(self):
        curve, trades, price_frames, replay_dates = self._build()

        core_curve = outputs.lane_position_value_curve(trades, price_frames, replay_dates, CORE_LANE)
        rerating_curve = outputs.lane_position_value_curve(trades, price_frames, replay_dates, RERATING_LANE)

        # AAA (core) holds a position from the first replay date onward.
        assert core_curve[0]["tickers_held"] == ["AAA"]
        assert core_curve[0]["positions_value"] > 0
        # BBB (rerating) has not entered yet at the first replay date.
        assert rerating_curve[0]["tickers_held"] == []
        assert rerating_curve[0]["positions_value"] == 0.0
        # By the last date both lanes hold their own ticker only.
        assert core_curve[-1]["tickers_held"] == ["AAA"]
        assert rerating_curve[-1]["tickers_held"] == ["BBB"]

    def test_empty_universe_returns_empty_curve_and_log(self):
        curve, trades = outputs.build_benchmark_curve({}, [pd.Timestamp("2023-01-02")], {}, CONFIG)
        assert curve == [] and trades == []


# ── build_result: assembly + JSON round trip ───────────────────────────


def _assert_json_native(value, path="$"):
    """Recursively assert `value` contains only str/int/float/bool/None/
    list/dict — no pd.Timestamp, no numpy scalar, no NaN/inf.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"{path}: non-finite float {value!r}"
        return
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"{path}: non-string key {k!r}"
            _assert_json_native(v, f"{path}.{k}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _assert_json_native(item, f"{path}[{i}]")
        return
    raise AssertionError(f"{path}: {type(value)!r} is not a JSON-native type ({value!r})")


class TestBuildResultAndRenderSummary:
    def _full_result(self) -> dict:
        universe_result = UniverseResult(excluded={"NEVER1": "too short"})
        gate_coverage_records = [
            {"date": date(2023, 5, 1), "ticker": "FASTCO", "lane_gate_result": gate_result(growth_intact=None)},
        ]
        exit_views = [{"exit_id": "A::1", "idle_days": 12}]
        reconciliation_failures = [
            {"ticker": "X", "date": "2023-01-01", "code": "never_fetched", "detail": "no face value"},
        ]
        calendar_result = {
            "battery_complete": {CORE_LANE: pd.Timestamp("2023-03-31"), RERATING_LANE: None},
            "battery_detail": {RERATING_LANE: {"reason": "never completes"}},
        }

        benchmark_curve, benchmark_trades = outputs.build_benchmark_curve(
            {"TICK": pd.Timestamp("2023-01-02")},
            [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-07-01")],
            {"TICK": pd.DataFrame({
                "date": pd.to_datetime(["2023-01-02", "2023-07-01"]),
                "close": [100.0, 150.0], "adj_close": [100.0, 150.0],
            })},
            CONFIG, ticker_lanes={"TICK": CORE_LANE},
        )

        return outputs.build_result(
            equity_curve=MICRO_EQUITY_CURVE,
            trade_log=MICRO_TRADE_LOG,
            benchmark_equity_curve=benchmark_curve,
            benchmark_trade_log=benchmark_trades,
            benchmark_per_lane={},
            lane_gate_records=gate_coverage_records,
            exit_views=exit_views,
            universe_result=universe_result,
            calendar_result=calendar_result,
            checkpoint_excluded_transitions=1,
            reconciliation_failures=reconciliation_failures,
            config=CONFIG,
        )

    def test_result_carries_every_top_level_section(self):
        result = self._full_result()
        assert set(result) == {
            "schema_version", "equity_curve", "trade_log", "benchmark",
            "metrics", "gate_coverage", "exclusions", "limitations",
        }
        assert result["schema_version"] == outputs.SCHEMA_VERSION
        assert set(result["metrics"]) == {
            "portfolio_cagr", "max_drawdown", "turnover",
            "per_lane_net_vs_gross", "fast_lane_break_even", "cash_drag",
        }

    def test_result_round_trips_through_json_unchanged(self):
        result = self._full_result()
        round_tripped = json.loads(json.dumps(result))
        assert round_tripped == result

    def test_result_contains_only_json_native_types(self):
        result = self._full_result()
        _assert_json_native(result)

    def test_result_without_a_benchmark_is_still_json_safe(self):
        result = outputs.build_result(equity_curve=MICRO_EQUITY_CURVE, trade_log=MICRO_TRADE_LOG)
        assert result["benchmark"] is None
        _assert_json_native(result)
        assert json.loads(json.dumps(result)) == result

    def test_render_summary_is_a_non_empty_string_and_does_not_crash(self):
        result = self._full_result()
        text = outputs.render_summary(result)
        assert isinstance(text, str) and text.strip()
        assert "Strategy" in text
        assert "Benchmark" in text
        assert "Gate coverage" in text
        assert "Exclusions" in text

    def test_render_summary_handles_a_bare_no_benchmark_no_lane_gate_result(self):
        result = outputs.build_result(equity_curve=MICRO_EQUITY_CURVE, trade_log=MICRO_TRADE_LOG)
        text = outputs.render_summary(result)
        assert "not supplied" in text
        assert "unmeasured" in text  # fast-lane break-even AND gate coverage
