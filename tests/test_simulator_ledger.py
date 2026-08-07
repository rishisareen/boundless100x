"""`simulator.ledger` — modeled cash, per-tranche lots, mark-to-market (U4, KTD4).

Three things get proven, mirroring the plan's own U4 test scenarios plus the
one place this file's own judgment call (tranche-notional sizing) needs its
own coverage, since nothing else in the codebase pins it:

  * **FIFO lot mechanics** — independent holding periods per tranche, and a
    partial exit that closes the oldest lot first.
  * **Regime consistency from the ledger's own call site** — slippage on
    both legs, STCG/LTCG/loss, without re-testing `friction_cash.settle_sale`
    itself (U5's own test file owns that).
  * **Mark-to-market bar hygiene and exact reconciliation** — "on or before",
    never "nearest"; a stale ticker carried at its last known mark; the
    equity curve reconciling to `cash + positions_value` exactly, no
    `pytest.approx` needed because nothing in the ledger rounds.

`CONFIG` mirrors `config.yaml`'s shipped `friction:`/`portfolio:` defaults as
literals — the same convention `test_simulator_friction_cash.py`'s `SETTINGS`
and `tests/test_friction.py` use — so a `config.yaml` edit shows up here as a
failing arithmetic assertion rather than silently different numbers under the
same expectations.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from boundless100x.simulator import friction_cash
from boundless100x.simulator.ledger import BASIS_MODELED_CAPITAL, Ledger

CONFIG = {
    "simulator": {"starting_pool": 100},
    "portfolio": {
        "sleeve_split": {"core": 0.7, "rerating": 0.3},
        "tranche_size_pct": {"core": 0.33, "rerating": 0.5},
    },
    "friction": {
        "stcg_pct": 20.0,
        "ltcg_pct": 12.5,
        "ltcg_holding_days": 365,
        "slippage_bps": 100,
    },
}


def bar(bar_date, price: float) -> dict:
    """The `{date, price}` contract `buy`/`sell` require — see the ledger's
    own module docstring for why they take an already-resolved bar rather
    than a price frame."""
    return {"date": bar_date, "price": price}


# ── tranche-notional sizing (the one judgment call this file must pin) ────


class TestTrancheSizing:
    def test_first_tranche_in_an_empty_sleeve(self):
        """By hand: sleeve_target = 0.7 * 100 = 70 (nothing deployed yet, so
        headroom == sleeve_target); tranche = 0.33 * 70 = 23.1, comfortably
        under headroom so the `min(..., headroom)` cap does not bind.
        """
        ledger = Ledger(config=CONFIG)
        result = ledger.buy("TICK", "core", bar(date(2023, 1, 2), 100.0), CONFIG)

        assert result["filled"] is True
        assert result["notional"] == pytest.approx(23.1)
        assert result["tranche_index"] == 0
        assert result["basis"] == BASIS_MODELED_CAPITAL

    def test_later_tranches_shrink_headroom_and_the_sleeve_eventually_saturates(self):
        """A core position "built in thirds" bought four times at a fixed
        price (isolating the sizing formula from any mark-to-market drift):
        the first three tranches are each ~23.1/23.07/23.05 (shrinking as
        the sleeve fills, per the module docstring's worked example), the
        fourth is capped hard by the sliver of headroom left
        (~0.54), and a fifth is refused outright because the sleeve is
        saturated.

        This is the direct rebuttal of the *rejected* reading (tranche_pct
        of whatever headroom remains, forever) — that formula never
        terminates. This one visibly does, in four buys.
        """
        ledger = Ledger(config=CONFIG)
        fixed_bar = bar(date(2023, 1, 2), 100.0)
        expected_notionals = [23.1, 23.0733195, 23.0466698159775, 0.5377407214165828]

        for tranche_index, expected in enumerate(expected_notionals):
            result = ledger.buy("TICK", "core", fixed_bar, CONFIG)
            assert result["filled"] is True, result
            assert result["tranche_index"] == tranche_index
            assert result["notional"] == pytest.approx(expected, rel=1e-6)

        assert len(ledger.positions["TICK"]) == 4

        refused = ledger.buy("TICK", "core", fixed_bar, CONFIG)
        assert refused["filled"] is False
        assert "headroom" in refused["reason"]
        # A refusal never mutates state.
        assert len(ledger.positions["TICK"]) == 4

    def test_an_unconfigured_lane_refuses_rather_than_guessing_a_size(self):
        ledger = Ledger(config=CONFIG)
        result = ledger.buy("TICK", "not_a_real_lane", bar(date(2023, 1, 2), 100.0), CONFIG)
        assert result["filled"] is False
        assert "not_a_real_lane" in result["reason"]
        assert ledger.positions == {}


# ── FIFO lots and independent holding periods ─────────────────────────────


class TestFifoLotsAndHoldingPeriods:
    def test_two_tranches_hold_independent_periods_and_partial_exit_is_fifo(self):
        ledger = Ledger(config=CONFIG)
        d1, d2, d3 = date(2023, 1, 2), date(2023, 6, 1), date(2023, 12, 1)

        first = ledger.buy("TICK", "core", bar(d1, 100.0), CONFIG)
        second = ledger.buy("TICK", "core", bar(d2, 110.0), CONFIG)
        assert first["filled"] and second["filled"]

        lot0_qty = ledger.positions["TICK"][0]["qty"]
        lot1_qty = ledger.positions["TICK"][1]["qty"]
        total_qty = lot0_qty + lot1_qty

        # Close all of the older lot plus half of the newer one.
        fraction = (lot0_qty + lot1_qty / 2) / total_qty
        settlements = ledger.sell(
            "TICK", fraction, bar(d3, 130.0), reason="partial-fifo-exit", config=CONFIG
        )

        assert len(settlements) == 2
        oldest, newest = settlements
        assert oldest["qty"] == pytest.approx(lot0_qty)
        assert newest["qty"] == pytest.approx(lot1_qty / 2)
        assert oldest["tranche_index"] == 0
        assert newest["tranche_index"] == 1

        expected_holding_oldest = (d3 - d1).days
        expected_holding_newest = (d3 - d2).days
        assert oldest["holding_days"] == expected_holding_oldest
        assert newest["holding_days"] == expected_holding_newest
        # The whole point: the two lots price against genuinely different
        # holding periods, not the newer lot's period applied to both.
        assert oldest["holding_days"] != newest["holding_days"]

        for settlement in settlements:
            assert settlement["reason"] == "partial-fifo-exit"
            assert settlement["basis"] == BASIS_MODELED_CAPITAL

        # Exactly one lot remains: the newer one, partially consumed.
        assert len(ledger.positions["TICK"]) == 1
        remaining = ledger.positions["TICK"][0]
        assert remaining["qty"] == pytest.approx(lot1_qty / 2)
        assert remaining["entry_bar_date"] == d2
        assert remaining["entry_price"] == 110.0
        assert remaining["tranche_index"] == 1

    def test_selling_a_ticker_with_no_open_lots_returns_no_settlements(self):
        ledger = Ledger(config=CONFIG)
        assert ledger.sell("NEVER_BOUGHT", 1.0, bar(date(2023, 1, 2), 100.0), "test") == []


# ── friction, proven from the ledger's own call site ──────────────────────


class TestFrictionFromTheLedgersOwnCallSite:
    """U5's tax/slippage arithmetic is `test_simulator_friction_cash.py`'s to
    prove; this class proves only that the ledger hands `settle_sale` the
    right `holding_days` (bar-to-bar, per lot) and passes its regime back
    unchanged.
    """

    def test_slippage_reduces_cash_on_the_buy_leg_by_exactly_its_own_cost(self):
        ledger = Ledger(config=CONFIG)
        result = ledger.buy("TICK", "core", bar(date(2023, 1, 2), 100.0), CONFIG)

        expected_slippage = friction_cash.cost_of_buy(result["notional"], CONFIG)
        assert result["slippage"] == pytest.approx(expected_slippage)
        assert ledger.cash == pytest.approx(100.0 - result["notional"] - expected_slippage)
        # And strictly less than the naive "just the notional" reading.
        assert ledger.cash < 100.0 - result["notional"]

    def test_sell_proceeds_are_gross_less_exactly_the_reported_slippage_and_tax(self):
        ledger = Ledger(config=CONFIG)
        entry = date(2023, 1, 2)
        buy_result = ledger.buy("TICK", "core", bar(entry, 100.0), CONFIG)
        exit_date = entry + timedelta(days=400)

        [settled] = ledger.sell("TICK", 1.0, bar(exit_date, 150.0), "test-exit", CONFIG)

        gross_proceeds = buy_result["qty"] * 150.0
        assert settled["proceeds"] < gross_proceeds
        assert settled["proceeds"] == pytest.approx(
            gross_proceeds - settled["slippage"] - settled["tax"]
        )

    def test_a_gain_under_the_ltcg_threshold_is_taxed_stcg(self):
        ledger = Ledger(config=CONFIG)
        entry = date(2023, 1, 2)
        ledger.buy("TICK", "core", bar(entry, 100.0), CONFIG)
        exit_date = entry + timedelta(days=100)  # well under 365

        [settled] = ledger.sell("TICK", 1.0, bar(exit_date, 150.0), "test-stcg", CONFIG)

        assert settled["holding_days"] == 100
        assert settled["regime"] == "stcg"
        assert settled["tax_pct"] == 20.0
        assert settled["taxed"] is True

    def test_a_gain_over_the_ltcg_threshold_is_taxed_ltcg(self):
        ledger = Ledger(config=CONFIG)
        entry = date(2023, 1, 2)
        ledger.buy("TICK", "core", bar(entry, 100.0), CONFIG)
        exit_date = entry + timedelta(days=400)  # >= 365

        [settled] = ledger.sell("TICK", 1.0, bar(exit_date, 150.0), "test-ltcg", CONFIG)

        assert settled["holding_days"] == 400
        assert settled["regime"] == "ltcg"
        assert settled["tax_pct"] == 12.5
        assert settled["taxed"] is True

    def test_a_loss_is_untaxed(self):
        ledger = Ledger(config=CONFIG)
        entry = date(2023, 1, 2)
        ledger.buy("TICK", "core", bar(entry, 100.0), CONFIG)
        exit_date = entry + timedelta(days=400)

        [settled] = ledger.sell("TICK", 1.0, bar(exit_date, 50.0), "test-loss", CONFIG)

        assert settled["gain"] < 0
        assert settled["taxed"] is False
        assert settled["tax"] == 0.0


# ── mark-to-market ──────────────────────────────────────────────────────


class TestMarkToMarket:
    def test_a_gap_date_resolves_to_the_prior_trading_bar_not_the_nearest(self):
        """Friday 2023-01-06 prices at 100, the next bar is Monday
        2023-01-09 at 110. Marking on Sunday 2023-01-08 sits one calendar
        day from Monday and two from Friday — a nearest-neighbour rule
        would wrongly pick Monday's 110; "last bar on or before" correctly
        stays on Friday's 100, because Monday's bar had not printed yet as
        of the mark date.
        """
        ledger = Ledger(config=CONFIG)
        ledger.buy("TICK", "core", bar(date(2023, 1, 6), 100.0), CONFIG)

        price_frame = pd.DataFrame({
            "date": [date(2023, 1, 6), date(2023, 1, 9)],
            "close": [100.0, 110.0],
            "adj_close": [100.0, 110.0],
        })

        result = ledger.mark_to_market(date(2023, 1, 8), {"TICK": price_frame})

        assert result["marks"]["TICK"] == 100.0
        assert result["stale_marks"] == []

    def test_a_ticker_with_no_usable_bar_at_all_is_carried_at_its_last_known_mark(self):
        ledger = Ledger(config=CONFIG)
        ledger.buy("TICK", "core", bar(date(2023, 1, 2), 100.0), CONFIG)
        qty = ledger.positions["TICK"][0]["qty"]

        result = ledger.mark_to_market(date(2023, 6, 1), price_frames={})

        assert result["stale_marks"] == ["TICK"]
        assert result["marks"]["TICK"] == 100.0
        assert result["positions_value"] == qty * 100.0
        assert result["total_value"] == result["cash"] + result["positions_value"]

    def test_an_empty_or_all_nan_frame_is_stale_too(self):
        ledger = Ledger(config=CONFIG)
        ledger.buy("TICK", "core", bar(date(2023, 1, 2), 100.0), CONFIG)

        nan_frame = pd.DataFrame({
            "date": [date(2023, 1, 2), date(2023, 6, 1)],
            "close": [float("nan"), float("nan")],
            "adj_close": [float("nan"), float("nan")],
        })
        result = ledger.mark_to_market(date(2023, 6, 1), {"TICK": nan_frame})

        assert result["stale_marks"] == ["TICK"]
        assert result["marks"]["TICK"] == 100.0

    def test_the_equity_curve_reconciles_to_cash_plus_marks_exactly(self):
        """Scripted sequence: two tranches, a partial exit, a mark — the
        final point must equal `cash + positions_value` exactly, no
        `pytest.approx`, because nothing in this module rounds.
        """
        ledger = Ledger(config=CONFIG)
        d1, d2, d3 = date(2023, 1, 2), date(2023, 4, 3), date(2023, 7, 3)

        ledger.buy("TICK", "core", bar(d1, 100.0), CONFIG)
        ledger.buy("TICK", "core", bar(d2, 120.0), CONFIG)
        settlements = ledger.sell("TICK", 0.4, bar(d3, 130.0), "scripted-partial-exit", CONFIG)
        assert settlements

        mark_date = date(2023, 7, 5)
        price_frame = pd.DataFrame({
            "date": [d1, d2, d3, mark_date],
            "close": [100.0, 120.0, 130.0, 135.0],
            "adj_close": [100.0, 120.0, 130.0, 135.0],
        })

        result = ledger.mark_to_market(mark_date, {"TICK": price_frame})

        remaining_qty = sum(lot["qty"] for lot in ledger.positions["TICK"])
        expected_positions_value = remaining_qty * 135.0

        assert result["cash"] == ledger.cash
        assert result["positions_value"] == expected_positions_value
        assert result["total_value"] == result["cash"] + result["positions_value"]
        assert result["total_value"] == ledger.cash + expected_positions_value
        assert result["basis"] == BASIS_MODELED_CAPITAL


# ── insufficient cash ──────────────────────────────────────────────────


class TestInsufficientCash:
    def test_insufficient_cash_refuses_cleanly_with_no_partial_mutation(self):
        # Deliberately aggressive sizing (a whole sleeve in one tranche) so
        # the notional alone consumes all of `cash`, and slippage on top
        # pushes the total cost just over what is available.
        tight_config = {
            **CONFIG,
            "portfolio": {
                "sleeve_split": {"core": 1.0},
                "tranche_size_pct": {"core": 1.0},
            },
        }
        ledger = Ledger(config=tight_config)

        result = ledger.buy("TICK", "core", bar(date(2023, 1, 2), 100.0), tight_config)

        assert result["filled"] is False
        assert "insufficient" in result["reason"].lower()
        assert result["basis"] == BASIS_MODELED_CAPITAL
        # No partial mutation: cash and positions are exactly as they were.
        assert ledger.cash == 100.0
        assert ledger.positions == {}
