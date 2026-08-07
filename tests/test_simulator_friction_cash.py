"""`simulator.friction_cash` — the `friction:` regime applied to traded
notional (U5, KTD5).

Two things are being proven, and they pull in opposite directions on
purpose. First, **regime consistency**: this module and
`lifecycle.friction.compute_net_return` must agree on tax bracket, the rate
that bracket applies, the `ltcg_holding_days` boundary (`>=`, not `>`),
slippage-before-tax ordering, and "a loss goes untaxed" — because both read
the exact same `friction.config_from` settings and are meant to be the same
regime expressed two ways. Second, **level disagreement**: a transform on a
return percentage and a charge on traded notional are different quantities,
and the plan (KTD5) is explicit that asserting they produce the *same net
number* would mean one of them stopped doing what it claims to do. The
+100%-gross case is checked in literally so a future change to either module
has to confront the 1.5pp-vs-1.0pp arithmetic rather than silently
reconciling it away.

`SETTINGS` mirrors `tests/test_friction.py`'s own convention: the shipped
regime restated as a literal, so a `config.yaml` edit shows up as a failing
arithmetic assertion here rather than as silently different numbers under
the same expectations.
"""

import pytest

from boundless100x.lifecycle import friction
from boundless100x.simulator import friction_cash as fc

SETTINGS = {
    "stcg_pct": 20.0,
    "ltcg_pct": 12.5,
    "ltcg_holding_days": 365,
    "slippage_bps": 100,
}


# ── hand-computed settlement ──────────────────────────────────────────────


class TestHandComputedSettlement:
    def test_a_known_lot_settles_to_the_hand_worked_figures(self):
        """qty=100, entry=100, exit=150, holding_days=400 (past the LTCG
        line), default 100bps slippage, LTCG at 12.5%.

        By hand:
          gross proceeds   = 100 * 150             = 15000
          exit slippage    = 15000 * (100/2)/10000 = 75.0   (half the round
                              trip bps, on the exit leg's own notional)
          post-slippage    = 15000 - 75             = 14925
          cost basis       = 100 * 100              = 10000
          gain             = 14925 - 10000           = 4925   (positive, so
                              taxed; holding_days=400 >= 365 -> LTCG)
          tax              = 4925 * 0.125            = 615.625
          net proceeds     = 14925 - 615.625         = 14309.375
        """
        lot = {"qty": 100, "entry_price": 100, "holding_days": 400}

        settled = fc.settle_sale(lot, 150, SETTINGS)

        assert settled["slippage"] == pytest.approx(75.0)
        assert settled["gain"] == pytest.approx(4925.0)
        assert settled["regime"] == "ltcg"
        assert settled["tax_pct"] == pytest.approx(12.5)
        assert settled["taxed"] is True
        assert settled["tax"] == pytest.approx(615.625)
        assert settled["proceeds"] == pytest.approx(14309.375)
        # Inputs travel back untouched, so the reading is self-explanatory.
        assert settled["holding_days"] == 400
        assert settled["qty"] == 100
        assert settled["entry_price"] == 100
        assert settled["exit_price"] == 150


# ── regime consistency against the return-percentage transform ───────────


class TestRegimeConsistency:
    """Same rates, same boundary, same ordering, same loss rule — never the
    same net *number*, because a percentage transform and a cash charge are
    different quantities (KTD5). Each test below constructs a lot whose
    `(exit/entry - 1) * 100` equals the gross return handed to
    `compute_net_return`, so the two paths are compared on the same trade.
    """

    def test_tax_regime_and_rate_agree_at_a_representative_return(self):
        lot = {"qty": 100, "entry_price": 100, "holding_days": 400}
        cash = fc.settle_sale(lot, 180, SETTINGS)  # +80% gross
        transform = friction.compute_net_return(80.0, 400, SETTINGS)

        assert cash["regime"] == transform["tax_regime"] == "ltcg"
        assert cash["tax_pct"] == transform["tax_pct"] == pytest.approx(12.5)
        assert cash["taxed"] is transform["taxed"] is True

    @pytest.mark.parametrize(
        "holding_days,expected_regime",
        [
            (364, "stcg"),  # one day short of the line
            (365, "ltcg"),  # exactly at the line: LTCG, not STCG (>=)
            (366, "ltcg"),  # one day past the line
        ],
    )
    def test_the_ltcg_boundary_is_at_or_beyond_in_both_paths(
        self, holding_days, expected_regime
    ):
        lot = {"qty": 100, "entry_price": 100, "holding_days": holding_days}
        cash = fc.settle_sale(lot, 180, SETTINGS)  # +80% gross, away from
        # any slippage-driven sign flip so the boundary is isolated
        transform = friction.compute_net_return(80.0, holding_days, SETTINGS)

        assert cash["regime"] == expected_regime
        assert transform["tax_regime"] == expected_regime
        assert cash["tax_pct"] == transform["tax_pct"]

    def test_slippage_flips_a_thin_gross_gain_to_a_net_loss_in_both_paths(self):
        """entry=100, exit=100.5 is +0.5% gross — thinner than the 1.00pp /
        0.50% (per leg) slippage bite, so both paths must see the flip and
        neither may tax the (nominal, pre-slippage) gain.

        Slippage only ever subtracts, so the only sign flip physically
        possible is gain-to-loss; there is no symmetric loss-to-gain case to
        construct, which is itself part of what "slippage before tax" means.
        """
        lot = {"qty": 100, "entry_price": 100, "holding_days": 400}
        cash = fc.settle_sale(lot, 100.5, SETTINGS)
        transform = friction.compute_net_return(0.5, 400, SETTINGS)

        assert cash["gain"] < 0
        assert transform["after_slippage_pct"] < 0
        assert cash["taxed"] is False
        assert transform["taxed"] is False
        assert cash["tax"] == 0.0

    def test_a_straight_loss_is_untaxed_in_both_paths(self):
        lot = {"qty": 100, "entry_price": 100, "holding_days": 400}
        cash = fc.settle_sale(lot, 80, SETTINGS)  # -20% gross
        transform = friction.compute_net_return(-20.0, 400, SETTINGS)

        assert cash["gain"] < 0
        assert cash["taxed"] is False
        assert cash["tax"] == 0.0
        assert transform["taxed"] is False


# ── level disagreement: the documented +100% worked example ──────────────


class TestLevelDisagreement:
    def test_the_plus_100_percent_case_costs_1_5pp_notional_vs_1_0pp_transform(self):
        """KTD5's own worked example (plan lines ~465-470), checked in as a
        literal.

        entry_price=100, exit_price=200 (a +100% gross move), qty=100,
        default 100bps slippage.

        Notional path (this module):
          entry notional  = 100 * 100 = 10000
          entry slippage  = cost_of_buy(10000)      = 10000*50/10000 = 50.0
          exit notional   = 100 * 200 = 20000 (the GROWN position)
          exit slippage   = settle_sale(...)["slippage"] = 20000*50/10000 = 100.0
          total slippage  = 150.0
          as pp of the original 10000 notional: 150.0 / 10000 * 100 = 1.5pp

        Transform path (`compute_net_return`):
          slippage_bps=100 -> a flat 1.00 percentage-point deduction off the
          gross return, regardless of the return's size:
          100.0 - after_slippage_pct == 1.0

        1.5 != 1.0: the two models share the regime (same rates, same
        boundary, same ordering, same loss rule — proven above) and diverge
        on level exactly as KTD5 predicts, because the exit leg's notional in
        the cash model is the grown position while the transform's deduction
        never scales with the return.
        """
        entry_notional = 100 * 100
        lot = {"qty": 100, "entry_price": 100, "holding_days": 400}

        entry_slippage = fc.cost_of_buy(entry_notional, SETTINGS)
        settled = fc.settle_sale(lot, 200, SETTINGS)
        total_notional_slippage = entry_slippage + settled["slippage"]
        notional_cost_pp = total_notional_slippage / entry_notional * 100

        transform = friction.compute_net_return(100.0, 400, SETTINGS)
        transform_cost_pp = 100.0 - transform["after_slippage_pct"]

        assert notional_cost_pp == pytest.approx(1.5)
        assert transform_cost_pp == pytest.approx(1.0)
        assert notional_cost_pp != pytest.approx(transform_cost_pp)


# ── the double-charge guard ───────────────────────────────────────────────


class TestRoundTripSlippageNeverDoubles:
    def test_a_flat_position_pays_exactly_slippage_bps_of_notional_once(self):
        """exit_price == entry_price: no gain, no loss on price alone — the
        only cash that moves is slippage, and it must total the CONFIGURED
        round-trip figure exactly once, never twice (the halving's whole
        reason to exist).
        """
        notional = 100 * 100  # qty=100, entry_price=100
        lot = {"qty": 100, "entry_price": 100, "holding_days": 10}

        entry_slippage = fc.cost_of_buy(notional, SETTINGS)
        settled = fc.settle_sale(lot, 100, SETTINGS)  # flat exit

        total = entry_slippage + settled["slippage"]
        expected = notional * SETTINGS["slippage_bps"] / 10_000.0

        assert total == expected  # exact, not approx: both sides are
        # products/quotients of the same round numbers and must not drift by
        # even a rounding hair — this module rounds nothing (see its
        # docstring) precisely so this equality holds bit-for-bit.
        assert entry_slippage == pytest.approx(settled["slippage"])  # same
        # notional in and out on a flat trade, so each leg's half-charge is
        # identical

    def test_a_flat_position_shows_a_small_loss_from_exit_slippage_alone(self):
        """The exit leg's slippage is charged inside `settle_sale` and does
        reduce cash below the (unchanged) cost basis — that loss is real and
        expected, and it must not be taxed.
        """
        lot = {"qty": 100, "entry_price": 100, "holding_days": 10}
        settled = fc.settle_sale(lot, 100, SETTINGS)

        assert settled["gain"] == pytest.approx(-50.0)  # -(100*100*50/10000)
        assert settled["taxed"] is False
        assert settled["tax"] == 0.0


# ── config edits flow through with no code change ─────────────────────────


class TestConfigFlowsThrough:
    def test_a_non_default_config_changes_both_functions_identically(self):
        custom = {
            "stcg_pct": 30.0,
            "ltcg_pct": 15.0,
            "ltcg_holding_days": 200,
            "slippage_bps": 50,
        }
        notional = 100 * 100

        # slippage_bps: 50 is HALF the default, so the cash slippage cost
        # halves too (50 bps round trip -> 25 bps per leg).
        default_cost = fc.cost_of_buy(notional, SETTINGS)
        custom_cost = fc.cost_of_buy(notional, custom)
        assert custom_cost == pytest.approx(default_cost / 2)

        # ltcg_holding_days: 200 pulls a 250-day holding into LTCG, where
        # the default (365) would still call it STCG.
        lot = {"qty": 100, "entry_price": 100, "holding_days": 250}
        assert fc.settle_sale(lot, 150, SETTINGS)["regime"] == "stcg"
        assert fc.settle_sale(lot, 150, custom)["regime"] == "ltcg"

        # stcg_pct / ltcg_pct: the custom LTCG rate (15%) applies, not the
        # default (12.5%).
        settled = fc.settle_sale(lot, 150, custom)
        assert settled["tax_pct"] == pytest.approx(15.0)


# ── edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_quantity_settles_to_all_zeros_with_no_error(self):
        lot = {"qty": 0, "entry_price": 100, "holding_days": 400}
        settled = fc.settle_sale(lot, 150, SETTINGS)

        assert settled["slippage"] == 0.0
        assert settled["gain"] == 0.0
        assert settled["taxed"] is False
        assert settled["tax"] == 0.0
        assert settled["proceeds"] == 0.0

    def test_zero_holding_days_is_short_term(self):
        """An exit on the same bar as the entry: 0 >= ltcg_holding_days is
        False for any positive boundary, so it must read STCG, not error or
        default to LTCG.
        """
        lot = {"qty": 100, "entry_price": 100, "holding_days": 0}
        settled = fc.settle_sale(lot, 150, SETTINGS)

        assert settled["regime"] == "stcg"
        assert settled["tax_pct"] == pytest.approx(20.0)

    def test_cost_of_buy_on_zero_notional_is_zero(self):
        assert fc.cost_of_buy(0, SETTINGS) == 0.0

    def test_config_none_falls_back_to_the_shipped_defaults(self):
        """No config supplied at all -> `friction.config_from`'s own
        defaults, not a silent local copy of them (there is no local copy in
        this module — see the docstring's KTD5 section).
        """
        assert fc.cost_of_buy(10000, None) == pytest.approx(
            fc.cost_of_buy(10000, SETTINGS)
        )
