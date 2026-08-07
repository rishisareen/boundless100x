"""`build_lane_context` — the one assembler both surfaces read.

A report must be able to render a position's lane, its gates and its friction
without `advance()` having run. That is the whole reason this is a pure
function taking an entry and a scored result rather than a field on an advance
outcome: `analyze` builds a report on a path that never advances anything, and
a figure only `advance` could produce would simply be missing there.

Three rules are asserted hardest, because each has an inviting wrong answer.

**An exited position reports what was recorded, never what today's bars say.**
The `exited` transition carries the friction payload as structured `details`
precisely so a reader can take it apart later; recomputing it would re-price a
sale that already happened against newer prices, and the number would drift
every time the report was regenerated. The fixture here makes the two visibly
different, so adopting and recomputing cannot both pass.

**An unreadable friction input is unavailable with its reason, never absent and
never zero.** Absent means there is no modeled position at all — nothing was
ever bought — and that is a different fact from a position nobody could price.

**An overdue catalyst is a display flag and nothing else.** §13 keeps the
system advisory: the clock feeds the time stop and nothing else, so noticing
that a window has passed must not move a company's state.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from boundless100x.lifecycle import friction as friction_module
from boundless100x.lifecycle.lane_view import build_lane_context
from boundless100x.watchlist import WatchlistManager
from tests.conftest import make_result
from tests.test_friction import price_frame

TODAY = date(2026, 8, 7)

# The six shipped lane gates, so a fixture asserting "all of them were asked"
# names them rather than counting.
LANE_GATE_IDS = {
    "quality_floor", "valuation_discount", "growth_intact",
    "institutional_accumulation", "catalyst_identified", "liquidity_floor",
}


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


# A holding period long enough to land in the long-term bracket, so the tests
# below read one regime rather than whichever one the calendar happened to give
# them. `test_confirm_exit` dates its exits the same way and for the same
# reason: a `probe` transition is stamped with the wall clock, so the only
# deterministic exit date is one measured from it.
HELD_DAYS = 400


def doubled_price(start: date, days: int = 500) -> pd.DataFrame:
    """100.00 for the first week, 150.00 thereafter — a clean +50% gross."""
    dates = pd.date_range(start, periods=days, freq="D")
    closes = [100.0] * 7 + [150.0] * (days - 7)
    return price_frame(dates, closes)


def scored(price=None):
    """A scored result with a price series a modeled position can be read off."""
    result = make_result()
    if price is not None:
        result.data["price"] = price
    return result


def tracked(wm, ticker="ASTRAL", lane="core", states=(), catalyst=None):
    """An entry walked into `states` through `transition`, so its dates are real.

    Hand-writing `state_history` would stamp a day the price fixture has never
    seen — the same reason `test_confirm_exit` builds its holdings this way.
    """
    wm.add(ticker, lane=lane)
    for state in states:
        wm.transition(ticker, state, f"{state}_trigger", evidence=f"moved to {state}")
    if catalyst:
        wm.record_catalyst(ticker, catalyst["description"], catalyst["expected_by"])
        if catalyst.get("status") == "spent":
            wm.mark_catalyst_spent(ticker)
    return wm.get(ticker)


class TestLaneAndState:
    def test_an_untracked_company_has_no_lane_context_at_all(self):
        """The CLI gates on membership; this is the same answer from inside."""
        assert build_lane_context(None, scored(), TODAY) is None

    def test_lane_and_state_are_reported_for_a_core_entry(self, wm):
        entry = tracked(wm, states=("qualify", "watch"))

        context = build_lane_context(entry, scored(), TODAY)

        assert context["lane"] == "core"
        assert context["state"] == "watch"

    def test_a_core_entry_is_never_given_lane_gates(self, wm):
        """The six gates are the fast lane's entry question, not a universal one."""
        entry = tracked(wm, states=("qualify",))

        assert build_lane_context(entry, scored(), TODAY)["lane_gates"] is None

    def test_a_rerating_entry_has_its_gates_evaluated_when_none_is_supplied(self, wm):
        entry = tracked(wm, lane="rerating", states=("qualify",))

        result = build_lane_context(entry, scored(), TODAY)["lane_gates"]

        assert set(result["gates"]) == LANE_GATE_IDS

    def test_a_supplied_gate_result_is_used_rather_than_recomputed(self, wm):
        """`advance_ticker` has already paid for this; it must not pay twice."""
        entry = tracked(wm, lane="rerating", states=("qualify",))
        already = {"verdict": "qualifies", "qualifies": True,
                   "gates": {}, "failed": [], "indeterminate": []}

        context = build_lane_context(entry, scored(), TODAY, lane_gate_result=already)

        assert context["lane_gates"] is already


class TestCatalyst:
    def test_a_catalyst_travels_with_its_window_and_status(self, wm):
        entry = tracked(wm, lane="rerating", catalyst={
            "description": "Demerger of the packaging arm",
            "expected_by": "2026-12-31",
        })

        catalyst = build_lane_context(entry, scored(), TODAY)["catalyst"]

        assert catalyst["description"] == "Demerger of the packaging arm"
        assert catalyst["expected_by"] == "2026-12-31"
        assert catalyst["status"] == "active"

    def test_an_active_catalyst_past_its_window_reads_overdue(self, wm):
        entry = tracked(wm, lane="rerating", catalyst={
            "description": "Capacity commissioning", "expected_by": "2026-01-31",
        })

        assert build_lane_context(entry, scored(), TODAY)["catalyst"]["overdue"] is True

    def test_an_overdue_catalyst_changes_no_state(self, wm):
        """§13: the clock feeds the time stop, and this is display only."""
        entry = tracked(wm, lane="rerating", states=("qualify",), catalyst={
            "description": "Capacity commissioning", "expected_by": "2026-01-31",
        })

        build_lane_context(entry, scored(), TODAY)

        assert wm.get("ASTRAL")["state"] == "qualify"
        assert [r["to"] for r in wm.get("ASTRAL")["state_history"]] == ["qualify"]

    def test_a_spent_catalyst_is_never_overdue(self, wm):
        """It happened. A window it happened after is not a warning."""
        entry = tracked(wm, lane="rerating", catalyst={
            "description": "Demerger", "expected_by": "2026-01-31",
            "status": "spent",
        })

        assert build_lane_context(entry, scored(), TODAY)["catalyst"]["overdue"] is False

    def test_a_window_still_ahead_is_not_overdue(self, wm):
        entry = tracked(wm, lane="rerating", catalyst={
            "description": "Demerger", "expected_by": "2026-12-31",
        })

        assert build_lane_context(entry, scored(), TODAY)["catalyst"]["overdue"] is False

    def test_a_window_nobody_can_parse_is_not_called_overdue(self, wm):
        """`expected_by` is free text. Unreadable is not a claim that it passed."""
        entry = tracked(wm, lane="rerating", catalyst={
            "description": "Demerger", "expected_by": "sometime in FY27",
        })

        assert build_lane_context(entry, scored(), TODAY)["catalyst"]["overdue"] is False

    def test_an_entry_with_no_catalyst_carries_none(self, wm):
        entry = tracked(wm, lane="rerating")

        assert build_lane_context(entry, scored(), TODAY)["catalyst"] is None


class TestFriction:
    @pytest.fixture
    def held(self, wm):
        """A position confirmed today, priced 400 days out at +50% gross."""
        class Held:
            entered = date.today()
            as_of = date.today() + timedelta(days=HELD_DAYS)
            price = doubled_price(date.today())

        return Held()

    def test_a_pre_position_entry_has_no_friction_reading(self, wm):
        """Nothing was bought, so there is no modeled position to price."""
        entry = tracked(wm, states=("qualify", "watch"))

        assert build_lane_context(entry, scored(), TODAY)["friction"] is None

    def test_a_positioned_entry_gets_an_estimate(self, wm, held):
        entry = tracked(wm, states=("qualify", "watch", "probe"))

        reading = build_lane_context(
            entry, scored(held.price), held.as_of
        )["friction"]

        assert reading["available"] is True
        assert reading["basis"] == friction_module.BASIS_ESTIMATE
        assert reading["gross_return_pct"] == pytest.approx(50.0)
        assert reading["holding_days"] == HELD_DAYS
        # gross 50.0 − 100bps slippage = 49.0, LTCG at 12.5% → 42.875
        assert reading["net_return_pct"] == pytest.approx(42.875)

    def test_an_exit_review_entry_gets_an_estimate_too(self, wm, held):
        entry = tracked(wm, states=("qualify", "watch", "probe", "exit_review"))

        reading = build_lane_context(
            entry, scored(held.price), held.as_of
        )["friction"]

        assert reading["available"] is True
        assert reading["basis"] == friction_module.BASIS_ESTIMATE

    def test_a_position_with_no_probe_in_its_history_has_no_reading(self, wm, held):
        """Absent, not unavailable: there is no modeled entry date to price from."""
        wm.add("ASTRAL")
        wm.transition("ASTRAL", "scale", "manual", evidence="hand-placed")

        context = build_lane_context(wm.get("ASTRAL"), scored(held.price), held.as_of)

        assert context["friction"] is None

    def test_an_unpriceable_position_is_unavailable_with_its_reason(self, wm, held):
        """A gap in the bars must never come out the far side as a zero return."""
        entry = tracked(wm, states=("qualify", "watch", "probe"))

        reading = build_lane_context(
            entry, scored(pd.DataFrame()), held.as_of
        )["friction"]

        assert reading["available"] is False
        assert reading["reason"]
        assert "gross_return_pct" not in reading

    def test_an_exited_entry_reports_the_payload_that_was_recorded(self, wm, held):
        """Not a re-price. The sale happened once, at one set of numbers."""
        recorded = {
            "available": True, "basis": friction_module.BASIS_RECORDED,
            "gross_return_pct": 12.5, "net_return_pct": 9.0,
            "holding_days": HELD_DAYS, "tax_regime": "ltcg", "tax_pct": 12.5,
            "slippage_bps": 100, "after_slippage_pct": 11.5,
        }
        tracked(wm, states=("qualify", "watch", "probe", "exit_review"))
        wm.transition("ASTRAL", "exited", "owner_confirmed",
                      evidence="sold", details=recorded)

        # The live series would price the same position at +50%, so a
        # recomputation is visibly distinguishable from an adoption.
        reading = build_lane_context(
            wm.get("ASTRAL"), scored(held.price), held.as_of
        )["friction"]

        assert reading["gross_return_pct"] == 12.5
        assert reading["basis"] == friction_module.BASIS_RECORDED

    def test_an_exit_recorded_without_a_payload_says_so_rather_than_re_pricing(
        self, wm, held
    ):
        tracked(wm, states=("qualify", "watch", "probe", "exit_review"))
        wm.transition("ASTRAL", "exited", "owner_confirmed", evidence="sold")

        reading = build_lane_context(
            wm.get("ASTRAL"), scored(held.price), held.as_of
        )["friction"]

        assert reading["available"] is False
        assert "recorded" in reading["reason"]
        assert "gross_return_pct" not in reading


class TestASuppliedEstimateIsUsedRatherThanRecomputed:
    """The same seam `lane_gate_result` already is, and for the same reason.

    On an exit-proposing ticker `advance_ticker` models this exact reading from
    this exact entry, price series and `as_of` before it builds the view — so
    modeling it again means rebuilding a frame over the whole daily series to
    reach an answer already in hand, and a *disagreement* between the two means
    the terminal's exit block and the report's lane section printing different
    net returns for one position.
    """

    @pytest.fixture
    def held(self, wm):
        class Held:
            as_of = date.today() + timedelta(days=HELD_DAYS)
            price = doubled_price(date.today())

        return Held()

    # Visibly not what the live series would produce, so adoption and
    # recomputation cannot both pass.
    ALREADY = {
        "available": True, "basis": friction_module.BASIS_ESTIMATE,
        "gross_return_pct": 3.5, "net_return_pct": 2.1, "holding_days": 40,
    }

    def test_a_positioned_entry_adopts_the_supplied_reading(self, wm, held):
        entry = tracked(wm, states=("qualify", "watch", "probe"))

        context = build_lane_context(
            entry, scored(held.price), held.as_of, friction_estimate=self.ALREADY
        )

        assert context["friction"] is self.ALREADY

    def test_a_pre_position_entry_still_reports_no_reading(self, wm, held):
        """The state dispatch runs first, and stays this module's question.

        A kill-switch may propose an exit review from a state where no capital
        is committed, so `advance` can hold an estimate for a company sitting at
        `watch`. Whether there is a position to report is decided here, not by
        whoever handed the reading in.
        """
        entry = tracked(wm, states=("qualify", "watch"))

        context = build_lane_context(
            entry, scored(held.price), held.as_of, friction_estimate=self.ALREADY
        )

        assert context["friction"] is None

    def test_an_exited_entry_still_reports_what_was_recorded(self, wm, held):
        """An estimate must never displace the payload of a sale that happened."""
        recorded = {
            "available": True, "basis": friction_module.BASIS_RECORDED,
            "gross_return_pct": 12.5, "net_return_pct": 9.0,
        }
        tracked(wm, states=("qualify", "watch", "probe", "exit_review"))
        wm.transition("ASTRAL", "exited", "owner_confirmed",
                      evidence="sold", details=recorded)

        context = build_lane_context(
            wm.get("ASTRAL"), scored(held.price), held.as_of,
            friction_estimate=self.ALREADY,
        )

        assert context["friction"]["gross_return_pct"] == 12.5

    def test_supplying_nothing_computes_the_reading_as_before(self, wm, held):
        """The CLI calls this fresh and has no estimate to hand in."""
        entry = tracked(wm, states=("qualify", "watch", "probe"))

        context = build_lane_context(entry, scored(held.price), held.as_of)

        assert context["friction"]["gross_return_pct"] == pytest.approx(50.0)


class TestTheCliGate:
    """`analyze` renders a lane section only for a company it is tracking.

    The same shape as `_record_checkpoints_if_tracked`, and for the same
    reason: an analysis the owner has already paid for must not be lost to a
    watchlist that could not be read.
    """

    @pytest.fixture
    def redirected(self, tmp_path, monkeypatch):
        from boundless100x import watchlist as watchlist_module

        path = tmp_path / "watchlist.json"
        monkeypatch.setattr(watchlist_module, "DEFAULT_WATCHLIST_PATH", path)
        return path

    def test_an_untracked_ticker_gets_no_lane_context(self, redirected):
        from boundless100x.cli import _lane_context_if_tracked

        assert _lane_context_if_tracked("ASTRAL", scored(), None) is None

    def test_a_tracked_ticker_gets_one(self, redirected):
        from boundless100x.cli import _lane_context_if_tracked

        WatchlistManager(path=str(redirected)).add("ASTRAL", lane="rerating")

        context = _lane_context_if_tracked("astral", scored(), None)

        assert context["lane"] == "rerating"
        assert context["state"] == "screen"

    def test_an_unreadable_watchlist_costs_the_section_and_nothing_else(
        self, redirected
    ):
        from boundless100x.cli import _lane_context_if_tracked

        redirected.write_text("{not json")

        assert _lane_context_if_tracked("ASTRAL", scored(), None) is None

    def test_the_terminal_line_states_the_lane_and_the_overdue_catalyst(
        self, redirected, monkeypatch, capsys
    ):
        """Gross and net travel together in the terminal too, and say `modeled`."""
        from rich.console import Console

        from boundless100x import cli, cli_common, cli_lifecycle
        from boundless100x.cli import _print_lane_status

        wide = Console(width=200)
        for module in (cli, cli_common, cli_lifecycle):
            monkeypatch.setattr(module, "console", wide)

        _print_lane_status({
            "lane": "rerating", "state": "probe",
            "catalyst": {"description": "Demerger", "expected_by": "2026-01-31",
                         "status": "active", "overdue": True},
            "friction": {
                "available": True, "basis": "estimate",
                "gross_return_pct": 48.0, "net_return_pct": 41.125,
                "holding_days": 420, "tax_regime": "ltcg", "tax_pct": 12.5,
                "slippage_bps": 100,
            },
        })

        printed = capsys.readouterr().out
        assert "rerating" in printed and "probe" in printed
        assert "Catalyst overdue" in printed and "no transition" in printed
        assert "gross +48.0%" in printed and "net +41.1%" in printed
        assert "modeled" in printed
        assert "realiz" not in printed.lower()


class TestAssumptions:
    def test_the_configured_friction_settings_travel_with_the_context(self, wm):
        """The break-even line lists them; it must not invent them."""
        entry = tracked(wm, lane="rerating", states=("qualify",))
        config = {"friction": {"stcg_pct": 30.0, "ltcg_pct": 15.0,
                               "ltcg_holding_days": 730, "slippage_bps": 250}}

        assumptions = build_lane_context(
            entry, scored(), TODAY, config=config
        )["friction_assumptions"]

        assert assumptions["stcg_pct"] == 30.0
        assert assumptions["slippage_bps"] == 250

    def test_no_config_still_states_the_shipped_rates(self, wm):
        entry = tracked(wm, lane="rerating")

        assumptions = build_lane_context(entry, scored(), TODAY)["friction_assumptions"]

        assert assumptions["stcg_pct"] == friction_module.DEFAULT_STCG_PCT
        assert assumptions["ltcg_pct"] == friction_module.DEFAULT_LTCG_PCT
