"""Friction: what a modeled exit keeps after tax and slippage (§8.2, R5).

The rule this unit carries is KTD7's, and it is as much about language as it is
about arithmetic. Every input here is a proxy — a `probe` confirmation date
rather than a fill, a market bar rather than a trade price, no cost basis
anywhere — so nothing this module produces is a *realized* return, and nothing
that renders it may imply otherwise. Two kinds of assertion carry that: the
arithmetic reproduces a hand-computed figure exactly, and every reading states
the basis it was computed on and never uses the forbidden word.

The second theme is bar selection, because lifecycle timestamps land on
Saturdays and a price series has no Saturday. Each direction rounds
conservatively — entry forward to the first bar a confirmed buy could have
used, exit back to the last bar on or before the exit date — and an empty range
reads unavailable *with its reason* rather than as a nearest-neighbour guess,
which is the same discipline `lifecycle.evaluator` applies to an unknown
elapsed time.

The third is the documented `adj_close` pitfall: a jugaad-data fallback aliases
`adj_close` to `close` and marks it `adj_close_is_estimated`, and the adjusted
column can trail the raw one by a bar. Both are dropped before any bar is
selected, so a split never reads as a crash and a trailing empty bar never
reads as a total loss.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from boundless100x.lifecycle import friction
from boundless100x.lifecycle.advance import advance_ticker
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.service import load_config
from boundless100x.watchlist import WatchlistManager
from tests.test_lifecycle_advance import (
    StubService,
    fast_lane_entry,
    fast_lane_metrics,
    metric,
)

# The shipped regime, restated here so a config edit that changes the owner's
# tax assumptions shows up as a failing arithmetic test rather than as silently
# different numbers under the same expectations.
SETTINGS = {
    "stcg_pct": 20.0,
    "ltcg_pct": 12.5,
    "ltcg_holding_days": 365,
    "slippage_bps": 100,
}


def price_frame(dates, closes, estimated=None, adjusted=True) -> pd.DataFrame:
    """A price series in the shape `price_volume.csv` actually carries."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "close": list(closes),
        "volume": [100_000] * len(list(dates)),
    })
    if adjusted:
        frame["adj_close"] = list(closes)
        if estimated is not None:
            frame["adj_close_is_estimated"] = list(estimated)
    return frame


# Two full trading weeks in March 2026: Mon 2nd–Fri 6th, Mon 9th–Fri 13th.
# Sat 7th / Sun 8th and Sat 14th / Sun 15th are the gaps a lifecycle timestamp
# can land in.
TRADING_FORTNIGHT = [
    "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06",
    "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
]


def fortnight_frame() -> pd.DataFrame:
    """One bar per trading day, each priced at its own day-of-month.

    Pricing each bar distinctly is what lets a test say *which* bar was chosen
    rather than only that some bar was.
    """
    return price_frame(
        TRADING_FORTNIGHT,
        [float(d.split("-")[-1]) for d in TRADING_FORTNIGHT],
    )


class TestTaxRegime:
    """STCG or LTCG by holding period, against the configured line."""

    def test_a_position_held_under_the_ltcg_line_is_taxed_at_stcg(self):
        reading = friction.compute_net_return(50.0, 364, SETTINGS)

        assert reading["tax_regime"] == "stcg"
        # 50.0 gross, less 100bps (1.00pp) round-trip slippage = 49.0,
        # taxed at 20%: 49.0 * 0.80 = 39.2
        assert reading["net_return_pct"] == pytest.approx(39.2)

    def test_a_position_held_at_the_line_is_taxed_at_ltcg(self):
        """At or beyond, not beyond — the boundary day is long-term."""
        reading = friction.compute_net_return(50.0, 365, SETTINGS)

        assert reading["tax_regime"] == "ltcg"
        # 49.0 after slippage, taxed at 12.5%: 49.0 * 0.875 = 42.875
        assert reading["net_return_pct"] == pytest.approx(42.875)

    def test_the_line_is_configuration_not_a_literal(self):
        """§8.2's own instruction: the regime is the owner's to edit."""
        settings = dict(SETTINGS, ltcg_holding_days=730)
        assert friction.compute_net_return(50.0, 400, settings)["tax_regime"] == "stcg"


class TestSlippage:
    def test_slippage_reduces_the_net_return_below_what_tax_alone_would(self):
        frictionless = friction.compute_net_return(
            50.0, 400, dict(SETTINGS, slippage_bps=0)
        )
        with_slippage = friction.compute_net_return(50.0, 400, SETTINGS)

        assert with_slippage["net_return_pct"] < frictionless["net_return_pct"]

    def test_a_gain_always_nets_below_its_gross(self):
        for gross in (0.5, 12.0, 50.0, 300.0):
            reading = friction.compute_net_return(gross, 400, SETTINGS)
            assert reading["net_return_pct"] < reading["gross_return_pct"]

    def test_a_loss_is_not_taxed(self):
        """Capital-gains behaviour, not a hardcoded floor: a loss pays slippage
        only. Taxing it would invent a rupee cost the owner never bore."""
        reading = friction.compute_net_return(-20.0, 400, SETTINGS)

        assert reading["net_return_pct"] == pytest.approx(-21.0)
        assert reading["taxed"] is False
        # The regime is still stated — which bracket it *would* have been is
        # part of reading the estimate, and silence would look like a gap.
        assert reading["tax_regime"] == "ltcg"

    def test_a_gain_slippage_wipes_out_is_not_taxed_either(self):
        reading = friction.compute_net_return(
            0.5, 400, dict(SETTINGS, slippage_bps=100)
        )
        assert reading["net_return_pct"] == pytest.approx(-0.5)
        assert reading["taxed"] is False


class TestNonNumbers:
    """An unreadable input is unavailable with its reason, never a figure."""

    @pytest.mark.parametrize("gross", [None, "50%", float("nan"), float("inf"), True])
    def test_a_gross_that_is_not_a_finite_number_is_refused(self, gross):
        reading = friction.compute_net_return(gross, 400, SETTINGS)
        assert reading["available"] is False
        assert reading["reason"]

    @pytest.mark.parametrize("days", [None, "400", float("nan"), float("inf")])
    def test_a_holding_period_that_is_not_a_finite_number_is_refused(self, days):
        reading = friction.compute_net_return(50.0, days, SETTINGS)
        assert reading["available"] is False

    def test_an_unusable_setting_falls_back_to_the_shipped_default(self):
        """Loudly, and the reading states the rate it actually applied — a
        malformed setting must not become a different tax rate in silence."""
        reading = friction.compute_net_return(
            50.0, 400, dict(SETTINGS, ltcg_pct="twelve and a half")
        )

        assert reading["tax_pct"] == friction.DEFAULT_LTCG_PCT
        assert reading["net_return_pct"] == pytest.approx(42.875)


class TestBarSelection:
    def test_a_saturday_entry_uses_the_following_monday(self):
        """A confirmed buy cannot predate its own confirmation, so entry rounds
        forward."""
        reading = friction.compute_position_return(
            fortnight_frame(), date(2026, 3, 7), date(2026, 3, 13)
        )

        assert reading["available"] is True
        assert reading["entry_date"] == "2026-03-09"
        assert reading["entry_price"] == pytest.approx(9.0)

    def test_a_sunday_exit_uses_the_preceding_friday(self):
        reading = friction.compute_position_return(
            fortnight_frame(), date(2026, 3, 2), date(2026, 3, 8)
        )

        assert reading["available"] is True
        assert reading["exit_date"] == "2026-03-06"
        assert reading["exit_price"] == pytest.approx(6.0)

    def test_an_entry_past_the_last_bar_is_unavailable_with_its_reason(self):
        """Not the nearest earlier bar: that would price a position the series
        cannot see, and price it at a bar from before the position existed."""
        reading = friction.compute_position_return(
            fortnight_frame(), date(2026, 4, 1), date(2026, 4, 30)
        )

        assert reading["available"] is False
        assert "on or after" in reading["reason"]
        assert "2026-04-01" in reading["reason"]

    def test_an_exit_before_the_first_bar_is_unavailable_with_its_reason(self):
        reading = friction.compute_position_return(
            fortnight_frame(), date(2026, 1, 1), date(2026, 1, 15)
        )

        assert reading["available"] is False
        assert "on or before" in reading["reason"]

    def test_an_exit_before_the_entry_bar_is_unavailable(self):
        reading = friction.compute_position_return(
            fortnight_frame(), date(2026, 3, 11), date(2026, 3, 4)
        )

        assert reading["available"] is False
        assert reading["reason"]

    def test_an_empty_series_is_unavailable_with_its_reason(self):
        assert friction.compute_position_return(
            None, date(2026, 3, 2), date(2026, 3, 13)
        )["available"] is False
        assert friction.compute_position_return(
            pd.DataFrame(), date(2026, 3, 2), date(2026, 3, 13)
        )["available"] is False

    def test_an_unreadable_date_is_unavailable_rather_than_assumed(self):
        reading = friction.compute_position_return(
            fortnight_frame(), "not a date", date(2026, 3, 13)
        )
        assert reading["available"] is False


class TestEstimatedAdjustedClose:
    """The documented `adj_close` / `adj_close_is_estimated` pitfall."""

    def test_rows_with_an_estimated_adjusted_close_are_skipped(self):
        """A jugaad-data fallback aliases `adj_close` to `close`; selecting a
        bar off it would read a split as a crash."""
        estimated = [d in ("2026-03-09", "2026-03-13") for d in TRADING_FORTNIGHT]
        frame = price_frame(
            TRADING_FORTNIGHT,
            [float(d.split("-")[-1]) for d in TRADING_FORTNIGHT],
            estimated=estimated,
        )

        reading = friction.compute_position_return(
            frame, date(2026, 3, 7), date(2026, 3, 13)
        )

        # Mon 9th and Fri 13th are aliased, so the first usable bar on or after
        # Sat 7th is Tue 10th and the last on or before Fri 13th is Thu 12th.
        assert reading["entry_date"] == "2026-03-10"
        assert reading["exit_date"] == "2026-03-12"

    def test_a_wholly_estimated_series_is_unavailable_rather_than_measured(self):
        frame = price_frame(
            TRADING_FORTNIGHT,
            [float(d.split("-")[-1]) for d in TRADING_FORTNIGHT],
            estimated=[True] * len(TRADING_FORTNIGHT),
        )

        reading = friction.compute_position_return(
            frame, date(2026, 3, 2), date(2026, 3, 13)
        )
        assert reading["available"] is False

    def test_an_empty_adjusted_close_bar_is_skipped(self):
        """The source publishes the raw close before the adjusted one, so a
        freshly fetched series routinely ends in an empty `adj_close`."""
        closes = [float(d.split("-")[-1]) for d in TRADING_FORTNIGHT]
        frame = price_frame(TRADING_FORTNIGHT, closes)
        frame.loc[frame.index[-1], "adj_close"] = float("nan")

        reading = friction.compute_position_return(
            frame, date(2026, 3, 2), date(2026, 3, 13)
        )
        assert reading["exit_date"] == "2026-03-12"

    def test_a_legacy_series_without_an_adjusted_column_still_reads(self):
        frame = price_frame(
            TRADING_FORTNIGHT,
            [float(d.split("-")[-1]) for d in TRADING_FORTNIGHT],
            adjusted=False,
        )

        reading = friction.compute_position_return(
            frame, date(2026, 3, 2), date(2026, 3, 13)
        )
        assert reading["available"] is True
        assert reading["price_series"] == "close"


class TestHandComputedVerification:
    """The unit's verification scenario: a known delta, derived by hand."""

    def test_a_known_delta_and_holding_period_reproduce_the_arithmetic(self):
        # 401 calendar bars, flat at 100.00 until the last, which prints 150.00.
        # Entry bar 2024-01-01 at 100.00; exit bar 2025-02-04 at 150.00.
        dates = pd.date_range("2024-01-01", periods=401, freq="D")
        closes = [100.0] * 400 + [150.0]

        position = friction.compute_position_return(
            price_frame(dates, closes), date(2024, 1, 1), date(2025, 2, 4)
        )

        #   gross  = (150.00 / 100.00 - 1) * 100                    = 50.0%
        #   days   = 2025-02-04 - 2024-01-01                        = 400
        assert position["gross_return_pct"] == pytest.approx(50.0)
        assert position["holding_days"] == 400

        net = friction.compute_net_return(
            position["gross_return_pct"], position["holding_days"], SETTINGS
        )

        #   slippage = 100bps round trip                            = 1.00pp
        #   after    = 50.0 - 1.0                                   = 49.0%
        #   400 days >= the 365-day line, so LTCG at 12.5% applies:
        #   net      = 49.0 * (1 - 0.125)                           = 42.875%
        assert net["tax_regime"] == "ltcg"
        assert net["net_return_pct"] == pytest.approx(42.875)


class TestShippedConfig:
    def test_the_config_block_ships_the_four_settings(self):
        config = load_config()

        # Asserted on the raw config, not only on the resolved settings: the
        # defaults mirror the shipped block, so a missing block would resolve
        # to identical numbers and this test would pass on a file that never
        # gave the owner anything to edit.
        assert "friction" in config
        settings = friction.config_from(config)

        assert settings["stcg_pct"] == 20.0
        assert settings["ltcg_pct"] == 12.5
        assert settings["ltcg_holding_days"] == 365
        assert settings["slippage_bps"] == 100

    def test_an_absent_block_falls_back_to_the_shipped_defaults(self):
        assert friction.config_from({}) == friction.config_from(
            {"friction": {}}
        )


class TestLanguage:
    """KTD7: nothing here may read as a statement about actual trades."""

    def test_a_reading_is_described_as_a_modeled_estimate(self):
        reading = friction.model_exit(
            fortnight_frame(), date(2026, 3, 2), date(2026, 3, 13), SETTINGS
        )
        text = friction.describe(reading)

        assert "realiz" not in text.lower()
        assert "realis" not in text.lower()
        assert "estimate" in text.lower() or "modeled" in text.lower()

    def test_an_unavailable_reading_describes_why(self):
        reading = friction.model_exit(
            fortnight_frame(), date(2026, 5, 1), date(2026, 5, 30), SETTINGS
        )
        text = friction.describe(reading)

        assert reading["available"] is False
        assert "realiz" not in text.lower()
        assert reading["reason"] in text

    def test_no_string_the_module_can_emit_says_realized(self):
        """A grep with teeth: every literal that could reach a reader.

        Docstrings are exempt — the prohibition has to be *explained*
        somewhere, and prose about the rule is not a claim about trades.
        Comments never reach a reader at all. Everything else, including the
        literal parts of f-strings, is checked."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(friction.__file__).read_text())
        docstrings = {
            ast.get_docstring(node, clean=False)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and ast.get_docstring(node, clean=False) is not None
        }
        offenders = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in docstrings
            and "realiz" in node.value.lower()
        ]

        assert offenders == []


class TestBasis:
    def test_a_proposal_time_reading_is_an_estimate(self):
        reading = friction.model_exit(
            fortnight_frame(), date(2026, 3, 2), date(2026, 3, 13), SETTINGS
        )
        assert reading["basis"] == "estimate"

    def test_a_recorded_reading_says_so(self):
        """`recorded` means the dates are fixed, not that it stopped being a
        model. U6's exit command is its only writer."""
        reading = friction.model_exit(
            fortnight_frame(), date(2026, 3, 2), date(2026, 3, 13), SETTINGS,
            basis=friction.BASIS_RECORDED,
        )
        assert reading["basis"] == "recorded"

    def test_an_unavailable_reading_still_carries_its_basis(self):
        reading = friction.model_exit(
            fortnight_frame(), date(2026, 5, 1), date(2026, 5, 30), SETTINGS
        )
        assert reading["basis"] == "estimate"


# ── the advance() seam ────────────────────────────────────────────────────


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


@pytest.fixture
def evaluator():
    return TriggerEvaluator(load_triggers())


def probed_position(wm, ticker="ZENSAR"):
    """A fast-lane holding that actually entered `probe`, and when it did."""
    fast_lane_entry(wm, ticker=ticker, state="probe")
    return date.fromisoformat(wm.get(ticker)["state_history"][-1]["at"][:10])


def stepped_price(entered: date, days: int = 500) -> pd.DataFrame:
    """Flat at 100.00 from the probe date, stepping to 150.00 a week later.

    Built off the *recorded* probe date rather than a fixed calendar, because
    `watchlist.transition` timestamps with the wall clock — a fixed fixture
    would price a position the series had never seen on most days of the year.
    """
    dates = pd.date_range(entered, periods=days, freq="D")
    closes = [100.0] * 7 + [150.0] * (days - 7)
    return price_frame(dates, closes)


def exiting_service(entered: date):
    metrics = fast_lane_metrics(
        rerating_headroom=metric(2.0, flags=["rerating_headroom_stretched"])
    )
    return StubService(metrics=metrics, data={"price": stepped_price(entered)})


class TestAdvanceAttachesTheReading:
    def test_an_exit_proposal_carries_a_friction_reading(self, wm, evaluator):
        entered = probed_position(wm)
        outcome = advance_ticker(
            exiting_service(entered), wm, "ZENSAR", evaluator,
            as_of=entered + timedelta(days=400),
        )

        proposal = outcome["proposal"]
        assert proposal["to"] == "exit_review"
        assert proposal["friction"]["available"] is True
        assert proposal["friction"]["basis"] == "estimate"

    def test_the_reading_reproduces_the_hand_computed_net(self, wm, evaluator):
        """100.00 -> 150.00 is 50.0% gross; less 1.00pp slippage is 49.0%;
        held past the 365-day line, LTCG at 12.5% leaves 49.0 * 0.875 =
        42.875%."""
        entered = probed_position(wm)
        outcome = advance_ticker(
            exiting_service(entered), wm, "ZENSAR", evaluator,
            as_of=entered + timedelta(days=400),
        )

        reading = outcome["proposal"]["friction"]
        assert reading["gross_return_pct"] == pytest.approx(50.0)
        assert reading["tax_regime"] == "ltcg"
        assert reading["net_return_pct"] == pytest.approx(42.875)

    def test_gross_and_net_travel_together_into_the_recorded_evidence(
        self, wm, evaluator
    ):
        """R5: recorded evidence carries net beside gross, never one alone."""
        entered = probed_position(wm)
        outcome = advance_ticker(
            exiting_service(entered), wm, "ZENSAR", evaluator, apply=True,
            as_of=entered + timedelta(days=400),
        )

        evidence = wm.get("ZENSAR")["state_history"][-1]["evidence"]
        assert "50.0" in evidence and "42.9" in evidence
        assert "realiz" not in evidence.lower()
        assert outcome["proposal"]["to"] == "exit_review"

    def test_a_company_that_never_entered_probe_gets_no_reading(self, wm, evaluator):
        """Not a raise, and not a zero: there is no modeled position to price."""
        fast_lane_entry(wm, ticker="ZENSAR", state="scale")
        service = StubService(
            metrics=fast_lane_metrics(
                rerating_headroom=metric(2.0, flags=["rerating_headroom_stretched"])
            ),
            data={"price": stepped_price(date(2024, 1, 1))},
        )

        outcome = advance_ticker(service, wm, "ZENSAR", evaluator)

        assert outcome["proposal"]["to"] == "exit_review"
        assert "friction" not in outcome["proposal"]

    def test_a_missing_price_series_reads_unavailable_with_its_reason(
        self, wm, evaluator
    ):
        """A position exists but cannot be priced — a different fact from no
        position at all, and one worth saying out loud."""
        entered = probed_position(wm)
        service = StubService(
            metrics=fast_lane_metrics(
                rerating_headroom=metric(2.0, flags=["rerating_headroom_stretched"])
            ),
            data={},
        )

        outcome = advance_ticker(
            service, wm, "ZENSAR", evaluator, as_of=entered + timedelta(days=400)
        )

        reading = outcome["proposal"]["friction"]
        assert reading["available"] is False
        assert reading["reason"]

    def test_the_cli_shows_net_beside_gross_with_its_assumptions(self):
        """R5 at the surface a person actually reads."""
        from boundless100x import cli

        reading = friction.model_exit(
            fortnight_frame(), date(2026, 3, 2), date(2026, 3, 13), SETTINGS
        )
        with cli.console.capture() as captured:
            cli._print_exit_friction(
                [{"ticker": "ZENSAR", "proposal": {"friction": reading}}]
            )
        text = captured.get()

        assert "gross" in text and "net" in text
        assert "estimate" in text.lower()
        assert "probe" in text  # the holding period's stated origin
        assert "realiz" not in text.lower()

    def test_the_evidence_cell_survives_the_markup_parser(self):
        """The bracketed summary is text, not a rich style.

        Rich reads `[gross +50.0% ...]` as a markup tag and renders nothing at
        all, so an unescaped evidence cell drops the very figures R5 requires —
        silently, and only in the column a person reads. The stored evidence
        stays byte-identical; only the rendering escapes.
        """
        from rich.console import Console
        from rich.table import Table

        from boundless100x import cli

        evidence = (
            "Re-rating target reached fired "
            "[gross +50.0% / net +42.9% (modeled estimate)]"
        )
        table = Table()
        table.add_column("Evidence")
        table.add_row(cli._evidence_cell(evidence))

        console = Console(width=200)
        with console.capture() as captured:
            console.print(table)

        assert "net +42.9%" in captured.get()

    def test_the_cli_shows_why_a_reading_was_unavailable(self):
        from boundless100x import cli

        reading = friction.model_exit(
            fortnight_frame(), date(2026, 5, 1), date(2026, 5, 30), SETTINGS
        )
        with cli.console.capture() as captured:
            cli._print_exit_friction(
                [{"ticker": "ZENSAR", "proposal": {"friction": reading}}]
            )

        assert "unavailable" in captured.get().lower()

    def test_the_cli_prints_nothing_when_no_proposal_carries_a_reading(self):
        from boundless100x import cli

        with cli.console.capture() as captured:
            cli._print_exit_friction([{"ticker": "ASTRAL", "proposal": None}])

        assert captured.get() == ""

    def test_an_entry_proposal_carries_no_friction_reading(self, wm, evaluator):
        """Nothing has been bought yet, so there is no holding to model."""
        fast_lane_entry(wm, ticker="ZENSAR", state="watch")
        outcome = advance_ticker(
            StubService(metrics=fast_lane_metrics()), wm, "ZENSAR", evaluator
        )

        assert outcome["proposal"]["to"] == "probe"
        assert "friction" not in outcome["proposal"]
