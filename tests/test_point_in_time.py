"""`point_in_time.truncate_to_date`: the shared "what was knowable on date D."

Never touches `raw_data/` — every fixture here is synthetic, built with the
same `tests/conftest.py` builders the rest of the suite uses so column names
match what the pipeline actually fetches.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.builtin._helpers import period_end_date
from boundless100x.compute_engine.point_in_time import (
    ANNUAL_REPORTING_LAG_MONTHS,
    NON_TRUNCATABLE_INPUTS,
    QUARTERLY_REPORTING_LAG_MONTHS,
    SHAREHOLDING_REPORTING_LAG_MONTHS,
    truncate_to_date,
)
from tests.conftest import (
    make_balance_sheet,
    make_cashflow,
    make_financials,
    make_metadata,
    make_price,
    make_quarterly,
    make_ratios,
    make_shareholding,
    quarter_labels,
    year_labels,
)


def _price_df(dates, closes, adj_closes=None) -> pd.DataFrame:
    df = pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100_000] * len(dates),
    })
    if adj_closes is not None:
        df["adj_close"] = adj_closes
    return df


def full_data(**overrides) -> dict:
    """A `data` dict carrying every truncatable frame, mirroring what a
    hypothetical simulator loader (not just the backtest's own `_load`,
    which never reads quarterly/shareholding) would assemble."""
    base = {
        "financials": make_financials(10),
        "balance_sheet": make_balance_sheet(10),
        "cashflow": make_cashflow(10),
        "ratios": make_ratios(10),
        "quarterly": make_quarterly(20),
        "shareholding": make_shareholding(20),
        "price": make_price(days=2600, end_close=400.0),
        "_metadata_raw": make_metadata(),
    }
    base.update(overrides)
    return base


class TestLeakage:
    """No frame row, price bar, or quarterly/shareholding row whose period
    ends after cutoff-minus-its-own-lag survives — checked per column."""

    CUTOFF = pd.Timestamp("2023-06-30")

    def test_annual_frames_exclude_every_row_past_their_lag_adjusted_boundary(self):
        data = full_data()
        truncated, reason = truncate_to_date(
            data, self.CUTOFF, non_truncatable_inputs=()
        )
        assert truncated is not None, reason

        included, excluded = 0, 0
        for name in ("financials", "balance_sheet", "cashflow", "ratios"):
            for label in year_labels(10):
                boundary = period_end_date(label) + pd.DateOffset(
                    months=ANNUAL_REPORTING_LAG_MONTHS
                )
                present = label in list(truncated[name]["year"])
                if boundary <= self.CUTOFF:
                    assert present, f"{name}/{label} should be visible"
                    included += 1
                else:
                    assert not present, f"{name}/{label} leaked past its lag"
                    excluded += 1
        assert included and excluded  # the cutoff must actually split the series

    def test_quarterly_frame_excludes_every_row_past_its_own_lag(self):
        data = full_data()
        truncated, reason = truncate_to_date(
            data, self.CUTOFF, non_truncatable_inputs=()
        )
        assert truncated is not None, reason

        included, excluded = 0, 0
        for label in quarter_labels(20):
            boundary = period_end_date(label) + pd.DateOffset(
                months=QUARTERLY_REPORTING_LAG_MONTHS
            )
            present = label in list(truncated["quarterly"]["quarter"])
            if boundary <= self.CUTOFF:
                assert present, f"quarterly/{label} should be visible"
                included += 1
            else:
                assert not present, f"quarterly/{label} leaked past its lag"
                excluded += 1
        assert included and excluded

    def test_shareholding_frame_excludes_every_row_past_its_own_lag(self):
        """Only checkable with the strip opted out — see TestShareholdingBacktestParity
        for the (unrelated) question of whether the backtest itself sees this."""
        data = full_data()
        truncated, reason = truncate_to_date(
            data, self.CUTOFF, non_truncatable_inputs=()
        )
        assert truncated is not None, reason

        included, excluded = 0, 0
        for label in quarter_labels(20):
            boundary = period_end_date(label) + pd.DateOffset(
                months=SHAREHOLDING_REPORTING_LAG_MONTHS
            )
            present = label in list(truncated["shareholding"]["quarter"])
            if boundary <= self.CUTOFF:
                assert present, f"shareholding/{label} should be visible"
                included += 1
            else:
                assert not present, f"shareholding/{label} leaked past its lag"
                excluded += 1
        assert included and excluded

    def test_price_excludes_every_bar_after_the_cutoff(self):
        data = full_data()
        truncated, reason = truncate_to_date(data, self.CUTOFF)
        assert truncated is not None, reason

        assert truncated["price"]["date"].max() <= self.CUTOFF
        assert len(truncated["price"]) < len(data["price"])  # cutoff actually bites


class TestShareholdingBacktestParity:
    """KTD2's own resolved contradiction: shareholding is truncatable, but the
    backtest's own strip stays byte-identical regardless."""

    def test_default_strip_matches_the_backtests_own_behaviour(self):
        data = full_data()
        truncated, reason = truncate_to_date(data, pd.Timestamp("2023-06-30"))
        assert truncated is not None, reason

        for leaky in NON_TRUNCATABLE_INPUTS:
            assert leaky not in truncated

    def test_opting_out_of_the_strip_keeps_a_genuinely_truncated_shareholding_frame(self):
        data = full_data()
        truncated, reason = truncate_to_date(
            data, pd.Timestamp("2023-06-30"), non_truncatable_inputs=()
        )
        assert truncated is not None, reason

        assert "shareholding" in truncated
        assert not truncated["shareholding"].empty
        assert truncated["shareholding"]["quarter"].map(period_end_date).max() < pd.Timestamp("2023-06-30")


class TestTooShallowFrames:
    """A too-shallow quarterly/shareholding frame must read as absent-with-
    reason (empty, in engine.py's own terms), never a populated frame a
    metric would mistake for "no rises"."""

    def test_quarterly_frame_entirely_after_the_cutoff_comes_back_empty(self):
        data = full_data(quarterly=make_quarterly(4))  # ends "Mar 2025"
        truncated, reason = truncate_to_date(data, pd.Timestamp("2020-01-01"))
        assert truncated is not None, reason

        assert truncated["quarterly"].empty

    def test_shareholding_frame_entirely_after_the_cutoff_comes_back_empty(self):
        data = full_data(shareholding=make_shareholding(4))  # ends "Mar 2025"
        truncated, reason = truncate_to_date(
            data, pd.Timestamp("2020-01-01"), non_truncatable_inputs=()
        )
        assert truncated is not None, reason

        assert truncated["shareholding"].empty


class TestDateAwareTruncation:
    def test_interim_row_sharing_the_cutoff_year_does_not_leak(self):
        """Screener appends part-year balance-sheet columns (e.g. "Sep 2025").
        One sharing a calendar year with the cutoff row covers a later period
        than it and must not leak in on a bare year comparison."""
        bs = make_balance_sheet(10)
        cutoff_row_label = bs.iloc[4]["year"]
        assert cutoff_row_label == "Mar 2020"
        leaking_row = bs.iloc[[4]].copy()
        leaking_row["year"] = "Sep 2020"
        bs = pd.concat([bs, leaking_row], ignore_index=True)

        cutoff_period_end = period_end_date(cutoff_row_label)
        cutoff = cutoff_period_end + pd.DateOffset(months=ANNUAL_REPORTING_LAG_MONTHS)
        data = full_data(balance_sheet=bs)

        truncated, reason = truncate_to_date(data, cutoff, annual_lag_months=ANNUAL_REPORTING_LAG_MONTHS)
        assert truncated is not None, reason
        assert "Sep 2020" not in list(truncated["balance_sheet"]["year"])
        assert "Mar 2020" in list(truncated["balance_sheet"]["year"])

    def test_cutoff_before_the_price_series_starts_yields_the_backtests_exclusion_shape(self):
        data = full_data(price=make_price(days=200, end_close=150.0))  # starts ~2015-08
        cutoff = pd.Timestamp("2014-01-01")  # before any price bar

        truncated, reason = truncate_to_date(data, cutoff)

        assert truncated is None
        assert reason == f"price history starts after {cutoff.date()}"


class TestValuationRebuildDefault:
    """`rebuild_valuation=False` is the default and matches the pre-KTD0
    backtest's own omission — but the *reason* now travels with the view."""

    def test_market_cap_and_stock_pe_are_absent_by_default(self):
        data = full_data()
        truncated, reason = truncate_to_date(data, pd.Timestamp("2023-06-30"))
        assert truncated is not None, reason

        assert "Market Cap" not in truncated["metadata"]
        assert "Stock P/E" not in truncated["metadata"]

    def test_absence_is_tagged_withheld_not_a_bare_omission(self):
        data = full_data()
        truncated, reason = truncate_to_date(data, pd.Timestamp("2023-06-30"))
        assert truncated is not None, reason

        assert truncated["metadata"]["_market_cap_exclusion"]["code"] == "withheld_to_prevent_leak"
        assert truncated["metadata"]["_stock_pe_exclusion"]["code"] == "withheld_to_prevent_leak"


class TestMarketCapReconciliation:
    """KTD0's two-level guard: against the stored figure at the corpus's
    latest date, and against an independent pat/eps share count at every
    replay date."""

    @staticmethod
    def _reconciling_fixture(latest_close: float, stored_market_cap: float | None):
        """equity_capital=100, face_value=10 -> shares=10cr; eps set so
        pat/eps also reads exactly 10cr, so guard (b) always agrees here —
        this fixture isolates guard (a)."""
        financials = make_financials(3, pat_growth=0.0)  # flat PAT: 150, 150, 150
        financials["eps"] = 15.0  # 150 / 15.0 = 10cr shares, matches equity route
        balance_sheet = make_balance_sheet(3)  # equity_capital 100 throughout
        price_dates = pd.bdate_range("2025-01-01", "2025-10-15")
        price = _price_df(price_dates, [latest_close] * len(price_dates))
        meta = make_metadata(market_cap=stored_market_cap) if stored_market_cap is not None else make_metadata()
        if stored_market_cap is None:
            meta.pop("Market Cap", None)
        return full_data(
            financials=financials, balance_sheet=balance_sheet, price=price,
            _metadata_raw=meta,
        )

    def test_reconciles_within_tolerance_at_the_corpus_latest_date(self):
        # rebuilt = equity(100)/face_value(10) * close(250) = 2500
        data = self._reconciling_fixture(latest_close=250.0, stored_market_cap=2510.0)
        cutoff = data["price"]["date"].max()  # the corpus's own latest date

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)

        assert truncated is not None, reason
        assert "_market_cap_exclusion" not in truncated["metadata"]
        assert truncated["metadata"]["Market Cap"] == pytest.approx(2500.0)

    def test_stored_figure_divergence_excludes_with_both_figures_named(self):
        # rebuilt = 2500; stored is 50% off -> guard (a) fails.
        data = self._reconciling_fixture(latest_close=250.0, stored_market_cap=3750.0)
        cutoff = data["price"]["date"].max()

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)

        assert truncated is not None, reason
        assert "Market Cap" not in truncated["metadata"]
        exclusion = truncated["metadata"]["_market_cap_exclusion"]
        assert exclusion["code"] == "reconciliation_failed"
        assert "2500" in exclusion["detail"] or "2500.0" in exclusion["detail"]
        assert "3750" in exclusion["detail"]

    def test_share_count_divergence_excludes_even_when_not_at_the_latest_date(self):
        """Guard (b) — the pat/eps cross-check — is the one that actually
        runs at a genuine historical replay cutoff, since guard (a) has
        nothing to check against there."""
        financials = make_financials(5, pat_growth=0.0)
        financials["eps"] = 1.5  # 150 / 1.5 = 100cr shares vs equity route's 10cr
        balance_sheet = make_balance_sheet(5)
        price = make_price(days=2600, end_close=250.0)
        data = full_data(financials=financials, balance_sheet=balance_sheet, price=price)
        # A cutoff mid-history: not the corpus's latest date, so guard (a)
        # cannot even engage — only guard (b) can catch this.
        cutoff = period_end_date(financials.iloc[2]["year"]) + pd.DateOffset(
            months=ANNUAL_REPORTING_LAG_MONTHS
        )
        assert cutoff < data["price"]["date"].max()

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)

        assert truncated is not None, reason
        assert "Market Cap" not in truncated["metadata"]
        exclusion = truncated["metadata"]["_market_cap_exclusion"]
        assert exclusion["code"] == "reconciliation_failed"
        assert "equity capital" in exclusion["detail"]
        assert "pat/eps" in exclusion["detail"]


class TestSplitSafety:
    def test_market_cap_rebuild_uses_raw_close_not_adjusted_close(self):
        """A 1:1 bonus issue doubles the share count and roughly halves the
        traded price. `close` records that drop; `adj_close` is smoothed
        across it. Using `close` (as KTD0 requires) must give the same
        market cap the pre-bonus state actually had; using `adj_close`
        would silently halve it — the double-count this choice exists to
        avoid.
        """
        financials = pd.DataFrame({
            "year": ["Mar 2019", "Mar 2020"],
            "pat": [150.0, 150.0],
            "eps": [15.0, 7.5],  # halves with the bonus-doubled share count
        })
        balance_sheet = pd.DataFrame({
            "year": ["Mar 2019", "Mar 2020"],
            "equity_capital": [100.0, 200.0],  # bonus issue doubles paid-up capital
        })
        pre_dates = pd.bdate_range("2019-01-01", "2019-09-30")
        post_dates = pd.bdate_range("2019-10-01", "2021-06-30")
        price = pd.concat([
            _price_df(pre_dates, [1000.0] * len(pre_dates), [500.0] * len(pre_dates)),
            _price_df(post_dates, [500.0] * len(post_dates), [500.0] * len(post_dates)),
        ], ignore_index=True)

        data = full_data(financials=financials, balance_sheet=balance_sheet, price=price)
        cutoff = pd.Timestamp("2019-09-30")  # pre-bonus: only "Mar 2019" is visible

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)

        assert truncated is not None, reason
        assert list(truncated["financials"]["year"]) == ["Mar 2019"]
        # equity(100)/face_value(10) * raw_close(1000) = 10,000 -- not the
        # 5,000 a wrongly-adj_close-based rebuild would give.
        assert truncated["metadata"]["Market Cap"] == pytest.approx(10_000.0)


class TestStockPERebuild:
    def test_non_positive_eps_refuses_the_multiple_rather_than_emitting_one(self):
        """The RAIN case: annual EPS at or below zero must not produce a
        rebuilt multiple three orders of magnitude from anything meaningful."""
        financials = make_financials(5, pat_growth=0.0)
        financials["eps"] = [10.0, 10.0, 10.0, 10.0, -0.02]  # latest row non-positive
        # Price must extend far enough past "Mar 2025" + the 6-month annual
        # lag for that row to even be visible at the cutoff — otherwise the
        # truncation would (correctly) fall back to an earlier, positive-EPS
        # row and this test would not exercise the refusal at all.
        price = make_price(days=2815, end_close=400.0)
        data = full_data(financials=financials, price=price)
        cutoff = data["price"]["date"].max()

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)

        assert truncated is not None, reason
        assert "Stock P/E" not in truncated["metadata"]
        exclusion = truncated["metadata"]["_stock_pe_exclusion"]
        assert exclusion["code"] == "non_positive_input"
        assert "Non-positive" in exclusion["detail"]

    def test_rebuilt_stock_pe_lands_on_the_exact_key_with_its_basis_beside_it(self):
        """KTD0: the rebuild must write to the literal `Stock P/E` key so
        `trailing_peg`/`peg_ratio`/`pe_vs_historical` read it unchanged, and
        the basis travels beside it rather than in a renamed key."""
        data = full_data()
        cutoff = data["price"]["date"].max()

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)

        assert truncated is not None, reason
        assert "Stock P/E" in truncated["metadata"]
        assert truncated["metadata"]["_stock_pe_basis"] == "annual_eps_reconstructed"
        assert truncated["metadata"]["Stock P/E"] > 0


class TestEngineIntegration:
    """A reconciling rebuild must actually unblock the production metrics
    that read `Market Cap`/`Stock P/E` from metadata — not just populate the
    keys in isolation. Sanity-checks the wiring KTD0 exists to enable."""

    def test_market_cap_metric_succeeds_on_a_reconciling_rebuild(self):
        from boundless100x.compute_engine.engine import ComputeEngine

        financials = make_financials(3, pat_growth=0.0)
        financials["eps"] = 15.0  # 150/15 = 10cr shares, matches the equity route
        balance_sheet = make_balance_sheet(3)  # equity_capital 100cr, face_value 10
        price = _price_df(
            pd.bdate_range("2025-01-01", "2025-10-15"), [250.0] * len(pd.bdate_range("2025-01-01", "2025-10-15"))
        )
        data = full_data(
            financials=financials, balance_sheet=balance_sheet, price=price,
            # This cutoff sits at the corpus's own latest date (guard (a)
            # engages), so the stored figure must agree with the rebuild
            # (equity(100)/face_value(10) * close(250) = 2500) for it to pass.
            _metadata_raw=make_metadata(market_cap=2500.0),
        )
        cutoff = data["price"]["date"].max()

        truncated, reason = truncate_to_date(data, cutoff, rebuild_valuation=True)
        assert truncated is not None, reason

        engine = ComputeEngine()
        results = engine.run_all(truncated)

        assert results["market_cap"].ok, results["market_cap"].error
        assert results["market_cap"].value == pytest.approx(2500.0)
