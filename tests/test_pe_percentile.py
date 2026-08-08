"""The historical P/E band must be built from historical prices.

The metric divided *today's* price by each past year's EPS. Because the price
term was constant, the resulting "distribution" was a rescaled reciprocal of
the EPS series, so the percentile measured where current earnings sat in their
own history — earnings growth wearing a valuation label. Any company with EPS
near a high scored as maximally cheap regardless of what it actually traded at.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.valuation import compute_pe_percentile
from tests.conftest import make_metadata, year_labels


def build(eps: list[float], year_end_prices: list[float], current_pe: float,
          current_price: float | None = None) -> dict:
    """Financials plus a daily price series pinned to each fiscal year end."""
    n = len(eps)
    years = year_labels(n)
    financials = pd.DataFrame({"year": years, "eps": eps})

    dates, closes = [], []
    for label, price in zip(years, year_end_prices):
        year = int(label.split()[-1])
        # A few sessions either side of 31 March, so the lookup has real bars.
        for offset in range(-5, 6):
            dates.append(pd.Timestamp(year=year, month=3, day=31) + pd.Timedelta(days=offset))
            closes.append(price)
    price_df = pd.DataFrame({"date": dates, "close": closes}).sort_values("date")

    meta = make_metadata()
    meta["Stock P/E"] = current_pe
    meta["Current Price"] = current_price if current_price is not None else year_end_prices[-1]
    return {"financials": financials, "metadata": meta, "price": price_df}


class TestUsesHistoricalPrices:
    def test_grower_at_a_high_multiple_is_not_called_cheap(self):
        """The bug's signature: rising EPS alone used to force percentile ~0."""
        eps = [10.0 * 1.25 ** i for i in range(10)]
        # The multiple expanded all the way through: today is the dearest it has been.
        prices = [e * (10 + 2 * i) for i, e in enumerate(eps)]

        result = compute_pe_percentile(build(eps, prices, current_pe=28.0), {"years": 10})

        assert result.ok
        assert result.value > 75
        assert "pe_above_historical_75th" in result.flags

    def test_grower_at_a_low_multiple_is_still_called_cheap(self):
        eps = [10.0 * 1.25 ** i for i in range(10)]
        prices = [e * (30 - 2 * i) for i, e in enumerate(eps)]

        result = compute_pe_percentile(build(eps, prices, current_pe=12.0), {"years": 10})

        assert result.value < 25
        assert "pe_below_historical_25th" in result.flags

    def test_flat_multiple_lands_mid_band(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        prices = [e * 20.0 for e in eps]

        result = compute_pe_percentile(build(eps, prices, current_pe=20.0), {"years": 10})

        assert 25 <= result.value <= 90

    def test_band_is_reported_for_inspection(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        prices = [e * (15 + i) for i, e in enumerate(eps)]

        meta = compute_pe_percentile(build(eps, prices, current_pe=22.0), {"years": 10}).metadata

        assert meta["pe_min"] < meta["pe_max"]
        assert meta["years_used"] >= 5

    def test_earnings_growth_alone_does_not_move_the_percentile(self):
        """Two companies at an identical, unchanged multiple; only growth differs."""
        slow_eps = [10.0 * 1.05 ** i for i in range(10)]
        fast_eps = [10.0 * 1.40 ** i for i in range(10)]

        slow = compute_pe_percentile(
            build(slow_eps, [e * 20.0 for e in slow_eps], current_pe=20.0), {"years": 10})
        fast = compute_pe_percentile(
            build(fast_eps, [e * 20.0 for e in fast_eps], current_pe=20.0), {"years": 10})

        assert slow.value == pytest.approx(fast.value, abs=1.0)


class TestDegradation:
    def test_missing_price_history_errors_rather_than_faking_a_band(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        data = build(eps, [e * 20 for e in eps], current_pe=20.0)
        data["price"] = pd.DataFrame({"date": [], "close": []})

        result = compute_pe_percentile(data, {"years": 10})

        assert not result.ok

    def test_price_history_shorter_than_financials_uses_the_overlap(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        data = build(eps, [e * 20 for e in eps], current_pe=20.0)
        cutoff = pd.Timestamp(year=2021, month=1, day=1)
        data["price"] = data["price"][data["price"]["date"] >= cutoff]

        result = compute_pe_percentile(data, {"years": 10})

        assert result.ok
        assert result.metadata["years_used"] < 10

    def test_too_little_overlap_errors(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        data = build(eps, [e * 20 for e in eps], current_pe=20.0)
        cutoff = pd.Timestamp(year=2024, month=1, day=1)
        data["price"] = data["price"][data["price"]["date"] >= cutoff]

        assert not compute_pe_percentile(data, {"years": 10}).ok

    def test_non_positive_eps_years_are_skipped(self):
        eps = [-5.0, -2.0] + [10.0 * 1.2 ** i for i in range(8)]
        prices = [abs(e) * 20 for e in eps]

        result = compute_pe_percentile(build(eps, prices, current_pe=20.0), {"years": 10})

        assert result.ok
        assert result.metadata["years_used"] == 8

    def test_missing_current_pe_errors(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        data = build(eps, [e * 20 for e in eps], current_pe=20.0)
        del data["metadata"]["Stock P/E"]

        assert not compute_pe_percentile(data, {"years": 10}).ok


class TestPriceBasisIsVisible:
    """Cached price files predate the raw/adjusted split. Adjusted closes
    understate past prices, so the band reads cheap and today's percentile
    reads high — the reader has to be told."""

    def test_legacy_basis_raises_a_flag(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        data = build(eps, [e * 20 for e in eps], current_pe=20.0)   # no adj_close column

        result = compute_pe_percentile(data, {"years": 10})

        assert result.metadata["price_basis"] == "legacy_close_unknown_adjustment"
        assert "pe_band_legacy_price_basis" in result.flags

    def test_raw_basis_raises_no_such_flag(self):
        eps = [10.0 * 1.2 ** i for i in range(10)]
        data = build(eps, [e * 20 for e in eps], current_pe=20.0)
        data["price"]["adj_close"] = data["price"]["close"] * 0.9

        result = compute_pe_percentile(data, {"years": 10})

        assert result.metadata["price_basis"] == "raw_close"
        assert "pe_band_legacy_price_basis" not in result.flags


class TestTheRenderedBandMatchesTheRenderedPercentile:
    """The range a reader sees must be the range the percentile was taken in.

    The metric was fixed to divide each past year-end close by that year's EPS.
    `_build_pe_band_summary` was not: it kept deriving the range as
    `current_price / historical_eps`, the exact anti-pattern
    `compute_pe_percentile`'s own docstring warns against. So the report quoted
    a percentile from one distribution beside the minimum and maximum of a
    different one, and the two could not be reconciled by any reader — PFC's
    real output placed a current 5.3x at the 70th percentile of a range whose
    floor was 5.4x, which is arithmetically impossible.

    Presentation-layer only. The scored value never moved; only the two numbers
    printed next to it were wrong.
    """

    @staticmethod
    def summary_for(eps, year_end_prices, current_pe, current_price=None):
        """Render the band the way the Markdown report does."""
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        data = build(eps, year_end_prices, current_pe, current_price)
        pe_hist = compute_pe_percentile(data, {"years": 10})
        assert pe_hist.ok, pe_hist.error

        result = make_result(
            metrics={
                "pe_vs_historical": pe_hist,
                "pe_ttm": MetricResult(value=float(current_pe)),
            },
        )
        result.data["financials"] = data["financials"]
        result.data["price"] = data["price"]
        result.data["metadata"] = data["metadata"]

        return ReportGenerator()._build_pe_band_summary(result), pe_hist

    def test_the_percentile_falls_inside_the_range_it_is_quoted_beside(self):
        """PFC's case, and the reason the two numbers could disagree at all.

        `current_pe` comes from Screener's `Stock P/E`, which is struck on
        *trailing-twelve-month* earnings. The old range divided `Current Price`
        by each past *annual* EPS. When TTM earnings have outgrown the last
        annual figure — an ordinary state for a company still growing — the
        cheapest ratio the old range could produce was already dearer than the
        multiple it was printed beside, so the band's floor sat above the
        current multiple no matter what the shares actually did.
        """
        eps = [10.0 * 1.25 ** i for i in range(10)]
        # A real spread of historical multiples, 4x to 8x, so the current
        # multiple has somewhere honest to sit inside the true band.
        multiples = [8.0, 4.0, 7.0, 5.0, 7.5, 4.5, 6.0, 8.0, 5.5, 7.0]
        prices = [e * m for e, m in zip(eps, multiples)]

        current_pe = 6.5
        ttm_eps = eps[-1] * 1.2                       # TTM ahead of last annual

        summary, _ = self.summary_for(
            eps, prices, current_pe=current_pe, current_price=ttm_eps * current_pe,
        )

        assert summary["pe_min"] <= summary["current_pe"] <= summary["pe_max"], (
            f"current {summary['current_pe']}x sits outside the rendered band "
            f"{summary['pe_min']}x-{summary['pe_max']}x"
        )

    def test_the_rendered_range_is_the_series_the_percentile_was_taken_in(self):
        eps = [10.0 * 1.15 ** i for i in range(10)]
        prices = [e * 18 for e in eps]

        summary, pe_hist = self.summary_for(eps, prices, current_pe=18.0)

        assert summary["pe_min"] == pytest.approx(min(pe_hist.raw_series), abs=0.01)
        assert summary["pe_max"] == pytest.approx(max(pe_hist.raw_series), abs=0.01)

    def test_a_company_at_its_historical_minimum_renders_at_the_bottom(self):
        """The 0th-percentile case: cheaper than every year it has traded."""
        eps = [10.0] * 10
        prices = [e * 20 for e in eps]

        summary, _ = self.summary_for(eps, prices, current_pe=5.0)

        assert summary["percentile"] == 0.0
        assert summary["current_pe"] <= summary["pe_min"]

    def test_the_scored_value_does_not_move(self):
        """R17: this is presentation. The percentile itself is untouched."""
        eps = [10.0 * 1.25 ** i for i in range(10)]
        prices = [e * 20 for e in eps[:-1]] + [eps[-1] * 5]

        summary, pe_hist = self.summary_for(eps, prices, current_pe=5.0)

        assert summary["percentile"] == pe_hist.value

    def test_a_metric_without_the_band_metadata_renders_nothing(self):
        """Degradation stays a blank section, not a half-built band."""
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        result = make_result(metrics={
            "pe_vs_historical": MetricResult(value=50.0),   # no metadata, no series
            "pe_ttm": MetricResult(value=18.0),
        })

        assert ReportGenerator()._build_pe_band_summary(result) == {}
