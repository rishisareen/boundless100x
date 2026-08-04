"""The historical P/E band must be built from historical prices.

The metric divided *today's* price by each past year's EPS. Because the price
term was constant, the resulting "distribution" was a rescaled reciprocal of
the EPS series, so the percentile measured where current earnings sat in their
own history — earnings growth wearing a valuation label. Any company with EPS
near a high scored as maximally cheap regardless of what it actually traded at.
"""

import pandas as pd
import pytest

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
