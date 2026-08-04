"""Macro assumptions come from config, and every price-lever bucket is reachable.

The old classification compared revenue_cagr against real_volume_growth + 3,
where real_volume_growth was revenue_cagr minus a hardcoded 5% inflation — so
the test was `cagr > cagr - 2`, always true, and `moderate_pricing` could never
be produced.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.builtin.growth import compute_price_lever
from boundless100x.compute_engine.metrics.builtin.valuation import (
    compute_earnings_yield_spread,
)
from tests.conftest import make_data, make_financials


def data_with_revenue_cagr(rate: float, n: int = 10) -> dict:
    data = make_data()
    data["financials"] = make_financials(n, revenue_growth=rate)
    return data


class TestPriceLeverBuckets:
    def test_moderate_pricing_is_reachable(self):
        """The bucket the constant-true comparison made unreachable."""
        result = compute_price_lever(data_with_revenue_cagr(0.10), {"years": 5})

        assert result.value == "moderate_pricing"

    def test_strong_pricing_power_for_growth_far_above_inflation(self):
        result = compute_price_lever(data_with_revenue_cagr(0.30), {"years": 5})

        assert result.value == "strong_pricing_power"

    def test_discounting_when_growth_trails_inflation(self):
        result = compute_price_lever(data_with_revenue_cagr(0.02), {"years": 5})

        assert result.value == "discounting"

    def test_all_configured_buckets_are_producible(self):
        """Guards against another silently-unreachable category."""
        produced = {
            compute_price_lever(data_with_revenue_cagr(rate), {"years": 5}).value
            for rate in (0.02, 0.10, 0.30)
        }

        assert produced == {"discounting", "moderate_pricing", "strong_pricing_power"}

    def test_insufficient_data_is_unknown_not_a_verdict(self):
        data = make_data()
        data["financials"] = make_financials(n=1)

        result = compute_price_lever(data, {"years": 5})

        assert not result.ok or result.value == "unknown"


class TestConfigurableInflation:
    def test_inflation_assumption_shifts_the_boundary(self):
        data = data_with_revenue_cagr(0.10)

        low = compute_price_lever(data, {"years": 5, "inflation_pct": 2.0})
        high = compute_price_lever(data, {"years": 5, "inflation_pct": 12.0})

        assert low.value != high.value
        assert high.value == "discounting"

    def test_assumption_is_reported_in_metadata(self):
        result = compute_price_lever(
            data_with_revenue_cagr(0.10), {"years": 5, "inflation_pct": 6.5}
        )

        assert result.metadata["inflation_assumption"] == 6.5


class TestConfigurableGSecYield:
    def test_spread_responds_to_configured_yield(self):
        data = make_data()

        low = compute_earnings_yield_spread(data, {"gsec_yield_pct": 5.0})
        high = compute_earnings_yield_spread(data, {"gsec_yield_pct": 9.0})

        assert low.value == pytest.approx(high.value + 4.0, abs=0.01)

    def test_default_preserves_prior_behaviour(self):
        data = make_data()

        assert compute_earnings_yield_spread(data, {}).metadata["gsec_yield"] == 7.0


class TestEngineMacroInjection:
    def test_macro_values_reach_metric_params(self):
        engine = ComputeEngine(macro={"inflation_pct": 11.0})
        data = data_with_revenue_cagr(0.10)

        results = engine.run_all(data)

        assert results["price_lever_signal"].metadata["inflation_assumption"] == 11.0

    def test_explicit_yaml_params_win_over_macro_defaults(self):
        engine = ComputeEngine(macro={"years": 99})

        assert engine.run_all(make_data())["revenue_cagr_5yr"].metadata[
            "years_requested"
        ] == 5

    def test_engine_without_macro_still_builds(self):
        assert ComputeEngine().run_all(make_data())["price_lever_signal"].ok
