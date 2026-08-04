"""Reverse DCF implied growth — especially saturation at the search bounds.

The binary search is bounded to [-10%, +50%]. A pinned result is an artifact of
the bound, not a measurement, and it feeds both scoring and the price-gate
veto — so saturation must be flagged rather than returned silently.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.builtin.valuation import compute_reverse_dcf


def make_data(market_cap: float) -> dict:
    """Steady company: ₹100 Cr FCF every year, ~10% revenue growth."""
    years = [f"Mar {y}" for y in range(2021, 2026)]
    return {
        "metadata": {"Market Cap": market_cap},
        "cashflow": pd.DataFrame({
            "year": years,
            "cfo": [150.0] * 5,
            "cfi": [-50.0] * 5,
        }),
        "financials": pd.DataFrame({
            "year": years,
            "revenue": [100.0, 110.0, 121.0, 133.1, 146.4],
        }),
    }


class TestInteriorSolution:
    def test_normal_price_gives_an_unsaturated_result(self):
        # ~20x average FCF prices in roughly 10% implied growth.
        result = compute_reverse_dcf(make_data(2_000.0), {})

        assert result.ok
        assert 0 < result.value < 40
        assert "reverse_dcf_saturated" not in result.flags
        assert result.metadata["saturated_at"] is None


class TestSaturation:
    def test_extreme_price_pins_to_the_ceiling_and_says_so(self):
        result = compute_reverse_dcf(make_data(50_000.0), {})

        assert result.ok
        assert result.value == pytest.approx(50.0)
        assert "reverse_dcf_saturated" in result.flags
        assert result.metadata["saturated_at"] == "ceiling"

    def test_dirt_cheap_price_pins_to_the_floor_and_says_so(self):
        result = compute_reverse_dcf(make_data(200.0), {})

        assert result.ok
        assert result.value == pytest.approx(-10.0)
        assert "reverse_dcf_saturated" in result.flags
        assert result.metadata["saturated_at"] == "floor"

    def test_saturation_does_not_suppress_the_overpriced_veto(self):
        """A ceiling-pinned company is still very plausibly overpriced."""
        result = compute_reverse_dcf(make_data(50_000.0), {})

        assert "reverse_dcf_overpriced" in result.flags
