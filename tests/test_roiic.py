"""Incremental returns on invested capital, and how much is reinvested.

ROIIC asks what the *marginal* rupee of capital earned, which is the signal
that separates a compounder from a company whose headline RoCE is carried by a
legacy asset base.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.builtin.profitability import (
    compute_capital_reinvestment_rate,
    compute_roiic,
)
from tests.conftest import make_balance_sheet, make_data, make_financials


def build(pat: list[float], reserves: list[float], borrowings: list[float] | None = None,
          equity: float = 100.0) -> dict:
    n = len(pat)
    data = make_data(n=n)
    financials = make_financials(n)
    financials["pat"] = pat
    financials["pbt"] = [p / 0.75 for p in pat]
    financials["tax_pct"] = [25.0] * n
    financials["operating_profit"] = [p * 1.4 for p in pat]
    financials["interest"] = [5.0] * n
    balance_sheet = make_balance_sheet(n)
    balance_sheet["reserves"] = reserves
    balance_sheet["equity_capital"] = [equity] * n
    balance_sheet["borrowings"] = borrowings if borrowings is not None else [50.0] * n
    data["financials"] = financials
    data["balance_sheet"] = balance_sheet
    return data


class TestROIIC:
    def test_steady_compounder_scores_near_the_incremental_rate(self):
        """NOPAT and capital both grow; the marginal return is ~25%."""
        pat = [100.0 + 25.0 * i for i in range(8)]
        reserves = [400.0 + 100.0 * i for i in range(8)]

        result = compute_roiic(build(pat, reserves), {"years": 5})

        assert result.ok
        assert result.value == pytest.approx(25.0, abs=6.0)

    def test_capital_piling_up_without_earnings_scores_near_zero(self):
        pat = [100.0] * 8
        reserves = [400.0 + 150.0 * i for i in range(8)]

        result = compute_roiic(build(pat, reserves), {"years": 5})

        assert result.ok
        assert result.value == pytest.approx(0.0, abs=2.0)

    def test_falling_earnings_on_rising_capital_flags_negative_returns(self):
        pat = [200.0 - 15.0 * i for i in range(8)]
        reserves = [400.0 + 120.0 * i for i in range(8)]

        result = compute_roiic(build(pat, reserves), {"years": 5})

        assert result.value < 0
        assert "negative_incremental_returns" in result.flags

    def test_high_incremental_returns_flag_a_compounder(self):
        pat = [100.0 + 60.0 * i for i in range(8)]
        reserves = [400.0 + 100.0 * i for i in range(8)]

        result = compute_roiic(build(pat, reserves), {"years": 5})

        assert "high_roiic_compounder" in result.flags

    def test_shrinking_capital_base_does_not_blow_up(self):
        """Buybacks and debt paydown shrink the denominator — no division blowup."""
        pat = [100.0 + 10.0 * i for i in range(8)]
        reserves = [800.0 - 50.0 * i for i in range(8)]

        result = compute_roiic(build(pat, reserves, borrowings=[200.0 - 20.0 * i for i in range(8)]),
                               {"years": 5})

        assert result.value is None or abs(result.value) < 1000
        if result.value is None:
            assert result.error

    def test_flat_capital_returns_no_meaningful_ratio(self):
        pat = [100.0 + 10.0 * i for i in range(8)]
        reserves = [500.0] * 8

        result = compute_roiic(build(pat, reserves, borrowings=[50.0] * 8), {"years": 5})

        assert not result.ok or result.value is None

    def test_missing_borrowings_column_errors_without_crashing_the_engine(self):
        data = build([100.0 + 10.0 * i for i in range(8)], [400.0 + 50.0 * i for i in range(8)])
        data["balance_sheet"] = data["balance_sheet"].drop(columns=["borrowings"])

        result = compute_roiic(data, {"years": 5})

        assert not result.ok
        assert result.error

    def test_insufficient_history_errors(self):
        result = compute_roiic(build([100.0, 110.0], [400.0, 450.0]), {"years": 5})

        assert not result.ok


class TestReinvestmentRate:
    def test_heavy_reinvestment_reads_high(self):
        pat = [100.0] * 8
        reserves = [400.0 + 90.0 * i for i in range(8)]

        result = compute_capital_reinvestment_rate(build(pat, reserves), {"years": 5})

        assert result.ok
        assert result.value > 50.0

    def test_capital_returned_rather_than_reinvested_is_flagged(self):
        pat = [100.0] * 8
        reserves = [400.0 + 5.0 * i for i in range(8)]

        result = compute_capital_reinvestment_rate(build(pat, reserves), {"years": 5})

        assert "capital_returned_not_reinvested" in result.flags

    def test_rate_is_reported_as_a_percentage(self):
        pat = [100.0] * 8
        reserves = [400.0 + 50.0 * i for i in range(8)]

        result = compute_capital_reinvestment_rate(build(pat, reserves), {"years": 5})

        assert 0 <= result.value <= 200


class TestRegistryIntegration:
    def test_both_metrics_are_registered_under_quality_business(self):
        engine = ComputeEngine()

        for metric_id in ("roiic", "capital_reinvestment_rate"):
            assert metric_id in engine.metrics
            assert engine.metrics[metric_id]["element"] == "quality_business"

    def test_metric_count_grew_by_two(self):
        assert len(ComputeEngine().metrics) == 50

    def test_engine_runs_both_metrics_on_synthetic_data(self):
        results = ComputeEngine().run_all(make_data())

        assert results["roiic"].ok
        assert results["capital_reinvestment_rate"].ok
