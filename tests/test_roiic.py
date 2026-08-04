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

    def test_registry_has_no_duplicate_ids(self, tmp_path):
        """A duplicate id silently dropped a scored metric before the guard."""
        (tmp_path / "registry.yaml").write_text(
            "element_weights: {size: 0.5, growth: 0.5}\n"
        )
        elements = tmp_path / "elements"
        elements.mkdir()
        body = (
            "element: {name}\nmetrics:\n  shared_id:\n    name: X\n"
            "    module: builtin.size\n    function: compute_market_cap\n"
            "    inputs: [price]\n    scoring: {{weight: 0.1, thresholds: [1]}}\n"
            "    display: {{format: '{{}}'}}\n"
        )
        (elements / "a.yaml").write_text(body.format(name="size"))
        (elements / "b.yaml").write_text(body.format(name="growth"))

        with pytest.raises(ValueError, match="Duplicate metric id"):
            ComputeEngine(registry_dir=str(tmp_path))

    def test_engine_runs_both_metrics_on_synthetic_data(self):
        results = ComputeEngine().run_all(make_data())

        assert results["roiic"].ok
        assert results["capital_reinvestment_rate"].ok


class TestRealWorldFrameShapes:
    """Screener's frames are ragged: the P&L ends with TTM, the balance sheet
    with a part-year column. Dropping only TTM left half a year of balance
    sheet paired against a full year of P&L in every cached company."""

    def test_interim_balance_sheet_row_does_not_reach_the_calculation(self):
        from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows
        from tests.conftest import make_balance_sheet

        rows = _get_annual_rows(make_balance_sheet(10, interim=True), 6)

        assert not rows["year"].astype(str).str.startswith("Sep").any()

    def test_roiic_is_unchanged_by_a_trailing_interim_row(self):
        clean = build([100.0 + 25.0 * i for i in range(8)], [400.0 + 100.0 * i for i in range(8)])
        ragged = build([100.0 + 25.0 * i for i in range(8)], [400.0 + 100.0 * i for i in range(8)])
        ragged["balance_sheet"] = make_balance_sheet(8, interim=True)
        ragged["balance_sheet"]["reserves"] = [400.0 + 100.0 * i for i in range(8)] + [50.0]
        ragged["balance_sheet"]["equity_capital"] = [100.0] * 9
        ragged["balance_sheet"]["borrowings"] = [50.0] * 9

        assert compute_roiic(ragged, {"years": 5}).value == pytest.approx(
            compute_roiic(clean, {"years": 5}).value, abs=0.01
        )

    def test_ttm_row_still_excluded_from_the_pnl(self):
        from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows
        from tests.conftest import make_financials as mf

        rows = _get_annual_rows(mf(10, ttm=True), 6)

        assert "TTM" not in rows["year"].astype(str).tolist()
