"""CAGR machinery: horizon slicing and endpoint smoothing.

`_compute_cagr_from_series` accepted a `years` argument and ignored it, so the
4-lever decomposition reported identical 3yr and 5yr CAGRs for every company
(observed on CDSL: both PAT CAGRs 37.5056024879207).
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.builtin.growth import (
    _compute_cagr_from_series,
    compute_cagr,
    compute_lever_decomposition_table,
)
from tests.conftest import make_data, make_financials, year_labels


def two_phase_series(early_rate: float, late_rate: float, n: int = 10,
                     base: float = 100.0, split: int = 5) -> list[float]:
    """Compounds at `early_rate` for the first `split` steps, then `late_rate`."""
    values = [base]
    for i in range(1, n):
        rate = early_rate if i <= split else late_rate
        values.append(values[-1] * (1 + rate))
    return values


class TestHorizonSlicing:
    def test_three_year_and_five_year_cagr_differ_on_two_phase_growth(self):
        series = pd.Series(two_phase_series(0.30, 0.10))

        cagr_3 = _compute_cagr_from_series(series, 3)
        cagr_5 = _compute_cagr_from_series(series, 5)

        assert cagr_3 == pytest.approx(10.0, abs=0.5)
        assert cagr_5 != pytest.approx(cagr_3, abs=0.5)

    def test_horizon_is_honoured_over_the_full_series(self):
        """A 3yr CAGR reads only the last four points, whatever else is present."""
        series = pd.Series(two_phase_series(0.50, 0.05, n=12, split=7))

        assert _compute_cagr_from_series(series, 3) == pytest.approx(5.0, abs=0.5)

    def test_decomposition_reports_distinct_three_and_five_year_cagrs(self):
        """The CDSL symptom, at the level the report and LLM actually consume."""
        financials = make_financials(n=10)
        financials["pat"] = two_phase_series(0.40, 0.08)
        financials["revenue"] = two_phase_series(0.35, 0.07)
        data = make_data()
        data["financials"] = financials

        table = compute_lever_decomposition_table(data, years=5)
        profile = table["earnings_profile"]

        assert profile["pat_cagr_3yr"] != pytest.approx(profile["pat_cagr_5yr"], abs=0.5)
        assert profile["pat_cagr_3yr"] == pytest.approx(8.0, abs=1.0)

    def test_shorter_series_falls_back_to_available_span(self):
        series = pd.Series([100.0, 120.0, 144.0])

        assert _compute_cagr_from_series(series, 5) == pytest.approx(20.0, abs=0.5)

    def test_single_point_returns_none(self):
        assert _compute_cagr_from_series(pd.Series([100.0]), 3) is None

    def test_non_positive_endpoints_return_none(self):
        assert _compute_cagr_from_series(pd.Series([-10.0, 5.0, 20.0]), 2) is None


class TestEndpointSmoothing:
    def test_terminal_spike_moves_smoothed_cagr_less(self):
        steady = [100.0 * 1.15 ** i for i in range(10)]
        spiked = steady[:-1] + [steady[-1] * 1.6]

        smoothed_delta = abs(
            _compute_cagr_from_series(pd.Series(spiked), 5, smooth=True)
            - _compute_cagr_from_series(pd.Series(steady), 5, smooth=True)
        )
        raw_delta = abs(
            _compute_cagr_from_series(pd.Series(spiked), 5, smooth=False)
            - _compute_cagr_from_series(pd.Series(steady), 5, smooth=False)
        )

        assert smoothed_delta < raw_delta

    def test_smoothing_is_neutral_on_a_clean_geometric_series(self):
        series = pd.Series([100.0 * 1.2 ** i for i in range(10)])

        assert _compute_cagr_from_series(series, 5, smooth=True) == pytest.approx(20.0, abs=0.1)

    def test_short_series_is_not_smoothed(self):
        """Below six points, averaging endpoints would leave too little signal."""
        series = pd.Series([100.0, 130.0, 150.0, 200.0])

        assert _compute_cagr_from_series(series, 3, smooth=True) == (
            _compute_cagr_from_series(series, 3, smooth=False)
        )


class TestComputeCagrMetric:
    def test_metric_records_endpoint_mode(self):
        data = make_data()

        result = compute_cagr(data, {"field": "revenue", "years": 5})

        assert result.ok
        assert result.metadata["endpoint_mode"] in {"smoothed", "single"}

    def test_metric_matches_clean_geometric_rate(self):
        data = make_data()  # revenue compounds at exactly 20%

        result = compute_cagr(data, {"field": "revenue", "years": 5})

        assert result.value == pytest.approx(20.0, abs=0.1)

    def test_short_history_still_flags(self):
        data = make_data()
        data["financials"] = make_financials(n=3)

        result = compute_cagr(data, {"field": "revenue", "years": 5})

        assert result.ok
        assert any("insufficient_history" in f for f in result.flags)

    def test_missing_field_errors_cleanly(self):
        data = make_data()

        result = compute_cagr(data, {"field": "not_a_column", "years": 5})

        assert not result.ok
        assert "not_a_column" in result.error


class TestBankNbfcPath:
    def test_decomposition_works_without_operating_profit_column(self):
        """Screener omits operating_profit for banks/NBFCs; _ensure_operating_profit derives it."""
        financials = make_financials(n=10).drop(columns=["operating_profit"])
        data = make_data()
        data["financials"] = financials

        table = compute_lever_decomposition_table(data, years=5)

        assert table["earnings_profile"]["pat_cagr_5yr"] is not None
        assert len(table["lever_table"]) == 3
