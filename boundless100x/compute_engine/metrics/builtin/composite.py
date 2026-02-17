"""Composite metrics: Quality-Growth Matrix classification."""

import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_qg_quadrant(data: dict, params: dict) -> MetricResult:
    """Quality-Growth Matrix position.

    High Quality (RoCE > 15%) + High Growth (PAT CAGR > 15%) = True Wealth Creator
    High Quality + Low Growth = Quality Trap
    Low Quality + High Growth = Growth Trap
    Low Quality + Low Growth = Wealth Destroyer
    """
    quality_threshold = params.get("quality_threshold", 15)
    growth_threshold = params.get("growth_threshold", 15)

    # Quality: 5yr avg RoCE
    ratios = _get_annual_rows(data["ratios"], 5)
    if "roce" not in ratios.columns:
        return MetricResult(error="No roce for QG matrix")

    roce_values = pd.to_numeric(ratios["roce"], errors="coerce").dropna()
    if len(roce_values) < 3:
        return MetricResult(error="Insufficient RoCE data for QG matrix")

    avg_roce = float(roce_values.mean())

    # Growth: 5yr PAT CAGR
    fin = _get_annual_rows(data["financials"], 6)
    pat = pd.to_numeric(fin["pat"], errors="coerce").dropna()
    if len(pat) < 2:
        return MetricResult(error="Insufficient PAT data for QG matrix")

    start = float(pat.iloc[0])
    end = float(pat.iloc[-1])
    actual_years = len(pat) - 1

    if start <= 0 or end <= 0:
        pat_cagr = 0.0
    else:
        pat_cagr = ((end / start) ** (1 / actual_years) - 1) * 100

    high_quality = avg_roce >= quality_threshold
    high_growth = pat_cagr >= growth_threshold

    if high_quality and high_growth:
        quadrant = "true_wealth_creator"
    elif high_quality and not high_growth:
        quadrant = "quality_trap"
    elif not high_quality and high_growth:
        quadrant = "growth_trap"
    else:
        quadrant = "wealth_destroyer"

    return MetricResult(
        value=quadrant,
        metadata={
            "avg_roce": avg_roce,
            "pat_cagr": pat_cagr,
            "quality_threshold": quality_threshold,
            "growth_threshold": growth_threshold,
        },
    )
