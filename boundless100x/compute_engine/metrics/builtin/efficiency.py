"""Efficiency metrics: Working Capital Days trend."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_wc_days_trend(data: dict, params: dict) -> MetricResult:
    """Working Capital Days trend over N years.

    Value = latest working capital days.
    Flags indicate whether trend is improving (declining) or worsening.
    """
    years = params.get("years", 5)
    df = _get_annual_rows(data["ratios"], years)

    if "working_capital_days" not in df.columns:
        return MetricResult(error="No working_capital_days in ratios")

    values = pd.to_numeric(df["working_capital_days"], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error="Insufficient WC days data")

    latest = float(values.iloc[-1])
    trend = float(values.iloc[-1] - values.iloc[0])

    flags = []
    if trend < -5:
        flags.append("improving_working_capital")
    elif trend > 10:
        flags.append("worsening_working_capital")

    return MetricResult(
        value=latest,
        raw_series=values.tolist(),
        flags=flags,
        metadata={"trend_change": trend, "years_used": len(values)},
    )
