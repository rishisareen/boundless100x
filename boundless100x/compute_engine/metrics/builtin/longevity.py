"""Longevity metrics: Consistency, streaks, stability, CAP proxy, reinvestment."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import detect_fcf_outliers
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_threshold_consistency(data: dict, params: dict) -> MetricResult:
    """Count years where a metric exceeds a threshold (e.g., RoCE > 15%)."""
    field = params.get("field", "roce")
    years = params.get("years", 10)
    threshold = params.get("threshold", 15)
    df = _get_annual_rows(data["ratios"], years)

    if field not in df.columns:
        return MetricResult(error=f"Field '{field}' not in ratios")

    values = pd.to_numeric(df[field], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error=f"Insufficient {field} data")

    count = int((values > threshold).sum())
    total = len(values)

    flags = []
    if count >= 8 and total >= 10:
        flags.append(f"consistently_high_{field}")

    return MetricResult(
        value=float(count),
        raw_series=values.tolist(),
        flags=flags,
        metadata={"total_years": total, "threshold": threshold},
    )


def compute_cap_proxy(data: dict, params: dict) -> MetricResult:
    """CAP Proxy = max consecutive years where RoCE > threshold."""
    roce_threshold = params.get("roce_threshold", 12)
    df = _get_annual_rows(data["ratios"], 15)

    if "roce" not in df.columns:
        return MetricResult(error="No roce column")

    values = pd.to_numeric(df["roce"], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error="Insufficient RoCE data")

    # Find max consecutive run above threshold
    max_streak = 0
    current = 0
    for v in values:
        if v > roce_threshold:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    flags = []
    if max_streak >= 8:
        flags.append("wide_moat_cap")
    elif max_streak >= 5:
        flags.append("moderate_moat_cap")

    return MetricResult(
        value=float(max_streak),
        flags=flags,
        metadata={"roce_threshold": roce_threshold, "data_years": len(values)},
    )


def compute_growth_streak(data: dict, params: dict) -> MetricResult:
    """Max consecutive years with YoY growth > threshold for a field."""
    field = params.get("field", "revenue")
    threshold_pct = params.get("threshold_pct", 10)
    df = _get_annual_rows(data["financials"], 15)

    if field not in df.columns:
        return MetricResult(error=f"Field '{field}' not in financials")

    values = pd.to_numeric(df[field], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error=f"Insufficient {field} data")

    # Compute YoY growth
    vals = values.values
    max_streak = 0
    current = 0
    for i in range(1, len(vals)):
        if vals[i - 1] > 0:
            growth = (vals[i] - vals[i - 1]) / vals[i - 1] * 100
            if growth > threshold_pct:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        else:
            current = 0

    return MetricResult(
        value=float(max_streak),
        metadata={"threshold_pct": threshold_pct, "data_years": len(values)},
    )


def compute_margin_stability(data: dict, params: dict) -> MetricResult:
    """Standard deviation of a margin field over N years."""
    field = params.get("field", "opm_pct")
    years = params.get("years", 10)
    df = _get_annual_rows(data["financials"], years)

    if field not in df.columns:
        return MetricResult(error=f"Field '{field}' not in financials")

    values = pd.to_numeric(df[field], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error=f"Insufficient {field} data")

    std = float(values.std())
    flags = []
    if std < 3:
        flags.append("highly_stable_margins")
    elif std > 8:
        flags.append("volatile_margins")

    return MetricResult(
        value=std,
        raw_series=values.tolist(),
        flags=flags,
        metadata={"years_used": len(values)},
    )


def compute_reinvestment_rate(data: dict, params: dict) -> MetricResult:
    """Reinvestment Rate = |CFI| / Depreciation (proxy for capex intensity)."""
    fin = _get_annual_rows(data["financials"], 3)
    cf = _get_annual_rows(data["cashflow"], 3)

    dep = pd.to_numeric(fin["depreciation"], errors="coerce").dropna()
    cfi = pd.to_numeric(cf["cfi"], errors="coerce").dropna()

    n = min(len(dep), len(cfi))
    if n < 2:
        return MetricResult(error="Insufficient data for reinvestment rate")

    dep_vals = dep.tail(n).values
    cfi_vals = cfi.tail(n).values

    ratios = []
    for c, d in zip(cfi_vals, dep_vals):
        if d and d > 0:
            ratios.append(abs(float(c)) / float(d))

    if not ratios:
        return MetricResult(error="Cannot compute reinvestment rate")

    avg = float(np.mean(ratios))
    flags = []
    if avg > 2.0:
        flags.append("heavy_reinvestment")

    return MetricResult(
        value=avg,
        raw_series=ratios,
        flags=flags,
    )


def compute_fcf_consistency(data: dict, params: dict) -> MetricResult:
    """Count of years with positive Free Cash Flow (CFO + CFI > 0).

    Also computes organic positive count excluding M&A outlier years.
    """
    years = params.get("years", 10)
    cf = _get_annual_rows(data["cashflow"], years)

    cfo = pd.to_numeric(cf["cfo"], errors="coerce")
    cfi = pd.to_numeric(cf["cfi"], errors="coerce")
    fcf = (cfo + cfi).dropna()

    if len(fcf) < 3:
        return MetricResult(error="Insufficient cash flow data")

    positive_count = int((fcf > 0).sum())
    total = len(fcf)

    # Detect outlier years (likely M&A) and compute organic consistency
    clean_fcf, outlier_flags = detect_fcf_outliers(fcf.values)
    organic_mask = ~np.isnan(clean_fcf)
    organic_positive = int(np.sum(clean_fcf[organic_mask] > 0)) if organic_mask.any() else 0
    organic_total = int(organic_mask.sum())

    flags = list(outlier_flags)
    if positive_count >= 8 and total >= 10:
        flags.append("consistent_fcf_generator")
    elif organic_positive >= 8 and organic_total >= 9:
        flags.append("consistent_organic_fcf_generator")

    return MetricResult(
        value=float(positive_count),
        raw_series=fcf.tolist(),
        flags=flags,
        metadata={
            "total_years": total,
            "organic_positive_years": organic_positive,
            "organic_total_years": organic_total,
            "outlier_years_excluded": len(outlier_flags),
        },
    )


def compute_dividend_consistency(data: dict, params: dict) -> MetricResult:
    """Count of years with dividend payout > 0 over N years."""
    years = params.get("years", 10)
    df = _get_annual_rows(data["financials"], years)

    if "dividend_payout_pct" not in df.columns:
        return MetricResult(error="No dividend_payout_pct column")

    values = pd.to_numeric(df["dividend_payout_pct"], errors="coerce")
    # Count non-zero, non-null values
    positive = values.dropna()
    count = int((positive > 0).sum())
    total = len(df)

    return MetricResult(
        value=float(count),
        metadata={"total_years": total},
    )
