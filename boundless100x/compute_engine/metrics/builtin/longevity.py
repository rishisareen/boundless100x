"""Longevity metrics: Consistency, streaks, stability, CAP proxy, reinvestment."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import (
    TREASURY_ADJUSTED_FLAG,
    detect_fcf_outliers,
    operating_free_cash_flow,
    treasury_flows,
)
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows
from boundless100x.compute_engine.sector import classify_sector, study_labels


def _short_window_flags(observed: int, designed: int) -> list[str]:
    """Flag a count scored against a window the data cannot fill.

    These metrics score an absolute count of good years against thresholds
    calibrated for `designed` years, so a company with less history is capped
    by arithmetic rather than by performance. The scorer decides what to do
    about it — see SQGLPScorer's history waiver.
    """
    if observed < designed:
        return [f"short_window_{observed}yr_of_{designed}yr"]
    return []


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

    flags = _short_window_flags(total, years)
    if count >= 8 and total >= 10:
        flags.append(f"consistently_high_{field}")

    return MetricResult(
        value=float(count),
        raw_series=values.tolist(),
        flags=flags,
        metadata={"total_years": total, "threshold": threshold},
    )


def compute_roe_consistency(data: dict, params: dict) -> MetricResult:
    """Count years where RoE cleared a threshold.

    `compute_threshold_consistency`'s sibling on the return measure a
    balance-sheet business is actually judged by. It cannot reuse that
    function: RoCE arrives pre-computed in Screener's `ratios` table and RoE
    does not, so the series has to be built from PAT over net worth here.

    The point of having both is that they disagree in a way that is
    informative rather than noisy. RoCE divides by equity **plus borrowings**,
    so for a lender it reports the spread on the whole funded book and sits
    structurally in the low teens no matter how good the business is — nought
    years above 15% is close to a definition of the sector, not a finding
    about the company. RoE asks what the owners earned, which is the question,
    and a lender that cannot clear 15% on equity while running 4x leverage has
    genuinely not earned its cost of capital.
    """
    years = params.get("years", 10)
    threshold = params.get("threshold", 15)

    fin = _get_annual_rows(data["financials"], years)
    bs = _get_annual_rows(data["balance_sheet"], years)

    required = ("equity_capital", "reserves")
    if any(col not in bs.columns for col in required):
        return MetricResult(error="Balance sheet lacks equity capital or reserves")

    pat = pd.to_numeric(fin["pat"], errors="coerce").dropna()
    equity = (
        pd.to_numeric(bs["equity_capital"], errors="coerce")
        + pd.to_numeric(bs["reserves"], errors="coerce")
    ).dropna()

    n = min(len(pat), len(equity))
    if n < 3:
        return MetricResult(error="Insufficient data for RoE consistency")

    series = [
        float(p / e * 100)
        for p, e in zip(pat.tail(n).values, equity.tail(n).values)
        if e and e > 0
    ]
    if len(series) < 3:
        return MetricResult(error="Insufficient valid RoE data points")

    count = int(sum(1 for v in series if v > threshold))
    total = len(series)

    flags = _short_window_flags(total, years)
    if count >= 8 and total >= 10:
        flags.append("consistently_high_roe")

    return MetricResult(
        value=float(count),
        raw_series=series,
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

    flags = _short_window_flags(len(values), params.get("designed_years", 10))
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
        # A streak cannot exceed the observations available to grow it.
        flags=_short_window_flags(len(values), params.get("designed_years", 10)),
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
    # |CFI| is a capex proxy only once treasury movement is out of it. Caplin
    # Point read 6.3x depreciation — "heavy reinvestment" — on an investing
    # line that was substantially mutual funds.
    treasury = treasury_flows(_get_annual_rows(data.get("balance_sheet", pd.DataFrame()), 4))
    cfi_raw = pd.to_numeric(cf["cfi"], errors="coerce")
    if treasury and "year" in cf.columns:
        cfi_raw = cfi_raw + cf["year"].astype(str).map(treasury).fillna(0.0)
    cfi = cfi_raw.dropna()

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

    # Counted on cash the business left over, not on the net of every
    # investing decision — a company that swept its surplus into deposits
    # otherwise records a year of "negative free cash flow" for saving.
    _, fcf, treasury = operating_free_cash_flow(
        cf, _get_annual_rows(data.get("balance_sheet", pd.DataFrame()), years + 1)
    )

    if len(fcf) < 3:
        return MetricResult(error="Insufficient cash flow data")

    positive_count = int((fcf > 0).sum())
    total = len(fcf)

    # Detect outlier years (likely M&A) and compute organic consistency
    clean_fcf, outlier_flags = detect_fcf_outliers(fcf.values)
    organic_mask = ~np.isnan(clean_fcf)
    organic_positive = int(np.sum(clean_fcf[organic_mask] > 0)) if organic_mask.any() else 0
    organic_total = int(organic_mask.sum())

    flags = list(outlier_flags) + _short_window_flags(total, years)
    if treasury.get("adjusted"):
        flags.append(TREASURY_ADJUSTED_FLAG)
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
        flags=_short_window_flags(total, years),
        metadata={"total_years": total},
    )


def compute_sector_tailwind(data: dict, params: dict) -> MetricResult:
    """Classify the company's sector against the Dec 2025 study's buckets.

    The study found compounders clustered in a handful of sectors and largely
    absent from others, so the sector a company sits in is a longevity signal
    independent of its own numbers.
    """
    metadata = data.get("metadata") or {}
    labels = study_labels(metadata)
    classification = classify_sector(labels)
    sector = metadata.get("sector")

    flags = []
    if classification == "strong_tailwind":
        flags.append("sector_strong_tailwind")
    elif classification == "non_consideration":
        flags.append("sector_non_consideration")
    elif classification == "unknown":
        # Common today: no cached ticker carries a sector until a re-fetch.
        flags.append("sector_unclassified")

    return MetricResult(
        value=classification,
        flags=flags,
        metadata={
            "sector": sector or "unavailable",
            "labels_considered": list(labels),
        },
    )
