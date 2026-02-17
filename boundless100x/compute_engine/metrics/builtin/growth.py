"""Growth metrics: CAGR, 4-lever decomposition, quality grade, consistency."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_cagr(data: dict, params: dict) -> MetricResult:
    """Compute CAGR for any financial field over N years."""
    field = params.get("field", "revenue")
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years + 1)  # Need N+1 for N-year CAGR

    if field not in df.columns:
        return MetricResult(error=f"Field '{field}' not in financials")

    values = pd.to_numeric(df[field], errors="coerce").dropna()
    if len(values) < 2:
        return MetricResult(error=f"Insufficient {field} data for CAGR")

    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    actual_years = len(values) - 1

    if start <= 0 or end <= 0:
        return MetricResult(error=f"Non-positive values for {field} CAGR")

    cagr = ((end / start) ** (1 / actual_years) - 1) * 100

    # Data quality flags
    flags = []
    if actual_years < years:
        flags.append(f"insufficient_history_{actual_years}yr_of_{years}yr")
    if actual_years < 3:
        flags.append("very_short_history_unreliable")

    return MetricResult(
        value=float(cagr),
        raw_series=values.tolist(),
        flags=flags,
        metadata={
            "start": start,
            "end": end,
            "years_actual": actual_years,
            "years_requested": years,
        },
    )


def compute_operating_leverage(data: dict, params: dict) -> MetricResult:
    """Operating Leverage = %Δ EBIT / %Δ Revenue (average over N years)."""
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years + 1)

    revenue = pd.to_numeric(df["revenue"], errors="coerce").dropna()
    op = pd.to_numeric(df["operating_profit"], errors="coerce").dropna()

    n = min(len(revenue), len(op))
    if n < 3:
        return MetricResult(error="Insufficient data for operating leverage")

    rev = revenue.tail(n).values
    ebit = op.tail(n).values

    leverages = []
    for i in range(1, len(rev)):
        rev_chg = (rev[i] - rev[i - 1]) / rev[i - 1] if rev[i - 1] != 0 else 0
        ebit_chg = (ebit[i] - ebit[i - 1]) / ebit[i - 1] if ebit[i - 1] != 0 else 0
        if rev_chg != 0:
            leverages.append(ebit_chg / rev_chg)

    if not leverages:
        return MetricResult(error="Cannot compute operating leverage")

    # Trim outliers (cap at ±5x)
    leverages = [max(-5, min(5, x)) for x in leverages]
    avg = float(np.mean(leverages))

    return MetricResult(
        value=avg,
        raw_series=leverages,
        metadata={"years_used": len(leverages)},
    )


def compute_financial_leverage(data: dict, params: dict) -> MetricResult:
    """Financial Leverage = %Δ EPS / %Δ EBIT (average over N years)."""
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years + 1)

    op = pd.to_numeric(df["operating_profit"], errors="coerce").dropna()
    eps = pd.to_numeric(df["eps"], errors="coerce").dropna()

    n = min(len(op), len(eps))
    if n < 3:
        return MetricResult(error="Insufficient data for financial leverage")

    ebit = op.tail(n).values
    eps_vals = eps.tail(n).values

    leverages = []
    for i in range(1, len(ebit)):
        ebit_chg = (ebit[i] - ebit[i - 1]) / ebit[i - 1] if ebit[i - 1] != 0 else 0
        eps_chg = (eps_vals[i] - eps_vals[i - 1]) / eps_vals[i - 1] if eps_vals[i - 1] != 0 else 0
        if ebit_chg != 0:
            leverages.append(eps_chg / ebit_chg)

    if not leverages:
        return MetricResult(error="Cannot compute financial leverage")

    leverages = [max(-5, min(5, x)) for x in leverages]
    avg = float(np.mean(leverages))

    return MetricResult(
        value=avg,
        raw_series=leverages,
        metadata={"years_used": len(leverages)},
    )


def compute_growth_quality(data: dict, params: dict) -> MetricResult:
    """Growth Quality Grade based on 4-lever decomposition.

    High Quality: Revenue growth + Operating leverage > 1
    Moderate: Revenue growth + some pricing power
    Low Quality: Financial leverage driven
    Risky: Earnings growth primarily from financial leverage
    """
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years + 1)

    revenue = pd.to_numeric(df["revenue"], errors="coerce").dropna()
    op = pd.to_numeric(df["operating_profit"], errors="coerce").dropna()
    eps = pd.to_numeric(df["eps"], errors="coerce").dropna()

    n = min(len(revenue), len(op), len(eps))
    if n < 3:
        return MetricResult(error="Insufficient data for growth quality")

    # Compute average operating and financial leverage
    rev = revenue.tail(n).values
    ebit = op.tail(n).values
    eps_vals = eps.tail(n).values

    op_levs = []
    fin_levs = []
    for i in range(1, len(rev)):
        rev_chg = (rev[i] - rev[i - 1]) / rev[i - 1] if rev[i - 1] != 0 else 0
        ebit_chg = (ebit[i] - ebit[i - 1]) / ebit[i - 1] if ebit[i - 1] != 0 else 0
        eps_chg = (eps_vals[i] - eps_vals[i - 1]) / eps_vals[i - 1] if eps_vals[i - 1] != 0 else 0

        if rev_chg != 0:
            op_levs.append(max(-5, min(5, ebit_chg / rev_chg)))
        if ebit_chg != 0:
            fin_levs.append(max(-5, min(5, eps_chg / ebit_chg)))

    avg_op_lev = np.mean(op_levs) if op_levs else 1.0
    avg_fin_lev = np.mean(fin_levs) if fin_levs else 1.0

    # Revenue CAGR
    rev_cagr = ((rev[-1] / rev[0]) ** (1 / (len(rev) - 1)) - 1) if rev[0] > 0 and rev[-1] > 0 else 0

    if rev_cagr > 0.10 and avg_op_lev > 1.0:
        grade = "high_quality"
    elif rev_cagr > 0.05 and avg_op_lev >= 0.8:
        grade = "moderate"
    elif avg_fin_lev > 1.5:
        grade = "risky"
    else:
        grade = "low_quality"

    flags = [f"growth_quality_{grade}"]

    return MetricResult(
        value=grade,
        flags=flags,
        metadata={
            "avg_operating_leverage": float(avg_op_lev),
            "avg_financial_leverage": float(avg_fin_lev),
            "revenue_cagr": float(rev_cagr * 100),
        },
    )


def compute_growth_consistency(data: dict, params: dict) -> MetricResult:
    """Standard deviation of year-over-year growth rate for a field."""
    field = params.get("field", "revenue")
    years = params.get("years", 10)
    df = _get_annual_rows(data["financials"], years + 1)

    if field not in df.columns:
        return MetricResult(error=f"Field '{field}' not in financials")

    values = pd.to_numeric(df[field], errors="coerce").dropna()
    if len(values) < 4:
        return MetricResult(error="Insufficient data for growth consistency")

    yoy_growth = []
    vals = values.values
    for i in range(1, len(vals)):
        if vals[i - 1] != 0:
            yoy_growth.append((vals[i] - vals[i - 1]) / vals[i - 1] * 100)

    if len(yoy_growth) < 3:
        return MetricResult(error="Insufficient growth data points")

    std = float(np.std(yoy_growth))
    return MetricResult(
        value=std,
        raw_series=yoy_growth,
        metadata={"years_used": len(yoy_growth)},
    )


def compute_share_dilution(data: dict, params: dict) -> MetricResult:
    """Growth in equity capital over N years, adjusted for bonus issues and stock splits.

    Detects single-year spikes >50% (likely bonus/split events) and separates
    organic dilution (genuine equity raises) from mechanical restructuring.
    """
    years = params.get("years", 10)
    df = _get_annual_rows(data["balance_sheet"], years + 1)

    if "equity_capital" not in df.columns:
        return MetricResult(error="No equity_capital column")

    values = pd.to_numeric(df["equity_capital"], errors="coerce").dropna()
    if len(values) < 2:
        return MetricResult(error="Insufficient equity capital data")

    start = float(values.iloc[0])
    end = float(values.iloc[-1])

    if start <= 0:
        return MetricResult(error="Non-positive starting equity capital")

    raw_growth_pct = (end / start - 1) * 100

    # Compute YoY changes and detect bonus/split events
    flags = []
    yoy_changes = []
    spike_years = []
    for i in range(1, len(values)):
        prev = float(values.iloc[i - 1])
        curr = float(values.iloc[i])
        if prev > 0:
            yoy_pct = (curr / prev - 1) * 100
            yoy_changes.append(yoy_pct)
            # >50% single-year jump is almost certainly a bonus issue or stock split
            if yoy_pct > 50:
                spike_years.append(i)
                flags.append(f"possible_bonus_split_year_{i}")
        else:
            yoy_changes.append(0.0)

    # Compute organic dilution: compound only the non-spike year changes
    organic_changes = [
        c for i, c in enumerate(yoy_changes) if (i + 1) not in spike_years
    ]
    if organic_changes:
        compound = 1.0
        for c in organic_changes:
            compound *= (1 + c / 100)
        adjusted_growth_pct = (compound - 1) * 100
    else:
        adjusted_growth_pct = 0.0  # All changes were bonus/split events

    # Flag if raw shows huge growth but organic is minimal
    if raw_growth_pct > 100 and adjusted_growth_pct < 30:
        flags.append("bonus_split_adjusted")

    if adjusted_growth_pct > 30:
        flags.append("high_dilution")
    elif adjusted_growth_pct < 5:
        flags.append("minimal_dilution")

    return MetricResult(
        value=float(adjusted_growth_pct),
        flags=flags,
        metadata={
            "start": start,
            "end": end,
            "raw_growth_pct": float(raw_growth_pct),
            "organic_yoy_changes": organic_changes,
            "spike_years_count": len(spike_years),
        },
    )
