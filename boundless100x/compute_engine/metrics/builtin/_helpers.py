"""Shared helpers for metric computation."""

import re

import numpy as np
import pandas as pd

MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def period_end_date(label) -> pd.Timestamp | None:
    """Turn a Screener column label such as 'Mar 2020' into that period's end.

    Returns None for labels this cannot parse (e.g. 'TTM') — a period whose
    actual end date is unknown must not be silently treated as any date, since
    comparing it against a cutoff either way would be a guess dressed up as a
    fact.
    """
    match = re.match(r"\s*([A-Za-z]{3})\w*\s+(\d{4})", str(label))
    if not match:
        return None
    month = MONTH_NUMBERS.get(match.group(1).lower())
    if month is None:
        return None
    return pd.Timestamp(year=int(match.group(2)), month=month, day=1) + pd.offsets.MonthEnd(0)


def detect_fcf_outliers(
    fcf_series: np.ndarray,
    threshold_std: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    """Detect outlier years in FCF series (likely M&A or one-time events).

    Uses Median Absolute Deviation (MAD) which is robust to single outliers,
    unlike standard deviation which is pulled by the outlier itself.

    Only flags NEGATIVE outliers (large capex spikes from M&A) — positive
    outliers (e.g., asset sales) are left untouched.

    Args:
        fcf_series: Array of FCF values (CFO + CFI) per year.
        threshold_std: Number of MAD-scaled standard deviations for outlier detection.

    Returns:
        (clean_series, flags) where clean_series has outliers replaced with NaN.
    """
    flags: list[str] = []

    if len(fcf_series) < 3:
        return fcf_series.copy(), flags

    median = np.median(fcf_series)
    mad = np.median(np.abs(fcf_series - median))

    if mad == 0:
        # All values are essentially the same — no outliers
        return fcf_series.copy(), flags

    # Scale MAD to approximate standard deviation (for normal distributions)
    mad_std = mad * 1.4826

    clean = fcf_series.copy().astype(float)
    for i, val in enumerate(fcf_series):
        deviation = abs(val - median) / mad_std
        # Only flag negative outliers (M&A-driven capex spikes)
        if deviation > threshold_std and val < median:
            flags.append(f"fcf_outlier_year_{i}_value_{val:.0f}")
            clean[i] = np.nan

    return clean, flags


# Below this many observations, averaging endpoints leaves too little signal
# to be worth the smoothing.
SMOOTHING_MIN_POINTS = 6


def smoothed_endpoints(values, min_points: int = SMOOTHING_MIN_POINTS) -> tuple[float, float, bool]:
    """Average each end over two periods once the series is long enough.

    A single acquisition, write-off or spike year otherwise sets the whole
    change being measured. Returns (start, end, smoothed).
    """
    if len(values) >= min_points:
        return float(values.iloc[:2].mean()), float(values.iloc[-2:].mean()), True
    return float(values.iloc[0]), float(values.iloc[-1]), False
