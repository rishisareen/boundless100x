"""Shared helpers for metric computation."""

import numpy as np


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
