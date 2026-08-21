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


def quarter_index(label) -> int | None:
    """An orderable index for a quarter label, or None when it cannot be read.

    `Mar 2026` sits one above `Dec 2025`, so "four quarters back" is arithmetic
    on real periods rather than a row offset — the distinction that stops a
    missing quarter silently pairing a period against one five or six earlier.
    Derived from `period_end_date` so there is exactly one label parser: the
    three-month bucket comes from the calendar month, which indexes a
    Jan/Apr/Jul/Oct filer just as consecutively as a March one.
    """
    end = period_end_date(label)
    return None if end is None else end.year * 4 + (end.month - 1) // 3


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


# ── Treasury deployment is not capital expenditure ────────────────────────
#
# Free cash flow here is `CFO + CFI`, and that is right for a company whose
# investing line is plant. It is wrong for a cash-rich one, because CFI also
# carries money moved into mutual funds, bonds and term deposits — cash that
# has been *parked*, not spent, and that a buyer of the whole company would get
# back.
#
# CAPLIPOINT is the case. Over five years its financial investments grew ₹848
# Cr against ₹495 Cr of growth in plant and work-in-progress, so most of its
# "negative free cash flow" was treasury. The consequences ran through four
# metrics and one gate: average FCF read ₹10.8 Cr for a company holding ₹2,875
# Cr of liquid assets, the DCF returned an intrinsic value of ₹43 against a
# ₹2,561 price (-98.3%, scored zero), the reverse DCF pinned at its +50%
# ceiling, and the `reverse_dcf_overpriced` veto that fired off it FAILED the
# entry-price eligibility gate whose own PEG conditions had passed.
#
# **The correction is an estimate and must not pretend otherwise.** The
# balance sheet gives the year-end *stock* of investments, so a rise is
# inferred to be cash that went in — but a rise can also be a mark-to-market
# gain, which consumed no cash and should not be added back. It is therefore
# reported beside the unadjusted figure rather than replacing it, and a metric
# that uses it says so in a flag.
#
# Only *increases* are added back. A fall in investments is cash coming out,
# and treating that as a deduction would flatter a company liquidating its
# treasury to cover an operating shortfall — the exact case the reader needs
# to see.

TREASURY_ADJUSTED_FLAG = "fcf_adjusted_for_treasury"

# Below this share of the reported investing outflow, treasury movement is
# noise rather than a distortion worth telling the reader about.
_TREASURY_MATERIAL_SHARE = 0.15


def treasury_flows(balance_sheet: pd.DataFrame) -> dict[str, float]:
    """Year label -> cash inferred to have moved INTO financial investments.

    Empty when the balance sheet carries no `investments` column, which is the
    honest answer: with no stock to difference, nothing can be inferred and the
    unadjusted investing line stands.
    """
    if balance_sheet is None or getattr(balance_sheet, "empty", True):
        return {}
    if not {"investments", "year"} <= set(balance_sheet.columns):
        return {}

    investments = pd.to_numeric(balance_sheet["investments"], errors="coerce")
    labels = balance_sheet["year"].astype(str)

    flows: dict[str, float] = {}
    previous = None
    for label, value in zip(labels, investments):
        if pd.isna(value):
            previous = None
            continue
        if previous is not None:
            flows[label] = max(0.0, float(value) - previous)
        previous = float(value)
    return flows


def operating_free_cash_flow(
    cashflow: pd.DataFrame, balance_sheet: pd.DataFrame
) -> tuple[pd.Series, pd.Series, dict]:
    """(reported_fcf, operating_fcf, detail) aligned to `cashflow`'s rows.

    `operating_fcf` adds back money that went into financial investments, so it
    answers "what did the operating business leave over" rather than "what was
    the net movement across every investing decision". Both are returned
    because they answer different questions and a reader is entitled to both.

    Aligned on the period LABEL rather than on row position: the two frames are
    filtered independently and a balance sheet carrying an interim column would
    otherwise pair each year against its neighbour.
    """
    empty = pd.Series(dtype=float)
    if cashflow is None or getattr(cashflow, "empty", True):
        return empty, empty, {"adjusted": False, "reason": "no cash flow data"}
    if not {"cfo", "cfi"} <= set(cashflow.columns):
        return empty, empty, {"adjusted": False, "reason": "no cfo/cfi columns"}

    cfo = pd.to_numeric(cashflow["cfo"], errors="coerce")
    cfi = pd.to_numeric(cashflow["cfi"], errors="coerce")
    reported = (cfo + cfi)

    flows = treasury_flows(balance_sheet)
    if not flows or "year" not in cashflow.columns:
        return (
            reported.dropna(),
            reported.dropna(),
            {"adjusted": False, "reason": "no investments column to difference"},
        )

    added_back = cashflow["year"].astype(str).map(flows).fillna(0.0)
    operating = (reported + added_back)

    outflow = float(cfi[cfi < 0].sum())
    total_added = float(added_back.sum())
    material = bool(outflow) and (total_added / abs(outflow)) >= _TREASURY_MATERIAL_SHARE

    return (
        reported.dropna(),
        operating.dropna(),
        {
            "adjusted": material,
            "treasury_added_back": round(total_added, 2),
            "reported_investing_outflow": round(outflow, 2),
            "years_adjusted": int((added_back > 0).sum()),
        },
    )
