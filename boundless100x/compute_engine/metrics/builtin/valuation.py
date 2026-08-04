"""Valuation metrics: P/E, PEG, trailing PEG, EV/EBITDA, DCF, reverse DCF."""

import re

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import detect_fcf_outliers
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_pe_ttm(data: dict, params: dict) -> MetricResult:
    """P/E ratio from metadata (TTM)."""
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None:
        return MetricResult(error="No P/E in metadata")

    flags = []
    if pe > 80:
        flags.append("very_expensive_pe")
    elif pe > 50:
        flags.append("expensive_pe")
    elif pe < 15:
        flags.append("cheap_pe")

    return MetricResult(value=float(pe), flags=flags)


def compute_peg(data: dict, params: dict) -> MetricResult:
    """PEG = P/E / EPS CAGR (5yr forward estimate, using historical as proxy)."""
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None or pe <= 0:
        return MetricResult(error="No P/E for PEG")

    df = _get_annual_rows(data["financials"], 6)
    eps = pd.to_numeric(df["eps"], errors="coerce").dropna()
    if len(eps) < 2:
        return MetricResult(error="Insufficient EPS data for PEG")

    start = float(eps.iloc[0])
    end = float(eps.iloc[-1])
    actual_years = len(eps) - 1

    if start <= 0 or end <= 0:
        return MetricResult(error="Non-positive EPS for PEG")

    eps_cagr = ((end / start) ** (1 / actual_years) - 1) * 100
    if eps_cagr <= 0:
        return MetricResult(error="Negative EPS CAGR, PEG undefined")

    peg = pe / eps_cagr

    flags = []
    if peg < 1.0:
        flags.append("attractively_valued_peg")
    elif peg > 2.5:
        flags.append("expensive_peg")

    return MetricResult(
        value=float(peg),
        flags=flags,
        metadata={"pe": pe, "eps_cagr": eps_cagr},
    )


def compute_trailing_peg(data: dict, params: dict) -> MetricResult:
    """Trailing PEG = P/E ÷ trailing 3yr PAT CAGR."""
    cagr_years = params.get("cagr_years", 3)
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None or pe <= 0:
        return MetricResult(error="No P/E for trailing PEG")

    df = _get_annual_rows(data["financials"], cagr_years + 1)
    pat = pd.to_numeric(df["pat"], errors="coerce").dropna()
    if len(pat) < 2:
        return MetricResult(error="Insufficient PAT data")

    start = float(pat.iloc[0])
    end = float(pat.iloc[-1])
    actual_years = len(pat) - 1

    if start <= 0 or end <= 0:
        return MetricResult(error="Non-positive PAT for trailing PEG")

    pat_cagr = ((end / start) ** (1 / actual_years) - 1) * 100
    if pat_cagr <= 0:
        return MetricResult(error="Negative PAT CAGR, trailing PEG undefined")

    tpeg = pe / pat_cagr

    flags = []
    if tpeg < 1.0:
        flags.append("attractive_trailing_peg")

    return MetricResult(
        value=float(tpeg),
        flags=flags,
        metadata={"pe": pe, "pat_cagr": pat_cagr, "years": actual_years},
    )


def compute_ev_ebitda(data: dict, params: dict) -> MetricResult:
    """EV/EBITDA = (Market Cap + Debt - Cash) / (Operating Profit + Depreciation)."""
    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None:
        return MetricResult(error="No market cap for EV/EBITDA")

    bs = _get_annual_rows(data["balance_sheet"], 1)
    fin = _get_annual_rows(data["financials"], 1)

    if bs.empty or fin.empty:
        return MetricResult(error="No BS/financials for EV/EBITDA")

    debt = pd.to_numeric(bs["borrowings"], errors="coerce").iloc[-1]
    debt = 0.0 if pd.isna(debt) else float(debt)

    op = pd.to_numeric(fin["operating_profit"], errors="coerce").iloc[-1]
    dep = pd.to_numeric(fin["depreciation"], errors="coerce").iloc[-1]

    if pd.isna(op) or pd.isna(dep):
        return MetricResult(error="Missing EBITDA components")

    ebitda = float(op) + float(dep)
    if ebitda <= 0:
        return MetricResult(error="Non-positive EBITDA")

    # Simplified EV (no cash subtraction — Screener doesn't provide cash directly)
    ev = mcap + debt
    ev_ebitda = ev / ebitda

    return MetricResult(
        value=float(ev_ebitda),
        metadata={"ev": ev, "ebitda": ebitda},
    )


MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# How far back a fiscal year end may reach for a traded price before the year
# is treated as uncovered — comfortably more than any exchange holiday run.
PRICE_LOOKBACK_DAYS = 45


def _period_end(label: str) -> pd.Timestamp | None:
    """Turn a Screener column label such as 'Mar 2020' into that period's end."""
    match = re.match(r"\s*([A-Za-z]{3})\w*\s+(\d{4})", str(label))
    if not match:
        return None
    month = MONTH_NUMBERS.get(match.group(1).lower())
    if month is None:
        return None
    return pd.Timestamp(year=int(match.group(2)), month=month, day=1) + pd.offsets.MonthEnd(0)


def _close_on_or_before(price: pd.DataFrame, when: pd.Timestamp) -> float | None:
    """The last traded close at or before a date, within the lookback window."""
    prior = price[price["date"] <= when]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    if (when - row["date"]).days > PRICE_LOOKBACK_DAYS:
        return None
    close = float(row["close"])
    return close if close > 0 else None


def compute_pe_percentile(data: dict, params: dict) -> MetricResult:
    """Where today's P/E sits in the company's own traded P/E history.

    The band must be built from the price at each past year end divided by that
    year's EPS. Dividing today's price by past EPS instead produces a rescaled
    reciprocal of the earnings series, so the percentile tracks earnings growth
    rather than valuation — and any company earning near a high scores as
    maximally cheap however dearly it actually trades.
    """
    years = params.get("years", 10)
    meta = data.get("metadata", {})
    current_pe = meta.get("Stock P/E")
    if current_pe is None:
        return MetricResult(error="No current P/E")

    price = data.get("price")
    if price is None or len(price) == 0:
        return MetricResult(error="No price history for historical P/E band")

    # The band divides each past year-end close by that year's as-reported
    # EPS, so it needs the raw traded close. Fetches after the adj_close
    # schema carry both series; older caches hold a single close whose
    # adjustment status is unknown — the band is computed either way, but the
    # basis is recorded so a distorted read is traceable.
    price_basis = (
        "raw_close" if "adj_close" in price.columns else "legacy_close_unknown_adjustment"
    )

    price = price.copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce", utc=True).dt.tz_localize(None)
    price = price.dropna(subset=["date"]).sort_values("date")
    if price.empty:
        return MetricResult(error="No usable price dates for historical P/E band")

    df = _get_annual_rows(data["financials"], years)
    if "eps" not in df.columns or "year" not in df.columns:
        return MetricResult(error="Financials lack eps/year for PE percentile")

    historical_pes = []
    for label, raw_eps in zip(df["year"], pd.to_numeric(df["eps"], errors="coerce")):
        if pd.isna(raw_eps) or raw_eps <= 0:
            continue
        period_end = _period_end(label)
        if period_end is None:
            continue
        close = _close_on_or_before(price, period_end)
        if close is None:
            continue
        historical_pes.append(close / float(raw_eps))

    if len(historical_pes) < 5:
        return MetricResult(
            error=f"Only {len(historical_pes)} years with both price and positive EPS"
        )

    # Percentile of current PE in historical distribution
    below = sum(1 for pe in historical_pes if pe <= current_pe)
    percentile = below / len(historical_pes) * 100

    flags = []
    if percentile > 75:
        flags.append("pe_above_historical_75th")
    elif percentile < 25:
        flags.append("pe_below_historical_25th")
    if price_basis != "raw_close":
        # Adjusted closes understate past prices, so the band reads cheaper
        # than it was and today's percentile reads higher. Say so rather than
        # presenting a distorted percentile as settled.
        flags.append("pe_band_legacy_price_basis")

    return MetricResult(
        value=float(percentile),
        raw_series=[round(pe, 2) for pe in historical_pes],
        flags=flags,
        metadata={
            "years_used": len(historical_pes),
            "pe_min": round(min(historical_pes), 2),
            "pe_max": round(max(historical_pes), 2),
            "pe_median": round(float(np.median(historical_pes)), 2),
            "current_pe": float(current_pe),
            "price_basis": price_basis,
        },
    )


def compute_dcf_margin(data: dict, params: dict) -> MetricResult:
    """DCF Margin of Safety = (Intrinsic Value - Current Price) / Current Price × 100."""
    projection_years = params.get("projection_years", 10)
    terminal_growth = params.get("terminal_growth", 0.04)
    discount_rate = params.get("discount_rate", 0.12)

    meta = data.get("metadata", {})
    current_price = meta.get("Current Price")
    if current_price is None or current_price <= 0:
        return MetricResult(error="No current price for DCF")

    cf = _get_annual_rows(data["cashflow"], 5)
    fin = _get_annual_rows(data["financials"], 6)

    cfo = pd.to_numeric(cf["cfo"], errors="coerce").dropna()
    cfi = pd.to_numeric(cf["cfi"], errors="coerce").dropna()

    n = min(len(cfo), len(cfi))
    if n < 3:
        return MetricResult(error="Insufficient cash flow for DCF")

    fcf_series = (cfo.tail(n).values + cfi.tail(n).values)
    avg_fcf_raw = float(np.mean(fcf_series))

    # Detect FCF outliers (likely M&A years)
    clean_fcf, outlier_flags = detect_fcf_outliers(fcf_series)
    avg_fcf = float(np.nanmean(clean_fcf)) if not np.all(np.isnan(clean_fcf)) else avg_fcf_raw

    if avg_fcf <= 0:
        # Even after removing outliers, FCF is negative
        all_flags = ["negative_average_fcf"] + outlier_flags
        if outlier_flags and avg_fcf_raw <= 0 and avg_fcf != avg_fcf_raw:
            all_flags.append("negative_fcf_even_after_outlier_removal")
        return MetricResult(
            value=-100.0,
            flags=all_flags,
            metadata={
                "avg_fcf_raw": avg_fcf_raw,
                "avg_fcf_organic": avg_fcf,
                "outlier_years": len(outlier_flags),
            },
        )

    # Estimate FCF growth from revenue CAGR
    revenue = pd.to_numeric(fin["revenue"], errors="coerce").dropna()
    if len(revenue) >= 2 and revenue.iloc[0] > 0:
        rev_cagr = (revenue.iloc[-1] / revenue.iloc[0]) ** (1 / (len(revenue) - 1)) - 1
        fcf_growth = min(rev_cagr, 0.25)  # Cap at 25%
    else:
        fcf_growth = 0.10

    # DCF: project FCF, discount, add terminal
    pv_fcfs = 0.0
    projected_fcf = avg_fcf
    for yr in range(1, projection_years + 1):
        projected_fcf *= (1 + fcf_growth)
        pv_fcfs += projected_fcf / (1 + discount_rate) ** yr

    terminal_value = projected_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
    intrinsic_total = pv_fcfs + pv_terminal

    # Per share: get shares from equity capital
    bs = _get_annual_rows(data.get("balance_sheet", pd.DataFrame()), 1)
    face_value = meta.get("Face Value", 1)
    if not bs.empty and "equity_capital" in bs.columns:
        eq_cap = pd.to_numeric(bs["equity_capital"], errors="coerce").iloc[-1]
        if not pd.isna(eq_cap) and face_value and face_value > 0:
            shares_cr = eq_cap / face_value
            intrinsic_per_share = intrinsic_total / shares_cr if shares_cr > 0 else 0
        else:
            intrinsic_per_share = 0
    else:
        intrinsic_per_share = 0

    if intrinsic_per_share <= 0:
        return MetricResult(error="Cannot compute per-share intrinsic value")

    margin = (intrinsic_per_share - current_price) / current_price * 100

    flags = []
    if margin > 20:
        flags.append("dcf_undervalued")
    elif margin < -30:
        flags.append("dcf_overvalued")

    return MetricResult(
        value=float(margin),
        flags=flags,
        metadata={
            "intrinsic_per_share": float(intrinsic_per_share),
            "current_price": current_price,
            "fcf_growth_assumed": float(fcf_growth * 100),
        },
    )


def compute_reverse_dcf(data: dict, params: dict) -> MetricResult:
    """Reverse DCF: solve for the growth rate implied by current market price."""
    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None or mcap <= 0:
        return MetricResult(error="No market cap for reverse DCF")

    cf = _get_annual_rows(data["cashflow"], 5)
    cfo = pd.to_numeric(cf["cfo"], errors="coerce").dropna()
    cfi = pd.to_numeric(cf["cfi"], errors="coerce").dropna()
    n = min(len(cfo), len(cfi))
    if n < 3:
        return MetricResult(error="Insufficient cash flow for reverse DCF")

    raw_fcf_series = cfo.tail(n).values + cfi.tail(n).values
    clean_fcf, outlier_flags = detect_fcf_outliers(raw_fcf_series)
    avg_fcf = float(np.nanmean(clean_fcf)) if not np.all(np.isnan(clean_fcf)) else float(np.mean(raw_fcf_series))
    if avg_fcf <= 0:
        return MetricResult(
            error="Negative average FCF, reverse DCF undefined",
            flags=outlier_flags,
        )

    discount_rate = float(params.get("discount_rate", 0.12))
    terminal_growth = float(params.get("terminal_growth", 0.04))
    projection_years = int(params.get("projection_years", 10))

    # Binary search for implied growth, bounded to [-10%, +50%]. A company
    # priced beyond a bound pins to it — directionally meaningful, but the
    # exact value is then an artifact of the bound rather than a measurement,
    # and it feeds both scoring and the price-gate veto, so it must say so.
    low, high = -0.10, 0.50
    for _ in range(50):
        mid = (low + high) / 2
        pv = 0.0
        proj_fcf = avg_fcf
        for yr in range(1, projection_years + 1):
            proj_fcf *= (1 + mid)
            pv += proj_fcf / (1 + discount_rate) ** yr
        tv = proj_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv += tv / (1 + discount_rate) ** projection_years

        if pv < mcap:
            low = mid
        else:
            high = mid

    implied_growth = (low + high) / 2 * 100
    saturated_at = None
    if low > 0.50 - 1e-6:
        saturated_at = "ceiling"  # market implies 50%+ growth — search ran out of room
    elif high < -0.10 + 1e-6:
        saturated_at = "floor"    # market prices below a 10% perpetual decline

    # Compare to actual revenue CAGR
    fin = _get_annual_rows(data["financials"], 6)
    revenue = pd.to_numeric(fin["revenue"], errors="coerce").dropna()
    actual_cagr = None
    if len(revenue) >= 2 and revenue.iloc[0] > 0:
        actual_cagr = ((revenue.iloc[-1] / revenue.iloc[0]) ** (1 / (len(revenue) - 1)) - 1) * 100

    flags = []
    if saturated_at is not None:
        flags.append("reverse_dcf_saturated")
    if actual_cagr is not None and implied_growth > actual_cagr * 1.5:
        flags.append("reverse_dcf_overpriced")
    elif actual_cagr is not None and implied_growth < actual_cagr * 0.7:
        flags.append("reverse_dcf_underpriced")

    return MetricResult(
        value=float(implied_growth),
        flags=flags,
        metadata={
            "actual_cagr": actual_cagr,
            "avg_fcf": avg_fcf,
            "saturated_at": saturated_at,
            "search_bounds_pct": [-10.0, 50.0],
        },
    )


def compute_earnings_yield_spread(data: dict, params: dict) -> MetricResult:
    """Earnings Yield (1/PE) minus the India 10yr G-Sec yield."""
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None or pe <= 0:
        return MetricResult(error="No P/E for earnings yield")

    earnings_yield = 100.0 / pe
    gsec_yield = float(params.get("gsec_yield_pct", 7.0))
    spread = earnings_yield - gsec_yield

    flags = []
    if spread > 0:
        flags.append("earnings_yield_above_gsec")
    else:
        flags.append("gsec_more_attractive")

    return MetricResult(
        value=float(spread),
        flags=flags,
        metadata={"earnings_yield": earnings_yield, "gsec_yield": gsec_yield},
    )
