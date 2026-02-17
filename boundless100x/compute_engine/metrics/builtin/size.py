"""Size metrics: Market cap, institutional holding, analyst coverage, turnover, promoter."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_market_cap(data: dict, params: dict) -> MetricResult:
    """Market Cap from metadata (already in ₹ Cr from Screener.in)."""
    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None:
        return MetricResult(error="No market cap in metadata")

    flags = []
    if mcap < 5000:
        flags.append("small_cap")
    elif mcap < 20000:
        flags.append("mid_cap")
    else:
        flags.append("large_cap")

    return MetricResult(value=float(mcap), flags=flags)


def compute_institutional_holding(data: dict, params: dict) -> MetricResult:
    """FII + DII holding from latest shareholding quarter."""
    sh = data["shareholding"]
    if sh.empty:
        return MetricResult(error="No shareholding data")

    latest = sh.iloc[-1]
    fii = pd.to_numeric(latest.get("fii_pct"), errors="coerce")
    dii = pd.to_numeric(latest.get("dii_pct"), errors="coerce")

    if pd.isna(fii) and pd.isna(dii):
        return MetricResult(error="Missing FII/DII data")

    fii = 0.0 if pd.isna(fii) else float(fii)
    dii = 0.0 if pd.isna(dii) else float(dii)
    total = fii + dii

    flags = []
    if total < 5:
        flags.append("low_institutional_ownership")
    elif total > 40:
        flags.append("heavily_institutional")

    return MetricResult(
        value=total,
        flags=flags,
        metadata={"fii_pct": fii, "dii_pct": dii},
    )


def compute_analyst_count(data: dict, params: dict) -> MetricResult:
    """Number of analysts covering the company."""
    ac = data.get("analyst_coverage", {})
    count = ac.get("count")
    if count is None:
        return MetricResult(error="No analyst coverage data")

    flags = []
    if count <= 3:
        flags.append("under_researched")
    elif count <= 5:
        flags.append("lightly_covered")

    return MetricResult(value=float(count), flags=flags)


def compute_turnover_ratio(data: dict, params: dict) -> MetricResult:
    """Daily Turnover Ratio = Avg Daily Volume × Price / Market Cap × 100.

    Measures liquidity. Very low = hard to accumulate/exit.
    """
    price_df = data["price"]
    if price_df.empty or len(price_df) < 20:
        return MetricResult(error="Insufficient price data")

    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None or mcap == 0:
        return MetricResult(error="No market cap for turnover calculation")

    # Last 90 trading days average
    recent = price_df.tail(90)
    avg_volume = recent["volume"].mean()
    avg_price = recent["close"].mean()

    if pd.isna(avg_volume) or pd.isna(avg_price):
        return MetricResult(error="Cannot compute average volume/price")

    # Daily turnover value in Cr (volume × price / 1e7)
    daily_turnover_cr = avg_volume * avg_price / 1e7
    ratio = float(daily_turnover_cr / mcap * 100)

    return MetricResult(value=ratio)


def compute_promoter_trend(data: dict, params: dict) -> MetricResult:
    """Promoter holding trend over N quarters."""
    quarters = params.get("quarters", 20)
    sh = data["shareholding"]
    if sh.empty:
        return MetricResult(error="No shareholding data")

    promoter = pd.to_numeric(sh["promoter_pct"], errors="coerce").dropna()
    if len(promoter) < 4:
        return MetricResult(error="Insufficient promoter holding data")

    latest = float(promoter.iloc[-1])
    earliest = float(promoter.iloc[0])
    change = latest - earliest

    flags = []
    if change > 2:
        flags.append("promoter_increasing_stake")
    elif change < -5:
        flags.append("promoter_reducing_stake")

    return MetricResult(
        value=latest,
        raw_series=promoter.tolist(),
        flags=flags,
        metadata={"change_pp": change, "quarters_used": len(promoter)},
    )


def compute_promoter_pledge(data: dict, params: dict) -> MetricResult:
    """Promoter pledge percentage (from BSE data or estimated as 0)."""
    sh = data["shareholding"]

    # Try BSE supplemental data first
    bse_sh = data.get("shareholding_bse")
    if bse_sh is not None and not bse_sh.empty and "promoter_pledge_pct" in bse_sh.columns:
        pledge = pd.to_numeric(bse_sh["promoter_pledge_pct"], errors="coerce").dropna()
        if len(pledge) > 0:
            val = float(pledge.iloc[-1])
            flags = []
            if val > 10:
                flags.append("promoter_pledge_red_flag")
            return MetricResult(value=val, flags=flags)

    # Default: assume 0 if not available
    return MetricResult(
        value=0.0,
        metadata={"note": "Pledge data not available, defaulting to 0"},
    )


def compute_owner_operator(data: dict, params: dict) -> MetricResult:
    """Owner-operator signal based on promoter holding level."""
    min_holding = params.get("min_promoter_holding", 40)
    sh = data["shareholding"]
    if sh.empty:
        return MetricResult(error="No shareholding data")

    promoter = pd.to_numeric(sh["promoter_pct"], errors="coerce").dropna()
    if len(promoter) == 0:
        return MetricResult(error="No promoter holding data")

    latest = float(promoter.iloc[-1])

    if latest >= 50:
        category = "founder_led_high_holding"
    elif latest >= min_holding:
        category = "founder_led_moderate"
    elif latest >= 20:
        category = "professional_mgmt"
    else:
        category = "low_promoter"

    return MetricResult(
        value=category,
        metadata={"promoter_pct": latest},
    )
