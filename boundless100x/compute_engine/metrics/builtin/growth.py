"""Growth metrics: CAGR, 4-lever decomposition, quality grade, consistency.

v4: Added compute_price_lever(), compute_lever_decomposition_table(), and
    helper functions for the expanded growth report section.
"""

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


# ─── v4: New functions for expanded growth report section ───


def compute_price_lever(data: dict, params: dict) -> MetricResult:
    """Detect pricing power by comparing revenue growth to volume proxy.

    Uses inflation proxy: Real volume growth = Revenue growth - Inflation.
    Categories:
        - "strong_pricing_power": Revenue CAGR > Volume proxy + 3pp
        - "moderate_pricing": Revenue CAGR > Volume proxy + 1-3pp
        - "discounting": Revenue CAGR < inflation
        - "unknown": Insufficient data
    """
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years + 1)

    if "revenue" not in df.columns:
        return MetricResult(error="No revenue column for price lever")

    values = pd.to_numeric(df["revenue"], errors="coerce").dropna()
    if len(values) < 2:
        return MetricResult(error="Insufficient revenue data for price lever")

    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    actual_years = len(values) - 1

    if start <= 0 or end <= 0:
        return MetricResult(error="Non-positive revenue for price lever")

    revenue_cagr = ((end / start) ** (1 / actual_years) - 1) * 100

    # Proxy: Use average WPI/CPI to deflate revenue → estimate real volume growth
    inflation_avg = 5.0  # Default assumption; can be parameterized
    real_volume_growth = revenue_cagr - inflation_avg

    if real_volume_growth <= 0:
        signal = "discounting" if revenue_cagr < inflation_avg else "unknown"
    elif revenue_cagr > real_volume_growth + 3:
        signal = "strong_pricing_power"
    elif revenue_cagr > real_volume_growth + 1:
        signal = "moderate_pricing"
    else:
        signal = "unknown"

    return MetricResult(
        value=signal,
        metadata={
            "revenue_cagr": float(revenue_cagr),
            "estimated_volume_growth": float(real_volume_growth),
            "inflation_assumption": inflation_avg,
        },
    )


def _compute_operating_leverage_avg(df: pd.DataFrame, years: int) -> float:
    """Compute average operating leverage from financials DataFrame."""
    if "revenue" not in df.columns or "operating_profit" not in df.columns:
        return 1.0
    revenue = pd.to_numeric(df["revenue"], errors="coerce").dropna()
    op = pd.to_numeric(df["operating_profit"], errors="coerce").dropna()

    n = min(len(revenue), len(op))
    if n < 3:
        return 1.0

    rev = revenue.tail(n).values
    ebit = op.tail(n).values

    leverages = []
    for i in range(1, len(rev)):
        rev_chg = (rev[i] - rev[i - 1]) / rev[i - 1] if rev[i - 1] != 0 else 0
        ebit_chg = (ebit[i] - ebit[i - 1]) / ebit[i - 1] if ebit[i - 1] != 0 else 0
        if rev_chg != 0:
            leverages.append(max(-5, min(5, ebit_chg / rev_chg)))

    return float(np.mean(leverages)) if leverages else 1.0


def _compute_financial_leverage_avg(df: pd.DataFrame, years: int) -> float:
    """Compute average financial leverage from financials DataFrame."""
    if "operating_profit" not in df.columns or "eps" not in df.columns:
        return 1.0
    op = pd.to_numeric(df["operating_profit"], errors="coerce").dropna()
    eps = pd.to_numeric(df["eps"], errors="coerce").dropna()

    n = min(len(op), len(eps))
    if n < 3:
        return 1.0

    ebit = op.tail(n).values
    eps_vals = eps.tail(n).values

    leverages = []
    for i in range(1, len(ebit)):
        ebit_chg = (ebit[i] - ebit[i - 1]) / ebit[i - 1] if ebit[i - 1] != 0 else 0
        eps_chg = (eps_vals[i] - eps_vals[i - 1]) / eps_vals[i - 1] if eps_vals[i - 1] != 0 else 0
        if ebit_chg != 0:
            leverages.append(max(-5, min(5, eps_chg / ebit_chg)))

    return float(np.mean(leverages)) if leverages else 1.0


def _compute_cagr_from_series(values: pd.Series, years: int) -> float | None:
    """Compute CAGR from a numeric series. Returns percentage or None."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 2:
        return None
    start = float(vals.iloc[0])
    end = float(vals.iloc[-1])
    actual_years = len(vals) - 1
    if start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1 / actual_years) - 1) * 100


def _classify_volume_status(rev_cagr_5: float, price_lever_result: MetricResult) -> str:
    """Classify volume growth: Strong / Moderate / Weak / Declining."""
    est_volume = (price_lever_result.metadata or {}).get("estimated_volume_growth", 0)
    if est_volume >= 15:
        return "Strong organic volume growth"
    elif est_volume >= 8:
        return "Moderate volume growth"
    elif est_volume >= 0:
        return "Weak volume growth"
    else:
        return "Volume declining"


def _classify_op_lever(op_lever_avg: float, ebit_cagr: float | None, rev_cagr: float | None) -> str:
    """Classify operating leverage status."""
    if op_lever_avg >= 1.3:
        return "Strong positive operating leverage"
    elif op_lever_avg >= 1.0:
        return "Mild operating leverage"
    elif op_lever_avg >= 0.8:
        return "Neutral — margins stable"
    else:
        return "Negative operating leverage (margin compression)"


def _classify_fin_lever(fin_lever_avg: float) -> str:
    """Classify financial leverage status."""
    if fin_lever_avg >= 1.5:
        return "High financial leverage — debt-amplified"
    elif fin_lever_avg >= 1.1:
        return "Moderate positive financial leverage"
    elif fin_lever_avg >= 0.8:
        return "Neutral — minimal debt impact"
    else:
        return "Negative financial leverage (deleveraging)"


def _classify_price_lever(price_lever: MetricResult) -> str:
    """Classify price lever status into human-readable description."""
    signal = price_lever.value if price_lever.ok else "unknown"
    labels = {
        "strong_pricing_power": "Strong pricing power",
        "moderate_pricing": "Moderate pricing power",
        "discounting": "Weak — discounting to maintain volumes",
        "unknown": "Insufficient data",
    }
    return labels.get(signal, signal.replace("_", " ").title() if isinstance(signal, str) else "Unknown")


def _generate_volume_analysis(rev_cagr: float | None, price_lever: MetricResult, df: pd.DataFrame) -> str:
    """Generate analysis text for volume lever."""
    est_vol = (price_lever.metadata or {}).get("estimated_volume_growth", 0)
    rev_str = f"{rev_cagr:.1f}%" if rev_cagr is not None else "N/A"
    return (
        f"Revenue CAGR of {rev_str} with estimated real volume growth of "
        f"{est_vol:.1f}% (after deflating for ~5% inflation assumption)."
    )


def _generate_price_analysis(rev_cagr: float | None, price_lever: MetricResult, df: pd.DataFrame) -> str:
    """Generate analysis text for price lever."""
    signal = price_lever.value
    meta = price_lever.metadata or {}
    est_vol = meta.get("estimated_volume_growth", 0)
    rev = meta.get("revenue_cagr", 0)
    price_contribution = rev - est_vol if rev and est_vol else 0

    if signal == "strong_pricing_power":
        return (
            f"Revenue growth exceeds volume proxy by ~{price_contribution:.1f}pp, "
            f"indicating strong pricing power or favorable product mix shift."
        )
    elif signal == "moderate_pricing":
        return (
            f"Revenue growth exceeds volume proxy by ~{price_contribution:.1f}pp, "
            f"suggesting moderate pricing power — partly raw material pass-through."
        )
    elif signal == "discounting":
        return (
            f"Revenue growth below inflation proxy — company may be discounting "
            f"to maintain volumes. Pricing power is weak."
        )
    else:
        return "Insufficient data to determine pricing power signal."


def _generate_op_lever_analysis(
    op_lever_avg: float, ebit_cagr: float | None, rev_cagr: float | None, df: pd.DataFrame
) -> str:
    """Generate analysis text for operating lever."""
    ebit_str = f"{ebit_cagr:.1f}%" if ebit_cagr is not None else "N/A"
    rev_str = f"{rev_cagr:.1f}%" if rev_cagr is not None else "N/A"
    return (
        f"EBIT grew at {ebit_str} vs revenue at {rev_str} "
        f"(operating leverage ratio: {op_lever_avg:.2f}x)."
    )


def _generate_fin_lever_analysis(
    fin_lever_avg: float, eps_cagr: float | None, ebit_cagr: float | None, df: pd.DataFrame
) -> str:
    """Generate analysis text for financial lever."""
    eps_str = f"{eps_cagr:.1f}%" if eps_cagr is not None else "N/A"
    ebit_str = f"{ebit_cagr:.1f}%" if ebit_cagr is not None else "N/A"
    return (
        f"EPS CAGR of {eps_str} vs EBIT CAGR of {ebit_str} "
        f"(financial leverage ratio: {fin_lever_avg:.2f}x)."
    )


def _synthesize_growth_quality(
    pat_3: float | None,
    pat_5: float | None,
    op_lever: float,
    fin_lever: float,
    price_lever: MetricResult,
) -> dict:
    """Determine the primary growth driver and flag quality."""
    drivers = []

    # Check volume
    vol_growth = (price_lever.metadata or {}).get("estimated_volume_growth", 0)
    if vol_growth >= 10:
        drivers.append("Volume expansion")

    # Check pricing power
    if price_lever.value in ("strong_pricing_power", "moderate_pricing"):
        drivers.append("Price realization")

    # Check operating leverage
    if op_lever >= 1.1:
        drivers.append("Operating leverage")

    # Check financial leverage
    if fin_lever >= 1.3:
        drivers.append("Financial leverage")

    # Determine quality flag
    if "Volume expansion" in drivers and "Operating leverage" in drivers:
        quality = "high_quality"
    elif "Volume expansion" in drivers and "Price realization" in drivers:
        quality = "moderate"
    elif "Financial leverage" in drivers and len(drivers) == 1:
        quality = "risky"
    elif "Financial leverage" in drivers:
        quality = "low_quality"
    else:
        quality = "moderate"

    # Build narrative
    narrative_parts = []
    if pat_3 is not None and pat_5 is not None:
        narrative_parts.append(
            f"3-year PAT CAGR of {pat_3:.1f}% and 5-year PAT CAGR of {pat_5:.1f}%."
        )
    if drivers:
        narrative_parts.append(f"Growth primarily driven by: {', '.join(drivers)}.")

    if quality == "high_quality":
        narrative_parts.append(
            "This is high-quality growth — organic volume expansion "
            "amplified by operating scale benefits."
        )
    elif quality == "risky":
        narrative_parts.append(
            "FLAG: Growth is primarily driven by financial leverage "
            "(debt amplification). This is low-quality, unsustainable growth "
            "that amplifies returns in good times but accelerates losses in downturns."
        )
    elif quality == "low_quality":
        narrative_parts.append(
            "FLAG: Significant financial leverage contribution detected. "
            "Growth quality is compromised — not purely operating-driven."
        )

    return {
        "primary_drivers": drivers,
        "quality_flag": quality,
        "narrative": " ".join(narrative_parts),
    }


def _peg_verdict(trailing_peg: float | None, quality_flag: str) -> str:
    """One-sentence PEG verdict per the 100-bagger golden rule."""
    if trailing_peg is None:
        return "PEG cannot be computed (negative or zero earnings growth)."

    if trailing_peg < 1.0:
        if quality_flag in ("high_quality", "moderate"):
            return (
                f"Trailing PEG of {trailing_peg:.2f}x is below 1.0 — "
                f"the golden rule for 100-baggers. Combined with "
                f"{quality_flag.replace('_', ' ')} earnings drivers, "
                f"the valuation appears justified and attractive."
            )
        else:
            return (
                f"Trailing PEG of {trailing_peg:.2f}x is below 1.0, but the "
                f"growth quality is flagged as '{quality_flag.replace('_', ' ')}'. "
                f"Low PEG driven by leveraged or unsustainable growth is a "
                f"value trap signal — proceed with caution."
            )
    elif trailing_peg < 2.0:
        return (
            f"Trailing PEG of {trailing_peg:.2f}x is between 1.0-2.0 — "
            f"fairly valued relative to growth. Not a screaming bargain "
            f"but acceptable if growth quality is high."
        )
    else:
        return (
            f"Trailing PEG of {trailing_peg:.2f}x is above 2.0 — "
            f"the market is pricing in significantly higher growth "
            f"than recent history. Risk of valuation correction if "
            f"growth decelerates."
        )


def _ensure_operating_profit(df: pd.DataFrame) -> pd.DataFrame:
    """Derive operating_profit if not present in the DataFrame.

    Screener.in omits operating_profit for financial companies (banks, NBFCs).
    Fallback: operating_profit = revenue - expenses - depreciation.
    If expenses or depreciation are also missing, use pbt + interest as proxy.
    """
    if "operating_profit" in df.columns:
        return df

    df = df.copy()

    if "revenue" in df.columns and "expenses" in df.columns:
        rev = pd.to_numeric(df["revenue"], errors="coerce")
        exp = pd.to_numeric(df["expenses"], errors="coerce")
        depr = pd.to_numeric(df.get("depreciation", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["operating_profit"] = rev - exp - depr
    elif "pbt" in df.columns and "interest" in df.columns:
        # EBIT proxy: PBT + Interest (before other_income adjustment)
        pbt = pd.to_numeric(df["pbt"], errors="coerce")
        interest = pd.to_numeric(df["interest"], errors="coerce")
        df["operating_profit"] = pbt + interest
    else:
        # Cannot derive — leave missing; callers handle gracefully
        pass

    return df


def compute_lever_decomposition_table(data: dict, years: int = 5) -> dict:
    """Full 4-lever decomposition for the expanded report section.

    Returns a structured dict with:
    - earnings_profile: {pat_cagr_3yr, pat_cagr_5yr}
    - lever_table: [{lever, status, analysis}, ...]
    - growth_synthesis: {primary_drivers, quality_flag, narrative}
    - valuation_check: {current_pe, pat_cagr_5yr, trailing_peg, verdict}

    This is consumed by the Jinja2 report template.
    """
    df = _get_annual_rows(data["financials"], years + 1)
    df = _ensure_operating_profit(df)

    # Compute all required CAGRs
    rev_cagr_3 = _compute_cagr_from_series(df["revenue"], 3) if "revenue" in df.columns else None
    rev_cagr_5 = _compute_cagr_from_series(df["revenue"], 5) if "revenue" in df.columns else None
    pat_cagr_3 = _compute_cagr_from_series(df["pat"], 3) if "pat" in df.columns else None
    pat_cagr_5 = _compute_cagr_from_series(df["pat"], 5) if "pat" in df.columns else None
    ebit_cagr_3 = _compute_cagr_from_series(df["operating_profit"], 3) if "operating_profit" in df.columns else None
    ebit_cagr_5 = _compute_cagr_from_series(df["operating_profit"], 5) if "operating_profit" in df.columns else None
    eps_cagr_5 = _compute_cagr_from_series(df["eps"], 5) if "eps" in df.columns else None

    # Operating Leverage = EBIT Growth / Revenue Growth (YoY, averaged)
    op_lever_avg = _compute_operating_leverage_avg(df, years)

    # Financial Leverage = EPS Growth / EBIT Growth (YoY, averaged)
    fin_lever_avg = _compute_financial_leverage_avg(df, years)

    # Volume & Price Lever (proxy-based)
    price_lever = compute_price_lever(data, {"years": years})

    # ─── 1. Earnings Growth Profile ───
    earnings_profile = {
        "pat_cagr_3yr": pat_cagr_3,
        "pat_cagr_5yr": pat_cagr_5,
    }

    # ─── 2. Lever Table ───
    lever_table = [
        {
            "lever": "Volume Growth",
            "status": _classify_volume_status(rev_cagr_5, price_lever),
            "analysis": _generate_volume_analysis(rev_cagr_5, price_lever, df),
        },
        {
            "lever": "Price Lever",
            "status": _classify_price_lever(price_lever),
            "analysis": _generate_price_analysis(rev_cagr_5, price_lever, df),
        },
        {
            "lever": "Operating Lever",
            "status": _classify_op_lever(op_lever_avg, ebit_cagr_5, rev_cagr_5),
            "analysis": _generate_op_lever_analysis(op_lever_avg, ebit_cagr_5, rev_cagr_5, df),
        },
        {
            "lever": "Financial Lever",
            "status": _classify_fin_lever(fin_lever_avg),
            "analysis": _generate_fin_lever_analysis(fin_lever_avg, eps_cagr_5, ebit_cagr_5, df),
        },
    ]

    # ─── 3. Growth Synthesis ───
    growth_synthesis = _synthesize_growth_quality(
        pat_cagr_3, pat_cagr_5, op_lever_avg, fin_lever_avg, price_lever
    )

    # ─── 4. Valuation Reality Check ───
    current_pe = None
    if "pe_ratio" in df.columns:
        pe_vals = pd.to_numeric(df["pe_ratio"], errors="coerce").dropna()
        if not pe_vals.empty:
            current_pe = float(pe_vals.iloc[-1])

    trailing_peg = (current_pe / pat_cagr_5) if (current_pe and pat_cagr_5 and pat_cagr_5 > 0) else None

    valuation_check = {
        "current_pe": current_pe,
        "pat_cagr_5yr": pat_cagr_5,
        "trailing_peg": trailing_peg,
        "verdict": _peg_verdict(trailing_peg, growth_synthesis["quality_flag"]),
    }

    return {
        "earnings_profile": earnings_profile,
        "lever_table": lever_table,
        "growth_synthesis": growth_synthesis,
        "valuation_check": valuation_check,
    }
