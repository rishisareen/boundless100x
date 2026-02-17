"""Profitability metrics: RoCE, RoE, OPM, DuPont decomposition, cash conversion."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult


def _get_annual_rows(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Get the last N annual rows, excluding TTM."""
    if "year" in df.columns:
        annual = df[~df["year"].astype(str).str.contains("TTM", case=False, na=False)]
    else:
        annual = df
    return annual.tail(years)


def compute_roce_avg(data: dict, params: dict) -> MetricResult:
    """Average RoCE over N years from ratios table."""
    years = params.get("years", 5)
    df = data["ratios"]
    rows = _get_annual_rows(df, years)

    if "roce" not in rows.columns:
        return MetricResult(error="No roce column in ratios")

    values = pd.to_numeric(rows["roce"], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error=f"Only {len(values)} RoCE data points")

    avg = float(values.mean())
    flags = []
    if (values > 15).all():
        flags.append("consistently_high_roce")
    if (values > 20).all():
        flags.append("exceptional_roce")
    if values.iloc[-1] > values.iloc[0]:
        flags.append("improving_roce")

    return MetricResult(
        value=avg,
        raw_series=values.tolist(),
        flags=flags,
        metadata={"years_used": len(values)},
    )


def compute_roe_avg(data: dict, params: dict) -> MetricResult:
    """Average RoE over N years = PAT / (Equity Capital + Reserves)."""
    years = params.get("years", 5)
    fin = _get_annual_rows(data["financials"], years)
    bs = _get_annual_rows(data["balance_sheet"], years)

    pat = pd.to_numeric(fin["pat"], errors="coerce").dropna()
    equity_capital = pd.to_numeric(bs["equity_capital"], errors="coerce")
    reserves = pd.to_numeric(bs["reserves"], errors="coerce")
    shareholders_equity = (equity_capital + reserves).dropna()

    # Align by taking min length
    n = min(len(pat), len(shareholders_equity))
    if n < 3:
        return MetricResult(error="Insufficient data for RoE")

    pat_vals = pat.tail(n).values
    eq_vals = shareholders_equity.tail(n).values
    roe_series = []
    for p, e in zip(pat_vals, eq_vals):
        if e and e > 0:
            roe_series.append(float(p / e * 100))

    if len(roe_series) < 3:
        return MetricResult(error="Insufficient valid RoE data points")

    avg = float(np.mean(roe_series))
    return MetricResult(
        value=avg,
        raw_series=roe_series,
        metadata={"years_used": len(roe_series)},
    )


def compute_opm_avg(data: dict, params: dict) -> MetricResult:
    """Average Operating Profit Margin over N years."""
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years)

    if "opm_pct" in df.columns:
        values = pd.to_numeric(df["opm_pct"], errors="coerce").dropna()
    elif "operating_profit" in df.columns and "revenue" in df.columns:
        op = pd.to_numeric(df["operating_profit"], errors="coerce")
        rev = pd.to_numeric(df["revenue"], errors="coerce")
        values = (op / rev * 100).dropna()
    else:
        return MetricResult(error="No OPM data available")

    if len(values) < 3:
        return MetricResult(error="Insufficient OPM data")

    avg = float(values.mean())
    flags = []
    if avg > 20:
        flags.append("high_operating_margin")
    if values.iloc[-1] > values.iloc[-3] if len(values) >= 3 else False:
        flags.append("improving_margins")

    return MetricResult(
        value=avg,
        raw_series=values.tolist(),
        flags=flags,
        metadata={"years_used": len(values)},
    )


def compute_dupont_margin(data: dict, params: dict) -> MetricResult:
    """DuPont: Net Profit Margin = PAT / Revenue (latest year)."""
    df = _get_annual_rows(data["financials"], 1)
    if df.empty:
        return MetricResult(error="No financial data")

    pat = pd.to_numeric(df["pat"], errors="coerce").iloc[-1]
    rev = pd.to_numeric(df["revenue"], errors="coerce").iloc[-1]

    if pd.isna(pat) or pd.isna(rev) or rev == 0:
        return MetricResult(error="Cannot compute net margin")

    npm = float(pat / rev * 100)
    return MetricResult(value=npm)


def compute_dupont_turnover(data: dict, params: dict) -> MetricResult:
    """DuPont: Asset Turnover = Revenue / Total Assets (latest year)."""
    fin = _get_annual_rows(data["financials"], 1)
    bs = _get_annual_rows(data["balance_sheet"], 1)

    if fin.empty or bs.empty:
        return MetricResult(error="No data for asset turnover")

    rev = pd.to_numeric(fin["revenue"], errors="coerce").iloc[-1]
    assets = pd.to_numeric(bs["total_assets"], errors="coerce").iloc[-1]

    if pd.isna(rev) or pd.isna(assets) or assets == 0:
        return MetricResult(error="Cannot compute asset turnover")

    at = float(rev / assets)
    return MetricResult(value=at)


def compute_dupont_leverage(data: dict, params: dict) -> MetricResult:
    """DuPont: Equity Multiplier = Total Assets / Shareholders' Equity."""
    bs = _get_annual_rows(data["balance_sheet"], 1)
    if bs.empty:
        return MetricResult(error="No balance sheet data")

    assets = pd.to_numeric(bs["total_assets"], errors="coerce").iloc[-1]
    eq_capital = pd.to_numeric(bs["equity_capital"], errors="coerce").iloc[-1]
    reserves = pd.to_numeric(bs["reserves"], errors="coerce").iloc[-1]
    equity = eq_capital + reserves if not (pd.isna(eq_capital) or pd.isna(reserves)) else None

    if pd.isna(assets) or equity is None or equity == 0:
        return MetricResult(error="Cannot compute equity multiplier")

    em = float(assets / equity)
    return MetricResult(value=em)


def compute_cash_conversion(data: dict, params: dict) -> MetricResult:
    """Cash Conversion = OCF / (Operating Profit + Depreciation) averaged over N years."""
    years = params.get("years", 5)
    fin = _get_annual_rows(data["financials"], years)
    cf = _get_annual_rows(data["cashflow"], years)

    op = pd.to_numeric(fin["operating_profit"], errors="coerce")
    dep = pd.to_numeric(fin["depreciation"], errors="coerce")
    ebitda = (op + dep).dropna()

    cfo = pd.to_numeric(cf["cfo"], errors="coerce").dropna()

    n = min(len(ebitda), len(cfo))
    if n < 3:
        return MetricResult(error="Insufficient data for cash conversion")

    ebitda_vals = ebitda.tail(n).values
    cfo_vals = cfo.tail(n).values
    ratios = []
    for c, e in zip(cfo_vals, ebitda_vals):
        if e and e != 0:
            ratios.append(float(c / e * 100))

    if not ratios:
        return MetricResult(error="Cannot compute cash conversion")

    avg = float(np.mean(ratios))
    flags = []
    if avg > 80:
        flags.append("cash_cow")

    return MetricResult(
        value=avg,
        raw_series=ratios,
        flags=flags,
        metadata={"years_used": len(ratios)},
    )


def compute_fcf_yield(data: dict, params: dict) -> MetricResult:
    """FCF Yield = (CFO - Capex) / Market Cap × 100.

    Flags when CFI is dominated by acquisitions (|CFI| > 5x depreciation).
    """
    cf = _get_annual_rows(data["cashflow"], 1)
    meta = data.get("metadata", {})

    if cf.empty:
        return MetricResult(error="No cashflow data")

    cfo = pd.to_numeric(cf["cfo"], errors="coerce").iloc[-1]
    cfi = pd.to_numeric(cf["cfi"], errors="coerce").iloc[-1]

    # Approximate capex as absolute value of CFI (investing outflows)
    # FCF ≈ CFO + CFI (since CFI is negative for capex)
    if pd.isna(cfo) or pd.isna(cfi):
        return MetricResult(error="Missing CFO/CFI data")

    fcf = cfo + cfi  # CFI is typically negative

    mcap = meta.get("Market Cap")
    if mcap is None or mcap == 0:
        return MetricResult(error="No market cap data")

    yield_pct = float(fcf / mcap * 100)

    # Detect if CFI is dominated by acquisitions
    flags = []
    fin = _get_annual_rows(data["financials"], 1)
    if not fin.empty and "depreciation" in fin.columns:
        dep = pd.to_numeric(fin["depreciation"], errors="coerce").iloc[-1]
        if not pd.isna(dep) and dep > 0 and abs(float(cfi)) > 5 * float(dep):
            flags.append("cfi_dominated_by_acquisitions")

    return MetricResult(
        value=yield_pct,
        flags=flags,
        metadata={"cfo": float(cfo), "cfi": float(cfi), "fcf": float(fcf)},
    )


def compute_tax_rate_variance(data: dict, params: dict) -> MetricResult:
    """Standard deviation of effective tax rate over N years."""
    years = params.get("years", 5)
    df = _get_annual_rows(data["financials"], years)

    if "tax_pct" not in df.columns:
        return MetricResult(error="No tax_pct column")

    values = pd.to_numeric(df["tax_pct"], errors="coerce").dropna()
    if len(values) < 3:
        return MetricResult(error="Insufficient tax rate data")

    std = float(values.std())
    flags = []
    if std > 10:
        flags.append("volatile_tax_rate")

    return MetricResult(
        value=std,
        raw_series=values.tolist(),
        flags=flags,
        metadata={"years_used": len(values)},
    )
