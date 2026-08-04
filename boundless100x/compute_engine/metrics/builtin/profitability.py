"""Profitability metrics: RoCE, RoE, OPM, DuPont decomposition, cash conversion."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import smoothed_endpoints


def _get_annual_rows(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Get the last N annual rows, excluding TTM and interim periods.

    Screener appends a part-year column to the balance sheet — every cached
    balance sheet here ends with one (`Sep 2025` for a March-year company).
    Dropping only TTM leaves that interim row looking like a full year, which
    silently pairs half a year of balance sheet against a full year of P&L.
    Annual rows are those sharing the frame's dominant period label, so this
    holds for companies whose financial year does not end in March.
    """
    if "year" not in df.columns:
        return df.tail(years)

    labels = df["year"].astype(str)
    annual = df[~labels.str.contains("TTM", case=False, na=False)]
    if annual.empty:
        return annual

    months = annual["year"].astype(str).str.extract(r"^([A-Za-z]{3})", expand=False)
    if months.notna().any():
        annual = annual[months == months.mode().iloc[0]]

    return annual.tail(years)


def _capital_employed(balance_sheet: pd.DataFrame) -> pd.Series | None:
    """Equity + reserves + borrowings, the capital the business is run on."""
    required = ("equity_capital", "reserves", "borrowings")
    if any(col not in balance_sheet.columns for col in required):
        return None
    parts = [pd.to_numeric(balance_sheet[col], errors="coerce") for col in required]
    return sum(parts[1:], parts[0]).dropna()


def _nopat_series(financials: pd.DataFrame) -> pd.Series | None:
    """Operating profit after tax — earnings before the capital structure."""
    if "operating_profit" not in financials.columns:
        return None
    ebit = pd.to_numeric(financials["operating_profit"], errors="coerce")
    if "tax_pct" in financials.columns:
        tax = pd.to_numeric(financials["tax_pct"], errors="coerce").clip(0, 50)
        tax_rate = float(tax.mean()) if tax.notna().any() else 25.0
    else:
        tax_rate = 25.0
    return (ebit * (1 - tax_rate / 100.0)).dropna()


def compute_roiic(data: dict, params: dict) -> MetricResult:
    """Return on incremental invested capital: change in NOPAT per rupee of new capital.

    Headline RoCE can stay high on a legacy asset base long after fresh capital
    stops earning. ROIIC asks what the marginal rupee bought, which is the
    signal that a company can keep compounding rather than merely look
    profitable.
    """
    years = params.get("years", 5)
    high_threshold = float(params.get("high_roiic_pct", 20.0))

    capital = _capital_employed(_get_annual_rows(data["balance_sheet"], years + 1))
    if capital is None:
        return MetricResult(error="Balance sheet lacks equity/reserves/borrowings for ROIIC")

    nopat = _nopat_series(_get_annual_rows(data["financials"], years + 1))
    if nopat is None:
        return MetricResult(error="No operating_profit column for ROIIC")

    if len(capital) < 3 or len(nopat) < 3:
        return MetricResult(error="Insufficient history for ROIIC")

    cap_start, cap_end, smoothed = smoothed_endpoints(capital)
    nopat_start, nopat_end, _ = smoothed_endpoints(nopat)

    delta_capital = cap_end - cap_start
    delta_nopat = nopat_end - nopat_start

    avg_capital = (cap_start + cap_end) / 2
    if avg_capital <= 0:
        return MetricResult(error="Non-positive capital employed")

    # A flat or shrinking base leaves no incremental capital to price.
    if delta_capital <= 0.01 * avg_capital:
        reason = (
            "Capital base shrinking — ROIIC undefined"
            if delta_capital < 0
            else "Capital base flat — no incremental capital to measure"
        )
        return MetricResult(
            error=reason,
            flags=["capital_base_shrinking"] if delta_capital < 0 else ["capital_base_flat"],
            metadata={"delta_capital": delta_capital, "avg_capital": avg_capital},
        )

    roiic = delta_nopat / delta_capital * 100

    flags = []
    if roiic < 0:
        flags.append("negative_incremental_returns")
    elif roiic >= high_threshold:
        flags.append("high_roiic_compounder")

    return MetricResult(
        value=float(roiic),
        raw_series=capital.tolist(),
        flags=flags,
        metadata={
            "delta_nopat": float(delta_nopat),
            "delta_capital": float(delta_capital),
            "years_used": len(capital) - 1,
            "endpoint_mode": "smoothed" if smoothed else "single",
        },
    )


def compute_capital_reinvestment_rate(data: dict, params: dict) -> MetricResult:
    """Share of NOPAT ploughed back into the capital base.

    Pairs with ROIIC: high incremental returns only compound when the company
    actually redeploys its earnings rather than paying them out. Distinct from
    longevity's reinvestment_rate, which measures capex against depreciation.
    """
    years = params.get("years", 5)
    low_threshold = float(params.get("low_reinvestment_pct", 20.0))

    capital = _capital_employed(_get_annual_rows(data["balance_sheet"], years + 1))
    if capital is None:
        return MetricResult(error="Balance sheet lacks equity/reserves/borrowings")

    nopat = _nopat_series(_get_annual_rows(data["financials"], years + 1))
    if nopat is None:
        return MetricResult(error="No operating_profit column for reinvestment rate")

    if len(capital) < 3 or len(nopat) < 2:
        return MetricResult(error="Insufficient history for reinvestment rate")

    delta_capital = float(capital.iloc[-1] - capital.iloc[0])
    # NOPAT earned across the same periods the capital change spans.
    nopat_total = float(nopat.iloc[1:].sum())

    if nopat_total <= 0:
        return MetricResult(error="Non-positive cumulative NOPAT")

    rate = delta_capital / nopat_total * 100

    flags = []
    if rate < low_threshold:
        flags.append("capital_returned_not_reinvested")
    elif rate > 80:
        # Distinct from longevity's heavy_reinvestment (capex vs depreciation).
        flags.append("high_capital_redeployment")

    return MetricResult(
        value=float(rate),
        flags=flags,
        metadata={
            "delta_capital": delta_capital,
            "nopat_total": nopat_total,
            "years_used": len(capital) - 1,
        },
    )


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
