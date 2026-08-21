"""Profitability metrics: RoCE, RoE, OPM, DuPont decomposition, cash conversion."""

import re

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import (
    TREASURY_ADJUSTED_FLAG,
    smoothed_endpoints,
    treasury_flows,
)


# A Screener column for a shortened accounting period carries its length as a
# suffix on the label — `Mar 20169m` is a nine-month stub, not the year 20169.
# Companies emit one whenever they change financial year end, which Caplin
# Point did (June to March), and the row looks like a full year to every filter
# that only checks the leading month.
STUB_PERIOD_LABEL = re.compile(r"\d+\s*m\s*$", re.IGNORECASE)


def stub_period_labels(df: pd.DataFrame) -> list[str]:
    """Labels in `df` that name a shortened accounting period.

    Exposed so a surface can say a period was dropped rather than leaving a
    reader to notice a missing year — `_get_annual_rows` removes them, and a
    silent removal of real reported data is worse than a stated one.
    """
    if "year" not in df.columns:
        return []
    labels = df["year"].astype(str)
    return sorted(set(labels[labels.str.contains(STUB_PERIOD_LABEL, na=False)]))


def _get_annual_rows(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Get the last N annual rows, excluding TTM, interim and stub periods.

    Screener appends a part-year column to the balance sheet — every cached
    balance sheet here ends with one (`Sep 2025` for a March-year company).
    Dropping only TTM leaves that interim row looking like a full year, which
    silently pairs half a year of balance sheet against a full year of P&L.
    Annual rows are those sharing the frame's dominant period label, so this
    holds for companies whose financial year does not end in March.

    **A transition stub is dropped too, and the month filter cannot catch it.**
    A company moving its year end files one shortened period whose label starts
    with the new month and so survives every check above: Caplin Point's
    `Mar 20169m` is nine months of trading that a ten-year window counted as a
    year, depressing every CAGR that reached back to it and printing "169m" as
    a column heading in the report's own snapshot table.
    """
    if "year" not in df.columns:
        return df.tail(years)

    labels = df["year"].astype(str)
    annual = df[~labels.str.contains("TTM", case=False, na=False)]
    if annual.empty:
        return annual

    stub = annual["year"].astype(str).str.contains(STUB_PERIOD_LABEL, na=False)
    if not stub.all():
        # Guarded: a frame that is somehow all stubs keeps them, because
        # returning nothing would read as "no data" rather than "no full year".
        annual = annual[~stub]

    months = annual["year"].astype(str).str.extract(r"^([A-Za-z]{3})", expand=False)
    if months.notna().any():
        annual = annual[months == months.mode().iloc[0]]

    return annual.tail(years)


def _capital_employed(
    balance_sheet: pd.DataFrame, exclude_treasury: bool = False
) -> pd.Series | None:
    """Equity + reserves + borrowings, the capital the business is run on.

    `exclude_treasury` nets off financial investments, which answers a
    different and narrower question: what capital is *working* in the
    business. **That is the right denominator for a return on INCREMENTAL
    capital**, because retained profit sitting in a mutual fund has not been
    deployed and cannot yet have earned anything.

    Caplin Point is the case. It grew capital employed by ₹1,878 Cr over five
    years, of which roughly ₹848 Cr became financial investments, and returned
    ROIIC of 14.86% against a 15.0 gate — failed by 0.14 of a percentage point
    for holding cash. On operating capital the same company earns far more,
    and the honest reading of an undeployed balance is a question about
    capital allocation rather than about the quality of what was reinvested.
    """
    required = ("equity_capital", "reserves", "borrowings")
    if any(col not in balance_sheet.columns for col in required):
        return None
    parts = [pd.to_numeric(balance_sheet[col], errors="coerce") for col in required]
    capital = sum(parts[1:], parts[0])

    if exclude_treasury and "investments" in balance_sheet.columns:
        investments = pd.to_numeric(balance_sheet["investments"], errors="coerce")
        capital = capital - investments.fillna(0.0)

    return capital.dropna()


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

    rows = _get_annual_rows(data["balance_sheet"], years + 1)
    capital = _capital_employed(rows, exclude_treasury=True)
    reported_capital = _capital_employed(rows)
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

    # How much of the retained capital never reached the business. Carried
    # rather than hidden: a company earning well on what it deployed while
    # parking the rest is a different proposition from one earning well on
    # everything, and only this number tells the two apart.
    undeployed = None
    if reported_capital is not None and len(reported_capital) == len(capital):
        rep_start, rep_end, _ = smoothed_endpoints(reported_capital)
        delta_reported = rep_end - rep_start
        if delta_reported > 0:
            undeployed = round((delta_reported - delta_capital) / delta_reported * 100, 1)
            if undeployed >= 25:
                flags.append("capital_partly_undeployed")

    return MetricResult(
        value=float(roiic),
        raw_series=capital.tolist(),
        flags=flags,
        metadata={
            "delta_nopat": float(delta_nopat),
            "delta_capital": float(delta_capital),
            "capital_basis": "operating (net of financial investments)",
            "undeployed_share_pct": undeployed,
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


def compute_roa_avg(data: dict, params: dict) -> MetricResult:
    """Average return on assets over N years = PAT / total assets.

    The one return measure that reads the same way for a lender and a
    manufacturer, which is why it is here. RoCE divides by capital employed
    and RoE by equity alone, so for a balance-sheet business the first is
    diluted by the borrowings that *are* its raw material and the second is
    inflated by them — a 9.4% RoE on a 9.5x equity multiple and a 1.6% RoA are
    the same fact reported three ways, and only the third says plainly how
    thin the spread is.

    Closing assets rather than the opening/closing average, matching
    `compute_roe_avg` beside it: two return metrics that disagreed about which
    denominator they meant would not be comparable to each other, and the
    company's own trend is what either is read for.
    """
    years = params.get("years", 5)
    fin = _get_annual_rows(data["financials"], years)
    bs = _get_annual_rows(data["balance_sheet"], years)

    pat = pd.to_numeric(fin["pat"], errors="coerce").dropna()
    if "total_assets" not in bs.columns:
        return MetricResult(error="Balance sheet carries no total_assets column")
    assets = pd.to_numeric(bs["total_assets"], errors="coerce").dropna()

    n = min(len(pat), len(assets))
    if n < 3:
        return MetricResult(error="Insufficient data for RoA")

    series = [
        float(p / a * 100)
        for p, a in zip(pat.tail(n).values, assets.tail(n).values)
        if a and a > 0
    ]
    if len(series) < 3:
        return MetricResult(error="Insufficient valid RoA data points")

    avg = float(np.mean(series))

    # Banded against a *lending* balance sheet deliberately: 1.5% is a
    # respectable RoA for an NBFC and a catastrophe for a manufacturer, and
    # the flag exists to mark the thin-spread case that leverage then hides.
    flags = []
    if avg < 1.0:
        flags.append("thin_asset_returns")

    return MetricResult(
        value=avg,
        raw_series=series,
        flags=flags,
        metadata={"years_used": len(series)},
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

    # Money swept into deposits and funds is not capex — see
    # `operating_free_cash_flow`. Caplin Point's latest year read -0.3% on an
    # investing line of which ₹269 Cr was treasury.
    treasury = treasury_flows(
        _get_annual_rows(data.get("balance_sheet", pd.DataFrame()), 3)
    )
    parked = float(treasury.get(str(cf["year"].iloc[-1]), 0.0)) if "year" in cf.columns else 0.0

    fcf = cfo + cfi + parked  # CFI is typically negative

    mcap = meta.get("Market Cap")
    if mcap is None or mcap == 0:
        return MetricResult(error="No market cap data")

    yield_pct = float(fcf / mcap * 100)

    # Detect if CFI is dominated by acquisitions
    flags = []
    fin = _get_annual_rows(data["financials"], 1)
    if not fin.empty and "depreciation" in fin.columns:
        dep = pd.to_numeric(fin["depreciation"], errors="coerce").iloc[-1]
        # Judged on the investing line net of treasury, or every cash-rich
        # company reads as acquisition-hungry for buying mutual funds.
        if not pd.isna(dep) and dep > 0 and abs(float(cfi) + parked) > 5 * float(dep):
            flags.append("cfi_dominated_by_acquisitions")
    if parked > 0:
        flags.append(TREASURY_ADJUSTED_FLAG)

    return MetricResult(
        value=yield_pct,
        flags=flags,
        metadata={
            "cfo": float(cfo),
            "cfi": float(cfi),
            "treasury_added_back": parked,
            "fcf": float(fcf),
        },
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
