"""Leverage metrics: Debt/Equity, Interest Coverage."""

import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_debt_equity(data: dict, params: dict) -> MetricResult:
    """Debt/Equity = Borrowings / (Equity Capital + Reserves)."""
    bs = _get_annual_rows(data["balance_sheet"], 1)
    if bs.empty:
        return MetricResult(error="No balance sheet data")

    borrowings = pd.to_numeric(bs["borrowings"], errors="coerce").iloc[-1]
    eq_capital = pd.to_numeric(bs["equity_capital"], errors="coerce").iloc[-1]
    reserves = pd.to_numeric(bs["reserves"], errors="coerce").iloc[-1]

    if pd.isna(eq_capital) or pd.isna(reserves):
        return MetricResult(error="Missing equity data")

    equity = eq_capital + reserves
    if equity <= 0:
        return MetricResult(error="Non-positive equity")

    borrowings = 0.0 if pd.isna(borrowings) else float(borrowings)
    de = borrowings / equity

    flags = []
    if de > 1.0:
        flags.append("debt_risk")
    elif de < 0.1:
        flags.append("virtually_debt_free")

    return MetricResult(value=float(de), flags=flags)


def compute_interest_coverage(data: dict, params: dict) -> MetricResult:
    """Interest Coverage = Operating Profit / Interest (latest year)."""
    df = _get_annual_rows(data["financials"], 1)
    if df.empty:
        return MetricResult(error="No financial data")

    op = pd.to_numeric(df["operating_profit"], errors="coerce").iloc[-1]
    interest = pd.to_numeric(df["interest"], errors="coerce").iloc[-1]

    if pd.isna(op):
        return MetricResult(error="Missing operating profit")
    if pd.isna(interest) or interest == 0:
        # No interest = effectively infinite coverage
        return MetricResult(value=100.0, flags=["no_interest_expense"])

    ic = float(op / interest)
    flags = []
    if ic < 2:
        flags.append("low_interest_coverage")
    elif ic > 10:
        flags.append("strong_interest_coverage")

    return MetricResult(value=ic, flags=flags)
