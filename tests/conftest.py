"""Shared synthetic fixtures.

Tests must not depend on `data_fetcher/raw_data/` — it is gitignored and
populated by live scraping. These builders mirror the real fetched schemas
(see `fetch_financials.py`) so tests exercise the same column names the
pipeline sees in production.
"""

import numpy as np
import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import MetricResult


def year_labels(n: int, end: int = 2025) -> list[str]:
    return [f"Mar {y}" for y in range(end - n + 1, end + 1)]


def compounding(base: float, rate: float, n: int) -> list[float]:
    """A clean geometric series — CAGR over the whole span equals `rate`."""
    return [base * (1 + rate) ** i for i in range(n)]


def make_financials(n: int = 10, revenue_growth: float = 0.20,
                    pat_growth: float = 0.25, ttm: bool = False,
                    **overrides) -> pd.DataFrame:
    revenue = compounding(1000.0, revenue_growth, n)
    pat = compounding(150.0, pat_growth, n)
    operating_profit = [r * 0.25 for r in revenue]
    df = pd.DataFrame({
        "year": year_labels(n),
        "revenue": revenue,
        "expenses": [r * 0.75 for r in revenue],
        "operating_profit": operating_profit,
        "opm_pct": [25.0] * n,
        "other_income": [10.0] * n,
        "interest": [5.0] * n,
        "depreciation": [20.0] * n,
        "pbt": [p / 0.75 for p in pat],
        "tax_pct": [25.0] * n,
        "pat": pat,
        "eps": [p / 10.0 for p in pat],
        "dividend_payout_pct": [20.0] * n,
    })
    for col, values in overrides.items():
        df[col] = values
    if ttm:
        # Screener appends a trailing TTM column to the P&L.
        trailing = df.iloc[[-1]].copy()
        trailing["year"] = "TTM"
        df = pd.concat([df, trailing], ignore_index=True)
    return df


def make_balance_sheet(n: int = 10, interim: bool = False, **overrides) -> pd.DataFrame:
    reserves = compounding(500.0, 0.22, n)
    df = pd.DataFrame({
        "year": year_labels(n),
        "equity_capital": [100.0] * n,
        "reserves": reserves,
        "borrowings": [50.0] * n,
        "other_liabilities": [80.0] * n,
        "total_liabilities": [r + 230.0 for r in reserves],
        "fixed_assets": [200.0] * n,
        "cwip": [10.0] * n,
        "investments": [r * 0.5 for r in reserves],
        "other_assets": [60.0] * n,
        "total_assets": [r + 230.0 for r in reserves],
    })
    for col, values in overrides.items():
        df[col] = values
    if interim:
        # Screener appends a part-year balance sheet column (e.g. "Sep 2025").
        part_year = df.iloc[[-1]].copy()
        part_year["year"] = f"Sep {2025}"
        part_year["reserves"] = float(part_year["reserves"].iloc[0]) * 0.5
        df = pd.concat([df, part_year], ignore_index=True)
    return df


def make_cashflow(n: int = 10, **overrides) -> pd.DataFrame:
    cfo = compounding(200.0, 0.20, n)
    df = pd.DataFrame({
        "year": year_labels(n),
        "cfo": cfo,
        "cfi": [-c * 0.3 for c in cfo],
        "cff": [-c * 0.2 for c in cfo],
        "net_cash_flow": [c * 0.5 for c in cfo],
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def make_ratios(n: int = 10, roce: float = 22.0, **overrides) -> pd.DataFrame:
    df = pd.DataFrame({
        "year": year_labels(n),
        "debtor_days": [30.0] * n,
        "inventory_days": [25.0] * n,
        "days_payable": [40.0] * n,
        "cash_conversion_cycle": [15.0] * n,
        "working_capital_days": [20.0] * n,
        "roce": [roce] * n,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def make_shareholding(quarters: int = 20, promoter: float = 60.0) -> pd.DataFrame:
    return pd.DataFrame({
        "quarter": [f"Q{i % 4 + 1} {2021 + i // 4}" for i in range(quarters)],
        "promoter_pct": [promoter] * quarters,
        "fii_pct": [8.0] * quarters,
        "dii_pct": [5.0] * quarters,
        "public_pct": [promoter and 100.0 - promoter - 13.0] * quarters,
        "num_shareholders": [50000] * quarters,
    })


def make_price(days: int = 2500, start_close: float = 100.0,
               end_close: float | None = None) -> pd.DataFrame:
    """Daily bars. When `end_close` is given, close compounds geometrically to it."""
    dates = pd.bdate_range("2015-01-01", periods=days)
    if end_close is None:
        closes = np.linspace(start_close, start_close * 3, days)
    else:
        ratio = (end_close / start_close) ** (1 / max(days - 1, 1))
        closes = start_close * ratio ** np.arange(days)
    return pd.DataFrame({
        "date": dates,
        "open": closes * 0.99,
        "high": closes * 1.02,
        "low": closes * 0.98,
        "close": closes,
        "volume": [100_000] * days,
    })


def make_metadata(market_cap: float = 5000.0, name: str = "Test Co",
                  **overrides) -> dict:
    meta = {
        "company_id": "1234",
        "warehouse_id": "5678",
        "consolidated": True,
        "Market Cap": market_cap,
        "Current Price": 300.0,
        "52w_high": 380.0,
        "52w_low": 210.0,
        "Stock P/E": 30.0,
        "Book Value": 90.0,
        "Dividend Yield": 0.5,
        "ROCE": 22.0,
        "ROE": 19.0,
        "Face Value": 10.0,
        "name": name,
    }
    meta.update(overrides)
    return meta


def make_data(n: int = 10, market_cap: float = 5000.0, **kwargs) -> dict:
    """The `data` dict the compute engine and report generator consume."""
    return {
        "metadata": make_metadata(market_cap=market_cap),
        "financials": make_financials(n, **kwargs.get("financials", {})),
        "balance_sheet": make_balance_sheet(n, **kwargs.get("balance_sheet", {})),
        "cashflow": make_cashflow(n, **kwargs.get("cashflow", {})),
        "ratios": make_ratios(n, **kwargs.get("ratios", {})),
        "shareholding": make_shareholding(),
        "price": make_price(),
        "analyst_coverage": {"analyst_count": 4},
    }


def make_scores(composite: float = 6.5, **elements) -> dict:
    base = {
        "size": 5.0, "quality_business": 7.0, "quality_management": 6.0,
        "growth": 7.0, "longevity": 6.0, "price": 4.0,
    }
    base.update(elements)
    return {"composite": composite, "elements": base, "details": {}}


def make_result(ticker: str = "TEST", metrics: dict | None = None,
                scores: dict | None = None, **data_kwargs):
    """A minimal but structurally faithful AnalysisResult."""
    from boundless100x.service import AnalysisResult

    return AnalysisResult(
        ticker=ticker,
        data=make_data(**data_kwargs),
        metrics=metrics if metrics is not None else {
            "pe_ttm": MetricResult(value=30.0),
            "roce_avg": MetricResult(value=22.0),
            "market_cap": MetricResult(value=data_kwargs.get("market_cap", 5000.0)),
        },
        scores=scores if scores is not None else make_scores(),
    )


@pytest.fixture
def analysis_result():
    return make_result()
