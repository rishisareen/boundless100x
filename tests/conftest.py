"""Shared synthetic fixtures.

Tests must not depend on `data_fetcher/raw_data/` — it is gitignored and
populated by live scraping. These builders mirror the real fetched schemas
(see `fetch_financials.py`) so tests exercise the same column names the
pipeline sees in production.
"""

import numpy as np
import pandas as pd
import pytest

from boundless100x import score_history
from boundless100x import watchlist as watchlist_module
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle import reinvestment as reinvestment_module


@pytest.fixture(autouse=True)
def isolate_live_stores(tmp_path, monkeypatch):
    """No test may write a store the owner's real decisions live in.

    Three files qualify, and the argument is the same for all of them.
    `service.analyze()` records every scored run in the score history, which is
    git-tracked and append-only by contract — a test run must never leave
    synthetic composites in it. `watchlist.json` holds live positions, and the
    reinvestment queue holds real sales and where their proceeds went; neither
    is generated state that can be rebuilt, and the queue is not even
    gitignored, so damage to it would be committed alongside whatever caused
    it.

    Redirecting the **module defaults** is what makes this catch the case worth
    catching. Every CLI entry point constructs these stores with no path at
    all, so a test that exercises a command without thinking about persistence
    reaches the real file by default rather than by mistake. Per-test
    monkeypatches that redirect them again are harmless and stay where they
    are: they name the path the test then asserts against.
    """
    monkeypatch.setattr(
        score_history, "DEFAULT_HISTORY_PATH", tmp_path / "score_history.jsonl"
    )
    monkeypatch.setattr(
        watchlist_module, "DEFAULT_WATCHLIST_PATH", tmp_path / "watchlist.json"
    )
    monkeypatch.setattr(
        reinvestment_module, "DEFAULT_QUEUE_PATH",
        tmp_path / "reinvestment_queue.json",
    )


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


def _per_quarter(value, quarters: int) -> list:
    """A per-row column from either one value or one value per row."""
    if isinstance(value, (list, tuple, pd.Series)):
        if len(value) != quarters:
            raise ValueError(f"expected {quarters} values, got {len(value)}")
        return list(value)
    return [value] * quarters


def make_shareholding(quarters: int = 20, promoter: float = 60.0,
                      fii=8.0, dii=5.0) -> pd.DataFrame:
    """Screener's shareholding table, column-for-column as the parser writes it.

    Two fidelity bugs lived here. Labels were `Q1 2021`-style, which no period
    parser reads, so anything matching shareholding by period silently fell
    back to position. And `govt_pct` was missing although `SH_LABEL_MAP`
    declares it and `report_generator` reads it — so report tests exercised an
    absent-column path no real ticker takes, and one test file had already
    added the column by hand to work around it.

    `fii` and `dii` take either a scalar (every quarter alike, the default a
    trendless series) or one value per quarter, which is what the
    institutional-accumulation streak needs: its whole subject is how the two
    move from quarter to quarter, and rows are ordered oldest first exactly as
    the fetched file is.
    """
    return pd.DataFrame({
        "quarter": quarter_labels(quarters),
        "promoter_pct": [promoter] * quarters,
        "fii_pct": _per_quarter(fii, quarters),
        "dii_pct": _per_quarter(dii, quarters),
        "govt_pct": [0.0] * quarters,
        "public_pct": [promoter and 100.0 - promoter - 13.0] * quarters,
        "num_shareholders": [50000] * quarters,
    })


def make_price(days: int = 2500, start_close: float = 100.0,
               end_close: float | None = None,
               adj_close: bool = True,
               adj_factor: float = 1.0,
               adj_close_is_estimated: bool | None = None) -> pd.DataFrame:
    """Daily bars. When `end_close` is given, close compounds geometrically to it.

    `adj_close` is **on by default, because every cached ticker now has one.**
    It was opt-in while 13 of 22 files predated the adjusted-series schema; the
    2026-08-07 refetch moved all 22, so the old default described no real
    ticker and quietly aimed most tests at a schema that no longer exists. That
    drift is what let a NaN in the newest `adj_close` bar go unnoticed until the
    refetch exposed it in the backtest.

    A test exercising the genuine legacy path passes `adj_close=False` and says
    so — an opt-out for a shape the corpus no longer holds, rather than a
    default. `adj_factor` scales the adjusted series away from the raw close so
    a test can tell the two apart.
    """
    dates = pd.bdate_range("2015-01-01", periods=days)
    if end_close is None:
        closes = np.linspace(start_close, start_close * 3, days)
    else:
        ratio = (end_close / start_close) ** (1 / max(days - 1, 1))
        closes = start_close * ratio ** np.arange(days)
    df = pd.DataFrame({
        "date": dates,
        "open": closes * 0.99,
        "high": closes * 1.02,
        "low": closes * 0.98,
        "close": closes,
        "volume": [100_000] * days,
    })
    if adj_close:
        df["adj_close"] = closes * adj_factor
        if adj_close_is_estimated is not None:
            df["adj_close_is_estimated"] = [adj_close_is_estimated] * days
    return df


# The quarterly results table Screener renders, column-for-column as
# `QTR_LABEL_MAP` in `fetch_financials.py` writes it. Every id in
# `checkpoint_vocabulary.yaml` under `source: quarterly` reads one of these,
# and so does `quarterly_momentum` — a fixture missing a column would make a
# metric look unavailable for a reason production never sees.
QUARTERLY_COLUMNS = (
    "quarter", "revenue", "expenses", "operating_profit", "opm_pct",
    "other_income", "interest", "depreciation", "pbt", "tax_pct", "pat", "eps",
)


def quarter_labels(n: int, end_year: int = 2025) -> list[str]:
    """Period labels oldest first, ending at the March quarter of `end_year`.

    Screener names a fiscal quarter by its end month, so an Indian March-year
    company cycles Jun / Sep / Dec / Mar with the calendar year rolling over on
    the March quarter.
    """
    months = ("Jun", "Sep", "Dec", "Mar")
    total = ((n + 3) // 4) * 4  # whole fiscal years, so the last label is a March
    start_year = end_year - 1 - (total - 1) // 4
    labels = [
        f"{months[i % 4]} {start_year + i // 4 + (1 if i % 4 == 3 else 0)}"
        for i in range(total)
    ]
    return labels[-n:]


def make_quarterly(periods: int = 12, revenue_yoy: float = 0.20,
                   base_revenue: float = 250.0, **overrides) -> pd.DataFrame:
    """Screener's quarterly results table.

    Grows `base_revenue` at a constant `revenue_yoy` against the same quarter a
    year earlier, so a YoY read is flat by construction and any momentum a test
    sees came from an override rather than the builder.
    """
    revenue = [base_revenue * (1 + revenue_yoy) ** (i / 4.0) for i in range(periods)]
    operating_profit = [r * 0.24 for r in revenue]
    pat = [r * 0.15 for r in revenue]
    df = pd.DataFrame({
        "quarter": quarter_labels(periods),
        "revenue": revenue,
        "expenses": [r * 0.76 for r in revenue],
        "operating_profit": operating_profit,
        "opm_pct": [24.0] * periods,
        "other_income": [3.0] * periods,
        "interest": [1.5] * periods,
        "depreciation": [6.0] * periods,
        "pbt": [p / 0.75 for p in pat],
        "tax_pct": [25.0] * periods,
        "pat": pat,
        "eps": [p / 10.0 for p in pat],
    })
    for col, values in overrides.items():
        df[col] = values
    return df


AR_SECTION_NAMES = ("mdna", "chairman", "governance")

# Section text that passes the KTD9 content gate for what it claims to be —
# the markers a genuine slice opens with, not a plausible-looking paraphrase.
_AR_SECTION_TEXT = {
    "mdna": (
        "MANAGEMENT DISCUSSION AND ANALYSIS\n"
        "ECONOMIC REVIEW\nThe Indian economy grew steadily through the year. "
        "INDUSTRY STRUCTURE AND DEVELOPMENTS\nDemand across our segments "
        "remained firm.\nOUTLOOK\nWe expect revenue of Rs 1,500 crore in "
        "FY2026.\nSEGMENT-WISE PERFORMANCE\nBoth segments grew double digits."
    ),
    "chairman": (
        "CHAIRMAN'S LETTER\nDear Shareholders,\nIt gives me great pleasure to "
        "present your Company's annual report for the year. The addressable "
        "market for our products is estimated at Rs 40,000 crore."
    ),
    "governance": (
        "REPORT ON CORPORATE GOVERNANCE\nThe Company's philosophy on corporate "
        "governance rests on transparency and accountability. The Board met "
        "four times during the year."
    ),
}

# The residual false positive KTD9 exists for: a bare heading line followed by
# governance prose. Taken from the real ASTRAL slice rather than invented.
AUDIT_COMMITTEE_TEXT = (
    "MANAGEMENT DISCUSSION AND ANALYSIS\n"
    "The terms of reference of the Audit Committee are in accordance with "
    "Section 177 of the Companies Act, 2013 and Regulation 18 of the SEBI "
    "Listing Regulations. The Committee reviewed the quarterly financial "
    "statements and the internal audit reports placed before it."
)


def make_ar_sections(years: list[str] | None = None, provenance: str = "found",
                     sections: dict[str, str] | None = None,
                     per_section_provenance: dict[str, str] | None = None) -> dict:
    """`{year: {section: {text, provenance, start_page}}}` as the fetcher writes it.

    `provenance` sets every section's tag; `per_section_provenance` overrides
    individual ones, which is how the mixed-provenance case is built — 16 of
    the 29 real report-years carry a mix, so a fixture that can only be
    uniformly found or uniformly fallback cannot reach the case that matters.
    """
    years = years or ["2025"]
    names = sections or {name: _AR_SECTION_TEXT[name] for name in AR_SECTION_NAMES}
    overrides = per_section_provenance or {}

    out: dict[str, dict] = {}
    for year in years:
        out[year] = {}
        for index, (name, text) in enumerate(names.items()):
            tag = overrides.get(name, provenance)
            out[year][name] = {
                "text": text,
                "provenance": tag,
                "start_page": 40 + index * 10 if tag == "found" else None,
            }
    return out


def make_history_rows(ticker: str = "TEST", dates: list[str] | None = None,
                      composites: list[float] | None = None,
                      config_hash: str = "abc123abc123",
                      forward_signal_hash: str = "fff000fff000",
                      elements: list[dict] | None = None,
                      synthetic: bool = False,
                      verdict: str = "eligible",
                      coverage: float = 0.95) -> list[dict]:
    """Score-history rows in the shape `score_history._row_from` writes.

    One call produces one regime. Momentum must refuse to diff across regimes
    (KTD5), so a two-regime fixture is two calls concatenated rather than a
    single builder with a hash list — that keeps the per-regime dates and
    composites obviously paired at the call site.
    """
    dates = dates or ["2026-01-01", "2026-04-01"]
    composites = composites if composites is not None else [6.0 + i * 0.4 for i in range(len(dates))]
    if elements is None:
        elements = [
            {
                "size": 5.0, "quality_business": 7.0 + i * 0.2,
                "quality_management": 6.0, "growth": 7.0, "longevity": 6.0,
                "price": 4.0,
            }
            for i in range(len(dates))
        ]

    return [
        {
            "schema_version": score_history.SCHEMA_VERSION,
            "ticker": ticker,
            "date": date,
            "composite": composites[i],
            "elements": elements[i],
            "verdict": verdict,
            "coverage": coverage,
            "flags": [],
            "config_hash": config_hash,
            "forward_signal_hash": forward_signal_hash,
            "synthetic": synthetic,
        }
        for i, date in enumerate(dates)
    ]


def write_history(path, rows: list[dict]) -> None:
    """Write history rows to a JSONL file the way `append_run` would."""
    import json

    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


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


def write_ticker_dir(
    root, ticker: str, *,
    years: int = 10, quarters: int | None = 13, shareholding_quarters: int | None = 12,
    price_days: int = 3200, market_cap: float = 5000.0,
    financials_kwargs: dict | None = None, balance_sheet_kwargs: dict | None = None,
    cashflow_kwargs: dict | None = None, ratios_kwargs: dict | None = None,
    price_kwargs: dict | None = None, quarterly_kwargs: dict | None = None,
    shareholding_kwargs: dict | None = None, metadata_overrides: dict | None = None,
):
    """Write one ticker's `raw_data/{TICKER}/`-shaped directory to disk —
    the CSV/JSON files `WalkForwardBacktest._load` and
    `simulator.universe.load_ticker_data` both read. Every column comes from
    this file's own `make_*` builders, so a simulator test exercises the
    same schema every other test does.

    `quarters`/`shareholding_quarters` of `None` skips writing that file
    entirely (a ticker fetched before Phase 0/before the shareholding
    truncation decision landed carries neither) — everything else is
    always written, since `financials.csv` alone is `TICKER_MARKER` and a
    ticker directory without a usable price series cannot be truncated at
    all.

    `price_days=3200` (~12.7 business-year years from `make_price`'s fixed
    2015-01-01 anchor) is chosen so the series comfortably outlives the
    default 2023-2026 simulator replay window; a test needing a shorter or
    longer tail overrides it directly.
    """
    import json
    from pathlib import Path

    ticker_dir = Path(root) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    make_financials(years, **(financials_kwargs or {})).to_csv(
        ticker_dir / "financials.csv", index=False
    )
    make_balance_sheet(years, **(balance_sheet_kwargs or {})).to_csv(
        ticker_dir / "balance_sheet.csv", index=False
    )
    make_cashflow(years, **(cashflow_kwargs or {})).to_csv(
        ticker_dir / "cashflow.csv", index=False
    )
    make_ratios(years, **(ratios_kwargs or {})).to_csv(
        ticker_dir / "ratios.csv", index=False
    )
    make_price(price_days, **(price_kwargs or {})).to_csv(
        ticker_dir / "price_volume.csv", index=False
    )
    if quarters is not None:
        make_quarterly(quarters, **(quarterly_kwargs or {})).to_csv(
            ticker_dir / "quarterly.csv", index=False
        )
    if shareholding_quarters is not None:
        make_shareholding(shareholding_quarters, **(shareholding_kwargs or {})).to_csv(
            ticker_dir / "shareholding.csv", index=False
        )

    meta = make_metadata(market_cap=market_cap, name=ticker, **(metadata_overrides or {}))
    (ticker_dir / "metadata.json").write_text(json.dumps(meta))

    return ticker_dir


def make_data(n: int = 10, market_cap: float = 5000.0, **kwargs) -> dict:
    """The `data` dict the compute engine and report generator consume.

    `quarterly` and `annual_report_sections` are carried because the fetcher
    always writes them; before Phase 2 no metric declared either as an input,
    so their presence changes nothing for existing callers.
    """
    return {
        "metadata": make_metadata(market_cap=market_cap),
        "financials": make_financials(n, **kwargs.get("financials", {})),
        "balance_sheet": make_balance_sheet(n, **kwargs.get("balance_sheet", {})),
        "cashflow": make_cashflow(n, **kwargs.get("cashflow", {})),
        "ratios": make_ratios(n, **kwargs.get("ratios", {})),
        "shareholding": make_shareholding(),
        "price": make_price(**kwargs.get("price", {})),
        "quarterly": make_quarterly(**kwargs.get("quarterly", {})),
        "annual_report_sections": make_ar_sections(
            **kwargs.get("annual_report_sections", {})
        ),
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


def latest_scores_for(ticker: str):
    """The most recent `scores.json` this machine holds for a ticker, or None.

    Globbed rather than named. Two fixtures used to open
    `PFC_20260808/scores.json` — a literal date, which stops existing the day
    after it is written. The failure mode is the quiet one: the test does not
    break, it *skips*, so the suite stays green while the case it was written
    for silently stops being checked. `analyze` writes one directory per ticker
    per date, so the latest is the one to read.
    """
    from boundless100x.output.report_expansion import DEFAULT_REPORTS_DIR

    for report in sorted(DEFAULT_REPORTS_DIR.glob(f"{ticker}_*"), reverse=True):
        scores = report / "scores.json"
        if scores.is_file():
            return scores
    return None
