"""`simulator.universe` — `raw_data/` discovery and per-ticker candidacy
under KTD8 ("every `raw_data/` ticker joins the simulated watchlist at
`screen` on the first replay date whose truncated financials meet the
engine's minimum-years bar; tickers that never clear it are exclusions,
never a silent omission").
"""

import pandas as pd

from boundless100x.compute_engine.point_in_time import truncate_to_date
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import universe as universe_module
from tests.conftest import write_ticker_dir


def test_discover_candidates_skips_non_ticker_directories(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10)

    # A BSE-code directory: annual-report PDFs only, no financials.csv —
    # the marker `discover_candidates` keys on.
    bse_only = root / "500999"
    bse_only.mkdir()
    (bse_only / "2025_annual_report.pdf").write_bytes(b"%PDF-1.4 stub")

    found = universe_module.discover_candidates(root)
    assert [d.name for d in found] == ["AAA"]


def test_discover_candidates_on_missing_directory_returns_empty(tmp_path):
    assert universe_module.discover_candidates(tmp_path / "does-not-exist") == []


def test_load_ticker_data_reads_every_frame(tmp_path):
    root = tmp_path / "raw_data"
    ticker_dir = write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12)

    data = universe_module.load_ticker_data(ticker_dir)

    for frame in ("financials", "balance_sheet", "cashflow", "ratios", "price", "quarterly", "shareholding"):
        assert frame in data, f"missing {frame}"
    assert isinstance(data["price"]["date"].iloc[0], pd.Timestamp)
    assert data["price"]["date"].dt.tz is None
    assert data["_metadata_raw"]["name"] == "AAA"
    assert data["_metadata_raw"]["Face Value"] == 10.0


def test_load_ticker_data_omits_frames_never_fetched(tmp_path):
    root = tmp_path / "raw_data"
    ticker_dir = write_ticker_dir(root, "AAA", years=10, quarters=None, shareholding_quarters=None)

    data = universe_module.load_ticker_data(ticker_dir)

    assert "quarterly" not in data
    assert "shareholding" not in data
    assert "financials" in data


def test_first_sufficient_history_date_finds_the_first_clearing_cutoff():
    from tests.conftest import make_financials, make_price

    data = {"financials": make_financials(10), "price": make_price(3200)}
    replay_dates = [pd.Timestamp(y, 3, 31) for y in range(2018, 2026)]

    date = universe_module.first_sufficient_history_date(
        data, replay_dates, min_total_years=8, annual_lag_months=6,
    )

    # 10 annual rows ending "Mar 2025" (year_labels' default end), 6-month
    # lag: the 8th row (Mar 2015 + ... ) needing 8 years means the cutoff
    # must fall on/after the 8th-from-last period-end + 6 months. Rather
    # than hand-deriving the exact date, assert the *property* the
    # function promises: this date clears the bar and the one before it
    # (chronologically) does not.
    assert date is not None
    truncated_at, _ = truncate_to_date(data, date, annual_lag_months=6, rebuild_valuation=False)
    assert len(truncated_at["financials"]) >= 8

    earlier_dates = [d for d in replay_dates if d < date]
    if earlier_dates:
        prior = max(earlier_dates)
        truncated_prior, _ = truncate_to_date(
            data, prior, annual_lag_months=6, rebuild_valuation=False,
        )
        assert len(truncated_prior["financials"]) < 8


def test_first_sufficient_history_date_none_when_never_enough():
    from tests.conftest import make_financials, make_price

    data = {"financials": make_financials(3), "price": make_price(3200)}
    replay_dates = [pd.Timestamp(y, 3, 31) for y in range(2018, 2026)]

    assert universe_module.first_sufficient_history_date(
        data, replay_dates, min_total_years=8, annual_lag_months=6,
    ) is None


def test_build_universe_excludes_ticker_with_too_few_years_with_reason(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "GOOD", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    write_ticker_dir(root, "SHORT", years=3, quarters=13, shareholding_quarters=12, price_days=3200)

    cal = calendar_module.compute_calendar(root)
    result = universe_module.build_universe(root, cal.dates)

    assert "GOOD" in result.eligible
    assert "SHORT" not in result.eligible
    assert "SHORT" in result.excluded
    assert "8 years" in result.excluded["SHORT"] or "years" in result.excluded["SHORT"]


def test_build_universe_excludes_ticker_missing_required_files(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "GOOD", years=10)

    # A directory that clears the TICKER_MARKER bar (has financials.csv)
    # but is missing price_volume.csv entirely.
    broken = root / "BROKEN"
    broken.mkdir()
    (broken / "financials.csv").write_text("year,revenue\nMar 2020,100\n")

    cal = calendar_module.compute_calendar(root)
    result = universe_module.build_universe(root, cal.dates)

    assert "BROKEN" in result.excluded
    assert "price_volume.csv" in result.excluded["BROKEN"]


def test_build_universe_first_eligible_date_is_a_calendar_date(tmp_path):
    """A ticker's KTD8 candidacy date must itself be a member of the
    replay-date list it was searched over — never an interpolated or
    off-grid date."""
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    cal = calendar_module.compute_calendar(root)
    result = universe_module.build_universe(root, cal.dates)

    assert result.eligible["AAA"] in cal.dates


def test_no_replay_dates_excludes_every_ticker_with_a_named_reason(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10)

    result = universe_module.build_universe(root, [])

    assert "AAA" in result.excluded
    assert "no replay dates" in result.excluded["AAA"]
