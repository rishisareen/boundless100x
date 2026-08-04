"""Walk-forward backtest: leakage guards, honest exclusions, correct returns."""

import json

import pandas as pd
import pytest

from boundless100x.compute_engine.backtest import WalkForwardBacktest
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import (
    make_balance_sheet,
    make_cashflow,
    make_financials,
    make_metadata,
    make_price,
    make_ratios,
    make_shareholding,
)


def write_ticker(root, ticker: str, years: int = 10, end_close: float = 400.0,
                 price_days: int = 2600, omit: tuple = ()) -> None:
    d = root / ticker
    d.mkdir(parents=True, exist_ok=True)
    frames = {
        "financials": make_financials(years),
        "balance_sheet": make_balance_sheet(years),
        "cashflow": make_cashflow(years),
        "ratios": make_ratios(years),
        "shareholding": make_shareholding(),
    }
    for name, df in frames.items():
        if name not in omit:
            df.to_csv(d / f"{name}.csv", index=False)
    if "price_volume" not in omit:
        make_price(days=price_days, end_close=end_close).to_csv(d / "price_volume.csv", index=False)
    (d / "metadata.json").write_text(json.dumps(make_metadata(name=f"{ticker} Ltd")))


def write_bse_code_dir(root, code: str = "500325") -> None:
    """A BSE-code folder: annual reports only, no CSVs."""
    reports = root / code / "annual_reports"
    reports.mkdir(parents=True)
    (reports / "2024_annual_report.txt").write_text("annual report text")


@pytest.fixture
def backtest_factory(tmp_path):
    def build(**kwargs):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        return WalkForwardBacktest(tmp_path, engine, scorer, **kwargs)
    return build


class TestDiscovery:
    def test_only_directories_with_the_required_csvs_are_candidates(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        write_bse_code_dir(tmp_path)

        names = [p.name for p in backtest_factory().discover_candidates()]

        assert names == ["GOODCO"]

    def test_bse_code_directories_never_appear_in_the_skip_list(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        write_bse_code_dir(tmp_path, "500325")
        write_bse_code_dir(tmp_path, "532830")

        report = backtest_factory().run()

        assert all(s["ticker"] != "500325" for s in report["skipped"])
        assert report["limitations"]["skipped_companies"] == 0

    def test_directory_without_price_history_is_not_a_candidate(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "NOPRICE", omit=("price_volume",))

        assert backtest_factory().discover_candidates() == []


class TestSkips:
    def test_short_history_is_skipped_and_reported(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "YOUNGCO", years=5)

        report = backtest_factory().run()

        assert report["companies"] == []
        assert report["skipped"][0]["ticker"] == "YOUNGCO"
        assert "years" in report["skipped"][0]["reason"]

    def test_price_starting_after_truncation_is_skipped_not_approximated(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "LATELIST", price_days=200)

        report = backtest_factory().run()

        assert report["companies"] == []
        assert report["skipped"]

    def test_no_silent_drops(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        write_ticker(tmp_path, "YOUNGCO", years=5)

        report = backtest_factory().run()

        assert len(report["companies"]) + len(report["skipped"]) == 2


class TestLeakageGuards:
    def test_market_cap_is_never_scored(self, tmp_path, backtest_factory):
        """Today's cap would penalise exactly the companies that later re-rated."""
        write_ticker(tmp_path, "GOODCO")

        excluded = {e["metric"] for e in backtest_factory().run()["excluded_metrics"]}

        assert "market_cap" in excluded

    def test_shareholding_and_analyst_metrics_are_excluded(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")

        excluded = {e["metric"] for e in backtest_factory().run()["excluded_metrics"]}

        assert "institutional_holding" in excluded
        assert "analyst_coverage" in excluded

    def test_changing_todays_market_cap_does_not_move_any_score(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        first = backtest_factory().run()["companies"][0]["composite_then"]

        meta = json.loads((tmp_path / "GOODCO" / "metadata.json").read_text())
        meta["Market Cap"] = 999_999.0
        (tmp_path / "GOODCO" / "metadata.json").write_text(json.dumps(meta))

        assert backtest_factory().run()["companies"][0]["composite_then"] == first

    def test_price_series_is_truncated_to_the_scoring_date(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        bt = backtest_factory()
        data = bt._load(tmp_path / "GOODCO")

        truncated, truncation_date, _ = bt._truncate(data)

        assert truncated["price"]["date"].max() <= truncation_date

    def test_every_exclusion_is_reported(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")

        report = backtest_factory().run()

        assert report["excluded_metrics"]
        for entry in report["excluded_metrics"]:
            assert entry["tickers_affected"] >= 1


class TestRealizedReturn:
    def test_return_matches_a_hand_computed_value(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        bt = backtest_factory()
        data = bt._load(tmp_path / "GOODCO")
        _, truncation_date, _ = bt._truncate(data)

        realized, span = bt._realized_return(data["price"], truncation_date)

        price = data["price"]
        start = float(price[price["date"] <= truncation_date].iloc[-1]["close"])
        end = float(price.iloc[-1]["close"])
        expected = ((end / start) ** (1 / span["years"]) - 1) * 100

        assert realized == pytest.approx(expected, abs=0.01)

    def test_span_is_reported(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")

        span = backtest_factory().run()["companies"][0]["forward_span"]

        assert span["years"] > 1
        assert "from" in span and "to" in span


class TestCorrelationAndOutput:
    def test_correlation_reproduces_a_known_ranking(self, backtest_factory):
        bt = backtest_factory()

        assert bt._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert bt._spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_correlation_needs_a_minimum_sample(self, backtest_factory):
        assert backtest_factory()._spearman([1, 2], [3, 4]) is None

    def test_report_carries_correlations_and_a_company_row_per_ticker(self, tmp_path, backtest_factory):
        for i, ticker in enumerate(("ACO", "BCO", "CCO")):
            write_ticker(tmp_path, ticker, end_close=200.0 + 100 * i)

        report = backtest_factory().run()

        assert len(report["companies"]) == 3
        assert "composite_vs_return" in report["correlations"]

    def test_limitations_block_is_always_present(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")

        limitations = backtest_factory().run()["limitations"]

        assert limitations["qualifying_companies"] == 1
        assert limitations["score_dates"]
        for key in ("survivorship", "shared_window", "truncated_history", "verdict"):
            assert limitations[key]

    def test_empty_universe_produces_an_empty_but_valid_report(self, backtest_factory):
        report = backtest_factory().run()

        assert report["companies"] == []
        assert report["limitations"]["qualifying_companies"] == 0
