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

    def test_ticker_missing_price_history_is_skipped_with_a_reason(self, tmp_path, backtest_factory):
        """It is a real ticker (ASTRAL is one today), so it must not vanish."""
        write_ticker(tmp_path, "NOPRICE", omit=("price_volume",))

        report = backtest_factory().run()

        assert report["companies"] == []
        assert report["skipped"] == [
            {"ticker": "NOPRICE", "reason": "missing price_volume.csv"}
        ]


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
        """Every genuine ticker lands in exactly one of companies or skipped."""
        write_ticker(tmp_path, "GOODCO")
        write_ticker(tmp_path, "YOUNGCO", years=5)
        write_ticker(tmp_path, "NOPRICE", omit=("price_volume",))
        write_bse_code_dir(tmp_path, "500325")

        report = backtest_factory().run()

        accounted = {r["ticker"] for r in report["companies"]} | {
            s["ticker"] for s in report["skipped"]
        }
        assert accounted == {"GOODCO", "YOUNGCO", "NOPRICE"}


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

        # Recompute independently — do not reuse the code's rounded span.
        price = data["price"]
        start_row = price[price["date"] <= truncation_date].iloc[-1]
        end_row = price.iloc[-1]
        years = (end_row["date"] - start_row["date"]).days / 365.25
        expected = ((float(end_row["close"]) / float(start_row["close"])) ** (1 / years) - 1) * 100

        assert realized == pytest.approx(expected, abs=1e-9)

    def test_span_is_reported(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")

        span = backtest_factory().run()["companies"][0]["forward_span"]

        assert span["years"] > 1
        assert "from" in span and "to" in span

    def test_a_trailing_unpublished_adjusted_close_does_not_nan_the_return(
        self, tmp_path, backtest_factory
    ):
        """The source publishes today's raw close before its adjusted one.

        A series fetched today therefore ends in a single NaN `adj_close`.
        Reading the last row unconditionally turned that into a NaN realized
        return on 5 of 15 corpus tickers the moment they gained an adjusted
        series — invisible until then, because they had been falling back to
        the raw close.
        """
        write_ticker(tmp_path, "GOODCO")
        path = tmp_path / "GOODCO" / "price_volume.csv"
        price = pd.read_csv(path)
        price["adj_close"] = price["close"]
        price["adj_close_is_estimated"] = False
        price.loc[price.index[-1], "adj_close"] = None
        price.to_csv(path, index=False)

        bt = backtest_factory()
        data = bt._load(tmp_path / "GOODCO")
        _, truncation_date, _ = bt._truncate(data)

        realized, span = bt._realized_return(data["price"], truncation_date)

        assert realized is not None and realized == realized  # not NaN
        assert span["price_series"] == "adj_close"
        assert span["to"] == str(price["date"].iloc[-2])[:10]

    def test_a_series_whose_adjusted_column_is_entirely_absent_of_values_is_skipped(
        self, tmp_path, backtest_factory
    ):
        write_ticker(tmp_path, "GOODCO")
        path = tmp_path / "GOODCO" / "price_volume.csv"
        price = pd.read_csv(path)
        price["adj_close"] = None
        price["adj_close_is_estimated"] = False
        price.to_csv(path, index=False)

        bt = backtest_factory()
        data = bt._load(tmp_path / "GOODCO")
        _, truncation_date, _ = bt._truncate(data)

        realized, span = bt._realized_return(data["price"], truncation_date)

        assert realized is None
        assert "both sides" in span["reason"]


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


class TestPostReviewHardening:
    def test_stock_pe_is_not_reconstructed_from_adjusted_closes(self, tmp_path, backtest_factory):
        """Stored closes are split/dividend-adjusted, so a rebuilt P/E is fiction."""
        write_ticker(tmp_path, "GOODCO")
        bt = backtest_factory()
        truncated, _, _ = bt._truncate(bt._load(tmp_path / "GOODCO"))

        assert "Stock P/E" not in truncated["metadata"]
        assert "Market Cap" not in truncated["metadata"]

    def test_observation_date_sits_after_the_reporting_lag(self, tmp_path, backtest_factory):
        """Scoring at fiscal year end would use accounts nobody could read yet."""
        write_ticker(tmp_path, "GOODCO")
        bt = backtest_factory()
        _, truncation_date, _ = bt._truncate(bt._load(tmp_path / "GOODCO"))

        assert truncation_date.month != 3

    def test_annual_frames_are_cut_by_fiscal_year_not_row_position(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        bt = backtest_factory()
        truncated, _, _ = bt._truncate(bt._load(tmp_path / "GOODCO"))

        cutoff = max(int(y.split()[-1]) for y in truncated["financials"]["year"])
        for frame in ("balance_sheet", "cashflow", "ratios"):
            assert all(int(y.split()[-1]) <= cutoff for y in truncated[frame]["year"])

    def test_size_gate_is_actually_excluded_from_the_backtest_verdict(self, tmp_path):
        from boundless100x.compute_engine.eligibility import EligibilityEvaluator

        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        write_ticker(tmp_path, "GOODCO")

        bt = WalkForwardBacktest(tmp_path, engine, scorer, EligibilityEvaluator(engine.gates))
        report = bt.run()

        assert "size" not in bt.gate_evaluator.gates
        assert "size" not in report["companies"][0]["eligibility_then"]["gates_evaluated"]

    def test_company_with_low_coverage_is_not_ranked(self, backtest_factory):
        bt = backtest_factory()
        rows = [
            {"composite_then": 0.0, "coverage_composite": 0.0, "realized_cagr_pct": -50.0,
             "elements_then": {}},
            {"composite_then": 7.0, "coverage_composite": 0.9, "realized_cagr_pct": 20.0,
             "elements_then": {}},
            {"composite_then": 5.0, "coverage_composite": 0.9, "realized_cagr_pct": 10.0,
             "elements_then": {}},
            {"composite_then": 3.0, "coverage_composite": 0.9, "realized_cagr_pct": 5.0,
             "elements_then": {}},
        ]

        assert bt._correlations(rows)["n"] == 3


class TestDateAwareTruncation:
    """A bare calendar-year comparison can't tell a fiscal year end from a
    trailing part-year column that happens to share its year — only the
    actual period-end date can. See period_end_date in builtin/_helpers.py."""

    def test_interim_row_sharing_the_cutoff_year_does_not_leak(self, tmp_path, backtest_factory):
        """Screener appends part-year balance-sheet columns (e.g. "Sep 2025").
        One landing on the same calendar year as the cutoff row covers a later
        period than it and was not yet public as of the truncation date — a
        year-only comparison would let it leak in as "past" data anyway."""
        d = tmp_path / "GOODCO"
        d.mkdir(parents=True)
        make_financials(10).to_csv(d / "financials.csv", index=False)
        make_cashflow(10).to_csv(d / "cashflow.csv", index=False)
        make_ratios(10).to_csv(d / "ratios.csv", index=False)
        make_shareholding().to_csv(d / "shareholding.csv", index=False)
        make_price(days=2600, end_close=400.0).to_csv(d / "price_volume.csv", index=False)
        (d / "metadata.json").write_text(json.dumps(make_metadata(name="GOODCO Ltd")))

        bs = make_balance_sheet(10)
        assert bs.iloc[4]["year"] == "Mar 2020"          # the row _truncate will cut on
        leaking_row = bs.iloc[[4]].copy()
        leaking_row["year"] = "Sep 2020"                 # same year, later period end
        bs = pd.concat([bs, leaking_row], ignore_index=True)
        bs.to_csv(d / "balance_sheet.csv", index=False)

        bt = backtest_factory()
        truncated, truncation_date, reason = bt._truncate(bt._load(d))

        assert truncated is not None, reason
        assert truncation_date == pd.Timestamp("2020-09-30")
        assert "Sep 2020" not in list(truncated["balance_sheet"]["year"])

    def test_non_march_fiscal_year_end_is_not_assumed_to_be_march(self, tmp_path, backtest_factory):
        """The old cutoff was hardcoded to 31 March regardless of the label.
        Deriving it from the parsed period end instead must track whatever
        fiscal year end the company actually reports on."""
        d = tmp_path / "DECCO"
        d.mkdir(parents=True)
        dec_labels = [f"Dec {y}" for y in range(2016, 2026)]
        for name, builder in (
            ("financials", make_financials), ("balance_sheet", make_balance_sheet),
            ("cashflow", make_cashflow), ("ratios", make_ratios),
        ):
            frame = builder(10)
            frame["year"] = dec_labels
            frame.to_csv(d / f"{name}.csv", index=False)
        make_shareholding().to_csv(d / "shareholding.csv", index=False)
        make_price(days=2600, end_close=400.0).to_csv(d / "price_volume.csv", index=False)
        (d / "metadata.json").write_text(json.dumps(make_metadata(name="DECCO Ltd")))

        bt = backtest_factory()
        truncated, truncation_date, reason = bt._truncate(bt._load(d))

        assert truncated is not None, reason
        # Cutoff row is "Dec 2020" (index 4 of 10) → period end 31 Dec 2020,
        # + the 6-month reporting lag → 30 Jun 2021. A March-31 assumption
        # would instead have produced 30 Sep 2020.
        assert truncation_date == pd.Timestamp("2021-06-30")


class TestCoverageThreshold:
    """The backtest's own bar for a "comparable enough" composite must be the
    same one production uses (scorer.low_coverage_threshold), not a second,
    undocumented constant free to drift from it."""

    def test_default_min_coverage_is_the_scorers_low_coverage_threshold(self, backtest_factory):
        bt = backtest_factory()

        assert bt.min_coverage == bt.scorer.low_coverage_threshold

    def test_a_stricter_scorer_threshold_tightens_the_backtest_too(self, tmp_path):
        engine = ComputeEngine()
        strict_scorer = SQGLPScorer(engine.metrics, engine.element_weights,
                                     low_coverage_threshold=0.95)

        bt = WalkForwardBacktest(tmp_path, engine, strict_scorer)

        assert bt.min_coverage == 0.95

    def test_explicit_min_coverage_overrides_the_scorers_default(self, tmp_path):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)

        bt = WalkForwardBacktest(tmp_path, engine, scorer, min_coverage=0.5)

        assert bt.min_coverage == 0.5

    def test_coverage_exactly_at_threshold_is_included(self, backtest_factory):
        bt = backtest_factory()
        rows = [
            {"composite_then": 6.0, "coverage_composite": bt.min_coverage, "realized_cagr_pct": 10.0, "elements_then": {}},
            {"composite_then": 5.0, "coverage_composite": bt.min_coverage, "realized_cagr_pct": 8.0, "elements_then": {}},
            {"composite_then": 4.0, "coverage_composite": bt.min_coverage, "realized_cagr_pct": 6.0, "elements_then": {}},
        ]

        assert bt._correlations(rows)["n"] == 3

    def test_coverage_just_below_threshold_is_excluded(self, backtest_factory):
        bt = backtest_factory()
        rows = [
            {"composite_then": 6.0, "coverage_composite": bt.min_coverage - 0.01, "realized_cagr_pct": 10.0, "elements_then": {}},
            {"composite_then": 5.0, "coverage_composite": bt.min_coverage, "realized_cagr_pct": 8.0, "elements_then": {}},
            {"composite_then": 4.0, "coverage_composite": bt.min_coverage, "realized_cagr_pct": 6.0, "elements_then": {}},
        ]

        assert bt._correlations(rows)["n"] == 2

    def test_missing_coverage_field_fails_closed_not_included(self, backtest_factory):
        bt = backtest_factory()
        rows = [{"composite_then": 6.0, "realized_cagr_pct": 10.0, "elements_then": {}}]

        assert bt._correlations(rows)["n"] == 0

    def test_run_populates_coverage_composite_from_the_real_scorer(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")

        row = backtest_factory().run()["companies"][0]

        assert "coverage_composite" in row
        assert 0.0 <= row["coverage_composite"] <= 1.0
        assert "scored_weight" not in row


class TestEligibilityCohorts:
    """The gates are a conjunctive filter, separate from the additive
    composite — each row already carries its own eligibility_then.verdict,
    but it was computed once per company and never rolled up. This is the
    one place that checks whether "eligible" companies actually went on to
    deliver bigger returns than "not_eligible" or "indeterminate" ones."""

    @staticmethod
    def eligibility_backtest(tmp_path):
        from boundless100x.compute_engine.eligibility import EligibilityEvaluator

        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        return WalkForwardBacktest(tmp_path, engine, scorer, EligibilityEvaluator(engine.gates))

    def test_returns_none_without_an_eligibility_evaluator(self, backtest_factory):
        bt = backtest_factory()

        cohorts = bt._eligibility_cohorts([
            {"realized_cagr_pct": 10.0, "eligibility_then": {"verdict": "eligible"}}
        ])

        assert cohorts is None

    def test_splits_returns_by_verdict(self, tmp_path):
        bt = self.eligibility_backtest(tmp_path)
        rows = [
            {"realized_cagr_pct": 50.0, "eligibility_then": {"verdict": "eligible"}},
            {"realized_cagr_pct": 70.0, "eligibility_then": {"verdict": "eligible"}},
            {"realized_cagr_pct": 5.0, "eligibility_then": {"verdict": "not_eligible"}},
            {"realized_cagr_pct": -10.0, "eligibility_then": {"verdict": "not_eligible"}},
            {"realized_cagr_pct": 20.0, "eligibility_then": {"verdict": "indeterminate"}},
        ]

        cohorts = bt._eligibility_cohorts(rows)

        assert cohorts["eligible"]["n"] == 2
        assert cohorts["eligible"]["mean_cagr_pct"] == pytest.approx(60.0)
        assert cohorts["not_eligible"]["n"] == 2
        assert cohorts["not_eligible"]["mean_cagr_pct"] == pytest.approx(-2.5)
        assert cohorts["indeterminate"]["n"] == 1

    def test_min_and_max_are_reported_per_cohort(self, tmp_path):
        bt = self.eligibility_backtest(tmp_path)
        rows = [
            {"realized_cagr_pct": -20.0, "eligibility_then": {"verdict": "eligible"}},
            {"realized_cagr_pct": 90.0, "eligibility_then": {"verdict": "eligible"}},
        ]

        cohorts = bt._eligibility_cohorts(rows)

        assert cohorts["eligible"]["min_cagr_pct"] == -20.0
        assert cohorts["eligible"]["max_cagr_pct"] == 90.0

    def test_rows_without_a_verdict_are_excluded_not_miscounted(self, tmp_path):
        bt = self.eligibility_backtest(tmp_path)
        rows = [
            {"realized_cagr_pct": 10.0},
            {"realized_cagr_pct": 20.0, "eligibility_then": {"verdict": "eligible"}},
        ]

        cohorts = bt._eligibility_cohorts(rows)

        assert set(cohorts.keys()) == {"eligible"}
        assert cohorts["eligible"]["n"] == 1

    def test_run_populates_eligibility_cohorts_end_to_end(self, tmp_path):
        bt = self.eligibility_backtest(tmp_path)
        write_ticker(tmp_path, "GOODCO")

        report = bt.run()

        assert report["eligibility_cohorts"] is not None
        verdict = report["companies"][0]["eligibility_then"]["verdict"]
        assert report["eligibility_cohorts"][verdict]["n"] == 1

    def test_run_without_eligibility_evaluator_reports_none(self, tmp_path, backtest_factory):
        write_ticker(tmp_path, "GOODCO")
        bt = backtest_factory()

        report = bt.run()

        assert report["eligibility_cohorts"] is None
