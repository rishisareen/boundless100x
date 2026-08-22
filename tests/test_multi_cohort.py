"""Multi-cohort backtest: cutoffs, fixed horizons, coverage denominator, distributions."""

import json

import pandas as pd
import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.multi_cohort import (
    MultiCohortBacktest,
    _WithheldEvidence,
)
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import (
    make_balance_sheet,
    make_cashflow,
    make_financials,
    make_metadata,
    make_price,
    make_ratios,
)


def write_ticker(root, ticker: str, years: int = 12, price_days: int = 3200,
                 end_close: float = 400.0, omit: tuple = (),
                 price_kwargs: dict | None = None) -> None:
    """One ticker's raw_data-shaped directory.

    No `shareholding.csv` — the backtest's `_load` never reads it (the
    leakage decision recorded in `point_in_time`), so a fixture carrying one
    would describe a corpus no caller sees.
    """
    d = root / ticker
    d.mkdir(parents=True, exist_ok=True)
    frames = {
        "financials": make_financials(years),
        "balance_sheet": make_balance_sheet(years),
        "cashflow": make_cashflow(years),
        "ratios": make_ratios(years),
    }
    for name, df in frames.items():
        if name not in omit:
            df.to_csv(d / f"{name}.csv", index=False)
    if "price_volume" not in omit:
        make_price(days=price_days, end_close=end_close,
                   **(price_kwargs or {})).to_csv(d / "price_volume.csv", index=False)
    (d / "metadata.json").write_text(json.dumps(make_metadata(name=f"{ticker} Ltd")))


def _bare_backtest(root, **kwargs):
    """A backtest over `root` with the real engine and a plain scorer."""
    engine = ComputeEngine()
    return MultiCohortBacktest(
        root, engine, SQGLPScorer(engine.metrics, engine.element_weights),
        None, **kwargs,
    )


@pytest.fixture
def factory(tmp_path):
    def build(**kwargs):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        return MultiCohortBacktest(tmp_path, engine, scorer, **kwargs)
    return build


class TestCutoffs:
    def test_many_cutoffs_one_per_qualifying_year(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")

        report = factory().run()

        cutoff_dates = {o["cutoff_date"] for o in report["observations"]}
        assert len(cutoff_dates) > 1
        # 12 financial years with min_history_years=5 -> candidate rows
        # i=4..11, eight candidates, every one accounted for.
        accounted = (
            len(report["observations"])
            + len(report["censored"])
            + len(report["failed_cutoffs"])
        )
        assert accounted == 8
        assert len(cutoff_dates) == len(report["observations"])

    def test_raising_stride_thins_cutoffs_not_their_span(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")
        bt = factory()
        yearly = bt._cutoff_dates(pd.read_csv(tmp_path / "GOODCO" / "financials.csv"))

        thinned = factory(stride_years=2)._cutoff_dates(
            pd.read_csv(tmp_path / "GOODCO" / "financials.csv")
        )

        assert len(thinned) < len(yearly)
        assert set(thinned) <= set(yearly)

    def test_no_observation_rests_on_less_history_than_promised(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")

        report = factory(min_history_years=6).run()

        assert report["observations"]
        assert all(o["years_scored"] >= 6 for o in report["observations"])

    def test_every_candidate_lands_in_exactly_one_bucket(self, tmp_path, factory):
        """No silent drops at the cutoff level: scored + censored + failed
        must account for every candidate the cutoff policy produced."""
        write_ticker(tmp_path, "GOODCO", price_days=2000)  # forward history ends early
        bt = factory()
        candidates = len(bt._cutoff_dates(
            pd.read_csv(tmp_path / "GOODCO" / "financials.csv"))
        )

        report = bt.run()

        accounted = (
            len(report["observations"])
            + len(report["censored"])
            + len(report["failed_cutoffs"])
        )
        assert accounted == candidates


class TestLeakage:
    def test_changing_todays_metadata_moves_no_score(self, tmp_path, factory):
        """Today's market cap and P/E would reward exactly the companies that
        already re-rated — the whole point of the rewind."""
        write_ticker(tmp_path, "GOODCO")
        first = {
            (o["ticker"], o["cutoff_date"]): o["composite_then"]
            for o in factory().run()["observations"]
        }

        meta = json.loads((tmp_path / "GOODCO" / "metadata.json").read_text())
        meta["Market Cap"] = 999_999.0
        meta["Stock P/E"] = 4242.0
        (tmp_path / "GOODCO" / "metadata.json").write_text(json.dumps(meta))

        second = {
            (o["ticker"], o["cutoff_date"]): o["composite_then"]
            for o in factory().run()["observations"]
        }
        assert second == first

    def test_a_window_starts_on_the_last_bar_at_or_before_its_cutoff(
        self, tmp_path, factory
    ):
        write_ticker(tmp_path, "GOODCO")
        price = pd.read_csv(tmp_path / "GOODCO" / "price_volume.csv")
        price["date"] = pd.to_datetime(price["date"])

        for obs in factory().run()["observations"]:
            start = pd.Timestamp(obs["forward_span"]["from"])
            cutoff = pd.Timestamp(obs["cutoff_date"])
            bars_before = price[price["date"] <= cutoff]["date"]
            assert start == bars_before.max()
            assert start <= cutoff


class TestFixedHorizonLabels:
    def test_forward_return_matches_a_hand_computed_value(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")
        bt = factory()
        obs = bt.run()["observations"][0]

        # Recompute independently from the CSV — do not reuse the code's span.
        price = pd.read_csv(tmp_path / "GOODCO" / "price_volume.csv")
        price["date"] = pd.to_datetime(price["date"])
        cutoff = pd.Timestamp(obs["cutoff_date"])
        target = cutoff + pd.DateOffset(years=bt.horizon_years)
        priced = price[price["adj_close"].notna()]
        start = float(priced[priced["date"] <= cutoff]["adj_close"].iloc[-1])
        end = float(priced[priced["date"] >= target]["adj_close"].iloc[0])
        years = (
            priced[priced["date"] >= target]["date"].iloc[0]
            - priced[priced["date"] <= cutoff]["date"].iloc[-1]
        ).days / 365.25
        expected = ((end / start) ** (1 / years) - 1) * 100

        assert obs["realized_cagr_pct"] == pytest.approx(expected, abs=0.01)

    def test_every_window_is_the_same_length(self, tmp_path, factory):
        """The one thing a rank correlation over mixed windows could never
        promise: like-for-like horizons."""
        write_ticker(tmp_path, "GOODCO")

        spans = [o["forward_span"]["years"] for o in factory().run()["observations"]]

        assert len(spans) > 1
        # The end bar is the FIRST bar at/after the target, so a weekend or
        # holiday can push a window a few days long — never short.
        assert all(3.0 <= s < 3.02 for s in spans)

    def test_an_incomplete_forward_window_is_censored_not_measured_short(
        self, tmp_path, factory
    ):
        write_ticker(tmp_path, "GOODCO", price_days=2000)  # ends ~Nov 2022

        report = factory().run()

        assert report["censored"], "a 2022-ending series must censor late cutoffs"
        for entry in report["censored"]:
            assert entry["status"] == "censored"
            assert "window incomplete" in entry["censor_reason"]
            assert entry["cutoff_date"] not in {
                o["cutoff_date"] for o in report["observations"]
            }

    def test_an_estimated_adj_close_series_is_refused_entirely(
        self, tmp_path, factory
    ):
        """The shared column policy: an unadjusted-close alias validates no
        realized return, so every label falls to the censored bucket with
        that reason rather than quietly measuring on raw closes."""
        write_ticker(tmp_path, "GOODCO", price_kwargs={"adj_close_is_estimated": True})

        report = factory().run()

        assert report["observations"] == []
        assert report["censored"]
        assert all(
            "unadjusted-close fallback" in e["censor_reason"]
            for e in report["censored"]
        )


class TestCoverageDenominator:
    def test_withheld_metrics_free_the_coverage_ceiling(self, tmp_path, factory):
        """Metrics no rewound date could score leave the denominator; what
        remains can clear the production coverage bar. Before this fix every
        backtest composite sat below it and the correlation reported n=0."""
        write_ticker(tmp_path, "GOODCO")

        report = factory().run()

        assert report["config"]["withheld_metric_count"] > 0
        best = max(o["coverage_composite"] for o in report["observations"])
        assert best > report["config"]["min_coverage"]
        assert report["correlations"]["n"] > 0

    def test_the_production_scorer_is_never_mutated(self, tmp_path):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        write_ticker(tmp_path, "GOODCO")

        MultiCohortBacktest(tmp_path, engine, scorer).run()

        assert scorer.applicability is None

    def test_withheld_means_failed_at_every_observation_not_any(self, tmp_path, factory):
        from boundless100x.compute_engine.metrics.base import MetricResult

        collected = [
            {"results": {"always": MetricResult(value=None, error="e"),
                         "sometimes": MetricResult(value=None, error="e")}},
            {"results": {"always": MetricResult(value=None, error="e"),
                         "sometimes": MetricResult(value=1.0)}},
        ]

        withheld = factory()._withheld_metrics(collected)

        assert set(withheld) == {"always"}

    def test_no_collected_observations_withhold_nothing(self, factory):
        assert factory()._withheld_metrics([]) == {}


class TestDistributionStatistics:
    @staticmethod
    def rows(*triples):
        return [
            {"ticker": f"T{i}", "cutoff_date": "2021-09-30",
             "composite_then": c, "coverage_composite": 1.0,
             "realized_cagr_pct": g, "fwd_multiple": m}
            for i, (c, g, m) in enumerate(triples)
        ]

    def test_quintile_buckets_are_rank_ordered_and_statistically_correct(self):
        rows = self.rows(
            (1.0, 10.0, 1.5), (2.0, 20.0, 2.5),      # bucket 1 (lowest scores)
            (3.0, 30.0, 3.5), (4.0, 40.0, 4.5),      # bucket 2
            (5.0, 50.0, 5.5), (6.0, 60.0, 6.5),      # bucket 3
            (7.0, 70.0, 7.5), (8.0, 80.0, 8.5),      # bucket 4
            (9.0, 90.0, 9.5), (10.0, 100.0, 10.5),   # bucket 5 (highest)
        )

        buckets = MultiCohortBacktest._quintile_buckets(rows)

        assert [b["bucket"] for b in buckets] == [1, 2, 3, 4, 5]
        assert all(b["n"] == 2 for b in buckets)
        assert buckets[0]["mean_cagr_pct"] == pytest.approx(15.0)
        assert buckets[-1]["mean_cagr_pct"] == pytest.approx(95.0)
        assert buckets[-1]["composite_range"] == [9.0, 10.0]
        assert buckets[2]["mean_multiple"] == pytest.approx(6.0)

    def test_quintiles_degrade_honestly_on_ties(self):
        assert MultiCohortBacktest._quintile_buckets(self.rows((1.0, 1.0, 1.0)), 5) == []

        # Scores tied everywhere: qcut labels every row NaN because nothing
        # ranks against itself — the answer is no distribution, not a fake
        # one-bucket one.
        tied = self.rows(*[(5.0, 10.0 * i, 2.0) for i in range(6)])
        assert MultiCohortBacktest._quintile_buckets(tied, 5) == []

        # One bucket is not a distribution either. Five tied scores and one
        # higher collapses to a single bin, whose "mean CAGR by bucket" is
        # just the pooled mean dressed as a comparison.
        one_bucket = self.rows(
            *([(5.0, 10.0 * i, 2.0) for i in range(5)] + [(6.0, 99.0, 3.0)])
        )
        assert MultiCohortBacktest._quintile_buckets(one_bucket, 5) == []

        # Near-tied but genuinely separable: duplicates dropped, everything
        # lands in fewer honest buckets and the counts still add up.
        near_tied = self.rows(
            *([(5.0, 10.0 * i, 2.0) for i in range(4)]
              + [(6.0, 99.0, 3.0), (7.0, 120.0, 4.0)])
        )
        buckets = MultiCohortBacktest._quintile_buckets(near_tied, 5)
        assert sum(b["n"] for b in buckets) == 6
        assert 2 <= len(buckets) < 5

    def test_tail_lift_counts_winners_in_the_top_fifth(self):
        # Winners are the two highest-scored rows of ten → share 1.0 against
        # a base rate of 0.2 → lift 5.0.
        rows = self.rows(
            (1.0, 5.0, 1.1), (2.0, 5.0, 1.2), (3.0, 5.0, 1.3),
            (4.0, 5.0, 1.4), (5.0, 5.0, 1.5), (6.0, 5.0, 1.6),
            (7.0, 5.0, 1.7), (8.0, 5.0, 1.8),
            (9.0, 40.0, 2.5), (10.0, 50.0, 3.0),
        )
        ranked_last_two_win = MultiCohortBacktest._tail_lift(rows, 2.0)
        assert ranked_last_two_win["winners"] == 2
        assert ranked_last_two_win["top_fifth_share"] == 1.0
        assert ranked_last_two_win["base_rate"] == 0.2
        assert ranked_last_two_win["lift"] == 5.0

    def test_tail_lift_without_winners_says_so_instead_of_dividing_by_zero(self):
        rows = self.rows(*[(float(i), 5.0, 1.1) for i in range(1, 11)])

        tail = MultiCohortBacktest._tail_lift(rows, 2.0)

        assert tail["winners"] == 0
        assert tail["lift"] is None

    def test_tail_lift_on_empty_rows_is_none(self):
        assert MultiCohortBacktest._tail_lift([], 2.0) is None

    def test_precision_at_k_skips_dates_too_small_to_mean_anything(self):
        big = [
            {"ticker": f"B{i}", "cutoff_date": "2021-09-30",
             "composite_then": float(i), "fwd_multiple": float(i) + 1}
            for i in range(6)  # k=3 needs 2*k=6 — exactly enough
        ]
        small = [
            {"ticker": f"S{i}", "cutoff_date": "2022-09-30",
             "composite_then": float(i), "fwd_multiple": float(i) + 1}
            for i in range(3)
        ]

        result = MultiCohortBacktest._precision_at_k(big + small, top_k=3)

        assert result["summary"]["dates_evaluated"] == 1
        only = result["dates"][0]
        assert only["picked"] == ["B5", "B4", "B3"]
        # picks: multiples 6, 5, 4 → mean 5.0; universe multiples 1..6 → mean 3.5
        assert only["pick_mean_multiple"] == pytest.approx(5.0)
        assert only["universe_mean_multiple"] == pytest.approx(3.5)
        assert result["summary"]["mean_pick_to_universe_ratio"] == pytest.approx(
            10 / 7, rel=0.001
        )


class TestReportAssembly:
    def test_young_tickers_are_skipped_once_with_a_reason(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")
        write_ticker(tmp_path, "YOUNGCO", years=4)

        report = factory().run()

        assert report["skipped"] == [{
            "ticker": "YOUNGCO",
            "reason": "only 4 scoreable years of financials (need 5)",
        }]
        assert "YOUNGCO" not in report["companies"]

    def test_eligibility_verdicts_roll_up_over_observations(self, tmp_path):
        from boundless100x.compute_engine.eligibility import EligibilityEvaluator

        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        write_ticker(tmp_path, "GOODCO")

        bt = MultiCohortBacktest(
            tmp_path, engine, scorer, EligibilityEvaluator(engine.gates)
        )
        report = bt.run()

        assert report["eligibility_cohorts"] is not None
        verdicts = {o["eligibility_then"]["verdict"] for o in report["observations"]}
        counted = sum(v["n"] for v in report["eligibility_cohorts"].values())
        assert counted == len(report["observations"])
        for o in report["observations"]:
            assert "size" not in o["eligibility_then"]["gates_evaluated"]
        assert verdicts  # every observation carries a verdict

    def test_excluded_metrics_are_reported_per_metric(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")

        excluded = factory().run()["excluded_metrics"]

        assert excluded
        for entry in excluded:
            assert entry["tickers_affected"] >= 1

    def test_run_is_deterministic(self, tmp_path, factory):
        write_ticker(tmp_path, "GOODCO")

        first = json.dumps(factory().run(), default=str, sort_keys=True)
        second = json.dumps(factory().run(), default=str, sort_keys=True)

        assert first == second

    def test_empty_universe_produces_an_empty_but_valid_report(self, factory):
        report = factory().run()

        assert report["observations"] == []
        assert report["correlations"]["n"] == 0
        assert report["quintiles"] == []
        assert report["tail_lift"] is None
        assert report["precision_at_k"]["dates"] == []
        assert report["limitations"]["qualifying_companies"] == 0


class TestTheBacktestScoresTheProductionRegime:
    """A diagnostic that scores on rules production abandoned tests nothing.

    The withheld-evidence shim originally REPLACED `SectorApplicability`
    rather than composing with it, so every sector rule vanished inside the
    rewind: EDELWEISS came back with `dcf_margin_of_safety` 0.0 at weight
    0.15, `dupont_turnover` 0.0, `cash_conversion` 0.0 and `debt_equity`
    0.0 — the exact "lender scored as a failing manufacturer" defect
    production removed.
    """

    def _shim(self, reasons=None):
        from boundless100x.compute_engine.engine import ComputeEngine
        from boundless100x.compute_engine.sector import SectorApplicability

        metrics = ComputeEngine().metrics
        return _WithheldEvidence(
            reasons or {}, SectorApplicability(metrics.keys())
        )

    def test_sector_rules_survive_alongside_the_withheld_set(self):
        merged = self._shim({"market_cap": "withheld"}).not_applicable_metrics(
            ("Finance", "Holding Company")
        )

        assert "market_cap" in merged, "withheld set lost"
        for lender_metric in ("dcf_margin_of_safety", "dupont_turnover",
                              "cash_conversion", "debt_equity"):
            assert lender_metric in merged, f"{lender_metric} scored as a manufacturer"

    def test_a_manufacturer_gains_nothing_it_should_not(self):
        """Composition must not hand every company the lender exclusions."""
        merged = self._shim().not_applicable_metrics(
            ("Industrial Products", "Plastic Products - Industrial")
        )

        assert merged == {}

    def test_the_shim_answers_flag_suppression_the_scorer_asks_for(self):
        """`SQGLPScorer.score()` calls this on every invocation. Without it the
        run depended on a defensive `except Exception` marked
        `pragma: no cover`."""
        shim = self._shim({"market_cap": "withheld"})

        assert hasattr(shim, "flag_suppressed_metrics")
        suppressed = shim.flag_suppressed_metrics(("Finance",))
        assert "market_cap" in suppressed
        # The base's own exceptions still stand: those two argue in the table
        # for keeping their warnings.
        assert "debt_equity" not in suppressed

    def test_a_scorer_without_an_applicability_table_still_works(self):
        assert _WithheldEvidence({"x": "y"}).not_applicable_metrics("Z") == {"x": "y"}
        assert _WithheldEvidence({"x": "y"}).flag_suppressed_metrics("Z") == {"x"}

    def test_the_real_backtest_scorer_carries_the_production_table(self, tmp_path):
        from boundless100x.compute_engine.engine import ComputeEngine
        from boundless100x.compute_engine.scorer import SQGLPScorer
        from boundless100x.compute_engine.sector import SectorApplicability

        engine = ComputeEngine()
        production = SQGLPScorer(
            engine.metrics, engine.element_weights,
            applicability=SectorApplicability(engine.metrics.keys()),
        )
        backtest = MultiCohortBacktest(tmp_path, engine, production)

        merged = backtest._backtest_scorer(
            {"market_cap": "withheld"}
        ).applicability.not_applicable_metrics(("Finance",))

        assert "dcf_margin_of_safety" in merged and "market_cap" in merged


class TestParametersAreRefusedRatherThanClamped:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_history_years": 0},   # range(-1, ...) -> rows.iloc[-1], duplicate cutoff
            {"horizon_years": 0},       # same bar both ends -> ZeroDivisionError
            {"stride_years": 0},
            {"min_history_years": -3},
        ],
    )
    def test_a_sub_one_parameter_is_refused_by_name(self, tmp_path, kwargs):
        with pytest.raises(ValueError) as excinfo:
            _bare_backtest(tmp_path, **kwargs)

        assert next(iter(kwargs)) in str(excinfo.value)

    def test_the_defaults_construct(self, tmp_path):
        assert _bare_backtest(tmp_path)


class TestEveryTickerLandsInABucket:
    def test_a_frame_with_no_parseable_labels_is_skipped_with_a_reason(
        self, tmp_path
    ):
        """Otherwise the ticker appears in none of observations, censored,
        failed or skipped, and the report understates its own universe."""
        ticker_dir = tmp_path / "NOLABELS"
        ticker_dir.mkdir()
        (ticker_dir / "financials.csv").write_text(
            "revenue\n" + "\n".join(str(100 + i) for i in range(8))
        )
        (ticker_dir / "price_volume.csv").write_text("date,close\n2020-01-01,10\n")

        backtest = _bare_backtest(tmp_path)
        cands, censored, failed, skip = backtest._collect_ticker(
            "NOLABELS", ticker_dir
        )

        assert (cands, censored, failed) == ([], [], [])
        assert skip and "parseable" in skip


class TestYearsScoredMatchesWhatTheMetricsSee:
    def test_a_stub_period_does_not_pad_the_history_count(self, tmp_path):
        """CAPLIPOINT changed year-end, so its financials carry `Jun 2015` and
        a nine-month `Mar 20169m`. Counting rows that every metric then
        discards made `years_scored` wrong and let a cutoff rest on less
        history than `min_history_years` promises."""
        import pandas as pd

        financials = pd.DataFrame({
            "year": ["Jun 2015", "Mar 20169m", "Mar 2017", "Mar 2018",
                     "Mar 2019", "Mar 2020", "Mar 2021"],
            "revenue": range(7),
        })
        backtest = _bare_backtest(tmp_path, min_history_years=5)

        cutoffs = backtest._cutoff_dates(financials)

        # Five metric-visible rows (Mar 2017..Mar 2021), so the first cutoff
        # is Mar 2021 — not Mar 2019, which only looks like the fifth year if
        # the stub and the superseded June year-end are counted.
        assert [str(c.date()) for c in cutoffs] == ["2021-09-30"]
