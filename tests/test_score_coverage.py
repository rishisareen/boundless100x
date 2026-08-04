"""A composite must say how much evidence it rests on.

Metrics that error are excluded and the remaining weights renormalise, so a
company scored on 84% of its weight prints the same clean X/10 as one scored on
96%. IRCTC loses RoCE average, RoCE consistency and the CAP proxy — most of
what Quality and Longevity mean — and still reads as directly comparable.
"""

import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import make_data


def scorer(**kwargs) -> tuple[SQGLPScorer, ComputeEngine]:
    engine = ComputeEngine()
    return SQGLPScorer(engine.metrics, engine.element_weights, **kwargs), engine


def score_with(dropped: set[str] = frozenset(), **kwargs) -> dict:
    sc, engine = scorer(**kwargs)
    results = engine.run_all(make_data())
    for metric_id in dropped:
        results[metric_id] = MetricResult(error="forced unavailable for test")
    return sc.score(results)


class TestCoverageIsReported:
    def test_scores_carry_a_coverage_block(self):
        coverage = score_with()["coverage"]

        assert 0 < coverage["composite"] <= 1
        assert set(coverage["elements"]) == set(ComputeEngine().element_weights)

    def test_dropping_a_metric_lowers_its_element_coverage(self):
        full = score_with()
        thinned = score_with({"roce_5yr_avg"})

        assert thinned["coverage"]["elements"]["quality_business"] < (
            full["coverage"]["elements"]["quality_business"]
        )

    def test_coverage_names_what_is_missing(self):
        coverage = score_with({"roce_5yr_avg", "peg_ratio"})["coverage"]

        assert "roce_5yr_avg" in coverage["unscored"]
        assert "peg_ratio" in coverage["unscored"]

    def test_full_data_reports_near_complete_coverage(self):
        """The synthetic fixture scores nearly everything."""
        assert score_with()["coverage"]["composite"] > 0.9

    def test_composite_coverage_falls_when_an_element_is_gutted(self):
        engine = ComputeEngine()
        price_metrics = {
            mid for mid, cfg in engine.metrics.items()
            if cfg["element"] == "price" and cfg["scoring"].get("weight", 0) > 0
        }

        gutted = score_with(price_metrics)

        assert gutted["coverage"]["elements"]["price"] == 0
        assert gutted["coverage"]["composite"] < 0.9


class TestConfidenceFlag:
    def test_thin_coverage_raises_a_flag(self):
        engine = ComputeEngine()
        heavy = {
            mid for mid, cfg in engine.metrics.items()
            if cfg["element"] in {"price", "longevity"}
            and cfg["scoring"].get("weight", 0) > 0
        }

        assert "low_data_coverage" in score_with(heavy)["flags"]

    def test_complete_data_raises_no_coverage_flag(self):
        assert "low_data_coverage" not in score_with()["flags"]

    def test_element_scored_on_nothing_is_none_not_zero(self):
        """An unmeasured element must not be scored as if it were measured badly."""
        engine = ComputeEngine()
        size_metrics = {
            mid for mid, cfg in engine.metrics.items()
            if cfg["element"] == "size" and cfg["scoring"].get("weight", 0) > 0
        }

        scores = score_with(size_metrics)

        assert scores["elements"]["size"] is None
        assert scores["coverage"]["elements"]["size"] == 0


class TestCoverageAccountsForWaivedMetrics:
    def test_history_waiver_shows_up_as_reduced_coverage(self):
        """Waived metrics are excluded on purpose — that still costs evidence."""
        sc, engine = scorer(history_waiver_mcap=5000.0)
        short = sc.score(engine.run_all(make_data(n=4, market_cap=3000.0)))
        full = sc.score(engine.run_all(make_data(n=10, market_cap=3000.0)))

        assert short["coverage"]["composite"] < full["coverage"]["composite"]
        assert "short_history_smallcap" in short["flags"]
