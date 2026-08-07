"""TTM revenue growth against the demonstrated CAGR — §9.2's growth gate.

The rule the fast lane states is "latest TTM growth ≥ historical CAGR", and no
existing metric answers it. `quarterly_momentum` is a *second difference* — it
asks whether growth is speeding up — so a company shrinking at a steady rate
reads as perfectly un-decelerating while its revenue falls away from everything
its five-year record promised. That case is tested here against both metrics at
once, because it is the entire reason this one exists.

Two contracts carry the rest of the file. The gap is stated in **percentage
points** (`ttm_growth_pct − revenue_cagr_5yr`), so the lane gate is a plain
`gte 0` and the sign means what a reader expects. And the historical anchor is
the existing `compute_cagr`, called rather than reimplemented, so the gate and
the scored growth element can never disagree about one company's CAGR.
"""

import shutil

import pandas as pd
import pytest
import yaml

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.builtin.forward_growth import (
    compute_quarterly_momentum,
)
from boundless100x.compute_engine.metrics.builtin.growth import (
    compute_ttm_growth_vs_cagr,
)
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import make_data, make_financials, make_quarterly


def gap(quarterly_yoy: float = 0.20, annual_growth: float = 0.20,
        periods: int = 12, **overrides):
    """The metric over a company growing `annual_growth` a year historically
    and `quarterly_yoy` right now."""
    data = make_data(
        financials={"revenue_growth": annual_growth},
        quarterly={"periods": periods, "revenue_yoy": quarterly_yoy},
    )
    data.update(overrides)
    return compute_ttm_growth_vs_cagr(data, {})


class TestTheGap:
    def test_ttm_growth_above_the_cagr_is_a_positive_gap(self):
        result = gap(quarterly_yoy=0.30, annual_growth=0.20)

        assert result.ok
        assert result.value == pytest.approx(10.0, abs=0.05)

    def test_ttm_growth_below_the_cagr_is_a_negative_gap(self):
        result = gap(quarterly_yoy=0.10, annual_growth=0.20)

        assert result.value == pytest.approx(-10.0, abs=0.05)

    def test_matching_growth_sits_exactly_on_the_gates_boundary(self):
        assert gap(quarterly_yoy=0.20, annual_growth=0.20).value == pytest.approx(
            0.0, abs=0.05
        )

    def test_both_readings_travel_in_metadata(self):
        meta = gap(quarterly_yoy=0.30, annual_growth=0.20).metadata

        assert meta["ttm_growth_pct"] == pytest.approx(30.0, abs=0.05)
        assert meta["cagr_pct"] == pytest.approx(20.0, abs=0.05)


class TestTheCaseThisMetricExistsFor:
    def test_a_company_shrinking_at_a_steady_rate_fails_this_gate(self):
        """Steady decline is un-decelerating, and still a broken thesis.

        Both metrics read the same fixture: revenue compounding at 20% across
        the annual record, now falling 10% year-over-year for eight straight
        quarters. Momentum sees no *change* in the rate and reports ~0 — no
        deceleration flag, nothing for a gate to catch. The gap sees the
        company has stopped doing what its record says it does.
        """
        data = make_data(
            financials={"revenue_growth": 0.20},
            quarterly={"periods": 12, "revenue_yoy": -0.10},
        )

        momentum = compute_quarterly_momentum(data, {})
        assert momentum.ok
        assert momentum.value == pytest.approx(0.0, abs=0.01)
        assert momentum.flags == []

        result = compute_ttm_growth_vs_cagr(data, {})
        assert result.ok
        assert result.value < 0
        assert result.value == pytest.approx(-30.0, abs=0.05)


class TestPeriodMatching:
    def test_an_interior_missing_quarter_errors_rather_than_fabricating(self):
        """Paired by label, so a hole inside the window is a refusal.

        Taking the latest four *rows* against the four before them would
        silently compare a TTM against a fifteen-month base — the same
        positional read that fabricated 1.4pp of movement in
        `quarterly_momentum`, here worth a whole quarter of revenue.
        """
        quarterly = make_quarterly(periods=12)
        assert quarterly["quarter"].iloc[9] == "Sep 2024"
        holed = quarterly.drop(index=9).reset_index(drop=True)

        result = gap(quarterly=holed)

        assert not result.ok
        assert "contiguous" in result.error
        assert "Mar 2025" in result.error

    def test_a_hole_outside_the_window_is_harmless(self):
        """Only the eight quarters actually read have to be contiguous."""
        quarterly = make_quarterly(periods=12).drop(index=0).reset_index(drop=True)

        assert gap(quarterly=quarterly).ok


class TestMissingData:
    def test_fewer_than_eight_resolvable_quarters_errors(self):
        result = gap(periods=7)

        assert not result.ok
        assert "8" in result.error

    def test_an_unreadable_period_label_does_not_count_toward_the_eight(self):
        quarterly = make_quarterly(periods=8)
        quarterly.loc[0, "quarter"] = "TTM"

        assert not gap(quarterly=quarterly).ok

    def test_an_unavailable_cagr_errors(self):
        """The anchor is the scored metric; without it there is no comparison."""
        financials = make_financials(10).drop(columns=["revenue"])

        result = gap(financials=financials)

        assert not result.ok
        assert "cagr" in result.error.lower() or "revenue" in result.error.lower()

    def test_a_non_positive_prior_year_base_errors(self):
        """A growth percentage off a zero base is not a reading."""
        quarterly = make_quarterly(periods=8, revenue=[0.0] * 4 + [10.0, 11.0, 12.0, 13.0])

        result = gap(quarterly=quarterly)

        assert not result.ok

    def test_an_absent_quarterly_frame_errors(self):
        assert not gap(quarterly=pd.DataFrame()).ok

    def test_a_missing_revenue_column_errors(self):
        quarterly = make_quarterly(periods=12).drop(columns=["revenue"])

        assert not gap(quarterly=quarterly).ok


class TestRegistration:
    def test_it_is_declared_at_zero_weight_in_the_growth_element(self):
        config = ComputeEngine().metrics["ttm_growth_vs_cagr"]

        assert config["element"] == "growth"
        assert config["scoring"]["weight"] == 0.0

    def test_it_appears_in_details_unscored_and_unweighted(self):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        scores = scorer.score(engine.run_all(make_data()))

        entry = scores["details"]["ttm_growth_vs_cagr"]
        assert entry["weight"] == 0
        assert entry["score"] is None
        assert entry["value"] is not None


class TestNonRegression:
    def test_registering_both_new_metrics_leaves_every_score_identical(self, tmp_path):
        """R8's proof for this unit, over *both* metrics at once.

        Phase 2 made the same proof one metric at a time; the fast lane adds
        two in two different weighted elements, and the failure mode being
        ruled out — a zero-weight metric leaking into an element mean or the
        coverage denominator — is a property of the pair being registered, not
        of either one alone.
        """
        shipped = ComputeEngine().registry_dir
        without = tmp_path / "without"
        shutil.copytree(shipped, without)

        for element, metric_id in (
            ("size", "institutional_accumulation_streak"),
            ("growth", "ttm_growth_vs_cagr"),
        ):
            path = without / "elements" / f"{element}.yaml"
            config = yaml.safe_load(path.read_text())
            assert metric_id in config["metrics"]
            del config["metrics"][metric_id]
            # sort_keys=False on purpose: the scorer accumulates each element's
            # weighted score in registry iteration order, so re-alphabetising
            # the metrics reorders that float summation and turns 6.5 into
            # 6.499999999999999 with nothing about the scoring having changed.
            path.write_text(yaml.safe_dump(config, sort_keys=False))

        data = make_data()

        def score(engine):
            scorer = SQGLPScorer(
                engine.metrics, engine.element_weights,
                history_waiver_mcap=engine.master.get("history_waiver_mcap"),
            )
            return scorer.score(engine.run_all(data))

        before = score(ComputeEngine(registry_dir=str(without)))
        after = score(ComputeEngine())

        assert after["composite"] == before["composite"]
        assert after["elements"] == before["elements"]
        assert after["coverage"] == before["coverage"]
