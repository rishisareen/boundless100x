"""Short history should inform, not punish — but only where it is excusable.

Count-based longevity metrics ("years RoCE exceeded 15%", "years FCF was
positive") score an absolute count against thresholds designed for a ten-year
window, so a four-year-old company is structurally capped: identical quality
scored 9.76 on longevity with ten years of data and 5.00 with four.

For a small company that is a fact about the data, not a verdict on the
business. For a large one, thin history is a genuine red flag — so the waiver
is gated on its own market-cap threshold, deliberately not the 100x size gate.
"""

import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import make_data

SMALL_CAP = 3_000.0        # below the waiver threshold
MID_CAP = 28_000.0         # above the waiver, below the 100x size gate
LARGE_CAP = 95_000.0
WAIVER = 5_000.0


def score_company(years: int, market_cap: float, waiver: float | None = WAIVER) -> dict:
    engine = ComputeEngine()
    scorer = SQGLPScorer(engine.metrics, engine.element_weights, history_waiver_mcap=waiver)
    return scorer.score(engine.run_all(make_data(n=years, market_cap=market_cap)))


class TestShortWindowFlagging:
    def test_count_metrics_flag_a_short_window(self):
        results = ComputeEngine().run_all(make_data(n=4))

        assert any("short_window" in f for f in results["roce_consistency"].flags)

    def test_full_history_carries_no_short_window_flag(self):
        results = ComputeEngine().run_all(make_data(n=10))

        assert not any("short_window" in f for f in results["roce_consistency"].flags)


class TestSmallCapWaiver:
    def test_small_cap_short_history_is_not_dragged_down(self):
        short = score_company(4, SMALL_CAP)
        full = score_company(10, SMALL_CAP)

        assert short["elements"]["longevity"] == pytest.approx(
            full["elements"]["longevity"], abs=1.5
        )

    def test_waived_metrics_are_renormalised_not_zeroed(self):
        scores = score_company(4, SMALL_CAP)

        waived = [d for d in scores["details"].values() if d.get("waived")]
        assert waived
        assert all(d["weight"] == 0 for d in waived)
        assert all(d["score"] is None for d in waived)

    def test_waiver_surfaces_a_flag_for_the_report_and_llm(self):
        assert "short_history_smallcap" in score_company(4, SMALL_CAP)["flags"]

    def test_composite_still_computed(self):
        assert score_company(4, SMALL_CAP)["composite"] > 0


class TestLargeCapsKeepThePenalty:
    def test_mid_cap_above_the_waiver_keeps_the_penalty(self):
        """28,000 Cr clears the waiver but sits under the 100x size gate."""
        short = score_company(4, MID_CAP)
        full = score_company(10, MID_CAP)

        assert short["elements"]["longevity"] < full["elements"]["longevity"] - 1.0

    def test_large_cap_keeps_the_penalty(self):
        short = score_company(4, LARGE_CAP)
        full = score_company(10, LARGE_CAP)

        assert short["elements"]["longevity"] < full["elements"]["longevity"] - 1.0

    def test_no_waiver_flag_for_a_large_cap(self):
        assert "short_history_smallcap" not in score_company(4, LARGE_CAP)["flags"]

    def test_waiver_threshold_is_not_the_size_gate_ceiling(self):
        """A 28,000 Cr company would be waived if the gate ceiling were reused."""
        engine = ComputeEngine()

        assert engine.master["history_waiver_mcap"] < (
            engine.gates["size"]["conditions"][0]["threshold"]
        )


class TestBoundaryAndFallback:
    def test_behaviour_switches_at_the_threshold(self):
        below = score_company(4, WAIVER - 1)
        above = score_company(4, WAIVER + 1)

        assert "short_history_smallcap" in below["flags"]
        assert "short_history_smallcap" not in above["flags"]

    def test_no_waiver_configured_preserves_prior_behaviour(self):
        scores = score_company(4, SMALL_CAP, waiver=None)

        assert "short_history_smallcap" not in scores["flags"]
        assert not any(d.get("waived") for d in scores["details"].values())

    def test_unknown_market_cap_does_not_waive(self):
        """Absent size evidence must not buy a company the benefit of the doubt."""
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights, history_waiver_mcap=WAIVER)
        data = make_data(n=4)
        data["metadata"].pop("Market Cap", None)

        scores = scorer.score(engine.run_all(data))

        assert "short_history_smallcap" not in scores["flags"]

    def test_full_history_small_cap_waives_nothing(self):
        scores = score_company(10, SMALL_CAP)

        assert not any(d.get("waived") for d in scores["details"].values())
