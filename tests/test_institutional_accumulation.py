"""FII + DII accumulation streak — the fast lane's institutional-flow gate.

§9.2 asks for "FII+DII rising for 2+ consecutive quarters". Two readings of
that sentence give different numbers, so both are pinned here rather than left
to the implementation to settle:

* **File order is chronological, oldest row first.** `compute_promoter_trend`
  reads `iloc[-1]` as the latest quarter and `iloc[0]` as the earliest, and the
  cached files run Dec 2024 to Jun 2026. The streak therefore walks *backward*
  from the last row. An inverted read would report a company being sold as one
  being accumulated, which is why the same fixture reversed is tested too.
* **The counted unit is rises, not observations.** Four strictly increasing
  quarters yield 3, because three comparisons happen between four points — so
  the gate's `>= 2` means two consecutive rises across three quarters.

The gap rule is the third pin and the one with a live failure mode behind it:
a rise counts only between rows exactly one quarter apart, because "FII+DII
rose across a hole in the data" is missing evidence rather than a rise. This is
deliberately stricter than `compute_promoter_trend`'s positional read — a
20-quarter *trend* tolerates a gap, a consecutive-quarters *gate* is defined by
adjacency.
"""

import numpy as np
import pandas as pd
import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.builtin.size import (
    compute_institutional_accumulation_trend,
)
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import make_data, make_shareholding


def streak(combined: list[float], labels: list[str] | None = None,
           **frame_overrides):
    """Run the metric over a frame whose FII+DII sums to `combined`.

    The split between the two legs is arbitrary — the metric reads their sum —
    so DII is held flat and FII carries the movement, which keeps a fixture's
    intent readable at the call site.
    """
    frame = make_shareholding(
        quarters=len(combined),
        fii=[value - 1.0 for value in combined],
        dii=1.0,
    )
    if labels is not None:
        frame["quarter"] = labels
    for column, values in frame_overrides.items():
        frame[column] = values
    return compute_institutional_accumulation_trend({"shareholding": frame}, {})


class TestTheCountedUnit:
    def test_four_strictly_rising_quarters_count_three_rises(self):
        """Three comparisons between four points — not four."""
        result = streak([10.0, 11.0, 12.0, 13.0])

        assert result.ok
        assert result.value == 3

    def test_two_rises_clear_the_gates_threshold(self):
        assert streak([10.0, 11.0, 12.0]).value == 2

    def test_a_rising_streak_is_flagged(self):
        assert "institutional_accumulation_rising" in streak([10.0, 11.0, 12.0]).flags
        assert streak([10.0, 11.0]).flags == []

    def test_a_fall_in_the_latest_quarter_ends_the_streak_at_zero(self):
        """However long the run before it — the walk starts at the latest row."""
        assert streak([10.0, 11.0, 12.0, 13.0, 12.0]).value == 0

    def test_a_flat_latest_quarter_is_not_a_rise(self):
        assert streak([10.0, 11.0, 12.0, 12.0]).value == 0

    def test_an_earlier_fall_caps_the_streak_at_the_rises_after_it(self):
        assert streak([10.0, 12.0, 11.0, 12.0, 13.0]).value == 2

    def test_the_combined_series_travels_as_raw_series(self):
        result = streak([10.0, 11.0, 12.0])

        assert result.raw_series == pytest.approx([10.0, 11.0, 12.0])


class TestOrdering:
    def test_it_reads_the_frame_in_file_order(self):
        """The test that would have caught an inverted read.

        Ascending rows are what the fetched file holds, and they describe
        accumulation. The same rows reversed describe distribution, and no
        reading of them may report a streak.
        """
        rising = make_shareholding(quarters=4, fii=[9.0, 10.0, 11.0, 12.0], dii=1.0)

        assert compute_institutional_accumulation_trend(
            {"shareholding": rising}, {}
        ).value == 3
        assert not compute_institutional_accumulation_trend(
            {"shareholding": rising.iloc[::-1].reset_index(drop=True)}, {}
        ).ok

    def test_a_reversed_file_errors_rather_than_reporting_no_streak(self):
        """Fail-closed was the safe direction and the wrong kind of wrong.

        The backward walk breaks its adjacency test at the very first step on a
        newest-first frame, so the metric returned `0` — a *fail*, not an
        error. The gate then read "no accumulation" indefinitely on a company
        being steadily accumulated, and nothing anywhere said why: a silent
        zero and a real zero are the same number, and only one of them is a
        reading. Verified rather than assumed, an unwalkable frame is
        indeterminate and names its own cause.
        """
        rising = make_shareholding(quarters=4, fii=[9.0, 10.0, 11.0, 12.0], dii=1.0)

        result = compute_institutional_accumulation_trend(
            {"shareholding": rising.iloc[::-1].reset_index(drop=True)}, {}
        )

        assert result.value is None
        assert "ascending order" in result.error
        assert "sold" in result.error

    def test_a_repeated_quarter_is_unwalkable_too(self):
        """Not strictly ascending either, and the adjacency arithmetic would be
        comparing a quarter against itself."""
        result = streak(
            [10.0, 11.0, 12.0],
            labels=["Jun 2024", "Sep 2024", "Sep 2024"],
        )

        assert result.value is None
        assert "ascending order" in result.error

    def test_the_error_names_the_pair_that_broke_the_order(self):
        """An unreadable frame is a fetch to go and look at, and the owner
        needs the row to look at rather than the file."""
        result = streak(
            [10.0, 11.0, 12.0],
            labels=["Jun 2024", "Dec 2024", "Sep 2024"],
        )

        assert "Dec 2024" in result.error
        assert "Sep 2024" in result.error

    def test_the_gate_reads_indeterminate_rather_than_failing(self):
        """The consequence that matters upstream: `lane_gates` treats an
        errored metric as indeterminate, never as a condition that failed. A
        silent zero would have been a fail, and a fast lane nobody can enter
        looks exactly like a lane with no qualifying candidates."""
        rising = make_shareholding(quarters=4, fii=[9.0, 10.0, 11.0, 12.0], dii=1.0)

        result = compute_institutional_accumulation_trend(
            {"shareholding": rising.iloc[::-1].reset_index(drop=True)}, {}
        )

        assert result.ok is False
        assert "institutional_accumulation_rising" not in result.flags


class TestGapsAndUnreadableRows:
    def test_a_gap_before_the_latest_quarter_yields_no_streak(self):
        """Rising values, but the newest pair spans a missing December."""
        result = streak(
            [10.0, 11.0, 12.0, 13.0],
            labels=["Jun 2024", "Sep 2024", "Dec 2024", "Dec 2025"],
        )

        assert result.ok
        assert result.value == 0

    def test_a_gap_deeper_in_the_series_caps_the_streak_at_the_rises_after_it(self):
        """Four rises by position, two by adjacency — the walk stops at the hole."""
        result = streak(
            [10.0, 11.0, 12.0, 13.0, 14.0],
            labels=["Jun 2024", "Sep 2024", "Mar 2025", "Jun 2025", "Sep 2025"],
        )

        assert result.value == 2

    def test_an_unparsable_quarter_label_errors(self):
        """Gate-indeterminate, never a pass: adjacency cannot be verified."""
        result = streak(
            [10.0, 11.0, 12.0],
            labels=["Jun 2024", "Sep 2024", "Q3 FY25"],
        )

        assert not result.ok
        assert "Q3 FY25" in result.error

    def test_no_period_labels_at_all_errors(self):
        frame = make_shareholding(quarters=4, fii=[9.0, 10.0, 11.0, 12.0], dii=1.0)
        frame = frame.drop(columns=["quarter"])

        assert not compute_institutional_accumulation_trend(
            {"shareholding": frame}, {}
        ).ok

    def test_a_row_missing_a_leg_is_not_readable_and_ends_the_walk(self):
        """A difference computed with one leg absent manufactures a rise.

        Point-in-time `compute_institutional_holding` can treat a missing FII
        as zero and lose only precision. A *difference* cannot: the row below
        would read as a jump from 1.0 to 13.0 and count as a rise built out of
        a data gap — the same error the adjacency rule exists to prevent, one
        level down.
        """
        frame = make_shareholding(
            quarters=4, fii=[9.0, 10.0, np.nan, 12.0], dii=1.0
        )

        assert compute_institutional_accumulation_trend(
            {"shareholding": frame}, {}
        ).value == 0


class TestMissingData:
    def test_a_single_row_errors(self):
        result = streak([10.0])

        assert not result.ok
        assert "1" in result.error

    def test_an_empty_frame_errors(self):
        assert not compute_institutional_accumulation_trend(
            {"shareholding": pd.DataFrame()}, {}
        ).ok

    def test_an_absent_frame_errors(self):
        assert not compute_institutional_accumulation_trend({}, {}).ok


class TestRegistration:
    def test_it_is_declared_at_zero_weight_in_the_size_element(self):
        config = ComputeEngine().metrics["institutional_accumulation_streak"]

        assert config["element"] == "size"
        assert config["scoring"]["weight"] == 0.0

    def test_it_appears_in_details_unscored_and_unweighted(self):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        scores = scorer.score(engine.run_all(make_data()))

        entry = scores["details"]["institutional_accumulation_streak"]
        assert entry["weight"] == 0
        assert entry["score"] is None
