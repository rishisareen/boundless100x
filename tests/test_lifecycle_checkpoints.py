"""Checkpoint vocabulary and evaluation.

Two properties carry the weight. The vocabulary is closed, so a checkpoint can
always actually come due — an annual-grain metric would produce a promise that
is never checkable. And a data gap is `indeterminate`, never `missed`: a
thesis is ended by evidence that it is not happening, not by an absence of
evidence.
"""

from datetime import date

import pandas as pd
import pytest

from boundless100x.lifecycle import checkpoints as cp
from tests.conftest import quarter_labels


def quarterly(revenue=None, opm=None, pat=None, periods=8) -> pd.DataFrame:
    """A frame with the real column names *and labels* Phase 0's parser writes.

    The labels used to be `Q0..Q7`, which Screener never produces and which no
    period parser can read — so every checkpoint reading in this file silently
    exercised the position-matched fallback rather than the period matching
    production uses.
    """
    return pd.DataFrame({
        "quarter": quarter_labels(periods),
        "revenue": revenue or [100.0 + 10 * i for i in range(periods)],
        "expenses": [50.0] * periods,
        "operating_profit": [40.0] * periods,
        "opm_pct": opm or [25.0] * periods,
        "other_income": [1.0] * periods,
        "interest": [2.0] * periods,
        "depreciation": [3.0] * periods,
        "pbt": [35.0] * periods,
        "tax_pct": [25.0] * periods,
        "pat": pat or [26.0] * periods,
        "eps": [5.0] * periods,
    })


def shareholding(promoter=60.0, fii=10.0, dii=5.0, periods=8) -> pd.DataFrame:
    return pd.DataFrame({
        "quarter": quarter_labels(periods),
        "promoter_pct": [promoter] * periods,
        "fii_pct": [fii] * periods,
        "dii_pct": [dii] * periods,
        "govt_pct": [0.0] * periods,
        "public_pct": [25.0] * periods,
        "num_shareholders": [1000] * periods,
    })


def checkpoint(**overrides) -> dict:
    base = {
        "metric_id": "quarterly_opm_pct",
        "comparator": "gte",
        "threshold": 20.0,
        "due_date": "2026-01-01",
    }
    base.update(overrides)
    return base


PAST = date(2026, 8, 6)


class TestVocabulary:
    def setup_method(self):
        self.vocab = cp.load_vocabulary()

    def test_the_shipped_vocabulary_loads(self):
        assert self.vocab

    def test_every_entry_resolves_against_the_real_column_names(self):
        """Guards the vocabulary drifting from what the fetcher writes."""
        data = {"quarterly": quarterly(), "shareholding": shareholding()}
        for metric_id, spec in self.vocab.items():
            value, explanation, _ = cp._series_value(spec, data)
            assert value is not None, f"{metric_id}: {explanation}"

    def test_annual_grain_metrics_are_deliberately_absent(self):
        """A checkpoint on a 5yr average could never come due quarterly."""
        assert "roce_5yr_avg" not in self.vocab
        assert "trailing_peg" not in self.vocab

    def test_the_prompt_block_lists_every_id(self):
        block = cp.vocabulary_prompt_block(self.vocab)
        assert all(metric_id in block for metric_id in self.vocab)


class TestValidation:
    def test_a_well_formed_checkpoint_validates(self):
        assert cp.validate(checkpoint()) == []

    def test_an_id_outside_the_vocabulary_is_refused(self):
        errors = cp.validate(checkpoint(metric_id="roce_next_year"))
        assert any("not in the checkpoint vocabulary" in e for e in errors)

    def test_an_unknown_comparator_is_refused(self):
        assert cp.validate(checkpoint(comparator="approximately"))

    def test_a_non_numeric_threshold_is_refused(self):
        assert cp.validate(checkpoint(threshold="twenty percent"))

    def test_a_boolean_threshold_is_refused(self):
        """bool is an int subclass — it must not slip through as numeric."""
        assert cp.validate(checkpoint(threshold=True))

    def test_a_malformed_due_date_is_refused(self):
        assert cp.validate(checkpoint(due_date="next quarter"))

    def test_a_non_mapping_is_refused_without_raising(self):
        assert cp.validate("track margins") == ["checkpoint must be a mapping"]


class TestEvaluation:
    def data(self, **kw):
        return {"quarterly": quarterly(**kw), "shareholding": shareholding()}

    def test_a_met_checkpoint(self):
        outcome = cp.evaluate(checkpoint(threshold=20.0), self.data(), as_of=PAST)
        assert outcome["status"] == cp.MET
        assert outcome["value"] == 25.0

    def test_a_missed_checkpoint(self):
        outcome = cp.evaluate(checkpoint(threshold=30.0), self.data(), as_of=PAST)
        assert outcome["status"] == cp.MISSED

    def test_a_future_checkpoint_is_pending_not_missed(self):
        outcome = cp.evaluate(
            checkpoint(threshold=99.0, due_date="2027-01-01"), self.data(), as_of=PAST
        )
        assert outcome["status"] == cp.PENDING

    def test_missing_data_is_indeterminate_not_missed(self):
        """The rule that stops a stale fetch from ending a thesis."""
        outcome = cp.evaluate(checkpoint(), {}, as_of=PAST)
        assert outcome["status"] == cp.INDETERMINATE
        assert "refetch" in outcome["detail"]

    def test_an_invalid_checkpoint_is_indeterminate_not_raised(self):
        outcome = cp.evaluate(checkpoint(metric_id="invented"), self.data(), as_of=PAST)
        assert outcome["status"] == cp.INDETERMINATE

    def test_the_detail_names_the_period_and_the_number(self):
        outcome = cp.evaluate(checkpoint(), self.data(), as_of=PAST)
        assert "25.00" in outcome["detail"]
        assert outcome["period"] == quarter_labels(8)[-1]


class TestYearOverYear:
    def test_yoy_compares_against_the_same_quarter_a_year_earlier(self):
        """Four periods back, not one — else seasonality reads as a trend."""
        revenue = [100.0, 200.0, 300.0, 400.0, 150.0, 250.0, 350.0, 450.0]
        outcome = cp.evaluate(
            checkpoint(metric_id="quarterly_revenue_yoy_pct", comparator="gte",
                       threshold=10.0),
            {"quarterly": quarterly(revenue=revenue)},
            as_of=PAST,
        )
        # 450 vs 400 four quarters back = +12.5%, not 450 vs 350 = +28.6%
        assert outcome["value"] == pytest.approx(12.5)
        assert outcome["status"] == cp.MET

    def test_a_series_too_short_for_yoy_is_indeterminate(self):
        outcome = cp.evaluate(
            checkpoint(metric_id="quarterly_revenue_yoy_pct", threshold=10.0),
            {"quarterly": quarterly(periods=3)},
            as_of=PAST,
        )
        assert outcome["status"] == cp.INDETERMINATE
        assert "year-over-year" in outcome["detail"]

    def test_a_decline_is_reported_negative(self):
        revenue = [400.0, 300.0, 200.0, 100.0, 200.0, 150.0, 100.0, 50.0]
        outcome = cp.evaluate(
            checkpoint(metric_id="quarterly_revenue_yoy_pct", comparator="gte",
                       threshold=0.0),
            {"quarterly": quarterly(revenue=revenue)},
            as_of=PAST,
        )
        assert outcome["value"] == pytest.approx(-50.0)
        assert outcome["status"] == cp.MISSED


class TestShareholdingSeries:
    def test_promoter_holding_reads_the_latest_quarter(self):
        outcome = cp.evaluate(
            checkpoint(metric_id="promoter_holding_pct", comparator="gte",
                       threshold=50.0),
            {"shareholding": shareholding(promoter=62.5)},
            as_of=PAST,
        )
        assert outcome["value"] == 62.5

    def test_institutional_holding_sums_fii_and_dii(self):
        outcome = cp.evaluate(
            checkpoint(metric_id="institutional_holding_pct", comparator="gte",
                       threshold=12.0),
            {"shareholding": shareholding(fii=8.0, dii=6.0)},
            as_of=PAST,
        )
        assert outcome["value"] == pytest.approx(14.0)
        assert outcome["status"] == cp.MET


class TestSummary:
    def test_counts_by_status(self):
        data = {"quarterly": quarterly()}
        outcomes = cp.evaluate_all([
            checkpoint(threshold=20.0),                        # met
            checkpoint(threshold=30.0),                        # missed
            checkpoint(threshold=1.0, due_date="2027-01-01"),  # pending
            checkpoint(metric_id="promoter_holding_pct", threshold=1.0),  # no data
        ], data, as_of=PAST)

        assert cp.summarise(outcomes) == {
            "met": 1, "missed": 1, "pending": 1, "indeterminate": 1,
            "total": 4, "due": 2,
        }

    def test_indeterminate_never_counts_as_missed(self):
        outcomes = cp.evaluate_all([checkpoint(), checkpoint()], {}, as_of=PAST)
        assert cp.summarise(outcomes)["missed"] == 0
        assert cp.summarise(outcomes)["indeterminate"] == 2

    def test_an_empty_checkpoint_list_summarises_cleanly(self):
        assert cp.summarise([])["total"] == 0


class TestPeriodMatching:
    """A reading is anchored to the period it actually came from.

    Two defects of the same shape as `quarterly_momentum`'s: reading
    `values.iloc[-1]` off a dropna'd series while labelling it with the frame's
    last row gives a value from one quarter under another quarter's name, and
    `iloc[-1 - 4]` assumes the rows between are contiguous.
    """

    def yoy(self, values):
        return cp._series_value(
            {"source": "quarterly", "columns": ["revenue"], "transform": "yoy_pct"},
            {"quarterly": quarterly(revenue=values)},
        )

    def test_a_clean_series_reads_its_real_yoy(self):
        value, _, period = self.yoy([100 * (1.2 ** (i / 4)) for i in range(8)])

        assert value == pytest.approx(20.0)
        assert period == quarter_labels(8)[-1]

    def test_a_gap_refuses_rather_than_comparing_across_it(self):
        values = [100 * (1.2 ** (i / 4)) for i in range(8)]
        values[3] = None

        value, why, _ = self.yoy(values)

        assert value is None
        assert "one year before" in why

    def test_the_period_names_the_row_the_value_came_from(self):
        """A value from Dec must not be reported under Mar's name."""
        values = [100.0 + 10 * i for i in range(8)]
        values[-1] = None

        value, _, period = cp._series_value(
            {"source": "quarterly", "columns": ["revenue"], "transform": "latest"},
            {"quarterly": quarterly(revenue=values)},
        )

        assert value == 160.0
        assert period == quarter_labels(8)[-2]

    def test_a_sum_takes_every_column_from_one_period(self):
        """A total assembled from three different quarters totals nothing."""
        frame = quarterly(periods=8)
        frame.loc[frame.index[-1], "interest"] = None

        value, _, period = cp._series_value(
            {"source": "quarterly", "columns": ["interest", "depreciation"],
             "transform": "sum"},
            {"quarterly": frame},
        )

        assert value == 5.0
        assert period == quarter_labels(8)[-2]

    def test_unreadable_labels_fall_back_and_say_so(self):
        """A source whose labels cannot be parsed still reads, but discloses it."""
        frame = quarterly(periods=8)
        frame["quarter"] = [f"Q{i}" for i in range(8)]

        value, why, period = cp._series_value(
            {"source": "quarterly", "columns": ["revenue"], "transform": "latest"},
            {"quarterly": frame},
        )

        assert value == 170.0
        assert "position-matched" in why
        assert period is None


class TestRecordingRejectsPastDueDates:
    """A checkpoint already due when recorded was never monitored.

    Taken from the first real `analyze` run: Pass 2 returned four well-formed
    monitorables, every one dated 2025-09-30 — eleven months before the run.
    The prompt asked for "the ISO date by which it should be true" without ever
    saying what today was, so the model answered from its training cutoff.
    """

    PAST_RUN = [
        {"metric_id": "quarterly_pat_yoy_pct", "comparator": "gte",
         "threshold": 12, "due_date": "2025-09-30"},
        {"metric_id": "quarterly_opm_pct", "comparator": "gte",
         "threshold": 15, "due_date": "2025-09-30"},
    ]

    def test_the_real_past_dated_run_is_demoted_to_prose(self):
        recorded = cp.record_from_pass2(
            {"structured_monitorables": self.PAST_RUN}, as_of=date(2026, 8, 7)
        )

        assert recorded["checkpoints"] == []
        assert len(recorded["demoted"]) == 2
        assert all("not in the future" in r
                   for item in recorded["demoted"] for r in item["reasons"])

    def test_a_future_dated_monitorable_is_kept(self):
        future = [dict(self.PAST_RUN[0], due_date="2026-12-31")]

        recorded = cp.record_from_pass2(
            {"structured_monitorables": future}, as_of=date(2026, 8, 7)
        )

        assert len(recorded["checkpoints"]) == 1
        assert recorded["demoted"] == []

    def test_a_checkpoint_due_today_is_not_a_monitorable(self):
        today = [dict(self.PAST_RUN[0], due_date="2026-08-07")]

        recorded = cp.record_from_pass2(
            {"structured_monitorables": today}, as_of=date(2026, 8, 7)
        )

        assert recorded["checkpoints"] == []

    def test_a_stored_checkpoint_may_still_become_due_with_time(self):
        """The restriction is on recording, not on evaluation — otherwise no
        checkpoint could ever come due, which is the point of having them."""
        assert cp.validate(self.PAST_RUN[0]) == []
