"""The two readings of `state_history` the whole lifecycle layer shares.

Both helpers live in `lifecycle/states.py` for the same reason, and it is the
reason these tests exist: `state_history` is the append-only record every
lifecycle decision argues from, and a helper that reads it must agree with
itself on every surface that reads it.

**`as_date` replaced three parsers that disagreed.** `evaluator` (the time
stop), `friction` (a holding period) and `reinvestment` (an idle-day count)
each carried a copy, each documenting the reconciliation as though it were the
only one, and the string branch differed: two parsed the whole string, one read
the first ten characters and discarded the rest. So one stored `at` could make
a time stop read indeterminate while the idle reading beside it printed a
confident number — a disagreement nobody looking at either surface could see.
The strict reading won, and the format table below is what pins that: a
timestamp that does not parse whole is a gap, and the layer's rule is that a
gap says so rather than becoming a date nobody supplied.

**`last_record_into` decides which visit is being measured.** The rule is
"last match, because re-entering a state restarts the clock", and it dates a
holding period, keys an `exit_id` and starts a time stop. Those must be the
same record or a report disagrees with the transition that produced it.
"""

from datetime import date, datetime

import pandas as pd
import pytest

from boundless100x.lifecycle import states


class TestAsDateShapes:
    """The types that actually arrive, from callers and from the stores."""

    def test_a_date_passes_through_unchanged(self):
        """`as_of` is a `date` throughout `lifecycle.checkpoints`."""
        assert states.as_date(date(2026, 8, 7)) == date(2026, 8, 7)

    def test_a_datetime_narrows_to_its_day(self):
        """A market bar has no time of day, and a tax bracket must not gain one."""
        assert states.as_date(datetime(2026, 8, 7, 23, 59)) == date(2026, 8, 7)

    def test_a_pandas_timestamp_narrows_to_its_day(self):
        """Lifted straight out of a price frame, and covered without a pandas
        branch: `pandas.Timestamp` subclasses `datetime.datetime`, which is why
        this module can stay standard-library-only and still be read by the
        layer that holds the frames."""
        assert states.as_date(pd.Timestamp("2026-08-07 10:30")) == date(2026, 8, 7)

    @pytest.mark.parametrize("value", [None, 20260807, 3.5, [], {"at": "2026-08-07"}])
    def test_anything_that_is_not_a_date_or_a_string_is_unknown(self, value):
        assert states.as_date(value) is None


class TestAsDateStrings:
    """The one branch the three copies disagreed on."""

    @pytest.mark.parametrize("value", [
        "2026-08-07",                    # an exit date: `str(as_of)`
        "2026-08-07T10:30:00",           # a transition `at`: `datetime.now()`
        "2026-08-07T10:30:00.123456",    # the same, with microseconds
        "2026-08-07T10:30:00+05:30",     # the same, with an offset
        "2026-08-07T10:30:00Z",
        "2026-08-07 10:30:00",
    ])
    def test_every_shape_the_stores_actually_write_is_read(self, value):
        assert states.as_date(value) == date(2026, 8, 7)

    @pytest.mark.parametrize("value", [
        "2026-08-07junk",
        "2026-08-07 10:30:00 IST",
        "07/08/2026",
        "H2 FY27",
        "",
    ])
    def test_a_string_that_does_not_parse_whole_is_unknown(self, value):
        """**The consolidation decision, pinned.**

        The lenient copy read `value[:10]` and threw the rest away, so the
        first two of these came back as a confident 2026-08-07 in the
        reinvestment queue while the evaluator's time stop called the identical
        string unreadable. Unifying strictly is what makes the two agree, and
        None here is what lets a caller name the value it could not read.
        """
        assert states.as_date(value) is None


class TestLastRecordInto:
    def test_the_last_matching_record_wins_not_the_first(self):
        """Re-entering a position restarts the clock. Dating a holding period
        from a stint that already ended puts it in the wrong tax bracket."""
        history = [
            {"to": "probe", "at": "2024-01-01T09:00:00"},
            {"to": "exited", "at": "2024-06-01T09:00:00"},
            {"to": "probe", "at": "2026-01-01T09:00:00"},
        ]

        assert states.last_record_into(history, "probe")["at"] == "2026-01-01T09:00:00"

    def test_a_state_never_reached_has_no_record(self):
        assert states.last_record_into([{"to": "watch", "at": "x"}], "probe") is None

    @pytest.mark.parametrize("history", [None, [], [42, "probe", None]])
    def test_an_absent_or_malformed_history_yields_no_record(self, history):
        """A hand-edited store is a fact of life; a non-dict row is skipped
        rather than raising, exactly as it was before this was shared."""
        assert states.last_record_into(history, "probe") is None

    def test_the_entry_wrapper_reads_the_same_rule(self):
        """`exit.py`, `advance.py` and `lane_view.py` hold whole entries; the
        evaluator holds the bare list. One implementation, two shapes — the
        alternative was the evaluator's own inline copy, which is what this
        replaced."""
        history = [
            {"to": "probe", "at": "2024-01-01T09:00:00"},
            {"to": "probe", "at": "2026-01-01T09:00:00"},
        ]
        entry = {"state_history": history}

        assert states.last_transition_into(entry, "probe") is (
            states.last_record_into(history, "probe")
        )

    def test_an_entry_with_no_history_key_is_not_an_error(self):
        assert states.last_transition_into({}, "probe") is None
