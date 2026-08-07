"""`boundless100x.price_bars` — the shared bar-selection hygiene extracted
from `simulator/ledger.py` and `simulator/outputs.py` (Phase 4 residual
fix). `lifecycle/friction.py`'s own `_usable_bars` stays a separate,
untouched implementation (see that module and `price_bars.py`'s own
docstrings for why), so this file does not touch `tests/test_friction.py`.

Three things get proven, mirroring what `test_simulator_ledger.py` and
`test_simulator_outputs.py` already covered only indirectly (through
`Ledger`/`outputs.py`), so extracting this module does not reduce direct
test coverage of the hygiene itself:

  * `is_estimated`'s three-shape read (None/NaN, string, bool).
  * `clean_price_bars`'s column preference and row-dropping rules.
  * `bar_on_or_before`'s "on or before, never nearest" bar selection, and
    its raw-vs-already-cleaned frame detection.
"""

from datetime import date

import pandas as pd
import pytest

from boundless100x import price_bars


# ── is_estimated ────────────────────────────────────────────────────────


class TestIsEstimated:
    def test_none_reads_false(self):
        assert price_bars.is_estimated(None) is False

    def test_nan_reads_false(self):
        assert price_bars.is_estimated(float("nan")) is False

    @pytest.mark.parametrize("value", ["true", "True", " TRUE ", "1", "yes", "Yes"])
    def test_true_like_strings_read_true(self, value):
        assert price_bars.is_estimated(value) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "garbage"])
    def test_other_strings_read_false(self, value):
        assert price_bars.is_estimated(value) is False

    def test_bools_pass_through(self):
        assert price_bars.is_estimated(True) is True
        assert price_bars.is_estimated(False) is False

    def test_truthy_non_string_reads_via_bool(self):
        assert price_bars.is_estimated(1) is True
        assert price_bars.is_estimated(0) is False


# ── clean_price_bars ────────────────────────────────────────────────────


class TestCleanPriceBars:
    def test_none_or_non_dataframe_or_empty_is_none(self):
        assert price_bars.clean_price_bars(None) is None
        assert price_bars.clean_price_bars("not a frame") is None
        assert price_bars.clean_price_bars(pd.DataFrame()) is None

    def test_missing_date_column_is_none(self):
        frame = pd.DataFrame({"close": [100.0]})
        assert price_bars.clean_price_bars(frame) is None

    def test_missing_close_and_adj_close_columns_is_none(self):
        frame = pd.DataFrame({"date": [date(2023, 1, 2)], "volume": [1000]})
        assert price_bars.clean_price_bars(frame) is None

    def test_prefers_adj_close_over_close_when_both_present(self):
        frame = pd.DataFrame({
            "date": [date(2023, 1, 2)], "close": [999.0], "adj_close": [100.0],
        })
        cleaned = price_bars.clean_price_bars(frame)
        assert cleaned is not None
        assert cleaned.iloc[0]["price"] == 100.0

    def test_falls_back_to_close_when_no_adj_close_column(self):
        """A file predating the adjusted schema — a single `close`, no alias
        flag at all — is read straight off `close`."""
        frame = pd.DataFrame({"date": [date(2023, 1, 2)], "close": [100.0]})
        cleaned = price_bars.clean_price_bars(frame)
        assert cleaned is not None
        assert cleaned.iloc[0]["price"] == 100.0

    def test_drops_nan_price_and_date_rows(self):
        frame = pd.DataFrame({
            "date": [date(2023, 1, 2), None, date(2023, 1, 4)],
            "close": [100.0, 110.0, float("nan")],
        })
        cleaned = price_bars.clean_price_bars(frame)
        assert cleaned is not None
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["price"] == 100.0

    def test_drops_adj_close_is_estimated_aliased_rows(self):
        frame = pd.DataFrame({
            "date": [date(2023, 1, 2), date(2023, 1, 3)],
            "close": [100.0, 105.0],
            "adj_close": [100.0, 105.0],
            "adj_close_is_estimated": [False, True],
        })
        cleaned = price_bars.clean_price_bars(frame)
        assert cleaned is not None
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["date"] == pd.Timestamp("2023-01-02")

    def test_all_rows_unusable_is_none(self):
        frame = pd.DataFrame({
            "date": [date(2023, 1, 2), date(2023, 1, 3)],
            "close": [100.0, 105.0],
            "adj_close": [100.0, 105.0],
            "adj_close_is_estimated": [True, True],
        })
        assert price_bars.clean_price_bars(frame) is None

    def test_result_is_sorted_by_date(self):
        frame = pd.DataFrame({
            "date": [date(2023, 1, 5), date(2023, 1, 2), date(2023, 1, 3)],
            "close": [130.0, 100.0, 110.0],
        })
        cleaned = price_bars.clean_price_bars(frame)
        assert list(cleaned["date"]) == sorted(cleaned["date"])
        assert list(cleaned["price"]) == [100.0, 110.0, 130.0]


# ── bar_on_or_before ─────────────────────────────────────────────────────


RAW_FRAME = pd.DataFrame({
    "date": [date(2023, 1, 6), date(2023, 1, 9)],
    "close": [100.0, 110.0],
    "adj_close": [100.0, 110.0],
})


class TestBarOnOrBefore:
    def test_exact_date_match(self):
        result = price_bars.bar_on_or_before(RAW_FRAME, date(2023, 1, 9))
        assert result == {"date": date(2023, 1, 9), "price": 110.0}

    def test_gap_date_resolves_to_the_prior_bar_not_the_nearest(self):
        """Sunday 2023-01-08 sits one calendar day from Monday's bar and two
        from Friday's — a nearest-neighbour rule would wrongly pick Monday;
        'last bar on or before' correctly stays on Friday, because Monday's
        bar had not printed yet as of the lookup date."""
        result = price_bars.bar_on_or_before(RAW_FRAME, date(2023, 1, 8))
        assert result == {"date": date(2023, 1, 6), "price": 100.0}

    def test_before_the_first_bar_is_none(self):
        assert price_bars.bar_on_or_before(RAW_FRAME, date(2023, 1, 1)) is None

    def test_unusable_frame_is_none(self):
        assert price_bars.bar_on_or_before(None, date(2023, 1, 8)) is None
        assert price_bars.bar_on_or_before(pd.DataFrame(), date(2023, 1, 8)) is None

    def test_unparseable_as_of_is_none(self):
        assert price_bars.bar_on_or_before(RAW_FRAME, "not a date") is None

    def test_none_as_of_is_none(self):
        assert price_bars.bar_on_or_before(RAW_FRAME, None) is None

    def test_accepts_an_already_cleaned_frame_directly(self):
        """A frame this module's own `clean_price_bars` already produced
        (carrying `price`, not `close`/`adj_close`) is used as-is rather
        than re-cleaned — the shape a caching caller reuses across lookups."""
        cleaned = price_bars.clean_price_bars(RAW_FRAME)
        result = price_bars.bar_on_or_before(cleaned, date(2023, 1, 9))
        assert result == {"date": date(2023, 1, 9), "price": 110.0}

    def test_accepts_a_pandas_timestamp_or_iso_string_as_of(self):
        assert price_bars.bar_on_or_before(
            RAW_FRAME, pd.Timestamp("2023-01-09")
        ) == {"date": date(2023, 1, 9), "price": 110.0}
        assert price_bars.bar_on_or_before(RAW_FRAME, "2023-01-09") == {
            "date": date(2023, 1, 9), "price": 110.0,
        }
