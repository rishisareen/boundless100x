"""Score momentum read off the append-only history.

The failure this module exists to prevent is a quiet one: at landing there is
no usable history (A3), and a zero delta and an unknown delta look identical in
a table while meaning opposite things. So the tests that matter most are not
the arithmetic ones — they are the ones that pin insufficient-history as a
distinct outcome, refuse a diff across a regime boundary, and make every figure
state the span it covers.
"""

import pytest

from boundless100x import trajectory
from tests.conftest import make_history_rows, write_history


def rows_at(dates, composites, **kw):
    return make_history_rows(dates=dates, composites=composites, **kw)


class TestArithmetic:
    def test_two_rows_in_one_regime_produce_a_composite_delta(self):
        rows = rows_at(["2026-01-01", "2026-04-01"], [6.0, 6.4])
        result = trajectory.compute_momentum("TEST", rows=rows)

        assert result["status"] == trajectory.OK
        assert result["latest"]["composite_delta"] == pytest.approx(0.4)
        assert result["latest"]["composite_from"] == 6.0
        assert result["latest"]["composite_to"] == 6.4

    def test_element_deltas_match_hand_computed_values(self):
        rows = rows_at(
            ["2026-01-01", "2026-04-01"], [6.0, 6.4],
            elements=[
                {"growth": 7.0, "price": 4.0},
                {"growth": 7.8, "price": 3.5},
            ],
        )
        latest = trajectory.compute_momentum("TEST", rows=rows)["latest"]

        assert latest["element_deltas"]["growth"] == pytest.approx(0.8)
        assert latest["element_deltas"]["price"] == pytest.approx(-0.5)

    def test_three_rows_produce_two_consecutive_steps(self):
        rows = rows_at(["2026-01-01", "2026-04-01", "2026-07-01"], [6.0, 6.4, 6.2])
        result = trajectory.compute_momentum("TEST", rows=rows)

        steps = result["regimes"][0]["steps"]
        assert [round(s["composite_delta"], 2) for s in steps] == [0.4, -0.2]
        # `latest` is the freshest step, not the first.
        assert result["latest"]["to_date"] == "2026-07-01"


class TestRegimePartitioning:
    def test_two_regimes_produce_two_series_never_a_bridging_figure(self):
        """Rows under different hashes came from different rulers (KTD5)."""
        rows = (
            rows_at(["2026-01-01", "2026-04-01"], [6.0, 6.4], config_hash="aaa")
            + rows_at(["2026-07-01", "2026-10-01"], [8.0, 8.5], config_hash="bbb")
        )
        result = trajectory.compute_momentum("TEST", rows=rows)

        assert len(result["regimes"]) == 2
        deltas = [
            round(step["composite_delta"], 2)
            for regime in result["regimes"]
            for step in regime["steps"]
        ]
        # 0.4 and 0.5 within each regime; never the 1.6 that bridging would give.
        assert deltas == [0.4, 0.5]

    def test_no_step_ever_spans_a_regime_boundary(self):
        rows = (
            rows_at(["2026-01-01", "2026-04-01"], [6.0, 6.4], config_hash="aaa")
            + rows_at(["2026-07-01", "2026-10-01"], [8.0, 8.5], config_hash="bbb")
        )
        result = trajectory.compute_momentum("TEST", rows=rows)

        for regime in result["regimes"]:
            for step in regime["steps"]:
                assert step["config_hash"] == regime["config_hash"]

    def test_one_row_per_regime_across_two_regimes_is_still_insufficient(self):
        rows = (
            rows_at(["2026-01-01"], [6.0], config_hash="aaa")
            + rows_at(["2026-07-01"], [8.0], config_hash="bbb")
        )
        result = trajectory.compute_momentum("TEST", rows=rows)

        assert result["status"] == trajectory.INSUFFICIENT_HISTORY
        assert result["latest"] is None


class TestSyntheticSeparation:
    def test_synthetic_and_organic_rows_are_not_averaged_into_one_figure(self):
        """A backfilled row and a real one are not two observations of one thing."""
        rows = (
            rows_at(["2026-01-01", "2026-02-01"], [5.0, 5.2], synthetic=True)
            + rows_at(["2026-04-01", "2026-07-01"], [6.0, 6.4])
        )
        result = trajectory.compute_momentum("TEST", rows=rows)

        assert len(result["regimes"]) == 2
        assert {r["synthetic"] for r in result["regimes"]} == {True, False}
        for regime in result["regimes"]:
            for step in regime["steps"]:
                assert step["synthetic"] == regime["synthetic"]

    def test_the_headline_reading_is_never_a_synthetic_one(self):
        rows = (
            rows_at(["2026-08-01", "2026-09-01"], [5.0, 5.9], synthetic=True)
            + rows_at(["2026-01-01", "2026-04-01"], [6.0, 6.4])
        )
        result = trajectory.compute_momentum("TEST", rows=rows)

        # The synthetic pair is more recent, but `latest` reports the organic one.
        assert result["latest"]["synthetic"] is False
        assert result["latest"]["to_date"] == "2026-04-01"

    def test_only_synthetic_history_reads_as_insufficient_organic_history(self):
        rows = rows_at(["2026-01-01", "2026-04-01"], [5.0, 5.2], synthetic=True)
        result = trajectory.compute_momentum("TEST", rows=rows)

        assert result["status"] == trajectory.INSUFFICIENT_HISTORY
        assert result["latest"] is None
        assert "synthetic" in result["reason"]


class TestIntervalHonesty:
    def test_the_interval_is_the_actual_day_gap(self):
        rows = rows_at(["2025-01-01", "2026-01-01"], [6.0, 6.4])
        latest = trajectory.compute_momentum("TEST", rows=rows)["latest"]

        assert latest["interval_days"] == 365

    def test_an_annual_gap_is_not_labelled_as_quarterly(self):
        annual = trajectory.compute_momentum(
            "TEST", rows=rows_at(["2025-01-01", "2026-01-01"], [6.0, 6.4])
        )["latest"]
        quarterly = trajectory.compute_momentum(
            "TEST", rows=rows_at(["2025-10-03", "2026-01-01"], [6.0, 6.4])
        )["latest"]

        assert annual["span"] != quarterly["span"]
        assert annual["interval_days"] > quarterly["interval_days"]
        assert "year" in annual["span"]
        assert "3 months" in quarterly["span"]

    def test_a_half_year_gap_is_not_labelled_as_a_quarter(self):
        """A bucket wide enough to cover 90 to 200 days calls half a year one
        quarter, which is exactly the misreading the label exists to prevent."""
        half = trajectory.compute_momentum(
            "TEST", rows=rows_at(["2026-02-01", "2026-08-06"], [6.1, 6.7])
        )["latest"]

        assert half["interval_days"] == 186
        assert "6 months" in half["span"]

    def test_every_step_states_its_own_span(self):
        rows = rows_at(["2024-01-01", "2025-01-01", "2025-04-01"], [6.0, 6.4, 6.5])
        steps = trajectory.compute_momentum("TEST", rows=rows)["regimes"][0]["steps"]

        assert [s["interval_days"] for s in steps] == [366, 90]
        assert all(s["span"] for s in steps)


class TestInsufficientHistoryIsDistinctFromFlat:
    def test_one_row_reports_insufficient_history(self):
        result = trajectory.compute_momentum("TEST", rows=rows_at(["2026-01-01"], [6.0]))

        assert result["status"] == trajectory.INSUFFICIENT_HISTORY
        assert result["latest"] is None
        assert result["reason"]

    def test_a_genuine_zero_delta_is_not_insufficient_history(self):
        """Two equal scores mean flat. One score means unknown. Never the same."""
        result = trajectory.compute_momentum(
            "TEST", rows=rows_at(["2026-01-01", "2026-04-01"], [6.0, 6.0])
        )

        assert result["status"] == trajectory.OK
        assert result["latest"]["composite_delta"] == 0.0

    def test_empty_history_reads_as_insufficient_rather_than_raising(self):
        result = trajectory.compute_momentum("TEST", rows=[])

        assert result["status"] == trajectory.INSUFFICIENT_HISTORY
        assert result["regimes"] == []
        assert result["latest"] is None

    def test_a_missing_history_file_reads_as_insufficient(self, tmp_path):
        result = trajectory.compute_momentum("TEST", path=tmp_path / "absent.jsonl")
        assert result["status"] == trajectory.INSUFFICIENT_HISTORY


class TestDegradedRows:
    def test_an_element_absent_from_the_earlier_row_yields_no_delta(self):
        """An absent element is unknown, not zero — the same rule as everywhere."""
        rows = rows_at(
            ["2026-01-01", "2026-04-01"], [6.0, 6.4],
            elements=[{"growth": 7.0}, {"growth": 7.5, "price": 4.0}],
        )
        latest = trajectory.compute_momentum("TEST", rows=rows)["latest"]

        assert "price" not in latest["element_deltas"]
        assert latest["element_deltas"]["growth"] == pytest.approx(0.5)

    def test_a_none_element_score_yields_no_delta(self):
        rows = rows_at(
            ["2026-01-01", "2026-04-01"], [6.0, 6.4],
            elements=[{"growth": None}, {"growth": 7.5}],
        )
        latest = trajectory.compute_momentum("TEST", rows=rows)["latest"]
        assert "growth" not in latest["element_deltas"]

    def test_a_row_with_no_composite_is_skipped_rather_than_diffed_as_zero(self):
        rows = rows_at(["2026-01-01", "2026-04-01", "2026-07-01"], [6.0, None, 6.4])
        result = trajectory.compute_momentum("TEST", rows=rows)

        steps = result["regimes"][0]["steps"]
        assert len(steps) == 1
        assert steps[0]["from_date"] == "2026-01-01"
        assert steps[0]["to_date"] == "2026-07-01"

    def test_an_unparseable_date_does_not_break_the_read(self):
        rows = rows_at(["not-a-date", "2026-04-01", "2026-07-01"], [6.0, 6.2, 6.4])
        result = trajectory.compute_momentum("TEST", rows=rows)

        assert result["status"] == trajectory.OK
        assert len(result["regimes"][0]["steps"]) == 1


class TestReadingFromDisk:
    def test_rows_are_read_through_load_history_for_the_named_ticker(self, tmp_path):
        path = tmp_path / "history.jsonl"
        write_history(path, make_history_rows(ticker="ASTRAL", composites=[6.0, 6.6]))
        write_history(path, make_history_rows(ticker="CDSL", composites=[4.0, 3.0]))

        astral = trajectory.compute_momentum("ASTRAL", path=path)
        assert astral["latest"]["composite_delta"] == pytest.approx(0.6)

    def test_same_day_reruns_are_resolved_before_diffing(self, tmp_path):
        """load_history keeps the later row; a same-day pair is not a step."""
        path = tmp_path / "history.jsonl"
        write_history(path, make_history_rows(dates=["2026-01-01"], composites=[6.0]))
        write_history(path, make_history_rows(dates=["2026-01-01"], composites=[6.9]))

        result = trajectory.compute_momentum("TEST", path=path)
        assert result["status"] == trajectory.INSUFFICIENT_HISTORY


class TestServiceIntegration:
    def test_a_scored_analysis_populates_momentum(self, monkeypatch, tmp_path):
        from tests.test_source_status import make_data, service_with_stub_suite

        data = make_data()
        data["source_status"] = {"financials": "ok", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        svc.history_path = tmp_path / "h.jsonl"

        result = svc.analyze("ASTRAL", use_llm=False)

        # One run, so one row: the honest answer on the day this lands (A3).
        assert result.momentum["status"] == trajectory.INSUFFICIENT_HISTORY

    def test_momentum_reproduces_from_stored_rows_without_rescoring(
        self, monkeypatch, tmp_path
    ):
        from tests.test_source_status import make_data, service_with_stub_suite

        data = make_data()
        data["source_status"] = {"financials": "ok", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        svc.history_path = tmp_path / "h.jsonl"

        # A prior run under the same regime, recorded on an earlier date.
        write_history(
            svc.history_path,
            make_history_rows(
                ticker="ASTRAL", dates=["2026-01-01"], composites=[5.0],
                config_hash=svc.engine.registry_hash,
            ),
        )
        result = svc.analyze("ASTRAL", use_llm=False)

        assert result.momentum["status"] == trajectory.OK
        assert result.momentum["latest"]["composite_from"] == 5.0
        assert result.momentum["latest"]["composite_to"] == result.scores["composite"]

    def test_a_momentum_failure_does_not_cost_the_caller_the_analysis(
        self, monkeypatch, tmp_path
    ):
        from tests.test_source_status import make_data, service_with_stub_suite

        data = make_data()
        data["source_status"] = {"financials": "ok", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        svc.history_path = tmp_path / "h.jsonl"
        monkeypatch.setattr(
            trajectory, "compute_momentum",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        result = svc.analyze("ASTRAL", use_llm=False)

        assert result.scores["composite"] is not None
        assert any("momentum" in e.lower() for e in result.errors)
