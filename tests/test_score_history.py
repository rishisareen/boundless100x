"""Score history: the append-only record momentum will be computed from.

Two properties carry the weight here. It must be append-only — a score not
written when the run happened cannot be recovered, and a rewrite that loses
earlier rows destroys history that no refetch can restore. And every row must
carry the registry hash, so a later diff can tell a company improving apart
from a threshold being edited.
"""

import json

import pytest

from boundless100x import score_history
from boundless100x.score_history import append_run, load_history, read_rows
from boundless100x.service import AnalysisResult


def scored_result(ticker="ASTRAL", composite=6.4, verdict="eligible", **kw) -> AnalysisResult:
    return AnalysisResult(
        ticker=ticker,
        scores={
            "composite": composite,
            "elements": {"growth": 7.1, "price": 4.2},
            "flags": kw.get("flags", []),
            "coverage": {"composite": kw.get("coverage", 0.93), "unscored": ["x", "y"]},
        },
        eligibility={"verdict": verdict} if verdict else None,
    )


@pytest.fixture
def log(tmp_path):
    return tmp_path / "score_history.jsonl"


class TestRowContent:
    def test_row_carries_the_registry_hash(self, log):
        row = append_run(scored_result(), "abc123def456", path=log)
        assert row["config_hash"] == "abc123def456"

    def test_row_records_score_verdict_and_coverage(self, log):
        row = append_run(scored_result(composite=6.4, coverage=0.93), "h", path=log)

        assert row["ticker"] == "ASTRAL"
        assert row["composite"] == 6.4
        assert row["elements"] == {"growth": 7.1, "price": 4.2}
        assert row["verdict"] == "eligible"
        assert row["coverage"] == 0.93

    def test_coverage_is_the_headline_share_not_the_unscored_list(self, log):
        """The log grows a row per run forever; the breakdown lives in scores.json."""
        row = append_run(scored_result(), "h", path=log)
        assert isinstance(row["coverage"], float)

    def test_rows_are_versioned_and_marked_organic(self, log):
        row = append_run(scored_result(), "h", path=log)
        assert row["schema_version"] == score_history.SCHEMA_VERSION
        assert row["synthetic"] is False

    def test_synthetic_rows_are_marked(self, log):
        """Backfill from truncated history must never pass as an organic run."""
        row = append_run(scored_result(), "h", path=log, synthetic=True)
        assert row["synthetic"] is True

    def test_absent_eligibility_records_indeterminate_not_a_pass(self, log):
        row = append_run(scored_result(verdict=None), "h", path=log)
        assert row["verdict"] == "indeterminate"


class TestAppendOnly:
    def test_two_runs_append_two_rows(self, log):
        append_run(scored_result(composite=6.4), "h", path=log)
        append_run(scored_result(composite=6.9), "h", path=log)

        assert [r["composite"] for r in read_rows(log)] == [6.4, 6.9]

    def test_appending_never_rewrites_existing_bytes(self, log):
        append_run(scored_result(composite=6.4), "h", path=log)
        first_line = log.read_text().splitlines()[0]

        append_run(scored_result(composite=6.9), "h", path=log)

        assert log.read_text().splitlines()[0] == first_line

    def test_each_row_is_one_self_contained_json_line(self, log):
        append_run(scored_result(), "h", path=log)
        append_run(scored_result(), "h", path=log)

        lines = log.read_text().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["ticker"] == "ASTRAL" for line in lines)

    def test_a_failed_scoring_run_records_nothing(self, log):
        """A run with no composite has nothing to remember."""
        assert append_run(AnalysisResult(ticker="NOPE"), "h", path=log) is None
        assert read_rows(log) == []

    def test_history_survives_a_truncated_final_line(self, log):
        append_run(scored_result(composite=6.4), "h", path=log)
        with open(log, "a") as f:
            f.write('{"ticker": "BROKEN", "compos')

        rows = read_rows(log)

        assert [r["composite"] for r in rows] == [6.4]


class TestReading:
    def test_rows_round_trip(self, log):
        written = append_run(scored_result(), "h", path=log)
        assert load_history("ASTRAL", path=log) == [written]

    def test_history_filters_by_ticker(self, log):
        append_run(scored_result(ticker="ASTRAL"), "h", path=log)
        append_run(scored_result(ticker="CDSL"), "h", path=log)

        assert [r["ticker"] for r in load_history("CDSL", path=log)] == ["CDSL"]

    def test_same_day_rerun_resolves_to_the_later_row(self, log):
        """The log keeps both; a reader sees one observation."""
        append_run(scored_result(composite=6.4), "h", path=log)
        append_run(scored_result(composite=6.9), "h", path=log)

        assert len(read_rows(log)) == 2
        assert [r["composite"] for r in load_history("ASTRAL", path=log)] == [6.9]

    def test_different_registry_regimes_are_both_kept(self, log):
        """Same day, different rulers — not one observation, and not comparable."""
        append_run(scored_result(composite=6.4), "regime_one", path=log)
        append_run(scored_result(composite=8.1), "regime_two", path=log)

        assert len(load_history("ASTRAL", path=log)) == 2

    def test_missing_file_reads_as_empty_not_an_error(self, tmp_path):
        assert load_history("ASTRAL", path=tmp_path / "absent.jsonl") == []


class TestServiceIntegration:
    def test_a_scored_analysis_records_a_row(self, monkeypatch, tmp_path):
        from tests.test_source_status import make_data, service_with_stub_suite

        data = make_data()
        data["source_status"] = {"financials": "ok", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        svc.history_path = tmp_path / "h.jsonl"

        result = svc.analyze("ASTRAL", use_llm=False)

        rows = read_rows(svc.history_path)
        assert len(rows) == 1
        assert rows[0]["composite"] == result.scores["composite"]
        assert rows[0]["config_hash"] == svc.engine.registry_hash

    def test_a_fatal_fetch_records_nothing(self, monkeypatch, tmp_path):
        import pandas as pd

        from tests.test_source_status import make_data, service_with_stub_suite

        data = make_data()
        data["financials"] = pd.DataFrame()
        data["source_status"] = {"financials": "failed: timeout", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        svc.history_path = tmp_path / "h.jsonl"

        svc.analyze("NOPE", use_llm=False)

        assert read_rows(svc.history_path) == []

    def test_a_history_write_failure_does_not_lose_the_analysis(
        self, monkeypatch, tmp_path
    ):
        from tests.test_source_status import make_data, service_with_stub_suite

        data = make_data()
        data["source_status"] = {"financials": "ok", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        monkeypatch.setattr(
            score_history,
            "append_run",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )

        result = svc.analyze("ASTRAL", use_llm=False)

        assert result.scores.get("composite") is not None
        assert any("Score history" in e for e in result.errors)
