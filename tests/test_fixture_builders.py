"""The Phase 2 fixture builders themselves.

Every later Phase 2 test reads its inputs from these builders rather than from
`raw_data/`, so a builder that drifts from the real fetched schema would make a
whole unit's tests agree with each other and disagree with production. These
tests pin the builders against the schemas they mirror: the quarterly parser's
column map, the section sidecar's shape, and the score-history row.
"""

import pytest
import yaml

from boundless100x.lifecycle.checkpoints import DEFAULT_VOCABULARY_PATH
from boundless100x import score_history
from tests.conftest import (
    QUARTERLY_COLUMNS,
    make_ar_sections,
    make_balance_sheet,
    make_cashflow,
    make_data,
    make_financials,
    make_history_rows,
    make_price,
    make_quarterly,
    make_ratios,
    make_shareholding,
    quarter_labels,
    write_history,
)


class TestQuarterly:
    def test_carries_every_column_the_checkpoint_vocabulary_reads(self):
        """A vocabulary entry cannot silently drift away from the fixture.

        `checkpoint_vocabulary.yaml` is the closed list Pass 2 is given, and
        every `source: quarterly` entry names a column. If the builder stopped
        emitting one, the checkpoint would read indeterminate in tests for a
        reason production never has.
        """
        vocabulary = yaml.safe_load(DEFAULT_VOCABULARY_PATH.read_text())["checkpoints"]
        needed = {
            column
            for spec in vocabulary.values()
            if spec.get("source") == "quarterly"
            for column in spec.get("columns", [])
        }
        assert needed  # the vocabulary really does reference quarterly columns
        assert needed <= set(make_quarterly().columns)

    def test_columns_match_the_screener_quarterly_map(self):
        """Asserted against the real parser's map, not against our own copy.

        `QUARTERLY_COLUMNS` claims parity with `QTR_LABEL_MAP`. Comparing the
        builder only to that claim would let a rename in the live-scraper
        adapter — the code most likely to change — pass silently while every
        quarterly fixture in the suite started testing a shape production no
        longer emits.
        """
        from boundless100x.data_fetcher.fetch_financials import QTR_LABEL_MAP

        # dict.fromkeys dedupes the Sales/Revenue -> revenue alias in order.
        expected = ["quarter"] + list(dict.fromkeys(QTR_LABEL_MAP.values()))

        assert list(QUARTERLY_COLUMNS) == expected
        assert list(make_quarterly().columns) == expected

    def test_period_count_is_what_was_asked_for(self):
        assert len(make_quarterly(periods=6)) == 6
        assert len(make_quarterly(periods=13)) == 13

    def test_labels_run_oldest_first_and_end_on_a_march_quarter(self):
        labels = quarter_labels(8, end_year=2025)
        assert labels[-1] == "Mar 2025"
        assert labels[0] == "Jun 2023"

    def test_growth_is_flat_year_over_year_by_construction(self):
        """A steady builder means momentum a test sees came from its override."""
        df = make_quarterly(periods=12, revenue_yoy=0.20)
        revenue = df["revenue"].tolist()
        for i in range(4, len(revenue)):
            assert revenue[i] / revenue[i - 4] == pytest.approx(1.20)

    def test_overrides_replace_a_column(self):
        df = make_quarterly(periods=4, pat=[1.0, 2.0, 3.0, 4.0])
        assert df["pat"].tolist() == [1.0, 2.0, 3.0, 4.0]


class TestAnnualReportSections:
    def test_fallback_provenance_is_uniform(self):
        sections = make_ar_sections(provenance="fallback")["2025"]
        assert {s["provenance"] for s in sections.values()} == {"fallback"}

    def test_found_provenance_is_uniform(self):
        sections = make_ar_sections(provenance="found")["2025"]
        assert {s["provenance"] for s in sections.values()} == {"found"}

    def test_mixed_provenance_is_expressible(self):
        """10 of 29 real report-years carry `mdna: fallback` beside a found sibling."""
        sections = make_ar_sections(
            provenance="found", per_section_provenance={"mdna": "fallback"}
        )["2025"]
        assert sections["mdna"]["provenance"] == "fallback"
        assert sections["chairman"]["provenance"] == "found"

    def test_shape_matches_the_sidecar(self):
        sections = make_ar_sections(years=["2024", "2025"])
        assert set(sections) == {"2024", "2025"}
        for year in sections.values():
            for section in year.values():
                assert set(section) == {"text", "provenance", "start_page"}

    def test_found_sections_carry_a_start_page_and_fallback_ones_do_not(self):
        found = make_ar_sections(provenance="found")["2025"]["mdna"]
        fallback = make_ar_sections(provenance="fallback")["2025"]["mdna"]
        assert isinstance(found["start_page"], int)
        assert fallback["start_page"] is None


class TestHistoryRows:
    def test_two_regimes_are_not_collapsed_by_load_history(self, tmp_path):
        """`load_history` dedupes within a regime, never across one."""
        path = tmp_path / "history.jsonl"
        write_history(path, make_history_rows(config_hash="aaa111aaa111"))
        write_history(path, make_history_rows(config_hash="bbb222bbb222"))

        rows = score_history.load_history("TEST", path=path)
        assert len(rows) == 4
        assert {r["config_hash"] for r in rows} == {"aaa111aaa111", "bbb222bbb222"}

    def test_same_day_rerun_in_one_regime_collapses_to_the_later_row(self, tmp_path):
        path = tmp_path / "history.jsonl"
        write_history(path, make_history_rows(dates=["2026-01-01"], composites=[6.0]))
        write_history(path, make_history_rows(dates=["2026-01-01"], composites=[6.9]))

        rows = score_history.load_history("TEST", path=path)
        assert len(rows) == 1
        assert rows[0]["composite"] == 6.9

    def test_row_shape_matches_what_append_run_writes(self, tmp_path):
        from tests.conftest import make_result

        written = score_history.append_run(
            make_result(), "abc123abc123", path=tmp_path / "history.jsonl"
        )
        assert set(make_history_rows()[0]) == set(written)

    def test_synthetic_rows_are_marked(self):
        assert make_history_rows(synthetic=True)[0]["synthetic"] is True
        assert make_history_rows()[0]["synthetic"] is False


class TestPrice:
    def test_the_default_carries_the_schema_every_cached_ticker_has(self):
        """22 of 22 real files carry `adj_close`; the default must too.

        It was opt-in while 13 files predated the adjusted-series schema. After
        the refetch the old default described no real ticker, which is the kind
        of drift that let a NaN in the newest adjusted bar hide until the
        backtest surfaced it.
        """
        df = make_price(days=10)
        assert "close" in df.columns
        assert "adj_close" in df.columns

    def test_the_legacy_shape_is_an_explicit_opt_out(self):
        """Files predating the split still exist in the wild, just not here."""
        df = make_price(days=10, adj_close=False)
        assert "close" in df.columns
        assert "adj_close" not in df.columns

    def test_adj_close_appears_on_request(self):
        df = make_price(days=10, adj_close=True, adj_factor=0.5)
        assert df["adj_close"].tolist() == [c * 0.5 for c in df["close"]]

    def test_estimated_alias_flag_is_opt_in(self):
        assert "adj_close_is_estimated" not in make_price(days=5, adj_close=True).columns
        flagged = make_price(days=5, adj_close=True, adj_close_is_estimated=True)
        assert flagged["adj_close_is_estimated"].all()


class TestMakeData:
    def test_carries_the_new_keys(self):
        data = make_data()
        assert not data["quarterly"].empty
        assert data["annual_report_sections"]

    def test_new_keys_are_overridable(self):
        data = make_data(
            quarterly={"periods": 6},
            annual_report_sections={"provenance": "fallback"},
        )
        assert len(data["quarterly"]) == 6
        assert data["annual_report_sections"]["2025"]["mdna"]["provenance"] == "fallback"


class TestBuildersMatchTheFetchedSchemas:
    """conftest promises the builders "mirror the real fetched schemas". This
    is that promise, enforced.

    Four defects in this codebase were masked by a builder that did not: a
    `Q0..Q7` quarter label no parser reads (checkpoints fell back to positional
    matching in every test), a `Q1 2021` shareholding label with the same
    effect, a missing `govt_pct` the report generator reads, and a `close`-only
    price frame after all 22 cached tickers had gained `adj_close`. In each
    case the fixture agreed with the code and both disagreed with reality, so
    the tests passed and the pipeline was wrong.

    Pinned against the fetchers' own label maps rather than against files in
    `raw_data/`, which is gitignored and absent on a clean checkout.
    """

    def test_quarterly_matches_the_parser_label_map(self):
        from boundless100x.data_fetcher.fetch_financials import QTR_LABEL_MAP

        expected = {"quarter"} | set(QTR_LABEL_MAP.values())
        assert set(make_quarterly().columns) == expected

    def test_shareholding_matches_the_parser_label_map(self):
        from boundless100x.data_fetcher.fetch_financials import SH_LABEL_MAP

        expected = {"quarter"} | set(SH_LABEL_MAP.values())
        assert set(make_shareholding().columns) == expected

    @pytest.mark.parametrize("builder,label_map_name", [
        (make_financials, "PL_LABEL_MAP"),
        (make_balance_sheet, "BS_LABEL_MAP"),
        (make_cashflow, "CF_LABEL_MAP"),
        (make_ratios, "RATIOS_LABEL_MAP"),
    ])
    def test_annual_frames_match_their_parser_label_maps(self, builder, label_map_name):
        from boundless100x.data_fetcher import fetch_financials

        expected = {"year"} | set(getattr(fetch_financials, label_map_name).values())
        assert set(builder().columns) == expected

    @pytest.mark.parametrize("builder", [make_quarterly, make_shareholding])
    def test_period_labels_are_parseable(self, builder):
        """A label no parser reads silently downgrades period matching."""
        from boundless100x.compute_engine.metrics.builtin._helpers import quarter_index

        labels = list(builder()["quarter"])
        assert all(quarter_index(label) is not None for label in labels)

    def test_period_labels_are_consecutive(self):
        """Gaps must come from a test that wants one, never from the builder."""
        from boundless100x.compute_engine.metrics.builtin._helpers import quarter_index

        indices = [quarter_index(q) for q in make_quarterly(periods=12)["quarter"]]
        assert indices == list(range(indices[0], indices[0] + 12))
