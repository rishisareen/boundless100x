"""U6 — the priced extraction sweep.

The dry run carries most of the weight here. It is the thing standing between
a mistyped flag and a corpus-wide spend, and unlike the live path it can be
tested exhaustively without an API key or a bill.
"""

import json

import pytest

from boundless100x import forward_growth_schema as schema
from boundless100x.llm_layer import sweep as sweep_module
from tests.conftest import AUDIT_COMMITTEE_TEXT, make_ar_sections, make_metadata


class RecordingLLM:
    """An orchestrator stub that records calls and meters usage like the real one."""

    def __init__(self, response=None, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)
        self._response = response or {"years": {}}
        self.forward_growth_model = "claude-sonnet-4-6"
        self.forward_growth_char_budget = 12000
        self._input = 0
        self._output = 0

    def build_forward_growth_prompt(self, ticker, company_name, submission):
        from boundless100x.llm_layer import forward_growth

        return forward_growth.prompt_template().format(
            ticker=ticker, company_name=company_name,
            vocabulary=forward_growth.vocabulary_prompt_block(),
            report_text=forward_growth.render_report_text(submission),
        )

    def run_forward_growth_extraction(self, ticker, company_name, submission):
        self.calls.append(ticker)
        self._input += 5000
        self._output += 500
        if ticker in self.fail_on:
            return {"error": "rate limited"}
        return self._response

    def usage_summary(self):
        from boundless100x.llm_layer.orchestrator import estimate_cost

        return {
            "total_input_tokens": self._input,
            "total_output_tokens": self._output,
            "estimated_cost_usd": round(
                estimate_cost("claude-sonnet-4-6", self._input, self._output), 6
            ),
        }

    def use_deep_models(self):
        pass

    def use_configured_models(self):
        pass


def write_ticker(root, ticker, code, sections=None, provenance="found",
                 with_reports=True):
    directory = root / ticker
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(
        make_metadata(name=f"{ticker} Ltd", bse_code=code)
    ))
    (directory / "financials.csv").write_text("year,revenue\nMar 2025,100\n")

    if not with_reports:
        return directory
    reports = root / code / "annual_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for year, payload in make_ar_sections(
        years=["2025"], provenance=provenance, sections=sections
    ).items():
        (reports / f"{year}_annual_report.sections.json").write_text(
            json.dumps(payload)
        )
    return directory


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A real service with the corpus and the sidecar path pointed at tmp_path."""
    from boundless100x.service import Boundless100xService

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    svc = Boundless100xService()
    svc.suite.raw_data_dir = str(tmp_path / "raw_data")
    svc.history_path = str(tmp_path / "score_history.jsonl")
    svc._llm = RecordingLLM()
    return svc


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL", "532830")                       # genuine MD&A
    write_ticker(root, "VBL", "540180")                          # genuine MD&A
    write_ticker(root, "RAIN", "500339",
                 sections={"mdna": AUDIT_COMMITTEE_TEXT})        # suspect
    write_ticker(root, "IDEA", "532822", with_reports=False)     # no reports
    return root


class TestScopeIsAlwaysExplicit:
    def test_no_list_and_no_all_flag_is_refused(self, service, corpus):
        with pytest.raises(ValueError) as excinfo:
            sweep_module.sweep(service)
        assert "no default" in str(excinfo.value)
        assert service._llm.calls == []

    def test_an_explicit_list_is_honoured_case_insensitively(self, service, corpus):
        report = sweep_module.sweep(service, tickers=["astral"], dry_run=True)
        assert [p["ticker"] for p in report["plans"]] == ["ASTRAL"]

    def test_all_tickers_prices_only_the_extractable_ones(self, service, corpus):
        report = sweep_module.sweep(service, all_tickers=True, dry_run=True)

        priced = [p["ticker"] for p in report["plans"] if not p["skipped"]]
        assert sorted(priced) == ["ASTRAL", "VBL"]
        assert report["estimate"]["tickers"] == 2

    def test_all_tickers_still_reports_what_it_skipped_and_why(
        self, service, corpus
    ):
        """The reasons are computed either way; dropping them is the silent cap.

        This is the case the earlier version of this test missed: it asserted
        only the priced plans, so a report naming 15 tickers to sweep and 0
        skipped — while 7 had just been excluded with reasons in hand — looked
        exactly like success.
        """
        report = sweep_module.sweep(service, all_tickers=True, dry_run=True)

        skipped = {e["ticker"]: e["reason"] for e in report["skipped"]}
        assert sorted(skipped) == ["IDEA", "RAIN"]
        assert "suspect" in skipped["RAIN"]
        assert "no annual-report sections" in skipped["IDEA"]


class TestDryRun:
    def test_a_dry_run_makes_no_call_and_still_prices_each_ticker(
        self, service, corpus
    ):
        report = sweep_module.sweep(
            service, tickers=["ASTRAL", "VBL"], dry_run=True
        )

        assert service._llm.calls == []
        for plan in report["plans"]:
            assert plan["submission_chars"] > 0
            assert plan["estimated_input_tokens"] > 0
            assert plan["estimated_cost_usd"] > 0
        assert report["estimate"]["tickers"] == 2
        assert report["actual"]["usd"] == 0.0

    def test_the_estimate_prices_the_whole_prompt_not_just_the_report_text(
        self, service, corpus
    ):
        """The template and vocabulary are the fixed cost every ticker pays."""
        plan = sweep_module.sweep(service, tickers=["ASTRAL"], dry_run=True)["plans"][0]

        assert plan["prompt_chars"] > plan["submission_chars"]

    def test_a_ticker_with_only_suspect_sections_is_skipped_with_the_reason(
        self, service, corpus
    ):
        report = sweep_module.sweep(service, tickers=["RAIN"], dry_run=True)

        assert report["skipped"] == [
            {"ticker": "RAIN", "reason": report["plans"][0]["skipped"]}
        ]
        assert "suspect" in report["skipped"][0]["reason"]
        assert report["estimate"]["tickers"] == 0
        assert report["estimate"]["usd"] == 0.0

    def test_a_ticker_with_no_reports_is_skipped_rather_than_priced(
        self, service, corpus
    ):
        report = sweep_module.sweep(service, tickers=["IDEA"], dry_run=True)

        assert report["estimate"]["tickers"] == 0
        assert "no annual-report sections" in report["skipped"][0]["reason"]

    def test_a_ticker_absent_from_the_corpus_is_skipped(self, service, corpus):
        report = sweep_module.sweep(service, tickers=["NOSUCH"], dry_run=True)

        assert "no metadata.json" in report["skipped"][0]["reason"]

    def test_the_dry_run_works_without_an_orchestrator(self, service, corpus):
        """Pricing a spend is what one does before configuring it."""
        service._llm = None

        report = sweep_module.sweep(service, tickers=["ASTRAL"], dry_run=True)

        assert report["plans"][0]["estimated_cost_usd"] > 0

    def test_the_reported_provenance_is_the_gated_one(self, service, corpus):
        plan = sweep_module.sweep(service, tickers=["RAIN"], dry_run=True)["plans"][0]

        assert plan["provenance"]["2025"]["mdna"] == schema.SUSPECT
        assert plan["gate_reasons"]["2025"]["mdna"]


class TestPricingIsSharedWithTheLivePath:
    def test_the_dry_run_prices_the_prompt_the_call_would_send(
        self, service, corpus
    ):
        """One assembly, not a reconstruction that happens to agree."""
        from boundless100x.llm_layer import forward_growth

        plan = sweep_module.plan_ticker(service, "ASTRAL")
        sections = sweep_module.load_context(
            service.suite.raw_data_dir, "ASTRAL"
        )["annual_report_sections"]
        submission = forward_growth.plan_submission(
            service.config, sections, llm=service._llm
        )["submission"]

        assert plan["prompt_chars"] == len(
            forward_growth.build_prompt("ASTRAL", "ASTRAL Ltd", submission)
        )

    def test_an_unpriceable_model_warns_that_the_ceiling_cannot_bind(
        self, service, corpus, caplog
    ):
        """A ceiling that meters on $0 enforces nothing; say so."""
        service._llm.forward_growth_model = "some-unreleased-model"
        service.config["llm"]["forward_growth_model"] = "some-unreleased-model"

        with caplog.at_level("WARNING"):
            sweep_module.sweep(service, tickers=["ASTRAL"], cost_ceiling_usd=1.0)

        assert "ceiling cannot bind" in caplog.text


class TestPilotBatch:
    def test_a_limit_runs_a_pilot_and_names_what_was_deferred(
        self, service, corpus
    ):
        report = sweep_module.sweep(
            service, all_tickers=True, dry_run=True, limit=1
        )

        assert report["estimate"]["tickers"] == 1
        assert len(report["deferred"]) == 1


class TestLiveRun:
    def _response(self):
        text = make_ar_sections()["2025"]["mdna"]["text"]
        sentence = "We expect revenue of Rs 1,500 crore in FY2026."
        assert sentence in text
        return {"years": {"2025": {"guidance": [{
            "metric": "revenue", "target_value": 1500, "target_period": "FY2026",
            "subject": schema.SUBJECT_COMPANY, "unit": schema.UNIT_INR_CR,
            "source_sentence": sentence, "section": "mdna",
        }]}}}

    def test_a_live_run_extracts_and_reports_what_was_kept(self, service, corpus):
        service._llm = RecordingLLM(response=self._response())

        report = sweep_module.sweep(service, tickers=["ASTRAL", "VBL"])

        assert service._llm.calls == ["ASTRAL", "VBL"]
        assert all(r["status"] == "ok" for r in report["results"])
        assert all(r["kept"] == 1 for r in report["results"])
        assert report["actual"]["usd"] > 0

    def test_the_live_run_stops_at_the_ceiling_and_names_what_it_missed(
        self, service, corpus
    ):
        service._llm = RecordingLLM(response=self._response())

        report = sweep_module.sweep(
            service, all_tickers=True, cost_ceiling_usd=0.0001
        )

        assert len(report["results"]) == 1
        assert report["not_reached"] == ["VBL"]

    def test_a_per_ticker_failure_does_not_end_the_sweep(self, service, corpus):
        service._llm = RecordingLLM(response=self._response(), fail_on=("ASTRAL",))

        report = sweep_module.sweep(service, tickers=["ASTRAL", "VBL"])

        by_ticker = {r["ticker"]: r for r in report["results"]}
        assert by_ticker["ASTRAL"]["status"] == "failed"
        assert by_ticker["VBL"]["status"] == "ok"

    def test_the_summary_groups_discard_reasons(self, service, corpus):
        """A systematic cause must be visible, not scattered across twenty rows."""
        service._llm = RecordingLLM(response={"years": {"2025": {"guidance": [
            {"metric": "revenue", "target_value": 1500, "target_period": "FY2026",
             "subject": schema.SUBJECT_COMPANY, "unit": schema.UNIT_INR_CR,
             "source_sentence": "A sentence that was never in the filing.",
             "section": "mdna"},
            {"metric": "pat", "target_value": 900, "target_period": "FY2027",
             "subject": schema.SUBJECT_COMPANY, "unit": schema.UNIT_INR_CR,
             "source_sentence": "Another sentence nobody wrote.",
             "section": "mdna"},
        ]}}})

        report = sweep_module.sweep(service, tickers=["ASTRAL"])

        assert report["discard_summary"] == {
            "quoted sentence is not in the submitted text": 2
        }


class TestDiscardGrouping:
    @pytest.mark.parametrize("reason,label", [
        ("source_sentence does not appear in the submitted mdna text for 2025 — x",
         "quoted sentence is not in the submitted text"),
        ("target_value does not appear in its own source_sentence as a inr_cr figure",
         "figure is not denominated as the entry claims"),
        ("target_period does not appear in its own source_sentence",
         "period is not in the quoted sentence"),
        ("missing required field(s): unit", "missing a required field"),
        ("unit 'eur_bn' is outside the closed set (inr_cr, inr, pct)",
         "unit outside the vocabulary"),
        ("guidance subject 'analysts' is outside the closed set (company, market)",
         "subject outside the vocabulary"),
    ])
    def test_known_failure_modes_group_into_one_bucket_each(self, reason, label):
        assert sweep_module.group_discards([{"reason": reason}]) == {label: 1}

    def test_an_unrecognised_reason_survives_as_its_own_bucket(self):
        grouped = sweep_module.group_discards([{"reason": "something new happened"}])
        assert grouped == {"something new happened": 1}

    def test_buckets_are_ordered_most_frequent_first(self):
        grouped = sweep_module.group_discards([
            {"reason": "missing required field(s): unit"},
            {"reason": "missing required field(s): subject"},
            {"reason": "target_period does not appear in its own source_sentence"},
        ])
        assert list(grouped) == [
            "missing a required field", "period is not in the quoted sentence",
        ]
