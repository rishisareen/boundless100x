"""Sector tailwind classification, scoring, prompt context, and applicability.

sector_context.yaml encoded the Dec 2025 study's sector findings but no code
read it, so two of the study's strongest empirical results influenced nothing.

sector_applicability.yaml answers the other sector question — not "is this a
good sector?" but "does this metric measure anything for a company of this
kind?". It exists because PFC, a lender, was scored at 0% on asset turnover,
equity multiplier and free cash flow yield for doing exactly what a lender
does, and nothing in the report said so.
"""

import json
from pathlib import Path

import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.sector import (
    APPLIES,
    DOES_NOT_APPLY,
    INDETERMINATE,
    SectorApplicability,
    classify_sector,
    load_sector_applicability,
    load_sector_context,
    validate_sector_applicability,
)
from boundless100x.compute_engine.metrics.builtin.longevity import compute_sector_tailwind
from boundless100x.llm_layer.checklist import build_sector_context
from tests.conftest import make_data

RAW_DATA_DIR = (
    Path(__file__).parent.parent / "boundless100x" / "data_fetcher" / "raw_data"
)

# Every metric the shipped table withdraws from a lender's score, by registry
# id. This started as the five the original plan named — all cash-flow-chain
# and DuPont readings that INVERT for a lender. It grew when the table started
# reaching the scorer rather than only the report: at that point an entry
# stopped being a footnote beside a number and became the difference between a
# metric counting and not counting, and eight more readings that were merely
# annotated as meaningless had to actually stop being scored.
LENDER_EXCLUSIONS = {
    # The original five — the free-cash-flow chain and the two DuPont terms
    # whose denominators are a lender's product rather than its plant.
    "dupont_turnover",
    "dupont_equity_multiplier",
    "fcf_yield",
    "fcf_consistency",
    "dcf_margin_of_safety",
    # Same broken chain, reached by two more routes.
    "reverse_dcf_growth",
    "ev_ebitda",
    "cash_conversion",
    # Capital deployed by a lender IS the loan book, which none of these look
    # at — so a company funding record lending reads as returning capital.
    "capital_reinvestment_rate",
    "reinvestment_rate",
    "working_capital_days_trend",
    # Calibrated for a manufacturer, where 4x leverage and 1.2x interest cover
    # mean distress. For an NBFC they are the business model. The values and
    # their flags still render and still reach the model; only the mark out of
    # ten is withdrawn.
    "debt_equity",
    "interest_coverage",
}


@pytest.fixture(scope="module")
def registry_metric_ids():
    return set(ComputeEngine().metrics)


@pytest.fixture(scope="module")
def applicability(registry_metric_ids):
    """The shipped table, validated against the real registry.

    Construction is where validation runs, so this fixture existing at all is
    the assertion that the shipped table names no metric the engine does not
    define.
    """
    return SectorApplicability(registry_metric_ids)


class TestClassification:
    @pytest.mark.parametrize("sector,expected", [
        ("Capital Market", "strong_tailwind"),
        ("Banks - Private Sector", "strong_tailwind"),
        ("Healthcare", "strong_tailwind"),
        ("IT", "moderate_tailwind"),
        ("Pharma", "moderate_tailwind"),
        ("Oil & Gas", "non_consideration"),
        ("Sugar", "non_consideration"),
    ])
    def test_known_sectors_map_to_their_study_bucket(self, sector, expected):
        assert classify_sector(sector) == expected

    def test_unlisted_sector_is_unknown(self):
        assert classify_sector("Interstellar Freight") == "unknown"

    def test_missing_sector_is_unknown(self):
        assert classify_sector(None) == "unknown"
        assert classify_sector("") == "unknown"

    def test_matching_ignores_case_and_spacing(self):
        assert classify_sector("  capital   market ") == "strong_tailwind"

    def test_short_codes_do_not_match_inside_unrelated_words(self):
        """'IT' must not match the letters in 'Securities'."""
        assert classify_sector("Securities Broking") != "moderate_tailwind"

    def test_context_file_loads_all_three_buckets(self):
        context = load_sector_context()

        assert context["strong_tailwind"]
        assert context["moderate_tailwind"]
        assert context["non_consideration"]


class TestMetric:
    def test_strong_tailwind_sector_scores_the_top_category(self):
        data = make_data()
        data["metadata"]["sector"] = "Capital Market"

        result = compute_sector_tailwind(data, {})

        assert result.ok
        assert result.value == "strong_tailwind"

    def test_non_consideration_sector_is_flagged(self):
        data = make_data()
        data["metadata"]["sector"] = "Sugar"

        result = compute_sector_tailwind(data, {})

        assert result.value == "non_consideration"
        assert "sector_non_consideration" in result.flags

    def test_missing_sector_degrades_to_unknown_without_error(self):
        """No cached ticker currently carries a sector, so this is the common path."""
        data = make_data()
        data["metadata"].pop("sector", None)

        result = compute_sector_tailwind(data, {})

        assert result.ok
        assert result.value == "unknown"

    def test_metric_is_registered_and_scored_under_longevity(self):
        engine = ComputeEngine()

        assert engine.metrics["sector_tailwind"]["element"] == "longevity"
        assert engine.metrics["sector_tailwind"]["scoring"]["weight"] > 0

    def test_engine_runs_the_metric(self):
        results = ComputeEngine().run_all(make_data())

        assert results["sector_tailwind"].ok


class TestPromptContext:
    def test_context_names_the_classification(self):
        text = build_sector_context({"sector": "Capital Market"})

        assert "Capital Market" in text
        assert "strong" in text.lower()

    def test_context_carries_the_study_findings(self):
        text = build_sector_context({"sector": "IT"})

        assert "B2C" in text
        assert "leader" in text.lower()

    def test_missing_sector_produces_usable_text_not_a_crash(self):
        text = build_sector_context({})

        assert text
        assert "unknown" in text.lower() or "not available" in text.lower()


class TestApplicabilityVerdicts:
    """The three-valued answer, and which absences produce which value."""

    def test_a_lender_reports_asset_turnover_as_not_applicable(self, applicability):
        outcome = applicability.evaluate("dupont_turnover", "Finance")

        assert outcome["verdict"] == DOES_NOT_APPLY
        assert outcome["applies"] is False
        assert "loan book" in outcome["reason"]

    def test_a_manufacturer_reports_the_same_metric_as_applicable(self, applicability):
        outcome = applicability.evaluate("dupont_turnover", "Industrial Products")

        assert outcome["verdict"] == APPLIES
        assert outcome["applies"] is True

    def test_an_unreviewed_sector_is_indeterminate_not_applicable(self, applicability):
        """The load-bearing case: most sectors are unreviewed."""
        outcome = applicability.evaluate("dupont_turnover", "Interstellar Freight")

        assert outcome["verdict"] == INDETERMINATE
        assert outcome["applies"] is None
        assert "not been reviewed" in outcome["reason"]

    def test_a_cached_sector_nobody_reviewed_is_indeterminate(self, applicability):
        """Capital Markets was deliberately left out, and must not read as fitting."""
        outcome = applicability.evaluate("fcf_yield", "Capital Markets")

        assert outcome["verdict"] == INDETERMINATE
        assert outcome["reason"]

    @pytest.mark.parametrize("sector", [None, "", "   "])
    def test_a_company_with_no_sector_is_indeterminate(self, applicability, sector):
        outcome = applicability.evaluate("dupont_turnover", sector)

        assert outcome["verdict"] == INDETERMINATE
        assert outcome["applies"] is None
        assert "No sector is recorded" in outcome["reason"]

    def test_a_metric_the_registry_does_not_define_is_indeterminate(self, applicability):
        outcome = applicability.evaluate("interstellar_freight_yield", "Finance")

        assert outcome["verdict"] == INDETERMINATE
        assert outcome["applies"] is None

    def test_every_outcome_carries_a_reason(self, applicability):
        """R4: unknown always renders with its reason, never as a blank."""
        cases = [
            ("dupont_turnover", "Finance"),
            ("dupont_turnover", "Industrial Products"),
            ("dupont_turnover", "Interstellar Freight"),
            ("dupont_turnover", None),
            ("no_such_metric", "Finance"),
        ]
        for metric_id, sector in cases:
            assert applicability.evaluate(metric_id, sector)["reason"].strip(), (
                f"{metric_id} / {sector} produced a verdict with no reason"
            )


class TestApplicabilityMatching:
    """Sector names match the way the study lists already match (KTD6)."""

    def test_matching_ignores_case_and_spacing(self, applicability):
        assert applicability.evaluate("fcf_yield", "  finance ")["applies"] is False

    def test_a_narrower_sector_inherits_without_a_new_entry(self, applicability):
        """The whole reason the table keys on a bucket rather than a company."""
        outcome = applicability.evaluate("dupont_equity_multiplier", "Housing Finance")

        assert outcome["verdict"] == DOES_NOT_APPLY
        assert outcome["matched_sectors"] == ["Finance"]

    def test_an_unrelated_sector_does_not_match_on_a_substring(self, applicability):
        assert applicability.matching_sectors("Chemicals & Petrochemicals") == []

    def test_a_more_specific_entry_supplies_the_wording(self, registry_metric_ids):
        """Two keys can both be true of one company; the narrower one is written
        about it, so its sentence is the one the reader gets."""
        table = {
            "Finance": {"not_applicable": {"fcf_yield": "the general reason"}},
            "Housing Finance": {"not_applicable": {"fcf_yield": "the specific reason"}},
        }
        evaluator = SectorApplicability(registry_metric_ids, table)

        outcome = evaluator.evaluate("fcf_yield", "Housing Finance")

        assert outcome["reason"] == "the specific reason"
        assert outcome["matched_sectors"] == ["Finance", "Housing Finance"]

    def test_exclusions_from_both_matching_entries_merge(self, registry_metric_ids):
        table = {
            "Finance": {"not_applicable": {"fcf_yield": "general"}},
            "Housing Finance": {"not_applicable": {"dupont_turnover": "specific"}},
        }
        evaluator = SectorApplicability(registry_metric_ids, table)

        assert set(evaluator.not_applicable_metrics("Housing Finance")) == {
            "fcf_yield",
            "dupont_turnover",
        }


class TestApplicabilityValidation:
    """A rule that can never fire looks exactly like one whose condition is
    never met, so every one of these is a startup error rather than a log line."""

    def test_a_metric_id_the_registry_does_not_define_is_a_startup_error(
        self, registry_metric_ids
    ):
        table = {"Finance": {"not_applicable": {"assett_turnover": "typo'd id"}}}

        with pytest.raises(ValueError, match="unknown metric id"):
            SectorApplicability(registry_metric_ids, table)

    def test_a_blank_reason_is_a_startup_error(self, registry_metric_ids):
        """R7 makes the sentence the deliverable; an exclusion with no reason
        reaches the reader as a shrug."""
        table = {"Finance": {"not_applicable": {"fcf_yield": "   "}}}

        with pytest.raises(ValueError, match="reason a reader can act on"):
            SectorApplicability(registry_metric_ids, table)

    def test_a_misspelled_entry_key_is_a_startup_error(self, registry_metric_ids):
        """`not_applicible:` would otherwise mark the sector reviewed with
        nothing excluded — the wrong answer, reached by a spelling mistake."""
        table = {"Finance": {"not_applicible": {"fcf_yield": "reason"}}}

        with pytest.raises(ValueError, match="unknown key"):
            SectorApplicability(registry_metric_ids, table)

    def test_a_non_mapping_exclusion_block_is_a_startup_error(self, registry_metric_ids):
        table = {"Finance": {"not_applicable": ["fcf_yield"]}}

        with pytest.raises(ValueError, match="must be a mapping"):
            SectorApplicability(registry_metric_ids, table)

    def test_a_reviewed_sector_may_exclude_nothing(self, registry_metric_ids):
        for empty in ({}, None):
            evaluator = SectorApplicability(
                registry_metric_ids, {"Widgets": {"not_applicable": empty}}
            )

            assert evaluator.evaluate("fcf_yield", "Widgets")["applies"] is True

    def test_the_shipped_table_validates_against_the_real_registry(
        self, registry_metric_ids
    ):
        errors = validate_sector_applicability(
            load_sector_applicability(), registry_metric_ids
        )

        assert errors == []

    def test_an_unreadable_table_degrades_to_indeterminate_not_to_applies(
        self, registry_metric_ids, tmp_path
    ):
        """The safe direction: a lost table costs a signal, never asserts a fit."""
        missing = tmp_path / "absent.yaml"
        evaluator = SectorApplicability(registry_metric_ids, load_sector_applicability(str(missing)))

        assert evaluator.evaluate("dupont_turnover", "Finance")["applies"] is None


class TestShippedApplicabilityTable:
    def test_finance_excludes_exactly_the_declared_lender_metrics(self, applicability):
        assert set(applicability.not_applicable_metrics("Finance")) == LENDER_EXCLUSIONS

    def test_no_lender_exclusion_empties_an_element(self, applicability):
        """The exclusions must not amount to refusing to judge a lender.

        Every excluded metric leaves both sides of the coverage ratio, so an
        element that lost all of its weight would score `None` and drop out of
        the composite entirely — a company silently judged on five elements
        while its peers were judged on six. Quality — Business and Longevity
        are the two the lender exclusions cut deepest, and the replacements
        added beside them (`roa_5yr_avg`, `roe_consistency`, `price_to_book`,
        `book_value_cagr_5yr`) exist to keep this true.
        """
        metrics = ComputeEngine().metrics
        excluded = set(applicability.not_applicable_metrics("Finance"))

        remaining: dict[str, float] = {}
        for metric_id, config in metrics.items():
            weight = (config.get("scoring") or {}).get("weight", 0) or 0
            if weight <= 0 or metric_id in excluded:
                continue
            remaining[config["element"]] = remaining.get(config["element"], 0) + weight

        for element in ("quality_business", "longevity", "price", "growth",
                        "size", "quality_management"):
            assert remaining.get(element, 0) >= 0.5, (
                f"{element} keeps only {remaining.get(element, 0):.2f} of its "
                f"declared weight for a lender"
            )

    def test_the_three_cached_lenders_resolve_all_five(self, applicability):
        """Read off the real cached metadata, not a fixture — the plan's
        verification is that these three tickers actually resolve."""
        if not RAW_DATA_DIR.is_dir():
            pytest.skip("no cached corpus on this machine")

        lenders = [
            d.name
            for d in sorted(RAW_DATA_DIR.iterdir())
            if (d / "metadata.json").exists()
            and json.loads((d / "metadata.json").read_text()).get("sector") == "Finance"
        ]

        assert lenders == ["EDELWEISS", "JIOFIN", "PFC"]
        for ticker in lenders:
            sector = json.loads(
                (RAW_DATA_DIR / ticker / "metadata.json").read_text()
            )["sector"]
            for metric_id in LENDER_EXCLUSIONS:
                outcome = applicability.evaluate(metric_id, sector)
                assert outcome["verdict"] == DOES_NOT_APPLY, f"{ticker}/{metric_id}"
                assert outcome["reason"].strip()

    def test_every_declared_reason_is_prose_a_reader_could_use(self, applicability):
        """Not a length check for its own sake: 'n/a' passes a non-blank test
        and fails a reader."""
        for sector in applicability.table:
            for metric_id, reason in applicability.not_applicable_metrics(sector).items():
                assert len(reason.split()) >= 10, f"{sector}.{metric_id}: {reason!r}"

    def test_industrial_products_is_reviewed_and_excludes_nothing(self, applicability):
        assert "Industrial Products" in applicability.table
        assert applicability.not_applicable_metrics("Industrial Products") == {}
