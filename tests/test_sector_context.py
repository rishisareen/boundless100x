"""Sector tailwind classification, scoring, and prompt context.

sector_context.yaml encoded the Dec 2025 study's sector findings but no code
read it, so two of the study's strongest empirical results influenced nothing.
"""

import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.sector import classify_sector, load_sector_context
from boundless100x.compute_engine.metrics.builtin.longevity import compute_sector_tailwind
from boundless100x.llm_layer.checklist import build_sector_context
from tests.conftest import make_data


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
