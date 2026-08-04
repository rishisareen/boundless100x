"""Screener eligibility gating.

`hidden_gems_100x` is the 100x preset — it must not surface a candidate the
conjunctive eligibility gates have already vetoed. The additive ratio filters
answer "is this a quality compounder?"; `require_eligibility` makes the
screen also enforce eligibility.py's separate "could this plausibly 100x?"
verdict, so a failed gate excludes a survivor instead of being silently
dropped on the floor.
"""

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.screener import Screener


def hidden_gems_metrics(**overrides) -> dict:
    metrics = {
        "market_cap": MetricResult(value=8_000.0),
        "pe_ttm": MetricResult(value=18.0),
        "institutional_holding": MetricResult(value=5.0),
        "analyst_coverage": MetricResult(value=3.0),
    }
    metrics.update(overrides)
    return metrics


def compounders_metrics(**overrides) -> dict:
    metrics = {
        "roe_5yr_avg": MetricResult(value=15.0),
        "pat_cagr_3yr": MetricResult(value=25.0),
        "trailing_peg": MetricResult(value=1.2),
    }
    metrics.update(overrides)
    return metrics


class TestHiddenGems100xRequiresEligibility:
    def test_eligible_verdict_survives(self):
        screener = Screener()
        universe = {"ASTRAL": hidden_gems_metrics()}
        eligibility = {"ASTRAL": {"verdict": "eligible"}}

        survivors = screener.screen(
            universe=universe, preset="hidden_gems_100x", eligibility=eligibility
        )

        assert [s["ticker"] for s in survivors] == ["ASTRAL"]
        assert survivors[0]["eligibility_verdict"] == "eligible"

    def test_not_eligible_verdict_is_excluded_despite_passing_ratios(self):
        screener = Screener()
        universe = {"ASTRAL": hidden_gems_metrics()}
        eligibility = {"ASTRAL": {"verdict": "not_eligible"}}

        survivors = screener.screen(
            universe=universe, preset="hidden_gems_100x", eligibility=eligibility
        )

        assert survivors == []

    def test_indeterminate_verdict_is_excluded(self):
        screener = Screener()
        universe = {"ASTRAL": hidden_gems_metrics()}
        eligibility = {"ASTRAL": {"verdict": "indeterminate"}}

        survivors = screener.screen(
            universe=universe, preset="hidden_gems_100x", eligibility=eligibility
        )

        assert survivors == []

    def test_missing_eligibility_data_raises_rather_than_silently_skipping_gates(self):
        screener = Screener()
        universe = {"ASTRAL": hidden_gems_metrics()}

        with pytest.raises(ValueError, match="eligibility"):
            screener.screen(universe=universe, preset="hidden_gems_100x")

    def test_mixed_universe_only_eligible_survives(self):
        screener = Screener()
        universe = {
            "ASTRAL": hidden_gems_metrics(),
            "OTHERCO": hidden_gems_metrics(),
        }
        eligibility = {
            "ASTRAL": {"verdict": "eligible"},
            "OTHERCO": {"verdict": "not_eligible"},
        }

        survivors = screener.screen(
            universe=universe, preset="hidden_gems_100x", eligibility=eligibility
        )

        assert [s["ticker"] for s in survivors] == ["ASTRAL"]


class TestCompoundersPresetDoesNotRequireEligibility:
    def test_survives_without_any_eligibility_data(self):
        screener = Screener()
        universe = {"TCS": compounders_metrics()}

        survivors = screener.screen(universe=universe, preset="compounders")

        assert [s["ticker"] for s in survivors] == ["TCS"]
        assert "eligibility_verdict" not in survivors[0]
