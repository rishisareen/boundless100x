"""The report's quadrant badge must speak the vocabulary the metric emits.

`compute_qg_quadrant` emits true_wealth_creator / quality_trap / growth_trap /
wealth_destroyer. The report's label map keyed on a different set, so the two
most consequential verdicts rendered as neutral title-case with no description
— a wealth destroyer looked the same as an unknown value.
"""

import inspect

import pytest

from boundless100x.compute_engine.metrics.builtin import composite
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.output.report_generator import ReportGenerator
from tests.conftest import make_result

EMITTED_QUADRANTS = [
    "true_wealth_creator",
    "quality_trap",
    "growth_trap",
    "wealth_destroyer",
]


def build_summary(quadrant: str) -> dict:
    result = make_result(metrics={
        "quality_growth_quadrant": MetricResult(
            value=quadrant,
            metadata={"avg_roce": 22.0, "pat_cagr": 18.0},
        ),
    })
    generator = ReportGenerator()
    return generator._build_executive_summary(result)["quadrant"]


def test_emitted_quadrant_values_are_the_four_expected():
    """Guards the vocabulary itself — if composite.py changes, this test leads."""
    source = inspect.getsource(composite.compute_qg_quadrant)
    for quadrant in EMITTED_QUADRANTS:
        assert f'"{quadrant}"' in source


@pytest.mark.parametrize("quadrant", EMITTED_QUADRANTS)
def test_every_emitted_quadrant_has_a_description(quadrant):
    """The fallback yields an empty description, so a non-empty one proves a hit."""
    assert build_summary(quadrant)["description"] != ""


@pytest.mark.parametrize("quadrant", EMITTED_QUADRANTS)
def test_no_emitted_quadrant_renders_neutral(quadrant):
    """Neutral is the unknown-value fallback; a known verdict must be good or bad."""
    assert build_summary(quadrant)["sentiment"] in {"good", "bad"}


def test_wealth_destroyer_renders_negative():
    badge = build_summary("wealth_destroyer")

    assert badge["sentiment"] == "bad"
    assert "Wealth Destroyer" in badge["label"]


def test_true_wealth_creator_renders_positive():
    badge = build_summary("true_wealth_creator")

    assert badge["sentiment"] == "good"


def test_unknown_quadrant_still_falls_back_safely():
    badge = build_summary("something_new")

    assert badge["sentiment"] == "neutral"
    assert badge["label"] == "Something New"
