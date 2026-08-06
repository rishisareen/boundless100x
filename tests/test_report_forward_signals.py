"""The Forward Signals report section.

Zero weight means these metrics never receive a score, so without a declared
direction and band the section would ship bare numbers rather than signal — a
reader cannot tell whether +40 is good news without recomputing the metric.
R8 is therefore about *presentation being load-bearing*, and that is what most
of this file asserts.

The other half is separation. The SQGLP drilldown skips `weight == 0` metrics
and reads display names from a hardcoded map, so reusing it would either need a
faked weight or would silently drop every one of these. It stays untouched, and
the new section says in as many words that nothing in it moved the composite.
"""

import re

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.output.report_generator import (
    FORWARD_SIGNALS,
    FORWARD_SIGNALS_DISCLAIMER,
    FORWARD_SIGNALS_ELEMENT,
    ReportGenerator,
)
from tests.conftest import make_result, make_scores


def signal_metrics(**overrides) -> dict:
    metrics = {
        "pe_ttm": MetricResult(value=30.0),
        "roce_avg": MetricResult(value=22.0),
        "market_cap": MetricResult(value=5000.0),
        "rerating_headroom": MetricResult(
            value=42.0,
            flags=["rerating_headroom_favourable"],
            metadata={"band": "favourable", "justified_multiple": 43.0,
                      "current_multiple": 30.3},
        ),
        "promises_kept_ratio": MetricResult(
            value=75.0, metadata={"kept": 3, "due": 4}
        ),
        "capex_pipeline": MetricResult(value=32.0, metadata={"announced_inr_cr": 900.0}),
        "tam_runway": MetricResult(value=12.0, metadata={"tam_inr_cr": 40000.0}),
        "quarterly_momentum": MetricResult(
            value=-5.4, flags=["quarterly_growth_decelerating"],
            metadata={"yoy_pct": [30.0, 20.0, 12.0]},
        ),
    }
    metrics.update(overrides)
    return metrics


def momentum(status="ok", **overrides):
    reading = {
        "ticker": "TEST",
        "status": status,
        "reason": "",
        "latest": {
            "from_date": "2026-01-01", "to_date": "2026-04-01",
            "interval_days": 90, "span": "90 days",
            "composite_from": 6.0, "composite_to": 6.6, "composite_delta": 0.6,
            "element_deltas": {"growth": 0.4, "price": -0.2},
            "synthetic": False, "config_hash": "abc",
        },
        "regimes": [],
    }
    reading.update(overrides)
    return reading


def result_with(metrics=None, momentum_reading=None):
    result = make_result(metrics=metrics if metrics is not None else signal_metrics())
    result.scores = make_scores()
    result.scores["details"] = {
        "roce_5yr_avg": {"value": 22.0, "score": 0.8, "weight": 0.15, "flags": []},
        "rerating_headroom": {"value": 42.0, "score": None, "weight": 0, "flags": []},
    }
    result.momentum = momentum_reading
    return result


@pytest.fixture
def generator():
    return ReportGenerator()


class TestTheBuilder:
    def test_every_available_signal_is_built(self, generator):
        section = generator._build_forward_signals(result_with())
        assert {s["id"] for s in section["signals"]} == set(FORWARD_SIGNALS)

    def test_each_signal_carries_its_direction_of_goodness(self, generator):
        for signal in generator._build_forward_signals(result_with())["signals"]:
            assert signal["direction"]

    def test_each_signal_carries_an_interpretation_band(self, generator):
        """R8: a bare number is not signal without one."""
        for signal in generator._build_forward_signals(result_with())["signals"]:
            assert signal["available"] is True
            assert signal["band"]

    def test_a_metrics_own_declared_band_wins_over_the_report_default(self, generator):
        """Headroom's bands are owner-editable in YAML params; honour them."""
        metrics = signal_metrics(rerating_headroom=MetricResult(
            value=42.0, metadata={"band": "stretched"}
        ))
        section = generator._build_forward_signals(result_with(metrics))
        headroom = next(s for s in section["signals"] if s["id"] == "rerating_headroom")

        assert headroom["band"] == "stretched"

    def test_bands_respond_to_the_value(self, generator):
        def band_at(value):
            metrics = signal_metrics(
                quarterly_momentum=MetricResult(value=value, metadata={})
            )
            section = generator._build_forward_signals(result_with(metrics))
            return next(
                s for s in section["signals"] if s["id"] == "quarterly_momentum"
            )["band"]

        assert band_at(8.0) != band_at(0.0) != band_at(-8.0)

    def test_an_indeterminate_signal_renders_as_unknown_with_its_reason(self, generator):
        metrics = signal_metrics(
            promises_kept_ratio=MetricResult(error="guidance from 1 report year(s)")
        )
        section = generator._build_forward_signals(result_with(metrics))
        promises = next(s for s in section["signals"] if s["id"] == "promises_kept_ratio")

        assert promises["available"] is False
        assert "1 report year" in promises["reason"]
        assert promises["formatted"] == "—"

    def test_an_absent_metric_is_omitted_rather_than_faked(self, generator):
        metrics = signal_metrics()
        del metrics["tam_runway"]
        section = generator._build_forward_signals(result_with(metrics))

        assert "tam_runway" not in {s["id"] for s in section["signals"]}

    def test_the_section_states_it_does_not_touch_the_composite(self, generator):
        note = generator._build_forward_signals(result_with())["disclaimer"]
        assert "composite" in note.lower()

    def test_momentum_comes_from_the_result_not_from_score_history(self, generator):
        section = generator._build_forward_signals(result_with(momentum_reading=momentum()))

        assert section["momentum"]["available"] is True
        assert section["momentum"]["composite_delta"] == 0.6
        assert section["momentum"]["span"] == "90 days"

    def test_insufficient_history_is_not_a_zero_delta(self, generator):
        reading = momentum(
            status="insufficient_history", latest=None,
            reason="no scored runs recorded yet",
        )
        section = generator._build_forward_signals(result_with(momentum_reading=reading))

        assert section["momentum"]["available"] is False
        assert section["momentum"]["composite_delta"] is None
        assert "not enough history yet" in section["momentum"]["label"].lower()

    def test_a_result_predating_the_phase_builds_no_section(self, generator):
        result = make_result(metrics={"pe_ttm": MetricResult(value=30.0)})
        result.momentum = None

        assert generator._build_forward_signals(result) == {}


class TestRendering:
    def rendered(self, generator, result):
        section = generator._build_forward_signals(result)
        drilldown = generator._build_score_drilldown(result)
        summary = generator._build_executive_summary(result)
        html = generator._render_html(
            result, {}, executive_summary=summary,
            score_drilldown=drilldown, forward_signals=section,
        )
        md = generator._render_markdown(
            result, executive_summary=summary,
            score_drilldown=drilldown, forward_signals=section,
        )
        return html, md

    def test_all_signals_render_in_both_formats(self, generator):
        html, md = self.rendered(generator, result_with(momentum_reading=momentum()))

        for config in FORWARD_SIGNALS.values():
            assert config["name"] in html
            assert config["name"] in md

    def test_each_rendered_signal_shows_its_band_and_direction(self, generator):
        html, md = self.rendered(generator, result_with())

        for output in (html, md):
            assert "favourable" in output
            assert "higher is better" in output

    def test_an_indeterminate_signal_renders_as_unknown_not_zero_or_blank(self, generator):
        metrics = signal_metrics(
            tam_runway=MetricResult(error="No numeric addressable-market figure")
        )
        html, md = self.rendered(generator, result_with(metrics))

        for output in (html, md):
            assert "No numeric addressable-market figure" in output

    def test_the_disclaimer_renders(self, generator):
        html, md = self.rendered(generator, result_with())

        for output in (html, md):
            assert "do not contribute" in output.lower()

    def test_insufficient_history_renders_as_such(self, generator):
        reading = momentum(status="insufficient_history", latest=None,
                           reason="no scored runs recorded yet")
        html, md = self.rendered(generator, result_with(momentum_reading=reading))

        for output in (html, md):
            assert "not enough history yet" in output.lower()
            assert "no scored runs recorded yet" in output

    def test_a_result_with_no_forward_signals_renders_without_the_section(self, generator):
        """A ticker analysed before this phase. No heading, and no exception."""
        result = make_result(metrics={"pe_ttm": MetricResult(value=30.0)})
        result.momentum = None
        html, md = self.rendered(generator, result)

        assert "<h2>Forward Signals</h2>" not in html
        assert "## Forward Signals" not in md
        assert FORWARD_SIGNALS_DISCLAIMER not in html

    def test_the_section_heading_renders_when_there_are_signals(self, generator):
        html, md = self.rendered(generator, result_with())

        assert "<h2>Forward Signals</h2>" in html
        assert "## Forward Signals" in md


class TestSeparationFromTheSqglpDrilldown:
    def test_the_drilldown_is_unchanged_by_the_new_metrics(self, generator):
        """Zero-weight metrics were always skipped there, and still are."""
        drilldown = generator._build_score_drilldown(result_with())
        rendered = {entry["name"] for rows in drilldown.values() for entry in rows}

        for config in FORWARD_SIGNALS.values():
            assert config["name"] not in rendered

    def test_no_forward_signal_flag_is_attributed_to_an_sqglp_element(self, generator):
        """KTD6: FLAG_ELEMENT_MAP falls back to 'composite' for anything unmapped."""
        flags = generator._collect_flags(signal_metrics())
        phase_two = [
            f for f in flags
            if f["raw"].startswith(("rerating_headroom_", "quarterly_growth_"))
        ]

        assert phase_two
        assert all(f["element"] == FORWARD_SIGNALS_ELEMENT for f in phase_two)

    def test_a_forward_signal_flag_does_not_render_under_an_element_heading(self, generator):
        result = result_with()
        flags = generator._collect_flags(result.metrics)
        md = generator._render_markdown(
            result,
            executive_summary=generator._build_executive_summary(result),
            score_drilldown=generator._build_score_drilldown(result),
            element_summaries=generator._build_element_summaries(
                result, generator._build_score_drilldown(result), flags
            ),
            flags_precomputed=flags,
            forward_signals=generator._build_forward_signals(result),
        )
        # The per-element "Signals:" lines are built from flags whose element
        # is an SQGLP key; a forward-signal flag must not appear in one.
        for line in re.findall(r"^\*\*Signals:\*\*.*$", md, flags=re.M):
            assert "Re-rating Headroom" not in line
            assert "Quarterly Growth" not in line
