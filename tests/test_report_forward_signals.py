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

import ast
import importlib
import inspect
import re
import textwrap

import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.output.report_generator import (
    FLAG_ELEMENT_MAP,
    FLAG_LABELS,
    FORWARD_SIGNALS,
    FORWARD_SIGNALS_DISCLAIMER,
    FORWARD_SIGNALS_ELEMENT,
    ReportGenerator,
)
from tests.conftest import make_result, make_scores


# ── Reading the registry for what a zero-weight metric can flag ──
#
# KTD6's rule is about *every* zero-weight metric, and the test that enforced
# it filtered on two hardcoded id prefixes — which is exactly why Phase 3's
# `institutional_accumulation_streak` shipped with an unregistered flag and a
# green suite. The prefix list was a convention nobody was reminded to extend.
# These helpers ask the engine instead, so a new zero-weight metric is caught
# on the run after it is added rather than on the day somebody remembers.


def _literal_strings(node) -> set[str]:
    """String constants inside a literal list, tuple, conditional or concat.

    Deliberately literal-only. An f-string flag cannot be known statically, and
    guessing at one would put a fabricated id in front of an assertion; the
    zero-weight metrics emit none, and a scored metric that does is not this
    rule's business.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_literal_strings(e) for e in node.elts)) if node.elts else set()
    if isinstance(node, ast.IfExp):
        return _literal_strings(node.body) | _literal_strings(node.orelse)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_strings(node.left) | _literal_strings(node.right)
    return set()


def emitted_flags(func) -> set[str]:
    """Every flag string a compute function can put in its `MetricResult`.

    Read off the source rather than by running the metric, because reaching
    every flag branch would mean a fixture per branch — and a branch nobody
    built a fixture for is precisely the one that ships unregistered.

    Three shapes appear in `builtin/`, and all three are collected: appending
    to a list whose name mentions flags, assigning a literal list to one
    (including `band, flags = "favourable", [...]`, which is how
    `rerating_headroom` does it), and passing one straight to `flags=`.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = node.func
            if (
                isinstance(called, ast.Attribute)
                and called.attr in ("append", "extend")
                and isinstance(called.value, ast.Name)
                and "flag" in called.value.id
            ):
                for arg in node.args:
                    found |= _literal_strings(arg)
            for keyword in node.keywords:
                if keyword.arg and "flag" in keyword.arg:
                    found |= _literal_strings(keyword.value)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and "flag" in target.id:
                    found |= _literal_strings(node.value)
                elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for name, value in zip(target.elts, node.value.elts):
                        if isinstance(name, ast.Name) and "flag" in name.id:
                            found |= _literal_strings(value)

    return found


def zero_weight_metrics() -> dict[str, dict]:
    """Every registered metric the scorer will never give a score to."""
    return {
        metric_id: config
        for metric_id, config in ComputeEngine().metrics.items()
        if float((config.get("scoring") or {}).get("weight") or 0) == 0
    }


def metric_function(config: dict):
    module = importlib.import_module(
        f"boundless100x.compute_engine.metrics.{config['module']}"
    )
    return getattr(module, config["function"])


def zero_weight_flags() -> dict[str, set[str]]:
    """`{metric_id: flags it can emit}` for every zero-weight metric."""
    return {
        metric_id: emitted_flags(metric_function(config))
        for metric_id, config in zero_weight_metrics().items()
    }


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
        """KTD6: FLAG_ELEMENT_MAP falls back to 'composite' for anything unmapped.

        The membership test is the registry's own answer rather than an id
        prefix — a prefix filter passes silently the moment a zero-weight
        metric is added whose flags do not match it.
        """
        flags = generator._collect_flags(signal_metrics())
        emitted = set().union(*zero_weight_flags().values())
        zero_weight = [f for f in flags if f["raw"] in emitted]

        assert zero_weight
        assert all(f["element"] == FORWARD_SIGNALS_ELEMENT for f in zero_weight)

    def test_every_zero_weight_metrics_flags_are_registered_in_both_maps(self):
        """KTD6, asked of the registry rather than of a naming convention.

        `FLAG_ELEMENT_MAP` falls back to `"composite"` and `FLAG_LABELS` falls
        back to a title-cased guess, so an unregistered flag does not fail —
        it renders as an SQGLP signal, with an invented label, on a ticker
        whose score did not move. Nothing about that looks wrong in a report.

        This is therefore the one place the rule can be made mechanical: every
        metric the scorer will never score, every flag its implementation can
        emit, present as a key in both maps. Phase 3's
        `institutional_accumulation_streak` is why — it shipped unregistered
        under a test that filtered on two hardcoded Phase 2 id prefixes.
        """
        unregistered = {
            metric_id: sorted(
                flag for flag in flags
                if flag not in FLAG_ELEMENT_MAP or flag not in FLAG_LABELS
            )
            for metric_id, flags in zero_weight_flags().items()
        }
        offenders = {mid: flags for mid, flags in unregistered.items() if flags}

        assert offenders == {}, (
            f"zero-weight metrics emit flags registered in neither "
            f"FLAG_ELEMENT_MAP nor FLAG_LABELS: {offenders}"
        )

    def test_no_zero_weight_flag_resolves_to_an_sqglp_element(self):
        """Registered is not enough — registered *where* is the rule.

        `composite` is a legitimate destination for a scored metric's flag and
        is also the fallback, so a zero-weight flag mapped there would be
        indistinguishable from one nobody mapped at all.
        """
        for metric_id, flags in zero_weight_flags().items():
            for flag in flags:
                assert FLAG_ELEMENT_MAP[flag] == FORWARD_SIGNALS_ELEMENT, (
                    f"{metric_id} emits {flag!r}, which renders under "
                    f"{FLAG_ELEMENT_MAP[flag]!r}"
                )

    def test_the_scanner_finds_the_flags_it_is_known_to_find(self):
        """The assertions above are only worth their green if this is not blind.

        A source scanner that quietly stops matching would make an empty
        offenders dict mean "nothing to check" rather than "nothing wrong", so
        the shapes actually used in `builtin/` are pinned here: an append loop,
        a tuple assignment, and a conditional passed straight to `flags=`.
        """
        found = zero_weight_flags()

        assert found["rerating_headroom"] == {
            "rerating_headroom_favourable", "rerating_headroom_stretched",
        }
        assert found["quarterly_momentum"] == {
            "quarterly_growth_accelerating", "quarterly_growth_decelerating",
        }
        assert found["tam_runway"] == {"tam_from_superseded_report"}
        assert found["institutional_accumulation_streak"] == {
            "institutional_accumulation_rising"
        }

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
