"""The CLI as R14's third surface, and R12/R15 on a line that has to fit.

No test covered `_print_scores`, `_print_coverage`, `_print_eligibility` or the
metric listing before this file, which is how the CLI came to keep its own
element-label map spelling three of the six elements differently from the
report's, print nineteen raw metric ids under `unscored:`, and pass
`service.py`'s `f"...failed: {e}"` warnings through to the reader verbatim.

Three properties carry the file, and each is a property rather than an example
for `test_report_components.py`'s reason — every one of these was an *omission*,
and a test that checks the strings this unit happens to produce passes on the
day the next label is forgotten.

**Registration is the R14 mechanism, and it is asserted here because this is
where it lands.** `test_report_components.py`'s
`test_every_surface_that_has_landed_renders_every_member` was vacuous for
`console` while nothing registered one; importing `cli_common` fills the slot.
Asserted here as well as there because a full-suite run happens to import this
module and a `pytest tests/test_report_components.py` run does not — the
guarantee should not depend on collection order.

**R12 has to survive the console's one liberty.** A terminal cannot carry three
sentences of interpretation and the surface clips them, so the test that matters
is not that a row has a unit but that the clip *cannot reach* the unit or the
direction: the console's figure and direction are compared byte-for-byte against
what the two document surfaces render from the same component.

**The vocabulary is compared against the report's own tables, never spelled.**
An assertion that hardcodes "Quality — Business" keeps passing after
`ELEMENT_CONFIG` stops saying it, which is the failure mode — two surfaces
drifting apart — restated as a test.
"""

import pytest

from boundless100x import cli, cli_common, cli_lifecycle
from boundless100x.cli_common import (
    ConsoleComponents,
    READING_BUDGET,
    clip,
    metric_row_line,
)
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.states import LANES, STATES
from boundless100x.output import report_components as rc
from boundless100x.output.report_reading import Quantity, Reading, read_metric
from boundless100x.output.report_surfaces import metric_cells
from boundless100x.output.report_vocabulary import (
    ACTION_LABELS,
    ELEMENT_CONFIG,
    LANE_LABELS,
    LANE_SHORT_LABELS,
    METRIC_DISPLAY_NAMES,
    STATE_LABELS,
)
from tests.conftest import make_result, make_scores


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine():
    return ComputeEngine()


class StubService:
    """Everything the display helpers ask a service for, and nothing else.

    A real `Boundless100xService` would fetch, and these tests are about what
    is printed. The registry is real, though — the labels, units, directions
    and bands under test are *its* declarations, and a hand-built config would
    make this a test of the fixture.
    """

    def __init__(self, engine, result):
        self.engine = engine
        self._result = result

    def get_element_summary(self, result):
        elements = (result.scores or {}).get("elements", {})
        weights = self.engine.element_weights
        summary = {
            element: {
                "score": round(score, 1) if score is not None else None,
                "weight": f"{weights.get(element, 0) * 100:.0f}%",
            }
            for element, score in elements.items()
        }
        summary["composite"] = (result.scores or {}).get("composite")
        return summary


def printed(render) -> str:
    """Everything a helper prints, markup intact and unwrapped.

    `console.print` is replaced rather than its output captured, following
    `tests/test_action_guard_integration.py`'s `printed()`: what comes back is
    the string the code composed rather than what Rich made of it, so no
    terminal width can defeat a substring assertion and the markup itself is
    assertable.

    Both CLI modules are patched. Each imported `console` by name and holds its
    own binding, so patching one leaves the other writing to the original — the
    trap `cli_common`'s docstring names.
    """
    captured: list[str] = []

    def record(*args, **kwargs):
        captured.append("" if not args else str(args[0]))

    originals = [(module, module.console) for module in (cli, cli_common, cli_lifecycle)]
    fake = type("Recorder", (), {"print": staticmethod(record)})()
    try:
        for module, _ in originals:
            module.console = fake
        render()
    finally:
        for module, original in originals:
            module.console = original
    return "\n".join(captured)


def rendered(render) -> str:
    """What a helper puts on screen, for a helper that prints a Rich `Table`.

    `printed()` cannot be used for those: the recorder holds the `Table` object
    and `str()` on one is its repr, so every assertion about a cell would pass
    vacuously. This renders through a very wide console instead — a wrapped or
    ellipsised cell fails a substring assertion for a reason that has nothing
    to do with what was put in it — and `export_text` strips the markup, which
    is the right trade here because these cases are about words rather than
    about where a style tag sits.
    """
    from rich.console import Console

    wide = Console(width=400, record=True)
    originals = [
        (module, module.console) for module in (cli, cli_common, cli_lifecycle)
    ]
    try:
        for module, _ in originals:
            module.console = wide
        render()
    finally:
        for module, original in originals:
            module.console = original
    return wide.export_text()


@pytest.fixture
def scored():
    """A result with one metric per shape the display has to handle."""
    result = make_result(
        ticker="TESTCO",
        metrics={
            "market_cap": MetricResult(value=138604.0, flags=["large_cap"]),
            "pe_ttm": MetricResult(value=5.32, flags=["cheap_pe"]),
            "roce_5yr_avg": MetricResult(error="Missing input(s): ratios"),
            "owner_operator_signal": MetricResult(value="founder_led_high_holding"),
        },
        scores=make_scores(),
    )
    result.scores["details"] = {"market_cap": {"score": 0.17}}
    result.scores["coverage"] = {
        "composite": 0.589,
        "elements": {"quality_business": 0.32, "size": 1.0},
        "unscored": ["roce_5yr_avg", "cash_conversion", "debt_equity"],
    }
    result.eligibility = {
        "verdict": "not_eligible",
        "gates": {
            "size": {
                "label": "Size headroom", "passed": False,
                "reason": "Size headroom not met: market_cap 138604.00 lt 30000",
            },
            "quiet": {"label": "Incremental returns", "passed": None, "reason": ""},
        },
        "failed": ["size"], "indeterminate": [],
    }
    return result


@pytest.fixture
def service(engine, scored):
    return StubService(engine, scored)


# ── R14: the console is a registered surface ──────────────────────────────


class TestTheConsoleIsTheThirdSurface:
    """`EXPECTED_SURFACES` named three and two had landed. This is the third."""

    def test_all_three_surfaces_are_registered(self):
        assert set(rc.SURFACES) == set(rc.EXPECTED_SURFACES)

    def test_every_registered_surface_renders_every_member(self):
        for name, surface in rc.SURFACES.items():
            assert rc.missing_members(surface) == (), name

    def test_the_console_surface_is_the_one_the_cli_renders_through(self):
        assert rc.SURFACES["console"] is ConsoleComponents

    def test_an_incomplete_console_renderer_cannot_be_defined(self):
        """The decorator's refusal, exercised against the name U11 owns.

        Worth its own case here rather than only in `test_report_components.py`:
        the slot is occupied now, so the interesting question is no longer
        whether registration works but whether occupying it still costs a
        missing handler its class statement.
        """
        with pytest.raises(rc.IncompleteSurface, match="render_caveat"):
            @rc.component_surface("console")
            class Forgot:
                def render_finding(self, component): ...
                def render_metric_row(self, component): ...
                def render_reading(self, component): ...
                def render_disclosure(self, component): ...
                def render_unknown(self, component): ...

        assert rc.SURFACES["console"] is ConsoleComponents


# ── R14: the element labels are the report's ──────────────────────────────


class TestElementLabelsComeFromTheReport:
    """The named defect: `Quality - Business` on the console against
    `Quality — Management` in the report, with nothing keeping them in step."""

    def test_the_cli_no_longer_keeps_an_element_label_map(self):
        """Asserted on the source, because the defect was a *second* table.

        A behavioural test would pass the moment the two tables happened to
        agree, which they did for three of the six elements. What must not
        exist is the second table.
        """
        source = (cli.__file__ and open(cli.__file__, encoding="utf-8").read()) or ""
        assert "element_names" not in source
        # One label from the retired map, chosen because it appears in no other
        # spelling of any element anywhere: the docstrings that explain the
        # defect quote `Quality - Business`, so that one would match itself.
        assert "Size (S)" not in source
        assert "ELEMENT_CONFIG" in source

    def test_every_element_renders_under_the_reports_label(self, scored, service):
        text = rendered(lambda: cli._print_scores(scored, service))

        for config in ELEMENT_CONFIG.values():
            assert str(config["label"]) in text

    def test_no_element_key_reaches_the_console(self, scored, service):
        text = rendered(lambda: cli._print_scores(scored, service))
        for element in ELEMENT_CONFIG:
            assert element not in text


# ── R12: the figure, its unit and its direction ───────────────────────────


class TestNoNumberLosesItsUnitOrDirection:
    def test_a_metric_line_carries_the_figure_with_its_unit(self, engine):
        row = rc.metric_row(
            "market_cap",
            read_metric("market_cap", engine.metrics["market_cap"],
                        MetricResult(value=138604.0)),
            rc.Vocabulary(engine.metrics),
        )
        _name, reading, _contribution = metric_row_line(row)

        assert "₹" in reading and "Cr" in reading
        assert "higher is better" in reading or "lower is better" in reading

    def test_the_console_reads_a_metric_the_same_way_the_documents_do(self, engine):
        """R14 where the console is allowed to differ: only in the middle.

        The figure and the direction are the two things R12 names, so they are
        the two the clip must not be able to reach — compared byte-for-byte
        against `metric_cells`, which is what both document surfaces render.
        """
        row = rc.metric_row(
            "pe_ttm",
            read_metric("pe_ttm", engine.metrics["pe_ttm"],
                        MetricResult(value=5.32)),
            rc.Vocabulary(engine.metrics),
        )
        name, document, contribution = metric_cells(row)
        console_name, console_reading, console_contribution = metric_row_line(row)

        assert console_name == name
        assert console_contribution == contribution
        assert console_reading.startswith(row.value)
        assert document.startswith(row.value)
        assert console_reading.endswith(f"({row.direction})")
        assert document.endswith(f"({row.direction})")

    def test_a_long_reason_is_clipped_and_the_direction_still_survives(self):
        """The tempting shortening is to cut the tail. The tail is R12's half."""
        long_reason = " ".join(["a reason that keeps going"] * 12)
        row = rc.MetricRow(
            label="Something",
            value="0.09x",
            reading=long_reason,
            direction="a middle range is best; both ends are worse",
        )
        _name, reading, _contribution = metric_row_line(row)

        assert len(reading) < len(metric_cells(row)[1])
        assert reading.startswith("0.09x")
        assert reading.endswith("(a middle range is best; both ends are worse)")
        assert "…" in reading

    def test_a_reading_inside_the_budget_is_left_exactly_alone(self):
        assert clip("short enough") == "short enough"

    def test_a_clip_lands_on_a_word_boundary(self):
        text = "x" + " word" * 60
        clipped = clip(text, 20)

        assert clipped.endswith("…")
        assert " …" not in clipped
        assert len(clipped) <= 21

    def test_the_budget_is_stated_rather_than_scattered(self):
        assert clip("y" * (READING_BUDGET + 40)).rstrip("…") == "y" * READING_BUDGET

    def test_a_metric_with_no_figure_says_so_rather_than_showing_a_dash(self, engine):
        row = rc.metric_row(
            "roce_5yr_avg",
            read_metric("roce_5yr_avg", engine.metrics["roce_5yr_avg"],
                        MetricResult(error="Missing input(s): ratios")),
            rc.Vocabulary(engine.metrics),
        )
        _name, reading, _contribution = metric_row_line(row)

        assert "—" in reading
        assert not reading.strip().startswith("—")


# ── R15: no id, enum or exception reaches the console ─────────────────────


class TestNoRawIdentifierReachesTheConsole:
    def test_the_unscored_line_names_metrics_rather_than_listing_ids(
        self, scored, service
    ):
        text = rendered(lambda: cli._print_scores(scored, service))

        assert "roce_5yr_avg" not in text
        assert "cash_conversion" not in text
        assert METRIC_DISPLAY_NAMES["roce_5yr_avg"][1] in text or "RoCE" in text

    def test_an_unnamed_metric_is_counted_rather_than_humanised(
        self, scored, service, caplog
    ):
        """A metric the registry does not define. The old line printed the id;
        auto-humanising it would be the same leak with better typography."""
        scored.scores["coverage"]["unscored"] = ["a_metric_nobody_declared"]

        with caplog.at_level("WARNING"):
            text = rendered(lambda: cli._print_scores(scored, service))

        assert "a_metric_nobody_declared" not in text
        assert "no name for" in text
        assert any("a_metric_nobody_declared" in r.message for r in caplog.records)

    def test_a_gate_reason_renders_its_metric_by_name(self, scored, service):
        text = rendered(lambda: cli._print_scores(scored, service))

        assert "market_cap" not in text
        assert "Market Cap" in text

    def test_a_gate_with_no_reason_falls_back_to_its_label_not_its_id(
        self, scored, service
    ):
        text = rendered(lambda: cli._print_scores(scored, service))

        assert "Incremental returns" in text
        assert "quiet" not in text

    def test_the_action_renders_as_a_label_and_never_as_its_key(self):
        result = make_result(scores=make_scores())
        result.eligibility = {
            "verdict": "eligible", "gates": {}, "failed": [], "indeterminate": [],
        }
        result.llm_analysis = {
            "pass2": {
                "thesis": "A quality compounder.",
                "conviction_level": "high",
                "suggested_action": "strong_buy",
                "target_holding_period": "10yr+",
            },
        }

        text = printed(lambda: cli._print_llm_summary(result))

        assert ACTION_LABELS["strong_buy"] in text
        assert "strong_buy" not in text

    def test_a_run_error_never_reaches_the_console_verbatim(self):
        """`service.py` writes `f"Data fetch failed: {e}"`; the tail is
        `str(exc)` and this block used to print all of it."""
        surface = ConsoleComponents()
        caveat = rc.caveat_from_run_error(
            "Data fetch failed: KeyError: 'borrowings' in /Users/x/raw_data"
        )

        line = surface.render_caveat(caveat)

        assert "Data fetch failed" in line
        assert "KeyError" not in line
        assert "borrowings" not in line
        assert "/Users/x" not in line


# ── R15: the lifecycle surface ────────────────────────────────────────────


class TestWatchlistShowRendersProse:
    def test_every_lane_has_a_label_and_a_short_label(self):
        assert set(LANE_LABELS) == set(LANES)
        assert set(LANE_SHORT_LABELS) == set(LANES)

    def test_a_short_lane_label_opens_its_long_one(self):
        """Two lengths, one name. `ELEMENT_CONFIG` carries `label` beside
        `short` for the same reason; what must not happen is the two coming to
        say different things."""
        for lane, short in LANE_SHORT_LABELS.items():
            assert LANE_LABELS[lane].startswith(short)

    def test_every_lifecycle_state_has_a_label(self):
        assert set(STATE_LABELS) == set(STATES)

    def test_no_state_label_is_its_own_key_wearing_a_hat(self):
        """`probe` → "Probe" is the identifier in title case, which is the
        fallback this whole plan exists to remove rather than the fix for it."""
        for state, label in STATE_LABELS.items():
            assert label.lower() != state.replace("_", " ")

    def test_show_renders_the_lane_and_the_state_as_words(self, tmp_path, monkeypatch):
        from boundless100x import watchlist as watchlist_module

        monkeypatch.setattr(
            watchlist_module, "DEFAULT_WATCHLIST_PATH", tmp_path / "watchlist.json"
        )
        manager = watchlist_module.WatchlistManager()
        manager.add("ASTRAL", lane="rerating")
        manager.transition("ASTRAL", "watch", "seed")

        table = rendered(cli_lifecycle.watchlist_show)

        assert LANE_SHORT_LABELS["rerating"] in table
        assert STATE_LABELS["watch"] in table
        assert "rerating" not in table
        assert "watch" not in table

    def test_the_lane_status_line_names_neither_key(self):
        text = printed(lambda: cli_lifecycle._print_lane_status(
            {"lane": "core", "state": "exit_review", "catalyst": {}, "friction": None}
        ))

        assert LANE_LABELS["core"] in text
        assert STATE_LABELS["exit_review"] in text
        assert "exit_review" not in text

    def test_a_lane_nobody_registered_says_so_rather_than_showing_its_key(self):
        text = printed(lambda: cli_lifecycle._print_lane_status(
            {"lane": "momentum", "state": "screen", "catalyst": {}, "friction": None}
        ))

        assert "momentum" not in text
        assert "no wording for" in text



# ── The metric listing ────────────────────────────────────────────────────


class TestTheMetricListing:
    def test_it_groups_under_the_reports_element_labels(self, scored, service):
        text = rendered(lambda: cli._print_metrics(scored, service))

        assert str(ELEMENT_CONFIG["size"]["label"]) in text
        assert str(ELEMENT_CONFIG["price"]["label"]) in text

    def test_a_named_grade_renders_its_label_and_never_its_enum(
        self, scored, service
    ):
        text = rendered(lambda: cli._print_metrics(scored, service))

        assert "founder_led_high_holding" not in text
        assert "Founder-led" in text

    def test_an_errored_metric_states_the_gap_rather_than_printing_the_error(
        self, scored, service
    ):
        text = rendered(lambda: cli._print_metrics(scored, service))

        assert "ERR:" not in text
        assert "no figure" in text

    def test_a_flag_renders_through_its_registered_label(self, scored, service):
        text = rendered(lambda: cli._print_metrics(scored, service))

        assert "large_cap" not in text
        assert "Large Cap" in text


# ── The score table ───────────────────────────────────────────────────────


class TestTheScoreTable:
    def test_the_composite_agrees_with_its_own_reading(self, scored, service):
        """The rounding rule `section_reading` states, applied to the one line
        no element-shaped builder covers: banding the raw figure while the
        headline rounds it puts `7.0 / 10` beside `middling`."""
        line = cli._composite_reading(6.96)

        assert line.headline.startswith("7.0")
        assert "strong" in line.text

    def test_an_unscorable_composite_reads_unknown_with_a_reason(self):
        line = cli._composite_reading(None)

        assert not line.known
        assert "not the same as a score of zero" in line.unknown.reason

    def test_a_thin_element_states_its_coverage_beside_its_score(
        self, scored, service
    ):
        """R18, where the reader meets the score it qualifies — the per-element
        line this replaced printed `quality_business 32%`, the raw key."""
        text = rendered(lambda: cli._print_scores(scored, service))

        assert "32%" in text
        assert "quality_business" not in text

    def test_an_element_at_full_coverage_recites_no_share(self, scored, service):
        text = rendered(lambda: cli._print_scores(scored, service))

        assert "100% of this element" not in text


# ── The surface itself ────────────────────────────────────────────────────


class TestConsoleComponentsCarryNoUnescapedMarkup:
    """Rich reads `[` as a style tag, so an unescaped one swallows the line.

    Component text cannot contain one — `guard_text` refuses brackets with that
    exact reason — and this is the test that the surface is relying on that
    rather than on nobody having tried.
    """

    def test_a_bracket_cannot_be_put_into_a_component_at_all(self):
        with pytest.raises(rc.ComponentContentError, match="Rich"):
            rc.Caveat(text="a note [with] brackets")

    def test_every_handler_returns_balanced_markup(self):
        surface = ConsoleComponents()
        components = [
            rc.ReadingLine(subject="Size", text="Reads strong.", headline="8.0 / 10"),
            rc.Finding(headline="Cash Cow", text="Strong conversion.", sentiment="good"),
            rc.MetricRow(label="PE TTM", value="5.3x", reading="cheap",
                         direction="lower is better"),
            rc.Unknown(subject="No reading", reason="nothing was computed"),
            rc.Caveat(text="part of this analysis did not complete"),
            rc.Disclosure(title="PE TTM", body="What it measures.", anchor="pe_ttm"),
        ]
        from rich.console import Console

        console = Console(width=200, record=True)
        for component in components:
            console.print(
                getattr(surface, rc.HANDLER_FOR_KIND[component.kind])(component)
            )
        text = console.export_text()

        assert "[" not in text and "]" not in text
        assert "PE TTM" in text and "Cash Cow" in text

    def test_the_disclosure_reference_carries_no_word_of_the_explanation(self):
        surface = ConsoleComponents()
        body = rc.Disclosure(
            title="PE TTM", body="Price divided by trailing earnings.",
            anchor="pe_ttm",
        )

        reference = surface.render_disclosure(body.ref)

        assert "Price divided by" not in reference
        assert "what this measures" in reference


class TestReadingsRenderTheSameWordsAsTheReport:
    def test_an_unknown_reading_renders_its_reason_rather_than_a_blank(self):
        surface = ConsoleComponents()
        line = rc.ReadingLine(
            subject="Longevity",
            unknown=rc.Unknown(subject="No score for Longevity",
                               reason="nothing in this section could be scored"),
        )

        assert "nothing in this section could be scored" in surface.render_reading(line)

    def test_a_quantity_reaches_the_console_with_its_unit(self):
        """`Quantity` is R12 as a constructor invariant; this is the surface
        end of it — the unit is in the text the console prints, not in a field
        a renderer has to remember to reach for."""
        quantity = Quantity(value=25.7, unit="percent", direction="higher_is_better")
        row = rc.MetricRow(label="RoE", value=quantity.text, reading="exceptional",
                           direction=quantity.direction_phrase)

        _name, reading, _contribution = metric_row_line(row)

        assert "25.7%" in reading
        assert "higher is better" in reading

    def test_a_reading_the_layer_could_not_produce_still_renders(self, engine):
        reading = Reading(
            metric_id="pe_ttm", status="value_absent",
            reason="the metric ran and produced no value",
        )
        row = rc.metric_row("pe_ttm", reading, rc.Vocabulary(engine.metrics))

        assert not row.known
        assert "no value" in metric_row_line(row)[1]
