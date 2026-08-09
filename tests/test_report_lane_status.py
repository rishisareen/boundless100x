"""The Lane & Friction report section.

Two claims, and the first is the one with teeth.

**Nothing changes for a report with no lane context.** A ticker analysed
outside the watchlist must render exactly what it rendered before this section
existed, which is asserted against a golden captured from the pre-change code
rather than by reading the new template and agreeing with it. The goldens in
`tests/golden/` are normalised for the two things that legitimately differ
between runs — the generation timestamp, and the uuid Plotly stamps on each
chart div. Regenerate them deliberately (never to make a red test green) with:

    from tests.test_report_lane_status import normalise
    normalise(Path(report_dir / "TEST_dashboard.html").read_text())

`pre_lane_section_dashboard.html` has been regenerated twice since it was
captured: once when the reading layer was folded into the dashboard — every
element section gained a one-line reading and a `<details>` around its metric
rows — and again when that reading layer's own presentation was polished (a
duplicated score number, stacked identical headlines, an overflowing table).
Both were deliberate, whole-report changes; the golden's job here is the lane
section, and it still holds it. `pre_lane_section_report.md` has **not** moved
and must not: the reading layer landed in the dashboard and nowhere else, so a
diff there means something leaked.

"Unchanged" is a claim about *untracked* reports and about the rest of a
tracked one. A tracked entry in either lane gains the section — a core entry
shows lane and state, a re-rating entry shows its gates and its catalyst too —
and that is the intended behaviour, not a leak.

**The figures may not be read as money that was made.** Gross and net always
render together (R5), every one of them is labelled a modeled estimate with the
basis it rests on, and the word "realized" appears nowhere in the section: the
holding period runs from a `probe` confirmation rather than a fill and the
prices are market bars, so there is no realised anything to report.

Two details are easy to get backwards and are pinned here. An unavailable
friction reading renders as *unavailable with its reason* and emits no numeric
field at all — a zero return means the position went nowhere, and in a table
that is indistinguishable from a reading nobody could take. And the fast lane's
break-even line states §8.2's rough 6–10 points-per-cycle estimate with the
configured tax and slippage rates listed beside it, but **computes no hurdle**:
a tax rate applies to a gain, not to a number of return points, so a single
figure would be arithmetic these inputs do not support.
"""

import re
from pathlib import Path

import pytest

from boundless100x.lifecycle import friction as friction_module
from boundless100x.output import report_charts, report_generator
from boundless100x.output.report_generator import (
    FRICTION_UNAVAILABLE_LABEL,
    ReportGenerator,
)
from boundless100x.watchlist import RERATING_LANE
from tests.conftest import make_result

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# The six shipped lane gates as a reader meets them.
GATE_LABELS = (
    "Quality floor", "Valuation discount", "Growth intact",
    "Institutional accumulation", "Catalyst identified", "Liquidity floor",
)

_CHART = re.compile(r'(<div class="chart-container">)(.*)')
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")

# What a container normalises to. Two placeholders, not one — see `_placeholder`.
CHART_RENDERED = "[chart]"
CHART_EMPTY = "[empty chart]"

# The opening tag itself, so a test counting containers and the regex matching
# them cannot drift into disagreeing about what one looks like.
CONTAINER = '<div class="chart-container">'


def _placeholder(match: re.Match) -> str:
    """Replace a chart payload, **recording whether there was one**.

    The earlier form rewrote the whole match to `\\1[chart]</div>`, which meant
    the `.*` swallowed the closing tag too: a full Plotly payload and a bare
    `<div class="chart-container"></div>` normalised to the *same* string.
    Every chart builder returns `""` on failure, so a data-contract change that
    made all seven charts silently stop rendering would have left this golden
    green — the one thing a golden is for.

    Still greedy to end of line, because a Plotly payload contains its own
    `</div>` and stopping at the first one would leave the trailing script in
    the comparison, where its fresh uuid differs every run. Emptiness is
    decided by stripping the container's own closing tag off the end and
    looking at what is left.
    """
    body = match.group(2)
    inner = body[:-len("</div>")] if body.endswith("</div>") else body
    marker = CHART_RENDERED if inner.strip() else CHART_EMPTY
    return f"{match.group(1)}{marker}</div>"


def normalise(text: str) -> str:
    """Strip the two things that differ between two identical runs.

    Plotly stamps a fresh uuid on every chart div, so the chart payloads are
    replaced rather than compared — but *whether a chart rendered at all* is
    not a run-to-run difference, and is preserved. A chart appearing,
    disappearing, or quietly failing to build all change the normalised text.
    """
    return _STAMP.sub("<generated>", _CHART.sub(_placeholder, text))


def gate(label, passed, reason):
    return {"label": label, "rationale": "", "passed": passed,
            "reason": reason, "conditions": []}


def lane_gate_result(verdict="not_qualified", failing=("liquidity_floor",)):
    """A lane-gate payload in the shape `LaneGateEvaluator.evaluate` returns."""
    gates = {
        "quality_floor": gate("Quality floor", True, "Quality floor met: score composite 6.50 gte 5.5"),
        "valuation_discount": gate("Valuation discount", True, "Valuation discount met: pe_vs_historical 22.00 lte 50"),
        "growth_intact": gate("Growth intact", True, "Growth intact met: ttm_growth_vs_cagr 4.00 gte 0"),
        "institutional_accumulation": gate("Institutional accumulation", True, "Institutional accumulation met: streak 3.00 gte 2"),
        "catalyst_identified": gate("Catalyst identified", True, "Catalyst identified met: catalyst is active"),
        "liquidity_floor": gate("Liquidity floor", True, "Liquidity floor met: daily_turnover_ratio 0.05 gte 0.02"),
    }
    for gate_id in failing:
        gates[gate_id]["passed"] = False
        gates[gate_id]["reason"] = (
            f"{gates[gate_id]['label']} not met: daily_turnover_ratio 0.01 gte 0.02"
        )
    return {
        "verdict": verdict,
        "qualifies": verdict == "qualifies",
        "gates": gates,
        "failed": [g for g, d in gates.items() if d["passed"] is False],
        "indeterminate": [g for g, d in gates.items() if d["passed"] is None],
    }


def friction_reading(**overrides):
    reading = {
        "available": True,
        "basis": friction_module.BASIS_ESTIMATE,
        "gross_return_pct": 48.0,
        "after_slippage_pct": 47.0,
        "net_return_pct": 41.125,
        "holding_days": 420,
        "tax_regime": "ltcg",
        "tax_pct": 12.5,
        "taxed": True,
        "ltcg_holding_days": 365,
        "slippage_bps": 100,
        "entry_date": "2025-06-10",
        "exit_date": "2026-08-04",
        "price_series": "adj_close",
    }
    reading.update(overrides)
    return reading


def lane_context(lane="rerating", state="probe", **overrides):
    context = {
        "lane": lane,
        "state": state,
        "as_of": "2026-08-07",
        "catalyst": None,
        "lane_gates": lane_gate_result() if lane == "rerating" else None,
        "friction": None,
        "friction_assumptions": friction_module.config_from(None),
    }
    context.update(overrides)
    return context


@pytest.fixture
def generator(tmp_path):
    return ReportGenerator(output_dir=str(tmp_path))


def rendered(generator, result, context):
    """The section as both surfaces show it, built the way `generate` builds it."""
    status = generator._build_lane_status(context)
    summary = generator._build_executive_summary(result)
    html = generator._render_html(
        result, {}, executive_summary=summary, lane_status=status
    )
    md = generator._render_markdown(
        result, executive_summary=summary, lane_status=status
    )
    return html, md


def lane_section(md: str) -> str:
    """The markdown between the section heading and the next one."""
    body = md.split("## Lane & Friction", 1)[1]
    return body.split("\n## ", 1)[0]


class TestTheBuilder:
    def test_no_lane_context_builds_no_section(self, generator):
        assert generator._build_lane_status(None) is None

    def test_lane_and_state_are_always_present(self, generator):
        status = generator._build_lane_status(lane_context(lane="core", state="watch"))

        assert status["lane"] == "core"
        assert status["state"] == "watch"
        assert status["lane_label"]
        assert status["state_label"]

    def test_a_core_context_carries_no_gates_and_no_break_even_line(self, generator):
        status = generator._build_lane_status(lane_context(lane="core", state="watch"))

        assert status["gates"] == []
        assert status["breakeven"] is None

    def test_the_six_gates_render_with_their_pass_fail_state(self, generator):
        status = generator._build_lane_status(lane_context())

        assert [g["label"] for g in status["gates"]] == list(GATE_LABELS)
        assert [g["passed"] for g in status["gates"]] == [True] * 5 + [False]

    def test_a_failing_lane_verdict_is_never_styled_as_a_pass(self, generator):
        status = generator._build_lane_status(lane_context())

        assert status["verdict"] == "not_qualified"
        assert status["sentiment"] == "bad"

    def test_the_lane_verdict_never_borrows_the_100x_vocabulary(self, generator):
        """`not_qualified` is not `not_eligible`; conflating them misleads twice."""
        for verdict in ("qualifies", "not_qualified", "indeterminate"):
            status = generator._build_lane_status(
                lane_context(lane_gates=lane_gate_result(verdict=verdict))
            )
            assert "eligib" not in status["verdict_label"].lower()
            assert "eligib" not in status["description"].lower()


class TestFrictionFigures:
    def test_gross_and_net_are_built_together(self, generator):
        figures = generator._build_lane_status(
            lane_context(friction=friction_reading())
        )["friction"]

        assert figures["gross_return_pct"] == 48.0
        assert figures["net_return_pct"] == 41.125

    def test_the_reading_states_its_basis_and_calls_itself_modeled(self, generator):
        figures = generator._build_lane_status(
            lane_context(friction=friction_reading())
        )["friction"]

        assert figures["basis"] == "estimate"
        assert "modeled" in figures["label"].lower()

    def test_an_unavailable_reading_emits_no_numbers_at_all(self, generator):
        """A zero return means the position went nowhere. This is not that."""
        figures = generator._build_lane_status(lane_context(friction={
            "available": False,
            "reason": "no usable adj_close bars for this position",
            "basis": "estimate",
        }))["friction"]

        assert figures["available"] is False
        assert figures["label"] == FRICTION_UNAVAILABLE_LABEL
        assert "no usable adj_close bars" in figures["reason"]
        assert "gross_return_pct" not in figures
        assert "net_return_pct" not in figures

    def test_no_friction_reading_builds_no_friction_subsection(self, generator):
        assert generator._build_lane_status(lane_context())["friction"] is None

    def test_a_payload_missing_half_the_pair_is_unavailable_not_half_rendered(
        self, generator
    ):
        """R5 has no half: a recorded payload is read back off a JSON store.

        `details` is written by `confirm_exit` today, but it is a stored dict
        like any other — hand-editable, and written by whatever version of this
        system recorded that sale. A gross figure with no net beside it is
        exactly the one-without-the-other R5 forbids, so the reading is refused
        rather than rendered short.
        """
        figures = generator._build_lane_status(lane_context(
            friction=friction_reading(net_return_pct=None)
        ))["friction"]

        assert figures["available"] is False
        assert figures["label"] == FRICTION_UNAVAILABLE_LABEL
        assert "net" in figures["reason"].lower()
        assert "gross_return_pct" not in figures

    def test_a_payload_without_the_intermediate_step_still_shows_the_pair(
        self, generator
    ):
        """Only gross and net are the pair. The slippage row is not."""
        html, md = rendered(generator, make_result(), lane_context(
            friction=friction_reading(after_slippage_pct=None)
        ))

        for output in (html, md):
            body = lane_body(output)
            assert "48.0" in body and "41.1" in body
            assert "After round-trip slippage" not in body

    def test_a_malformed_payload_does_not_take_down_the_report(self, generator):
        html, md = rendered(generator, make_result(), lane_context(
            friction={"available": True, "basis": "recorded"}
        ))

        for output in (html, md):
            assert FRICTION_UNAVAILABLE_LABEL in lane_body(output)


class TestBreakEven:
    def test_the_fast_lane_states_the_roadmap_estimate(self, generator):
        breakeven = generator._build_lane_status(lane_context())["breakeven"]

        assert "6–10" in breakeven["estimate"]
        assert "estimate" in breakeven["statement"].lower()

    def test_the_configured_assumptions_are_listed_beside_it(self, generator):
        context = lane_context(friction_assumptions=friction_module.config_from(
            {"friction": {"stcg_pct": 30.0, "ltcg_pct": 15.0,
                          "ltcg_holding_days": 730, "slippage_bps": 250}}
        ))

        listed = " | ".join(generator._build_lane_status(context)["breakeven"]["assumptions"])

        assert "30.0" in listed and "15.0" in listed
        assert "730" in listed and "250" in listed

    def test_no_hurdle_number_is_computed(self, generator):
        """A rate applied to gains is not a number of return points."""
        breakeven = generator._build_lane_status(lane_context())["breakeven"]

        assert not [k for k in breakeven if "hurdle" in k]
        assert "no hurdle" in breakeven["caveat"].lower()

    def test_a_core_report_never_renders_it(self, generator):
        assert generator._build_lane_status(
            lane_context(lane="core", state="probe")
        )["breakeven"] is None

    def test_the_lane_it_gates_on_is_the_shared_constant(self, generator):
        """One statement of what the fast lane is called.

        `advance.py` and `lane_view.py` both import `RERATING_LANE` from
        `watchlist`; a bare `"rerating"` literal here is a third spelling of the
        same word with nothing keeping it in step, and this gate decides whether
        a whole section renders.
        """
        assert report_generator.RERATING_LANE is RERATING_LANE
        assert generator._build_lane_status(
            lane_context(lane=RERATING_LANE, state="probe")
        )["breakeven"] is not None


class TestRendering:
    def test_the_section_heading_renders_in_both_formats(self, generator):
        html, md = rendered(generator, make_result(), lane_context())

        assert "<h2>Lane &amp; Friction</h2>" in html
        assert "## Lane & Friction" in md

    def test_the_six_gate_labels_and_their_state_render(self, generator):
        html, md = rendered(generator, make_result(), lane_context())

        for label in GATE_LABELS:
            assert label in html
            assert label in md
        for output in (html, md):
            assert "fail" in lane_body(output)

    def test_a_failed_gate_is_not_dressed_as_an_eligible_verdict(self, generator):
        html, md = rendered(generator, make_result(), lane_context())

        for output in (html, md):
            body = lane_body(output)
            assert "eligib" not in body.lower()
            assert "does not qualify" in body.lower()

    def test_a_core_context_renders_lane_and_state_and_nothing_else(self, generator):
        html, md = rendered(
            generator, make_result(), lane_context(lane="core", state="watch")
        )

        for output in (html, md):
            body = lane_body(output)
            assert "watch" in body
            assert "6–10" not in body
            assert FRICTION_UNAVAILABLE_LABEL not in body
            for label in GATE_LABELS:
                assert label not in body

    def test_gross_and_net_render_together_never_net_alone(self, generator):
        html, md = rendered(
            generator, make_result(), lane_context(friction=friction_reading())
        )

        for output in (html, md):
            body = lane_body(output)
            assert "48.0" in body
            assert "41.1" in body
            assert "gross" in body.lower() and "net" in body.lower()

    def test_every_figure_is_labelled_a_modeled_estimate_with_its_basis(self, generator):
        html, md = rendered(
            generator, make_result(), lane_context(friction=friction_reading())
        )

        for output in (html, md):
            body = lane_body(output)
            assert "modeled" in body.lower()
            assert "estimate" in body.lower()

    def test_the_word_realized_appears_nowhere_in_the_section(self, generator):
        html, md = rendered(
            generator, make_result(), lane_context(friction=friction_reading())
        )

        for output in (html, md):
            body = lane_body(output).lower()
            assert "realiz" not in body
            assert "realis" not in body

    def test_an_unavailable_reading_renders_its_reason_and_no_figures(self, generator):
        html, md = rendered(generator, make_result(), lane_context(friction={
            "available": False,
            "reason": "no probe transition to measure a holding period from",
            "basis": "estimate",
        }))

        for output in (html, md):
            body = lane_body(output)
            assert FRICTION_UNAVAILABLE_LABEL in body
            assert "no probe transition" in body
            # Not one numeric field beside an absent reading: a rendered zero
            # is indistinguishable from a measurement of zero.
            assert "Gross return" not in body
            assert "Net return" not in body

    def test_the_break_even_line_renders_for_the_fast_lane_only(self, generator):
        fast_html, fast_md = rendered(generator, make_result(), lane_context())
        core_html, core_md = rendered(
            generator, make_result(), lane_context(lane="core", state="probe")
        )

        for output in (fast_html, fast_md):
            body = lane_body(output)
            assert "6–10" in body
            assert "20.0" in body and "12.5" in body and "100" in body
            assert "no hurdle" in body.lower()
        for output in (core_html, core_md):
            assert "6–10" not in lane_body(output)

    def test_an_overdue_catalyst_renders_a_warning(self, generator):
        html, md = rendered(generator, make_result(), lane_context(catalyst={
            "description": "Demerger of the packaging arm",
            "expected_by": "2026-01-31", "status": "active", "overdue": True,
        }))

        for output in (html, md):
            body = lane_body(output)
            assert "Demerger of the packaging arm" in body
            assert "2026-01-31" in body
            assert "overdue" in body.lower()

    def test_a_spent_catalyst_renders_without_the_warning(self, generator):
        html, md = rendered(generator, make_result(), lane_context(catalyst={
            "description": "Demerger of the packaging arm",
            "expected_by": "2026-01-31", "status": "spent", "overdue": False,
        }))

        for output in (html, md):
            body = lane_body(output)
            assert "spent" in body.lower()
            assert "overdue" not in body.lower()


class TestTheGoldenSeesChartsStopRendering:
    """What actually protects the golden from silently losing every chart.

    A residual finding held that it did not: `_CHART` rewrote a container line
    to `[chart]</div>` whether or not anything was inside, so — the argument
    went — every chart builder returning `""` would leave the golden green.

    **The conclusion was wrong, and this class is why.** The template guards
    every container with `{% if chart %}`, so an empty chart removes the
    container *and its card* rather than rendering an empty one: the count
    drops from three to one, which the comparison sees whichever normaliser is
    used. Verified against the real generator before the normaliser was
    touched.

    The observation behind the finding was still correct — the two cases did
    normalise identically — and that conflation is closed here, because it is
    one `{% if %}` away from mattering. Both facts get a test, so neither is
    rediscovered as a surprise.
    """

    def dashboard(self, tmp_path, dead_charts=False, monkeypatch=None):
        """A rendered dashboard, optionally with every chart builder silenced.

        Patched on `report_charts`, which is where the builders are defined —
        they were never methods in anything but name, and none of them touched
        `self`. `render_charts` calls them as module globals, so rebinding them
        there is what a real failure would look like: a builder returning `""`
        because its data contract moved.
        """
        if dead_charts:
            for builder in ("roce_trend_chart", "pe_band_chart",
                            "growth_chart", "cashflow_quality_chart"):
                monkeypatch.setattr(
                    report_charts, builder, lambda *a, **k: ""
                )
        generator = ReportGenerator(output_dir=str(tmp_path))
        report_dir = Path(generator.generate(make_result("TEST"), formats=["html"]))
        return (report_dir / "TEST_dashboard.html").read_text()

    def test_dead_charts_change_the_normalised_report(self, tmp_path, monkeypatch):
        """The property the golden actually needs, asserted end to end against
        the real generator rather than against a hand-written fragment."""
        healthy = normalise(self.dashboard(tmp_path / "a"))
        dead = normalise(self.dashboard(tmp_path / "b", True, monkeypatch))

        assert healthy != dead

    def test_it_is_the_container_count_that_falls(self, tmp_path, monkeypatch):
        """Naming the mechanism, because the finding assumed a different one.
        An empty chart does not render an empty container — the template's
        `{% if %}` removes the container entirely."""
        healthy = self.dashboard(tmp_path / "a")
        dead = self.dashboard(tmp_path / "b", True, monkeypatch)

        assert healthy.count(CONTAINER) > dead.count(CONTAINER)

    def test_an_empty_container_would_not_be_mistaken_for_a_full_one(self):
        """The latent conflation, closed. Nothing emits an empty container
        today; one removed `{% if %}` would, and the golden must not read it as
        a chart that rendered."""
        full = (
            f'  {CONTAINER}<div id="abc" class="plotly-graph-div">x</div>'
            f'<script>Plotly.newPlot("abc")</script></div>'
        )
        empty = f"  {CONTAINER}</div>"

        assert normalise(full) != normalise(empty)
        assert CHART_RENDERED in normalise(full)
        assert CHART_EMPTY in normalise(empty)

    def test_a_rendered_chart_still_normalises_away_its_uuid(self):
        """The reason any of this is normalised: Plotly stamps a fresh uuid
        every run, and two identical runs must compare equal."""
        def chart(uuid):
            return (
                f'  {CONTAINER}<div id="{uuid}" class="plotly-graph-div"></div>'
                f'<script>Plotly.newPlot("{uuid}")</script></div>'
            )

        assert normalise(chart("aaa-111")) == normalise(chart("bbb-222"))


class TestUntrackedReportsAreUnchanged:
    def test_generate_without_lane_context_matches_the_pre_change_golden(self, generator):
        report_dir = generator.generate(make_result(), formats=["html", "md"])

        for name, golden in (
            ("TEST_dashboard.html", "pre_lane_section_dashboard.html"),
            ("TEST_report.md", "pre_lane_section_report.md"),
        ):
            produced = normalise((report_dir / name).read_text())
            assert produced == (GOLDEN_DIR / golden).read_text(), (
                f"{name} differs from the pre-U7 golden. A ticker with no lane "
                f"context must render exactly as it did before this section "
                f"existed; regenerate the golden only for a change you meant."
            )

    def test_generate_without_lane_context_renders_no_section(self, generator):
        report_dir = generator.generate(make_result(), formats=["html", "md"])
        html = (report_dir / "TEST_dashboard.html").read_text()
        md = (report_dir / "TEST_report.md").read_text()

        assert "Lane &amp; Friction" not in html
        assert "## Lane & Friction" not in md

    def test_generate_with_a_lane_context_renders_the_six_gates(self, generator):
        report_dir = generator.generate(
            make_result(), formats=["html"], lane_context=lane_context()
        )
        html = (report_dir / "TEST_dashboard.html").read_text()

        assert "<h2>Lane &amp; Friction</h2>" in html
        for label in GATE_LABELS:
            assert label in html
        assert "fail" in lane_body(html)


def lane_body(output: str) -> str:
    """Just the Lane & Friction section, so an assertion cannot pass on the rest.

    A word like "eligible" occurs all over a report; the claim being made is
    about this section only.
    """
    if "## Lane & Friction" in output:
        return lane_section(output)
    body = output.split("<h2>Lane &amp; Friction</h2>", 1)[1]
    return body.split("<h2>", 1)[0]
