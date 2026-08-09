"""The closed component set — what may be said, and what may never be.

Three properties carry this unit, and each of them is a property rather than an
example, because every one of the five R15 leaks the plan names is an
*omission*. Nobody decided to render `cfi_dominated_by_acquisitions` in title
case; somebody forgot a label and the fallback was helpful. A test that checks
the components this unit happens to build would pass on the day the next label
is forgotten.

So: the markup ban is a scan over every string field of every component built
from the real registry, not an inspection. The R15 ban is asserted at the
constructor, on shapes rather than on a list of known-bad strings. And the
surface contract is asserted on a renderer that does not exist yet — U10 and
U11 are the consumers, and the mechanism has to bind them when they arrive
rather than depend on their authors having read the module.

Two limits are asserted deliberately, as tests, because an undocumented limit
gets mistaken for a guarantee: a single-word key is invisible mid-sentence, and
an exception whose text happens to be ordinary prose is invisible everywhere.
Both are recorded below with the reason they cannot be closed by scanning.
"""

import json
import re
from pathlib import Path

import pytest

from boundless100x.compute_engine.eligibility import effective_gates
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.sector import (
    MODERATE,
    NON_CONSIDERATION,
    STRONG,
    UNKNOWN as SECTOR_UNKNOWN,
    SectorApplicability,
)
from boundless100x.output import report_components as rc
from boundless100x.output.contradiction import ContradictionPairs
from boundless100x.output.report_expansion import (
    SECTOR_MISMATCH,
    TRIGGER_LABELS,
    ExpansionDecider,
    ScoredCorpus,
    load_scored_corpus,
)
from boundless100x.output.report_reading import (
    BANDS_NOT_DECLARED,
    Quantity,
    Reading,
    read_metric,
    read_metrics,
)
from boundless100x.output.report_vocabulary import (
    CATEGORICAL_VALUE_LABELS,
    ELEMENT_CONFIG,
    FLAG_LABELS,
)
from tests.conftest import latest_scores_for


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine():
    return ComputeEngine()


@pytest.fixture(scope="module")
def metric_configs(engine):
    return engine.metrics


@pytest.fixture(scope="module")
def vocabulary(metric_configs):
    """The real vocabulary, over the real registry.

    Module-scoped, and the disclosure cache it carries is the reason to say so:
    it is keyed on the declaration rather than on the company, so sharing one
    across tests is sharing a lookup table, not sharing state about a ticker.
    """
    return rc.Vocabulary(metric_configs)


@pytest.fixture(autouse=True)
def _surfaces_registry_is_restored():
    """Guards the shared `rc.SURFACES` registry itself, not just one test.

    A test that registers under an occupied name (e.g. "console", which
    `ConsoleComponents` in `cli_common.py` claims at import time) and then
    tears down by popping rather than restoring silently deletes the real
    entry instead of undoing its own mutation — invisible within this file,
    and only a `KeyError`/`KeyError`-shaped failure at a distance, in
    whichever other test file happens to run after this one (finding #9).
    Snapshotting before every test and comparing after turns that class of
    leak into an immediate, loud failure in the test that caused it, rather
    than a pass that depends on collection order.
    """
    before = dict(rc.SURFACES)
    yield
    assert rc.SURFACES == before, (
        "a test left boundless100x.output.report_components.SURFACES "
        "mutated — restore whatever was there before the test, don't pop it"
    )


def reading_for(metric_configs, metric_id, value, *, error=None, applicability=None):
    """One `Reading` through the real reading layer, as U6 produces it."""
    return read_metric(
        metric_id,
        metric_configs.get(metric_id),
        None if (value is None and error is None)
        else MetricResult(value=value, error=error),
        applicability=applicability,
    )


class CompleteSurface:
    """A renderer that handles the whole set. The shape U10 and U11 must take."""

    def render_finding(self, component): ...
    def render_metric_row(self, component): ...
    def render_reading(self, component): ...
    def render_disclosure(self, component): ...
    def render_unknown(self, component): ...
    def render_caveat(self, component): ...


def strings_in(component) -> list[tuple[str, str]]:
    """Every reader-facing string a surface could print, nested ones included.

    Walks `TEXT_FIELDS` and recurses into the components a component may hold —
    a row's unknown, a row's disclosure reference — because a leak nested one
    level down renders exactly as visibly as one at the top.
    """
    found: list[tuple[str, str]] = []
    for name in getattr(type(component), "TEXT_FIELDS", ()):
        found.append((f"{type(component).__name__}.{name}",
                      str(getattr(component, name) or "")))
    for name in getattr(type(component), "HANDLE_FIELDS", ()):
        nested = getattr(component, name, None)
        if isinstance(nested, rc._Component):
            found.extend(strings_in(nested))
    return found


# ── The set is closed ─────────────────────────────────────────────────────


class TestTheSetIsClosed:
    """R13. Six members, enumerated once, and a split every field must declare."""

    def test_the_six_members_are_the_six_the_plan_names(self):
        assert rc.COMPONENT_KINDS == (
            "finding", "metric_row", "reading", "disclosure", "unknown", "caveat",
        )

    def test_every_kind_has_a_type_and_every_type_has_a_kind(self):
        assert set(rc.COMPONENT_TYPES) == set(rc.COMPONENT_KINDS)
        for kind, types in rc.COMPONENT_TYPES.items():
            assert types, f"{kind} declares no type"
            for component_type in types:
                assert component_type.KIND == kind

    def test_every_field_is_declared_either_reader_facing_or_a_handle(self):
        """The guard only runs on fields somebody said it should run on.

        A field added without that decision would be unguarded by default,
        which is the quiet direction. Asserting the union rather than the
        guarded set is what makes the omission visible: a new field belongs to
        one list or the other, and choosing is the point.
        """
        for types in rc.COMPONENT_TYPES.values():
            for component_type in types:
                declared = set(component_type.TEXT_FIELDS) | set(
                    component_type.HANDLE_FIELDS
                )
                assert declared == set(rc.component_fields(component_type)), (
                    f"{component_type.__name__} has fields in neither "
                    f"TEXT_FIELDS nor HANDLE_FIELDS"
                )
                assert not (
                    set(component_type.TEXT_FIELDS)
                    & set(component_type.HANDLE_FIELDS)
                )

    def test_a_section_refuses_anything_outside_the_set(self, vocabulary):
        line = rc.section_reading("size", vocabulary, score=5.0)
        with pytest.raises(rc.ComponentContentError, match="outside the closed set"):
            rc.Section(key="size", title="Size", reading=line, findings=("a string",))


# ── Data, not markup ──────────────────────────────────────────────────────


class TestAComponentCarriesDataAndNotMarkup:
    """R14. Three surfaces render one component three ways, or none of them can.

    The scan is mechanical because inspection is what let the current report
    grow a string with a backtick in it: fine in Markdown, punctuation in HTML,
    punctuation on the console.
    """

    @pytest.mark.parametrize("bad, shape", [
        ("<b>RoCE</b>", "HTML tag"),
        ("RoCE &amp; ROE", "HTML entity"),
        ("Metric | Value", "table pipe"),
        ("[bold]RoCE[/bold]", "bracket"),
        ("The `categories` table", "backtick"),
        ("**Strong**", "bold"),
        ("Line one\nLine two", "line break"),
        ("- A bullet", "Markdown block marker"),
        ("## A heading", "Markdown block marker"),
        ("\x1b[31mred\x1b[0m", "bracket"),
    ])
    def test_markup_is_refused_at_the_constructor(self, bad, shape):
        with pytest.raises(rc.ComponentContentError):
            rc.Caveat(text=bad)

    def test_an_angle_bracket_that_is_not_a_tag_is_allowed(self):
        """Two shipped labels read "(>25%)" and "RoCE > 15% Count (10yr)".

        A rule that banned `<` and `>` outright would be a rule the vocabulary
        has to be written around, and a vocabulary written around a guard stops
        being the plain wording R15 exists to produce.
        """
        assert rc.Finding(headline="Exceptional RoCE (>25%)").headline

    def test_no_component_built_from_the_real_registry_carries_markup(
        self, metric_configs, vocabulary
    ):
        """Every metric in the registry, read and rowed, scanned field by field.

        The guard already ran at each constructor, so this cannot fail without
        the guard having a hole — which is exactly what it is here to find. It
        is the only test in the file that would notice a shape nobody thought
        of, because it is the only one running over text nobody wrote for it.
        """
        offenders = []
        for metric_id in metric_configs:
            reading = reading_for(metric_configs, metric_id, 12.0)
            row = rc.metric_row(metric_id, reading, vocabulary, score=0.5)
            for where, text in strings_in(row):
                for name, pattern in rc._MARKUP_SHAPES:
                    if pattern.search(text):
                        offenders.append((metric_id, where, name))
        assert not offenders


# ── R15 ───────────────────────────────────────────────────────────────────


class TestR15IsAConstructorInvariant:
    """No raw id, enum, lifecycle key or exception string reaches a reader."""

    @pytest.mark.parametrize("raw", [
        "cfi_dominated_by_acquisitions",
        "The metric roce_5yr_avg came out low",
        "quality_business scored 4.7",
        "founder_led_high_holding",
        "Moved to exit_review this quarter",
    ])
    def test_a_snake_case_token_is_refused_anywhere_in_the_string(self, raw):
        with pytest.raises(rc.ComponentContentError, match="raw identifier"):
            rc.Caveat(text=raw)

    @pytest.mark.parametrize("key", ["core", "probe", "exited", "watchlist",
                                     "eligible", "indeterminate", "percent"])
    def test_a_whole_field_that_is_a_reserved_key_is_refused(self, key):
        """The action badge's defect: a field whose entire content is an enum."""
        with pytest.raises(rc.ComponentContentError, match="raw key"):
            rc.Finding(headline=key)

    @pytest.mark.parametrize("raw", [
        "Data fetch failed: Traceback (most recent call last)",
        'Compute engine failed: File "/x/engine.py", line 44',
        "Scoring failed: ValueError: no rows",
        "LLM analysis failed: anthropic.APIError",
        "Data fetch failed: [Errno 2] No such file",
        "Compute engine failed: boundless100x.compute_engine.engine broke",
        "Could not read /Users/someone/reports/x.json",
        "the metric reported: 'borrowings'",
    ])
    def test_an_exception_shaped_string_is_refused(self, raw):
        with pytest.raises(rc.ComponentContentError):
            rc.Caveat(text=raw)

    def test_the_limit_shape_detection_cannot_close(self):
        """An exception whose text is ordinary prose is undetectable. Stated.

        `str(Exception("Screener returned 404"))` is a sentence. No scanner
        distinguishes it from one somebody wrote, so this asserts that the
        guard lets it through — recording the limit rather than implying a
        guarantee the code does not give. `caveat_from_run_error` is what
        actually protects this case, by never rendering the untrusted half.
        """
        assert rc.Caveat(text="Screener returned 404").text

    def test_the_limit_a_single_word_key_is_invisible_mid_sentence(self):
        """`probe` is an English word. Routing, not detection, covers it.

        Caught only as a whole field, which the test above asserts. Buried in a
        sentence it reads as prose and is left alone, which is correct: a guard
        that refused the word "probe" would refuse most sentences about a
        probe.
        """
        assert rc.Caveat(text="A probe position was opened in core.").text

    def test_a_reserved_key_list_covers_the_action_vocabulary(self):
        """`action_policy`'s five actions, pinned rather than imported.

        The module deliberately does not import `action_policy` — that module
        reaches into the report layer's siblings and a cycle is the one thing
        this leaf must not acquire — so the copy is checked against the
        original here instead.
        """
        from boundless100x.action_policy import ACTION_ORDER

        assert set(ACTION_ORDER) <= rc.RESERVED_KEYS


class TestARunErrorNeverReachesTheReaderVerbatim:
    """The fifth named leak: `service.py`'s warnings, printed as they arrive."""

    def test_the_untrusted_half_is_dropped_and_the_authored_half_is_kept(self):
        caveat = rc.caveat_from_run_error(
            "Data fetch failed: KeyError('promoter_pct')"
        )

        assert caveat.text.startswith("Data fetch failed")
        assert "KeyError" not in caveat.text
        assert "promoter_pct" not in caveat.text
        assert "run log" in caveat.text

    def test_prose_that_happens_to_be_an_exception_is_dropped_too(self):
        """The case shape detection cannot see, handled structurally.

        The split is made on the colon rather than on whether the tail looks
        dangerous, because whether it looks dangerous is exactly the thing that
        cannot be known.
        """
        caveat = rc.caveat_from_run_error("Data fetch failed: Screener returned 404")

        assert "404" not in caveat.text
        assert caveat.text.startswith("Data fetch failed")

    def test_an_error_with_no_authored_clause_says_only_what_is_known(self):
        caveat = rc.caveat_from_run_error("KeyError('promoter_pct')")

        assert "promoter_pct" not in caveat.text
        assert "did not complete" in caveat.text

    def test_it_is_a_warning_by_default_because_something_was_lost(self):
        assert rc.caveat_from_run_error("Scoring failed: x").severity == rc.WARNING


# ── The unregistered flag ─────────────────────────────────────────────────


class TestAnUnregisteredFlagBecomesUnknown:
    """The named defect: `f.replace("_", " ").title()` is a leak, not a fallback."""

    def test_an_unregistered_flag_produces_the_unknown_component(self):
        built = rc.finding_from_flag("some_signal_nobody_registered")

        assert isinstance(built, rc.Unknown)
        assert "some_signal_nobody_registered" not in built.sentence
        assert "Some Signal Nobody Registered" not in built.sentence
        assert built.reason.strip()

    def test_a_registered_flag_produces_a_finding_with_its_own_words(self):
        built = rc.finding_from_flag("cfi_dominated_by_acquisitions")

        assert isinstance(built, rc.Finding)
        assert built.headline == "Capex Dominated by Acquisitions"
        assert built.sentiment == "bad"
        assert built.source == "cfi_dominated_by_acquisitions"

    def test_every_shipped_flag_label_survives_the_guard(self):
        """The vocabulary is the safe path, so the safe path has to be safe.

        A label with a bracket or a raw id in it would raise at every call site
        and the report would be the place that found out.
        """
        for flag in FLAG_LABELS:
            assert isinstance(rc.finding_from_flag(flag), rc.Finding)

    def test_the_scorer_flag_that_caps_an_action_is_registered(self):
        """`low_data_coverage` was the one unregistered flag in the corpus.

        Emitted by `SQGLPScorer` rather than by a metric, which is how every
        audit of `FLAG_LABELS` walked past it — and it is the flag
        `action_policy` caps a `buy` on, so the single signal that can change
        the displayed action had no wording of its own.
        """
        built = rc.finding_from_flag("low_data_coverage")

        assert isinstance(built, rc.Finding)
        assert built.sentiment == "bad"

    def test_an_unregistered_metric_is_shown_as_unknown_rather_than_dropped(
        self, vocabulary
    ):
        """The drill-down's defect is the opposite one: a silent omission.

        A row that is not there is invisible in a way a stated absence is not,
        so the metric appears, saying that nothing names it.
        """
        reading = Reading(
            metric_id="a_metric_nobody_declared",
            status="value_absent",
            reason="this metric was not computed for this company",
        )
        built = rc.metric_row("a_metric_nobody_declared", reading, vocabulary)

        assert isinstance(built, rc.Unknown)
        assert "a_metric_nobody_declared" not in built.sentence


# ── The named grades ──────────────────────────────────────────────────────


class TestNamedGradesHaveRegisteredLabels:
    """The gap U6 left open: five metrics whose value is a raw enum."""

    def test_the_labelled_metrics_are_exactly_the_categorical_ones(
        self, metric_configs
    ):
        """Derived from the registry, never listed.

        A sixth categorical metric added tomorrow fails here rather than
        rendering its grade raw, which is the only version of this test worth
        having — a hardcoded list of five is a list that goes stale silently.
        """
        categorical = {
            metric_id
            for metric_id, config in metric_configs.items()
            if ((config.get("presentation") or {}).get("unit")) == "category"
        }

        assert set(CATEGORICAL_VALUE_LABELS) == categorical

    def test_every_declared_grade_has_a_label_and_no_label_is_invented(
        self, metric_configs
    ):
        """The key set comes from each metric's own `scoring.categories` table.

        Both directions matter. A grade with no label leaks; a label for a
        grade the metric cannot emit is dead vocabulary that reads as coverage.
        """
        for metric_id, labels in CATEGORICAL_VALUE_LABELS.items():
            declared = set(
                (metric_configs[metric_id].get("scoring") or {}).get("categories")
                or {}
            )
            assert set(labels) == declared, metric_id

    def test_the_grades_the_sector_classifier_can_return_are_all_labelled(self):
        """`classify_sector`'s constants, checked against the labels directly.

        The YAML table and the implementation are two statements of one
        vocabulary; this is the one place they are compared, so a bucket added
        to the classifier without a label cannot pass as a table nobody
        updated.
        """
        labels = CATEGORICAL_VALUE_LABELS["sector_tailwind"]

        assert {STRONG, MODERATE, NON_CONSIDERATION, SECTOR_UNKNOWN} <= set(labels)

    def test_every_label_and_gloss_survives_the_guard(self):
        for metric_id, labels in CATEGORICAL_VALUE_LABELS.items():
            for value, (label, gloss) in labels.items():
                rc.guard_text(label, field=f"{metric_id}.{value} label")
                rc.guard_text(gloss, field=f"{metric_id}.{value} gloss")

    def test_a_grade_reaches_the_row_as_its_label_not_as_its_value(
        self, metric_configs, vocabulary
    ):
        reading = reading_for(
            metric_configs, "owner_operator_signal", "founder_led_high_holding"
        )
        row = rc.metric_row("owner_operator_signal", reading, vocabulary)

        assert row.value == "Founder-led, majority stake"
        assert "founder_led" not in row.value
        # And the gloss is the reading, not the declared bands-absent reason,
        # which explains to a developer why a band walk was skipped.
        assert "at least half the company" in row.reading

    def test_a_grade_nobody_labelled_reads_unknown_rather_than_humanised(
        self, metric_configs, vocabulary
    ):
        reading = reading_for(
            metric_configs, "owner_operator_signal", "a_grade_from_the_future"
        )
        row = rc.metric_row("owner_operator_signal", reading, vocabulary)

        assert isinstance(row, rc.MetricRow)
        assert row.value == ""
        assert not row.known
        assert "a_grade_from_the_future" not in row.unknown.sentence


# ── The disclosure ────────────────────────────────────────────────────────


class TestTheDisclosureIsReachableAndNeverInline:
    """R3, both halves. U3 wrote the explanation; this is what makes it reach."""

    def test_the_reading_flow_contains_no_explanation(self, pfc_section):
        assert not any(
            isinstance(component, rc.Disclosure) for component in pfc_section.flow
        )

    def test_a_row_holds_a_reference_that_carries_no_body(self, pfc_section):
        refs = [row.disclosure for row in pfc_section.rows if row.disclosure]

        assert refs, "no row in a thirteen-metric section reached an explanation"
        for ref in refs:
            assert isinstance(ref, rc.DisclosureRef)
            assert not hasattr(ref, "body")

    def test_the_explanations_are_reachable_from_the_section(self, pfc_section):
        anchors = {d.anchor for d in pfc_section.disclosures}
        for row in pfc_section.rows:
            if row.disclosure:
                assert row.disclosure.anchor in anchors

    def test_a_disclosure_declares_itself_deferred(self):
        assert rc.Disclosure.DEFERRED is True

    def test_a_disclosure_cannot_be_placed_among_the_findings(self, vocabulary):
        line = rc.section_reading("size", vocabulary, score=5.0)
        explanation = rc.Disclosure(title="Size", body="What it measures.",
                                    anchor="market_cap")

        with pytest.raises(rc.ComponentContentError, match="outside the closed set"):
            rc.Section(key="size", title="Size", reading=line,
                       findings=(explanation,))

    def test_an_empty_explanation_is_refused(self):
        with pytest.raises(rc.ComponentContentError):
            rc.Disclosure(title="Size", body="   ", anchor="market_cap")

    def test_every_metric_that_declares_a_meaning_reaches_one(
        self, metric_configs, vocabulary
    ):
        """R3 says *every* metric, so the count is the assertion.

        Two of the fifty-seven are expected to fall short and they are named:
        their explanations are clean, but their bands-absent reasons name
        Python parameters. That is a declaration defect this unit surfaces
        rather than one it may fix — `elements/*.yaml` is out of scope here.
        """
        missing = []
        for metric_id, config in metric_configs.items():
            meaning = (config.get("presentation") or {}).get("meaning")
            if not str(meaning or "").strip():
                continue
            reading = reading_for(metric_configs, metric_id, 12.0)
            if rc.disclosure_for(metric_id, reading, vocabulary) is None:
                missing.append(metric_id)

        assert not missing


# ── R12 and R4 on a row ───────────────────────────────────────────────────


class TestARowNeverShowsABareNumberOrAnEmptyCell:
    """R12 and R4, as the pair of invariants KD6 turns on."""

    def test_a_bare_figure_is_refused(self):
        """KD6's own example: a uniform row showing `0.84` with no unit."""
        with pytest.raises(rc.ComponentContentError, match="bare figure"):
            rc.MetricRow(label="Debt / Equity", value="0.84", reading="low")

    def test_a_row_carries_either_a_reading_or_a_reason_never_neither(self):
        with pytest.raises(rc.ComponentContentError, match="R4"):
            rc.MetricRow(label="Debt / Equity", value="0.84x")

    def test_a_row_carries_either_a_reading_or_a_reason_never_both(self):
        with pytest.raises(rc.ComponentContentError, match="R4"):
            rc.MetricRow(
                label="Debt / Equity", value="0.84x", reading="low",
                unknown=rc.Unknown(subject="No reading", reason="because"),
            )

    def test_ae4_a_metric_with_no_declared_bands_shows_its_value_and_the_reason(
        self, metric_configs, vocabulary
    ):
        """`pe_ttm` declares no bands on purpose — it is scored against peers."""
        reading = reading_for(metric_configs, "pe_ttm", 24.0)
        row = rc.metric_row("pe_ttm", reading, vocabulary)

        assert row.value == "24.0x"
        assert not row.known
        assert "sector peers" in row.unknown.reason
        # And the sibling metric the declaration names in backticks arrives as
        # its label, not as an id in punctuation.
        assert "pe_vs_historical" not in row.unknown.reason
        assert "P/E Percentile" in row.unknown.reason
        assert "`" not in row.unknown.reason

    def test_a_row_carries_the_direction_of_goodness_beside_the_figure(
        self, metric_configs, vocabulary
    ):
        """R12 is two things, and the unit is only one of them.

        "0.09x — poor" leaves a reader unable to say whether they wanted the
        number higher. The direction has nowhere to live but the row, so it is
        a field rather than something each of three surfaces looks up.
        """
        row = rc.metric_row(
            "roce_5yr_avg",
            reading_for(metric_configs, "roce_5yr_avg", 22.0), vocabulary,
        )
        assert row.direction == "higher is better"

        grade = rc.metric_row(
            "owner_operator_signal",
            reading_for(metric_configs, "owner_operator_signal",
                        "founder_led_high_holding"),
            vocabulary,
        )
        assert grade.direction == "a named grade, with no better or worse direction"

    def test_every_row_with_a_figure_states_its_direction(
        self, metric_configs, vocabulary
    ):
        for metric_id in metric_configs:
            reading = reading_for(metric_configs, metric_id, 12.0)
            row = rc.metric_row(metric_id, reading, vocabulary)
            if isinstance(row, rc.MetricRow) and row.value:
                assert row.direction.strip(), metric_id

    def test_a_score_always_carries_its_scale(self, metric_configs, vocabulary):
        reading = reading_for(metric_configs, "roce_5yr_avg", 22.0)
        row = rc.metric_row("roce_5yr_avg", reading, vocabulary, score=0.82)

        assert row.score == "82% of full marks"

    def test_the_two_things_called_score_cannot_be_swapped(
        self, metric_configs, vocabulary
    ):
        """`details[id]["score"]` is a 0–1 share; `elements[key]` is out of ten.

        Rendering an element's 7.0 through the row path would read as "700% of
        full marks", and rendering a metric's 0.8 through the section path as
        "0.8 / 10" — the second is the dangerous one, because it looks
        plausible. The row refuses the out-of-range figure rather than
        formatting it.
        """
        reading = reading_for(metric_configs, "roce_5yr_avg", 22.0)

        with pytest.raises(rc.ComponentContentError, match="0–1 share"):
            rc.metric_row("roce_5yr_avg", reading, vocabulary, score=7.0)

    def test_a_section_headline_is_out_of_ten(self, vocabulary):
        assert rc.section_reading(
            "quality_business", vocabulary, score=4.6875
        ).headline == "4.7 / 10"

    def test_a_unit_phrase_is_appended_when_a_format_leaves_a_naked_number(self):
        """The fallback that keeps R12 true even against a wrong declaration.

        No shipped `display.format` produces a bare numeral today, so this is
        the only place the branch runs — and it has to exist, because the
        alternative when one appears is a row that raises during report
        generation.
        """
        reading = Reading(
            metric_id="debt_equity",
            band="low",
            quantity=Quantity(value=0.84, unit="multiple",
                              direction="lower_is_better", display_format="{:.2f}"),
        )
        vocab = rc.Vocabulary({"debt_equity": {"name": "Debt / Equity"}})
        row = rc.metric_row("debt_equity", reading, vocab)

        assert row.value == "0.84 (times)"

    def test_a_section_that_could_not_be_scored_says_so(self, vocabulary):
        line = rc.section_reading("price", vocabulary, score=None)

        assert not line.known
        assert line.headline == ""
        assert "not the same as a score of zero" in line.unknown.reason

    def test_a_section_nobody_named_cannot_be_titled(self, vocabulary):
        with pytest.raises(rc.ComponentContentError, match="declared sections"):
            rc.section_reading("forward_signals", vocabulary, score=5.0)

    def test_the_element_labels_are_the_report_vocabularys_own(self, vocabulary):
        """R14's other half: one spelling of each element across three surfaces.

        The CLI keeps a second map today that spells the same elements
        differently; U11 replaces it with this one, and this is the assertion
        that there is a single source for it to point at.
        """
        for element, config in ELEMENT_CONFIG.items():
            assert vocabulary.element_title(element) == config["label"]


# ── The surface contract ──────────────────────────────────────────────────


class TestASurfaceMustRenderEveryMember:
    """R14's hardest clause, asserted against renderers that do not exist yet.

    U10 and U11 are the surfaces. The binding mechanism has to fail *them*, not
    this file, so it is the decorator that carries it: a renderer missing a
    handler cannot be defined at all, and the traceback points at the class
    that omitted it rather than at a test three directories away.
    """

    def test_a_complete_surface_is_missing_nothing(self):
        assert rc.missing_members(CompleteSurface) == ()

    def test_an_incomplete_surface_names_what_it_cannot_render(self):
        class HalfASurface:
            def render_finding(self, component): ...
            def render_reading(self, component): ...

        assert rc.missing_members(HalfASurface) == (
            "metric_row", "disclosure", "unknown", "caveat",
        )

    def test_verify_surface_raises_naming_the_handlers_to_add(self):
        class NoCaveats(CompleteSurface):
            render_caveat = None

        with pytest.raises(rc.IncompleteSurface, match="render_caveat"):
            rc.verify_surface(NoCaveats, "html")

    def test_the_decorator_refuses_an_incomplete_renderer_at_import_time(self):
        """The mechanism that will bind U10 and U11.

        Registration is not a later check somebody runs; it is the class
        statement itself. A surface that forgets `render_caveat` fails on the
        line that defines it.
        """
        with pytest.raises(rc.IncompleteSurface, match="render_unknown"):
            @rc.component_surface("html")
            class Forgot:
                def render_finding(self, component): ...
                def render_metric_row(self, component): ...
                def render_reading(self, component): ...
                def render_disclosure(self, component): ...
                def render_caveat(self, component): ...

        # Not `"html" not in rc.SURFACES` any more: U10 landed and registered a
        # real HTML renderer, so the slot is legitimately occupied. What the
        # refusal has to guarantee is that the *incomplete* class did not take
        # it — checked by name, because the class statement raised and never
        # bound one.
        assert "Forgot" not in {
            surface.__name__ for surface in rc.SURFACES.values()
        }

    def test_the_decorator_refuses_a_surface_r14_does_not_name(self):
        with pytest.raises(rc.IncompleteSurface, match="not one of the surfaces"):
            @rc.component_surface("pdf")
            class Extra(CompleteSurface):
                pass

    def test_a_complete_surface_registers_under_its_name(self):
        # "console" is claimed by the real `ConsoleComponents` (`cli_common.py`)
        # the moment that module is imported anywhere in the suite — which may
        # or may not have happened yet, depending on run order. Save whatever
        # is there now and put it back, rather than popping: popping deletes
        # the real registration outright when it got there first (finding #9).
        previous = rc.SURFACES.get("console")
        try:
            @rc.component_surface("console")
            class Console(CompleteSurface):
                pass

            assert rc.SURFACES["console"] is Console
        finally:
            if previous is None:
                rc.SURFACES.pop("console", None)
            else:
                rc.SURFACES["console"] = previous

    def test_r14_names_three_surfaces(self):
        assert rc.EXPECTED_SURFACES == ("html", "markdown", "console")

    def test_every_surface_that_has_landed_renders_every_member(self):
        """Vacuous today and load-bearing the moment U10 or U11 registers.

        Kept because the decorator can be bypassed — a renderer that never
        decorates itself is a renderer nothing checks — and because this is the
        assertion that reads as the requirement rather than as its mechanism.
        """
        for name, surface in rc.SURFACES.items():
            assert rc.missing_members(surface) == (), name

    def test_no_surface_registers_under_a_name_nobody_expects(self):
        assert set(rc.SURFACES) <= set(rc.EXPECTED_SURFACES)


# ── Against the company the plan was written about ────────────────────────



@pytest.fixture(scope="module")
def pfc_scores():
    path = latest_scores_for("PFC")
    if path is None:
        pytest.skip("PFC has not been analysed on this machine")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def pfc_section(metric_configs, engine, vocabulary, pfc_scores):
    """PFC's real Quality — Business section, built through the whole stack.

    Real readings from real scored values, the real applicability table, the
    real contradiction pairs and the real corpus on disk. A hand-built fixture
    would prove that the components accept what the test author gives them,
    which is not the property under question.
    """
    results = {
        metric_id: MetricResult(value=detail.get("value"), error=detail.get("error"))
        for metric_id, detail in pfc_scores["details"].items()
    }
    readings = read_metrics(
        metric_configs, results, sector="Finance",
        applicability=SectorApplicability(list(metric_configs)),
    )
    decider = ExpansionDecider(
        metric_configs,
        ContradictionPairs(metric_configs, effective_gates(engine.gates)),
        load_scored_corpus(),
    )
    decision = decider.evaluate(readings, pfc_scores)["quality_business"]
    shares = {
        metric_id: share
        for metric_id in metric_configs
        if (share := decider.weight_share(metric_id)) is not None
    }
    return rc.build_section(
        "quality_business", decision, readings, vocabulary, pfc_scores,
        weight_shares=shares,
    )


class TestPfcsQualityBusinessSection:
    """AE1 and AE7, rendered as components rather than as a decision."""

    def test_ae1_the_section_expands_and_names_the_three_lender_mismatches(
        self, pfc_section
    ):
        assert pfc_section.expanded

        mismatches = [
            finding for finding in pfc_section.findings
            if finding.source == SECTOR_MISMATCH
        ]
        assert {finding.subject for finding in mismatches} == {
            "DuPont: Asset Turnover", "DuPont: Equity Multiplier", "FCF Yield",
        }
        for finding in mismatches:
            assert finding.headline == TRIGGER_LABELS[SECTOR_MISMATCH]
            # R7: the table's own sentence travels, so each says what a lender
            # is instead of repeating the trigger's headline three times.
            assert finding.subject in finding.text
            assert "lender" in finding.text

    def test_ae7_the_collapsed_reading_states_the_coverage(self, pfc_section):
        assert pfc_section.reading.headline == "4.7 / 10"
        assert "32%" in pfc_section.reading.qualifier

    def test_every_row_carries_a_reading_or_a_reason(self, pfc_section):
        assert len(pfc_section.rows) == 13
        for row in pfc_section.rows:
            assert row.label.strip()
            # R4 as the type states it: exactly one of the two, never a blank
            # cell and never a reason with no absence to explain.
            assert bool(row.reading.strip()) is not (row.unknown is not None)
            if row.known:
                assert row.reading.strip()
            else:
                assert row.unknown.reason.strip()

    def test_a_lenders_asset_turnover_keeps_its_number_and_loses_its_band(
        self, pfc_section
    ):
        """R7 and AE1: the row is shown, the reading is withheld.

        Calling PFC asset-heavy for lending is the misreading the whole path
        exists to stop, and hiding the row instead would leave a reader
        wondering why a DuPont decomposition has two of three legs.

        The argument for withholding it lives in the section's finding, not in
        the row. Rendering it in both put the same paragraph on screen twice —
        six times over this card's three inapplicable metrics — so the row now
        states the fact and the finding keeps the explanation. Asserted as a
        split rather than as a substring, because "the essay is in exactly one
        of these two places" is the property that was wrong.
        """
        row = next(r for r in pfc_section.rows if r.metric_id == "dupont_turnover")

        assert row.value == "0.09x"
        assert not row.known
        assert "loan book" not in row.unknown.reason
        assert "does not measure anything" in row.unknown.reason

        findings = " ".join(f.text for f in pfc_section.findings)
        assert "loan book" in findings

    def test_nothing_in_the_flow_carries_a_raw_identifier(self, pfc_section):
        """The end-to-end statement of R15, over text nobody wrote for a test."""
        offenders = []
        for component in pfc_section.flow:
            for where, text in strings_in(component):
                match = rc._SNAKE_TOKEN.search(text)
                if match:
                    offenders.append((where, match.group(0)))
        assert not offenders

    def test_the_errored_metrics_reasons_are_cleaned_or_replaced(self, pfc_section):
        """PFC is the case that proves the guard is doing work, not decoration.

        Two of its rows carry a `MetricResult` error naming a dataframe column
        (`operating_profit`) and one carries a re-raised `KeyError` whose whole
        text is `'borrowings'`. All three reach the current report verbatim.
        Here they become a stated absence with no identifier in it.
        """
        by_id = {row.metric_id: row for row in pfc_section.rows}
        for metric_id in ("cash_conversion", "interest_coverage", "debt_equity"):
            row = by_id[metric_id]
            assert not row.known
            assert "operating_profit" not in row.unknown.reason
            assert "'borrowings'" not in row.unknown.reason
            assert row.unknown.reason.strip()


class TestBuildingASectionWithNothingToWorkFrom:
    """The degenerate inputs, because they are what an early run looks like."""

    def test_a_section_with_no_scores_reads_unknown_rather_than_zero(
        self, metric_configs, engine, vocabulary
    ):
        decider = ExpansionDecider(
            metric_configs,
            ContradictionPairs(metric_configs, effective_gates(engine.gates)),
            ScoredCorpus(),
        )
        decision = decider.evaluate_section("size", {}, None)
        section = rc.build_section("size", decision, {}, vocabulary, None)

        assert not section.reading.known
        assert section.rows == ()
        assert section.unknowns
        # The section-level unresolved lines become caveats, so the reader is
        # told which checks could not be run rather than reading a clean
        # section that nobody checked.
        assert section.caveats

    def test_a_reason_a_surface_could_not_render_becomes_an_honest_absence(
        self, vocabulary
    ):
        """A declaration the vocabulary cannot clean must not blow up a report.

        Hand-built, not read off `price_lever_signal`: its shipped
        `bands_absent_reason` was rewritten into clean prose once R15 started
        enforcing this (see `growth.yaml`'s comment on that key), so calling
        `reading_for(..., "price_lever_signal", "not_a_grade")` — the previous
        version of this test — produces a reason with no raw identifier in it
        at all. The assertion that two specific parameter names were absent
        then passed whether or not `metric_row`'s `except ComponentContentError`
        fallback worked, because nothing ever reached it.

        A reason containing a genuinely unregistered identifier is what
        `vocabulary.narrate` cannot clean (it only resolves *registered*
        metric ids) and what `Unknown.__post_init__`'s R15 guard then refuses
        — which is what `metric_row` must catch and degrade, rather than let
        the exception propagate and take the whole row down with it.
        """
        reading = Reading(
            metric_id="price_lever_signal",
            status=BANDS_NOT_DECLARED,
            reason="explained only in terms of internal_capex_ratio_helper",
        )

        row = rc.metric_row("price_lever_signal", reading, vocabulary)

        assert not row.known
        assert row.unknown.reason == (
            "could not be computed from this company's financial statements "
            "— the technical detail is in the run log rather than here"
        )


class TestTheModuleStaysALeaf:
    """The dependency direction U6 established, kept closed by the same means."""

    def test_it_imports_neither_the_llm_layer_nor_the_report_generator(self):
        import ast

        source = Path(rc.__file__).read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        assert not any(name.startswith("boundless100x.llm_layer")
                       for name in imported)
        assert "boundless100x.output.report_generator" not in imported

    def test_the_guard_is_reachable_without_building_a_component(self):
        """U10 and U11 will hold strings of their own — headings, totals, notes.

        Exposing the guard is what lets them check one without inventing a
        component to hold it, which is the shape a surface reaches for when the
        set does not quite fit and is how a seventh member gets born.
        """
        assert rc.guard_text("Quality — Business", field="title")
        with pytest.raises(rc.ComponentContentError):
            rc.guard_text("quality_business", field="title")


# ── Regression guard on the shape of the regexes ──────────────────────────


class TestTheDetectorsDoNotOverreach:
    """A guard that refuses good prose gets bypassed, which is worse than none."""

    @pytest.mark.parametrize("safe", [
        "Scored on 32% of this element's declared weight, below the 85% bar.",
        "Exceptional RoCE (>25%)",
        "RoCE > 15% Count (10yr)",
        "Capital Reinvestment Rate (ΔCapital / NOPAT)",
        "P/E Percentile (10yr traded range)",
        "FII + DII Holding",
        "A probe position in the core lane was opened at 6–10 percentage points.",
        "Money lent out is counted as an operating outflow.",
        "the metric reported: ratios.csv has no roce column",
        "Core — the compounder lane",
    ])
    def test_real_prose_passes(self, safe):
        assert rc.guard_text(safe, field="text") == safe

    def test_every_element_label_and_trigger_label_passes(self):
        for config in ELEMENT_CONFIG.values():
            rc.guard_text(config["label"], field="label")
        for label in TRIGGER_LABELS.values():
            rc.guard_text(label, field="label")

    def test_a_blank_required_field_is_refused(self):
        with pytest.raises(rc.ComponentContentError, match="blank"):
            rc.Unknown(subject="   ", reason="because")

    def test_the_snake_case_rule_needs_two_segments(self):
        """A single lowercase word is prose. `_SNAKE_TOKEN` must not fire on it."""
        assert not re.search(rc._SNAKE_TOKEN, "borrowings revenue roce")
        assert re.search(rc._SNAKE_TOKEN, "operating_profit")
