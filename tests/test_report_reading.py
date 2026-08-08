"""The reading layer — what a number *means*, and what it means that there is none.

Written unknown-first, deliberately. The band walk is the easy half and it
already existed as `report_generator._forward_band`; what that helper does with
an absence is return `""`, and an empty string is exactly the blank R4 forbids.
Every rule the rest of this plan turns on — R4's "unknown together with its
reason", R12's "no number without its unit and its direction", R18's coverage
clause — is a rule about the cases where the obvious implementation says
nothing. So those are the cases asserted first and at the greatest length.
"""

import ast
import pathlib

import pytest

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.sector import SectorApplicability
from boundless100x.output import report_reading as rr


# ── Fixtures small enough to read at the assertion ────────────────────────
#
# The layer is pure, so a test passes it a declaration dict and a value and
# nothing else. That is the point of KTD2 and it is what makes these readable.


def declaration(**overrides) -> dict:
    """A `presentation:` block in the shape `elements/*.yaml` writes one."""
    block = {
        "unit": "percent",
        "direction": "higher_is_better",
        "meaning": "Return on capital employed, averaged over five years.",
        "bands": [[25, "exceptional"], [15, "solid"]],
        "low_label": "weak",
    }
    block.update(overrides)
    return block


def lender_table() -> SectorApplicability:
    """A two-sector applicability table: one reviewed, one excluded metric."""
    return SectorApplicability(
        ["roce_5yr_avg", "dupont_turnover"],
        {
            "Finance": {
                "label": "Lenders",
                "not_applicable": {
                    "dupont_turnover": (
                        "A lender's assets are the loan book it earns on, not "
                        "plant it has to sweat."
                    )
                },
            },
            "Industrial Products": {"label": "Manufacturers", "not_applicable": {}},
        },
    )


class TestAnUnknownAlwaysCarriesItsReason:
    """R4, six ways. Each absence is a different fact and says which it is.

    A single "unknown" bucket would satisfy the letter of R4 and lose the only
    thing the reader can act on: whether the number is missing because the data
    was not there, because nobody declared how to read it, or because reading
    it would mean nothing for a company of this kind.
    """

    def test_a_metric_that_errored_names_the_error_rather_than_scoring_zero(self):
        reading = rr.read_metric(
            "roce_5yr_avg",
            {"presentation": declaration(), "display": {"format": "{:.1f}%"}},
            MetricResult(error="ratios.csv has no roce column"),
        )

        assert not reading.known
        assert reading.status == rr.METRIC_ERROR
        assert "roce column" in reading.reason
        assert reading.source_error == "ratios.csv has no roce column"
        # The failure mode this replaces: a scorer that skips an errored metric
        # and a table that then shows nothing, which reads as a zero.
        assert reading.band == ""
        assert "0" not in reading.sentence.replace("roce", "")

    def test_a_value_that_never_computed_is_unknown_rather_than_absent(self):
        reading = rr.read_value("roce_5yr_avg", declaration(), None)

        assert reading.status == rr.VALUE_ABSENT
        assert reading.reason
        assert reading.sentence.strip()

    def test_a_metric_that_never_ran_says_so_distinctly(self):
        """`None` for the whole result is not the same as a result with no value.

        One means the engine never reached the metric; the other means it ran
        and the data was not there. Same blank on the page, different fix.
        """
        never_ran = rr.read_metric("roce_5yr_avg", {"presentation": declaration()}, None)
        ran_empty = rr.read_metric(
            "roce_5yr_avg", {"presentation": declaration()}, MetricResult()
        )

        assert never_ran.status == ran_empty.status == rr.VALUE_ABSENT
        assert never_ran.reason != ran_empty.reason

    def test_no_declaration_at_all_names_the_missing_declaration(self):
        reading = rr.read_value("mystery_metric", None, 22.4)

        assert reading.status == rr.NO_DECLARATION
        assert "declar" in reading.reason
        # No declaration means no unit and no direction, so under R12 there is
        # nothing this layer may render the number as.
        assert reading.quantity is None
        assert "22.4" not in reading.sentence

    def test_an_incomplete_band_declaration_is_a_missing_declaration_too(self):
        """Bands with no `low_label` name a reading for high values and none
        for low ones. The validator rejects it at startup; if one ever reached
        here it must not render as the bottom band by accident."""
        reading = rr.read_value(
            "roce_5yr_avg", declaration(low_label=None), 3.0, display_format="{:.1f}%"
        )

        assert reading.status == rr.NO_DECLARATION
        assert reading.reason
        # The value survives, because AE4 says the row still shows it.
        assert reading.quantity is not None
        assert "3.0%" in reading.sentence

    def test_deliberately_absent_bands_render_their_declared_reason(self):
        """AE4, and the case nine shipped metrics are in on purpose.

        `bands: []` plus a `bands_absent_reason` is a first-class declaration,
        not an oversight — the reason string is what the row renders instead of
        a reading.
        """
        why = (
            "Scored against sector peers rather than an absolute line, because "
            "the same multiple is cheap for a consumer franchise and dear for a "
            "commodity processor."
        )
        reading = rr.read_value(
            "pe_ttm",
            declaration(unit="multiple", direction="lower_is_better", bands=[],
                        low_label=None, bands_absent_reason=why),
            5.3,
            display_format="{:.1f}x",
        )

        assert reading.status == rr.BANDS_NOT_DECLARED
        assert reading.reason == why
        # Never a bare number and never an empty cell: the value, its unit, and
        # the reason there is no reading.
        assert "5.3x" in reading.sentence
        assert why in reading.sentence
        assert reading.quantity.unit == "multiple"
        assert reading.quantity.direction == "lower_is_better"

    def test_a_metric_excluded_for_this_sector_reads_as_not_applicable(self):
        """AE1's per-metric half. The table's own sentence reaches the reader
        verbatim, per R7 — "does not apply" alone tells nobody anything."""
        reading = rr.read_metric(
            "dupont_turnover",
            {"presentation": declaration(unit="multiple"), "display": {"format": "{:.2f}x"}},
            MetricResult(value=0.09),
            applicability=lender_table().evaluate("dupont_turnover", "Finance"),
        )

        assert reading.status == rr.NOT_APPLICABLE
        assert "loan book" in reading.reason
        assert not reading.known
        # The number is still shown — the expanded section puts the reason in
        # front of the table rather than hiding the row.
        assert "0.09x" in reading.sentence
        # And the band that would have called a lender "asset-heavy" is withheld.
        assert reading.band == ""

    def test_a_non_numeric_value_against_numeric_bands_is_unknown(self):
        """`_forward_band` returns `""` here. An empty string in a reading
        column is indistinguishable from a reading nobody wrote."""
        reading = rr.read_value("roce_5yr_avg", declaration(), "n/a")

        assert reading.status == rr.VALUE_NOT_BANDABLE
        assert reading.reason
        assert reading.sentence.strip()

    def test_a_boolean_is_not_a_number_the_bands_may_walk(self):
        """`True >= 25` is False and `True >= 0` is True, so a bool silently
        lands in a band. `_forward_band` guards this and so does this layer."""
        reading = rr.read_value("roce_5yr_avg", declaration(), True)

        assert reading.status == rr.VALUE_NOT_BANDABLE

    def test_every_absence_path_produces_a_distinct_reason(self):
        """The property, not six spot checks: no two absences read alike."""
        applic = lender_table()
        reasons = [
            rr.read_value("roce_5yr_avg", None, 22.4).reason,
            rr.read_value("roce_5yr_avg", declaration(low_label=None), 3.0).reason,
            rr.read_value("roce_5yr_avg", declaration(), None).reason,
            rr.read_metric("roce_5yr_avg", {"presentation": declaration()}, None).reason,
            rr.read_value(
                "roce_5yr_avg", declaration(), None, error="no ratios file"
            ).reason,
            rr.read_value(
                "pe_ttm",
                declaration(bands=[], low_label=None, bands_absent_reason="peers only"),
                5.3,
            ).reason,
            rr.read_value("roce_5yr_avg", declaration(), "n/a").reason,
            rr.read_metric(
                "dupont_turnover",
                {"presentation": declaration(unit="multiple")},
                MetricResult(value=0.09),
                applicability=applic.evaluate("dupont_turnover", "Finance"),
            ).reason,
        ]

        assert all(r and r.strip() for r in reasons)
        assert len(set(reasons)) == len(reasons)

    def test_a_reading_cannot_be_built_unknown_without_a_reason(self):
        """R4 as a constructor invariant rather than a convention. A future
        caller inside this module cannot forget the reason."""
        with pytest.raises(ValueError, match="reason"):
            rr.Reading(metric_id="x", status=rr.VALUE_ABSENT, reason="")

    def test_a_reading_cannot_claim_a_band_it_did_not_resolve(self):
        with pytest.raises(ValueError, match="band"):
            rr.Reading(metric_id="x", status=rr.READ, band="")


class TestApplicabilityIsThreeValuedAndNeverSilentlyPasses:
    """R4 against the question "does this metric mean anything here?".

    The asymmetry from `sector.py` carries through: `applies` is the answer
    that lets a lender be marked down for lending, so it is the one that has to
    be earned. An unreviewed sector is indeterminate, and indeterminate is
    reported rather than resolved into either answer.
    """

    def test_an_unreviewed_sector_is_indeterminate_not_applicable(self):
        reading = rr.read_metric(
            "roce_5yr_avg",
            {"presentation": declaration()},
            MetricResult(value=22.4),
            applicability=lender_table().evaluate("roce_5yr_avg", "Capital Markets"),
        )

        assert reading.applicability.verdict == rr.INDETERMINATE
        assert not reading.applicability.known
        assert "has not been reviewed" in reading.applicability.reason

    def test_an_indeterminate_sector_does_not_withhold_the_reading(self):
        """The one place the plan's wording had to be read against its own
        arithmetic. Two of the twenty-six cached sectors are reviewed, so
        treating indeterminate applicability as grounds to withhold a reading
        would render every metric on nearly every company as unknown — and R1
        asks for the opposite. Indeterminate is reported *about applicability*
        and leaves the band walk alone; `sector.py` says the same, that the
        cost of an unreviewed sector falls on the expansion trigger."""
        reading = rr.read_metric(
            "roce_5yr_avg",
            {"presentation": declaration(), "display": {"format": "{:.1f}%"}},
            MetricResult(value=22.4),
            applicability=lender_table().evaluate("roce_5yr_avg", "Capital Markets"),
        )

        assert reading.known
        assert reading.band == "solid"
        assert not reading.applicability.known

    def test_a_reviewed_sector_that_did_not_exclude_the_metric_applies(self):
        reading = rr.read_metric(
            "roce_5yr_avg",
            {"presentation": declaration()},
            MetricResult(value=22.4),
            applicability=lender_table().evaluate("roce_5yr_avg", "Industrial Products"),
        )

        assert reading.applicability.verdict == rr.APPLIES
        assert reading.applicability.known

    def test_not_consulting_the_table_never_reads_as_applies(self):
        """A caller that did not ask has not been told. The default has to be
        indeterminate with a reason, or "nobody looked" and "we looked and it
        fits" become the same value on the page."""
        reading = rr.read_value("roce_5yr_avg", declaration(), 22.4)

        assert reading.applicability.verdict == rr.INDETERMINATE
        assert reading.applicability.reason
        assert not reading.applicability.known


class TestNoNumberTravelsWithoutItsUnitAndDirection:
    """R12, asserted against the type rather than against a habit.

    The defence is that a number can only leave this layer inside a `Quantity`,
    and a `Quantity` cannot be built without both. Documenting the rule would
    hold exactly as long as the next person read the docstring.
    """

    def test_a_quantity_cannot_exist_without_a_declared_unit(self):
        with pytest.raises(ValueError, match="unit"):
            rr.Quantity(value=22.4, unit="", direction="higher_is_better")
        with pytest.raises(ValueError, match="unit"):
            rr.Quantity(value=22.4, unit="percentage", direction="higher_is_better")

    def test_a_quantity_cannot_exist_without_a_declared_direction(self):
        with pytest.raises(ValueError, match="direction"):
            rr.Quantity(value=22.4, unit="percent", direction="")
        with pytest.raises(ValueError, match="direction"):
            rr.Quantity(value=22.4, unit="percent", direction="up_is_good")

    def test_a_reading_exposes_no_bare_value_attribute(self):
        """The route to the number is `reading.quantity.value` and nothing
        shorter. A `Reading.value` would be the path of least resistance
        straight past R12."""
        reading = rr.read_value("roce_5yr_avg", declaration(), 22.4,
                                display_format="{:.1f}%")

        assert not hasattr(reading, "value")
        assert reading.quantity.value == 22.4

    def test_the_fallback_rendering_never_emits_a_naked_numeral(self):
        """A caller holding only a `presentation` block has no `display.format`
        to render through. Every unit in the closed vocabulary still has to
        come out marked — including `count`, which has no affix of its own."""
        for unit in rr.PRESENTATION_UNITS:
            if unit == "category":
                continue  # A grade is not a number; R12 has nothing to say.
            quantity = rr.Quantity(value=4, unit=unit, direction="higher_is_better")
            assert quantity.text != "4", f"{unit} rendered a bare numeral"
            assert str(quantity) == quantity.text

    def test_a_declared_format_wins_over_the_fallback(self):
        """`display.format` is the metric's own typography and the only
        statement of it. The fallback is a floor, not a competitor."""
        quantity = rr.Quantity(value=5000.0, unit="inr_crore",
                               direction="lower_is_better",
                               display_format="₹{:,.0f} Cr")

        assert quantity.text == "₹5,000 Cr"

    def test_an_unusable_format_string_loses_the_typography_not_the_number(self):
        quantity = rr.Quantity(value="a_grade", unit="percent",
                               direction="higher_is_better",
                               display_format="{:.1f}%")

        assert "a_grade" in quantity.text

    def test_the_closed_vocabularies_are_covered_mechanically(self):
        """KTD6's lesson, applied here: derived from the validator rather than
        remembered. Adding a unit or a direction to `validator.py` without
        teaching this layer how to speak it is a test failure, not a metric
        that renders `None` beside its number."""
        assert set(rr.UNIT_AFFIXES) == set(rr.PRESENTATION_UNITS)
        assert set(rr.UNIT_PHRASES) == set(rr.PRESENTATION_UNITS)
        assert set(rr.DIRECTION_PHRASES) == set(rr.PRESENTATION_DIRECTIONS)
        assert all(p.strip() for p in rr.UNIT_PHRASES.values())
        assert all(p.strip() for p in rr.DIRECTION_PHRASES.values())

    def test_every_status_has_a_sentence_opening(self):
        assert set(rr.STATUS_PREFIXES) == rr.READING_STATUSES - {rr.READ}


class TestTheBandWalk:
    """R1: the value placed against the declaration, first threshold wins."""

    def test_a_value_inside_a_band_returns_that_label_and_the_direction(self):
        reading = rr.read_value("roce_5yr_avg", declaration(), 22.4,
                                display_format="{:.1f}%")

        assert reading.known
        assert reading.band == "solid"
        assert reading.quantity.direction == "higher_is_better"
        assert reading.sentence == "22.4% — solid (higher is better)"

    def test_the_top_band_opens_exactly_at_its_threshold(self):
        assert rr.resolve_band([[25, "exceptional"], [15, "solid"]], "weak", 25) == "exceptional"
        assert rr.resolve_band([[25, "exceptional"], [15, "solid"]], "weak", 24.9) == "solid"

    def test_below_every_band_falls_to_the_low_label(self):
        assert rr.resolve_band([[25, "exceptional"], [15, "solid"]], "weak", 3) == "weak"

    def test_a_lower_is_better_metrics_low_label_names_the_best_outcome(self):
        """The trap in the declaration format: thresholds descend, so for a
        `lower_is_better` metric the labels run worst-first and `low_label` is
        the good news. `debt_equity`'s real declaration, read both ends."""
        bands, low = [[0.5, "leveraged"], [0.1, "modest debt"]], "virtually debt-free"

        assert rr.resolve_band(bands, low, 1.4) == "leveraged"
        assert rr.resolve_band(bands, low, 0.02) == "virtually debt-free"

    def test_the_walk_matches_the_helper_it_generalises(self):
        """Differential against `report_generator._forward_band` on every value
        it can place. The semantics must not drift while the two coexist — a
        later unit collapses them, and a collapse onto changed behaviour would
        move readings nobody meant to move.

        The test imports the old helper; the module under test must not, which
        is what `TestTheLayerIsPure` asserts.
        """
        from boundless100x.output.report_generator import ReportGenerator

        config = {"bands": [(25.0, "exceptional"), (15.0, "solid")], "low_label": "weak"}
        for value in (-40, -0.1, 0, 14.99, 15, 15.01, 24.99, 25, 25.0, 900):
            assert rr.resolve_band(config["bands"], config["low_label"], value) == (
                ReportGenerator._forward_band(config, value)
            ), f"drifted at {value}"

    def test_an_unplaceable_value_returns_none_not_an_empty_label(self):
        """The single behavioural difference from `_forward_band`, and the
        reason this module exists: `""` is a reading nobody can distinguish
        from a reading nobody wrote."""
        assert rr.resolve_band([[25, "exceptional"]], "weak", None) is None
        assert rr.resolve_band([[25, "exceptional"]], "weak", "n/a") is None
        assert rr.resolve_band([], "weak", 22.4) == "weak"


class TestElementCoverage:
    """R18. The scorer already computes this and nothing has ever read it."""

    def test_an_element_below_the_bar_states_its_coverage(self):
        """AE7: PFC's Quality — Business, five scored metrics on 32% of the
        element's declared weight."""
        coverage = rr.read_element_coverage("quality_business", 0.32)

        assert coverage.status == rr.COVERAGE_LOW
        assert coverage.low
        assert "32%" in coverage.clause
        assert "85%" in coverage.clause
        # R15: `quality_business` is a raw key and never reaches a reader. The
        # caller prepends the label; the clause is a statement about a share.
        assert "quality_business" not in coverage.clause

    def test_an_element_above_the_bar_says_nothing(self):
        coverage = rr.read_element_coverage("growth", 0.94)

        assert coverage.status == rr.COVERAGE_ADEQUATE
        assert coverage.clause == ""
        assert not coverage.low

    def test_the_bar_itself_reads_as_adequate(self):
        assert rr.read_element_coverage("growth", 0.85).status == rr.COVERAGE_ADEQUATE
        assert rr.read_element_coverage("growth", 0.8499).status == rr.COVERAGE_LOW

    def test_unknown_coverage_is_not_adequate_coverage(self):
        """An element with no declared weight has a missing denominator, not a
        full numerator. Silence would make it read like a fully measured one —
        the confusion R18 exists to end, arrived at from the other side."""
        coverage = rr.read_element_coverage("forward_signals", None)

        assert coverage.status == rr.COVERAGE_UNKNOWN
        assert not coverage.known
        assert coverage.reason
        assert coverage.clause

    def test_a_share_outside_zero_to_one_is_unknown_rather_than_rendered(self):
        coverage = rr.read_element_coverage("growth", 4.2)

        assert coverage.status == rr.COVERAGE_UNKNOWN
        assert "4.200" in coverage.reason

    def test_the_threshold_is_the_scorers_own_and_is_not_restated(self):
        """Two statements of one bar drift silently: the composite would flag
        thin evidence at 0.85 while a section said nothing at 0.9."""
        from boundless100x.compute_engine.scorer import SQGLPScorer

        assert rr.LOW_COVERAGE_THRESHOLD == SQGLPScorer({}, {}).low_coverage_threshold

        # And the bar appears nowhere in this module as a literal *value* — the
        # AST rather than a text search, so the prose may explain the rule
        # while the code stays unable to restate it.
        tree = ast.parse(pathlib.Path(rr.__file__).read_text())
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        assert rr.LOW_COVERAGE_THRESHOLD not in literals

    def test_a_caller_with_a_tuned_scorer_may_pass_its_threshold(self):
        coverage = rr.read_element_coverage("growth", 0.9, threshold=0.95)

        assert coverage.status == rr.COVERAGE_LOW
        assert "95%" in coverage.clause

    def test_the_whole_scorer_coverage_block_reads_in_one_call(self):
        block = {
            "composite": 0.61,
            "elements": {"quality_business": 0.32, "growth": 0.94,
                         "longevity": 0.476, "size": None},
            "unscored": [],
        }

        readings = rr.read_element_coverages(block)

        assert readings["quality_business"].low
        assert not readings["growth"].low
        assert "48%" in readings["longevity"].clause
        assert readings["size"].status == rr.COVERAGE_UNKNOWN


class TestTheLayerIsPure:
    """KTD2's import boundary, walked off the file's own AST.

    Two failure modes to avoid, and the second is the reason for the AST.
    Importing `llm_layer` would make R2 hold by discipline rather than by
    construction, and a report generated on a `--no-llm` path would open on a
    blank the moment somebody reached for a model-written line. Importing
    `report_generator` would invert the direction the plan sequences these
    units in — the report imports the reading layer, not the reverse.

    A `sys.modules` check would pass vacuously whenever the offending module
    simply had not been imported yet in that process. Parsing the source cannot
    be fooled that way, and the anti-vacuity controls below prove the collector
    is doing work rather than looking at nothing.
    """

    # Reaching any of these from a layer that must never touch the network or
    # the filesystem would be a purity break even without an `llm_layer` in it.
    FORBIDDEN_ROOTS = frozenset({
        "requests", "urllib", "urllib3", "http", "httpx", "socket", "anthropic",
        "subprocess", "yfinance", "boto3",
    })

    def module_path(self) -> pathlib.Path:
        """Located off the installed package, not the working directory — a
        relative path would quietly find nothing and pass."""
        import boundless100x

        return pathlib.Path(boundless100x.__file__).parent / "output" / "report_reading.py"

    def package_of(self, path: pathlib.Path) -> str:
        """The dotted package containing `path`, needed to resolve `node.level`.

        `from . import x` names a different target depending on which package
        the importing file sits in — `report_reading.py`'s "current package"
        is `boundless100x.output`, not `report_reading` itself. Derived from
        the file's location under the installed `boundless100x` tree; the
        negative control below exercises a file that was never written to
        disk there, so it passes `package` to `imports_of` explicitly instead
        of going through this.
        """
        import boundless100x

        root = pathlib.Path(boundless100x.__file__).parent.parent
        parts = path.relative_to(root).with_suffix("").parts
        return ".".join(parts[:-1])

    def imports_of(self, path: pathlib.Path, *, package: str | None = None) -> set[str]:
        """Every dotted module name the file imports, relative or absolute.

        A relative import's `node.module` is `None` for `from . import x` and
        a bare, unqualified name for `from ..llm_layer import y` — neither
        reads as `boundless100x.llm_layer` on its own. The old walk either
        dropped the import outright (`module is None`, nothing to add at
        all — a `from . import report_generator` vanished completely) or kept
        a name too short for `test_the_only_project_imports_are_from_the_
        compute_engine` to recognise as a `boundless100x` import in the first
        place (finding #11: a relative escape from this layer was invisible to
        the very guard meant to catch it). `node.level` — the leading-dot
        count — is resolved against the file's own package the way Python's
        import system actually resolves it, and `alias.name` is folded in
        for the `from . import x` shape, where the imported name is the only
        place the target module appears at all.
        """
        if package is None:
            package = self.package_of(path)
        package_parts = package.split(".") if package else []

        tree = ast.parse(path.read_text())
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    # level=1 is "this package" (drop nothing); each further
                    # dot climbs one more level (drop one more trailing part).
                    base_parts = package_parts[: len(package_parts) - (node.level - 1)]
                    base = ".".join(base_parts)
                    resolved = f"{base}.{node.module}" if node.module else base
                else:
                    resolved = node.module or ""
                if resolved:
                    found.add(resolved)
                if node.module is None:
                    found.update(
                        f"{resolved}.{alias.name}" if resolved else alias.name
                        for alias in node.names
                    )
            elif isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
        return found

    def test_the_module_imports_nothing_from_the_llm_layer(self):
        imports = self.imports_of(self.module_path())

        offenders = [m for m in imports if "llm_layer" in m.split(".")]
        assert offenders == []

    def test_the_collector_is_not_looking_at_an_empty_file(self):
        """Anti-vacuity, first control: the file exists and the walk finds
        imports at all. An unparsed or renamed file would otherwise satisfy
        every assertion above by having nothing in it."""
        path = self.module_path()

        assert path.exists()
        imports = self.imports_of(path)
        assert len(imports) >= 3
        assert any(m.startswith("boundless100x.compute_engine") for m in imports)

    def test_the_collector_does_flag_a_module_that_really_imports_the_llm_layer(self):
        """Anti-vacuity, second control: the same walk over `service.py`, which
        imports `llm_layer` three times. A rule that never fires and a rule
        that cannot fire look identical from the passing side."""
        import boundless100x

        service = pathlib.Path(boundless100x.__file__).parent / "service.py"
        offenders = [m for m in self.imports_of(service) if "llm_layer" in m.split(".")]

        assert offenders, "the boundary check cannot detect the thing it forbids"

    def test_the_collector_does_flag_a_relative_import_of_a_forbidden_module(
        self, tmp_path
    ):
        """Anti-vacuity, third control, for the blind spot finding #11 closes.

        The two controls above only ever exercise an absolute import
        (`service.py`'s `from boundless100x.llm_layer import ...`). A relative
        one is a different AST shape — `from . import x` has `node.module is
        None`, and `from ..llm_layer import y` has `node.module == "llm_layer"`
        with no `boundless100x` prefix in it anywhere — and a collector that
        only worked on the absolute shape would pass every test in this class
        while being blind to exactly the escape KTD2 exists to prevent.

        Stand-in file, not `report_reading.py` itself: the point is that the
        collector resolves relative imports *given* a package, which is
        exercised here against a package (`boundless100x.output`) that never
        actually needs a passing grade.
        """
        fake = tmp_path / "fake_module.py"
        fake.write_text(
            "from . import report_generator\n"
            "from ..llm_layer import forward_growth\n"
        )
        imports = self.imports_of(fake, package="boundless100x.output")

        # `from . import report_generator` used to vanish outright
        # (`node.module is None`) — it must now resolve to something naming
        # the module the report layer must never import.
        assert any(
            m.startswith("boundless100x.output") and m.endswith("report_generator")
            for m in imports
        ), imports
        # `from ..llm_layer import forward_growth` used to resolve to the bare
        # name "llm_layer" (no `boundless100x` prefix) — properly qualified,
        # it both trips the llm_layer-specific check and now actually counts
        # as a `boundless100x` import for the "only from compute_engine" one.
        offenders = [m for m in imports if "llm_layer" in m.split(".")]
        assert offenders, imports
        assert any(m.split(".")[0] == "boundless100x" for m in offenders), imports

    def test_the_only_project_imports_are_from_the_compute_engine(self):
        """The dependency direction, stated positively. This catches the
        `report_generator` inversion and any future reach into `service`,
        `lifecycle` or the fetchers, none of which a pure reading layer needs."""
        project = sorted(
            m for m in self.imports_of(self.module_path())
            if m.split(".")[0] == "boundless100x"
        )

        assert project
        assert all(m.startswith("boundless100x.compute_engine") for m in project), project

    def test_the_module_reaches_neither_the_network_nor_a_subprocess(self):
        roots = {m.split(".")[0] for m in self.imports_of(self.module_path())}

        assert not roots & self.FORBIDDEN_ROOTS


class TestAgainstTheShippedDeclarations:
    """The real registry and the real applicability table, not hand-built dicts.

    A declaration *shape* mismatch is exactly the class of bug this layer would
    otherwise hide: every hand-written fixture in this file uses lists of
    two-element lists because that is what YAML produces, but a fixture agrees
    with itself by construction. These exercise all 57 shipped metrics and the
    table as it actually ships.
    """

    def engine(self):
        from boundless100x.compute_engine.engine import ComputeEngine

        return ComputeEngine()

    def probe_value(self, presentation):
        """A value guaranteed to land in the top band, or a plausible stand-in
        for a metric that declares none."""
        if presentation["bands"]:
            return presentation["bands"][0][0]
        return "some_grade" if presentation["unit"] == "category" else 1.0

    def test_every_shipped_metric_produces_a_reading_or_a_declared_reason(self):
        engine = self.engine()
        assert len(engine.metrics) >= 57

        outcomes = {}
        for metric_id, config in engine.metrics.items():
            reading = rr.read_metric(
                metric_id, config,
                MetricResult(value=self.probe_value(config["presentation"])),
            )
            outcomes[metric_id] = reading.status
            # R1/R4: never a blank, whichever branch it took.
            assert reading.sentence.strip(), metric_id
            # R3: the explanation is reachable for every metric.
            assert reading.meaning.strip(), metric_id

        unexpected = {
            m: s for m, s in outcomes.items()
            if s not in (rr.READ, rr.BANDS_NOT_DECLARED)
        }
        assert unexpected == {}
        assert sum(s == rr.BANDS_NOT_DECLARED for s in outcomes.values()) == 9

    def test_the_top_band_label_resolves_for_every_banded_metric(self):
        for metric_id, config in self.engine().metrics.items():
            presentation = config["presentation"]
            if not presentation["bands"]:
                continue
            threshold, label = presentation["bands"][0]
            reading = rr.read_metric(metric_id, config, MetricResult(value=threshold))

            assert reading.band == label, metric_id

    def test_a_value_under_every_band_reaches_the_low_label(self):
        for metric_id, config in self.engine().metrics.items():
            presentation = config["presentation"]
            if not presentation["bands"]:
                continue
            floor = presentation["bands"][-1][0]
            reading = rr.read_metric(metric_id, config, MetricResult(value=floor - 1))

            assert reading.band == presentation["low_label"], metric_id

    def test_no_shipped_metric_renders_a_bare_numeral(self):
        """R12 across the whole registry. `category` is exempt because a named
        grade is not a number; everything else must come out marked."""
        for metric_id, config in self.engine().metrics.items():
            presentation = config["presentation"]
            if presentation["unit"] == "category":
                continue
            value = self.probe_value(presentation)
            reading = rr.read_metric(metric_id, config, MetricResult(value=value))
            text = reading.quantity.text

            assert text not in (str(value), f"{value:g}"), f"{metric_id}: {text!r}"
            assert reading.quantity.direction in rr.PRESENTATION_DIRECTIONS
            assert reading.quantity.unit in rr.PRESENTATION_UNITS

    def test_the_nine_unbanded_metrics_render_their_own_declared_reason(self):
        """Each carries a different sentence — a shared placeholder would mean
        the reason had stopped being the deliverable."""
        engine = self.engine()
        unbanded = {
            metric_id: config for metric_id, config in engine.metrics.items()
            if not config["presentation"]["bands"]
        }
        assert len(unbanded) == 9

        for metric_id, config in unbanded.items():
            reading = rr.read_metric(
                metric_id, config,
                MetricResult(value=self.probe_value(config["presentation"])),
            )

            assert reading.status == rr.BANDS_NOT_DECLARED
            assert reading.reason == config["presentation"]["bands_absent_reason"]
            assert reading.reason in reading.sentence

    def test_the_shipped_table_excludes_five_metrics_for_a_lender(self):
        """The three cached lenders — PFC, JIOFIN and EDELWEISS — all carry the
        breadcrumb sector `Finance`, verified against `raw_data/` by hand
        (tests must not read it: it is gitignored and live-scraped). What is
        asserted here is the table as it ships, keyed on that same string."""
        from boundless100x.compute_engine.sector import SectorApplicability

        engine = self.engine()
        applicability = SectorApplicability(engine.metrics)
        excluded = [
            "dupont_turnover", "dupont_equity_multiplier", "fcf_yield",
            "fcf_consistency", "dcf_margin_of_safety",
        ]
        values = [0.09, 9.37, -5.7, 2, -100.0]

        for metric_id, value in zip(excluded, values):
            reading = rr.read_metric(
                metric_id, engine.metrics[metric_id], MetricResult(value=value),
                applicability=applicability.evaluate(metric_id, "Finance"),
            )

            assert reading.status == rr.NOT_APPLICABLE, metric_id
            # The number survives and the band does not: PFC's turnover still
            # shows as 0.09x, and it is no longer called "asset-heavy" for
            # being a lender.
            assert reading.quantity is not None
            assert reading.band == ""
            assert reading.reason.strip()

    def test_a_lenders_other_metrics_still_read_normally(self):
        """The exclusions are five metrics, not a sector-wide silence."""
        from boundless100x.compute_engine.sector import SectorApplicability

        engine = self.engine()
        applicability = SectorApplicability(engine.metrics)
        reading = rr.read_metric(
            "roce_5yr_avg", engine.metrics["roce_5yr_avg"], MetricResult(value=9.1),
            applicability=applicability.evaluate("roce_5yr_avg", "Finance"),
        )

        assert reading.known
        assert reading.sentence == "9.1% — weak (higher is better)"

    def test_the_same_metric_applies_to_a_reviewed_manufacturer(self):
        from boundless100x.compute_engine.sector import SectorApplicability

        engine = self.engine()
        applicability = SectorApplicability(engine.metrics)
        reading = rr.read_metric(
            "dupont_turnover", engine.metrics["dupont_turnover"],
            MetricResult(value=1.13),
            applicability=applicability.evaluate("dupont_turnover", "Industrial Products"),
        )

        assert reading.known
        assert reading.applicability.verdict == rr.APPLIES
        assert reading.sentence == "1.13x — moderate (higher is better)"

    def test_a_whole_company_reads_in_one_call(self):
        """`read_metrics` is the batch surface the report and CLI reach for.

        The union of declared and computed, so a metric with no declaration
        reads unknown-with-reason rather than vanishing — the drill-down's
        current behaviour is to drop it, and a dropped row is invisible in a
        way a stated absence is not.
        """
        from boundless100x.compute_engine.sector import SectorApplicability

        engine = self.engine()
        readings = rr.read_metrics(
            engine.metrics,
            {
                "roce_5yr_avg": MetricResult(value=9.1),
                "dupont_turnover": MetricResult(value=0.09),
                "pe_ttm": MetricResult(error="no EPS in the accounts"),
                "a_metric_nobody_declared": MetricResult(value=3.0),
            },
            sector="Finance",
            applicability=SectorApplicability(engine.metrics),
        )

        assert len(readings) == len(engine.metrics) + 1
        assert readings["roce_5yr_avg"].known
        assert readings["dupont_turnover"].status == rr.NOT_APPLICABLE
        # `pe_ttm` is not on the Finance exclusion list, so a lender's failed
        # P/E is an ordinary computation failure and says so.
        assert readings["pe_ttm"].status == rr.METRIC_ERROR
        assert readings["a_metric_nobody_declared"].status == rr.NO_DECLARATION
        # Everything the engine declares but this company did not compute.
        assert readings["tam_runway"].status == rr.VALUE_ABSENT
        # R4 holds across the whole company, not just the metrics under test.
        assert all(r.known or r.reason.strip() for r in readings.values())

    def test_sector_mismatch_is_decided_before_the_computation_failed(self):
        """F1's ordering, on a metric that is both excluded and broken.

        `working_capital_days_trend` errors outright for PFC. When a metric is
        *also* excluded for the sector, the reader wants the sentence saying it
        would mean nothing here — why it additionally failed to compute is a
        detail that changes nothing they would do about it.
        """
        from boundless100x.compute_engine.sector import SectorApplicability

        engine = self.engine()
        applicability = SectorApplicability(engine.metrics)
        reading = rr.read_metric(
            "fcf_yield", engine.metrics["fcf_yield"],
            MetricResult(error="cashflow.csv has no cfo column"),
            applicability=applicability.evaluate("fcf_yield", "Finance"),
        )

        assert reading.status == rr.NOT_APPLICABLE
        assert "lent out" in reading.reason
        # The error is not thrown away — it is carried for anyone who needs it,
        # just not made the headline.
        assert reading.source_error == "cashflow.csv has no cfo column"
