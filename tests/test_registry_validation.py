"""Registry validation: what a metric must declare, and what it is excused.

Two rules live here. The older one is about **scoring**: a metric at
`weight: 0` is display-only — the scorer returns from its `weight == 0` branch
before `_compute_raw_score` is reached, so thresholds, ranges and categories
declared for it are never read. Demanding them anyway produced exactly what
you would expect: zero-weight metrics carrying invented category tables purely
to get past startup.

The newer one is about **presentation** (R11, R12, R3) and runs the other way:
every metric must declare its unit, its direction of goodness, one plain line
on what it measures, and its interpretation bands — and a zero-weight metric
needs that more than a scored one, not less, because the number and its band
are the only reading it will ever get. A missing block is a startup error, the
same class of error as a duplicate metric id, because the failure it prevents
is silent: a value column holding `25.7`, `0.09`, `2.0` and `0.84` with
nothing to say which is a percentage, which a ratio, which a count of years.
"""

import pytest

from boundless100x.compute_engine.metrics.validator import (
    PRESENTATION_DIRECTIONS,
    PRESENTATION_UNITS,
    validate_registry,
)

PRESENTATION = {
    "unit": "percent",
    "direction": "higher_is_better",
    "meaning": "What the thing is, and what good looks like.",
    "bands": [[20.0, "strong"], [12.0, "adequate"]],
    "low_label": "weak",
}


def metric(weight, presentation=PRESENTATION, **scoring):
    return {
        "m": {
            "_source_file": "test.yaml",
            "name": "M",
            "module": "builtin.profitability",
            "function": "compute_roce_avg",
            "inputs": ["ratios"],
            "scoring": {"weight": weight, **scoring},
            "display": {"format": "{}", "section": "s"},
            # Not copied: some tests pass a deliberately malformed block that
            # is not a mapping at all, which is the case being validated.
            "presentation": presentation,
        }
    }


def presented(**overrides):
    """A metric whose presentation block is the default with edits applied.

    A key set to `None` is removed, so a test can say "this block is missing
    its low_label" without restating the other four fields.

    Weight is 0 so the scoring-config checks stay out of the way — every test
    reaching for this helper is asking about presentation, and a scored metric
    would need thresholds and a direction it has nothing to say about.
    """
    block = {**PRESENTATION, **overrides}
    return metric(0.0, presentation={k: v for k, v in block.items() if v is not None})


class TestScoredMetricsMustDeclareTheirScoring:
    def test_a_weighted_threshold_metric_needs_thresholds_and_direction(self):
        errors = validate_registry(metric(0.1, mode="threshold"))
        assert any("thresholds" in e for e in errors)
        assert any("direction" in e for e in errors)

    def test_a_weighted_range_metric_needs_its_range(self):
        assert validate_registry(metric(0.1, mode="range_optimal"))

    def test_a_weighted_categorical_metric_needs_its_categories(self):
        assert validate_registry(metric(0.1, mode="categorical"))


class TestZeroWeightMetricsAreExcusedOnlyFromDeadConfig:
    def test_no_thresholds_are_required(self):
        assert validate_registry(metric(0.0)) == []

    def test_no_categories_are_required(self):
        assert validate_registry(metric(0.0, mode="categorical")) == []

    def test_weight_is_still_required(self):
        definition = metric(0.0)
        del definition["m"]["scoring"]["weight"]
        assert any("weight" in e for e in validate_registry(definition))

    def test_an_unrecognised_mode_is_still_an_error(self):
        """A typo is a typo whatever the weight."""
        assert any("invalid mode" in e for e in validate_registry(metric(0.0, mode="nonsense")))

    def test_the_structural_fields_are_still_required(self):
        definition = metric(0.0)
        del definition["m"]["function"]
        assert any("function" in e for e in validate_registry(definition))

    def test_presentation_is_not_among_the_things_it_is_excused(self):
        """The excusal is about *scoring* config, which nothing reads at
        weight 0. Presentation is the opposite case: it is the only thing a
        reader gets, so it is required exactly as hard here (R8)."""
        definition = metric(0.0)
        del definition["m"]["presentation"]
        assert any("presentation" in e for e in validate_registry(definition))


class TestEveryMetricMustDeclareItsPresentation:
    def test_a_missing_block_is_an_error(self):
        definition = metric(0.1)
        del definition["m"]["presentation"]
        assert any("missing 'presentation'" in e for e in validate_registry(definition))

    def test_a_missing_block_stops_the_engine_at_construction(self, tmp_path):
        """The same class of startup failure as a duplicate metric id.

        Catching it here rather than at render time is the whole point: a
        metric added without a declaration would otherwise reach a reader as
        a bare number, and a bare number in this table is unreadable.
        """
        import shutil

        import yaml

        from boundless100x.compute_engine.engine import ComputeEngine

        registry = tmp_path / "metrics"
        shutil.copytree(ComputeEngine().registry_dir, registry)
        target = registry / "elements" / "quality_business.yaml"
        config = yaml.safe_load(target.read_text())
        config["metrics"]["roce_5yr_avg"].pop("presentation")
        target.write_text(yaml.safe_dump(config))

        with pytest.raises(ValueError, match="Registry validation failed"):
            ComputeEngine(registry_dir=str(registry))

    def test_a_block_that_is_not_a_mapping_is_an_error(self):
        assert any(
            "must be a mapping" in e
            for e in validate_registry(metric(0.1, presentation=["percent"]))
        )

    def test_an_unknown_unit_is_an_error(self):
        """The vocabulary is closed for the reason KTD5 gives about extraction
        units: left open, `pct`, `percent` and `%` all appear and the field
        stops being something anyone can group on."""
        assert any(
            "presentation.unit" in e for e in validate_registry(presented(unit="pct"))
        )

    def test_an_unknown_direction_is_an_error(self):
        assert any(
            "presentation.direction" in e
            for e in validate_registry(presented(direction="up"))
        )

    def test_a_missing_meaning_is_an_error(self):
        """R3: the explanation must exist for every metric before any later
        unit can make it reachable."""
        assert any(
            "presentation.meaning" in e for e in validate_registry(presented(meaning=None))
        )

    def test_a_blank_meaning_is_an_error(self):
        assert any(
            "presentation.meaning" in e for e in validate_registry(presented(meaning="  "))
        )


class TestPresentationMustNotContradictScoring:
    """One metric, one direction.

    A metric that told the scorer `lower_is_better` and the reader
    `higher_is_better` would render a reading contradicting its own score, on
    every company, with nothing anywhere saying so.
    """

    def test_a_direction_disagreeing_with_the_scoring_config_is_an_error(self):
        errors = validate_registry(
            metric(
                0.1,
                presentation={**PRESENTATION, "direction": "lower_is_better"},
                mode="threshold",
                thresholds=[1, 2, 3],
                direction="higher_is_better",
            )
        )
        assert any("contradicts scoring.direction" in e for e in errors)

    def test_agreement_passes(self):
        errors = validate_registry(
            metric(
                0.1,
                mode="threshold",
                thresholds=[1, 2, 3],
                direction="higher_is_better",
            )
        )
        assert errors == []

    def test_a_trend_direction_is_outside_the_rule(self):
        """`declining_is_better` describes the *series* the scorer reads, not
        the level on the page, so the two are allowed to differ — and
        `working_capital_days_trend` in the shipped registry does exactly
        that."""
        errors = validate_registry(
            metric(
                0.1,
                presentation={**PRESENTATION, "direction": "lower_is_better"},
                mode="trend_direction",
                direction="declining_is_better",
            )
        )
        assert errors == []


class TestBandsMustBeWalkable:
    """The walk takes the first threshold a value reaches, so order is not
    cosmetic — an ascending list makes every band but the first unreachable
    and renders a wrong reading on every company."""

    def test_ascending_thresholds_are_an_error(self):
        errors = validate_registry(
            presented(bands=[[12.0, "adequate"], [20.0, "strong"]])
        )
        assert any("unreachable" in e for e in errors)

    def test_repeated_thresholds_are_an_error(self):
        errors = validate_registry(
            presented(bands=[[20.0, "strong"], [20.0, "also strong"]])
        )
        assert any("unreachable" in e for e in errors)

    def test_a_non_numeric_threshold_is_an_error(self):
        errors = validate_registry(presented(bands=[["high", "strong"]]))
        assert any("is not a number" in e for e in errors)

    def test_a_blank_label_is_an_error(self):
        errors = validate_registry(presented(bands=[[20.0, ""]]))
        assert any("non-empty string" in e for e in errors)

    def test_a_malformed_pair_is_an_error(self):
        errors = validate_registry(presented(bands=[[20.0]]))
        assert any("[threshold, label] pair" in e for e in errors)

    def test_bands_must_be_a_list(self):
        for not_a_list in ({"20": "strong"}, 1, "strong"):
            errors = validate_registry(presented(bands=not_a_list))
            assert any("must be a list" in e for e in errors), not_a_list

    def test_declared_bands_need_a_low_label(self):
        errors = validate_registry(presented(low_label=None))
        assert any("low_label" in e for e in errors)

    def test_descending_thresholds_pass_whichever_way_goodness_runs(self):
        """For a `lower_is_better` metric the thresholds still descend; only
        the labels run worst-first, with `low_label` naming the best outcome.
        `debt_equity` is the shipped example."""
        errors = validate_registry(
            metric(
                0.1,
                presentation={
                    **PRESENTATION,
                    "direction": "lower_is_better",
                    "bands": [[0.5, "leveraged"], [0.1, "modest debt"]],
                    "low_label": "virtually debt-free",
                },
                mode="threshold",
                thresholds=[2.0, 1.0, 0.5],
                direction="lower_is_better",
            )
        )
        assert errors == []


class TestAnUndeclaredBandMustSayWhy:
    """An empty `bands` list is a legitimate declaration — a categorical
    metric's value is already its own reading, and a sector-relative multiple
    has no honest absolute band. A wrong band is worse than a declared
    unknown, so what is required is the *reason*, which is what the row
    renders in place of a reading (AE4)."""

    def test_no_bands_and_no_reason_is_an_error(self):
        errors = validate_registry(presented(bands=[], low_label=None))
        assert any("bands_absent_reason" in e for e in errors)

    def test_no_bands_with_a_reason_passes(self):
        errors = validate_registry(
            presented(bands=[], low_label=None, bands_absent_reason="Sector-relative.")
        )
        assert errors == []

    def test_a_low_label_with_no_bands_is_an_error(self):
        errors = validate_registry(
            presented(bands=[], bands_absent_reason="Sector-relative.")
        )
        assert any("names a reading nothing can produce" in e for e in errors)

    def test_bands_and_a_reason_together_are_an_error(self):
        errors = validate_registry(presented(bands_absent_reason="Sector-relative."))
        assert any("only one of them can be true" in e for e in errors)


class TestCategoricalMetricsCannotBeBanded:
    """The band walk compares with `>=`, which a string value never satisfies:
    a numeric band on a category-valued metric renders blank rather than
    erroring. The ranking of its grades belongs in the scoring config's
    `categories` table, stated once."""

    def test_a_category_with_numeric_bands_is_an_error(self):
        errors = validate_registry(presented(unit="category", direction="not_directional"))
        assert any("already its own reading" in e for e in errors)

    def test_a_category_with_a_monotone_direction_is_an_error(self):
        errors = validate_registry(
            presented(
                unit="category",
                direction="higher_is_better",
                bands=[],
                low_label=None,
                bands_absent_reason="A grade.",
            )
        )
        assert any("names have no ordering" in e for e in errors)


class TestTheShippedRegistryIsSound:
    def test_it_loads_without_errors(self):
        from boundless100x.compute_engine.engine import ComputeEngine

        assert validate_registry(ComputeEngine().metrics) == []


# ── Coverage, asked of the registry rather than of a list ──
#
# The expected set is derived by introspection, the way
# `tests/test_report_forward_signals.py` derives the zero-weight flag set. A
# hardcoded roll of fifty-seven ids would go stale the first time a metric is
# added and would then be silently short one metric forever — the exact
# failure mode KTD6 records from Phase 3.


def shipped_metrics() -> dict[str, dict]:
    from boundless100x.compute_engine.engine import ComputeEngine

    return ComputeEngine().metrics


class TestEveryShippedMetricDeclaresItsPresentation:
    def test_none_is_missing_the_block(self):
        missing = [mid for mid, c in shipped_metrics().items() if "presentation" not in c]
        assert missing == [], f"metrics with no presentation declaration: {missing}"

    def test_every_one_declares_a_unit_a_direction_and_a_meaning(self):
        """R11 and R12 in one assertion: no number can reach a reader without
        its unit and its direction, and R3's explanation must exist for all of
        them before a later unit can make it reachable."""
        for mid, config in shipped_metrics().items():
            block = config["presentation"]
            assert block["unit"] in PRESENTATION_UNITS, mid
            assert block["direction"] in PRESENTATION_DIRECTIONS, mid
            assert block["meaning"].strip(), mid

    def test_a_metric_without_bands_says_why(self):
        for mid, config in shipped_metrics().items():
            block = config["presentation"]
            if not block["bands"]:
                assert block.get("bands_absent_reason", "").strip(), mid

    def test_the_vocabularies_carry_nothing_the_registry_does_not_use(self):
        """A unit vocabulary is built from what is actually measured, not from
        first principles — the rule KTD5 arrived at for extraction units after
        `usd_tn` and `inr_lakh_cr` had to be added by meeting them. A member
        no metric uses is a reading nobody can check, and it invites the next
        author to reach for it instead of the right one."""
        blocks = [c["presentation"] for c in shipped_metrics().values()]
        assert set(PRESENTATION_UNITS) == {b["unit"] for b in blocks}
        assert set(PRESENTATION_DIRECTIONS) == {b["direction"] for b in blocks}


class TestDeclaredBandsResolveThroughTheExistingWalk:
    """The declarations must be readable by `_forward_band`, which is the walk
    the validator's docstring names and the one the four zero-weight signals
    have been rendered through since Phase 2. Testing against it rather than
    against a re-implementation is what makes the declaration usable rather
    than merely well-formed."""

    @staticmethod
    def band(metric_id):
        from boundless100x.output.report_generator import ReportGenerator

        block = shipped_metrics()[metric_id]["presentation"]
        return block, ReportGenerator._forward_band

    def test_a_known_value_lands_in_the_expected_band(self):
        block, walk = self.band("roce_5yr_avg")
        assert walk(block, 31.0) == "exceptional"
        assert walk(block, 18.0) == "solid"
        assert walk(block, 8.0) == "weak"

    def test_a_lower_is_better_metric_reads_the_right_way_round(self):
        """The thresholds still descend; it is the labels that run worst-first,
        so `low_label` is the good news rather than the bad."""
        block, walk = self.band("debt_equity")
        assert block["direction"] == "lower_is_better"
        assert walk(block, 1.8) == "leveraged"
        assert walk(block, 0.35) == "modest debt"
        assert walk(block, 0.02) == "virtually debt-free"

    def test_every_banded_metric_resolves_its_extremes(self):
        """Derived, so a metric added later is covered without being listed.

        A value above the top threshold must return the top label and a value
        below the bottom one must return `low_label`. That is weak as a claim
        about any single metric and strong as a claim about all of them: it
        fails the moment a band list is ordered so that some band is
        unreachable, or a label is not the string the walk returns.
        """
        for mid, config in shipped_metrics().items():
            block = config["presentation"]
            if not block["bands"]:
                continue
            _, walk = self.band(mid)
            top_threshold, top_label = block["bands"][0]
            bottom_threshold, _ = block["bands"][-1]
            assert walk(block, top_threshold + 1) == top_label, mid
            assert walk(block, bottom_threshold - 1) == block["low_label"], mid

    def test_a_category_valued_metric_is_never_walked(self):
        """`_forward_band` returns "" for a non-numeric value, so a categorical
        metric must not be carrying bands for it to walk in the first place —
        the validator refuses that combination, and this is the registry-side
        half of the same claim."""
        for mid, config in shipped_metrics().items():
            block = config["presentation"]
            if block["unit"] == "category":
                assert block["bands"] == [], mid
                assert block["direction"] == "not_directional", mid
