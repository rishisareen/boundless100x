"""R6's expansion triggers — the declared contradiction pairs (U7).

The trigger this file covers is the curated one. KTD4 settled that
contradiction is a list somebody writes down rather than a detector, because
two of the three examples that motivated it were not contradictions:
`growth_quality_grade` beside its element score measures composition against
magnitude, and the P/E percentile discrepancy was a computation bug. A
sentiment-diff detector would have fired on both, on every company.

So the tests that matter most here are the ones that must **not** fire, and the
ones that turn a mis-declaration into a startup error. A pair that can never
match looks exactly like a pair whose condition is never met, and a pair that
matches something it should not makes a section longer while teaching the
reader to skim the trigger that was supposed to earn their attention.

U8 adds the section-level decision to this file.
"""

import pytest

from boundless100x.compute_engine.eligibility import (
    EligibilityEvaluator,
    effective_gates,
)
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.sector import SectorApplicability
from boundless100x.output.contradiction import (
    AGREES,
    CONTRADICTS,
    INDETERMINATE,
    NOT_DECLARED,
    PARTICIPANT_KINDS,
    ContradictionPairs,
    declared_band_labels,
    load_contradiction_pairs,
    validate_contradiction_pairs,
)
from boundless100x.output.report_reading import read_metric

# The one shipped pair, and the two readings it names.
SHIPPED_PAIR = "cheap_on_dcf_but_entry_price_gate_failed"
FAVOURABLE_DCF = 30.0        # -> "comfortable margin"
UNFAVOURABLE_DCF = -50.0     # -> "above fair value"


@pytest.fixture(scope="module")
def engine():
    return ComputeEngine()


@pytest.fixture(scope="module")
def metric_configs(engine):
    return engine.metrics


@pytest.fixture(scope="module")
def gate_specs(engine):
    """The gates that will actually be enforced, not the raw registry section.

    Resolved through `effective_gates` for the reason that function exists: an
    install with no declared gates falls back to `DEFAULT_GATES`, and
    validating a pair against the raw section would validate it against an
    empty mapping while the run enforced three gates.
    """
    return effective_gates(engine.gates)


@pytest.fixture(scope="module")
def pairs(metric_configs, gate_specs):
    """The shipped declaration, validated against the real registry.

    Construction is where validation runs, so this fixture existing at all is
    the assertion that the shipped file names no metric the engine does not
    define, no gate the registry does not declare, and no band label the metric
    does not produce.
    """
    return ContradictionPairs(metric_configs, gate_specs)


# ── Builders ──────────────────────────────────────────────────────────────


def reading_for(metric_configs, metric_id, value, *, error=None, applicability=None):
    """One `Reading`, built through the real reading layer (U6).

    Deliberately not a stub: the pair conditions are stated over declared band
    labels, so a hand-rolled reading could carry a band the registry does not
    declare and the test would pass on a pair that could never fire in a real
    report.
    """
    return read_metric(
        metric_id,
        metric_configs[metric_id],
        MetricResult(value=value, error=error),
        applicability=applicability,
    )


def gate_reading(passed, *, label="Entry price sanity", reason="because"):
    """The shape `EligibilityEvaluator.evaluate` returns, for one gate."""
    return {"gates": {"price": {"label": label, "passed": passed, "reason": reason}}}


def side(kind, participant_id, condition_key, values):
    return {"kind": kind, "id": participant_id, "when": {condition_key: values}}


GOOD_METRIC_SIDE = side(
    "metric", "dcf_margin_of_safety", "band_in", ["comfortable margin"]
)
GOOD_GATE_SIDE = side("eligibility_gate", "price", "verdict_in", ["not_met"])


def one_pair(sides, **overrides):
    entry = {
        "label": "A label",
        "reason": "a sentence that reconciles the two readings for a reader",
        "sides": sides,
    }
    entry.update(overrides)
    return {"p": entry}


# ── Firing and not firing ─────────────────────────────────────────────────


class TestTheDeclaredPairFires:
    def test_a_declared_pair_in_its_disagreeing_state_fires(
        self, pairs, metric_configs
    ):
        """The motivating instance: cheap on the cash-flow model, refused by
        the entry-price gate."""
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        outcome = pairs.evaluate(
            "dcf_margin_of_safety", readings, gate_reading(False)
        )

        assert outcome["contradicts"] is True
        assert outcome["verdict"] == CONTRADICTS
        assert outcome["reasons"] == [pairs.pairs[SHIPPED_PAIR]["reason"]]

    def test_the_fired_reason_is_the_declared_sentence_verbatim(
        self, pairs, metric_configs
    ):
        """R7 puts the declaration in front of the reader; nothing here
        paraphrases it."""
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        outcome = pairs.evaluate_pair(
            SHIPPED_PAIR, readings, gate_reading(False)
        )

        assert outcome["reason"] == pairs.pairs[SHIPPED_PAIR]["reason"]

    def test_the_fired_outcome_states_what_both_readings_actually_said(
        self, pairs, metric_configs
    ):
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        details = [
            s["detail"]
            for s in pairs.evaluate_pair(
                SHIPPED_PAIR, readings, gate_reading(False)
            )["sides"]
        ]

        assert any("comfortable margin" in d for d in details)
        assert any("was not met" in d for d in details)

    def test_no_side_detail_leaks_a_raw_id(self, pairs, metric_configs):
        """R15. `source_reason` may carry ids from `eligibility.py`'s condition
        summaries and is kept separate for exactly that reason; `detail` is the
        field a surface may render."""
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        for s in pairs.evaluate_pair(
            SHIPPED_PAIR, readings, gate_reading(False)
        )["sides"]:
            assert "dcf_margin_of_safety" not in s["detail"]
            assert "_" not in s["detail"]

    def test_the_gate_side_reads_the_real_evaluators_output(
        self, pairs, metric_configs, gate_specs
    ):
        """Pins the contract between this module and `eligibility.py`.

        The gate tri-state is read off `passed`, so a hand-built dict could
        agree with a mapping the evaluator never actually produces. This one
        goes through the evaluator: a demanding PEG on both windows fails the
        entry-price gate, with the reverse-DCF veto source available so the
        gate reads `not_met` rather than indeterminate.
        """
        eligibility = EligibilityEvaluator(gate_specs).evaluate({
            "trailing_peg": MetricResult(value=3.5),
            "peg_ratio": MetricResult(value=2.1),
            "reverse_dcf_growth": MetricResult(value=12.0),
        })
        assert eligibility["gates"]["price"]["passed"] is False

        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        assert pairs.evaluate(
            "dcf_margin_of_safety", readings, eligibility
        )["contradicts"] is True


class TestTheDeclaredPairDoesNotFire:
    def test_the_same_pair_in_agreement_does_not_fire(self, pairs, metric_configs):
        """Cheap on the model and accepted by the gate — two readings saying
        the same thing, which is the ordinary case and earns no space."""
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        outcome = pairs.evaluate(
            "dcf_margin_of_safety", readings, gate_reading(True)
        )

        assert outcome["contradicts"] is False
        assert outcome["verdict"] == AGREES
        assert outcome["reasons"] and outcome["reasons"][0].strip()

    def test_an_unfavourable_reading_beside_a_failed_gate_does_not_fire(
        self, pairs, metric_configs
    ):
        """Both readings say the price is wrong. Agreement, not a puzzle."""
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", UNFAVOURABLE_DCF
            )
        }

        assert pairs.evaluate(
            "dcf_margin_of_safety", readings, gate_reading(False)
        )["contradicts"] is False

    def test_an_indeterminate_gate_is_not_a_disagreement(
        self, pairs, metric_configs
    ):
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        assert pairs.evaluate(
            "dcf_margin_of_safety", readings, gate_reading(None)
        )["contradicts"] is False


class TestUnknownIsNeverASilentPass:
    """The rule the three sibling evaluators share: an unknown reads
    indeterminate rather than quietly agreeing."""

    def test_a_metric_that_could_not_be_read_is_indeterminate(
        self, pairs, metric_configs
    ):
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", None,
                error="cashflow series unavailable",
            )
        }

        outcome = pairs.evaluate(
            "dcf_margin_of_safety", readings, gate_reading(False)
        )

        assert outcome["contradicts"] is None
        assert outcome["verdict"] == INDETERMINATE
        assert "could not be read" in outcome["reasons"][0]

    def test_a_metric_absent_from_the_readings_is_indeterminate(self, pairs):
        outcome = pairs.evaluate("dcf_margin_of_safety", {}, gate_reading(False))

        assert outcome["contradicts"] is None
        assert "was not read" in outcome["reasons"][0]

    def test_gates_that_were_never_evaluated_are_indeterminate(
        self, pairs, metric_configs
    ):
        """`watchlist advance` re-scores with no LLM and a caller may hand over
        a result with no eligibility block at all. Absent is not 'the gate was
        fine'."""
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", FAVOURABLE_DCF
            )
        }

        outcome = pairs.evaluate("dcf_margin_of_safety", readings, None)

        assert outcome["contradicts"] is None
        assert "was not evaluated" in outcome["reasons"][0]

    def test_a_metric_the_sector_excludes_has_no_reading_to_disagree_with(
        self, pairs, metric_configs
    ):
        """The interaction with R6's first trigger. `dcf_margin_of_safety` is
        excluded for lenders, and the reading layer withholds the band rather
        than calling a lending-distorted cash flow cheap — so this pair must
        not fire a second time off a number the report has just said means
        nothing here. EDELWEISS reads +147% on this metric, which is exactly
        the favourable band, so the case is real rather than hypothetical.
        """
        applicability = SectorApplicability(set(metric_configs))
        readings = {
            "dcf_margin_of_safety": reading_for(
                metric_configs, "dcf_margin_of_safety", 147.0,
                applicability=applicability.evaluate(
                    "dcf_margin_of_safety", "Finance"
                ),
            )
        }

        outcome = pairs.evaluate(
            "dcf_margin_of_safety", readings, gate_reading(False)
        )

        assert outcome["contradicts"] is None
        assert "does not apply to a company of this kind" in outcome["reasons"][0]

    def test_every_outcome_carries_at_least_one_reason(self, pairs, metric_configs):
        """R4's posture: no verdict reaches a caller with nothing to say."""
        cases = [
            (FAVOURABLE_DCF, None, gate_reading(False)),
            (FAVOURABLE_DCF, None, gate_reading(True)),
            (None, "boom", gate_reading(False)),
            (FAVOURABLE_DCF, None, None),
        ]
        for value, error, eligibility in cases:
            readings = {
                "dcf_margin_of_safety": reading_for(
                    metric_configs, "dcf_margin_of_safety", value, error=error
                )
            }
            outcome = pairs.evaluate(
                "dcf_margin_of_safety", readings, eligibility
            )
            assert outcome["reasons"], f"{value}/{error} produced no reason"
            assert all(r.strip() for r in outcome["reasons"])

        assert pairs.evaluate("roce_5yr_avg", {}, None)["reasons"]


# ── The false positive KTD4 exists to prevent ─────────────────────────────


class TestGrowthQualityGradeNeverFires:
    """KTD4's named false positive, held down three ways.

    A naive sentiment-diff detector would compare the categorical grade to the
    Growth element score and fire on every company whose growth was fast and
    low-quality — a description that is simply true of such a company, not a
    contradiction in it. Each test below breaks a different route by which such
    a detector could be reintroduced.
    """

    def test_no_declared_pair_names_it(self, pairs, metric_configs):
        readings = {
            "growth_quality_grade": reading_for(
                metric_configs, "growth_quality_grade", "low_quality"
            )
        }

        outcome = pairs.evaluate(
            "growth_quality_grade", readings, gate_reading(False)
        )

        assert pairs.pairs_for("growth_quality_grade") == []
        assert outcome["contradicts"] is False
        assert outcome["verdict"] == NOT_DECLARED
        assert "No declared contradiction pair" in outcome["reasons"][0]

    def test_not_declared_is_kept_distinct_from_agreement(
        self, pairs, metric_configs
    ):
        """Both are False for R6, and they are different facts. Collapsing them
        would hide how thin the declared coverage is behind a page of confident
        agreement."""
        assert (
            pairs.evaluate("growth_quality_grade", {}, gate_reading(False))["verdict"]
            != AGREES
        )

    def test_declaring_it_in_a_pair_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        """The structural half. It declares no interpretation bands, so there
        is no band a condition could name — a later attempt to wire the
        detector back in as a declaration fails at load rather than shipping.
        """
        assert declared_band_labels(metric_configs["growth_quality_grade"]) == set()

        table = one_pair([
            side("metric", "growth_quality_grade", "band_in", ["low_quality"]),
            GOOD_GATE_SIDE,
        ])

        with pytest.raises(ValueError, match="declares no interpretation bands"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_an_element_score_is_not_a_participant_kind(
        self, metric_configs, gate_specs
    ):
        """The other half of the same detector: the score it would have been
        diffed against is not in the closed vocabulary, so the pair cannot be
        expressed at all."""
        assert "element_score" not in PARTICIPANT_KINDS

        table = one_pair([
            {"kind": "element_score", "id": "growth", "when": {"band_in": ["low"]}},
            GOOD_METRIC_SIDE,
        ])

        with pytest.raises(ValueError, match="unknown participant kind"):
            ContradictionPairs(metric_configs, gate_specs, table)


# ── Validation ────────────────────────────────────────────────────────────


class TestValidation:
    """Every one of these is a startup error rather than a log line, for the
    reason the sector table and the trigger registry both give: a rule that can
    never fire is indistinguishable from a rule whose condition is never met.
    """

    def test_a_metric_the_registry_does_not_define_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        table = one_pair([
            side("metric", "dcf_margin_of_saftey", "band_in", ["comfortable margin"]),
            GOOD_GATE_SIDE,
        ])

        with pytest.raises(ValueError, match="unknown metric id"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_zero_weight_metric_is_rejected_at_load(
        self, metric_configs, gate_specs
    ):
        """KTD5. Expansion is prominence, and a signal that deliberately cannot
        move a score must not move the report's shape instead."""
        table = one_pair([
            side("metric", "rerating_headroom", "band_in", ["favourable"]),
            GOOD_GATE_SIDE,
        ])

        with pytest.raises(ValueError, match="carries zero weight"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_every_zero_weight_metric_is_rejected(
        self, engine, metric_configs, gate_specs
    ):
        """Derived from the registry rather than a hardcoded list of ids — the
        mechanical form the forward-signals rule was rewritten into after a
        remembered rule let one through."""
        zero_weight = [
            metric_id
            for metric_id, config in metric_configs.items()
            if not engine._scored(config)
        ]
        assert zero_weight, "the registry has stopped carrying forward signals"

        for metric_id in zero_weight:
            labels = declared_band_labels(metric_configs[metric_id])
            # Either rejection is correct and both are KTD5-safe; a bandless
            # signal simply trips the earlier rule.
            table = one_pair([
                side("metric", metric_id, "band_in", sorted(labels) or ["anything"]),
                GOOD_GATE_SIDE,
            ])
            with pytest.raises(
                ValueError, match="carries zero weight|declares no interpretation"
            ):
                ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_band_label_the_metric_does_not_declare_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        table = one_pair([
            side("metric", "dcf_margin_of_safety", "band_in", ["screaming bargain"]),
            GOOD_GATE_SIDE,
        ])

        with pytest.raises(ValueError, match="declares no band"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_an_unknown_participant_kind_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        """The participant vocabulary is closed. An open string field would let
        a typo'd kind sit here forever, never matching."""
        table = one_pair([
            {"kind": "metrics", "id": "dcf_margin_of_safety", "when": {}},
            GOOD_GATE_SIDE,
        ])

        with pytest.raises(ValueError, match="unknown participant kind"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_an_unknown_gate_id_is_a_startup_error(self, metric_configs, gate_specs):
        table = one_pair([
            GOOD_METRIC_SIDE,
            side("eligibility_gate", "valuation", "verdict_in", ["not_met"]),
        ])

        with pytest.raises(ValueError, match="unknown eligibility gate"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_gate_verdict_outside_the_tri_state_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        table = one_pair([
            GOOD_METRIC_SIDE,
            side("eligibility_gate", "price", "verdict_in", ["failed"]),
        ])

        with pytest.raises(ValueError, match="is not a gate verdict"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_condition_key_from_the_wrong_kind_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        """One kind, one condition key. A `band_in` on a gate would read as a
        declaration nobody could satisfy."""
        table = one_pair([
            GOOD_METRIC_SIDE,
            {"kind": "eligibility_gate", "id": "price", "when": {"band_in": ["x"]}},
        ])

        with pytest.raises(ValueError, match="unknown condition key"):
            ContradictionPairs(metric_configs, gate_specs, table)

    @pytest.mark.parametrize("when", [None, {}, {"verdict_in": []}, "not_met"])
    def test_a_malformed_condition_is_a_startup_error(
        self, metric_configs, gate_specs, when
    ):
        table = one_pair([
            GOOD_METRIC_SIDE,
            {"kind": "eligibility_gate", "id": "price", "when": when},
        ])

        with pytest.raises(ValueError):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_blank_reason_is_a_startup_error(self, metric_configs, gate_specs):
        """R7 makes the reconciling sentence the deliverable; a pair that fires
        with nothing to say expands a section and then shrugs."""
        table = one_pair([GOOD_METRIC_SIDE, GOOD_GATE_SIDE], reason="   ")

        with pytest.raises(ValueError, match="reason that reconciles"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_blank_label_is_a_startup_error(self, metric_configs, gate_specs):
        table = one_pair([GOOD_METRIC_SIDE, GOOD_GATE_SIDE], label="")

        with pytest.raises(ValueError, match="label naming the disagreement"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_misspelled_entry_key_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        """`reasson:` would otherwise ship a pair with a blank reason and an
        extra key nobody reads."""
        table = one_pair([GOOD_METRIC_SIDE, GOOD_GATE_SIDE], reasson="typo")

        with pytest.raises(ValueError, match="unknown key"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_misspelled_side_key_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        table = one_pair([
            {
                "kind": "metric",
                "id": "dcf_margin_of_safety",
                "when": {"band_in": ["comfortable margin"]},
                "wen": {"band_in": ["at fair value"]},
            },
            GOOD_GATE_SIDE,
        ])

        with pytest.raises(ValueError, match="unknown key"):
            ContradictionPairs(metric_configs, gate_specs, table)

    @pytest.mark.parametrize("sides", [
        [GOOD_METRIC_SIDE],
        [GOOD_METRIC_SIDE, GOOD_GATE_SIDE, GOOD_METRIC_SIDE],
        "both of them",
    ])
    def test_a_pair_that_is_not_two_sides_is_a_startup_error(
        self, metric_configs, gate_specs, sides
    ):
        with pytest.raises(ValueError, match="exactly two sides"):
            ContradictionPairs(metric_configs, gate_specs, one_pair(sides))

    def test_a_reading_paired_with_itself_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        table = one_pair([GOOD_METRIC_SIDE, dict(GOOD_METRIC_SIDE)])

        with pytest.raises(ValueError, match="cannot disagree with itself"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_pair_with_no_metric_side_is_a_startup_error(
        self, metric_configs, gate_specs
    ):
        """R6 fires this trigger on a metric, so a pair naming none could never
        reach a section — dead on arrival rather than merely rare."""
        table = one_pair([
            GOOD_GATE_SIDE,
            side("eligibility_gate", "size", "verdict_in", ["not_met"]),
        ])

        with pytest.raises(ValueError, match="at least one side must be a metric"):
            ContradictionPairs(metric_configs, gate_specs, table)

    def test_a_non_mapping_entry_is_a_startup_error(self, metric_configs, gate_specs):
        with pytest.raises(ValueError, match="entry must be a mapping"):
            ContradictionPairs(metric_configs, gate_specs, {"p": ["nope"]})

    def test_an_unreadable_file_declares_nothing_rather_than_firing(
        self, metric_configs, gate_specs, tmp_path
    ):
        """The safe direction. A lost declaration costs the trigger its signal,
        loudly; it never invents a disagreement, and the other two triggers in
        R6 are evaluated independently of it."""
        evaluator = ContradictionPairs(
            metric_configs,
            gate_specs,
            load_contradiction_pairs(str(tmp_path / "absent.yaml")),
        )

        assert evaluator.pairs == {}
        assert evaluator.evaluate(
            "dcf_margin_of_safety", {}, gate_reading(False)
        )["contradicts"] is False


# ── The shipped declaration ───────────────────────────────────────────────


class TestShippedDeclaration:
    def test_the_shipped_file_validates_against_the_real_registry(
        self, metric_configs, gate_specs
    ):
        errors = validate_contradiction_pairs(
            load_contradiction_pairs(), metric_configs, gate_specs
        )

        assert errors == []

    def test_the_shipped_file_declares_the_one_surviving_instance(self, pairs):
        """One entry, on purpose. KTD4 bounds coverage to what somebody wrote
        down, and padding the list to justify the trigger would defeat the
        reason it is curated rather than detected. If this count changes, the
        new entry needs its own firing and non-firing tests above."""
        assert list(pairs.pairs) == [SHIPPED_PAIR]

    def test_every_declared_reason_reconciles_rather_than_labels(self, pairs):
        """Not a length check for its own sake: "these disagree" passes a
        non-blank test and leaves the reader with the same two numbers and no
        way to choose between them."""
        for pair_id, entry in pairs.pairs.items():
            assert len(entry["reason"].split()) >= 25, f"{pair_id}: too terse"

    def test_no_shipped_side_names_a_zero_weight_metric(
        self, engine, metric_configs, pairs
    ):
        """KTD5, read off the registry rather than off a remembered list."""
        for pair_id, entry in pairs.pairs.items():
            for s in entry["sides"]:
                if s["kind"] != "metric":
                    continue
                assert engine._scored(metric_configs[s["id"]]), (
                    f"{pair_id} names zero-weight {s['id']}"
                )
