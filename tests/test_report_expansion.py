"""R6's expansion triggers, and the section-level decision built on them.

Two units live here. The first half covers U7's declared contradiction pairs;
the second half covers U8's `report_expansion`, which evaluates all three
triggers in F1's order and ORs them across a section.

The trigger the first half covers is the curated one. KTD4 settled that
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

U8's half turns on two properties that are easy to get subtly wrong. Order:
sector mismatch is decided first and is *not* subject to the corpus-relative
suppression, so a metric that is both inapplicable and a corpus-wide zero must
still expand. And direction: an unknown trigger condition must not fire — it
would expand everything and therefore say nothing about anyone — while an
unknown *suppression* must not suppress, because suppression is the mechanism
that hides a real gap. Those two pull opposite ways on purpose, and each has
its own test below.
"""

import json
from pathlib import Path

import pytest

from boundless100x.compute_engine.eligibility import (
    EligibilityEvaluator,
    effective_gates,
)
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.scorer import SQGLPScorer
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
from boundless100x.output.report_expansion import (
    CONTRADICTION,
    DEFAULT_REPORTS_DIR,
    MIN_COMPARABLE_REPORTS,
    MIN_WEIGHT_SHARE,
    SECTOR_MISMATCH,
    TRIGGER_LABELS,
    TRIGGERS,
    ZERO_SCORE_GAP,
    ZERO_SCORE_NOT_COMPARABLE,
    ExpansionDecider,
    ExpansionReason,
    ScoredCorpus,
    expanded_sections,
    load_scored_corpus,
    section_applicability_line,
)
from boundless100x.output.report_reading import read_metric, read_metrics
from tests.conftest import latest_scores_for

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


# ══════════════════════════════════════════════════════════════════════════
# U8 — the section-level expansion decision
# ══════════════════════════════════════════════════════════════════════════

# Three sectors, chosen for what the applicability table says about each.
LENDER = "Finance"                    # reviewed, five metrics excluded
MANUFACTURER = "Industrial Products"  # reviewed, nothing excluded
UNREVIEWED = "Power"                  # in no entry at all — indeterminate

# Metrics picked for their weight share rather than their meaning: one heavy
# enough that a zero on it fires R6's trigger, one far too light to.
#
# **Derived, not written down**, for the reason
# `test_the_weight_share_is_of_the_elements_declared_total` states in its own
# docstring: an element gaining a metric moves every share in it. These were
# hardcoded to `roiic` at exactly 0.10 of a Quality — Business that summed to
# exactly 1.0, and adding `roa_5yr_avg` to the element moved `roiic` to 9.1%
# — every test below then read as a claim about the trigger when it was a
# claim about that coincidence. The exact-boundary case that assumption also
# used to cover is now tested directly, on a registry built to have one, in
# `TestTheZeroScoreTrigger`.
BAR_ELEMENT = "quality_business"


def _pick_bar_metrics() -> tuple[str, str]:
    """(heaviest-under-the-bar, lightest-at-or-above-it) within BAR_ELEMENT."""
    metrics = ComputeEngine().metrics
    shares = {}
    total = sum(
        (c.get("scoring") or {}).get("weight", 0) or 0
        for c in metrics.values()
        if c["element"] == BAR_ELEMENT
    )
    for metric_id, config in metrics.items():
        if config["element"] != BAR_ELEMENT:
            continue
        weight = (config.get("scoring") or {}).get("weight", 0) or 0
        if weight > 0:
            shares[metric_id] = weight / total

    at = min(
        (m for m, s in shares.items() if s >= MIN_WEIGHT_SHARE),
        key=lambda m: (shares[m], m),
    )
    under = max(
        (m for m, s in shares.items() if s < MIN_WEIGHT_SHARE),
        key=lambda m: (-shares[m], m),
    )
    return at, under


AT_THE_BAR, UNDER_THE_BAR = _pick_bar_metrics()


def lender_exclusions_in(element: str) -> set[str]:
    """The metrics the shipped table withdraws from a lender, in one element.

    Read off the table for the same reason the bar metrics above are derived:
    these tests are about whether the mismatch trigger fires and names what it
    withdrew, not about the size of the table on the day they were written.
    """
    metrics = ComputeEngine().metrics
    excluded = SectorApplicability(set(metrics)).not_applicable_metrics(LENDER)
    return {
        metric_id for metric_id in excluded
        if metrics[metric_id]["element"] == element
    }


# ── Builders ──────────────────────────────────────────────────────────────


def write_reports(root: Path, reports: dict) -> Path:
    """A corpus on disk, in the shape `ReportGenerator` writes.

    Tests never read the real `output/reports/` directory: it is gitignored and
    machine-local, so a suite that depended on it would pass here and fail in a
    checkout that had never run `analyze`. The one test that does look at it
    skips when it is absent, and observes rather than asserts.
    """
    for name, details in reports.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scores.json").write_text(json.dumps({
            "details": {mid: {"score": score} for mid, score in details.items()}
        }))
    return root


def corpus_from(root: Path, reports: dict, *, minimum=MIN_COMPARABLE_REPORTS):
    return load_scored_corpus(write_reports(root, reports), minimum=minimum)


def corpus_where(root: Path, metric_id: str, *, zero: int, comparable: int,
                 errored: int = 0, minimum=MIN_COMPARABLE_REPORTS):
    """A corpus in which `metric_id` reads zero in `zero` of `comparable`.

    `errored` adds companies whose reading was `None` — present in the corpus,
    absent from both sides of the rate.
    """
    reports = {
        f"T{index}_20260808": {metric_id: 0.0 if index < zero else 1.0}
        for index in range(comparable)
    }
    reports.update({
        f"E{index}_20260808": {metric_id: None} for index in range(errored)
    })
    return corpus_from(root, reports, minimum=minimum)


def readings_for(metric_configs, values=None, *, sector=MANUFACTURER, errors=None):
    """Readings through the real U6 layer and the real U4 table.

    Not stubs, for the reason U7's builder gives: the decision reads
    `Reading.applicability`, and a hand-built reading could carry a verdict the
    shipped table never produces.
    """
    errors = errors or {}
    results = {
        metric_id: MetricResult(value=value, error=errors.get(metric_id))
        for metric_id, value in (values or {}).items()
    }
    for metric_id, error in errors.items():
        results.setdefault(metric_id, MetricResult(value=None, error=error))
    return read_metrics(
        metric_configs, results,
        sector=sector, applicability=SectorApplicability(set(metric_configs)),
    )


def scores_for(scored: dict, *, coverage=None):
    """The block `SQGLPScorer.score` returns, cut down to what U8 reads."""
    return {
        "details": {mid: {"score": score} for mid, score in scored.items()},
        "coverage": {"elements": coverage or {}},
    }


@pytest.fixture
def decide(metric_configs, pairs):
    """One company, one corpus, one decision."""
    def _decide(corpus, *, sector=MANUFACTURER, values=None, errors=None,
                scored=None, eligibility=None, element=None, coverage=None):
        readings = readings_for(metric_configs, values, sector=sector, errors=errors)
        scores = scores_for(scored or {}, coverage=coverage)
        decider = ExpansionDecider(metric_configs, pairs, corpus)
        sections = decider.evaluate(
            readings, scores, eligibility=eligibility,
            elements=None if element is None else [element],
        )
        # One section or all of them, but always through `evaluate`, so the
        # R18 coverage clause is derived the way the real caller gets it rather
        # than handed in by the test.
        return sections if element is None else sections[element]
    return _decide


# ── The corpus rule (R8, KTD5) ────────────────────────────────────────────


class TestTheCorpusRule:
    """What is counted, what is not, and what an absence resolves to."""

    def test_only_the_latest_report_per_ticker_is_counted(self, tmp_path):
        """A company re-analysed weekly must not get one vote per run.

        Counting directories would let a single ticker carry a majority in a
        test whose entire claim is that several *companies* read the same way.
        """
        corpus = corpus_from(tmp_path / "c", {
            "AAA_20260101": {AT_THE_BAR: 0.0},   # superseded
            "AAA_20260808": {AT_THE_BAR: 1.0},
            "BBB_20260808": {AT_THE_BAR: 0.0},
        })

        assert corpus.reports == 2
        assert corpus.tickers == ("AAA", "BBB")
        rate = corpus.rate_for(AT_THE_BAR)
        assert (rate.zero, rate.comparable) == (1, 2)

    def test_a_metric_that_errored_is_absent_from_numerator_and_denominator(
        self, tmp_path
    ):
        """R18's wording, and it has to be both sides.

        Dropped from the numerator alone, an element whose metrics mostly error
        would read as a metric that mostly does not come out zero, and the
        suppression rule would quietly stop suppressing.
        """
        corpus = corpus_from(tmp_path / "c", {
            "AAA_20260808": {AT_THE_BAR: 0.0},
            "BBB_20260808": {AT_THE_BAR: None},
            "CCC_20260808": {AT_THE_BAR: None},
        })

        rate = corpus.rate_for(AT_THE_BAR)
        assert (rate.zero, rate.comparable) == (1, 1)
        assert rate.share == 1.0

    @pytest.mark.parametrize("zero,comparable,expected", [
        (4, 6, True),
        (3, 6, False),   # exactly half is not a majority
        (4, 7, True),
        (3, 7, False),
        (6, 6, True),
        (0, 6, False),
    ])
    def test_a_simple_majority_is_strictly_more_than_half(
        self, tmp_path, zero, comparable, expected
    ):
        """KTD5. Three of six is the live case: `ebit_cagr_3yr`,
        `roce_consistency` and `analyst_coverage` all sit exactly there in the
        measured corpus, and a `>=` would suppress all three."""
        corpus = corpus_where(
            tmp_path / "c", AT_THE_BAR, zero=zero, comparable=comparable
        )

        assert corpus.rate_for(AT_THE_BAR).suppresses is expected

    def test_a_corpus_below_the_minimum_cannot_suppress(self, tmp_path):
        """Five of five is unanimous and still says nothing — six comparable
        readings is where dropping any one company stops changing the answer."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=5, comparable=5)

        assert corpus.rate_for(AT_THE_BAR).suppresses is None

    def test_the_minimum_counts_readings_of_this_metric_not_reports_on_disk(
        self, tmp_path
    ):
        """A fifty-report corpus in which this metric computed twice still
        cannot say whether its zero is unusual. The minimum gates the
        denominator of the test being run, not the size of the folder."""
        corpus = corpus_where(
            tmp_path / "c", AT_THE_BAR, zero=2, comparable=2, errored=20
        )

        assert corpus.reports == 22
        assert corpus.rate_for(AT_THE_BAR).suppresses is None

    def test_an_unreadable_corpus_reads_as_below_the_minimum_never_as_clean(
        self, tmp_path
    ):
        """The dangerous direction, named. `False` would mean "asked, and this
        is not a corpus-wide zero" — a claim of corpus evidence nobody has —
        and would let a missing folder hide every real gap on every company."""
        corpus = load_scored_corpus(tmp_path / "never-generated")

        assert corpus.reports == 0
        assert corpus.error and "could not be read" in corpus.error
        assert corpus.rate_for(AT_THE_BAR).suppresses is None

    def test_the_corpus_error_is_prose_rather_than_an_exception_string(
        self, tmp_path
    ):
        """R15. This sentence reaches a reader inside a fired reason."""
        corpus = load_scored_corpus(tmp_path / "never-generated")

        assert "Errno" not in corpus.error
        assert "FileNotFound" not in corpus.error
        assert corpus.error[0].islower() or corpus.error[0].isalpha()

    def test_one_malformed_report_does_not_cost_the_whole_corpus(self, tmp_path):
        """But it does not vanish either: it shifts every denominator it should
        have been in, so the count travels with the reading."""
        root = write_reports(tmp_path / "c", {
            "AAA_20260808": {AT_THE_BAR: 0.0},
            "BBB_20260808": {AT_THE_BAR: 1.0},
        })
        broken = root / "CCC_20260808"
        broken.mkdir()
        (broken / "scores.json").write_text("{ this is not json")

        corpus = load_scored_corpus(root)

        assert corpus.reports == 2
        assert corpus.unreadable == ("CCC",)

    def test_a_report_directory_with_no_scores_file_is_skipped(self, tmp_path):
        root = write_reports(tmp_path / "c", {"AAA_20260808": {AT_THE_BAR: 0.0}})
        (root / "BBB_20260808").mkdir()

        corpus = load_scored_corpus(root)

        assert corpus.tickers == ("AAA",)
        assert corpus.unreadable == ("BBB",)


class TestTheSubjectDoesNotVoteOnItself:
    """`exclude`, and the reason it is not a nicety.

    `ReportGenerator.generate` writes this run's own `scores.json` before it
    builds the note, into the same directory the note then scans. R8 asks
    whether *other* companies read zero here; a corpus that includes the
    subject is answering a different question, and at the minimum comparable
    count it can answer it the other way round.
    """

    def test_the_subject_leaves_the_corpus_entirely(self, tmp_path):
        root = write_reports(tmp_path / "c", {
            "AAA_20260808": {AT_THE_BAR: 0.0},
            "BBB_20260808": {AT_THE_BAR: 1.0},
            "SUBJ_20260808": {AT_THE_BAR: 0.0},
        })

        corpus = load_scored_corpus(root, exclude="SUBJ")

        assert corpus.tickers == ("AAA", "BBB")
        rate = corpus.rate_for(AT_THE_BAR)
        assert (rate.zero, rate.comparable) == (1, 2)

    def test_every_dated_report_of_the_subject_goes_with_it(self, tmp_path):
        """The exclusion happens after the latest-per-ticker map resolves, so
        one `pop` removes a company analysed weekly for a year rather than the
        most recent run only."""
        root = write_reports(tmp_path / "c", {
            "SUBJ_20260101": {AT_THE_BAR: 0.0},
            "SUBJ_20260401": {AT_THE_BAR: 0.0},
            "SUBJ_20260808": {AT_THE_BAR: 0.0},
            "AAA_20260808": {AT_THE_BAR: 1.0},
        })

        corpus = load_scored_corpus(root, exclude="SUBJ")

        assert corpus.tickers == ("AAA",)
        assert corpus.rate_for(AT_THE_BAR).zero == 0

    def test_one_self_vote_is_enough_to_flip_the_majority(self, tmp_path):
        """The measured hazard, not a hypothetical. Six other companies split
        three-three, which is not a majority and therefore does not suppress —
        adding the subject's own zero makes it four of seven and buries the
        very gap the section was about to explain."""
        reports = {
            f"Z{index}_20260808": {AT_THE_BAR: 0.0 if index < 3 else 1.0}
            for index in range(6)
        }
        reports["SUBJ_20260808"] = {AT_THE_BAR: 0.0}
        root = write_reports(tmp_path / "c", reports)

        assert load_scored_corpus(root).rate_for(AT_THE_BAR).suppresses is True
        assert (
            load_scored_corpus(root, exclude="SUBJ")
            .rate_for(AT_THE_BAR).suppresses
            is False
        )

    def test_the_subjects_own_unreadable_report_is_not_counted_either(
        self, tmp_path
    ):
        """Dropped before the read, not after. An unreadable report shifts every
        denominator it should have been in and the count travels with the
        reading — but the subject was never entitled to a denominator here, so
        its own malformed file must not make the corpus look short."""
        root = write_reports(tmp_path / "c", {"AAA_20260808": {AT_THE_BAR: 0.0}})
        broken = root / "SUBJ_20260808"
        broken.mkdir()
        (broken / "scores.json").write_text("{ this is not json")

        corpus = load_scored_corpus(root, exclude="SUBJ")

        assert corpus.unreadable == ()
        assert corpus.tickers == ("AAA",)

    def test_a_corpus_of_only_the_subject_says_so_rather_than_blaming_the_disk(
        self, tmp_path
    ):
        """Both emptinesses read as "nothing to compare against", and neither
        suppresses — but they send a reader to different places, and "could not
        be read" would send them hunting a bug that is not there."""
        root = write_reports(tmp_path / "c", {"SUBJ_20260808": {AT_THE_BAR: 0.0}})

        corpus = load_scored_corpus(root, exclude="SUBJ")

        assert corpus.reports == 0
        assert "no other company" in corpus.error
        assert "could not be read" not in corpus.error
        assert corpus.rate_for(AT_THE_BAR).suppresses is None

    def test_a_ticker_nobody_analysed_excludes_nothing(self, tmp_path):
        root = write_reports(tmp_path / "c", {
            "AAA_20260808": {AT_THE_BAR: 0.0},
            "BBB_20260808": {AT_THE_BAR: 1.0},
        })

        corpus = load_scored_corpus(root, exclude="NEVER_SEEN")

        assert corpus.tickers == ("AAA", "BBB")
        assert corpus.error == ""


# ── The zero-score trigger (R6) ───────────────────────────────────────────


class TestTheZeroScoreTrigger:
    def test_a_zero_at_or_above_the_weight_bar_fires(self, tmp_path, decide):
        """The lightest metric in the element that still clears R6's bar."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=1, comparable=7)

        section = decide(corpus, element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0})

        assert section.expand
        assert section.fired_triggers == (ZERO_SCORE_GAP,)

    def test_a_zero_below_the_weight_bar_does_not_fire(self, tmp_path, decide):
        """The heaviest metric that still misses the bar. A zero there cannot
        have moved the score enough to need explaining."""
        corpus = corpus_where(tmp_path / "c", UNDER_THE_BAR, zero=1, comparable=7)

        section = decide(corpus, element=BAR_ELEMENT, scored={UNDER_THE_BAR: 0.0})

        assert not section.expand

    def test_the_bar_is_inclusive_at_exactly_the_share(self, tmp_path, pairs):
        """`>=` not `>`, tested on a registry built to sit on the boundary.

        No shipped metric lands on exactly MIN_WEIGHT_SHARE of its element and
        none is obliged to — the shares move whenever an element gains a
        metric. So the inclusivity of the comparison is pinned here, against
        two metrics weighted to put one of them exactly on the line, rather
        than resting on a coincidence in the shipped weights that a later
        registry edit would silently retire.
        """
        configs = {
            "on_the_line": {
                "element": BAR_ELEMENT,
                "name": "On The Line",
                "scoring": {"weight": MIN_WEIGHT_SHARE},
            },
            "the_rest": {
                "element": BAR_ELEMENT,
                "name": "The Rest",
                "scoring": {"weight": 1.0 - MIN_WEIGHT_SHARE},
            },
        }
        decider = ExpansionDecider(
            configs, pairs,
            corpus_where(tmp_path / "c", "on_the_line", zero=1, comparable=7),
        )

        assert decider.weight_share("on_the_line") == pytest.approx(MIN_WEIGHT_SHARE)

        section = decider.evaluate(
            read_metrics(configs, {"on_the_line": 1.0, "the_rest": 1.0}),
            scores_for({"on_the_line": 0.0}),
            elements=[BAR_ELEMENT],
        )[BAR_ELEMENT]

        assert section.expand
        assert ZERO_SCORE_GAP in section.fired_triggers

    def test_the_weight_share_is_of_the_elements_declared_total(
        self, engine, metric_configs, pairs
    ):
        """R6 says "of its element's weight" — three things it is not.

        Not the composite: Quality — Business is 20% of it, so `roiic` at a
        tenth of the element is 2% of the composite and the trigger would never
        fire on anything. Not the raw declared weight either, which happens to
        coincide only for the four elements whose declared weights sum to
        exactly 1.0 — Growth sums to 1.07 and Longevity to 1.05, so the check
        below is mechanical over the whole registry rather than resting on one
        conveniently chosen metric. And derived, not written down: an element
        gaining a metric moves every share in it.
        """
        decider = ExpansionDecider(metric_configs, pairs, ScoredCorpus())
        totals = decider.declared_element_weights
        differs = 0

        for metric_id, config in metric_configs.items():
            weight = (config.get("scoring") or {}).get("weight", 0) or 0
            if weight <= 0:
                assert decider.weight_share(metric_id) is None, metric_id
                continue
            total = totals[config["element"]]
            assert decider.weight_share(metric_id) == pytest.approx(weight / total)
            differs += weight != pytest.approx(weight / total)

        assert differs, "no element's declared weights sum to anything but 1.0"
        # The fixture self-check: whatever `_pick_bar_metrics` chose really
        # does straddle the bar, so every test using them is testing the
        # trigger rather than an assumption about the shipped weights.
        assert decider.weight_share(AT_THE_BAR) >= MIN_WEIGHT_SHARE
        assert decider.weight_share(UNDER_THE_BAR) < MIN_WEIGHT_SHARE

    def test_a_metric_that_errored_is_not_a_zero(self, tmp_path, decide):
        """R18's exclusion. A `None` score is a gap in the evidence, which the
        collapsed section's coverage clause states — not a finding about the
        company, which is what a fired trigger would claim."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=1, comparable=7)

        section = decide(
            corpus, element=BAR_ELEMENT, scored={AT_THE_BAR: None},
            errors={AT_THE_BAR: "Missing input(s): ratios"},
        )

        assert not section.expand

    def test_a_corpus_wide_zero_does_not_expand_for_any_company(
        self, tmp_path, decide
    ):
        """R8. It is describing the model, and a finding true of everybody is a
        finding about nobody."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=5, comparable=7)

        section = decide(corpus, element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0})

        assert not section.expand

    def test_the_declared_element_weights_agree_with_the_scorer(
        self, metric_configs, pairs
    ):
        """`_declared_element_weights` restates `SQGLPScorer._declared_weights`
        rather than importing a private method. The risk in restating is drift,
        so the agreement is pinned rather than assumed."""
        decider = ExpansionDecider(metric_configs, pairs, ScoredCorpus())

        assert (
            decider.declared_element_weights
            == SQGLPScorer(metric_configs, {})._declared_weights()
        )


# ── Order: sector mismatch is decided first (R8, F1) ──────────────────────


class TestSectorMismatchIsDecidedFirst:
    """R8's closing sentence, held structurally rather than remembered.

    Sector mismatch terminates the metric's walk, so a metric that is both
    inapplicable *and* a corpus-wide zero never reaches the test that would
    have suppressed it. A later `or` over three independently computed booleans
    would look equivalent and silently lose this.
    """

    def test_a_sector_inapplicable_metric_that_is_also_a_corpus_wide_zero_expands(
        self, tmp_path, decide
    ):
        """The real case: `dcf_margin_of_safety` reads zero in five of the seven
        analysed companies *and* is excluded for lenders."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=5, comparable=7
        )
        assert corpus.rate_for("dcf_margin_of_safety").suppresses is True

        section = decide(
            corpus, sector=LENDER, element="price",
            values={"dcf_margin_of_safety": -100.0},
            scored={"dcf_margin_of_safety": 0.0},
        )

        assert section.expand
        assert section.fired_triggers == (SECTOR_MISMATCH,)

    def test_the_same_metric_in_a_reviewed_manufacturing_sector_is_suppressed(
        self, tmp_path, decide
    ):
        """The control that makes the test above mean something: with the sector
        trigger out of the way, the corpus rule really does suppress it."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=5, comparable=7
        )

        section = decide(
            corpus, sector=MANUFACTURER, element="price",
            values={"dcf_margin_of_safety": -100.0},
            scored={"dcf_margin_of_safety": 0.0},
        )

        assert not section.expand

    def test_a_fired_sector_mismatch_terminates_that_metrics_walk(
        self, tmp_path, decide
    ):
        """One reason for the metric, and it is the mismatch — not a second one
        off a number the report has just said means nothing here."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=0, comparable=7
        )

        section = decide(
            corpus, sector=LENDER, element="price",
            values={"dcf_margin_of_safety": FAVOURABLE_DCF},
            scored={"dcf_margin_of_safety": 0.0},
            eligibility=gate_reading(False),
        )

        fired = [m for m in section.metrics if m.metric_id == "dcf_margin_of_safety"]
        assert len(fired[0].reasons) == 1
        assert fired[0].reasons[0].trigger == SECTOR_MISMATCH


class TestContradictionIsDecidedBeforeTheZeroScore:
    def test_a_declared_disagreement_outranks_a_zero_at_weight(
        self, tmp_path, decide
    ):
        """Both would fire. F1 puts the pair first, and the pair's declared
        sentence is the more useful thing to say."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=0, comparable=7
        )

        section = decide(
            corpus, sector=MANUFACTURER, element="price",
            values={"dcf_margin_of_safety": FAVOURABLE_DCF},
            scored={"dcf_margin_of_safety": 0.0},
            eligibility=gate_reading(False),
        )

        fired = [m for m in section.metrics if m.fired]
        assert [r.trigger for m in fired for r in m.reasons] == [CONTRADICTION]

    def test_the_fired_text_carries_the_declared_sentence_verbatim(
        self, tmp_path, decide, pairs
    ):
        """R7 puts the declaration in front of the reader; this layer supplies a
        lead-in and never paraphrases the explanation."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=0, comparable=7
        )

        section = decide(
            corpus, sector=MANUFACTURER, element="price",
            values={"dcf_margin_of_safety": FAVOURABLE_DCF},
            eligibility=gate_reading(False),
        )

        assert any(
            pairs.pairs[SHIPPED_PAIR]["reason"] in reason.text
            for reason in section.reasons
        )


# ── The OR across a section, and across sections (R6, R7, R9, KD5) ────────


class TestTheSectionLevelOr:
    def test_two_triggers_on_different_metrics_produce_two_reasons(
        self, tmp_path, decide
    ):
        """R7 names every trigger that fired, not the first one.

        Longevity for a lender trips both at once: the sector table excludes
        some of its metrics, and `roce_consistency` carries enough of the
        element that a zero there clears R6's bar on its own. The excluded set
        is read off the table rather than listed here — it grew from one metric
        to two when the table started reaching the scorer, and a list written
        down would have made that read as a broken trigger.
        """
        corpus = corpus_where(
            tmp_path / "c", "roce_consistency", zero=0, comparable=7
        )

        section = decide(
            corpus, sector=LENDER, element="longevity",
            values={"fcf_consistency": 2.0},
            scored={"roce_consistency": 0.0},
        )

        assert set(section.fired_triggers) == {SECTOR_MISMATCH, ZERO_SCORE_GAP}

        mismatched = {
            r.metric_id for r in section.reasons if r.trigger == SECTOR_MISMATCH
        }
        zeroed = {r.metric_id for r in section.reasons if r.trigger == ZERO_SCORE_GAP}

        assert mismatched == lender_exclusions_in("longevity")
        assert zeroed == {"roce_consistency"}
        assert len(section.reasons) == len(mismatched) + 1

    def test_a_finding_reached_by_several_sections_is_stated_in_each(
        self, tmp_path, decide
    ):
        """R9 and KD4: no roll-up. A lender trips the same structural mismatch
        in Quality — Business, Longevity and Price, and each section says so."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        sections = decide(corpus, sector=LENDER)

        mismatched = {
            element: [r for r in decision.reasons if r.trigger == SECTOR_MISMATCH]
            for element, decision in sections.items()
        }
        assert [e for e, rs in mismatched.items() if rs] == [
            "longevity", "price", "quality_business"
        ]
        for element, reasons in mismatched.items():
            assert {r.metric_id for r in reasons} == lender_exclusions_in(element), element

    def test_no_cap_limits_how_many_sections_expand(self, tmp_path, decide):
        """KD5. The length is the verdict, so nothing here budgets it."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        sections = decide(
            corpus, sector=LENDER,
            scored={
                AT_THE_BAR: 0.0, "market_cap": 0.0, "cap_proxy": 0.0,
                "peg_ratio": 0.0, "equity_dilution": 0.0, "revenue_cagr_5yr": 0.0,
            },
        )

        assert len(expanded_sections(sections)) == 6

    def test_reasons_within_a_section_are_ordered_deterministically(
        self, tmp_path, decide
    ):
        """Two runs of one report must not shuffle their own paragraphs."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)
        kwargs = dict(sector=LENDER, element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0})

        first = decide(corpus, **kwargs)
        second = decide(corpus, **kwargs)

        assert [r.text for r in first.reasons] == [r.text for r in second.reasons]
        assert [r.metric_name for r in first.reasons] == sorted(
            r.metric_name for r in first.reasons
        )


# ── The acceptance examples ───────────────────────────────────────────────


class TestAE1QualityBusinessNamesTheSectorMismatch:
    """AE1. Covers R6, R7.

    PFC's measured Quality — Business: asset turnover 0.09x, equity multiplier
    9.37x and FCF yield -5.7%, each scored at zero for doing exactly what a
    lender does, in an element resting on 32% of its declared weight. Built from
    those figures rather than read off disk, so the case survives a checkout
    that has never run `analyze`.
    """

    def test_ae1_pfc_quality_business_expands_and_names_the_sector_mismatch(
        self, tmp_path, decide
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=LENDER, element=BAR_ELEMENT,
            values={
                "dupont_turnover": 0.09,
                "dupont_equity_multiplier": 9.37,
                "fcf_yield": -5.7,
                "dupont_margin": 22.4,
                "roe_5yr_avg": 18.1,
            },
            errors={
                "roce_5yr_avg": "Missing input(s): ratios",
                "operating_margin_5yr": "Missing input(s): ratios",
                AT_THE_BAR: "Missing input(s): ratios",
            },
            scored={
                "dupont_turnover": 0.0,
                "dupont_equity_multiplier": 0.0,
                "fcf_yield": 0.0,
                "dupont_margin": 1.0,
                "roe_5yr_avg": 0.83,
            },
            coverage={BAR_ELEMENT: 0.32},
        )

        assert section.expand
        assert section.fired_triggers == (SECTOR_MISMATCH,)
        # The three PFC figures this case is built from are named, along with
        # every other Quality — Business reading the table withdraws from a
        # lender. Asserting the set rather than those three keeps the case
        # about the trigger rather than about how many entries the table held
        # on the day it was written.
        named = {r.metric_id for r in section.reasons}
        assert named == lender_exclusions_in(BAR_ELEMENT)
        assert {"dupont_turnover", "dupont_equity_multiplier", "fcf_yield"} <= named

    def test_ae1_each_reason_explains_what_a_lender_is_instead(
        self, tmp_path, decide
    ):
        """R7's substance. "Does not apply" leaves the reader with the same
        three zeros and no way to read them."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=LENDER, element=BAR_ELEMENT,
            values={"dupont_turnover": 0.09, "dupont_equity_multiplier": 9.37,
                    "fcf_yield": -5.7},
            scored={"dupont_turnover": 0.0, "dupont_equity_multiplier": 0.0,
                    "fcf_yield": 0.0},
        )

        joined = " ".join(r.text for r in section.reasons)
        assert "loan book" in joined
        assert "capital-adequacy" in joined
        assert "operating outflow" in joined
        for reason in section.reasons:
            assert len(reason.text.split()) >= 30, reason.metric_name

    def test_ae7_the_collapsed_reading_still_carries_the_coverage(
        self, tmp_path, decide
    ):
        """AE7 rides along: R18's clause is derived from the same `scores` block
        the size was decided from, so a section can never state one run's
        coverage beside another run's decision."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=LENDER, element=BAR_ELEMENT, coverage={BAR_ELEMENT: 0.32}
        )

        assert section.coverage.low
        assert "32%" in section.coverage.clause


class TestAE2TheCorpusWideZeroDoesNotExpand:
    """AE2. Covers R8.

    `dcf_margin_of_safety` scores zero in five of the seven analysed companies,
    which is what KTD5 measured the majority threshold against.
    """

    def test_ae2_dcf_margin_of_safety_does_not_fire_the_zero_score_trigger(
        self, tmp_path, decide
    ):
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=5, comparable=7
        )

        section = decide(
            corpus, sector=MANUFACTURER, element="price",
            values={"dcf_margin_of_safety": -100.0},
            scored={"dcf_margin_of_safety": 0.0},
        )

        assert not section.expand
        assert corpus.rate_for("dcf_margin_of_safety").share == pytest.approx(5 / 7)

    def test_ae2_the_measured_thresholds_still_separate_the_same_way(
        self, tmp_path
    ):
        """KTD5's evidence, as an invariant rather than a note: five of seven is
        suppressed at a simple majority and at 60%, and not at 75% — which is
        why the threshold is where it is."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=5, comparable=7
        )
        rate = corpus.rate_for("dcf_margin_of_safety")

        assert rate.share > 0.5
        assert rate.share > 0.6
        assert rate.share < 0.75


class TestAE5NothingFiresAndEverySectionCollapses:
    """AE5. Covers R5, R9.

    A company in a reviewed sector, nothing scoring zero, no declared pair
    disagreeing. The report is short, and its shortness is the verdict.
    """

    def test_ae5_a_company_firing_no_trigger_collapses_every_section(
        self, tmp_path, decide
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        sections = decide(
            corpus, sector=MANUFACTURER,
            values={"dcf_margin_of_safety": UNFAVOURABLE_DCF},
            scored={AT_THE_BAR: 0.8, "market_cap": 1.0, "cap_proxy": 0.6,
                    "peg_ratio": 0.5, "equity_dilution": 0.9,
                    "revenue_cagr_5yr": 0.7},
            eligibility=gate_reading(True),
        )

        assert expanded_sections(sections) == []
        assert all(not decision.reasons for decision in sections.values())

    def test_ae5_a_collapsed_section_still_carries_its_metric_decisions(
        self, tmp_path, decide
    ):
        """Collapsed is a rendering, not a gap in the data: U10 still needs the
        per-metric outcome to render the score and the one-line reading."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(corpus, sector=MANUFACTURER, element=BAR_ELEMENT)

        assert not section.expand
        # Every metric in the element, counted off the registry. A literal here
        # asserts the size of the element rather than the completeness of the
        # decision, and goes red for the one change it should not care about —
        # the element gaining a metric.
        assert {d.metric_id for d in section.metrics} == {
            metric_id
            for metric_id, config in ComputeEngine().metrics.items()
            if config["element"] == BAR_ELEMENT
        }


class TestAE8ACorpusBelowTheMinimum:
    """AE8. Covers R8.

    The corpus cannot answer, so it does not get to suppress — and the reading
    says how far short it is rather than shrugging.
    """

    def test_ae8_a_corpus_below_the_minimum_expands(self, tmp_path, decide):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=3, comparable=3)

        section = decide(corpus, element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0})

        assert section.expand
        assert section.fired_triggers == (ZERO_SCORE_NOT_COMPARABLE,)

    def test_ae8_the_reading_states_how_many_exist_and_how_many_are_needed(
        self, tmp_path, decide
    ):
        """Not a generic "insufficient data": AE8 asks for both numbers, because
        a reader cannot tell "two more reports away" from "nobody will ever run
        this test" without them."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=3, comparable=3)

        section = decide(corpus, element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0})
        text = section.reasons[0].text

        assert "3 of the 3 scored reports" in text
        assert f"{MIN_COMPARABLE_REPORTS} needed" in text

    def test_ae8_an_unreadable_corpus_says_so_rather_than_quoting_a_zero(
        self, tmp_path, decide
    ):
        """Zero of zero reports is arithmetically a fine sentence and a useless
        one. The load failure carries its own wording."""
        section = decide(
            load_scored_corpus(tmp_path / "never-generated"),
            element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0},
        )

        assert section.fired_triggers == (ZERO_SCORE_NOT_COMPARABLE,)
        assert "could not be read" in section.reasons[0].text

    def test_ae8_a_skipped_report_is_named_in_the_shortfall(self, tmp_path, decide):
        """A corpus short because two of its reports were unreadable is a
        different problem from one short because nobody has run enough
        analyses, and only one of them is fixed by waiting."""
        root = write_reports(tmp_path / "c", {
            f"T{i}_20260808": {AT_THE_BAR: 1.0} for i in range(3)
        })
        for name in ("X_20260808", "Y_20260808"):
            broken = root / name
            broken.mkdir()
            (broken / "scores.json").write_text("nope")

        section = decide(
            load_scored_corpus(root), element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0}
        )

        assert "2 further reports" in section.reasons[0].text


# ── Unknowns, in both directions ──────────────────────────────────────────


class TestUnknownIsNeverASilentPassInTheDecision:
    """The spine, applied to a decision where the unknowns pull two ways.

    An unknown trigger condition must not fire — most sectors are unreviewed and
    most runs evaluate no gates, so firing would expand everything and mean
    nothing. An unknown suppression must not suppress. Neither may be silent.
    """

    def test_an_unreviewed_sector_does_not_fire_but_is_recorded(
        self, tmp_path, decide
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=UNREVIEWED, element=BAR_ELEMENT,
            values={"dupont_turnover": 0.09},
        )

        assert not section.expand
        assert any("has not been checked" in u for u in section.unresolved)
        assert any(UNREVIEWED in u for u in section.unresolved)

    def test_a_company_with_no_recorded_sector_does_not_fire_but_is_recorded(
        self, tmp_path, decide
    ):
        """The refetch case. A ticker fetched before the breadcrumb fix carries
        no `metadata.sector`, and a missing sector must not read as a fitting
        one."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=None, element=BAR_ELEMENT, values={"dupont_turnover": 0.09}
        )

        assert not section.expand
        assert any("has not been checked" in u for u in section.unresolved)

    def test_gates_that_were_never_evaluated_do_not_fire_but_are_recorded(
        self, tmp_path, decide
    ):
        """`watchlist advance` re-scores with no LLM and evaluates no gates, so
        every declared pair reads indeterminate on that path. Firing there would
        expand the Price section of every company on every advance."""
        corpus = corpus_where(
            tmp_path / "c", "dcf_margin_of_safety", zero=0, comparable=7
        )

        section = decide(
            corpus, sector=MANUFACTURER, element="price",
            values={"dcf_margin_of_safety": FAVOURABLE_DCF}, eligibility=None,
        )

        assert not section.expand
        assert any("could not be run" in u for u in section.unresolved)

    def test_a_metric_nobody_read_does_not_fire_but_is_recorded(
        self, metric_configs, pairs, tmp_path
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)
        decider = ExpansionDecider(metric_configs, pairs, corpus)

        decision = decider.evaluate_metric(AT_THE_BAR, {}, scores_for({}))

        assert not decision.fired
        assert decision.unresolved
        assert "was not read for this company" in decision.unresolved[0]

    def test_a_section_with_no_readings_says_so_once_rather_than_per_metric(
        self, metric_configs, pairs, tmp_path
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)
        decider = ExpansionDecider(metric_configs, pairs, corpus)

        section = decider.evaluate_section(BAR_ELEMENT, {}, scores_for({}))

        assert not section.expand
        assert len(section.unresolved) == 2   # no readings, and no scores
        assert all(decision.unresolved for decision in section.metrics)

    def test_a_run_with_no_scores_says_the_zero_test_could_not_run(
        self, tmp_path, decide
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(corpus, element=BAR_ELEMENT, scored={})

        assert any("No scores were supplied" in u for u in section.unresolved)

    def test_an_unknown_sector_still_lets_the_other_triggers_fire(
        self, tmp_path, decide
    ):
        """Indeterminate on one check is not indeterminate on the section. The
        zero-score test does not need the sector table to run."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=UNREVIEWED, element=BAR_ELEMENT, scored={AT_THE_BAR: 0.0}
        )

        assert section.fired_triggers == (ZERO_SCORE_GAP,)
        assert section.unresolved


class TestAnUnreviewedSectorSaysItOnce:
    """The same collapse `readings_absent` already makes, for the caveat that
    actually fires in production.

    Two sectors are reviewed; the other twenty-two cached tickers get
    indeterminate applicability for **every** metric, with one shared sentence
    and only the metric's name changing. Thirteen near-identical lines in
    Quality — Business is not thoroughness — it buries the section's real
    caveats under a wall of one caveat, which is the density problem this
    report exists to replace.
    """

    def test_a_section_states_the_unreviewed_sector_once_not_per_metric(
        self, tmp_path, decide
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=UNREVIEWED, element=BAR_ELEMENT,
            values={"dupont_turnover": 0.09}, scored={AT_THE_BAR: 0.5},
        )

        stated = [u for u in section.unresolved if "has not been checked" in u]
        affected = [d for d in section.metrics if d.applicability_reason]
        assert len(stated) == 1
        assert len(affected) > 1, "the fixture must have several metrics to collapse"
        assert stated[0] == section_applicability_line(
            len(affected), affected[0].applicability_reason
        )
        assert UNREVIEWED in stated[0]

    def test_the_per_metric_lines_survive_on_the_decision(self, tmp_path, decide):
        """Only the *rendered* section-level output collapses. A caller that
        wants a line per row has lost nothing, and the decision is still the
        record of what was checked for each metric."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=UNREVIEWED, element=BAR_ELEMENT,
            values={"dupont_turnover": 0.09}, scored={AT_THE_BAR: 0.5},
        )

        affected = [d for d in section.metrics if d.applicability_reason]
        for decision in affected:
            assert decision.applicability_unresolved in decision.unresolved
            assert decision.metric_name in decision.applicability_unresolved
            assert (
                decision.applicability_unresolved
                not in decision.unresolved_beyond_applicability
            )

    def test_a_reviewed_sector_states_nothing_at_all(self, tmp_path, decide):
        """The collapse must not become a line every section always carries."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)

        section = decide(
            corpus, sector=MANUFACTURER, element=BAR_ELEMENT,
            values={"dupont_turnover": 0.09}, scored={AT_THE_BAR: 0.5},
        )

        assert not [u for u in section.unresolved if "has not been checked" in u]
        assert not [d for d in section.metrics if d.applicability_reason]

    def test_one_affected_metric_keeps_its_own_name(
        self, metric_configs, pairs, tmp_path
    ):
        """"These 1 metrics" is both ungrammatical and less informative than
        naming the one metric, so the single case keeps the per-metric wording
        rather than being described as a group."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)
        decider = ExpansionDecider(
            {AT_THE_BAR: metric_configs[AT_THE_BAR]}, pairs, corpus
        )
        readings = readings_for(
            metric_configs, {AT_THE_BAR: 12.0}, sector=UNREVIEWED
        )

        section = decider.evaluate_section(
            BAR_ELEMENT, readings, scores_for({AT_THE_BAR: 0.5})
        )

        stated = [u for u in section.unresolved if "has not been checked" in u]
        assert len(stated) == 1
        assert stated[0].startswith(f"Whether {decider._metric_name(AT_THE_BAR)} ")

    def test_a_section_nobody_read_still_says_that_instead(
        self, metric_configs, pairs, tmp_path
    ):
        """The two collapses do not stack. With no readings at all there is no
        applicability answer to have missed, and the section says the one true
        thing rather than two overlapping ones."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)
        decider = ExpansionDecider(metric_configs, pairs, corpus)

        section = decider.evaluate_section(BAR_ELEMENT, {}, scores_for({}))

        assert len(section.unresolved) == 2   # no readings, and no scores
        assert not [u for u in section.unresolved if "has not been checked" in u]


# ── Zero-weight metrics stay out of the shape of the report (KTD5) ────────


class TestZeroWeightMetricsCannotExpandASection:
    """Expansion is prominence. A signal that deliberately cannot move a score
    must not move the report's shape instead — the coupling the whole
    forward-signals design exists to keep separate.
    """

    def test_every_zero_weight_metric_is_excluded_from_the_decision(
        self, engine, metric_configs, pairs, tmp_path
    ):
        """Derived from the registry rather than a hardcoded list of ids, the
        mechanical form the forward-signals rule was rewritten into after a
        remembered rule let one through."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=7)
        decider = ExpansionDecider(metric_configs, pairs, corpus)
        zero_weight = [
            metric_id for metric_id, config in metric_configs.items()
            if not engine._scored(config)
        ]
        assert zero_weight, "the registry has stopped carrying forward signals"

        for metric_id in zero_weight:
            decision = decider.evaluate_metric(
                metric_id,
                readings_for(metric_configs, {metric_id: 0.0}, sector=LENDER),
                scores_for({metric_id: 0.0}),
                gate_reading(False),
            )
            assert not decision.fired, metric_id
            assert not decision.considered, metric_id
            assert decision.excluded_reason

    def test_a_metric_the_registry_does_not_define_is_excluded_with_a_reason(
        self, metric_configs, pairs
    ):
        decider = ExpansionDecider(metric_configs, pairs, ScoredCorpus())

        decision = decider.evaluate_metric("dcf_margin_of_saftey", {}, scores_for({}))

        assert not decision.fired
        assert "not a metric this registry computes" in decision.excluded_reason


# ── R7 and R15: the reasons are the deliverable ───────────────────────────


class TestReasonsAreWrittenForAReader:
    def test_no_reason_leaks_a_raw_metric_id(self, tmp_path, decide):
        """R15. A reason reading `zero_score_trigger: dcf_margin_of_safety`
        fails the requirement however correct the decision behind it."""
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=3)

        sections = decide(
            corpus, sector=LENDER,
            values={"dcf_margin_of_safety": FAVOURABLE_DCF},
            scored={AT_THE_BAR: 0.0, "market_cap": 0.0, "cap_proxy": 0.0},
            eligibility=gate_reading(False),
        )

        texts = [r.text for d in sections.values() for r in d.reasons]
        assert texts
        for metric_id in ("roiic", "market_cap", "cap_proxy", "dupont_turnover",
                          "dcf_margin_of_safety", "fcf_yield"):
            assert not any(metric_id in text for text in texts), metric_id

    def test_every_fired_reason_is_a_sentence_rather_than_a_label(
        self, tmp_path, decide
    ):
        corpus = corpus_where(tmp_path / "c", AT_THE_BAR, zero=0, comparable=3)

        sections = decide(
            corpus, sector=LENDER,
            scored={AT_THE_BAR: 0.0, "market_cap": 0.0},
        )

        reasons = [r for d in sections.values() for r in d.reasons]
        assert reasons
        for reason in reasons:
            assert len(reason.text.split()) >= 25, reason.metric_name
            assert reason.text.strip().endswith(".")

    def test_a_trigger_that_fires_with_nothing_to_say_is_rejected(self):
        with pytest.raises(ValueError, match="must carry the reason it fired"):
            ExpansionReason(
                trigger=SECTOR_MISMATCH, metric_id="x", metric_name="X", text="  "
            )

    def test_an_unknown_trigger_kind_is_rejected(self):
        with pytest.raises(ValueError, match="is not one of"):
            ExpansionReason(
                trigger="flag_fired", metric_id="x", metric_name="X", text="a reason"
            )

    def test_a_fired_flag_is_not_a_trigger_kind(self):
        """KD4, structurally. A quality flag stays in the section's Signals line
        and does not buy space, so there is no kind for it to arrive under."""
        assert len(TRIGGERS) == 4
        assert not any("flag" in trigger for trigger in TRIGGERS)

    def test_every_trigger_kind_has_a_label(self):
        """A label map covering three of four kinds would render the fourth
        through whatever fallback a surface happens to have — the
        auto-humanising failure the problem frame names."""
        assert set(TRIGGER_LABELS) == set(TRIGGERS)
        assert all(label.strip() for label in TRIGGER_LABELS.values())


# ── The corpus that actually exists ───────────────────────────────────────


class TestAgainstTheGeneratedCorpus:
    """Observation, not assertion, and skipped when the folder is absent.

    `output/reports/` is gitignored and machine-local: a suite that asserted on
    it would pass here and fail in a checkout that has never run `analyze`. What
    it can still do is confirm that the numbers KTD5 and AE1/AE2/AE7 were
    measured from are the numbers this code reads back.
    """

    @pytest.fixture
    def real_corpus(self):
        if not DEFAULT_REPORTS_DIR.is_dir():
            pytest.skip("no generated reports on this machine")
        corpus = load_scored_corpus()
        if corpus.reports < 1:
            pytest.skip("the generated-report corpus is empty")
        return corpus

    def test_ae2_the_shipped_corpus_suppresses_dcf_margin_of_safety(
        self, real_corpus
    ):
        rate = real_corpus.rate_for("dcf_margin_of_safety")
        if not rate.comparable_enough:
            pytest.skip(
                f"only {rate.comparable} reports computed it; "
                f"{rate.minimum} are needed"
            )

        assert rate.suppresses is True

    def test_ae7_pfcs_quality_business_coverage_is_a_third(self, real_corpus):
        pfc = latest_scores_for("PFC")
        if pfc is None:
            pytest.skip("PFC has not been analysed on this machine")

        coverage = json.loads(pfc.read_text())["coverage"]["elements"]

        assert coverage["quality_business"] == pytest.approx(0.32, abs=0.005)
