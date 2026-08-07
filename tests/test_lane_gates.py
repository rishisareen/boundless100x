"""Fast-lane entry gates — the third sibling evaluator.

`compute_engine/eligibility.py` asks "could this plausibly 100x?" and
`lifecycle/evaluator.py` asks "what transition is due?". These gates ask a
third question — "does this qualify for the fast lane, right now?" — and answer
it with the same three-valued rule both siblings use: **a gate whose inputs are
missing reads indeterminate, never a silent pass.** A lane gate that could not
be evaluated must not look like one that was cleared, because on this side of
the system a cleared gate is what lets capital move.

One condition kind is new. `catalyst_status` reads the owner-recorded catalyst
on the watchlist entry rather than a computed metric, and it has two distinct
empty cases that this file keeps apart: an entry carrying no catalyst has been
*looked at* and has none (a plain fail), while evaluating with no watchlist
context at all is an unknown. Collapsing them would let a company with no
recorded catalyst read the same as one nobody has checked.
"""

import pytest
import yaml

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.lane_gates import (
    DEFAULT_LANE_GATES,
    DEFAULT_LANE_GATES_PATH,
    INDETERMINATE,
    LANE_VERDICTS,
    NOT_QUALIFIED,
    QUALIFIES,
    LaneGateEvaluator,
    effective_lane_gates,
    load_lane_gates,
)
from tests.conftest import make_scores


def shipped_gates() -> dict:
    """The YAML gates production actually runs, not the Python constant."""
    return load_lane_gates()


def evaluator(gates: dict | None = None) -> LaneGateEvaluator:
    return LaneGateEvaluator(gates if gates is not None else shipped_gates())


def passing_metrics(**overrides) -> dict:
    """A candidate clearing every declared threshold."""
    metrics = {
        "pe_vs_historical": MetricResult(value=35.0),
        # Ran, and emitted no risky-growth flag: the absence is confirmed.
        "growth_quality_grade": MetricResult(
            value="high_quality", flags=["growth_quality_high_quality"]
        ),
        "rerating_headroom": MetricResult(value=40.0),
        "ttm_growth_vs_cagr": MetricResult(value=6.0),
        "institutional_accumulation_streak": MetricResult(
            value=3.0, flags=["institutional_accumulation_rising"]
        ),
        "daily_turnover_ratio": MetricResult(value=0.05),
    }
    metrics.update(overrides)
    return metrics


def active_catalyst() -> dict:
    return {
        "description": "Capacity commissioning at the Dahej plant",
        "expected_by": "2026-12-31",
        "status": "active",
        "recorded_at": "2026-08-07T09:00:00",
    }


def qualifying(metrics=None, scores=None, catalyst=None) -> dict:
    return evaluator().evaluate(
        metrics if metrics is not None else passing_metrics(),
        scores if scores is not None else make_scores(composite=6.5),
        catalyst if catalyst is not None else active_catalyst(),
    )


class TestVerdicts:
    def test_a_fast_lane_eligible_candidate_qualifies(self):
        verdict = qualifying()

        assert verdict["qualifies"] is True
        assert verdict["verdict"] == "qualifies"
        assert verdict["failed"] == []
        assert verdict["indeterminate"] == []

    def test_every_declared_gate_reports_its_own_detail(self):
        gates = qualifying()["gates"]

        assert set(gates) == set(shipped_gates())
        for detail in gates.values():
            assert "label" in detail and "passed" in detail and "reason" in detail

    @pytest.mark.parametrize("gate_id,kwargs", [
        ("quality_floor", {"scores": make_scores(composite=4.9)}),
        (
            "valuation_discount",
            {"metrics": passing_metrics(
                pe_vs_historical=MetricResult(value=85.0),
                rerating_headroom=MetricResult(value=-40.0),
            )},
        ),
        (
            "growth_intact",
            {"metrics": passing_metrics(ttm_growth_vs_cagr=MetricResult(value=-6.0))},
        ),
        (
            "institutional_accumulation",
            {"metrics": passing_metrics(
                institutional_accumulation_streak=MetricResult(value=1.0)
            )},
        ),
        ("catalyst_identified", {"catalyst": {"status": "spent"}}),
        (
            "liquidity_floor",
            {"metrics": passing_metrics(daily_turnover_ratio=MetricResult(value=0.001))},
        ),
    ])
    def test_each_gate_fails_alone(self, gate_id, kwargs):
        """One gate at a time, so a fixture cannot pass for the wrong reason."""
        verdict = qualifying(**kwargs)

        assert verdict["qualifies"] is False
        assert verdict["verdict"] == "not_qualified"
        assert verdict["failed"] == [gate_id]
        assert verdict["gates"][gate_id]["passed"] is False

    def test_risky_growth_alone_fails_the_growth_gate(self):
        """§9.2's second clause: growth intact means not FinLev-driven."""
        verdict = qualifying(metrics=passing_metrics(
            growth_quality_grade=MetricResult(
                value="risky", flags=["growth_quality_risky"]
            )
        ))

        assert verdict["failed"] == ["growth_intact"]

    def test_a_favourable_rerating_flag_substitutes_for_the_percentile(self):
        """`mode: any` — either route into the valuation discount is enough."""
        verdict = qualifying(metrics=passing_metrics(
            pe_vs_historical=MetricResult(value=85.0),
            rerating_headroom=MetricResult(
                value=40.0, flags=["rerating_headroom_favourable"]
            ),
        ))

        assert verdict["qualifies"] is True

    def test_a_failure_reason_names_the_threshold(self):
        verdict = qualifying(metrics=passing_metrics(
            institutional_accumulation_streak=MetricResult(value=1.0)
        ))

        assert "2" in verdict["gates"]["institutional_accumulation"]["reason"]


class TestIndeterminate:
    def test_an_errored_source_metric_reads_indeterminate_not_failed(self):
        """No shareholding data is not evidence that nobody is accumulating."""
        verdict = qualifying(metrics=passing_metrics(
            institutional_accumulation_streak=MetricResult(
                error="No shareholding data"
            )
        ))

        assert verdict["qualifies"] is None
        assert verdict["verdict"] == "indeterminate"
        assert verdict["failed"] == []
        assert verdict["indeterminate"] == ["institutional_accumulation"]

    def test_a_missing_metric_reads_indeterminate(self):
        metrics = passing_metrics()
        del metrics["daily_turnover_ratio"]

        verdict = qualifying(metrics=metrics)

        assert verdict["indeterminate"] == ["liquidity_floor"]

    def test_an_errored_flag_source_makes_the_absence_unconfirmable(self):
        """The price gate's caveat, inherited: absence proves nothing alone."""
        verdict = qualifying(metrics=passing_metrics(
            growth_quality_grade=MetricResult(error="Insufficient data")
        ))

        assert verdict["verdict"] == "indeterminate"
        assert verdict["indeterminate"] == ["growth_intact"]

    def test_absent_scores_make_the_quality_floor_indeterminate(self):
        verdict = qualifying(scores={})

        assert verdict["indeterminate"] == ["quality_floor"]

    def test_a_failure_outranks_an_unknown(self):
        """One broken gate settles the question; the unknown one cannot undo it."""
        verdict = qualifying(
            metrics=passing_metrics(
                daily_turnover_ratio=MetricResult(value=0.001),
                institutional_accumulation_streak=MetricResult(error="no data"),
            )
        )

        assert verdict["verdict"] == "not_qualified"
        assert verdict["failed"] == ["liquidity_floor"]
        assert verdict["indeterminate"] == ["institutional_accumulation"]


class TestCatalystStatus:
    def test_an_active_catalyst_passes(self):
        assert qualifying()["gates"]["catalyst_identified"]["passed"] is True

    def test_an_entry_with_no_catalyst_recorded_fails(self):
        """Looked at, and there is none — a real "not yet identified"."""
        verdict = qualifying(catalyst={})

        assert verdict["gates"]["catalyst_identified"]["passed"] is False
        assert verdict["verdict"] == "not_qualified"

    def test_no_watchlist_context_at_all_is_indeterminate(self):
        """Distinct from the empty dict above, and it must stay distinct.

        `{}` is falsy, so a truthiness check would collapse "this company has
        no catalyst" into "nobody asked" — and one of those is a decision while
        the other is an absence of one.
        """
        verdict = evaluator().evaluate(passing_metrics(), make_scores(composite=6.5))

        assert verdict["gates"]["catalyst_identified"]["passed"] is None
        assert verdict["verdict"] == "indeterminate"

    def test_a_spent_catalyst_fails(self):
        assert qualifying(catalyst={"status": "spent"})["failed"] == [
            "catalyst_identified"
        ]


class TestTheShippedRegistry:
    def test_the_yaml_declares_the_six_gates(self):
        assert set(shipped_gates()) == {
            "quality_floor",
            "valuation_discount",
            "growth_intact",
            "institutional_accumulation",
            "catalyst_identified",
            "liquidity_floor",
        }

    def test_the_yaml_and_the_shipped_defaults_agree(self):
        """Two statements of one regime must not be able to drift apart."""
        assert shipped_gates() == DEFAULT_LANE_GATES

    def test_no_declared_gates_falls_back_to_the_shipped_defaults(self):
        assert effective_lane_gates({}) == DEFAULT_LANE_GATES
        assert effective_lane_gates(None) == DEFAULT_LANE_GATES
        assert LaneGateEvaluator(effective_lane_gates({})).gates == DEFAULT_LANE_GATES

    def test_every_metric_the_yaml_names_exists_in_the_registry(self):
        """The startup check, run against the real registry."""
        from boundless100x.compute_engine.engine import ComputeEngine

        LaneGateEvaluator(known_metric_ids=set(ComputeEngine().metrics))

    def test_the_declared_thresholds_are_the_documented_starting_points(self):
        gates = shipped_gates()

        assert gates["quality_floor"]["conditions"][0]["threshold"] == 5.5
        assert gates["valuation_discount"]["conditions"][0]["threshold"] == 50
        assert gates["growth_intact"]["conditions"][0]["threshold"] == 0
        assert gates["institutional_accumulation"]["conditions"][0]["threshold"] == 2
        assert gates["liquidity_floor"]["conditions"][0]["threshold"] == 0.02

    def test_the_yaml_marks_its_thresholds_as_starting_points(self):
        assert "STARTING POINT" in DEFAULT_LANE_GATES_PATH.read_text()


class TestStartupValidation:
    """A registry error must be loud at construction.

    The failure it prevents is silent: a gate naming a metric that does not
    exist would read indeterminate forever, and a lane nothing can ever enter
    looks exactly like a lane with no qualifying candidates.
    """

    def gates_with(self, condition: dict) -> dict:
        return {
            "made_up": {
                "label": "Made up",
                "conditions": [condition],
            }
        }

    def test_an_unknown_metric_id_raises(self):
        gates = self.gates_with(
            {"metric": "no_such_metric", "comparator": "gte", "threshold": 1}
        )

        with pytest.raises(ValueError, match="no_such_metric"):
            LaneGateEvaluator(gates, known_metric_ids={"market_cap"})

    def test_an_unknown_comparator_raises(self):
        gates = self.gates_with(
            {"metric": "market_cap", "comparator": "approximately", "threshold": 1}
        )

        with pytest.raises(ValueError):
            LaneGateEvaluator(gates, known_metric_ids={"market_cap"})

    def test_a_missing_threshold_raises(self):
        gates = self.gates_with({"metric": "market_cap", "comparator": "gte"})

        with pytest.raises(ValueError):
            LaneGateEvaluator(gates, known_metric_ids={"market_cap"})

    def test_a_condition_naming_two_kinds_raises(self):
        gates = self.gates_with(
            {"metric": "market_cap", "score": "composite", "comparator": "gte",
             "threshold": 1}
        )

        with pytest.raises(ValueError):
            LaneGateEvaluator(gates, known_metric_ids={"market_cap"})

    def test_a_condition_naming_no_kind_raises(self):
        with pytest.raises(ValueError):
            LaneGateEvaluator(self.gates_with({"comparator": "gte", "threshold": 1}))

    def test_a_gate_with_no_conditions_raises(self):
        with pytest.raises(ValueError):
            LaneGateEvaluator({"empty": {"label": "Empty", "conditions": []}})

    def test_an_unknown_mode_raises(self):
        gates = self.gates_with({"score": "composite", "comparator": "gte",
                                 "threshold": 1})
        gates["made_up"]["mode"] = "most"

        with pytest.raises(ValueError):
            LaneGateEvaluator(gates)

    def test_an_unknown_catalyst_status_raises(self):
        with pytest.raises(ValueError):
            LaneGateEvaluator(self.gates_with({"catalyst_status": "pending"}))

    def test_an_unknown_flag_source_raises(self):
        gates = self.gates_with(
            {"flag_absent": "growth_quality_risky", "sources": ["no_such_metric"]}
        )

        with pytest.raises(ValueError, match="no_such_metric"):
            LaneGateEvaluator(gates, known_metric_ids={"growth_quality_grade"})

    def test_a_sound_registry_constructs(self):
        LaneGateEvaluator(
            self.gates_with({"score": "composite", "comparator": "gte",
                             "threshold": 5.0})
        )


class TestTheYamlIsReadable:
    def test_the_file_parses_and_carries_a_lane_gates_section(self):
        loaded = yaml.safe_load(DEFAULT_LANE_GATES_PATH.read_text())

        assert set(loaded) == {"lane_gates"}


class TestTheVerdictVocabularyIsExportedFromItsSource:
    """One statement of three words that four modules act on.

    Each acts on them differently and each breaks differently on a rename
    nobody propagated: the trigger evaluator validates `lane_verdict:`
    conditions against them, so a stale word is a startup error; `advance`
    routes capital on `QUALIFIES` and blocks everything else, so a stale word
    is a silent capital freeze; the report keys a label map on them, so a stale
    word renders a blank badge. Three failure modes from one edit is the whole
    argument for the words living where they are produced — and for these
    assertions, which are what would catch the edit.
    """

    def test_the_evaluator_emits_exactly_the_exported_words(self):
        """The constants are only worth importing if they are what comes out."""
        assert LANE_VERDICTS == (QUALIFIES, NOT_QUALIFIED, INDETERMINATE)
        assert qualifying()["verdict"] == QUALIFIES
        assert qualifying(scores=make_scores(composite=4.9))["verdict"] == (
            NOT_QUALIFIED
        )
        assert qualifying(scores={})["verdict"] == INDETERMINATE

    def test_the_trigger_evaluator_validates_against_the_same_tuple(self):
        from boundless100x.lifecycle import evaluator as trigger_evaluator

        assert trigger_evaluator.LANE_VERDICTS is LANE_VERDICTS

    def test_the_report_labels_every_verdict_that_can_arrive(self):
        """A verdict with no label renders a blank badge, which reads as a
        company nobody evaluated rather than as a missing entry in a map."""
        from boundless100x.output import report_generator

        assert set(report_generator.LANE_VERDICT_LABELS) == set(LANE_VERDICTS)
