"""Registry hashes: the regime stamps carried by every score-history row.

Phase 2 reads score history as fundamental momentum — a company moving 6.2 to
6.8 is treated as improving. If a weight, threshold, gate, or macro assumption
changed between those two runs, the movement is an artefact of the ruler, not
the company. The hash exists so a trajectory diff can refuse to compare across
regimes; these tests pin what counts as a regime change.

There are **two** regimes, and the split is load-bearing (KTD8). `registry_hash`
covers only what can move a composite; `forward_signal_hash` covers the
zero-weight forward signals and the extraction schema. Folding both into one
hash would make Phase 5 circular: it needs trajectory evidence to calibrate the
forward signals, and calibrating them would reset the baseline that evidence
lives in — unrecoverably, since history is append-only.
"""

import shutil

import pytest
import yaml

from boundless100x.compute_engine.eligibility import DEFAULT_GATES, effective_gates
from boundless100x.compute_engine.engine import ComputeEngine


@pytest.fixture
def registry_dir(tmp_path):
    """A writable copy of the shipped registry."""
    src = ComputeEngine().registry_dir
    dst = tmp_path / "metrics"
    shutil.copytree(src, dst)
    return dst


def engine_at(registry_dir, **kwargs) -> ComputeEngine:
    return ComputeEngine(registry_dir=str(registry_dir), **kwargs)


def edit_yaml(path, mutate):
    config = yaml.safe_load(path.read_text())
    mutate(config)
    path.write_text(yaml.safe_dump(config))


class TestStability:
    def test_two_engines_over_the_same_registry_agree(self, registry_dir):
        assert engine_at(registry_dir).registry_hash == engine_at(registry_dir).registry_hash

    def test_the_shipped_registry_hashes_identically_every_load(self):
        assert ComputeEngine().registry_hash == ComputeEngine().registry_hash

    def test_hash_is_a_short_hex_digest(self):
        digest = ComputeEngine().registry_hash
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)


class TestScoringRegimeChanges:
    """Anything that can move a composite must move the hash."""

    def test_element_weight_change_flips_the_hash(self, registry_dir):
        before = engine_at(registry_dir).registry_hash
        edit_yaml(
            registry_dir / "registry.yaml",
            lambda c: c["element_weights"].update({"growth": 0.30}),
        )
        assert engine_at(registry_dir).registry_hash != before

    def test_metric_threshold_change_flips_the_hash(self, registry_dir):
        before = engine_at(registry_dir).registry_hash
        target = registry_dir / "elements" / "quality_business.yaml"
        config = yaml.safe_load(target.read_text())
        metric = next(iter(config["metrics"].values()))
        metric["scoring"]["thresholds"] = [1, 2, 3, 4, 5, 6]
        target.write_text(yaml.safe_dump(config))

        assert engine_at(registry_dir).registry_hash != before

    def test_gate_threshold_change_flips_the_hash(self, registry_dir):
        before = engine_at(registry_dir).registry_hash
        edit_yaml(
            registry_dir / "registry.yaml",
            lambda c: c["eligibility_gates"]["size"]["conditions"][0].update(
                {"threshold": 999_999}
            ),
        )
        assert engine_at(registry_dir).registry_hash != before

    def test_custom_metric_dropin_flips_the_hash(self, registry_dir):
        """`custom/` ships absent; dropping a metric in is how users extend."""
        before = engine_at(registry_dir).registry_hash
        (registry_dir / "custom").mkdir(exist_ok=True)
        (registry_dir / "custom" / "extra.yaml").write_text(
            yaml.safe_dump(
                {
                    "element": "quality_business",
                    "metrics": {
                        "made_up_metric": {
                            "name": "Made Up",
                            "module": "builtin.profitability",
                            "function": "compute_roce_avg",
                            "inputs": ["financials"],
                            "scoring": {
                                "mode": "threshold",
                                "direction": "higher_is_better",
                                "thresholds": [1, 2, 3, 4, 5, 6],
                                "weight": 0.01,
                            },
                            "display": {
                                "format": "{:.1f}%",
                                "section": "quality_scorecard",
                            },
                        }
                    },
                }
            )
        )
        assert engine_at(registry_dir).registry_hash != before

    def test_macro_assumption_change_flips_the_hash(self, registry_dir):
        """Macro reaches every metric as a parameter default — DCF moves with it."""
        base = engine_at(registry_dir, macro={"discount_rate": 0.12})
        shifted = engine_at(registry_dir, macro={"discount_rate": 0.14})

        assert base.registry_hash != shifted.registry_hash

    def test_history_waiver_change_flips_the_hash(self, registry_dir):
        before = engine_at(registry_dir).registry_hash
        edit_yaml(
            registry_dir / "registry.yaml",
            lambda c: c.update({"history_waiver_mcap": 1.0}),
        )
        assert engine_at(registry_dir).registry_hash != before


class TestEffectiveGates:
    """A registry with no gate section is still governed by the code defaults."""

    def test_absent_gate_section_hashes_as_the_shipped_defaults(self, registry_dir):
        edit_yaml(
            registry_dir / "registry.yaml",
            lambda c: c.pop("eligibility_gates", None),
        )
        engine = engine_at(registry_dir)

        assert engine.gates == {}
        assert effective_gates(engine.gates) == DEFAULT_GATES

    def test_resolution_helper_matches_what_the_evaluator_applies(self):
        from boundless100x.compute_engine.eligibility import EligibilityEvaluator

        assert EligibilityEvaluator(effective_gates({})).gates == DEFAULT_GATES
        assert EligibilityEvaluator(effective_gates(None)).gates == DEFAULT_GATES

    def test_declared_gates_win_over_the_defaults(self):
        declared = {"g": {"label": "G", "conditions": []}}
        assert effective_gates(declared) == declared


class TestProvenanceIsNotSemantics:
    def test_renaming_a_metric_file_does_not_fragment_history(self, registry_dir):
        """`_source_file` moves; what the metrics compute does not."""
        before = engine_at(registry_dir).registry_hash
        elements = registry_dir / "elements"
        (elements / "longevity.yaml").rename(elements / "longevity_renamed.yaml")

        assert engine_at(registry_dir).registry_hash == before


# The two regimes the shipped registry currently declares. Pinned so a
# presentation-layer change cannot move them without a test saying so.
#
# These are NOT arbitrary constants to re-baseline when a test goes red. A
# change here is a change of scoring regime: every score-history row written
# before it becomes uncomparable to every row after, permanently, because the
# log is append-only. Update them only alongside a deliberate weight, gate,
# threshold, macro or metric-definition change — never to make a red test green.
SHIPPED_REGISTRY_HASH = "8e3c4115e9cf"
SHIPPED_FORWARD_SIGNAL_HASH = "f693039de714"


def add_key(registry_dir, filename: str, metric_id: str, key: str, value):
    """Add one top-level key to one metric's definition."""
    target = registry_dir / "elements" / filename
    config = yaml.safe_load(target.read_text())
    config["metrics"][metric_id][key] = value
    target.write_text(yaml.safe_dump(config))


PRESENTATION_BLOCK = {
    "unit": "percent",
    "direction": "higher_is_better",
    "bands": [[20.0, "strong"], [12.0, "adequate"]],
    "low_label": "weak",
}


class TestPresentationIsNotSemantics:
    """A metric's *presentation* declaration must not move either regime.

    R11 puts the unit, direction of goodness and interpretation bands beside
    the scoring config, in the same YAML the hash payload is built from. R17
    forbids that from moving a hash. Nothing about how a number is displayed
    can change the number, so a presentation key that fragmented history would
    be recording a regime change that never happened — and the fragmentation
    is unrecoverable, since score history is append-only.
    """

    def test_the_shipped_registry_hashes_are_what_they_were(self):
        """The filter widening is a no-op until the first declaration exists.

        This is what makes the change provably safe at the moment it lands:
        excluding a key no metric carries yet cannot alter the payload.
        """
        engine = ComputeEngine()
        assert engine.registry_hash == SHIPPED_REGISTRY_HASH
        assert engine.forward_signal_hash == SHIPPED_FORWARD_SIGNAL_HASH

    def test_declaring_presentation_on_a_scored_metric_leaves_the_scoring_hash(
        self, registry_dir
    ):
        before = engine_at(registry_dir).registry_hash
        add_key(
            registry_dir, "quality_business.yaml", "roce_5yr_avg",
            "presentation", PRESENTATION_BLOCK,
        )

        assert engine_at(registry_dir).registry_hash == before

    def test_declaring_presentation_on_a_zero_weight_metric_leaves_the_forward_hash(
        self, registry_dir
    ):
        """The split reads in both directions, so the exclusion must too."""
        before = engine_at(registry_dir).forward_signal_hash
        add_key(
            registry_dir, "forward_growth.yaml", "promises_kept_ratio",
            "presentation", PRESENTATION_BLOCK,
        )

        assert engine_at(registry_dir).forward_signal_hash == before

    def test_an_ordinary_key_still_moves_the_scoring_hash(self, registry_dir):
        """The filter excludes one named key, not everything unrecognised.

        Without this, a filter that quietly stopped hashing anything would pass
        every test above while destroying the regime stamp entirely.
        """
        before = engine_at(registry_dir).registry_hash
        add_key(
            registry_dir, "quality_business.yaml", "roce_5yr_avg",
            "some_new_semantic_key", {"weight_multiplier": 2.0},
        )

        assert engine_at(registry_dir).registry_hash != before


def drop_in(registry_dir, weight: float, element: str = "forward_growth",
            metric_id: str = "made_up_signal", **scoring):
    """Write a custom metric at a given weight and return the engine."""
    scoring = {
        "mode": "threshold",
        "direction": "higher_is_better",
        "thresholds": [1, 2, 3, 4, 5, 6],
        "weight": weight,
        **scoring,
    }
    (registry_dir / "custom").mkdir(exist_ok=True)
    (registry_dir / "custom" / "extra.yaml").write_text(
        yaml.safe_dump(
            {
                "element": element,
                "metrics": {
                    metric_id: {
                        "name": "Made Up",
                        "module": "builtin.profitability",
                        "function": "compute_roce_avg",
                        "inputs": ["ratios"],
                        "scoring": scoring,
                        "display": {"format": "{:.1f}%", "section": "forward_signals"},
                    }
                },
            }
        )
    )
    return engine_at(registry_dir)


class TestForwardSignalRegimeIsSeparate:
    """A metric that provably cannot move a score is not in the scoring hash.

    This is the whole point of KTD8: tuning a zero-weight signal must leave the
    scoring regime — and therefore every ticker's momentum baseline — alone.
    """

    def test_zero_weight_metric_does_not_move_the_scoring_hash(self, registry_dir):
        before = engine_at(registry_dir).registry_hash
        assert drop_in(registry_dir, weight=0.0).registry_hash == before

    def test_zero_weight_metric_does_move_the_forward_signal_hash(self, registry_dir):
        before = engine_at(registry_dir).forward_signal_hash
        assert drop_in(registry_dir, weight=0.0).forward_signal_hash != before

    def test_tuning_a_zero_weight_metric_leaves_the_scoring_hash_alone(self, registry_dir):
        """The Phase 5 case: recalibrating a forward signal must not reset momentum."""
        base = drop_in(registry_dir, weight=0.0, thresholds=[1, 2, 3])
        tuned = drop_in(registry_dir, weight=0.0, thresholds=[9, 8, 7])

        assert base.registry_hash == tuned.registry_hash
        assert base.forward_signal_hash != tuned.forward_signal_hash

    def test_a_scored_threshold_change_leaves_the_forward_signal_hash_alone(
        self, registry_dir
    ):
        """The split reads in both directions, or it is not a split."""
        before = engine_at(registry_dir).forward_signal_hash
        target = registry_dir / "elements" / "quality_business.yaml"
        config = yaml.safe_load(target.read_text())
        config["metrics"]["roce_5yr_avg"]["scoring"]["thresholds"] = [1, 2, 3, 4, 5, 6]
        target.write_text(yaml.safe_dump(config))

        assert engine_at(registry_dir).forward_signal_hash == before

    def test_element_weights_do_not_reach_the_forward_signal_hash(self, registry_dir):
        before = engine_at(registry_dir).forward_signal_hash
        edit_yaml(
            registry_dir / "registry.yaml",
            lambda c: c["element_weights"].update({"growth": 0.30}),
        )
        assert engine_at(registry_dir).forward_signal_hash == before

    def test_macro_moves_both_regimes(self, registry_dir):
        """Macro is a parameter default for every metric, scored or not."""
        base = engine_at(registry_dir, macro={"discount_rate": 0.12})
        shifted = engine_at(registry_dir, macro={"discount_rate": 0.14})

        assert base.registry_hash != shifted.registry_hash
        assert base.forward_signal_hash != shifted.forward_signal_hash

    def test_forward_signal_hash_covers_the_extraction_schema(self, registry_dir):
        """A prompt/field-schema change must be visible in score history."""
        from boundless100x import forward_growth_schema

        engine = engine_at(registry_dir)
        before = engine.forward_signal_hash
        original = forward_growth_schema.SCHEMA_VERSION
        try:
            forward_growth_schema.SCHEMA_VERSION = original + 1
            assert engine_at(registry_dir).forward_signal_hash != before
        finally:
            forward_growth_schema.SCHEMA_VERSION = original

    def test_forward_signal_hash_is_a_short_hex_digest(self):
        digest = ComputeEngine().forward_signal_hash
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)

    def test_forward_signal_hash_is_stable_across_loads(self):
        assert ComputeEngine().forward_signal_hash == ComputeEngine().forward_signal_hash

    def test_the_two_hashes_are_not_the_same_digest(self):
        engine = ComputeEngine()
        assert engine.registry_hash != engine.forward_signal_hash
