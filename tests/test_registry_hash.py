"""Registry hash: the scoring-regime stamp carried by every score-history row.

Phase 2 reads score history as fundamental momentum — a company moving 6.2 to
6.8 is treated as improving. If a weight, threshold, gate, or macro assumption
changed between those two runs, the movement is an artefact of the ruler, not
the company. The hash exists so a trajectory diff can refuse to compare across
regimes; these tests pin what counts as a regime change.
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
