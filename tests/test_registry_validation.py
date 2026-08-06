"""Registry validation, and what a zero-weight metric is excused from.

A metric at `weight: 0` is display-only — the scorer returns from its
`weight == 0` branch before `_compute_raw_score` is reached, so thresholds,
ranges and categories declared for it are never read. Demanding them anyway
produced exactly what you would expect: zero-weight metrics carrying invented
category tables purely to get past startup. Phase 2 adds five more, so the
rule is stated here rather than worked around five more times.
"""

from boundless100x.compute_engine.metrics.validator import validate_registry


def metric(weight, **scoring):
    return {
        "m": {
            "_source_file": "test.yaml",
            "name": "M",
            "module": "builtin.profitability",
            "function": "compute_roce_avg",
            "inputs": ["ratios"],
            "scoring": {"weight": weight, **scoring},
            "display": {"format": "{}", "section": "s"},
        }
    }


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


class TestTheShippedRegistryIsSound:
    def test_it_loads_without_errors(self):
        from boundless100x.compute_engine.engine import ComputeEngine

        assert validate_registry(ComputeEngine().metrics) == []
