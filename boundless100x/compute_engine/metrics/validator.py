"""YAML schema validation for metric registry files."""

REQUIRED_FIELDS = ["name", "module", "function", "inputs", "scoring", "display"]
VALID_DIRECTIONS = ["higher_is_better", "lower_is_better"]
VALID_MODES = [
    "threshold",
    "range_optimal",
    "categorical",
    "sector_relative_percentile",
    "trend_direction",
    "comparison_to_actual",
]


def validate_registry(metrics: dict) -> list[str]:
    """Validate all metric definitions. Returns list of errors (empty = valid)."""
    errors = []
    for metric_id, config in metrics.items():
        src = config.get("_source_file", "unknown")
        prefix = f"[{src}] {metric_id}"

        for field in REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"{prefix}: missing '{field}'")

        scoring = config.get("scoring", {})
        if "weight" not in scoring:
            errors.append(f"{prefix}: scoring.weight required")

        mode = scoring.get("mode", "threshold")
        if mode not in VALID_MODES:
            errors.append(f"{prefix}: invalid mode '{mode}'")

        if mode == "threshold":
            if "thresholds" not in scoring:
                errors.append(f"{prefix}: threshold mode needs 'thresholds'")
            if scoring.get("direction") not in VALID_DIRECTIONS:
                errors.append(f"{prefix}: invalid direction '{scoring.get('direction')}'")
        elif mode == "range_optimal":
            if "optimal_range" not in scoring:
                errors.append(f"{prefix}: range_optimal mode needs 'optimal_range'")
        elif mode == "categorical":
            if "categories" not in scoring:
                errors.append(f"{prefix}: categorical mode needs 'categories'")
        elif mode == "trend_direction":
            if "direction" not in scoring:
                errors.append(f"{prefix}: trend_direction mode needs 'direction'")

    return errors
