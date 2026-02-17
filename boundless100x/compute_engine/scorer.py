"""SQGLP Scorer — maps metric values to 0-10 element scores and a weighted composite."""

import logging

from boundless100x.compute_engine.metrics.base import MetricResult

logger = logging.getLogger(__name__)


class SQGLPScorer:
    """Compute per-element scores (0-10) and weighted composite from metric results."""

    def __init__(self, metrics_config: dict, element_weights: dict):
        self.metrics_config = metrics_config
        self.element_weights = element_weights

    def score(self, results: dict[str, MetricResult]) -> dict:
        """Compute SQGLP scores.

        Returns:
            {
                "elements": {"size": 7.2, "quality_business": 8.1, ...},
                "composite": 7.6,
                "details": {metric_id: {"value": X, "score": Y, "weight": Z}, ...}
            }
        """
        element_weighted_scores: dict[str, float] = {}
        element_total_weights: dict[str, float] = {}
        details: dict[str, dict] = {}

        for metric_id, result in results.items():
            if not result.ok:
                details[metric_id] = {
                    "value": None,
                    "score": None,
                    "weight": 0,
                    "error": result.error,
                }
                continue

            config = self.metrics_config.get(metric_id)
            if config is None:
                continue

            element = config["element"]
            scoring_config = config["scoring"]
            weight = scoring_config.get("weight", 0.0)

            if weight == 0:
                # Display-only metric (like composite QG quadrant)
                details[metric_id] = {
                    "value": result.value,
                    "score": None,
                    "weight": 0,
                    "flags": result.flags,
                }
                continue

            raw_score = self._compute_raw_score(result, scoring_config)

            details[metric_id] = {
                "value": result.value,
                "score": raw_score,
                "weight": weight,
                "flags": result.flags,
            }

            element_weighted_scores.setdefault(element, 0.0)
            element_total_weights.setdefault(element, 0.0)
            element_weighted_scores[element] += raw_score * weight
            element_total_weights[element] += weight

        # Normalize element scores to 0-10
        elements = {}
        for el in self.element_weights:
            total_w = element_total_weights.get(el, 0)
            if total_w > 0:
                elements[el] = element_weighted_scores[el] / total_w * 10
            else:
                elements[el] = None

        # Weighted composite (exclude None elements)
        total_weight = 0.0
        composite = 0.0
        for el, w in self.element_weights.items():
            if elements.get(el) is not None:
                composite += elements[el] * w
                total_weight += w

        if total_weight > 0:
            composite = composite / total_weight
        else:
            composite = 0.0

        return {
            "elements": elements,
            "composite": round(composite, 2),
            "details": details,
        }

    def _compute_raw_score(self, result: MetricResult, config: dict) -> float:
        """Map a metric value to a 0-1 score using the configured method.

        Args:
            result: Full MetricResult (needed for trend_direction mode to access metadata).
            config: Scoring config from YAML (mode, thresholds, direction, etc.).
        """
        mode = config.get("mode", "threshold")
        value = result.value

        if mode == "threshold":
            return self._threshold_score(
                value, config.get("thresholds", []), config.get("direction", "higher_is_better")
            )
        elif mode == "range_optimal":
            return self._range_score(value, config.get("optimal_range", [0, 100]))
        elif mode == "categorical":
            categories = config.get("categories", {})
            return categories.get(value, 0) / 10.0
        elif mode == "sector_relative_percentile":
            # Without sector data, use absolute thresholds as fallback
            direction = config.get("direction", "lower_is_better")
            if direction == "lower_is_better":
                return self._threshold_score(value, [80, 60, 45, 30, 20, 12], "lower_is_better")
            else:
                return self._threshold_score(value, [5, 10, 15, 20, 30, 50], "higher_is_better")
        elif mode == "trend_direction":
            return self._trend_score(result, config)
        elif mode == "comparison_to_actual":
            return self._threshold_score(value, [40, 30, 25, 20, 15, 10], "lower_is_better")
        else:
            return 0.5  # Unknown mode fallback

    def _threshold_score(
        self, value: float, thresholds: list, direction: str
    ) -> float:
        """Score using threshold buckets.

        Thresholds define 7 zones (below all thresholds through above all).
        Each zone maps to an evenly-spaced score between 0 and 1.
        """
        if not isinstance(value, (int, float)):
            return 0.0

        if direction == "lower_is_better":
            # Thresholds are in descending order: [worst, ..., best]
            # Below the smallest threshold = best score
            for i, t in enumerate(thresholds):
                if value >= t:
                    return i / len(thresholds)
            return 1.0
        else:
            # Thresholds are in ascending order: [worst, ..., best]
            # Above the largest threshold = best score
            for i, t in enumerate(thresholds):
                if value < t:
                    return i / len(thresholds)
            return 1.0

    def _range_score(self, value: float, optimal_range: list) -> float:
        """Score based on distance from optimal range. In range = 1.0."""
        if not isinstance(value, (int, float)):
            return 0.0

        low, high = optimal_range
        if low <= value <= high:
            return 1.0

        # Score decreases with distance from range
        if value < low:
            distance = low - value
            range_width = high - low
            return max(0.0, 1.0 - distance / (range_width * 3))
        else:
            distance = value - high
            range_width = high - low
            return max(0.0, 1.0 - distance / (range_width * 3))

    def _trend_score(self, result: MetricResult, config: dict) -> float:
        """Score based on trend direction from metadata.

        Trend metrics store the current level as .value but the TREND
        (change over time) in .metadata. We score the trend, not the level.

        Supported directions:
        - "stable_or_rising_is_better": promoter holding (rising/stable = good)
        - "declining_is_better": working capital days (declining = good)
        """
        preferred = config.get("direction", "declining_is_better")

        # Extract trend magnitude from metadata
        trend = None
        if result.metadata:
            # Try known metadata keys in order
            for key in ("change_pp", "trend_change", "trend"):
                if key in result.metadata:
                    try:
                        trend = float(result.metadata[key])
                    except (TypeError, ValueError):
                        pass
                    break

        if trend is not None:
            if preferred == "stable_or_rising_is_better":
                # Promoter holding: rising or stable = good, declining = bad
                if trend > 5:
                    return 1.0
                elif trend > 2:
                    return 0.85
                elif trend > -2:
                    return 0.65  # Stable (±2pp)
                elif trend > -5:
                    return 0.35
                elif trend > -10:
                    return 0.15
                else:
                    return 0.0
            else:
                # declining_is_better: e.g., working capital days
                if trend < -10:
                    return 1.0
                elif trend < -5:
                    return 0.8
                elif trend < 0:
                    return 0.65
                elif trend == 0:
                    return 0.5
                elif trend < 5:
                    return 0.35
                elif trend < 10:
                    return 0.2
                else:
                    return 0.0

        # Fallback: use flags if no quantitative trend
        has_improving = any("improving" in f or "increasing" in f for f in result.flags)
        has_worsening = any("worsening" in f or "reducing" in f for f in result.flags)

        if preferred == "stable_or_rising_is_better":
            if has_improving:
                return 0.8
            elif has_worsening:
                return 0.2
        else:
            if has_improving:
                return 0.8
            elif has_worsening:
                return 0.2

        return 0.5  # No trend data available
