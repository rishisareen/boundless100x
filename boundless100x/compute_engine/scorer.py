"""SQGLP Scorer — maps metric values to 0-10 element scores and a weighted composite.

**A metric that measures nothing for this kind of company does not score it.**
`sector_applicability.yaml` has always declared which metrics are meaningless
for which sector, but for its first life only the *report* read it: the reader
was told that a lender's DCF margin means nothing while the composite that
lender was ranked on had already banked it. The distortion ran both ways at
once — EDELWEISS took a perfect 1.0 on `dcf_margin_of_safety` and a perfect 0.0
on `dupont_turnover`, one flattering and one damning, both computed off series
that inverted for the same reason. That is the failure the table was written to
prevent, arrived at one layer below where it was fixed.

The exclusion is **not** the same event as a metric that errored, and the two
must not share an outcome. A missing metric leaves the score thinner and
`_coverage` says so, because the evidence was wanted and could not be got. An
inapplicable one was never evidence here at all, so it leaves the denominator
entirely: counting it as absent would drive every lender under
`low_coverage_threshold` and cap its action for the crime of being a lender —
the same penalty in a new costume.
"""

import logging

from boundless100x.compute_engine.metrics.base import MetricResult, is_scorable

logger = logging.getLogger(__name__)


class SQGLPScorer:
    """Compute per-element scores (0-10) and weighted composite from metric results."""

    def __init__(self, metrics_config: dict, element_weights: dict,
                 history_waiver_mcap: float | None = None,
                 low_coverage_threshold: float = 0.85,
                 applicability=None):
        self.metrics_config = metrics_config
        self.element_weights = element_weights
        # Below this share of declared weight, the composite is flagged as
        # resting on thin evidence rather than presented as a peer of a
        # fully measured one.
        self.low_coverage_threshold = low_coverage_threshold
        # Below this market cap, metrics capped by a short observation window
        # are treated as missing rather than scored low. See _waived_for_history.
        self.history_waiver_mcap = history_waiver_mcap
        # A `SectorApplicability`, or None to score every metric everywhere.
        # None is the old behaviour and stays the default so a caller
        # constructing a scorer directly — the backtest, tests — is not
        # silently given a different regime than it asked for. The service
        # passes one.
        self.applicability = applicability

    def _excluded_for_sector(self, sector: str | None) -> dict[str, str]:
        """Metric id -> the table's own sentence saying why it does not apply.

        Empty for an unreviewed sector, which is the common case and is NOT a
        claim that everything applies — see `SectorApplicability`. Empty is
        also what a missing table gives, so scoring degrades to the
        pre-applicability regime rather than to no scoring at all.
        """
        if self.applicability is None or not sector:
            return {}
        try:
            return self.applicability.not_applicable_metrics(sector)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"Sector applicability lookup failed for {sector!r}: {exc} — "
                f"scoring every metric"
            )
            return {}

    def _flag_suppressed(self, sector: str | None) -> set[str]:
        """Inapplicable metrics whose flags must not reach a reader either."""
        if self.applicability is None or not sector:
            return set()
        try:
            return self.applicability.flag_suppressed_metrics(sector)
        except Exception:  # pragma: no cover - defensive
            return set()

    @staticmethod
    def _surviving_flags(metric_id, result, suppressed: set[str]) -> list:
        """A withdrawn metric's flags, or none when the table suppresses them."""
        if metric_id in suppressed:
            return []
        return list(getattr(result, "flags", None) or [])

    def _waived_for_history(self, results: dict) -> bool:
        """Whether short-window metrics should be excused for this company.

        Count-based longevity metrics score an absolute number of good years
        against a ten-year window, so a young company is capped by arithmetic
        rather than performance. For a small company that is a fact about the
        data; for a large one, thin history is itself a red flag — hence a
        threshold of its own, well below the 100x size gate. An unknown market
        cap never earns the waiver.
        """
        if not self.history_waiver_mcap:
            return False

        mcap = results.get("market_cap")
        if mcap is None or not getattr(mcap, "ok", False):
            return False
        if not isinstance(mcap.value, (int, float)):
            return False

        return mcap.value < self.history_waiver_mcap

    def score(self, results: dict[str, MetricResult], sector: str | None = None) -> dict:
        """Compute SQGLP scores.

        `sector` is the company's Screener breadcrumb sector. Supplied, any
        metric the applicability table declares meaningless for that sector is
        excluded from its element and from the coverage denominator. Omitted,
        every metric scores — the pre-applicability regime.

        Returns:
            {
                "elements": {"size": 7.2, "quality_business": 8.1, ...},
                "composite": 7.6,
                "details": {metric_id: {"value": X, "score": Y, "weight": Z}, ...},
                "not_applicable": {metric_id: reason, ...},
            }
        """
        element_weighted_scores: dict[str, float] = {}
        element_total_weights: dict[str, float] = {}
        details: dict[str, dict] = {}
        score_flags: list[str] = []

        waive_history = self._waived_for_history(results)
        excluded = self._excluded_for_sector(sector)
        suppressed = self._flag_suppressed(sector)
        # Only the exclusions that actually bit — a table entry for a metric
        # this registry never computed would otherwise be reported to the
        # reader as a metric withheld from them.
        applied_exclusions: dict[str, str] = {}

        for metric_id, result in results.items():
            config = self.metrics_config.get(metric_id)

            # **Inapplicability is checked before the error branch, and that
            # order is the whole of it.** A metric this company was never going
            # to be judged on does not become missing evidence by also failing
            # to compute — and for a lender the two arrive together as a rule
            # rather than a coincidence: `reverse_dcf_growth` errors with
            # "Negative average FCF" on every lender growing its book, which is
            # exactly why the table calls it meaningless. Checked the other way
            # round, JIOFIN's Price coverage read 0.553 instead of 0.627, and a
            # composite pushed under `low_coverage_threshold` that way has its
            # displayed action capped.
            if config is not None and metric_id in excluded:
                applied_exclusions[metric_id] = excluded[metric_id]
                entry = {
                    "value": result.value if result.ok else None,
                    "score": None,
                    "weight": 0,
                    "not_applicable": excluded[metric_id],
                    "flags": self._surviving_flags(metric_id, result, suppressed),
                }
                if not result.ok:
                    entry["error"] = result.error
                details[metric_id] = entry
                continue

            if not result.ok:
                details[metric_id] = {
                    "value": None,
                    "score": None,
                    "weight": 0,
                    "error": result.error,
                }
                continue

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

            # Computed, and arithmetically correct, but not evidence — see
            # UNSCORABLE_FLAGS. Waived like a short-window metric rather than
            # scored, so the remaining weights renormalise and coverage records
            # the gap: the evidence really is missing, unlike an inapplicable
            # metric above.
            if not is_scorable(result):
                details[metric_id] = {
                    "value": result.value,
                    "score": None,
                    "weight": 0,
                    "waived": "not_a_reading",
                    "flags": result.flags,
                }
                if "unscorable_readings" not in score_flags:
                    score_flags.append("unscorable_readings")
                continue

            if waive_history and any(f.startswith("short_window_") for f in result.flags):
                # Excluded rather than scored low; remaining weights renormalise.
                details[metric_id] = {
                    "value": result.value,
                    "score": None,
                    "weight": 0,
                    "waived": "short_history_smallcap",
                    "flags": result.flags,
                }
                if "short_history_smallcap" not in score_flags:
                    score_flags.append("short_history_smallcap")
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

        coverage = self._coverage(element_total_weights, details, applied_exclusions)
        if coverage["composite"] < self.low_coverage_threshold:
            score_flags.append("low_data_coverage")

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
            "flags": score_flags,
            "coverage": coverage,
            # What this company was NOT judged on, and the table's reason for
            # each. Rendered rather than merely honoured: a composite that
            # quietly stopped counting five metrics is a different number, and
            # a reader comparing it against another company's is entitled to
            # know which questions were not asked here.
            "not_applicable": applied_exclusions,
            # The subset whose flags were dropped too, so a surface rebuilding
            # a signal list from raw metric results can apply the same rule
            # rather than reaching its own conclusion.
            "flags_suppressed": sorted(suppressed & set(applied_exclusions)),
            "sector": sector,
        }

    def _declared_weights(
        self, excluded: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Total weight each element would carry if every metric computed.

        `excluded` metrics are struck from the total rather than counted as
        unmet: they are not evidence this company was missing, they are
        questions that do not apply to it.
        """
        skip = excluded or {}
        declared: dict[str, float] = {}
        for metric_id, config in self.metrics_config.items():
            if metric_id in skip:
                continue
            weight = config.get("scoring", {}).get("weight", 0) or 0
            if weight > 0:
                declared[config["element"]] = declared.get(config["element"], 0) + weight
        return declared

    def _coverage(
        self, scored_weights: dict, details: dict,
        excluded: dict[str, str] | None = None,
    ) -> dict:
        """How much of the declared evidence actually reached the score.

        A renormalised composite reads like a full one, so the share of weight
        behind it has to travel with it. Both errored and deliberately waived
        metrics count as absent — the score is thinner either way.

        Sector-inapplicable metrics are the exception, and leave *both* sides
        of the ratio. Counting them as absent would report a lender as thinly
        evidenced for the whole of a metric set that was never about lenders,
        and `low_data_coverage` caps the displayed action — so the penalty
        removed from the score would have come straight back as a penalty on
        the recommendation.
        """
        declared = self._declared_weights(excluded)

        by_element = {}
        for element in self.element_weights:
            total = declared.get(element, 0)
            by_element[element] = (
                round(scored_weights.get(element, 0) / total, 3) if total > 0 else None
            )

        weighted, total_weight = 0.0, 0.0
        for element, element_weight in self.element_weights.items():
            if declared.get(element, 0) <= 0:
                continue
            weighted += (by_element[element] or 0) * element_weight
            total_weight += element_weight

        return {
            "composite": round(weighted / total_weight, 3) if total_weight else 0.0,
            "elements": by_element,
            # Evidence that was wanted and not got. A sector-inapplicable
            # metric is not listed: it left the denominator too, so reporting
            # it here would describe the score as short of something it was
            # never measured against.
            "unscored": sorted(
                mid for mid, d in details.items()
                if d.get("score") is None
                and "not_applicable" not in d
                and self.metrics_config.get(mid, {})
                .get("scoring", {}).get("weight", 0)
            ),
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
