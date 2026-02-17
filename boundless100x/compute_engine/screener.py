"""Screener — apply preset or custom filters to a universe of companies."""

import logging
from pathlib import Path

import yaml

from boundless100x.compute_engine.metrics.base import MetricResult

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).parent / "metrics" / "presets"


class Screener:
    """Apply filter criteria to computed metric results and rank survivors."""

    def __init__(self):
        self.presets = self._load_presets()

    def _load_presets(self) -> dict:
        presets = {}
        if not PRESETS_DIR.exists():
            return presets
        for f in PRESETS_DIR.glob("*.yaml"):
            with open(f) as fh:
                data = yaml.safe_load(fh)
                key = f.stem
                presets[key] = data
        return presets

    def list_presets(self) -> list[dict]:
        """Return available screening presets."""
        return [
            {"key": k, "name": v.get("name", k), "description": v.get("description", "")}
            for k, v in self.presets.items()
        ]

    def screen(
        self,
        universe: dict[str, dict[str, MetricResult]],
        scores: dict[str, dict] | None = None,
        preset: str | None = None,
        filters: dict | None = None,
        rankings: dict | None = None,
    ) -> list[dict]:
        """Screen a universe of companies.

        Args:
            universe: {ticker: {metric_id: MetricResult}} for each company.
            scores: {ticker: scores_dict} with composite scores.
            preset: Name of a preset filter set (e.g., "compounders").
            filters: Custom filter dict {metric_id: {min: X, max: Y}}.
            rankings: Custom ranking config {primary: metric_id, secondary: ...}.

        Returns:
            Sorted list of dicts with ticker, metric values, and rank.
        """
        if preset:
            preset_config = self.presets.get(preset)
            if not preset_config:
                raise ValueError(
                    f"Unknown preset '{preset}'. Available: {list(self.presets.keys())}"
                )
            filters = preset_config.get("filters", {})
            rankings = preset_config.get("rankings", {})
            logger.info(f"Using preset: {preset_config.get('name', preset)}")

        if not filters:
            filters = {}

        # Apply filters
        survivors = []
        for ticker, metrics in universe.items():
            passes = True
            metric_vals = {}

            for metric_id, criteria in filters.items():
                # Special case: sqglp_composite comes from scores dict
                if metric_id == "sqglp_composite" and scores:
                    val = scores.get(ticker, {}).get("composite")
                else:
                    result = metrics.get(metric_id)
                    if not result or not result.ok:
                        passes = False
                        break
                    val = result.value

                if not isinstance(val, (int, float)):
                    passes = False
                    break

                metric_vals[metric_id] = val

                if "min" in criteria and val < criteria["min"]:
                    passes = False
                    break
                if "max" in criteria and val > criteria["max"]:
                    passes = False
                    break

            if passes:
                # Collect all numeric metrics for the survivor
                entry = {"ticker": ticker}
                for mid, result in metrics.items():
                    if result.ok and isinstance(result.value, (int, float)):
                        entry[mid] = result.value
                if scores and ticker in scores:
                    entry["sqglp_composite"] = scores[ticker].get("composite")
                survivors.append(entry)

        logger.info(
            f"Screening: {len(survivors)}/{len(universe)} passed "
            f"({len(filters)} filters)"
        )

        # Rank survivors
        if rankings:
            primary = rankings.get("primary", "sqglp_composite")
            secondary = rankings.get("secondary")

            # Determine sort direction (lower is better for PE/PEG, higher for others)
            lower_is_better = {"pe_ttm", "peg_ratio", "trailing_peg", "ev_ebitda",
                               "debt_equity", "earnings_yield_spread"}

            reverse_primary = primary not in lower_is_better
            survivors.sort(
                key=lambda x: (
                    x.get(primary, float("inf") if not reverse_primary else float("-inf")),
                ),
                reverse=reverse_primary,
            )

        # Add rank
        for i, entry in enumerate(survivors, 1):
            entry["rank"] = i

        return survivors

    def screen_quick(
        self,
        tickers: list[str],
        service,
        preset: str = "compounders",
    ) -> list[dict]:
        """Screen a list of tickers using quick (no-peer) analysis.

        Args:
            tickers: List of NSE symbols to screen.
            service: Boundless100xService instance.
            preset: Preset name to apply.

        Returns:
            Sorted list of qualifying companies.
        """
        universe = {}
        scores_map = {}

        for ticker in tickers:
            try:
                result = service.analyze_quick(ticker)
                universe[ticker] = result.metrics
                scores_map[ticker] = result.scores
                ok = sum(1 for m in result.metrics.values() if m.ok)
                logger.info(f"  {ticker}: {ok} metrics computed")
            except Exception as e:
                logger.warning(f"  {ticker}: failed — {e}")

        return self.screen(
            universe=universe,
            scores=scores_map,
            preset=preset,
        )
