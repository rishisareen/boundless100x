"""Peer comparison — runs the compute engine on target + peers side by side."""

import logging

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.base import MetricResult

logger = logging.getLogger(__name__)


def build_comparison_table(
    engine: ComputeEngine,
    target_ticker: str,
    target_results: dict[str, MetricResult],
    peer_results: dict[str, dict[str, MetricResult]],
) -> dict:
    """Build a side-by-side peer comparison table.

    Returns:
        {
            "metrics": ["roce_5yr_avg", "opm_5yr", ...],
            "companies": {
                "ASTRAL": {"roce_5yr_avg": 19.7, ...},
                "SUPREMEIND": {"roce_5yr_avg": 22.1, ...},
            },
            "ranks": {
                "ASTRAL": {"roce_5yr_avg": 2, ...},
            },
            "best_in_class": {"roce_5yr_avg": "SUPREMEIND", ...},
        }
    """
    comparable_metrics = engine.get_peer_comparable_metrics()

    # Collect values
    companies = {}

    # Target
    companies[target_ticker] = {}
    for mid in comparable_metrics:
        result = target_results.get(mid)
        if result and result.ok and isinstance(result.value, (int, float)):
            companies[target_ticker][mid] = result.value
        else:
            companies[target_ticker][mid] = None

    # Peers
    for peer_ticker, results in peer_results.items():
        companies[peer_ticker] = {}
        for mid in comparable_metrics:
            result = results.get(mid)
            if result and result.ok and isinstance(result.value, (int, float)):
                companies[peer_ticker][mid] = result.value
            else:
                companies[peer_ticker][mid] = None

    # Compute ranks per metric
    ranks = {ticker: {} for ticker in companies}
    best_in_class = {}

    for mid in comparable_metrics:
        config = engine.metrics.get(mid, {})
        scoring = config.get("scoring", {})
        direction = scoring.get("direction", "higher_is_better")

        # Get non-null values with tickers
        values = [
            (ticker, companies[ticker].get(mid))
            for ticker in companies
            if companies[ticker].get(mid) is not None
        ]

        if not values:
            continue

        # Sort: best first
        reverse = direction != "lower_is_better"
        values.sort(key=lambda x: x[1], reverse=reverse)

        for rank, (ticker, val) in enumerate(values, 1):
            ranks[ticker][mid] = rank

        best_in_class[mid] = values[0][0]

    return {
        "metrics": comparable_metrics,
        "companies": companies,
        "ranks": ranks,
        "best_in_class": best_in_class,
    }
