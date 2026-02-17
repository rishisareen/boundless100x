"""Fetch analyst coverage data from Trendlyne."""

import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

TRENDLYNE_BASE = "https://trendlyne.com"


class AnalystCoverageFetcher(BaseFetcher):
    """Fetch analyst coverage count and consensus from Trendlyne.

    Used for the "unknown-ness" metric in SQGLP Size element.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch(
        self,
        ticker: str,
        output_dir: str | None = None,
    ) -> dict:
        """Fetch analyst coverage for a ticker.

        Returns dict with keys: count, avg_target, consensus, source.
        """
        cache_key = f"analyst_coverage_{ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache hit: {cache_key}")
            return cached

        result = self._fetch_trendlyne(ticker)

        if result:
            self.cache.set(cache_key, result)

        if output_dir and result:
            path = Path(output_dir) / ticker
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "analyst_coverage.json", "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved {ticker}/analyst_coverage.json")

        return result or {"count": None, "avg_target": None, "consensus": None, "source": "trendlyne"}

    def _fetch_trendlyne(self, ticker: str) -> dict | None:
        """Scrape analyst coverage from Trendlyne.

        Trendlyne URLs use a slug format. We try the forecasts page
        which is more reliably structured for analyst data.
        """
        # Try multiple URL patterns
        urls_to_try = [
            f"{TRENDLYNE_BASE}/fundamentals/stock-summary/{ticker}/",
            f"{TRENDLYNE_BASE}/equity/{ticker}/",
        ]

        resp = None
        for url in urls_to_try:
            try:
                resp = self._get(url)
                break
            except Exception:
                continue

        if resp is None:
            logger.warning(f"All Trendlyne URL patterns failed for {ticker}")
            return None

        try:
            soup = BeautifulSoup(resp.text, "html.parser")

            result = {
                "count": None,
                "avg_target": None,
                "consensus": None,
                "source": "trendlyne",
            }

            # Look for analyst count in various possible locations
            text = soup.get_text()

            # Try to find "X analysts" pattern
            match = re.search(r"(\d+)\s+analyst", text, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))

            # Try to find target price
            match = re.search(r"target.*?(?:Rs\.?|₹)\s*([\d,]+)", text, re.IGNORECASE)
            if match:
                result["avg_target"] = float(match.group(1).replace(",", ""))

            return result
        except Exception as e:
            logger.warning(f"Trendlyne fetch failed for {ticker}: {e}")
            return None
