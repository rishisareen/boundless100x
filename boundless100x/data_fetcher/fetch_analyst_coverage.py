"""Fetch analyst coverage data from Trendlyne."""

import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

TRENDLYNE_BASE = "https://trendlyne.com"
TRENDLYNE_SEARCH_URL = f"{TRENDLYNE_BASE}/member/api/ac_snames/stock/"


class AnalystCoverageFetcher(BaseFetcher):
    """Fetch analyst coverage count and consensus from Trendlyne.

    Used for the "unknown-ness" metric in SQGLP Size element.

    Trendlyne stock pages use numeric IDs in URLs:
        /equity/{numeric_id}/{TICKER}/{slug}/
    We first resolve the ticker to its numeric ID via the search API,
    then fetch the consensus estimates page for analyst data.
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

    def _resolve_trendlyne_id(self, ticker: str) -> dict | None:
        """Resolve NSE ticker to Trendlyne stock metadata via search API.

        Returns dict with keys: k (numeric ID), slugname, nexturl, etc.
        Returns None if ticker not found.
        """
        try:
            # Trendlyne search API requires X-Requested-With header
            resp = self._get(
                TRENDLYNE_SEARCH_URL,
                params={"term": ticker},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            data = resp.json()

            if not isinstance(data, list) or not data:
                return None

            # Check for "no matches" response
            if len(data) == 1 and "headline" in data[0]:
                logger.debug(f"Trendlyne: no match for '{ticker}'")
                return None

            # Find exact NSE ticker match first, then fall back to first result
            for entry in data:
                if entry.get("NSEcode", "").upper() == ticker.upper():
                    return entry

            # Fall back to first result if no exact match
            return data[0]

        except Exception as e:
            logger.debug(f"Trendlyne search API failed for {ticker}: {e}")
            return None

    def _fetch_trendlyne(self, ticker: str) -> dict | None:
        """Fetch analyst coverage from Trendlyne.

        1. Resolve ticker to Trendlyne numeric ID via search API
        2. Fetch the consensus estimates page
        3. Parse analyst count and target price
        """
        # Step 1: Resolve ticker to Trendlyne URL
        stock_info = self._resolve_trendlyne_id(ticker)
        if stock_info is None:
            logger.warning(f"Could not resolve {ticker} on Trendlyne")
            return None

        numeric_id = stock_info.get("k")
        slugname = stock_info.get("slugname", "")
        nse_code = stock_info.get("NSEcode", ticker)
        sector_name = stock_info.get("sectorName")

        if numeric_id is None:
            logger.warning(f"No Trendlyne numeric ID for {ticker}")
            return None

        logger.debug(f"Trendlyne: {ticker} → ID={numeric_id}, slug={slugname}, sector={sector_name}")

        # Step 2: Try consensus estimates page first (richer data), fallback to overview
        urls_to_try = [
            f"{TRENDLYNE_BASE}/equity/consensus-estimates/{numeric_id}/{nse_code}/{slugname}/",
            f"{TRENDLYNE_BASE}/equity/{numeric_id}/{nse_code}/{slugname}/",
        ]

        resp = None
        for url in urls_to_try:
            try:
                resp = self._get(url)
                break
            except Exception:
                continue

        if resp is None:
            logger.warning(f"All Trendlyne page fetches failed for {ticker}")
            return None

        # Step 3: Parse analyst data
        try:
            soup = BeautifulSoup(resp.text, "html.parser")

            result = {
                "count": None,
                "avg_target": None,
                "consensus": None,
                "sector": sector_name,
                "source": "trendlyne",
            }

            text = soup.get_text()

            # Find analyst count: "X analysts" pattern
            match = re.search(r"(\d+)\s+analyst", text, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))

            # Find target price: "target ... Rs/₹ XXXX" pattern
            match = re.search(r"target.*?(?:Rs\.?|₹)\s*([\d,]+)", text, re.IGNORECASE)
            if match:
                result["avg_target"] = float(match.group(1).replace(",", ""))

            # Determine consensus from Buy/Hold/Sell mentions
            consensus_matches = re.findall(
                r"\b(strong\s+buy|buy|hold|sell|strong\s+sell|outperform|underperform)\b",
                text,
                re.IGNORECASE,
            )
            if consensus_matches:
                # Normalize and find the most frequent
                normalized = [m.lower().strip() for m in consensus_matches]
                from collections import Counter
                most_common = Counter(normalized).most_common(1)
                if most_common:
                    result["consensus"] = most_common[0][0]

            if result["count"]:
                logger.info(
                    f"Trendlyne: {ticker} — {result['count']} analysts, "
                    f"target ₹{result['avg_target']}, consensus: {result['consensus']}"
                )

            return result
        except Exception as e:
            logger.warning(f"Trendlyne parse failed for {ticker}: {e}")
            return None
