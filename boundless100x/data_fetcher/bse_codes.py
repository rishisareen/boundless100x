"""Resolve a ticker to its BSE scrip code from BSE's own scrip master.

Screener used to render a bseindia.com link on the company page and the code
was scraped out of its URL. It no longer does — the live page contains no
occurrence of 'bseindia' and no scrip code — so `bse_code` came back None on
every fresh fetch, and annual-report downloads and BSE shareholding silently
degraded with it.

BSE publishes the full active equity list, which is the authoritative mapping
and changes rarely, so it is cached for a week and shared across tickers.

Two outcomes must stay distinguishable. A company genuinely not listed on BSE
(CDSL and BSE Ltd among the cached tickers — CDSL trades on NSE only, and BSE
Ltd cannot list on itself) is a fact about the company, not a failure. A
lookup that could not run is a failure. Reporting both as "no code" would have
the suite log an error every run for companies that are fine.
"""

import json
import logging
import re

import requests

from boundless100x.data_fetcher.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

SCRIP_MASTER_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
)
CACHE_KEY = "bse_scrip_master_active_equity"
CACHE_TTL_HOURS = 24 * 7

# Corporate suffixes carry no identifying signal when matching names.
NAME_NOISE = re.compile(
    r"\b(ltd|limited|ltd\.|inc|corporation|corp|company|co|the|india|"
    r"private|pvt|public)\b",
    re.I,
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json, text/plain, */*",
}


def _normalise_name(name: str) -> str:
    cleaned = NAME_NOISE.sub(" ", str(name))
    return re.sub(r"[^a-z0-9]+", "", cleaned.lower())


class BseCodeResolver:
    """Ticker -> BSE scrip code, from the exchange's own list."""

    def __init__(self, cache_dir: str | None = None, timeout: int = 30):
        self.cache = CacheManager(cache_dir=cache_dir, ttl_hours=CACHE_TTL_HOURS)
        self.timeout = timeout
        self._scrips: list[dict] | None = None
        self._index: dict | None = None
        self._lookup_failed = False

    # ── data ─────────────────────────────────────────────────────────────

    def _download_scrips(self) -> list[dict]:
        response = requests.get(
            SCRIP_MASTER_URL, headers=REQUEST_HEADERS, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _load(self) -> bool:
        """Populate the index, from cache when warm. False if unavailable."""
        if self._index is not None:
            return True
        if self._lookup_failed:
            return False

        cached = self.cache.get(CACHE_KEY)
        scrips = cached.get("scrips") if isinstance(cached, dict) else None

        if not scrips:
            try:
                scrips = self._download_scrips()
                self.cache.set(CACHE_KEY, {"scrips": scrips})
                logger.info(f"BSE scrip master: {len(scrips)} active equities")
            except Exception as e:
                logger.warning(f"Could not fetch the BSE scrip master: {e}")
                self._lookup_failed = True
                return False

        self._scrips = scrips
        self._index = self._build_index(scrips)
        return True

    @staticmethod
    def _build_index(scrips: list[dict]) -> dict:
        by_symbol, by_name = {}, {}
        for row in scrips:
            code = str(row.get("SCRIP_CD", "")).strip()
            if not code:
                continue
            symbol = str(row.get("scrip_id", "")).strip().upper()
            if symbol:
                by_symbol.setdefault(symbol, code)
            for field in ("Scrip_Name", "Issuer_Name"):
                key = _normalise_name(row.get(field, ""))
                if key:
                    by_name.setdefault(key, code)
        return {"symbol": by_symbol, "name": by_name}

    # ── resolution ───────────────────────────────────────────────────────

    def describe(self, ticker: str, company_name: str | None = None) -> dict:
        """Resolve, reporting how it went so callers can log honestly."""
        if not self._load():
            return {"status": "lookup_failed", "bse_code": None, "matched_on": None}

        symbol = (ticker or "").strip().upper()
        if symbol:
            code = self._index["symbol"].get(symbol)
            if code:
                return {"status": "resolved", "bse_code": code, "matched_on": "symbol"}

        if company_name:
            key = _normalise_name(company_name)
            # A name that normalises to almost nothing ("Ltd") would match
            # arbitrarily; require enough signal to be worth trusting.
            if len(key) >= 4:
                code = self._index["name"].get(key)
                if code:
                    return {
                        "status": "resolved",
                        "bse_code": code,
                        "matched_on": "company_name",
                    }

        return {"status": "not_listed_on_bse", "bse_code": None, "matched_on": None}

    def resolve(self, ticker: str, company_name: str | None = None) -> str | None:
        """The scrip code, or None when unlisted or unavailable."""
        return self.describe(ticker, company_name)["bse_code"]
