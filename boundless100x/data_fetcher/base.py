"""Base fetcher with retry logic, rate limiting, and caching."""

import logging
import time
from typing import Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from boundless100x.data_fetcher.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class BaseFetcher:
    """Base class for all data fetchers.

    Provides:
    - requests.Session with retry logic and sensible headers
    - Rate limiting between requests
    - TTL-based local caching
    - Standardized error handling
    """

    def __init__(
        self,
        cache_ttl_hours: int = 24,
        rate_limit_seconds: float = 2.0,
        retry_count: int = 3,
        retry_delay_seconds: float = 5.0,
    ):
        self.cache = CacheManager(ttl_hours=cache_ttl_hours)
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_time: float = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

        retries = Retry(
            total=retry_count,
            backoff_factor=retry_delay_seconds,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _rate_limit(self) -> None:
        """Enforce minimum delay between HTTP requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

    def _get(self, url: str, **kwargs) -> requests.Response:
        """Make a rate-limited GET request with error handling."""
        self._rate_limit()
        logger.debug(f"GET {url}")

        response = self.session.get(url, timeout=30, **kwargs)
        self._last_request_time = time.time()
        response.raise_for_status()

        return response

    def fetch_with_cache(
        self,
        key: str,
        fetch_fn: Callable[[], pd.DataFrame | dict],
    ) -> pd.DataFrame | dict:
        """Return cached data if available, otherwise call fetch_fn and cache the result."""
        cached = self.cache.get(key)
        if cached is not None:
            logger.info(f"Cache hit: {key}")
            return cached

        logger.info(f"Cache miss: {key} — fetching fresh data")
        data = fetch_fn()
        self.cache.set(key, data)
        return data
