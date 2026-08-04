"""Financials fetch: page caching and sector metadata extraction.

The fetch used to bypass its cache entirely (dead `_do_fetch`/`cache_key`
code), so every run re-scraped Screener.in. And the sector selector grabbed
the first anchor of `p.sub`, which no longer holds the sector — Screener now
renders a market breadcrumb there, and every cached ticker's metadata came
back sectorless, neutralising the sector-tailwind metric.
"""

import pytest
from bs4 import BeautifulSoup

from boundless100x.compute_engine.sector import classify_sector
from boundless100x.data_fetcher.cache.cache_manager import CacheManager
from boundless100x.data_fetcher.fetch_financials import FinancialsFetcher

# Mirrors the live Screener.in structure (checked against /company/CDSL/consolidated/).
SCREENER_HTML = """
<html><body>
<div id="company-info" data-company-id="123" data-warehouse-id="456"
     data-consolidated="true"></div>
<h1>Central Depository Services (India) Ltd</h1>
<p class="sub font-size-14">
  <span>*</span>
  <span>The pros and cons are machine generated.</span>
</p>
<p class="sub">
  <i class="icon-globe"></i>
  <a href="/market/IN05/" title="Broad Sector">Financial Services</a>
  <i class="icon-right"></i>
  <a href="/market/IN05/IN0501/" title="Sector">Financial Services</a>
  <i class="icon-right"></i>
  <a href="/market/IN05/IN0501/IN050103/" title="Broad Industry">Capital Markets</a>
  <i class="icon-right"></i>
  <a href="/market/IN05/IN0501/IN050103/IN050103002/"
     title="Industry">Depositories, Clearing Houses and Other Intermediaries</a>
</p>
<ul id="top-ratios">
  <li><span class="name">Market Cap</span><span class="number">27,634</span></li>
  <li><span class="name">Stock P/E</span><span class="number">66.6</span></li>
  <li><span class="name">High / Low</span>
      <span class="number">1,989</span><span class="number">1,200</span></li>
</ul>
<a href="https://www.bseindia.com/stock-share-price/x/cdsl/541540/">BSE</a>
</body></html>
"""


def make_fetcher(tmp_path, ttl_hours: float = 24) -> FinancialsFetcher:
    fetcher = FinancialsFetcher()
    fetcher.cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=ttl_hours)
    return fetcher


class TestPageCaching:
    def test_repeat_fetch_within_ttl_hits_the_network_once(self, tmp_path, monkeypatch):
        fetcher = make_fetcher(tmp_path)
        calls = []
        monkeypatch.setattr(
            fetcher, "_get_company_page_html",
            lambda ticker: calls.append(ticker) or SCREENER_HTML,
        )

        first = fetcher.fetch_all("CDSL", output_dir=str(tmp_path / "raw"))
        second = fetcher.fetch_all("CDSL", output_dir=str(tmp_path / "raw"))

        assert calls == ["CDSL"]
        assert first["metadata"]["name"] == second["metadata"]["name"]

    def test_expired_cache_refetches(self, tmp_path, monkeypatch):
        fetcher = make_fetcher(tmp_path, ttl_hours=0)  # everything is immediately stale
        calls = []
        monkeypatch.setattr(
            fetcher, "_get_company_page_html",
            lambda ticker: calls.append(ticker) or SCREENER_HTML,
        )

        fetcher.fetch_all("CDSL")
        fetcher.fetch_all("CDSL")

        assert calls == ["CDSL", "CDSL"]


class TestSectorMetadata:
    def setup_method(self):
        self.meta = FinancialsFetcher()._get_company_metadata(
            BeautifulSoup(SCREENER_HTML, "html.parser")
        )

    def test_sector_comes_from_the_broad_industry_breadcrumb(self):
        assert self.meta["sector"] == "Capital Markets"

    def test_all_breadcrumb_levels_are_kept(self):
        assert self.meta["sector_broad"] == "Financial Services"
        assert self.meta["sector_industry"] == (
            "Depositories, Clearing Houses and Other Intermediaries"
        )

    def test_extracted_sector_classifies_into_a_study_bucket(self):
        assert classify_sector(self.meta["sector"]) == "strong_tailwind"

    def test_screener_plural_labels_match_the_study_lists(self):
        """The study lists 'Capital Market'; Screener prints 'Capital Markets'."""
        assert classify_sector("Capital Markets") == "strong_tailwind"
