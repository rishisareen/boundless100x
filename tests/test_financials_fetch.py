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
from boundless100x.data_fetcher.fetch_financials import (
    PL_LABEL_MAP,
    QTR_LABEL_MAP,
    FinancialsFetcher,
    _parse_table,
)

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
<section id="quarters" class="card card-large">
  <h2>Quarterly Results</h2>
  <div class="responsive-holder fill-card-width" data-result-table>
  <table class="data-table responsive-text-nowrap">
    <thead><tr>
      <th class="text"></th><th>Jun 2023</th><th>Sep 2023</th><th>Dec 2023</th>
    </tr></thead>
    <tbody>
      <tr class="stripe">
        <td class="text"><button class="button-plain">Sales&nbsp;<span>+</span></button></td>
        <td>150</td><td>207</td><td>214</td>
      </tr>
      <tr>
        <td class="text"><button class="button-plain">Expenses&nbsp;<span>+</span></button></td>
        <td>70</td><td>79</td><td>84</td>
      </tr>
      <tr><td class="text">Operating Profit</td><td>80</td><td>128</td><td>130</td></tr>
      <tr><td class="text">OPM %</td><td>53%</td><td>62%</td><td>61%</td></tr>
      <tr><td class="text">Net Profit</td><td>74</td><td>109</td><td>107</td></tr>
      <tr><td class="text">EPS in Rs</td><td>3.52</td><td>5.21</td><td>5.14</td></tr>
      <tr><td class="text">Raw PDF</td><td><a href="/x.pdf">PDF</a></td><td></td><td></td></tr>
    </tbody>
  </table>
  </div>
</section>
</body></html>
"""

# Same page with the quarterly section absent — Screener does not render it for
# every listing, and a missing section must degrade to an empty frame rather
# than take the whole fetch down with it.
SCREENER_HTML_NO_QUARTERS = SCREENER_HTML[: SCREENER_HTML.index('<section id="quarters"')] + (
    "</body></html>"
)


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


class TestQuarterlyResults:
    """The quarterly table is the grain the v05 lifecycle checkpoints run on.

    It is structurally identical to the annual P&L, so it shares `_parse_table`
    — what differs is the period column name and the absence of annual-only
    rows.
    """

    def parse(self, html: str = SCREENER_HTML):
        return _parse_table(
            BeautifulSoup(html, "html.parser"),
            "quarters",
            QTR_LABEL_MAP,
            period_col="quarter",
        )

    def test_periods_land_in_a_quarter_column_not_a_year_column(self):
        df = self.parse()
        assert list(df.columns)[0] == "quarter"
        assert list(df["quarter"]) == ["Jun 2023", "Sep 2023", "Dec 2023"]

    def test_one_row_per_quarter_with_mapped_metric_columns(self):
        df = self.parse()
        assert len(df) == 3
        assert {"revenue", "expenses", "operating_profit", "opm_pct", "pat", "eps"} <= set(
            df.columns
        )

    def test_values_are_numeric_with_percent_signs_stripped(self):
        df = self.parse()
        assert df["revenue"].tolist() == [150.0, 207.0, 214.0]
        assert df["opm_pct"].tolist() == [53.0, 62.0, 61.0]
        assert df["eps"].tolist() == [3.52, 5.21, 5.14]

    def test_unmapped_rows_are_dropped(self):
        """Screener's 'Raw PDF' row carries links, not numbers."""
        assert "Raw PDF" not in self.parse().columns

    def test_annual_only_columns_never_appear(self):
        """Dividend payout is annual; the quarterly map must not invent it."""
        assert "dividend_payout_pct" not in self.parse().columns

    def test_missing_section_yields_an_empty_frame(self):
        assert self.parse(SCREENER_HTML_NO_QUARTERS).empty

    def test_annual_table_still_uses_the_year_column(self):
        """The shared parser must not have renamed the annual period column."""
        html = SCREENER_HTML.replace('id="quarters"', 'id="profit-loss"')
        df = _parse_table(BeautifulSoup(html, "html.parser"), "profit-loss", PL_LABEL_MAP)
        assert list(df.columns)[0] == "year"


class TestQuarterlyPersistence:
    def test_quarterly_csv_is_written_alongside_the_other_tables(
        self, tmp_path, monkeypatch
    ):
        fetcher = make_fetcher(tmp_path)
        monkeypatch.setattr(fetcher, "_get_company_page_html", lambda t: SCREENER_HTML)

        result = fetcher.fetch_all("CDSL", output_dir=str(tmp_path / "raw"))

        assert not result["quarterly"].empty
        written = tmp_path / "raw" / "CDSL" / "quarterly.csv"
        assert written.exists()
        assert "quarter" in written.read_text().splitlines()[0]

    def test_a_page_without_quarters_still_completes_the_fetch(
        self, tmp_path, monkeypatch
    ):
        """Graceful absence: no exception, no file, other outputs unaffected."""
        fetcher = make_fetcher(tmp_path)
        monkeypatch.setattr(
            fetcher, "_get_company_page_html", lambda t: SCREENER_HTML_NO_QUARTERS
        )

        result = fetcher.fetch_all("CDSL", output_dir=str(tmp_path / "raw"))

        assert result["quarterly"].empty
        assert not (tmp_path / "raw" / "CDSL" / "quarterly.csv").exists()
        assert result["metadata"]["name"] == "Central Depository Services (India) Ltd"


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
