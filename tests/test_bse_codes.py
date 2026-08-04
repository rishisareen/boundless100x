"""Resolving a BSE scrip code without Screener's help.

Screener stopped rendering bseindia.com links server-side — the live page now
contains no occurrence of 'bseindia' and no scrip code at all — so `bse_code`
came back None on every fresh fetch and the annual-report and BSE shareholding
fetches quietly degraded. The code now comes from BSE's own scrip master.

A company that simply is not listed on BSE (CDSL and BSE Ltd among the cached
tickers) must resolve to a clean "not listed" rather than an error, or the
suite will log a failure every run for a fact about the company.
"""

import json

import pytest

from boundless100x.data_fetcher.bse_codes import BseCodeResolver

SCRIPS = [
    {"SCRIP_CD": "532830", "scrip_id": "ASTRAL", "Scrip_Name": "Astral Ltd",
     "Issuer_Name": "Astral Ltd", "ISIN_NUMBER": "INE006I01046", "Status": "Active"},
    {"SCRIP_CD": "500339", "scrip_id": "RAIN", "Scrip_Name": "Rain Industries Ltd",
     "Issuer_Name": "Rain Industries Ltd", "ISIN_NUMBER": "INE855B01025", "Status": "Active"},
    {"SCRIP_CD": "500777", "scrip_id": "TNPETRO", "Scrip_Name": "Tamilnadu Petroproducts Ltd",
     "Issuer_Name": "Tamilnadu Petroproducts Ltd", "ISIN_NUMBER": "INE148A01019", "Status": "Active"},
    {"SCRIP_CD": "543232", "scrip_id": "CAMS", "Scrip_Name": "Computer Age Management Services Ltd",
     "Issuer_Name": "Computer Age Management Services Ltd", "ISIN_NUMBER": "INE596I01020",
     "Status": "Active"},
]


@pytest.fixture
def resolver(tmp_path):
    r = BseCodeResolver(cache_dir=str(tmp_path))
    r._scrips = SCRIPS          # skip the network entirely
    r._index = r._build_index(SCRIPS)
    return r


class TestExactSymbolMatch:
    def test_resolves_a_listed_ticker(self, resolver):
        assert resolver.resolve("ASTRAL") == "532830"

    def test_match_is_case_and_space_insensitive(self, resolver):
        assert resolver.resolve("  astral ") == "532830"

    @pytest.mark.parametrize("ticker,code", [
        ("RAIN", "500339"), ("TNPETRO", "500777"), ("CAMS", "543232"),
    ])
    def test_resolves_each_cached_ticker(self, resolver, ticker, code):
        assert resolver.resolve(ticker) == code


class TestNotListedIsAFactNotAFailure:
    def test_unlisted_company_resolves_to_none(self, resolver):
        """CDSL trades on NSE only — no BSE code exists to find."""
        assert resolver.resolve("CDSL") is None

    def test_unlisted_company_is_reported_as_not_listed(self, resolver):
        assert resolver.describe("CDSL")["status"] == "not_listed_on_bse"

    def test_listed_company_reports_how_it_matched(self, resolver):
        described = resolver.describe("ASTRAL")

        assert described["status"] == "resolved"
        assert described["bse_code"] == "532830"
        assert described["matched_on"] == "symbol"


class TestCompanyNameFallback:
    def test_name_resolves_when_the_symbol_differs(self, resolver):
        """BSE's symbol need not equal the NSE ticker."""
        code = resolver.resolve("TNPL", company_name="Tamilnadu Petroproducts Ltd")

        assert code == "500777"

    def test_name_match_ignores_suffixes_and_case(self, resolver):
        assert resolver.resolve("XX", company_name="rain industries limited") == "500339"

    def test_name_match_is_reported_as_such(self, resolver):
        described = resolver.describe("XX", company_name="Astral Limited")

        assert described["matched_on"] == "company_name"

    def test_ambiguous_name_does_not_guess(self, resolver):
        assert resolver.resolve("XX", company_name="Ltd") is None


class TestCaching:
    def test_scrip_master_is_fetched_once(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(self):
            calls.append(1)
            return SCRIPS

        monkeypatch.setattr(BseCodeResolver, "_download_scrips", fake_download)
        r = BseCodeResolver(cache_dir=str(tmp_path))

        r.resolve("ASTRAL")
        r.resolve("RAIN")

        assert len(calls) == 1

    def test_a_second_resolver_reuses_the_cached_master(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(self):
            calls.append(1)
            return SCRIPS

        monkeypatch.setattr(BseCodeResolver, "_download_scrips", fake_download)
        BseCodeResolver(cache_dir=str(tmp_path)).resolve("ASTRAL")
        BseCodeResolver(cache_dir=str(tmp_path)).resolve("RAIN")

        assert len(calls) == 1


class TestDegradation:
    def test_download_failure_returns_none_rather_than_raising(self, tmp_path, monkeypatch):
        def boom(self):
            raise RuntimeError("BSE unreachable")

        monkeypatch.setattr(BseCodeResolver, "_download_scrips", boom)

        assert BseCodeResolver(cache_dir=str(tmp_path)).resolve("ASTRAL") is None

    def test_download_failure_is_reported_distinctly_from_not_listed(self, tmp_path, monkeypatch):
        """'We could not look it up' is not 'it is not there'."""
        def boom(self):
            raise RuntimeError("BSE unreachable")

        monkeypatch.setattr(BseCodeResolver, "_download_scrips", boom)

        assert BseCodeResolver(cache_dir=str(tmp_path)).describe("ASTRAL")["status"] == "lookup_failed"

    def test_blank_ticker_is_handled(self, resolver):
        assert resolver.resolve("") is None


@pytest.mark.network
class TestAgainstLiveBse:
    def test_live_scrip_master_resolves_a_known_company(self, tmp_path):
        code = BseCodeResolver(cache_dir=str(tmp_path)).resolve("ASTRAL")

        assert code == "532830"
