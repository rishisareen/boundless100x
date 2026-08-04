"""Trendlyne ticker resolution.

The search API returns fuzzy matches — accepting the first result when no
NSE code matches exactly can attach one company's analyst coverage to a
different, similarly-named issuer.
"""

from unittest.mock import MagicMock

from boundless100x.data_fetcher.fetch_analyst_coverage import AnalystCoverageFetcher


def fetcher() -> AnalystCoverageFetcher:
    return AnalystCoverageFetcher()


def mock_response(payload) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestResolveTrendlyneId:
    def test_exact_nse_match_is_returned(self):
        f = fetcher()
        f._get = MagicMock(return_value=mock_response([
            {"NSEcode": "OTHERCO", "k": 111, "slugname": "other-co"},
            {"NSEcode": "ASTRAL", "k": 222, "slugname": "astral-ltd"},
        ]))

        result = f._resolve_trendlyne_id("ASTRAL")

        assert result["k"] == 222

    def test_no_exact_match_returns_none_instead_of_first_result(self):
        f = fetcher()
        f._get = MagicMock(return_value=mock_response([
            {"NSEcode": "ASTRALPOLY", "k": 111, "slugname": "astral-poly"},
            {"NSEcode": "ASTRALFOAM", "k": 222, "slugname": "astral-foam"},
        ]))

        result = f._resolve_trendlyne_id("ASTRAL")

        assert result is None

    def test_match_is_case_insensitive(self):
        f = fetcher()
        f._get = MagicMock(return_value=mock_response([
            {"NSEcode": "astral", "k": 222, "slugname": "astral-ltd"},
        ]))

        result = f._resolve_trendlyne_id("ASTRAL")

        assert result["k"] == 222

    def test_no_matches_headline_response_returns_none(self):
        f = fetcher()
        f._get = MagicMock(return_value=mock_response([{"headline": "No matches"}]))

        assert f._resolve_trendlyne_id("NOTREAL") is None
