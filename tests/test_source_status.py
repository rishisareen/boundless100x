"""Fetch failures must not reach scoring or a recommendation.

Previously a Screener.in or price fetch failure left an empty DataFrame with
no signal attached to it, so the pipeline scored, evaluated eligibility, and
could still ask the LLM for a recommendation on essentially no data. Financials
and price are load-bearing for every stage after the fetch — a failure there
must stop the pipeline, not degrade it silently into a complete-looking report.
"""

import pandas as pd
import pytest

from boundless100x.data_fetcher.suite import DataFetcherSuite
from boundless100x.service import Boundless100xService
from tests.conftest import make_data


class FailingFetcher:
    def fetch_all(self, ticker, output_dir=None):
        raise RuntimeError("Screener.in unreachable")


class FailingPriceFetcher:
    def fetch(self, ticker, years=10, output_dir=None):
        raise RuntimeError("yfinance and jugaad-data both failed")


class EmptyPriceFetcher:
    def fetch(self, ticker, years=10, output_dir=None):
        return pd.DataFrame()


class StubAnalystCoverage:
    def fetch(self, ticker, output_dir=None):
        return {}


def build_suite() -> DataFetcherSuite:
    return DataFetcherSuite({})


class TestSourceStatus:
    def test_financials_exception_is_recorded_as_failed(self):
        suite = build_suite()
        suite.financials = FailingFetcher()

        data = suite.fetch_all("NOPE")

        assert data["source_status"]["financials"].startswith("failed:")
        assert data["financials"].empty

    def test_price_exception_is_recorded_as_failed(self):
        suite = build_suite()
        suite.price_volume = FailingPriceFetcher()
        suite.analyst_coverage = StubAnalystCoverage()

        data = suite.fetch_all("NOPE")

        assert data["source_status"]["price"].startswith("failed:")

    def test_price_empty_without_exception_is_recorded_as_empty(self):
        """Screener.in and yfinance can both succeed and just return nothing —
        that is not an exception, so it needs its own signal."""
        suite = build_suite()
        suite.price_volume = EmptyPriceFetcher()
        suite.analyst_coverage = StubAnalystCoverage()

        data = suite.fetch_all("NOPE")

        assert data["source_status"]["price"].startswith("empty:")


def service_with_stub_suite(monkeypatch, fetch_all_return: dict) -> Boundless100xService:
    svc = Boundless100xService()
    monkeypatch.setattr(svc.suite, "fetch_all", lambda ticker, bse_code=None: fetch_all_return)
    return svc


class TestFatalCoreDataStop:
    def test_missing_financials_stops_before_scoring(self, monkeypatch):
        data = make_data()
        data["financials"] = pd.DataFrame()
        data["source_status"] = {"financials": "failed: timeout", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)

        result = svc.analyze("NOPE", use_llm=False)

        assert result.metrics == {}
        assert result.scores == {}
        assert any("Fatal" in e for e in result.errors)

    def test_missing_price_stops_before_scoring(self, monkeypatch):
        data = make_data()
        data["price"] = pd.DataFrame()
        data["source_status"] = {"financials": "ok", "price": "empty: no rows"}
        svc = service_with_stub_suite(monkeypatch, data)

        result = svc.analyze("NOPE", use_llm=False)

        assert result.metrics == {}
        assert any("Fatal" in e for e in result.errors)

    def test_llm_is_never_invoked_when_core_data_is_missing(self, monkeypatch):
        data = make_data()
        data["financials"] = pd.DataFrame()
        data["source_status"] = {"financials": "empty: no rows", "price": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)
        svc._llm = object()  # would blow up if ever touched — analyze() must not reach it

        result = svc.analyze("NOPE", use_llm=True)

        assert result.llm_analysis is None

    def test_good_core_data_proceeds_past_the_gate(self, monkeypatch):
        data = make_data()
        data["source_status"] = {"financials": "ok", "price": "ok", "analyst_coverage": "ok"}
        svc = service_with_stub_suite(monkeypatch, data)

        result = svc.analyze("ASTRAL", use_llm=False)

        assert result.metrics != {}
        assert result.scores.get("composite") is not None
        assert not any("Fatal" in e for e in result.errors)

    def test_missing_source_status_key_is_treated_as_not_ok(self, monkeypatch):
        """A caller that forgets to set source_status must fail closed, not
        be waved through as if the fetch had succeeded."""
        data = make_data()
        data.pop("source_status", None)
        svc = service_with_stub_suite(monkeypatch, data)

        result = svc.analyze("NOPE", use_llm=False)

        assert result.metrics == {}
        assert any("Fatal" in e for e in result.errors)
