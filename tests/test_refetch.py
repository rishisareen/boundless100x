"""U2 — the refetch loop, offline.

The thing this command is really testing is the network and the corpus, and
neither belongs in a unit test. What is testable offline is everything that
decides *which* tickers are touched and *whether* the network is reached at
all: enumeration, per-ticker isolation, resume, and the cache bypass. Those
are also the parts whose failure is silent — a skipped ticker and a
cache-served one both look like success.
"""

import json

import pytest

from boundless100x.data_fetcher import bse_codes, corpus_snapshot, refetch
from boundless100x.data_fetcher.cache.cache_manager import CacheManager


class StubSuite:
    """A DataFetcherSuite shaped just enough for the refetch loop.

    Carries six fetchers because the real suite does and the bypass has to
    reach every distinct cache, but they all share one CacheManager exactly as
    production's default cache directory makes them share one directory.
    """

    def __init__(self, raw_data_dir, cache, fail_on=()):
        self.raw_data_dir = str(raw_data_dir)
        self.calls = []
        self.fail_on = set(fail_on)
        for name in ("financials", "price_volume", "shareholding_bse",
                     "corporate_actions", "analyst_coverage", "annual_reports"):
            setattr(self, name, type("F", (), {"cache": cache})())

    def fetch_all(self, ticker, bse_code=None):
        self.calls.append(ticker)
        if ticker in self.fail_on:
            raise RuntimeError(f"Screener markup changed for {ticker}")
        return {"source_status": {"financials": "ok", "price": "ok"}}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "raw_data"
    for ticker in ("ASTRAL", "VBL", "CDSL"):
        directory = root / ticker
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(json.dumps({"bse_code": "500001"}))
        (directory / "financials.csv").write_text("year,revenue\n")
    # A BSE-code directory of annual report PDFs, and the dead one (A3).
    (root / "500001" / "annual_reports").mkdir(parents=True)
    (root / "ZYDUS").mkdir()
    (root / "ZYDUS" / "analyst_coverage.json").write_text("{}")
    return root


@pytest.fixture
def suite(corpus, tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
    return StubSuite(corpus, cache)


def run(suite, tmp_path, **kwargs):
    kwargs.setdefault("require_snapshot", False)
    kwargs.setdefault("run_log_path", tmp_path / "refetch_log.json")
    return refetch.refetch(suite, **kwargs)


class TestEnumeration:
    def test_returns_real_tickers_and_names_every_exclusion(self, corpus):
        tickers, skipped = refetch.enumerate_tickers(corpus)

        assert tickers == ["ASTRAL", "CDSL", "VBL"]
        reasons = {entry["name"]: entry["reason"] for entry in skipped}
        assert set(reasons) == {"500001", "ZYDUS"}
        assert "BSE-code" in reasons["500001"]
        assert "metadata.json" in reasons["ZYDUS"]

    def test_an_absent_corpus_enumerates_to_nothing(self, tmp_path):
        assert refetch.enumerate_tickers(tmp_path / "nope") == ([], [])


class TestIsolation:
    def test_a_failing_ticker_is_recorded_and_the_loop_continues(
        self, corpus, tmp_path
    ):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        suite = StubSuite(corpus, cache, fail_on=("CDSL",))

        report = run(suite, tmp_path)

        assert suite.calls == ["ASTRAL", "CDSL", "VBL"]
        by_ticker = {o["ticker"]: o for o in report["outcomes"]}
        assert by_ticker["CDSL"]["status"] == "failed"
        assert "Screener markup" in by_ticker["CDSL"]["detail"]
        assert by_ticker["ASTRAL"]["status"] == "ok"
        assert by_ticker["VBL"]["status"] == "ok"


class TestDegradedSources:
    """`fetch_all` swallows a source failure into `source_status` and returns
    normally, so the return value alone says nothing about what was refreshed."""

    def test_a_run_where_every_source_failed_is_not_reported_as_ok(
        self, corpus, tmp_path
    ):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        suite = StubSuite(corpus, cache)
        suite.fetch_all = lambda t, bse_code=None: {
            "source_status": {"financials": "failed: Screener 500",
                              "price": "failed: no data"}
        }

        report = run(suite, tmp_path)

        assert {o["status"] for o in report["outcomes"]} == {"degraded"}

    def test_a_degraded_ticker_is_retried_rather_than_skipped_on_resume(
        self, corpus, tmp_path
    ):
        """The dangerous half: marked complete, a resume would never revisit it."""
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        broken = StubSuite(corpus, cache)
        broken.fetch_all = lambda t, bse_code=None: {
            "source_status": {"financials": "failed: Screener 500"}
        }
        run(broken, tmp_path)

        recovered = StubSuite(corpus, cache)
        run(recovered, tmp_path)

        assert recovered.calls == ["ASTRAL", "CDSL", "VBL"]

    def test_a_partially_degraded_ticker_still_counts_as_degraded(
        self, corpus, tmp_path
    ):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        suite = StubSuite(corpus, cache)
        suite.fetch_all = lambda t, bse_code=None: {
            "source_status": {"financials": "ok", "price": "failed: timeout"}
        }

        report = run(suite, tmp_path)

        assert report["outcomes"][0]["status"] == "degraded"


class TestUnmatchedTickers:
    def test_a_requested_ticker_that_is_not_cached_is_reported(self, suite, tmp_path):
        """Refetching 1 of the 2 someone asked for must not look like success."""
        report = run(suite, tmp_path, tickers=["VBL", "NOSUCHCO"])

        assert suite.calls == ["VBL"]
        reasons = {e["name"]: e["reason"] for e in report["skipped"]}
        assert "NOSUCHCO" in reasons
        assert "not a cached ticker" in reasons["NOSUCHCO"]


class TestResume:
    def test_a_resumed_run_skips_tickers_the_log_records_as_complete(
        self, suite, tmp_path
    ):
        log_path = tmp_path / "refetch_log.json"
        run(suite, tmp_path)
        assert suite.calls == ["ASTRAL", "CDSL", "VBL"]

        suite.calls.clear()
        report = run(suite, tmp_path, run_log_path=log_path)

        assert suite.calls == []
        assert report["resumed"] == ["ASTRAL", "CDSL", "VBL"]

    def test_a_failed_ticker_is_retried_on_resume(self, corpus, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        failing = StubSuite(corpus, cache, fail_on=("CDSL",))
        run(failing, tmp_path)

        recovered = StubSuite(corpus, cache)
        run(recovered, tmp_path)

        assert recovered.calls == ["CDSL"]

    def test_a_completed_run_does_not_suppress_a_later_refetch(
        self, suite, tmp_path
    ):
        """The log resumes an interrupted pass; it is not a permanent record.

        Unbounded, running `corpus refetch` again a month later skipped all 22
        tickers and reported them resumed — the command silently no-opping on
        its second use, which is the use it exists for.
        """
        run(suite, tmp_path)
        suite.calls.clear()

        run(suite, tmp_path, resume_within_hours=0)

        assert suite.calls == ["ASTRAL", "CDSL", "VBL"]

    def test_an_interrupted_pass_still_resumes_inside_the_window(
        self, suite, tmp_path
    ):
        run(suite, tmp_path)
        suite.calls.clear()

        run(suite, tmp_path)  # default 24h window, entries written seconds ago

        assert suite.calls == []

    def test_resume_off_reruns_everything(self, suite, tmp_path):
        run(suite, tmp_path)
        suite.calls.clear()

        run(suite, tmp_path, resume=False)

        assert suite.calls == ["ASTRAL", "CDSL", "VBL"]


class TestCacheBypass:
    def _warm(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        cache.set("screener_page_ASTRAL", "<html>cached</html>")
        cache.set(bse_codes.CACHE_KEY, {"scrips": [{"code": "500001"}]})
        return cache

    def test_the_flag_makes_a_fresh_entry_miss(self, corpus, tmp_path):
        cache = self._warm(tmp_path)
        suite = StubSuite(corpus, cache)

        run(suite, tmp_path, bypass_cache=True)

        assert cache.get("screener_page_ASTRAL") is None

    def test_without_the_flag_a_fresh_entry_is_still_served(self, corpus, tmp_path):
        cache = self._warm(tmp_path)
        suite = StubSuite(corpus, cache)

        run(suite, tmp_path, bypass_cache=False)

        assert cache.get("screener_page_ASTRAL") == "<html>cached</html>"

    def test_the_bse_scrip_master_survives_the_bypass(self, corpus, tmp_path):
        cache = self._warm(tmp_path)
        suite = StubSuite(corpus, cache)

        run(suite, tmp_path, bypass_cache=True)

        assert cache.get(bse_codes.CACHE_KEY) == {"scrips": [{"code": "500001"}]}

    def test_nothing_is_cleared_when_every_ticker_was_already_done(
        self, corpus, tmp_path
    ):
        cache = self._warm(tmp_path)
        suite = StubSuite(corpus, cache)
        run(suite, tmp_path, bypass_cache=False)

        run(suite, tmp_path, bypass_cache=True)

        assert cache.get("screener_page_ASTRAL") == "<html>cached</html>"


class TestSnapshotGuard:
    def test_starting_with_no_snapshot_is_refused(self, suite, tmp_path, monkeypatch):
        monkeypatch.setattr(
            corpus_snapshot, "DEFAULT_SNAPSHOT_DIR", tmp_path / "no_snapshots"
        )

        with pytest.raises(corpus_snapshot.SnapshotError) as excinfo:
            refetch.refetch(
                suite, require_snapshot=True,
                run_log_path=tmp_path / "refetch_log.json",
            )
        assert "only copy" in str(excinfo.value)
        assert suite.calls == []

    def test_a_present_snapshot_lets_the_run_start(self, suite, tmp_path, monkeypatch):
        base = tmp_path / "snapshots"
        made = base / f"{corpus_snapshot.SNAPSHOT_PREFIX}20260807-060000"
        (made / "raw_data").mkdir(parents=True)
        (made / corpus_snapshot.MANIFEST_NAME).write_text("{}")
        monkeypatch.setattr(corpus_snapshot, "DEFAULT_SNAPSHOT_DIR", base)

        refetch.refetch(
            suite, require_snapshot=True,
            run_log_path=tmp_path / "refetch_log.json",
        )

        assert suite.calls == ["ASTRAL", "CDSL", "VBL"]


class TestTickerFilter:
    def test_an_explicit_list_narrows_the_run(self, suite, tmp_path):
        run(suite, tmp_path, tickers=["vbl"])

        assert suite.calls == ["VBL"]


class TestCacheManagerKeep:
    def test_clear_all_keeps_named_keys(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        cache.set("keep_me", {"a": 1})
        cache.set("drop_me", {"b": 2})

        removed = cache.clear_all(keep=("keep_me",))

        assert cache.get("keep_me") == {"a": 1}
        assert cache.get("drop_me") is None
        assert removed == 2  # the .json and the .meta of drop_me

    def test_clear_all_with_no_argument_still_clears_everything(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        cache.set("anything", {"a": 1})

        cache.clear_all()

        assert cache.get("anything") is None
