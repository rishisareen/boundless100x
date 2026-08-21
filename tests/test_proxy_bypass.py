"""Data fetches go around the machine's system proxy by default.

Magic Lasso Adblock runs as a macOS packet-tunnel extension, which advertises
itself system-wide on an ephemeral loopback port — so `HTTP_PROXY` is set for
every process without anyone configuring it. While it reloads its filter lists
it refuses the CONNECT handshake, and every HTTPS fetch in that window dies
three times over with "Tunnel connection failed: 400 Bad Request", for a
failure retrying cannot fix. The port changes on each restart, so a NO_PROXY
entry is not a durable answer and this is a setting instead.
"""

import os

import pytest
import requests

from boundless100x.data_fetcher.base import BaseFetcher
from boundless100x.data_fetcher.bse_codes import BseCodeResolver
from boundless100x.data_fetcher.proxy import (
    apply_to_session,
    bypassed,
    system_proxy_in_use,
)


@pytest.fixture
def proxied(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:49595")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:49595")


class TestSessionsIgnoreTheSystemProxy:
    def test_a_fetcher_session_does_not_trust_the_environment(self, proxied):
        """`trust_env` rather than `session.proxies`: an empty proxies dict does
        not override the environment, because requests merges environment
        proxies in at request time unless trust_env is off."""
        assert BaseFetcher().session.trust_env is False

    def test_opting_in_restores_it(self, proxied):
        """A machine that reaches the internet only through its proxy."""
        assert BaseFetcher(use_system_proxy=True).session.trust_env is True

    def test_apply_is_idempotent_and_safe_without_a_proxy(self, monkeypatch):
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.delenv(key, raising=False)
        session = requests.Session()

        apply_to_session(session)

        assert session.trust_env is False


class TestClientsThatReadTheEnvironmentThemselves:
    """yfinance and jugaad-data take no session of ours."""

    def test_the_variables_are_gone_inside_the_block(self, proxied):
        with bypassed():
            assert system_proxy_in_use() is None

    def test_they_are_restored_afterwards(self, proxied):
        with bypassed():
            pass

        assert system_proxy_in_use() == "http://127.0.0.1:49595"

    def test_they_are_restored_even_when_the_block_raises(self, proxied):
        """A fetch that throws must not leave the process without the proxy
        settings the rest of it may need."""
        with pytest.raises(RuntimeError):
            with bypassed():
                raise RuntimeError("fetch blew up")

        assert system_proxy_in_use() == "http://127.0.0.1:49595"

    def test_opting_in_leaves_the_environment_alone(self, proxied):
        with bypassed(use_system_proxy=True):
            assert system_proxy_in_use() == "http://127.0.0.1:49595"

    def test_lowercase_spellings_are_removed_too(self, monkeypatch):
        """requests is case-insensitive about these; urllib and curl_cffi are
        not, so both cases have to go."""
        monkeypatch.setenv("https_proxy", "http://127.0.0.1:49595")

        with bypassed():
            assert os.environ.get("https_proxy") is None


class TestEveryNetworkPathIsCovered:
    def test_the_scrip_resolver_carries_the_preference(self):
        """The one network call outside `BaseFetcher`'s session."""
        assert BseCodeResolver().use_system_proxy is False
        assert BseCodeResolver(use_system_proxy=True).use_system_proxy is True

    def test_the_suite_threads_the_setting_to_every_fetcher(self):
        from boundless100x.data_fetcher.suite import DataFetcherSuite

        suite = DataFetcherSuite({"fetching": {"use_system_proxy": True}})

        assert suite.financials.session.trust_env is True
        assert suite.price_volume.use_system_proxy is True
        assert suite._bse_codes.use_system_proxy is True

    def test_the_default_config_bypasses(self):
        from boundless100x.data_fetcher.suite import DataFetcherSuite
        from boundless100x.service import load_config

        suite = DataFetcherSuite(load_config())

        assert suite.financials.session.trust_env is False
        assert suite.price_volume.use_system_proxy is False
        assert suite._bse_codes.use_system_proxy is False

    def test_the_llm_transport_is_deliberately_untouched(self):
        """One module's fetching preference has no business reaching the
        headless CLI's environment, which allowlists the proxy variables on
        purpose for the corporate-egress case."""
        from boundless100x.llm_layer.transport import INHERITED_ENV_KEYS

        assert "HTTPS_PROXY" in INHERITED_ENV_KEYS
