"""Whether this pipeline's fetches go through the machine's system proxy.

**Default: they do not.** Every source here is a public financial site —
Screener, BSE, Yahoo, Trendlyne — reached from a personal machine, and a system
proxy on such a machine is far more often an ad blocker or a VPN than required
egress. Those interpose on connections they add nothing to, and when one
reloads its filter lists or wakes from sleep it refuses the CONNECT handshake
for a few seconds. Every HTTPS fetch in that window dies with

    ProxyError: Unable to connect to proxy —
    Tunnel connection failed: 400 Bad Request

three times over, once per retry, for a failure retrying cannot fix. That is
what Magic Lasso Adblock's packet-tunnel extension did to a BSE fetch here: it
advertises itself system-wide on an ephemeral loopback port, so `HTTP_PROXY`
and `HTTPS_PROXY` are set for every process without anyone configuring
anything, and the port number changes each time the extension restarts — which
is why a `NO_PROXY` entry is not a durable answer and this is a setting rather
than a hostname list.

Set `fetching.use_system_proxy: true` in config.yaml on a machine where the
proxy is the only way out. **The LLM transport is deliberately untouched by
this**: `llm_layer/transport.py` allowlists the proxy variables into the
headless CLI's environment on purpose, for exactly the corporate-egress case,
and one module's fetching preference has no business reaching that decision.
"""

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Every spelling of the proxy variables that a client library might read.
# `requests` is case-insensitive about these; `urllib` and `curl_cffi` are not,
# so both cases have to go.
_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)

_warned = False


def system_proxy_in_use() -> str | None:
    """The proxy the environment advertises, or None."""
    for key in _PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _announce(proxy: str) -> None:
    """Say once per process that a proxy was found and is being stepped around.

    Once, because this is asked per fetcher and per ticker — and stated at all,
    because silently ignoring a proxy somebody configured on purpose is the
    same class of failure as silently obeying one they did not.
    """
    global _warned
    if _warned:
        return
    _warned = True
    logger.info(
        f"System proxy detected ({proxy}) and bypassed for data fetches. "
        f"Set fetching.use_system_proxy: true in config.yaml if this machine "
        f"reaches the internet only through it."
    )


def apply_to_session(session, use_system_proxy: bool = False) -> None:
    """Point a `requests.Session` around the system proxy, or leave it alone.

    `trust_env` is the switch rather than `session.proxies`, because an empty
    proxies dict does not override the environment — `requests` merges
    environment proxies in at request time unless trust_env is off.

    Note this also stops the session reading `REQUESTS_CA_BUNDLE` and
    `.netrc`. Neither is used by any source here, and a machine that needs a
    custom CA bundle is the same machine that needs the proxy — so both travel
    together under one setting rather than pretending to be independent.
    """
    if use_system_proxy:
        return
    proxy = system_proxy_in_use()
    if proxy:
        _announce(proxy)
    session.trust_env = False


@contextmanager
def bypassed(use_system_proxy: bool = False):
    """Run a block with the proxy variables removed from the environment.

    For clients that read `os.environ` themselves and take no session —
    yfinance and jugaad-data both do. Scoped to the block and restored in a
    `finally`, so a fetch cannot leave the process without the proxy settings
    the rest of it may need.
    """
    if use_system_proxy:
        yield
        return

    proxy = system_proxy_in_use()
    if proxy:
        _announce(proxy)

    saved = {key: os.environ.pop(key) for key in _PROXY_ENV_KEYS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)
