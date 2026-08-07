"""Refresh every cached ticker in one operation, reaching the network.

The corpus predates most of what the pipeline now reads. `quarterly.csv`
postdates the fetch of 17 of the 22 cached tickers, `annual_reports.max_reports`
applies only from its landing forward, and 13 tickers still carry the legacy
price schema with no `adj_close`. None of that is a metric defect; it is fetch
vintage, and one operation fixes it.

Three properties this loop must have, each for a reason that has already bitten
something in this repo:

**It must actually reach the network** (KTD2). Five tickers were fetched within
the last day, and the fetch cache's TTL is 24 hours — without a bypass a
refetch run soon would silently serve them from cache and report success.
`CacheManager.clear_all` and `invalidate` existed with zero call sites; the
flag wires them up rather than asking an operator to hand-delete a directory or
edit `cache_ttl_hours` (a persistent change made for a transient reason, and
the easiest to forget to undo). The BSE scrip master is exempted: it is a
shared week-TTL index this run has no reason to invalidate.

**One ticker's failure must not end the run**, exactly as `advance()` isolates
each tracked company. Twenty-one good refetches are not worth losing to one
Screener markup change.

**An interrupted run must resume.** A full pass is 15-35 minutes of rate-limited
scraping, so the log records what completed and a resumed run skips it.

Deliberately refuses to start without a snapshot (U1). The corpus is gitignored
and is the only copy; a partial-write regression with nothing to compare
against is the failure this whole plan is arranged around.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from boundless100x.data_fetcher import bse_codes, corpus_snapshot

logger = logging.getLogger(__name__)

# Operational state, outside the repository and outside `raw_data/` — a restore
# replaces the corpus wholesale, and a run log living inside it would vanish
# with the run it describes.
DEFAULT_RUN_LOG = corpus_snapshot.DEFAULT_SNAPSHOT_DIR.parent / "refetch_log.json"

# A completed fetch always writes this. Its absence is what separates a real
# ticker directory from the wreckage of a fetch that failed under a wrong
# symbol (A3: `raw_data/ZYDUS` holds one stray analyst-coverage file).
METADATA_MARKER = "metadata.json"


def enumerate_tickers(raw_data_dir) -> tuple[list[str], list[dict]]:
    """The real tickers to refetch, and what was skipped with the reason.

    Two kinds of directory are not tickers. A numeric one is a BSE scrip code
    holding annual report PDFs — refreshed as part of its ticker's own fetch,
    never on its own, since BSE data is reached through the code the ticker
    resolves to. A directory with no `metadata.json` never completed a fetch.

    Both are *reported* rather than silently dropped: a run that quietly
    skipped a ticker would be indistinguishable from one that refreshed it.
    """
    root = Path(raw_data_dir)
    if not root.is_dir():
        return [], []

    tickers: list[str] = []
    skipped: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.isdigit():
            skipped.append({
                "name": child.name,
                "reason": "BSE-code directory (annual reports) — refreshed "
                          "through its ticker, not on its own",
            })
            continue
        if not (child / METADATA_MARKER).exists():
            skipped.append({
                "name": child.name,
                "reason": f"no {METADATA_MARKER} — never completed a fetch, so "
                          f"there is no cached company to refresh",
            })
            continue
        tickers.append(child.name)
    return tickers, skipped


def bypass_fetch_cache(suite) -> int:
    """Drop the fetch cache so this run reaches the network (KTD2).

    Scoped to the fetch cache. Every fetcher shares one cache directory, and
    the BSE scrip master lives in it on a week-long TTL — it is an index across
    all tickers rather than any ticker's data, so clearing it would only cost
    the run a re-download before it could resolve a single code.
    """
    caches = {
        id(fetcher.cache): fetcher.cache
        for fetcher in (
            suite.financials, suite.price_volume, suite.shareholding_bse,
            suite.corporate_actions, suite.analyst_coverage, suite.annual_reports,
        )
    }
    removed = sum(
        cache.clear_all(keep=(bse_codes.CACHE_KEY,)) for cache in caches.values()
    )
    logger.info(f"Fetch cache bypassed: {removed} cached entries removed")
    return removed


def read_run_log(path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Ignoring unreadable refetch log {target}: {e}")
        return {}
    return stored if isinstance(stored, dict) else {}


def _write_run_log(path, log: dict) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    except OSError as e:
        # Losing the log costs a resume, never the fetches already on disk.
        logger.warning(f"Could not write refetch log {target}: {e}")


def refetch(
    suite,
    tickers: list[str] | None = None,
    bypass_cache: bool = True,
    run_log_path=None,
    resume: bool = True,
    require_snapshot: bool = True,
    snapshot_config: dict | None = None,
) -> dict:
    """Refetch every cached ticker, one at a time, isolated and resumable.

    Returns per-ticker outcomes plus what was skipped and why. Nothing here
    scores, computes, or writes score history — this is a fetch, and mixing a
    scoring pass into it would make the run's 15-35 minutes hostage to a
    metric bug.
    """
    if require_snapshot and corpus_snapshot.latest_snapshot(
        config=snapshot_config
    ) is None:
        raise corpus_snapshot.SnapshotError(
            f"no corpus snapshot found under "
            f"{corpus_snapshot.snapshot_root(snapshot_config)} — refusing to "
            f"refetch. The corpus is gitignored and is the only copy, and a "
            f"partial parse failure mixes fresh files with stale ones "
            f"undetectably. Run `corpus snapshot` first, or pass "
            f"require_snapshot=False if you have a copy elsewhere."
        )

    candidates, skipped = enumerate_tickers(suite.raw_data_dir)
    if tickers is not None:
        requested = [t.upper() for t in tickers]
        candidates = [t for t in candidates if t.upper() in requested]

    log_path = Path(run_log_path) if run_log_path else DEFAULT_RUN_LOG
    log = read_run_log(log_path) if resume else {}
    completed = log.get("completed") or {}
    if not resume:
        completed = {}

    already = [t for t in candidates if t in completed]
    pending = [t for t in candidates if t not in completed]
    if already:
        logger.info(
            f"Resuming: skipping {len(already)} ticker(s) already completed "
            f"in {log_path.name}"
        )

    if bypass_cache and pending:
        bypass_fetch_cache(suite)

    log = {
        "started_at": log.get("started_at") or datetime.now().isoformat(timespec="seconds"),
        "raw_data": str(suite.raw_data_dir),
        "completed": dict(completed),
        "failed": dict(log.get("failed") or {}),
    }

    outcomes: list[dict] = []
    for index, ticker in enumerate(pending, start=1):
        logger.info(f"[{index}/{len(pending)}] Refetching {ticker}")
        started = time.time()
        try:
            data = suite.fetch_all(ticker)
        except Exception as e:
            # Per-ticker isolation, the same rule `advance()` follows: one
            # ticker's failure is not a reason to abandon the other twenty-one.
            elapsed = round(time.time() - started, 1)
            logger.error(f"{ticker}: refetch failed after {elapsed}s: {e}")
            outcomes.append({
                "ticker": ticker, "status": "failed",
                "detail": str(e), "seconds": elapsed,
            })
            log["failed"][ticker] = {
                "at": datetime.now().isoformat(timespec="seconds"), "error": str(e),
            }
            log["completed"].pop(ticker, None)
            _write_run_log(log_path, log)
            continue

        elapsed = round(time.time() - started, 1)
        status = data.get("source_status", {}) if isinstance(data, dict) else {}
        outcomes.append({
            "ticker": ticker, "status": "ok",
            "detail": "; ".join(f"{k}={v}" for k, v in sorted(status.items())),
            "seconds": elapsed,
        })
        log["completed"][ticker] = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "seconds": elapsed,
            "source_status": status,
        }
        log["failed"].pop(ticker, None)
        _write_run_log(log_path, log)

    return {
        "outcomes": outcomes,
        "skipped": skipped,
        "resumed": already,
        "run_log": str(log_path),
        "tickers": candidates,
    }
