"""What a refetch actually changed, read off the corpus rather than the run.

**The pipeline cannot answer this question about itself** (KTD3). `_save_to_disk`
writes each artifact only `if not df.empty` (`fetch_financials.py:397`), so a
run where Screener's quarters section fails to parse leaves the old
`quarterly.csv` in place, refreshes everything around it, and reports nothing
wrong. `source_status["financials"]` reflects the P&L table alone
(`suite.py:105-107`). From inside the pipeline, a refetch that did not fix the
missing quarterly series is indistinguishable from one that did.

So this counts files on disk, before against after, with "before" coming from
the snapshot's manifest — the one record written while the old corpus was still
intact.

**Regressions are reported apart from gains, and never rolled into a net.** A
file that got smaller is the partial-write signature the whole snapshot exists
for; averaged against twenty tickers that gained a quarterly series it would
disappear entirely. An unchanged corpus reports every directory as unchanged
rather than reporting nothing, because "no output" and "nothing moved" must not
look the same either.
"""

import logging
from pathlib import Path

from boundless100x.data_fetcher.corpus_snapshot import describe_corpus, load_manifest

logger = logging.getLogger(__name__)

# Presence transitions, named so a report reads without a legend.
GAINED = "gained"
HELD = "held"
LOST = "lost"
ABSENT = "still_absent"
UNKNOWN = "unknown"


def _presence(before, after) -> str:
    """How a boolean corpus fact moved. `None` means the file was not there."""
    if before is None and after is None:
        return UNKNOWN
    if bool(after) and not bool(before):
        return GAINED
    if bool(before) and not bool(after):
        return LOST
    return HELD if after else ABSENT


def _compare_years(before: list, after: list) -> dict:
    before_set, after_set = set(before or []), set(after or [])
    return {
        "before": sorted(before_set),
        "after": sorted(after_set),
        "added": sorted(after_set - before_set),
        "removed": sorted(before_set - after_set),
    }


def _compare_files(before: dict, after: dict) -> dict:
    before = before or {}
    after = after or {}
    shrank = [
        {
            "path": path,
            "before": before[path],
            "after": after[path],
            "shrink_pct": round((before[path] - after[path]) / before[path] * 100, 1)
            if before[path] else 0.0,
        }
        for path in sorted(set(before) & set(after))
        if after[path] < before[path]
    ]
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "shrank": shrank,
    }


def audit(raw_data_dir, manifest: dict) -> dict:
    """Compare the corpus on disk against the manifest written before the run."""
    after = describe_corpus(raw_data_dir)
    before_entries = (manifest or {}).get("entries") or {}
    after_entries = after["entries"]

    directories: dict[str, dict] = {}
    regressions: list[dict] = []

    for name in sorted(set(before_entries) | set(after_entries)):
        old = before_entries.get(name)
        new = after_entries.get(name)

        if new is None:
            directories[name] = {"status": "disappeared", "is_ticker": old["is_ticker"]}
            regressions.append({
                "directory": name, "kind": "directory_disappeared",
                "detail": f"{old['file_count']} file(s) were here before the run",
            })
            continue

        entry = {
            "status": "new" if old is None else "present",
            "is_ticker": new["is_ticker"],
            "quarterly": _presence(
                (old or {}).get("has_quarterly", False), new["has_quarterly"]
            ),
            "adj_close": _presence(
                (old or {}).get("has_adj_close"), new["has_adj_close"]
            ),
            "annual_report_years": _compare_years(
                (old or {}).get("annual_report_years", []), new["annual_report_years"]
            ),
            "mdna_found_years": _compare_years(
                (old or {}).get("mdna_found_years", []), new["mdna_found_years"]
            ),
            "files": _compare_files((old or {}).get("files", {}), new["files"]),
        }
        directories[name] = entry

        for path in entry["files"]["removed"]:
            regressions.append({
                "directory": name, "kind": "file_removed", "detail": path,
            })
        for shrunk in entry["files"]["shrank"]:
            regressions.append({
                "directory": name, "kind": "file_shrank",
                "detail": (
                    f"{shrunk['path']}: {shrunk['before']} -> {shrunk['after']} bytes "
                    f"({shrunk['shrink_pct']}% smaller)"
                ),
            })
        if entry["quarterly"] == LOST:
            regressions.append({
                "directory": name, "kind": "quarterly_lost",
                "detail": "quarterly.csv was present before the run and is not now",
            })
        if entry["adj_close"] == LOST:
            regressions.append({
                "directory": name, "kind": "adj_close_lost",
                "detail": "price_volume.csv carried adj_close before the run",
            })
        for year in entry["annual_report_years"]["removed"]:
            regressions.append({
                "directory": name, "kind": "annual_report_year_removed",
                "detail": f"{year} annual report is no longer on disk",
            })

    # A directory that vanished carries no after-state to roll up; it is
    # already in `regressions`, which is where it belongs.
    surviving = {
        n: e for n, e in directories.items() if e.get("status") != "disappeared"
    }
    tickers = {n: e for n, e in surviving.items() if e.get("is_ticker")}
    two_plus = [
        name for name, entry in surviving.items()
        if len(entry["mdna_found_years"]["after"]) >= 2
    ]
    two_plus_before = [
        name for name, entry in before_entries.items()
        if len(entry.get("mdna_found_years") or []) >= 2
    ]

    headline = {
        "tickers_total": len(tickers),
        "gained_quarterly": sorted(
            n for n, e in tickers.items() if e["quarterly"] == GAINED
        ),
        "still_without_quarterly": sorted(
            n for n, e in tickers.items() if e["quarterly"] == ABSENT
        ),
        "gained_adj_close": sorted(
            n for n, e in tickers.items() if e["adj_close"] == GAINED
        ),
        "still_without_adj_close": sorted(
            n for n, e in tickers.items() if e["adj_close"] == ABSENT
        ),
        "gained_report_years": sorted(
            n for n, e in surviving.items() if e["annual_report_years"]["added"]
        ),
        "report_years_added": sum(
            len(e["annual_report_years"]["added"]) for e in surviving.values()
        ),
        "two_or_more_mdna_years_before": sorted(two_plus_before),
        "two_or_more_mdna_years_after": sorted(two_plus),
        "regressions": len(regressions),
    }

    if regressions:
        logger.warning(
            f"Corpus audit: {len(regressions)} regression(s) — investigate before "
            f"discarding the snapshot"
        )

    return {
        "before": {
            "source": (manifest or {}).get("source"),
            "created_at": (manifest or {}).get("created_at"),
            "totals": (manifest or {}).get("totals"),
        },
        "after": {"source": after["source"], "totals": after["totals"]},
        "directories": directories,
        "regressions": regressions,
        "headline": headline,
    }


def audit_against_snapshot(raw_data_dir, snapshot_path) -> dict:
    """The audit, with the before-side read from a snapshot's manifest."""
    return audit(Path(raw_data_dir), load_manifest(snapshot_path))
