"""Snapshot and restore the fetched corpus, so a refetch is reversible.

`raw_data/` is gitignored (`.gitignore:15`) and is the only copy of everything
the fetchers have ever pulled. There is no revert. A refetch that degrades a
ticker — Screener's quarters section failing to parse, a price feed returning a
short series — writes over the good copy and leaves nothing to compare against,
and `_save_to_disk`'s per-file conditional write means the damage is *partial*:
fresh files beside stale ones, with nothing in the pipeline's own account of
itself saying which is which (KTD3). So the snapshot comes first.

**The destination is deliberately outside the repository tree** (KTD1). The
repo sits in an iCloud-synced directory and the corpus is ~370MB; an in-tree
copy would push the full amount through sync, might not settle into a stable
copy while it is being read, and would double the working set of every later
`du`. The default is a plain local directory under the home tree, config can
move it, and a destination inside the repo is refused with the reason.

The manifest is what makes the snapshot *auditable* rather than merely
restorable. It records, per corpus directory, every file with its size plus the
three facts U3's coverage audit compares against — whether a quarterly series
exists, whether the price series carries `adj_close`, and which annual-report
years are held — so the audit can diff before against after without walking
370MB a second time.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from boundless100x.data_fetcher.download_annual_reports import (
    cached_report_years,
    load_cached_sections,
)

logger = logging.getLogger(__name__)

# The repository root: this file is `<repo>/boundless100x/data_fetcher/`.
REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_NAME = "manifest.json"
SNAPSHOT_PREFIX = "raw_data-"

# Shared state, outside the repository and outside the corpus. Named once so
# the snapshot directory and the refetch run log cannot drift apart — deriving
# one from the other by `.parent` made the coupling invisible from both ends.
STATE_DIR = Path.home() / ".boundless100x"

# A directory holding this is a real ticker rather than a BSE-code directory of
# annual report PDFs. **This is the definition** — `backtest` and `pace` import
# it from here. `data_fetcher` writes `financials.csv`, so the constant belongs
# in the layer that produces it, and `pace.py` already carries the note saying
# two copies could drift into disagreeing about which companies the corpus
# contains. This module briefly made that a third copy.
TICKER_MARKER = "financials.csv"

# Outside the synced tree, and outside the repo. Overridable from config under
# `corpus_snapshot.dir`, read the way `service.history_path` reads its own.
DEFAULT_SNAPSHOT_DIR = STATE_DIR / "corpus_snapshots"


class SnapshotError(RuntimeError):
    """A snapshot or restore that must not proceed silently."""


def snapshot_root(config: dict | None = None) -> Path:
    """Where snapshots live: config's `corpus_snapshot.dir`, or the default."""
    configured = (config or {}).get("corpus_snapshot", {}).get("dir")
    return Path(configured).expanduser() if configured else DEFAULT_SNAPSHOT_DIR


def _refuse_if_inside_repo(destination: Path) -> None:
    resolved = destination.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SnapshotError(
            f"refusing to write a corpus snapshot inside the repository "
            f"({resolved}). The repo sits in an iCloud-synced directory and the "
            f"corpus is hundreds of megabytes — an in-tree copy churns the full "
            f"amount through sync and may not settle into a stable copy while "
            f"it is being written. Point corpus_snapshot.dir at a local path "
            f"outside {REPO_ROOT}."
        )


def _price_has_adj_close(path: Path) -> bool | None:
    """Whether a price series carries the adjusted column, or None if absent.

    Reads the header line only — the corpus holds ten years of daily bars per
    ticker and the question is about a column name.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline()
    except OSError:
        return None
    return "adj_close" in [column.strip() for column in header.split(",")]


def _mdna_found_years(directory: Path) -> list[str]:
    """Report years whose sections sidecar tags MD&A as `found`.

    Raw detection provenance, **before the content gate** — that gate lives in
    `llm_layer`, which this layer must not import. So this is an upper bound on
    usable MD&A years, and anything reporting it must say so; U7's own
    measurement is the gated figure.
    """
    return sorted(
        year
        for year, sections in load_cached_sections(directory.parent, directory.name).items()
        if isinstance((sections or {}).get("mdna"), dict)
        and sections["mdna"].get("provenance") == "found"
    )


def describe_directory(directory: Path) -> dict:
    """One corpus directory's files, sizes, and the facts the audit compares.

    Files are keyed by their path relative to the directory, so an annual
    report nested under `annual_reports/` is distinguishable from a top-level
    CSV and a file that vanished from either is visible.
    """
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(directory))] = path.stat().st_size

    return {
        "files": files,
        "file_count": len(files),
        "bytes": sum(files.values()),
        "is_ticker": (directory / TICKER_MARKER).exists(),
        "has_quarterly": (directory / "quarterly.csv").exists(),
        "has_adj_close": _price_has_adj_close(directory / "price_volume.csv"),
        "annual_report_years": cached_report_years(directory.parent, directory.name),
        "mdna_found_years": _mdna_found_years(directory),
    }


def describe_corpus(source) -> dict:
    """The manifest body for a corpus directory, without copying anything."""
    root = Path(source)
    entries = {
        child.name: describe_directory(child)
        for child in sorted(root.iterdir())
        if child.is_dir()
    }
    return {
        "source": str(root.resolve()),
        "entries": entries,
        "totals": {
            "directories": len(entries),
            "files": sum(entry["file_count"] for entry in entries.values()),
            "bytes": sum(entry["bytes"] for entry in entries.values()),
        },
    }


def snapshot(source, destination=None, config: dict | None = None) -> dict:
    """Copy the corpus to a timestamped directory and write its manifest.

    Fails rather than producing an empty snapshot when the corpus is absent or
    holds no directories: an empty snapshot restores as a *deletion*, which is
    the one outcome a recovery mechanism must never quietly offer.
    """
    root = Path(source)
    if not root.is_dir():
        raise SnapshotError(
            f"no corpus at {root} — refusing to write an empty snapshot, which "
            f"would restore as a deletion of whatever is there later"
        )

    body = describe_corpus(root)
    if not body["entries"]:
        raise SnapshotError(
            f"corpus at {root} holds no ticker directories — refusing to write "
            f"an empty snapshot, which would restore as a deletion"
        )

    base = Path(destination) if destination else snapshot_root(config)
    _refuse_if_inside_repo(base)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = base / f"{SNAPSHOT_PREFIX}{stamp}"
    if target.exists():
        raise SnapshotError(f"snapshot {target} already exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Snapshotting {body['totals']['files']} files "
        f"({body['totals']['bytes'] / 1e6:.0f}MB) to {target}"
    )
    shutil.copytree(root, target / "raw_data")

    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"), **body}
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": target, "manifest": manifest}


def load_manifest(snapshot_path) -> dict:
    """A snapshot's manifest, so the audit need not re-walk the copy."""
    path = Path(snapshot_path) / MANIFEST_NAME
    if not path.exists():
        raise SnapshotError(f"no manifest at {path} — not a corpus snapshot")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_snapshot(base=None, config: dict | None = None) -> Path | None:
    """The newest snapshot under `base`, or None.

    Names are timestamped to the second in a sortable format, so lexical order
    is chronological order and no filesystem mtime is consulted.
    """
    root = Path(base) if base else snapshot_root(config)
    if not root.is_dir():
        return None
    candidates = sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name.startswith(SNAPSHOT_PREFIX)
        and (child / MANIFEST_NAME).exists()
    )
    return candidates[-1] if candidates else None


def restore(snapshot_path, destination) -> dict:
    """Put a snapshot back, replacing whatever is at `destination`.

    Replaces rather than merges. A half-restored corpus — some tickers from the
    snapshot, some from the run that went wrong — is worse than either state on
    its own, because nothing afterwards can tell which files came from where.
    """
    source = Path(snapshot_path)
    payload = source / "raw_data"
    if not payload.is_dir():
        raise SnapshotError(f"no raw_data payload inside {source}")
    manifest = load_manifest(source)  # refuses anything that is not a snapshot

    target = Path(destination)
    if target.exists():
        logger.info(f"Removing {target} before restore (replace, never merge)")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(payload, target)

    # The walk that describes what landed also *checks* it against the manifest
    # written when the snapshot was taken. It was already being paid for and
    # its result thrown away; a recovery path that reports a file count without
    # comparing it to the one it promised is not a recovery path anyone should
    # trust. A mismatch warns rather than raises — the files are already back,
    # and refusing after the fact would help nobody.
    restored = describe_corpus(target)
    expected = (manifest or {}).get("totals") or {}
    if expected and expected != restored["totals"]:
        logger.warning(
            f"Restored corpus does not match the snapshot manifest: "
            f"expected {expected}, found {restored['totals']} — the files are "
            f"in place, but investigate before discarding {source}"
        )
    else:
        logger.info(
            f"Restored {restored['totals']['files']} files to {target} from "
            f"{source}, matching its manifest"
        )
    return restored
