"""U1 — the corpus snapshot is the only thing standing between a refetch and
an unrecoverable corpus, so its failure modes are tested harder than its
happy path.
"""

import json

import pytest

from boundless100x.data_fetcher import corpus_snapshot


def build_corpus(root, tickers=("ASTRAL", "VBL"), with_quarterly=("ASTRAL",),
                 with_adj_close=("ASTRAL",)):
    """A miniature raw_data tree with the real file names and layout."""
    root.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        directory = root / ticker
        directory.mkdir()
        (directory / "financials.csv").write_text("year,revenue\nMar 2025,100\n")
        (directory / "metadata.json").write_text(json.dumps({"bse_code": "500001"}))
        if ticker in with_quarterly:
            (directory / "quarterly.csv").write_text("quarter,revenue\nMar 2025,25\n")
        header = (
            "date,open,high,low,close,adj_close,adj_close_is_estimated,volume"
            if ticker in with_adj_close
            else "date,open,high,low,close,volume"
        )
        (directory / "price_volume.csv").write_text(f"{header}\n")
    return root


def build_bse_dir(root, code="500001", years=("2025",), mdna_found=("2025",)):
    reports = root / code / "annual_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for year in years:
        (reports / f"{year}_annual_report.pdf").write_bytes(b"%PDF-1.4 stub")
        provenance = "found" if year in mdna_found else "fallback"
        (reports / f"{year}_annual_report.sections.json").write_text(
            json.dumps({"mdna": {"text": "x", "provenance": provenance}})
        )
    return reports


def test_restore_reproduces_every_file_byte_identically(tmp_path):
    source = build_corpus(tmp_path / "raw_data")
    build_bse_dir(source)
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }

    made = corpus_snapshot.snapshot(source, destination=tmp_path / "snaps")
    corpus_snapshot.restore(made["path"], source)

    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }
    assert after == before


def test_manifest_records_per_directory_file_counts_and_audit_facts(tmp_path):
    source = build_corpus(tmp_path / "raw_data")
    build_bse_dir(source, years=("2024", "2025"), mdna_found=("2025",))

    manifest = corpus_snapshot.snapshot(
        source, destination=tmp_path / "snaps"
    )["manifest"]

    entries = manifest["entries"]
    assert set(entries) == {"ASTRAL", "VBL", "500001"}
    assert entries["ASTRAL"]["file_count"] == 4
    assert entries["ASTRAL"]["is_ticker"] is True
    assert entries["ASTRAL"]["has_quarterly"] is True
    assert entries["ASTRAL"]["has_adj_close"] is True
    assert entries["VBL"]["has_quarterly"] is False
    assert entries["VBL"]["has_adj_close"] is False
    assert entries["500001"]["is_ticker"] is False
    assert entries["500001"]["annual_report_years"] == ["2024", "2025"]
    assert entries["500001"]["mdna_found_years"] == ["2025"]
    assert manifest["totals"]["directories"] == 3


def test_manifest_alone_answers_the_audit_without_rewalking_the_copy(tmp_path):
    source = build_corpus(tmp_path / "raw_data")
    made = corpus_snapshot.snapshot(source, destination=tmp_path / "snaps")

    reloaded = corpus_snapshot.load_manifest(made["path"])
    assert reloaded["entries"] == made["manifest"]["entries"]


def test_destination_inside_the_repository_is_refused_with_the_reason(tmp_path):
    source = build_corpus(tmp_path / "raw_data")

    with pytest.raises(corpus_snapshot.SnapshotError) as excinfo:
        corpus_snapshot.snapshot(
            source, destination=corpus_snapshot.REPO_ROOT / "snapshots"
        )

    message = str(excinfo.value)
    assert "sync" in message
    assert "outside" in message


def test_restore_replaces_rather_than_merges(tmp_path):
    source = build_corpus(tmp_path / "raw_data")
    made = corpus_snapshot.snapshot(source, destination=tmp_path / "snaps")

    # A refetch that invented a ticker and corrupted an existing one.
    (source / "GHOST").mkdir()
    (source / "GHOST" / "financials.csv").write_text("year,revenue\n")
    (source / "ASTRAL" / "financials.csv").write_text("truncated\n")

    corpus_snapshot.restore(made["path"], source)

    assert not (source / "GHOST").exists()
    assert (source / "ASTRAL" / "financials.csv").read_text().startswith("year,revenue")


def test_restore_verifies_what_landed_against_the_manifest(tmp_path, caplog):
    """The walk that describes the restore also checks it — it was free anyway."""
    source = build_corpus(tmp_path / "raw_data")
    made = corpus_snapshot.snapshot(source, destination=tmp_path / "snaps")

    with caplog.at_level("INFO"):
        corpus_snapshot.restore(made["path"], source)

    assert "matching its manifest" in caplog.text


def test_a_restore_that_does_not_match_its_manifest_warns(tmp_path, caplog):
    source = build_corpus(tmp_path / "raw_data")
    made = corpus_snapshot.snapshot(source, destination=tmp_path / "snaps")
    # A payload that no longer matches the manifest written beside it.
    (made["path"] / "raw_data" / "ASTRAL" / "financials.csv").unlink()

    with caplog.at_level("WARNING"):
        corpus_snapshot.restore(made["path"], source)

    assert "does not match the snapshot manifest" in caplog.text
    assert (source / "ASTRAL").exists()  # the files are still put back


def test_ticker_marker_has_one_definition(tmp_path):
    """`pace.py` carries the note saying two copies could drift; this is it."""
    from boundless100x.compute_engine import backtest

    assert backtest.TICKER_MARKER is corpus_snapshot.TICKER_MARKER


def test_snapshot_of_an_absent_corpus_fails_clearly(tmp_path):
    with pytest.raises(corpus_snapshot.SnapshotError) as excinfo:
        corpus_snapshot.snapshot(tmp_path / "nothing_here",
                                 destination=tmp_path / "snaps")
    assert "deletion" in str(excinfo.value)


def test_snapshot_of_an_empty_corpus_directory_fails_clearly(tmp_path):
    empty = tmp_path / "raw_data"
    empty.mkdir()

    with pytest.raises(corpus_snapshot.SnapshotError) as excinfo:
        corpus_snapshot.snapshot(empty, destination=tmp_path / "snaps")
    assert "deletion" in str(excinfo.value)


def test_restoring_something_that_is_not_a_snapshot_is_refused(tmp_path):
    source = build_corpus(tmp_path / "raw_data")
    fake = tmp_path / "fake"
    (fake / "raw_data").mkdir(parents=True)

    with pytest.raises(corpus_snapshot.SnapshotError):
        corpus_snapshot.restore(fake, source)


def test_latest_snapshot_picks_the_newest_by_sortable_name(tmp_path):
    base = tmp_path / "snaps"
    for stamp in ("20260801-090000", "20260807-120000", "20260803-235959"):
        made = base / f"{corpus_snapshot.SNAPSHOT_PREFIX}{stamp}"
        (made / "raw_data").mkdir(parents=True)
        (made / corpus_snapshot.MANIFEST_NAME).write_text("{}")

    assert corpus_snapshot.latest_snapshot(base).name.endswith("20260807-120000")


def test_latest_snapshot_ignores_a_directory_with_no_manifest(tmp_path):
    base = tmp_path / "snaps"
    good = base / f"{corpus_snapshot.SNAPSHOT_PREFIX}20260801-090000"
    (good / "raw_data").mkdir(parents=True)
    (good / corpus_snapshot.MANIFEST_NAME).write_text("{}")
    (base / f"{corpus_snapshot.SNAPSHOT_PREFIX}20260809-090000").mkdir(parents=True)

    assert corpus_snapshot.latest_snapshot(base) == good


def test_snapshot_root_reads_config_then_falls_back(tmp_path):
    assert corpus_snapshot.snapshot_root({}) == corpus_snapshot.DEFAULT_SNAPSHOT_DIR
    assert corpus_snapshot.snapshot_root(
        {"corpus_snapshot": {"dir": str(tmp_path / "elsewhere")}}
    ) == tmp_path / "elsewhere"
