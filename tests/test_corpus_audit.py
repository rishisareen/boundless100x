"""U3 — the audit reads the corpus, not the pipeline's account of itself.

Every case here is a shape the pipeline would report as a clean run: per-file
conditional writes mean a partial parse failure leaves stale files in place and
says nothing (KTD3). The audit's whole value is that it notices anyway.
"""

import json

from boundless100x.data_fetcher import corpus_audit, corpus_snapshot


def write_ticker(root, ticker, quarterly=False, adj_close=False, rows=1):
    directory = root / ticker
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "financials.csv").write_text(
        "year,revenue\n" + "".join(f"Mar 202{i},100\n" for i in range(rows))
    )
    (directory / "metadata.json").write_text(json.dumps({"bse_code": "500001"}))
    header = (
        "date,open,high,low,close,adj_close,adj_close_is_estimated,volume"
        if adj_close else "date,open,high,low,close,volume"
    )
    (directory / "price_volume.csv").write_text(f"{header}\n")
    if quarterly:
        (directory / "quarterly.csv").write_text("quarter,revenue\nMar 2025,25\n")
    else:
        (directory / "quarterly.csv").unlink(missing_ok=True)
    return directory


def write_reports(root, code, years, with_sidecar=(), mdna="found"):
    reports = root / code / "annual_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for year in years:
        (reports / f"{year}_annual_report.pdf").write_bytes(b"%PDF stub")
        if year in with_sidecar:
            (reports / f"{year}_annual_report.sections.json").write_text(
                json.dumps({"mdna": {"text": "x", "provenance": mdna}})
            )
    return reports


def manifest_of(root):
    return corpus_snapshot.describe_corpus(root)


def test_two_tickers_that_gained_a_quarterly_series_are_reported_exactly(tmp_path):
    root = tmp_path / "raw_data"
    for ticker in ("ASTRAL", "VBL", "CDSL"):
        write_ticker(root, ticker, quarterly=False)
    before = manifest_of(root)

    write_ticker(root, "ASTRAL", quarterly=True)
    write_ticker(root, "VBL", quarterly=True)

    report = corpus_audit.audit(root, before)

    assert report["headline"]["gained_quarterly"] == ["ASTRAL", "VBL"]
    assert report["headline"]["still_without_quarterly"] == ["CDSL"]
    assert report["regressions"] == []


def test_a_shrunk_file_is_a_regression_not_a_gain(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL", quarterly=True, rows=8)
    before = manifest_of(root)

    # The partial-write signature: quarters parsed short, everything else fine.
    (root / "ASTRAL" / "financials.csv").write_text("year,revenue\nMar 2025,100\n")

    report = corpus_audit.audit(root, before)

    kinds = [r["kind"] for r in report["regressions"]]
    assert kinds == ["file_shrank"]
    assert "financials.csv" in report["regressions"][0]["detail"]
    assert report["headline"]["regressions"] == 1
    assert report["headline"]["gained_quarterly"] == []


def test_a_ticker_that_lost_a_file_is_a_regression(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL", quarterly=True)
    before = manifest_of(root)

    (root / "ASTRAL" / "quarterly.csv").unlink()

    report = corpus_audit.audit(root, before)

    kinds = {r["kind"] for r in report["regressions"]}
    assert kinds == {"file_removed", "quarterly_lost"}
    assert report["directories"]["ASTRAL"]["quarterly"] == corpus_audit.LOST


def test_a_ticker_that_vanished_entirely_is_a_regression(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL")
    write_ticker(root, "VBL")
    before = manifest_of(root)

    import shutil
    shutil.rmtree(root / "VBL")

    report = corpus_audit.audit(root, before)

    assert [r["kind"] for r in report["regressions"]] == ["directory_disappeared"]
    assert report["directories"]["VBL"]["status"] == "disappeared"


def test_an_unchanged_corpus_reports_every_directory_and_no_movement(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL", quarterly=True, adj_close=True)
    write_ticker(root, "VBL")
    before = manifest_of(root)

    report = corpus_audit.audit(root, before)

    assert set(report["directories"]) == {"ASTRAL", "VBL"}
    assert report["regressions"] == []
    assert report["headline"]["gained_quarterly"] == []
    assert report["headline"]["gained_adj_close"] == []
    assert report["directories"]["ASTRAL"]["quarterly"] == corpus_audit.HELD
    assert report["directories"]["VBL"]["quarterly"] == corpus_audit.ABSENT


def test_a_report_year_with_no_sections_sidecar_still_counts_as_held(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL")
    write_reports(root, "500001", ["2025"], with_sidecar=["2025"])
    before = manifest_of(root)

    # A refetch that downloaded two older years but has not extracted them yet.
    write_reports(root, "500001", ["2023", "2024", "2025"], with_sidecar=["2025"])

    report = corpus_audit.audit(root, before)

    years = report["directories"]["500001"]["annual_report_years"]
    assert years["added"] == ["2023", "2024"]
    assert years["after"] == ["2023", "2024", "2025"]
    assert report["headline"]["report_years_added"] == 2
    assert report["headline"]["gained_report_years"] == ["500001"]


def test_adj_close_gain_is_reported_and_a_missing_price_file_is_not(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "IDEA", adj_close=False)
    write_ticker(root, "RAIN", adj_close=True)
    (root / "NOPRICE").mkdir()
    (root / "NOPRICE" / "metadata.json").write_text("{}")
    before = manifest_of(root)

    write_ticker(root, "IDEA", adj_close=True)

    report = corpus_audit.audit(root, before)

    assert report["headline"]["gained_adj_close"] == ["IDEA"]
    assert report["directories"]["RAIN"]["adj_close"] == corpus_audit.HELD
    assert report["directories"]["NOPRICE"]["adj_close"] == corpus_audit.UNKNOWN


def test_mdna_year_counts_move_from_before_to_after(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL")
    write_reports(root, "500001", ["2025"], with_sidecar=["2025"])
    before = manifest_of(root)

    write_reports(root, "500001", ["2024", "2025"], with_sidecar=["2024", "2025"])

    report = corpus_audit.audit(root, before)

    assert report["headline"]["two_or_more_mdna_years_before"] == []
    assert report["headline"]["two_or_more_mdna_years_after"] == ["500001"]


def test_a_brand_new_ticker_directory_is_a_gain_not_a_regression(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL", quarterly=True)
    before = manifest_of(root)

    write_ticker(root, "NEWCO", quarterly=True)

    report = corpus_audit.audit(root, before)

    assert report["directories"]["NEWCO"]["status"] == "new"
    assert report["headline"]["gained_quarterly"] == ["NEWCO"]
    assert report["regressions"] == []


def test_audit_against_snapshot_reads_the_manifest(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker(root, "ASTRAL")
    made = corpus_snapshot.snapshot(root, destination=tmp_path / "snaps")

    write_ticker(root, "ASTRAL", quarterly=True)
    report = corpus_audit.audit_against_snapshot(root, made["path"])

    assert report["headline"]["gained_quarterly"] == ["ASTRAL"]
    assert report["before"]["created_at"] == made["manifest"]["created_at"]
