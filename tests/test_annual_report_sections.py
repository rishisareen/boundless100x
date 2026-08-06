"""Section-targeted annual report extraction, and the provenance that guards it.

Guidance and capacity plans live in MD&A, which a survey of the fetched
reports found starting between pages 20 and 145 — a first-30-pages read never
reaches it. Detection is heuristic, so the load-bearing property is not that
it always finds the section but that it never claims to have: a slot holding
first-N-pages text is tagged `fallback`, and Phase 2 sub-metrics are required
to read that tag rather than mine a chairman's letter for guidance.
"""

import json

import pytest

from boundless100x.data_fetcher.download_annual_reports import (
    AnnualReportDownloader,
    combined_text,
    find_section_starts,
)

fitz = pytest.importorskip("fitz", reason="PyMuPDF required to build PDF fixtures")

SECTIONS = {"mdna": 12000, "chairman": 6000, "governance": 8000}


def make_pdf(path, pages: list[str]) -> str:
    """A PDF whose pages carry the given text."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def body(marker: str, lines: int = 12) -> str:
    return "\n".join(f"{marker} substantive paragraph line {i}" for i in range(lines))


def structured_pages() -> list[str]:
    """A report shaped like a real one: cover, contents, then real sections."""
    return [
        "ANNUAL REPORT 2025\nSome Company Limited",
        # Contents page — lists every section name in heading form.
        "Contents\nChairman's Letter\nBoard's Report\n"
        "Management Discussion and Analysis\nCorporate Governance Report",
        "Chairman's Letter\n" + body("chairman"),
        body("filler"),
        "Management Discussion and Analysis\n" + body("mdna"),
        body("mdna continued"),
        "Report on Corporate Governance\n" + body("governance"),
    ]


@pytest.fixture
def downloader():
    return AnnualReportDownloader()


class TestSectionDetection:
    def test_finds_each_section_past_the_contents_page(self):
        starts = find_section_starts(structured_pages())
        assert starts == {"chairman": 2, "mdna": 4, "governance": 6}

    def test_contents_page_is_not_mistaken_for_a_section(self):
        """The commonest trap: a contents page lists every heading verbatim."""
        assert all(page != 1 for page in find_section_starts(structured_pages()).values())

    def test_a_page_titled_index_is_also_a_contents_page(self):
        pages = ["Index\nManagement Discussion and Analysis\nBoard's Report",
                 "Management Discussion and Analysis\n" + body("mdna")]
        assert find_section_starts(pages)["mdna"] == 1

    def test_bulk_page_number_entries_mark_a_listing(self):
        """Some reports title contents pages oddly — the page numbers give it away."""
        listing = "ACROSS THE PAGES\n" + "\n".join(
            f"{n:02d}\nSome Section Title" for n in range(2, 30, 2)
        ) + "\nManagement Discussion and Analysis"
        pages = [listing, "Management Discussion and Analysis\n" + body("mdna")]
        assert find_section_starts(pages)["mdna"] == 1

    def test_a_passing_mention_in_prose_is_not_a_heading(self):
        pages = [
            "As required we note that the management discussion and analysis "
            "forms part of this document and should be read together with it.",
            "Management Discussion and Analysis\n" + body("mdna"),
        ]
        assert find_section_starts(pages)["mdna"] == 1

    def test_absent_sections_are_simply_absent(self):
        assert find_section_starts(["cover page", body("nothing")]) == {}


class TestProvenance:
    def test_located_sections_are_marked_found(self, downloader, tmp_path):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        result = downloader.extract_sections(pdf, sections=SECTIONS)

        assert {name: s["provenance"] for name, s in result.items()} == {
            "mdna": "found", "chairman": "found", "governance": "found"
        }

    def test_a_located_section_holds_its_own_text(self, downloader, tmp_path):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        result = downloader.extract_sections(pdf, sections=SECTIONS)

        assert "mdna substantive" in result["mdna"]["text"]
        assert "chairman substantive" not in result["mdna"]["text"]

    def test_missing_sections_fall_back_and_say_so(self, downloader, tmp_path):
        """A two-page stub has no sections — real reports like this exist."""
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", ["cover", body("stub")])
        result = downloader.extract_sections(pdf, sections=SECTIONS)

        assert {s["provenance"] for s in result.values()} == {"fallback"}
        assert all(s["start_page"] is None for s in result.values())
        assert all("stub substantive" in s["text"] for s in result.values())

    def test_fallback_never_raises(self, downloader, tmp_path):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", ["only a cover page"])
        assert downloader.extract_sections(pdf, sections=SECTIONS)

    def test_an_unreadable_pdf_yields_nothing_rather_than_guessing(
        self, downloader, tmp_path
    ):
        broken = tmp_path / "2025_annual_report.pdf"
        broken.write_bytes(b"not a pdf at all")
        assert downloader.extract_sections(str(broken), sections=SECTIONS) == {}

    def test_per_section_char_caps_are_enforced(self, downloader, tmp_path):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        result = downloader.extract_sections(pdf, sections={"mdna": 50, "chairman": 40})

        assert len(result["mdna"]["text"]) <= 50
        assert len(result["chairman"]["text"]) <= 40


class TestSidecarCache:
    def test_sections_are_cached_beside_the_pdf(self, downloader, tmp_path):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        downloader.extract_sections(pdf, sections=SECTIONS)

        sidecar = tmp_path / "2025_annual_report.sections.json"
        assert sidecar.exists()
        assert set(json.loads(sidecar.read_text())) == set(SECTIONS)

    def test_a_cached_run_does_not_reopen_the_pdf(self, downloader, tmp_path, monkeypatch):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        first = downloader.extract_sections(pdf, sections=SECTIONS)

        monkeypatch.setattr(
            downloader, "_page_texts", lambda *a, **k: pytest.fail("reopened the PDF")
        )
        assert downloader.extract_sections(pdf, sections=SECTIONS) == first

    def test_changing_the_configured_sections_invalidates_the_cache(
        self, downloader, tmp_path
    ):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        downloader.extract_sections(pdf, sections={"mdna": 500})

        result = downloader.extract_sections(pdf, sections=SECTIONS)

        assert set(result) == set(SECTIONS)

    def test_a_corrupt_sidecar_is_ignored_not_fatal(self, downloader, tmp_path):
        pdf = make_pdf(tmp_path / "2025_annual_report.pdf", structured_pages())
        (tmp_path / "2025_annual_report.sections.json").write_text("{not json")

        assert downloader.extract_sections(pdf, sections=SECTIONS)["mdna"]["provenance"] == "found"


class TestCombinedText:
    """The single-string view older consumers still read."""

    def test_found_sections_are_joined_in_page_order(self):
        sections = {
            "governance": {"text": "GOV", "provenance": "found", "start_page": 6},
            "chairman": {"text": "CHAIR", "provenance": "found", "start_page": 2},
            "mdna": {"text": "MDNA", "provenance": "found", "start_page": 4},
        }
        assert combined_text(sections) == "CHAIR\n\nMDNA\n\nGOV"

    def test_all_fallback_returns_the_extract_once_not_three_times(self):
        """Repeating one extract per slot would triple today's string."""
        text = "first thirty pages of text"
        sections = {
            name: {"text": text, "provenance": "fallback", "start_page": None}
            for name in SECTIONS
        }
        assert combined_text(sections) == text

    def test_fallback_only_sections_are_dropped_when_others_were_found(self):
        sections = {
            "mdna": {"text": "MDNA", "provenance": "found", "start_page": 4},
            "chairman": {"text": "COVER PAGES", "provenance": "fallback", "start_page": None},
        }
        assert combined_text(sections) == "MDNA"

    def test_no_sections_is_an_empty_string(self):
        assert combined_text({}) == ""


class TestMultiYearRetention:
    def test_every_retained_report_is_extracted_not_just_the_newest(
        self, downloader, tmp_path, monkeypatch
    ):
        """Promises-kept compares one year's guidance to the next year's delivery."""
        paths = [
            make_pdf(tmp_path / f"{year}_annual_report.pdf", structured_pages())
            for year in (2025, 2024)
        ]
        monkeypatch.setattr(downloader, "download", lambda *a, **k: paths)

        by_year = downloader.download_and_extract(
            "500001", str(tmp_path), sections=SECTIONS
        )

        assert sorted(by_year) == ["2024", "2025"]
        assert by_year["2024"]["mdna"]["provenance"] == "found"

    def test_no_downloads_yields_an_empty_mapping(self, downloader, tmp_path, monkeypatch):
        monkeypatch.setattr(downloader, "download", lambda *a, **k: [])
        assert downloader.download_and_extract("500001", str(tmp_path)) == {}
