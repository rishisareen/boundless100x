"""Download annual report PDFs from BSE filings and extract text.

Extraction is section-targeted rather than first-N-pages. The distinction
matters: guidance, capacity plans, and related-party detail live in MD&A and
the statutory reports, which a survey of 20 real Indian annual reports found
starting at pages 20 to 145 — a first-30-pages read reaches the chairman's
letter and almost nothing else.

Detection is heuristic and expected to miss. Indian annual reports have no
reliable structural anchors, so a section is located by heading text, and the
commonest trap is the contents page — it lists every section name in exactly
the format a heading takes. What makes imperfection safe is provenance: every
section is tagged `found` or `fallback`, and consumers of a `fallback` slot
are required to treat it as unknown rather than mine a chairman's letter for
guidance it never contained.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

BSE_ANNUAL_REPORT_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
BSE_PDF_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachHis"

# Heading patterns per section. Written against the wording actually observed
# across 20 fetched reports — "Chairman's Communique" and "From the MD's Desk"
# are as common as "Chairman's Letter".
SECTION_PATTERNS = {
    "mdna": re.compile(
        r"management[’'\s]+discussion\s*(?:&|and)?\s*analysis", re.I
    ),
    "chairman": re.compile(
        r"(chairman|managing\s+director|md)[’'s]*\s*"
        r"(letter|message|statement|communiqu\w*|desk)"
        r"|letter\s+from\s+(our\s+|the\s+)?chairman",
        re.I,
    ),
    "governance": re.compile(
        r"(report\s+on\s+corporate\s+governance|corporate\s+governance\s+report"
        r"|related\s+party\s+transaction|board[’']?s?\s+report"
        r"|directors[’']?\s+report)",
        re.I,
    ),
}

# A page whose title is literally "Contents" or "Index" is never a section.
_CONTENTS_TITLE = re.compile(r"^\s*(table\s+of\s+)?(contents|index)\s*$", re.I)
# Contents pages list page numbers, either prefixed ("14  Board's Report") or
# on their own line between titles. Either shape, in bulk, means a listing.
_NUMBERED_ENTRY = re.compile(r"^\s*\d{1,3}\s*[\t.\s]")
_BARE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_CONTENTS_ENTRY_THRESHOLD = 8

# Numbering and bullets that may precede a heading without making it prose:
# "2. Management Discussion and Analysis", "• Management Discussion…".
_LEADING_ORNAMENT = re.compile(r"^[\s\d.)\]\-–—•*|>]+")

# Pages of body text taken for a section when no later section bounds it.
_MAX_SECTION_PAGES = 15


def _is_heading_like(line: str, match: re.Match) -> bool:
    """Whether a matched line reads as a heading rather than a sentence.

    The original "short line, mostly the section name" test let annual reports
    through on their own cross-references. Measured across the fetched corpus,
    8 of 18 `found` MD&A slices were actually auditor's reports, governance,
    CSR or HR text, every one of them anchored on a line like
    "provided in the Management Discussion and Analysis" or
    "Management Discussion and Analysis of financial statements" — prose that
    merely names the section, at 58–77% of its line, comfortably inside the old
    50% bar.

    Two structural properties separate the two, and both are needed:

    1. **A heading opens its line.** A cross-reference sits inside a sentence,
       so something precedes it ("included in the…", "Moreover, a report on…").
       Leading numbering or bullets are ornament, not prose, and are stripped
       first.
    2. **A heading is not continued in lowercase.** What follows a real heading
       is nothing, punctuation, or a subtitle in caps ("MANAGEMENT DISCUSSION
       AND ANALYSIS ECONOMIC REVIEW"). A lowercase word after the match means
       the sentence is still running ("…and Analysis of financial statements").

    Raising the coverage ratio instead would not work: it rejects the genuine
    "MANAGEMENT DISCUSSION AND ANALYSIS ECONOMIC REVIEW" heading at 69% while
    still admitting "Management Discussion and Analysis describing…" at 77%.
    Coverage does not distinguish these; position and continuation do.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False

    head = _LEADING_ORNAMENT.sub("", stripped)
    matched = match.group(0)
    if not head.lower().startswith(matched.lower()):
        return False

    tail = head[len(matched):].strip(" \t:.,;–—-")
    return not tail or tail[0].isupper()


def _is_contents_page(lines: list[str], hits: dict) -> bool:
    """Whether a page is a contents listing rather than a section start.

    Three independent signals, because contents pages vary: an explicit
    Contents/Index title, a bulk of page-number entries, or several different
    section names on one page — no real section starts twice over.
    """
    if any(_CONTENTS_TITLE.match(line) for line in lines):
        return True
    entries = sum(
        1 for line in lines if _NUMBERED_ENTRY.match(line) or _BARE_NUMBER.match(line)
    )
    if entries >= _CONTENTS_ENTRY_THRESHOLD:
        return True
    return len(hits) >= 2


def annual_reports_dir(raw_data_dir, bse_code) -> Path:
    """Where a scrip code's annual reports live.

    The one statement of the layout `download()` writes. It had been restated
    at five call sites across three layers, none of which imports the fetcher
    that owns it, so renaming the directory would have broken `llm_layer` last
    and silently — every reader swallows a missing directory as "no reports".
    """
    return Path(raw_data_dir) / str(bse_code) / "annual_reports"


def report_year(path) -> str:
    """The report year in a `{year}_annual_report.*` filename."""
    return Path(path).name.split("_")[0]


def load_cached_sections(raw_data_dir, bse_code) -> dict[str, dict]:
    """`{year: {section: {...}}}` from the sidecars already on disk.

    The offline counterpart to `download_and_extract`, which always reaches
    BSE. Anything wanting to read what has already been fetched — the sweep's
    dry run, the corpus audit — needs this and had been open-coding it, so the
    filename convention above leaked into three modules.

    An unreadable sidecar is skipped with a warning rather than raising: this
    is a read of a cache, and one corrupt file must not make a whole corpus
    look empty.
    """
    reports = annual_reports_dir(raw_data_dir, bse_code)
    if not reports.is_dir():
        return {}

    by_year: dict[str, dict] = {}
    for sidecar in sorted(reports.glob("*_annual_report.sections.json")):
        try:
            by_year[report_year(sidecar)] = json.loads(
                sidecar.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Ignoring unreadable sections sidecar {sidecar}: {e}")
    return by_year


def cached_report_years(raw_data_dir, bse_code) -> list[str]:
    """Report years held as PDFs, which is what "held" means.

    Deliberately the PDFs rather than the `.sections.json` sidecars: a year
    downloaded but not yet section-extracted is still a year the corpus
    gained, and counting sidecars would report it as missing.
    """
    reports = annual_reports_dir(raw_data_dir, bse_code)
    if not reports.is_dir():
        return []
    return sorted(report_year(p) for p in reports.glob("*_annual_report.pdf"))


def combined_text(sections: dict[str, dict]) -> str:
    """The single AR string for consumers that predate section extraction.

    When sections were located, this is those sections in page order — the
    substance the LLM should read. When none were, every slot holds the same
    first-N-pages text, so the longest one is returned once: that reproduces
    the pre-section behaviour exactly rather than repeating one extract three
    times over. Callers apply their own character cap; provided every
    per-section cap is at least that cap, the all-fallback string is
    byte-identical to what the pipeline produced before sections existed.
    """
    if not sections:
        return ""

    found = [s for s in sections.values() if s.get("provenance") == "found"]
    if not found:
        return max(
            (s.get("text", "") for s in sections.values()), key=len, default=""
        )

    ordered = sorted(found, key=lambda s: s.get("start_page") or 0)
    return "\n\n".join(s.get("text", "") for s in ordered)


def find_section_starts(pages: list[str]) -> dict[str, tuple[int, int]]:
    """Map section name to where it starts, as `(page_index, line_index)`.

    Scans in document order and keeps the first non-contents page whose
    heading matches. Sections that never match are simply absent.

    The line index matters as much as the page. A heading frequently sits near
    the bottom of its page — one real report has the MD&A heading at line 40 of
    62 — so taking the page from its top prepends the *previous* section's
    tail. That is how a correctly-detected MD&A came back opening on CSR prose.
    Callers slice from the line, not the page.
    """
    starts: dict[str, tuple[int, int]] = {}

    for index, text in enumerate(pages):
        lines = text.splitlines()
        hits: dict[str, tuple[int, int]] = {}
        for name, pattern in SECTION_PATTERNS.items():
            for line_no, line in enumerate(lines):
                match = pattern.search(line)
                if match and _is_heading_like(line, match):
                    hits[name] = (index, line_no)
                    break

        if not hits or _is_contents_page(lines, hits):
            continue

        for name, position in hits.items():
            starts.setdefault(name, position)

    return starts


class AnnualReportDownloader(BaseFetcher):
    """Download annual report PDFs from BSE filings and extract text."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session.headers.update(
            {
                "Referer": "https://www.bseindia.com/",
                "Origin": "https://www.bseindia.com",
            }
        )

    def download(
        self,
        bse_code: str,
        output_dir: str,
        max_reports: int = 3,
    ) -> list[str]:
        """Download the most recent annual report PDFs.

        Args:
            bse_code: BSE scrip code
            output_dir: Directory to save PDFs
            max_reports: Maximum number of reports to download

        Returns list of saved file paths.
        """
        ar_dir = Path(output_dir) / bse_code / "annual_reports"
        ar_dir.mkdir(parents=True, exist_ok=True)

        pdf_urls = self._find_annual_report_urls(bse_code)
        if not pdf_urls:
            logger.warning(f"No annual report URLs found for {bse_code}")
            return []

        saved = []
        for url, year in pdf_urls[:max_reports]:
            filename = f"{year}_annual_report.pdf"
            filepath = ar_dir / filename

            if filepath.exists():
                logger.info(f"Already downloaded: {filename}")
                saved.append(str(filepath))
                continue

            try:
                resp = self._get(url)
                filepath.write_bytes(resp.content)
                logger.info(f"Downloaded: {filename} ({len(resp.content)} bytes)")
                saved.append(str(filepath))
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")

        return saved

    def extract_text(self, pdf_path: str, max_pages: int = 30) -> str:
        """Extract and clean text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum number of pages to extract from.

        Returns cleaned text string, or empty string on failure.
        """
        txt_path = Path(pdf_path).with_suffix(".txt")

        # Cache: if .txt already exists and is non-empty, return it
        if txt_path.exists() and txt_path.stat().st_size > 0:
            logger.info(f"Using cached text: {txt_path.name}")
            return txt_path.read_text(encoding="utf-8")

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages_to_read = min(len(doc), max_pages)
            raw_parts = []

            for page_num in range(pages_to_read):
                page = doc[page_num]
                raw_parts.append(page.get_text())

            doc.close()

            raw_text = "\n".join(raw_parts)
            cleaned = self._clean_extracted_text(raw_text)

            # Save alongside PDF for caching
            txt_path.write_text(cleaned, encoding="utf-8")
            logger.info(
                f"Extracted text from {pages_to_read} pages of "
                f"{Path(pdf_path).name} ({len(cleaned)} chars)"
            )

            return cleaned

        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed — cannot extract PDF text")
            return ""
        except Exception as e:
            logger.warning(f"PDF text extraction failed for {pdf_path}: {e}")
            return ""

    def _page_texts(self, pdf_path: str, scan_pages: int) -> list[str]:
        """Raw per-page text for the scan window, or [] if unreadable.

        Detection runs on raw text, not cleaned text: `_clean_extracted_text`
        strips standalone page numbers, which are one of the signals that
        identifies a contents page.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages = [doc[i].get_text() for i in range(min(len(doc), scan_pages))]
            doc.close()
            return pages
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed — cannot extract PDF text")
            return []
        except Exception as e:
            logger.warning(f"PDF page read failed for {pdf_path}: {e}")
            return []

    def extract_sections(
        self,
        pdf_path: str,
        sections: dict[str, int] | None = None,
        max_pages: int = 30,
        scan_pages: int = 150,
    ) -> dict[str, dict]:
        """Extract the configured sections, tagging each with its provenance.

        Args:
            pdf_path: Path to the PDF.
            sections: {section_name: char_cap}. Names must be keys of
                SECTION_PATTERNS; unknown names are ignored.
            max_pages: Fallback window — the first-N-pages text handed to a
                section that could not be located.
            scan_pages: How deep to search for headings. Generous by default:
                MD&A commonly starts past page 100 in a full annual report.

        Returns {section: {"text", "provenance", "start_page"}}. `provenance`
        is `found` when the section was located and `fallback` when the slot
        holds first-N-pages text instead — never silently interchangeable.

        Results are cached in a `.sections.json` sidecar beside the PDF.
        """
        sections = sections or {}
        sidecar = Path(pdf_path).with_suffix(".sections.json")

        if sidecar.exists() and sidecar.stat().st_size > 0:
            try:
                cached = json.loads(sidecar.read_text(encoding="utf-8"))
                if set(cached) == set(sections):
                    logger.info(f"Using cached sections: {sidecar.name}")
                    return cached
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Ignoring unreadable sections sidecar {sidecar.name}: {e}")

        pages = self._page_texts(pdf_path, scan_pages)
        if not pages:
            return {}

        starts = find_section_starts(pages)
        fallback = self._clean_extracted_text("\n".join(pages[:max_pages]))
        boundaries = sorted(starts.values())

        result: dict[str, dict] = {}
        for name, cap in sections.items():
            start = starts.get(name)
            if start is None:
                result[name] = {
                    "text": fallback[:cap],
                    "provenance": "fallback",
                    "start_page": None,
                }
                continue

            start_page, start_line = start

            # Run to the next section start, or a bounded number of pages.
            later = [b for b in boundaries if b > start]
            end = min(
                later[0][0] if later else len(pages),
                start_page + _MAX_SECTION_PAGES,
            )
            end = max(end, start_page + 1)

            # Begin at the heading line, not the top of its page — everything
            # above it belongs to the preceding section.
            body_pages = pages[start_page:end]
            body_pages[0] = "\n".join(body_pages[0].splitlines()[start_line:])
            body = self._clean_extracted_text("\n".join(body_pages))
            result[name] = {
                "text": body[:cap],
                "provenance": "found",
                "start_page": start_page,
                "start_line": start_line,
            }

        located = [n for n, s in result.items() if s["provenance"] == "found"]
        logger.info(
            f"Sections in {Path(pdf_path).name}: "
            f"{len(located)}/{len(result)} found ({', '.join(sorted(located)) or 'none'})"
        )

        try:
            sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not cache sections for {Path(pdf_path).name}: {e}")

        return result

    def download_and_extract(
        self,
        bse_code: str,
        output_dir: str,
        max_reports: int = 1,
        max_pages: int = 30,
        sections: dict[str, int] | None = None,
        scan_pages: int = 150,
    ) -> dict[str, dict]:
        """Download the retained annual reports and extract sections from each.

        Returns {year: {section: {...}}}, newest year first. Several years are
        kept because the forward-growth module compares what management
        promised in one report against what the next one delivered — a
        single report cannot answer that.
        """
        pdf_paths = self.download(bse_code, output_dir, max_reports=max_reports)
        if not pdf_paths:
            return {}

        by_year: dict[str, dict] = {}
        for path in pdf_paths:
            # Filenames are written as "{year}_annual_report.pdf" by download().
            year = Path(path).stem.split("_")[0]
            extracted = self.extract_sections(
                path,
                sections=sections,
                max_pages=max_pages,
                scan_pages=scan_pages,
            )
            if extracted:
                by_year[year] = extracted

        return by_year

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Clean raw PDF text for LLM consumption.

        - Collapse excessive whitespace
        - Remove standalone page numbers
        - Remove repeated header/footer lines
        """
        # Remove lines that are just a page number (1-3 digits, optionally with whitespace)
        text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

        # Collapse 3+ consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse runs of spaces/tabs (not newlines) to single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove common footer patterns: lines appearing 3+ times (likely headers/footers)
        lines = text.split("\n")
        line_counts = Counter(line.strip() for line in lines if line.strip())
        repeated = {
            line for line, count in line_counts.items()
            if count >= 3 and len(line) < 80
        }
        if repeated:
            lines = [line for line in lines if line.strip() not in repeated]
            text = "\n".join(lines)

        return text.strip()

    def _find_annual_report_urls(self, bse_code: str) -> list[tuple[str, str]]:
        """Find annual report PDF URLs from BSE AnnualReport API.

        Uses the dedicated BSE Annual Report API endpoint which returns
        structured data with year and file_name (UUID).

        Returns list of (url, year) tuples, most recent first.
        """
        try:
            params = {"scripcode": bse_code}
            resp = self._get(BSE_ANNUAL_REPORT_API, params=params)
            data = resp.json()

            results = []
            if isinstance(data, dict) and "Table" in data:
                for entry in data["Table"]:
                    year = entry.get("year", "")
                    file_name = entry.get("file_name", "")

                    if not file_name:
                        continue

                    # Clean file_name: strip leading backslashes, extract UUID
                    file_name = file_name.lstrip("\\")

                    # Construct download URL
                    # API returns filenames like "UUID.pdf.pdf" or "UUID.pdf" or "NNN.pdf"
                    # The actual BSE URL pattern is: /xml-data/corpfiling/AttachHis/{UUID}.pdf
                    if file_name.endswith(".pdf.pdf"):
                        # Strip the double .pdf extension, keep just UUID.pdf
                        clean_name = file_name[:-4]  # remove trailing .pdf
                    elif file_name.endswith(".pdf"):
                        clean_name = file_name
                    else:
                        clean_name = f"{file_name}.pdf"

                    url = f"{BSE_PDF_BASE}/{clean_name}"
                    results.append((url, year))

            # Already sorted by year descending from API, but ensure it
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as e:
            logger.warning(f"BSE annual report API failed for {bse_code}: {e}")
            return []
