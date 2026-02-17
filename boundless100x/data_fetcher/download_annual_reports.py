"""Download annual report PDFs from BSE filings and extract text."""

import logging
import re
from collections import Counter
from pathlib import Path

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

BSE_ANNUAL_REPORT_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
BSE_PDF_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachHis"


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

    def download_and_extract(
        self,
        bse_code: str,
        output_dir: str,
        max_reports: int = 1,
        max_pages: int = 30,
    ) -> str:
        """Download the most recent annual report and extract text.

        Args:
            bse_code: BSE scrip code.
            output_dir: Base directory for raw data.
            max_reports: Number of reports to download.
            max_pages: Pages to extract per PDF.

        Returns extracted text from the most recent report, or empty string.
        """
        pdf_paths = self.download(bse_code, output_dir, max_reports=max_reports)

        if not pdf_paths:
            return ""

        # Extract from the most recent (first in list — download returns newest first)
        return self.extract_text(pdf_paths[0], max_pages=max_pages)

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
