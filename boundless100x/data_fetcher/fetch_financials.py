"""Fetch 10-year financial data from Screener.in (P&L, Balance Sheet, Cash Flow, Ratios)."""

import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

SCREENER_BASE = "https://www.screener.in"

# Map Screener.in row labels to normalized column names
PL_LABEL_MAP = {
    "Sales": "revenue",
    "Revenue": "revenue",
    "Expenses": "expenses",
    "Operating Profit": "operating_profit",
    "OPM %": "opm_pct",
    "Other Income": "other_income",
    "Interest": "interest",
    "Depreciation": "depreciation",
    "Profit before tax": "pbt",
    "Tax %": "tax_pct",
    "Net Profit": "pat",
    "EPS in Rs": "eps",
    "Dividend Payout %": "dividend_payout_pct",
}

BS_LABEL_MAP = {
    "Equity Capital": "equity_capital",
    "Reserves": "reserves",
    "Borrowings": "borrowings",
    "Other Liabilities": "other_liabilities",
    "Total Liabilities": "total_liabilities",
    "Fixed Assets": "fixed_assets",
    "CWIP": "cwip",
    "Investments": "investments",
    "Other Assets": "other_assets",
    "Total Assets": "total_assets",
}

CF_LABEL_MAP = {
    "Cash from Operating Activity": "cfo",
    "Cash from Investing Activity": "cfi",
    "Cash from Financing Activity": "cff",
    "Net Cash Flow": "net_cash_flow",
}

RATIOS_LABEL_MAP = {
    "Debtor Days": "debtor_days",
    "Inventory Days": "inventory_days",
    "Days Payable": "days_payable",
    "Cash Conversion Cycle": "cash_conversion_cycle",
    "Working Capital Days": "working_capital_days",
    "ROCE %": "roce",
}


def _clean_label(td: Tag) -> str:
    """Extract clean label text from a table cell, stripping expand buttons."""
    button = td.find("button")
    if button:
        text = button.get_text(strip=True)
        return text.rstrip("+").strip()
    return td.get_text(strip=True)


def _parse_value(text: str) -> float | None:
    """Parse a numeric value from Screener.in cell text."""
    text = text.strip().replace(",", "")
    if text in ("", "-", "—"):
        return None
    # Handle percentage values like "22%"
    text = text.rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_table(
    soup: BeautifulSoup,
    section_id: str,
    label_map: dict[str, str],
) -> pd.DataFrame:
    """Parse a Screener.in financial data table into a DataFrame.

    Returns a DataFrame with 'year' column and one column per mapped label.
    """
    section = soup.find(id=section_id)
    if section is None:
        logger.warning(f"Section #{section_id} not found on page")
        return pd.DataFrame()

    table = section.find("table", class_="data-table")
    if table is None:
        logger.warning(f"No data-table found in #{section_id}")
        return pd.DataFrame()

    # Extract year headers
    thead = table.find("thead")
    if thead is None:
        logger.warning(f"No thead in #{section_id}")
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    # First header is empty (label column), rest are dates like "Mar 2014" or "TTM"
    year_headers = headers[1:]

    # Parse rows
    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame()

    row_data: dict[str, list[float | None]] = {}

    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue

        raw_label = _clean_label(cells[0])
        col_name = label_map.get(raw_label)
        if col_name is None:
            continue

        values = [_parse_value(td.get_text(strip=True)) for td in cells[1:]]
        row_data[col_name] = values

    if not row_data:
        return pd.DataFrame()

    # Build DataFrame: years as rows, metrics as columns
    df = pd.DataFrame(row_data)
    df.insert(0, "year", year_headers[: len(df)])

    return df


def _parse_ranges_tables(soup: BeautifulSoup, section_id: str) -> dict:
    """Parse the CAGR/growth summary tables below a financial section."""
    section = soup.find(id=section_id)
    if section is None:
        return {}

    result = {}
    for rt in section.find_all("table", class_="ranges-table"):
        th = rt.find("th")
        if th is None:
            continue
        title = th.get_text(strip=True)
        data = {}
        for tr in rt.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) >= 2:
                period = tds[0].get_text(strip=True).rstrip(":")
                value = tds[1].get_text(strip=True)
                data[period] = value
        result[title] = data

    return result


class FinancialsFetcher(BaseFetcher):
    """Fetch 10-year financial data from Screener.in."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_company_page(self, ticker: str) -> BeautifulSoup:
        """Fetch and parse the consolidated company page."""
        url = f"{SCREENER_BASE}/company/{ticker}/consolidated/"
        resp = self._get(url)
        return BeautifulSoup(resp.text, "html.parser")

    def _get_company_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract company metadata from the page."""
        info = soup.find(id="company-info")
        meta = {}
        if info:
            meta["company_id"] = info.get("data-company-id")
            meta["warehouse_id"] = info.get("data-warehouse-id")
            meta["consolidated"] = info.get("data-consolidated") == "true"

        # Extract top ratios
        ratios_ul = soup.find(id="top-ratios")
        if ratios_ul:
            for li in ratios_ul.find_all("li"):
                name_el = li.find("span", class_="name")
                value_els = li.find_all("span", class_="number")
                if name_el and value_els:
                    name = name_el.get_text(strip=True)
                    if name == "High / Low":
                        meta["52w_high"] = _parse_value(value_els[0].get_text(strip=True))
                        if len(value_els) > 1:
                            meta["52w_low"] = _parse_value(value_els[1].get_text(strip=True))
                    else:
                        meta[name] = _parse_value(value_els[0].get_text(strip=True))

        # Company name from page title
        title_el = soup.find("h1")
        if title_el:
            meta["name"] = title_el.get_text(strip=True)

        # Sector from company-info area
        sub_el = soup.find("p", class_="sub")
        if sub_el:
            sector_link = sub_el.find("a")
            if sector_link:
                meta["sector"] = sector_link.get_text(strip=True)

        # BSE code from bseindia.com link on the page
        # Pattern: https://www.bseindia.com/stock-share-price/.../TICKER/532830/
        import re
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "bseindia.com/stock-share-price" in href:
                bse_match = re.search(r"/(\d{5,6})/?$", href)
                if bse_match:
                    meta["bse_code"] = bse_match.group(1)
                    break

        return meta

    def fetch_all(self, ticker: str, output_dir: str | None = None) -> dict:
        """Fetch all financial data for a ticker.

        Returns dict with keys: financials, balance_sheet, cashflow, ratios,
        metadata, growth_summary.

        Also saves to CSV files in output_dir if provided.
        """

        def _do_fetch():
            soup = self._get_company_page(ticker)
            return self._parse_all(soup, ticker, output_dir)

        cache_key = f"financials_{ticker}"
        # We cache the metadata dict; individual DataFrames are saved to disk
        # For cache, we just track that fetch was done
        result = self._do_fetch_with_save(ticker, output_dir)
        return result

    def _do_fetch_with_save(self, ticker: str, output_dir: str | None) -> dict:
        """Internal: fetch, parse, and optionally save all data."""
        soup = self._get_company_page(ticker)
        return self._parse_all(soup, ticker, output_dir)

    def _parse_all(
        self, soup: BeautifulSoup, ticker: str, output_dir: str | None
    ) -> dict:
        """Parse all financial sections from a Screener.in page."""
        metadata = self._get_company_metadata(soup)

        financials = _parse_table(soup, "profit-loss", PL_LABEL_MAP)
        balance_sheet = _parse_table(soup, "balance-sheet", BS_LABEL_MAP)
        cashflow = _parse_table(soup, "cash-flow", CF_LABEL_MAP)
        ratios = _parse_table(soup, "ratios", RATIOS_LABEL_MAP)
        shareholding = self._parse_shareholding_table(soup)
        growth_summary = _parse_ranges_tables(soup, "profit-loss")

        result = {
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow,
            "ratios": ratios,
            "shareholding": shareholding,
            "metadata": metadata,
            "growth_summary": growth_summary,
        }

        if output_dir:
            self._save_to_disk(result, ticker, output_dir)

        return result

    def _parse_shareholding_table(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Parse the shareholding pattern table."""
        section = soup.find(id="shareholding")
        if section is None:
            return pd.DataFrame()

        tables = section.find_all("table", class_="data-table")
        if not tables:
            return pd.DataFrame()

        # Use the first (quarterly) table
        table = tables[0]
        thead = table.find("thead")
        if thead is None:
            return pd.DataFrame()

        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        quarter_headers = headers[1:]

        sh_label_map = {
            "Promoters": "promoter_pct",
            "FIIs": "fii_pct",
            "DIIs": "dii_pct",
            "Government": "govt_pct",
            "Public": "public_pct",
            "No. of Shareholders": "num_shareholders",
        }

        row_data: dict[str, list[float | None]] = {}
        tbody = table.find("tbody")
        if tbody is None:
            return pd.DataFrame()

        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            raw_label = _clean_label(cells[0])
            col_name = sh_label_map.get(raw_label)
            if col_name is None:
                continue
            values = []
            for td in cells[1:]:
                text = td.get_text(strip=True).rstrip("%").replace(",", "")
                values.append(_parse_value(text))
            row_data[col_name] = values

        if not row_data:
            return pd.DataFrame()

        df = pd.DataFrame(row_data)
        df.insert(0, "quarter", quarter_headers[: len(df)])
        return df

    def _save_to_disk(
        self, result: dict, ticker: str, output_dir: str
    ) -> None:
        """Save parsed data to standardized CSV/JSON files."""
        base = Path(output_dir) / ticker
        base.mkdir(parents=True, exist_ok=True)

        for key in ("financials", "balance_sheet", "cashflow", "ratios", "shareholding"):
            df = result.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(base / f"{key}.csv", index=False)
                logger.info(f"Saved {ticker}/{key}.csv ({len(df)} rows)")

        # Save metadata and growth summary as JSON
        import json

        for key in ("metadata", "growth_summary"):
            data = result.get(key)
            if data:
                with open(base / f"{key}.json", "w") as f:
                    json.dump(data, f, indent=2, default=str)

    def get_warehouse_id(self, ticker: str) -> str | None:
        """Get the warehouse ID for peer API calls."""
        soup = self._get_company_page(ticker)
        info = soup.find(id="company-info")
        if info:
            return info.get("data-warehouse-id")
        return None
