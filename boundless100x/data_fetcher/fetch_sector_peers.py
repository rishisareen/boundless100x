"""Fetch peer comparison data from Screener.in (JS-loaded via API)."""

import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

SCREENER_BASE = "https://www.screener.in"


class SectorPeersFetcher(BaseFetcher):
    """Fetch the peer comparison table from Screener.in.

    The peers table is loaded via AJAX, not in the initial HTML.
    We first get the warehouse_id from the company page, then hit the peers API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch(
        self,
        ticker: str,
        warehouse_id: str | None = None,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Fetch peer comparison for a ticker.

        Args:
            ticker: NSE symbol (e.g., "ASTRAL")
            warehouse_id: If known, skip the initial page fetch
            output_dir: Optional directory to save CSV

        Returns DataFrame with columns:
            name, ticker_url, cmp, pe, market_cap, div_yield,
            np_qtr, qtr_profit_var, sales_qtr, qtr_sales_var, roce
        """
        cache_key = f"sector_peers_{ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache hit: {cache_key}")
            return cached

        # Step 1: Get warehouse_id if not provided
        if warehouse_id is None:
            warehouse_id = self._get_warehouse_id(ticker)
            if warehouse_id is None:
                logger.error(f"Could not get warehouse_id for {ticker}")
                return pd.DataFrame()

        # Step 2: Fetch peers API
        peers_url = f"{SCREENER_BASE}/api/company/{warehouse_id}/peers/"
        try:
            resp = self._get(
                peers_url,
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        except Exception as e:
            logger.error(f"Failed to fetch peers for {ticker}: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(resp.text, "html.parser")
        df = self._parse_peers_table(soup)

        if df.empty:
            logger.warning(f"No peer data found for {ticker}")
            return df

        self.cache.set(cache_key, df)

        if output_dir:
            path = Path(output_dir) / ticker
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "sector_peers.csv", index=False)
            logger.info(f"Saved {ticker}/sector_peers.csv ({len(df)} rows)")

        return df

    def _get_warehouse_id(self, ticker: str) -> str | None:
        """Get warehouse_id from the company page."""
        url = f"{SCREENER_BASE}/company/{ticker}/consolidated/"
        try:
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            info = soup.find(id="company-info")
            if info:
                return info.get("data-warehouse-id")
        except Exception as e:
            logger.error(f"Failed to get warehouse_id for {ticker}: {e}")
        return None

    def _parse_peers_table(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Parse the peers API HTML response into a DataFrame."""
        table = soup.find("table", class_="data-table")
        if table is None:
            return pd.DataFrame()

        rows = table.find_all("tr")
        if not rows:
            return pd.DataFrame()

        # First row is headers (uses <th>)
        header_row = rows[0]
        raw_headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

        # Map header text to normalized column names
        header_map = {
            "S.No.": "sno",
            "Name": "name",
            "CMPRs.": "cmp",
            "P/E": "pe",
            "Mar CapRs.Cr.": "market_cap",
            "Div Yld%": "div_yield",
            "NP QtrRs.Cr.": "np_qtr",
            "Qtr Profit Var%": "qtr_profit_var",
            "Sales QtrRs.Cr.": "sales_qtr",
            "Qtr Sales Var%": "qtr_sales_var",
            "ROCE%": "roce",
        }

        # Build normalized header list
        norm_headers = []
        for h in raw_headers:
            # Strip any embedded unit spans
            clean = h.replace("Rs.", "").replace("Rs.Cr.", "").replace("%", "").strip()
            # Try exact match first, then look through map
            if h in header_map:
                norm_headers.append(header_map[h])
            else:
                # Fuzzy match: find map entry contained in header
                matched = False
                for key, val in header_map.items():
                    if key.replace(" ", "") in h.replace(" ", ""):
                        norm_headers.append(val)
                        matched = True
                        break
                if not matched:
                    norm_headers.append(h.lower().replace(" ", "_"))

        # Parse data rows
        records = []
        for row in rows[1:]:
            tds = row.find_all("td")
            if not tds:
                continue

            # Skip median row
            company_id = row.get("data-row-company-id")
            if company_id is None:
                # Could be median row
                name_text = tds[1].get_text(strip=True) if len(tds) > 1 else ""
                if "Median" in name_text:
                    continue

            record = {}
            for i, td in enumerate(tds):
                if i >= len(norm_headers):
                    break
                col = norm_headers[i]

                if col == "name":
                    link = td.find("a")
                    record["name"] = link.get_text(strip=True) if link else td.get_text(strip=True)
                    if link:
                        href = link.get("href", "")
                        # Extract ticker from URL like /company/SUPREMEIND/consolidated/
                        parts = [p for p in href.split("/") if p and p != "company" and p != "consolidated"]
                        record["peer_ticker"] = parts[0] if parts else ""
                elif col == "sno":
                    continue
                else:
                    text = td.get_text(strip=True).replace(",", "")
                    try:
                        record[col] = float(text) if text and text != "-" else None
                    except ValueError:
                        record[col] = None

            if record.get("name"):
                records.append(record)

        return pd.DataFrame(records) if records else pd.DataFrame()
