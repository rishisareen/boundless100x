"""Fetch quarterly shareholding patterns from BSE India."""

import logging
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

BSE_SHP_URL = "https://api.bseindia.com/BseIndiaAPI/api/CorporateShareholding/w"


class ShareholdingFetcher(BaseFetcher):
    """Fetch quarterly shareholding patterns from BSE India.

    Note: Screener.in also has shareholding data on the company page,
    which the FinancialsFetcher already parses. This fetcher provides
    BSE-sourced data as an alternative/supplement with more granular detail.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session.headers.update(
            {
                "Referer": "https://www.bseindia.com/",
                "Origin": "https://www.bseindia.com",
            }
        )

    def fetch(
        self,
        bse_code: str,
        quarters: int = 20,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Fetch shareholding pattern for the last N quarters.

        Args:
            bse_code: BSE scrip code (e.g., "532830" for Astral)
            quarters: Number of historical quarters to attempt
            output_dir: Optional directory to save CSV

        Returns DataFrame with columns:
            quarter, promoter_pct, promoter_pledge_pct, fii_pct, dii_pct,
            public_pct, govt_pct, total_shares
        """
        cache_key = f"shareholding_bse_{bse_code}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache hit: {cache_key}")
            return cached

        records = []
        try:
            params = {
                "scripcode": bse_code,
                "qtrid": "",
            }
            resp = self._get(BSE_SHP_URL, params=params)
            data = resp.json()

            if isinstance(data, list):
                for entry in data:
                    record = self._parse_bse_entry(entry)
                    if record:
                        records.append(record)
        except Exception as e:
            logger.warning(f"BSE shareholding fetch failed for {bse_code}: {e}")

        if not records:
            logger.warning(f"No shareholding data from BSE for {bse_code}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("quarter").reset_index(drop=True)

        # Limit to requested quarters
        if len(df) > quarters:
            df = df.tail(quarters).reset_index(drop=True)

        self.cache.set(cache_key, df)

        if output_dir:
            path = Path(output_dir) / bse_code
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "shareholding.csv", index=False)
            logger.info(f"Saved {bse_code}/shareholding.csv ({len(df)} rows)")

        return df

    def _parse_bse_entry(self, entry: dict) -> dict | None:
        """Parse a single BSE shareholding JSON entry."""
        try:
            return {
                "quarter": entry.get("SHPDate", ""),
                "promoter_pct": self._safe_float(entry.get("PromotersPer")),
                "promoter_pledge_pct": self._safe_float(entry.get("PromoterPledge")),
                "fii_pct": self._safe_float(entry.get("FIIPer")),
                "dii_pct": self._safe_float(entry.get("DIIPer")),
                "public_pct": self._safe_float(entry.get("PublicPer")),
                "govt_pct": self._safe_float(entry.get("GovtPer")),
            }
        except Exception:
            return None

    @staticmethod
    def _safe_float(val) -> float | None:
        if val is None:
            return None
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return None
