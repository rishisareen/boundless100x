"""Fetch corporate actions (splits, bonuses, dividends) from BSE India."""

import logging
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

BSE_CORP_ACTIONS_URL = "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w"


class CorporateActionsFetcher(BaseFetcher):
    """Fetch corporate actions from BSE India API."""

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
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Fetch corporate actions for a BSE scrip code.

        Returns DataFrame with columns: date, type, details, ex_date
        """
        cache_key = f"corporate_actions_{bse_code}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache hit: {cache_key}")
            return cached

        records = []
        for purpose in ["Bonus", "Split", "Dividend"]:
            try:
                params = {
                    "scripcode": bse_code,
                    "purpose": purpose,
                }
                resp = self._get(BSE_CORP_ACTIONS_URL, params=params)
                data = resp.json()

                if isinstance(data, list):
                    for entry in data:
                        record = {
                            "date": entry.get("ANNOUNCEMENT_DT", ""),
                            "ex_date": entry.get("EX_DT", ""),
                            "type": purpose.lower(),
                            "details": entry.get("PURPOSE", ""),
                        }
                        records.append(record)
            except Exception as e:
                logger.warning(f"BSE corporate actions ({purpose}) failed for {bse_code}: {e}")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)

        self.cache.set(cache_key, df)

        if output_dir:
            path = Path(output_dir) / bse_code
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "corporate_actions.csv", index=False)
            logger.info(f"Saved {bse_code}/corporate_actions.csv ({len(df)} rows)")

        return df
