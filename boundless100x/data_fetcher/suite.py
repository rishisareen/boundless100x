"""DataFetcherSuite — orchestrates all data fetchers for a company."""

import logging
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.fetch_financials import FinancialsFetcher
from boundless100x.data_fetcher.fetch_price_volume import PriceVolumeFetcher
from boundless100x.data_fetcher.fetch_shareholding import ShareholdingFetcher
from boundless100x.data_fetcher.fetch_sector_peers import SectorPeersFetcher
from boundless100x.data_fetcher.fetch_corporate_actions import CorporateActionsFetcher
from boundless100x.data_fetcher.fetch_analyst_coverage import AnalystCoverageFetcher
from boundless100x.data_fetcher.download_annual_reports import AnnualReportDownloader

logger = logging.getLogger(__name__)


class DataFetcherSuite:
    """Orchestrates all data fetchers to gather complete company data.

    Coordinates fetching from multiple sources and merges into a unified
    data dict that the compute engine can consume.
    """

    def __init__(self, config: dict):
        fetch_config = config.get("fetching", {})
        rate_limit = fetch_config.get("rate_limit_seconds", 2.0)
        cache_ttl = fetch_config.get("cache_ttl_hours", 24)
        retry_count = fetch_config.get("retry_count", 3)

        common_kwargs = {
            "rate_limit_seconds": rate_limit,
            "cache_ttl_hours": cache_ttl,
            "retry_count": retry_count,
        }

        self.financials = FinancialsFetcher(**common_kwargs)
        self.price_volume = PriceVolumeFetcher(**common_kwargs)
        self.shareholding_bse = ShareholdingFetcher(**common_kwargs)
        self.sector_peers = SectorPeersFetcher(**common_kwargs)
        self.corporate_actions = CorporateActionsFetcher(**common_kwargs)
        self.analyst_coverage = AnalystCoverageFetcher(**common_kwargs)
        self.annual_reports = AnnualReportDownloader(**common_kwargs)

        self.analysis_years = config.get("analysis_period", {}).get("financials_years", 10)
        self.price_years = config.get("analysis_period", {}).get("price_history_years", 10)
        self.sh_quarters = config.get("analysis_period", {}).get("shareholding_quarters", 20)

        # Annual report config
        ar_config = config.get("annual_reports", {})
        self.ar_enabled = ar_config.get("enabled", True)
        self.ar_max_reports = ar_config.get("max_reports", 1)
        self.ar_max_pages = ar_config.get("max_pages", 30)
        self.ar_max_text_chars = ar_config.get("max_text_chars", 5000)

        self.raw_data_dir = str(
            Path(__file__).parent / "raw_data"
        )

    def fetch_all(self, ticker: str, bse_code: str | None = None) -> dict:
        """Fetch all available data for a company.

        Args:
            ticker: NSE symbol (e.g., "ASTRAL")
            bse_code: BSE scrip code (optional, for BSE-specific data)

        Returns dict with keys matching compute engine expected inputs:
            financials, balance_sheet, cashflow, ratios, shareholding,
            price, analyst_coverage, metadata, sector_peers
        """
        logger.info(f"Fetching all data for {ticker}")
        data = {"annual_report_text": None}

        # 1. Screener.in financials (P&L, BS, CF, Ratios, Shareholding)
        try:
            screener_data = self.financials._do_fetch_with_save(
                ticker, self.raw_data_dir
            )
            data["financials"] = screener_data.get("financials", pd.DataFrame())
            data["balance_sheet"] = screener_data.get("balance_sheet", pd.DataFrame())
            data["cashflow"] = screener_data.get("cashflow", pd.DataFrame())
            data["ratios"] = screener_data.get("ratios", pd.DataFrame())
            data["shareholding"] = screener_data.get("shareholding", pd.DataFrame())
            data["metadata"] = screener_data.get("metadata", {})
            data["growth_summary"] = screener_data.get("growth_summary", {})
        except Exception as e:
            logger.error(f"Screener.in fetch failed for {ticker}: {e}")
            data["financials"] = pd.DataFrame()
            data["balance_sheet"] = pd.DataFrame()
            data["cashflow"] = pd.DataFrame()
            data["ratios"] = pd.DataFrame()
            data["shareholding"] = pd.DataFrame()
            data["metadata"] = {}

        # 2. Price & volume
        logger.info(f"Fetching price & volume for {ticker} (may take a moment)...")
        try:
            data["price"] = self.price_volume.fetch(
                ticker, years=self.price_years, output_dir=self.raw_data_dir
            )
        except Exception as e:
            logger.error(f"Price fetch failed for {ticker}: {e}")
            data["price"] = pd.DataFrame()

        # 3. Analyst coverage
        logger.info(f"Fetching analyst coverage for {ticker}...")
        try:
            data["analyst_coverage"] = self.analyst_coverage.fetch(
                ticker, output_dir=self.raw_data_dir
            )
        except Exception as e:
            logger.error(f"Analyst coverage fetch failed for {ticker}: {e}")
            data["analyst_coverage"] = {}

        # 4. BSE-specific data (if BSE code provided or auto-detected)
        resolved_bse = bse_code or data.get("metadata", {}).get("bse_code")
        if resolved_bse:
            self.fetch_bse_data(data, ticker, resolved_bse)

        # 5. Sector peers
        logger.info(f"Fetching sector peers for {ticker}...")
        warehouse_id = data.get("metadata", {}).get("warehouse_id")
        try:
            data["sector_peers"] = self.sector_peers.fetch(
                ticker, warehouse_id=warehouse_id, output_dir=self.raw_data_dir
            )
        except Exception as e:
            logger.warning(f"Sector peers fetch failed for {ticker}: {e}")
            data["sector_peers"] = pd.DataFrame()

        logger.info(f"Data fetch complete for {ticker}")
        return data

    def fetch_bse_data(self, data: dict, ticker: str, bse_code: str) -> None:
        """Fetch BSE-specific data: shareholding, corporate actions, annual report.

        Mutates the data dict in-place.
        """
        logger.info(f"Fetching BSE data for {ticker} (code={bse_code})...")

        try:
            bse_sh = self.shareholding_bse.fetch(
                bse_code, quarters=self.sh_quarters, output_dir=self.raw_data_dir
            )
            if not bse_sh.empty and (
                data.get("shareholding") is None
                or (hasattr(data.get("shareholding"), "empty") and data["shareholding"].empty)
                or len(bse_sh) > len(data.get("shareholding", []))
            ):
                data["shareholding_bse"] = bse_sh
        except Exception as e:
            logger.warning(f"BSE shareholding fetch failed: {e}")

        try:
            data["corporate_actions"] = self.corporate_actions.fetch(
                bse_code, output_dir=self.raw_data_dir
            )
        except Exception as e:
            logger.warning(f"Corporate actions fetch failed: {e}")
            data["corporate_actions"] = pd.DataFrame()

        # Annual report download + text extraction
        if self.ar_enabled:
            logger.info(f"Fetching annual report for {ticker} (BSE={bse_code})...")
            try:
                ar_text = self.annual_reports.download_and_extract(
                    bse_code=bse_code,
                    output_dir=self.raw_data_dir,
                    max_reports=self.ar_max_reports,
                    max_pages=self.ar_max_pages,
                )
                if ar_text:
                    data["annual_report_text"] = ar_text[:self.ar_max_text_chars]
                    logger.info(
                        f"Annual report text: {len(data['annual_report_text'])} chars"
                    )
                else:
                    logger.info(f"No annual report text extracted for {ticker}")
            except Exception as e:
                logger.warning(f"Annual report fetch failed for {ticker}: {e}")
