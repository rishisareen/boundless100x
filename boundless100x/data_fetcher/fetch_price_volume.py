"""Fetch historical price and volume data via yfinance (primary) with jugaad-data fallback."""

import logging
import signal
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.base import BaseFetcher

# Timeout for jugaad-data calls (seconds) — NSE often blocks/hangs
JUGAAD_TIMEOUT = 15

logger = logging.getLogger(__name__)


class PriceVolumeFetcher(BaseFetcher):
    """Fetch daily OHLCV data from Yahoo Finance (primary) with jugaad-data fallback."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch(
        self,
        ticker: str,
        years: int = 10,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for a ticker.

        Tries yfinance first (reliable for Indian stocks), falls back to jugaad-data.
        Returns DataFrame with columns: date, open, high, low, close, volume.
        """
        cache_key = f"price_volume_{ticker}_{years}yr"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache hit: {cache_key}")
            return cached

        end_date = date.today()
        start_date = end_date - timedelta(days=years * 365)

        # Try yfinance first (reliable, fast)
        df = self._fetch_yfinance(ticker, start_date, end_date)
        if df is None or df.empty:
            logger.info(f"yfinance failed for {ticker}, trying jugaad-data (may be slow)")
            df = self._fetch_jugaad(ticker, start_date, end_date)

        if df is None or df.empty:
            logger.error(f"Failed to fetch price data for {ticker} from all sources")
            return pd.DataFrame()

        # Normalize columns
        df = self._normalize(df)

        self.cache.set(cache_key, df)

        if output_dir:
            path = Path(output_dir) / ticker
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "price_volume.csv", index=False)
            logger.info(f"Saved {ticker}/price_volume.csv ({len(df)} rows)")

        return df

    def _fetch_yfinance(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame | None:
        """Primary: fetch from Yahoo Finance (append .NS for NSE tickers)."""
        try:
            import yfinance as yf

            yf_ticker = f"{ticker}.NS"
            logger.debug(f"yfinance: downloading {yf_ticker}")
            df = yf.download(
                yf_ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
            )
            if df.empty:
                logger.warning(f"yfinance returned empty data for {yf_ticker}")
                return None
            df = df.reset_index()
            logger.info(f"yfinance: got {len(df)} rows for {ticker}")
            return df
        except ImportError:
            logger.warning("yfinance not installed, skipping")
            return None
        except Exception as e:
            logger.warning(f"yfinance error for {ticker}: {e}")
            return None

    def _fetch_jugaad(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame | None:
        """Fallback: fetch from NSE via jugaad-data with timeout protection."""
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"jugaad-data timed out after {JUGAAD_TIMEOUT}s for {ticker}")

        try:
            from jugaad_data.nse import stock_df

            # Set alarm-based timeout (Unix only) — NSE frequently hangs
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(JUGAAD_TIMEOUT)
            try:
                df = stock_df(
                    symbol=ticker,
                    from_date=start,
                    to_date=end,
                    series="EQ",
                )
            finally:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)
            return df
        except TimeoutError as e:
            logger.warning(str(e))
            return None
        except ImportError:
            logger.warning("jugaad-data not installed, skipping")
            return None
        except Exception as e:
            logger.warning(f"jugaad-data error for {ticker}: {e}")
            return None

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        # Handle yfinance multi-level columns (e.g., ('Close', 'ASTRAL.NS'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        col_map = {}
        for col in df.columns:
            lower = str(col).lower().strip()
            if lower in ("date", "ch_timestamp"):
                col_map[col] = "date"
            elif lower in ("open", "ch_opening_price"):
                col_map[col] = "open"
            elif lower in ("high", "ch_trade_high_price"):
                col_map[col] = "high"
            elif lower in ("low", "ch_trade_low_price"):
                col_map[col] = "low"
            elif lower in ("close", "adj close", "ch_closing_price"):
                col_map[col] = "close"
            elif lower in ("volume", "ch_tot_traded_qty"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)

        # Keep only standard columns
        standard = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in standard if c in df.columns]
        df = df[available].copy()

        # Sort by date ascending
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df
