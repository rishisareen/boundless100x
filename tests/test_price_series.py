"""Price series basis: raw close vs adjusted close must not be conflated.

Valuation (historical P/E band) needs the raw traded close against as-reported
EPS; return measurement needs the split/dividend-adjusted series. yfinance's
default auto_adjust=True silently handed every consumer the adjusted close.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.backtest import WalkForwardBacktest
from boundless100x.data_fetcher.fetch_price_volume import PriceVolumeFetcher


def yfinance_frame() -> pd.DataFrame:
    """What yf.download(auto_adjust=False) returns for a stock that split 1:2."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "Open": [100.0, 102.0, 51.5],
        "High": [102.0, 103.0, 52.0],
        "Low": [99.0, 100.5, 50.5],
        "Close": [101.0, 102.0, 51.0],          # raw: halves at the split
        "Adj Close": [50.5, 51.0, 51.0],        # adjusted: continuous
        "Volume": [1000, 1100, 2200],
    })


class TestNormalize:
    def test_raw_and_adjusted_close_are_kept_separate(self):
        df = PriceVolumeFetcher()._normalize(yfinance_frame())

        assert list(df["close"]) == [101.0, 102.0, 51.0]
        assert list(df["adj_close"]) == [50.5, 51.0, 51.0]

    def test_multiindex_columns_are_flattened(self):
        raw = yfinance_frame()
        raw.columns = pd.MultiIndex.from_tuples(
            [(c, "TEST.NS") for c in raw.columns]
        )

        df = PriceVolumeFetcher()._normalize(raw)

        assert "close" in df.columns and "adj_close" in df.columns

    def test_source_without_adjusted_close_gets_adj_equal_close(self):
        raw = yfinance_frame().drop(columns=["Adj Close"])

        df = PriceVolumeFetcher()._normalize(raw)

        assert list(df["adj_close"]) == list(df["close"])

    def test_aliased_adjusted_close_is_marked_estimated(self):
        """jugaad-data has no adjusted series — the alias must be labeled so a
        consumer computing returns doesn't mistake it for a real one."""
        raw = yfinance_frame().drop(columns=["Adj Close"])

        df = PriceVolumeFetcher()._normalize(raw)

        assert df["adj_close_is_estimated"].all()

    def test_genuine_adjusted_close_is_not_marked_estimated(self):
        df = PriceVolumeFetcher()._normalize(yfinance_frame())

        assert not df["adj_close_is_estimated"].any()


class TestRealizedReturnBasis:
    def test_adjusted_close_is_preferred_when_present(self):
        """A 1:2 split must not read as a 50% loss."""
        price = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2024-01-02", "2025-01-02"]),
            "close": [100.0, 51.0, 52.0],       # raw: split in year one
            "adj_close": [50.0, 51.0, 52.0],    # adjusted: steady +2%/yr-ish
        })

        ret, span = WalkForwardBacktest._realized_return(
            price, pd.Timestamp("2023-06-30")
        )

        # Window: last close ≤ truncation (2023-01-02, adj 50) → final close
        # (2025-01-02, adj 52), 731 days apart.
        years = 731 / 365.25
        assert span["price_series"] == "adj_close"
        assert ret == pytest.approx(((52.0 / 50.0) ** (1 / years) - 1) * 100, rel=1e-6)
        # The raw close would have read the split as a deep loss instead.
        raw_ret = ((52.0 / 100.0) ** (1 / years) - 1) * 100
        assert abs(ret - raw_ret) > 20

    def test_legacy_files_without_adj_close_still_work(self):
        price = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2025-01-02"]),
            "close": [100.0, 121.0],
        })

        ret, span = WalkForwardBacktest._realized_return(
            price, pd.Timestamp("2023-06-30")
        )

        assert span["price_series"] == "close"
        assert ret == pytest.approx(10.0, abs=0.2)

    def test_estimated_adjusted_close_is_excluded_not_trusted(self):
        """A jugaad-data fetch aliases adj_close = close for a split-affected
        stock. Scoring a return on it would read the split as a crash — the
        fix is to refuse, not to compute a wrong number confidently."""
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2024-01-02", "2025-01-02"]),
            "close": [100.0, 51.0, 52.0],
        })
        price = PriceVolumeFetcher()._normalize(raw)
        assert price["adj_close_is_estimated"].all()

        ret, span = WalkForwardBacktest._realized_return(
            price, pd.Timestamp("2023-06-30")
        )

        assert ret is None
        assert "estimated" in span["reason"] or "fallback" in span["reason"]

    def test_genuine_adjusted_close_from_normalize_is_still_used(self):
        raw = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-02", "2025-01-02"]),
            "Open": [100.0, 52.5], "High": [102.0, 53.0], "Low": [99.0, 51.5],
            "Close": [100.0, 52.0], "Adj Close": [50.0, 52.0],
            "Volume": [1000, 1200],
        })
        price = PriceVolumeFetcher()._normalize(raw)

        ret, span = WalkForwardBacktest._realized_return(
            price, pd.Timestamp("2023-06-30")
        )

        assert span["price_series"] == "adj_close"
        assert ret is not None
