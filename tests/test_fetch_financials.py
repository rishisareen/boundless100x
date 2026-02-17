"""Smoke test: fetch financials for ASTRAL from Screener.in and validate structure."""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from boundless100x.data_fetcher.fetch_financials import FinancialsFetcher


def test_fetch_astral():
    """Fetch Astral Ltd financials and verify data structure."""
    fetcher = FinancialsFetcher(cache_ttl_hours=1, rate_limit_seconds=2)

    output_dir = str(ROOT / "boundless100x" / "data_fetcher" / "raw_data")
    result = fetcher._do_fetch_with_save("ASTRAL", output_dir)

    # Check all expected keys are present
    expected_keys = [
        "financials", "balance_sheet", "cashflow", "ratios",
        "shareholding", "metadata", "growth_summary",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    # Financials (P&L) checks
    fin = result["financials"]
    print(f"\n=== Financials (P&L) ===")
    print(f"Shape: {fin.shape}")
    print(f"Columns: {list(fin.columns)}")
    if not fin.empty:
        print(f"Years: {list(fin['year'])}")
        print(f"Sample row:\n{fin.iloc[-1]}")

    assert not fin.empty, "Financials DataFrame is empty"
    assert "year" in fin.columns
    assert "revenue" in fin.columns
    assert "pat" in fin.columns
    assert "eps" in fin.columns
    assert len(fin) >= 5, f"Expected 5+ years of data, got {len(fin)}"

    # Balance Sheet checks
    bs = result["balance_sheet"]
    print(f"\n=== Balance Sheet ===")
    print(f"Shape: {bs.shape}")
    print(f"Columns: {list(bs.columns)}")

    assert not bs.empty, "Balance Sheet is empty"
    assert "total_assets" in bs.columns
    assert "borrowings" in bs.columns

    # Cash Flow checks
    cf = result["cashflow"]
    print(f"\n=== Cash Flow ===")
    print(f"Shape: {cf.shape}")
    print(f"Columns: {list(cf.columns)}")

    assert not cf.empty, "Cash Flow is empty"
    assert "cfo" in cf.columns

    # Ratios checks
    ratios = result["ratios"]
    print(f"\n=== Ratios ===")
    print(f"Shape: {ratios.shape}")
    print(f"Columns: {list(ratios.columns)}")

    if not ratios.empty:
        assert "roce" in ratios.columns

    # Metadata checks
    meta = result["metadata"]
    print(f"\n=== Metadata ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    assert meta.get("warehouse_id") is not None
    assert meta.get("Market Cap") is not None or meta.get("name") is not None

    # Shareholding checks
    sh = result["shareholding"]
    print(f"\n=== Shareholding ===")
    print(f"Shape: {sh.shape}")
    if not sh.empty:
        print(f"Columns: {list(sh.columns)}")
        print(f"Sample:\n{sh.iloc[-1]}")

    # Growth summary
    gs = result["growth_summary"]
    print(f"\n=== Growth Summary ===")
    for title, data in gs.items():
        print(f"  {title}: {data}")

    # Verify files were saved
    ticker_dir = Path(output_dir) / "ASTRAL"
    assert ticker_dir.exists(), "Output directory not created"
    for fname in ["financials.csv", "balance_sheet.csv", "cashflow.csv"]:
        fpath = ticker_dir / fname
        assert fpath.exists(), f"Missing file: {fname}"
        print(f"\nFile {fname}: {fpath.stat().st_size} bytes")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    test_fetch_astral()
