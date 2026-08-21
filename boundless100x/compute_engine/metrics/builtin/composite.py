"""Composite metrics: Quality-Growth Matrix classification."""

import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows
from boundless100x.compute_engine.sector import group_structure, structure_caveat


# Above this share of total assets, the balance sheet is a portfolio with a
# business attached rather than the other way round. Half is a deliberately
# unambitious line: an operating company parking surplus cash rarely approaches
# it, and the cases this is written about sit far above — JIOFIN at 81%.
_INVESTMENT_HEAVY = 0.50


def _investment_share(balance_sheet) -> float | None:
    """Investments as a share of total assets, or None if not readable."""
    if balance_sheet is None or getattr(balance_sheet, "empty", True):
        return None
    if not {"investments", "total_assets"} <= set(balance_sheet.columns):
        return None

    rows = _get_annual_rows(balance_sheet, 1)
    investments = pd.to_numeric(rows["investments"], errors="coerce").dropna()
    assets = pd.to_numeric(rows["total_assets"], errors="coerce").dropna()
    if investments.empty or assets.empty or float(assets.iloc[-1]) <= 0:
        return None
    return float(investments.iloc[-1]) / float(assets.iloc[-1])


def compute_qg_quadrant(data: dict, params: dict) -> MetricResult:
    """Quality-Growth Matrix position.

    High Quality (RoCE > 15%) + High Growth (PAT CAGR > 15%) = True Wealth Creator
    High Quality + Low Growth = Quality Trap
    Low Quality + High Growth = Growth Trap
    Low Quality + Low Growth = Wealth Destroyer
    """
    quality_threshold = params.get("quality_threshold", 15)
    growth_threshold = params.get("growth_threshold", 15)

    # Quality: 5yr avg RoCE
    ratios = _get_annual_rows(data["ratios"], 5)
    if "roce" not in ratios.columns:
        return MetricResult(error="No roce for QG matrix")

    roce_values = pd.to_numeric(ratios["roce"], errors="coerce").dropna()
    if len(roce_values) < 3:
        return MetricResult(error="Insufficient RoCE data for QG matrix")

    avg_roce = float(roce_values.mean())

    # Growth: 5yr PAT CAGR
    fin = _get_annual_rows(data["financials"], 6)
    pat = pd.to_numeric(fin["pat"], errors="coerce").dropna()
    if len(pat) < 2:
        return MetricResult(error="Insufficient PAT data for QG matrix")

    start = float(pat.iloc[0])
    end = float(pat.iloc[-1])
    actual_years = len(pat) - 1

    if start <= 0 or end <= 0:
        pat_cagr = 0.0
    else:
        pat_cagr = ((end / start) ** (1 / actual_years) - 1) * 100

    high_quality = avg_roce >= quality_threshold
    high_growth = pat_cagr >= growth_threshold

    if high_quality and high_growth:
        quadrant = "true_wealth_creator"
    elif high_quality and not high_growth:
        quadrant = "quality_trap"
    elif not high_quality and high_growth:
        quadrant = "growth_trap"
    else:
        quadrant = "wealth_destroyer"

    # A quadrant is a claim about one business, and a holding company is not
    # one business. Both inputs here are consolidated: EDELWEISS blends a
    # lending book in deliberate run-off with fee businesses compounding at
    # 27-63%, lands at 2.6% revenue growth and 11.8% RoCE, and is filed as a
    # Growth Trap — a verdict describing an average that no segment resembles.
    # The corner is still computed and still shown; what travels with it is
    # the warning that its inputs were blended.
    flags = []
    caveat = structure_caveat(data.get("metadata"))
    if group_structure(data.get("metadata")).get("is_group"):
        flags.append("consolidated_group_ratios")

    # **How much of the balance sheet is somebody else's equity.** The single
    # number that separates "this company allocates capital badly" from "this
    # company is a holding vehicle and most of its return never touches the
    # income statement". JIOFIN reads 81% — ₹133,089 Cr of investments against
    # ₹163,467 Cr of assets, mostly a 6.1% stake in Reliance — and on that
    # basis a 0.91% RoE is arithmetic rather than a verdict on management.
    # Without it, a reader has the low returns and no way to explain them.
    investment_share = _investment_share(data.get("balance_sheet"))
    if investment_share is not None and investment_share >= _INVESTMENT_HEAVY:
        flags.append("investment_heavy_balance_sheet")
        caveat = (caveat + " " if caveat else "") + (
            f"INVESTMENT-HEAVY: {investment_share:.0%} of total assets are "
            f"investments in other companies rather than assets this one "
            f"operates. A listed holding of that size returns through its own "
            f"share price rather than through this company's profit, so return "
            f"measures computed on the full equity base understate it by "
            f"construction — read them as a statement about how hard the "
            f"capital is being worked, not about the quality of the operating "
            f"businesses, and value those separately."
        )

    return MetricResult(
        value=quadrant,
        flags=flags,
        metadata={
            "avg_roce": avg_roce,
            "pat_cagr": pat_cagr,
            "quality_threshold": quality_threshold,
            "growth_threshold": growth_threshold,
            "investment_share": investment_share,
            "structure_caveat": caveat,
        },
    )
