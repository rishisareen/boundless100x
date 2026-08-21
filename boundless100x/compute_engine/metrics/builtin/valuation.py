"""Valuation metrics: P/E, PEG, trailing PEG, EV/EBITDA, DCF, reverse DCF."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import (
    TREASURY_ADJUSTED_FLAG,
    detect_fcf_outliers,
    operating_free_cash_flow,
    period_end_date,
)
from boundless100x.compute_engine.metrics.builtin.growth import (
    NEGLIGIBLE_BASE_FLAG,
    _base_effect_reason,
    _negligible_base,
)
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_pe_ttm(data: dict, params: dict) -> MetricResult:
    """P/E ratio from metadata (TTM)."""
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None:
        return MetricResult(error="No P/E in metadata")

    flags = []
    if pe > 80:
        flags.append("very_expensive_pe")
    elif pe > 50:
        flags.append("expensive_pe")
    elif pe < 15:
        flags.append("cheap_pe")

    return MetricResult(value=float(pe), flags=flags)


def _net_worth(balance_sheet: pd.DataFrame) -> pd.Series | None:
    """Equity capital + reserves — the book the shareholders own."""
    required = ("equity_capital", "reserves")
    if any(col not in balance_sheet.columns for col in required):
        return None
    parts = [pd.to_numeric(balance_sheet[col], errors="coerce") for col in required]
    return (parts[0] + parts[1]).dropna()


def compute_price_to_book(data: dict, params: dict) -> MetricResult:
    """Price / book value per share.

    The valuation metric a lender is actually judged on, and the one this
    registry had no equivalent of. Every price metric here divided by a flow —
    earnings, EBITDA, free cash flow — and for a financial the last two are
    structurally broken while the first swings with the credit cycle. Book
    value is the stock the business is run on, so P/B is the reading that
    survives when the others do not.

    Read against RoE, never alone: a bank earning 20% on equity is worth
    several times book and one earning 8% is not, so the same 2x means
    opposite things. That relationship is why this is banded rather than
    scored sector-relative — the peer set that would make a percentile
    meaningful is not fetched, and an unqualified percentile would call every
    high-RoE franchise expensive.

    Prefers Screener's own `Book Value` (per share, same basis as the quoted
    price) and falls back to the balance sheet, which needs the share count
    reconstructed from face value.
    """
    meta = data.get("metadata", {}) or {}
    price = meta.get("Current Price")
    if price is None or float(price) <= 0:
        return MetricResult(error="No current price in metadata")
    price = float(price)

    book_per_share = meta.get("Book Value")
    basis = "screener_book_value"

    if book_per_share is None or float(book_per_share) <= 0:
        # Reconstruct from the balance sheet. Screener reports equity capital
        # at face value, so shares = equity_capital / face_value — the same
        # derivation the dilution metric uses.
        bs = data.get("balance_sheet")
        face_value = meta.get("Face Value")
        if bs is None or getattr(bs, "empty", True) or not face_value:
            return MetricResult(
                error="No book value in metadata and no balance sheet to derive one"
            )
        net_worth = _net_worth(_get_annual_rows(bs, 1))
        equity_capital = pd.to_numeric(
            _get_annual_rows(bs, 1)["equity_capital"], errors="coerce"
        ).dropna()
        if net_worth is None or net_worth.empty or equity_capital.empty:
            return MetricResult(error="Cannot derive book value from balance sheet")
        shares_cr = float(equity_capital.iloc[-1]) / float(face_value)
        if shares_cr <= 0:
            return MetricResult(error="Cannot derive share count for book value")
        book_per_share = float(net_worth.iloc[-1]) / shares_cr
        basis = "derived_from_balance_sheet"

    book_per_share = float(book_per_share)
    if book_per_share <= 0:
        # Negative book is a real state (accumulated losses exceed capital)
        # and P/B is undefined there rather than infinite — reporting it as a
        # very large multiple would read as "expensive" when it means
        # "insolvent on a book basis".
        return MetricResult(
            error="Book value is zero or negative — price/book undefined"
        )

    pb = price / book_per_share

    flags = []
    if pb < 1.0:
        flags.append("below_book_value")
    elif pb > 5.0:
        flags.append("rich_price_to_book")

    return MetricResult(
        value=float(pb),
        flags=flags,
        metadata={
            "price": price,
            "book_value_per_share": book_per_share,
            "book_value_basis": basis,
        },
    )


def compute_peg(data: dict, params: dict) -> MetricResult:
    """PEG = P/E ÷ trailing 5yr EPS CAGR.

    Despite the "forward" framing this used to carry, no forward estimate
    feeds this: it is a trailing PEG, on a longer window than
    `compute_trailing_peg`'s 3yr PAT CAGR, nothing more. A real forward PEG
    would need timestamped analyst EPS estimates, which this pipeline does
    not fetch.
    """
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None or pe <= 0:
        return MetricResult(error="No P/E for PEG")

    df = _get_annual_rows(data["financials"], 6)
    eps = pd.to_numeric(df["eps"], errors="coerce").dropna()
    if len(eps) < 2:
        return MetricResult(error="Insufficient EPS data for PEG")

    start = float(eps.iloc[0])
    end = float(eps.iloc[-1])
    actual_years = len(eps) - 1

    if start <= 0 or end <= 0:
        return MetricResult(error="Non-positive EPS for PEG")

    eps_cagr = ((end / start) ** (1 / actual_years) - 1) * 100
    if eps_cagr <= 0:
        return MetricResult(error="Negative EPS CAGR, PEG undefined")

    peg = pe / eps_cagr

    flags = []
    if peg < 1.0:
        flags.append("attractively_valued_peg")
    elif peg > 2.5:
        flags.append("expensive_peg")

    metadata = {"pe": pe, "eps_cagr": eps_cagr}
    if _negligible_base(start, end):
        flags.append(NEGLIGIBLE_BASE_FLAG)
        metadata["base_effect_reason"] = _base_effect_reason(
            "EPS", start, end, actual_years
        )

    return MetricResult(
        value=float(peg),
        flags=flags,
        metadata=metadata,
    )


def compute_trailing_peg(data: dict, params: dict) -> MetricResult:
    """Trailing PEG = P/E ÷ trailing 3yr PAT CAGR."""
    cagr_years = params.get("cagr_years", 3)
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None or pe <= 0:
        return MetricResult(error="No P/E for trailing PEG")

    df = _get_annual_rows(data["financials"], cagr_years + 1)
    pat = pd.to_numeric(df["pat"], errors="coerce").dropna()
    if len(pat) < 2:
        return MetricResult(error="Insufficient PAT data")

    start = float(pat.iloc[0])
    end = float(pat.iloc[-1])
    actual_years = len(pat) - 1

    if start <= 0 or end <= 0:
        return MetricResult(error="Non-positive PAT for trailing PEG")

    pat_cagr = ((end / start) ** (1 / actual_years) - 1) * 100
    if pat_cagr <= 0:
        return MetricResult(error="Negative PAT CAGR, trailing PEG undefined")

    tpeg = pe / pat_cagr

    flags = []
    if tpeg < 1.0:
        flags.append("attractive_trailing_peg")

    # A PEG is only as good as its denominator. JIOFIN divided a P/E of 78 by
    # a 269% "CAGR" measured from a post-demerger base of ₹31 Cr and came out
    # at 0.29x — the single heaviest metric in its Price element, and the one
    # that carried its entry-price gate. The number is kept and shown; the flag
    # is what stops it scoring and stops it gating (UNSCORABLE_FLAGS).
    metadata = {"pe": pe, "pat_cagr": pat_cagr, "years": actual_years}
    if _negligible_base(start, end):
        flags.append(NEGLIGIBLE_BASE_FLAG)
        metadata["base_effect_reason"] = _base_effect_reason(
            "PAT", start, end, actual_years
        )

    return MetricResult(
        value=float(tpeg),
        flags=flags,
        metadata=metadata,
    )


def compute_ev_ebitda(data: dict, params: dict) -> MetricResult:
    """EV/EBITDA = (Market Cap + Debt - Cash) / (Operating Profit + Depreciation)."""
    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None:
        return MetricResult(error="No market cap for EV/EBITDA")

    bs = _get_annual_rows(data["balance_sheet"], 1)
    fin = _get_annual_rows(data["financials"], 1)

    if bs.empty or fin.empty:
        return MetricResult(error="No BS/financials for EV/EBITDA")

    debt = pd.to_numeric(bs["borrowings"], errors="coerce").iloc[-1]
    debt = 0.0 if pd.isna(debt) else float(debt)

    op = pd.to_numeric(fin["operating_profit"], errors="coerce").iloc[-1]
    dep = pd.to_numeric(fin["depreciation"], errors="coerce").iloc[-1]

    if pd.isna(op) or pd.isna(dep):
        return MetricResult(error="Missing EBITDA components")

    ebitda = float(op) + float(dep)
    if ebitda <= 0:
        return MetricResult(error="Non-positive EBITDA")

    # Simplified EV (no cash subtraction — Screener doesn't provide cash directly)
    ev = mcap + debt
    ev_ebitda = ev / ebitda

    return MetricResult(
        value=float(ev_ebitda),
        metadata={"ev": ev, "ebitda": ebitda},
    )


# How far back a fiscal year end may reach for a traded price before the year
# is treated as uncovered — comfortably more than any exchange holiday run.
PRICE_LOOKBACK_DAYS = 45


def _price_basis(price: pd.DataFrame) -> str:
    """What the `close` column can be trusted to mean.

    Presence of `adj_close` means the fetch carried both series, so `close` is
    genuinely the raw traded price. Its absence means a legacy single-close
    cache whose adjustment status is unknown — recorded rather than assumed,
    because an adjusted close silently understates past prices and every
    valuation metric built on one reads cheaper than the company traded.
    """
    return "raw_close" if "adj_close" in price.columns else "legacy_close_unknown_adjustment"


def _close_on_or_before(price: pd.DataFrame, when: pd.Timestamp) -> float | None:
    """The last traded close at or before a date, within the lookback window."""
    prior = price[price["date"] <= when]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    if (when - row["date"]).days > PRICE_LOOKBACK_DAYS:
        return None
    close = float(row["close"])
    return close if close > 0 else None


def compute_pe_percentile(data: dict, params: dict) -> MetricResult:
    """Where today's P/E sits in the company's own traded P/E history.

    The band must be built from the price at each past year end divided by that
    year's EPS. Dividing today's price by past EPS instead produces a rescaled
    reciprocal of the earnings series, so the percentile tracks earnings growth
    rather than valuation — and any company earning near a high scores as
    maximally cheap however dearly it actually trades.
    """
    years = params.get("years", 10)
    meta = data.get("metadata", {})
    current_pe = meta.get("Stock P/E")
    if current_pe is None:
        return MetricResult(error="No current P/E")

    price = data.get("price")
    if price is None or len(price) == 0:
        return MetricResult(error="No price history for historical P/E band")

    # The band divides each past year-end close by that year's as-reported
    # EPS, so it needs the raw traded close. The band is computed either way,
    # but the basis is recorded so a distorted read is traceable.
    price_basis = _price_basis(price)

    price = price.copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce", utc=True).dt.tz_localize(None)
    price = price.dropna(subset=["date"]).sort_values("date")
    if price.empty:
        return MetricResult(error="No usable price dates for historical P/E band")

    df = _get_annual_rows(data["financials"], years)
    if "eps" not in df.columns or "year" not in df.columns:
        return MetricResult(error="Financials lack eps/year for PE percentile")

    historical_pes = []
    for label, raw_eps in zip(df["year"], pd.to_numeric(df["eps"], errors="coerce")):
        if pd.isna(raw_eps) or raw_eps <= 0:
            continue
        period_end = period_end_date(label)
        if period_end is None:
            continue
        close = _close_on_or_before(price, period_end)
        if close is None:
            continue
        historical_pes.append(close / float(raw_eps))

    if len(historical_pes) < 5:
        return MetricResult(
            error=f"Only {len(historical_pes)} years with both price and positive EPS"
        )

    # Percentile of current PE in historical distribution
    below = sum(1 for pe in historical_pes if pe <= current_pe)
    percentile = below / len(historical_pes) * 100

    flags = []
    if percentile > 75:
        flags.append("pe_above_historical_75th")
    elif percentile < 25:
        flags.append("pe_below_historical_25th")
    if price_basis != "raw_close":
        # Adjusted closes understate past prices, so the band reads cheaper
        # than it was and today's percentile reads higher. Say so rather than
        # presenting a distorted percentile as settled.
        flags.append("pe_band_legacy_price_basis")

    return MetricResult(
        value=float(percentile),
        raw_series=[round(pe, 2) for pe in historical_pes],
        flags=flags,
        metadata={
            "years_used": len(historical_pes),
            "pe_min": round(min(historical_pes), 2),
            "pe_max": round(max(historical_pes), 2),
            "pe_median": round(float(np.median(historical_pes)), 2),
            "current_pe": float(current_pe),
            "price_basis": price_basis,
        },
    )


def compute_dcf_margin(data: dict, params: dict) -> MetricResult:
    """DCF Margin of Safety = (Intrinsic Value - Current Price) / Current Price × 100."""
    projection_years = params.get("projection_years", 10)
    terminal_growth = params.get("terminal_growth", 0.04)
    discount_rate = params.get("discount_rate", 0.12)

    meta = data.get("metadata", {})
    current_price = meta.get("Current Price")
    if current_price is None or current_price <= 0:
        return MetricResult(error="No current price for DCF")

    cf = _get_annual_rows(data["cashflow"], 5)
    fin = _get_annual_rows(data["financials"], 6)

    # Money parked in mutual funds and deposits is not capital expenditure, so
    # it does not reduce what the business generated. See
    # `operating_free_cash_flow` — CAPLIPOINT valued at ₹43 against a ₹2,561
    # price because eight years of treasury deployment were read as spending.
    _, fcf, treasury = operating_free_cash_flow(
        cf, _get_annual_rows(data.get("balance_sheet", pd.DataFrame()), 6)
    )
    if len(fcf) < 3:
        return MetricResult(error="Insufficient cash flow for DCF")

    fcf_series = fcf.tail(5).values
    avg_fcf_raw = float(np.mean(fcf_series))

    # Detect FCF outliers (likely M&A years)
    clean_fcf, outlier_flags = detect_fcf_outliers(fcf_series)
    if treasury.get("adjusted"):
        outlier_flags = list(outlier_flags) + [TREASURY_ADJUSTED_FLAG]
    avg_fcf = float(np.nanmean(clean_fcf)) if not np.all(np.isnan(clean_fcf)) else avg_fcf_raw

    if avg_fcf <= 0:
        # Even after removing outliers, FCF is negative
        all_flags = ["negative_average_fcf"] + outlier_flags
        if outlier_flags and avg_fcf_raw <= 0 and avg_fcf != avg_fcf_raw:
            all_flags.append("negative_fcf_even_after_outlier_removal")
        return MetricResult(
            value=-100.0,
            flags=all_flags,
            metadata={
                "avg_fcf_raw": avg_fcf_raw,
                "avg_fcf_organic": avg_fcf,
                "outlier_years": len(outlier_flags),
            },
        )

    # Estimate FCF growth from revenue CAGR
    revenue = pd.to_numeric(fin["revenue"], errors="coerce").dropna()
    if len(revenue) >= 2 and revenue.iloc[0] > 0:
        rev_cagr = (revenue.iloc[-1] / revenue.iloc[0]) ** (1 / (len(revenue) - 1)) - 1
        fcf_growth = min(rev_cagr, 0.25)  # Cap at 25%
    else:
        fcf_growth = 0.10

    # DCF: project FCF, discount, add terminal
    pv_fcfs = 0.0
    projected_fcf = avg_fcf
    for yr in range(1, projection_years + 1):
        projected_fcf *= (1 + fcf_growth)
        pv_fcfs += projected_fcf / (1 + discount_rate) ** yr

    terminal_value = projected_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
    intrinsic_total = pv_fcfs + pv_terminal

    # Per share: get shares from equity capital
    bs = _get_annual_rows(data.get("balance_sheet", pd.DataFrame()), 1)
    face_value = meta.get("Face Value", 1)
    if not bs.empty and "equity_capital" in bs.columns:
        eq_cap = pd.to_numeric(bs["equity_capital"], errors="coerce").iloc[-1]
        if not pd.isna(eq_cap) and face_value and face_value > 0:
            shares_cr = eq_cap / face_value
            intrinsic_per_share = intrinsic_total / shares_cr if shares_cr > 0 else 0
        else:
            intrinsic_per_share = 0
    else:
        intrinsic_per_share = 0

    if intrinsic_per_share <= 0:
        return MetricResult(error="Cannot compute per-share intrinsic value")

    margin = (intrinsic_per_share - current_price) / current_price * 100

    flags = []
    if margin > 20:
        flags.append("dcf_undervalued")
    elif margin < -30:
        flags.append("dcf_overvalued")

    return MetricResult(
        value=float(margin),
        flags=flags,
        metadata={
            "intrinsic_per_share": float(intrinsic_per_share),
            "current_price": current_price,
            "fcf_growth_assumed": float(fcf_growth * 100),
        },
    )


def compute_reverse_dcf(data: dict, params: dict) -> MetricResult:
    """Reverse DCF: solve for the growth rate implied by current market price."""
    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None or mcap <= 0:
        return MetricResult(error="No market cap for reverse DCF")

    cf = _get_annual_rows(data["cashflow"], 5)
    _, fcf, treasury = operating_free_cash_flow(
        cf, _get_annual_rows(data.get("balance_sheet", pd.DataFrame()), 6)
    )
    if len(fcf) < 3:
        return MetricResult(error="Insufficient cash flow for reverse DCF")

    raw_fcf_series = fcf.tail(5).values
    clean_fcf, outlier_flags = detect_fcf_outliers(raw_fcf_series)
    if treasury.get("adjusted"):
        outlier_flags = list(outlier_flags) + [TREASURY_ADJUSTED_FLAG]
    avg_fcf = float(np.nanmean(clean_fcf)) if not np.all(np.isnan(clean_fcf)) else float(np.mean(raw_fcf_series))
    if avg_fcf <= 0:
        return MetricResult(
            error="Negative average FCF, reverse DCF undefined",
            flags=outlier_flags,
        )

    discount_rate = float(params.get("discount_rate", 0.12))
    terminal_growth = float(params.get("terminal_growth", 0.04))
    projection_years = int(params.get("projection_years", 10))

    # Binary search for implied growth, bounded to [-10%, +50%]. A company
    # priced beyond a bound pins to it — directionally meaningful, but the
    # exact value is then an artifact of the bound rather than a measurement,
    # and it feeds both scoring and the price-gate veto, so it must say so.
    low, high = -0.10, 0.50
    for _ in range(50):
        mid = (low + high) / 2
        pv = 0.0
        proj_fcf = avg_fcf
        for yr in range(1, projection_years + 1):
            proj_fcf *= (1 + mid)
            pv += proj_fcf / (1 + discount_rate) ** yr
        tv = proj_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv += tv / (1 + discount_rate) ** projection_years

        if pv < mcap:
            low = mid
        else:
            high = mid

    implied_growth = (low + high) / 2 * 100
    saturated_at = None
    if low > 0.50 - 1e-6:
        saturated_at = "ceiling"  # market implies 50%+ growth — search ran out of room
    elif high < -0.10 + 1e-6:
        saturated_at = "floor"    # market prices below a 10% perpetual decline

    # Compare to actual revenue CAGR
    fin = _get_annual_rows(data["financials"], 6)
    revenue = pd.to_numeric(fin["revenue"], errors="coerce").dropna()
    actual_cagr = None
    if len(revenue) >= 2 and revenue.iloc[0] > 0:
        actual_cagr = ((revenue.iloc[-1] / revenue.iloc[0]) ** (1 / (len(revenue) - 1)) - 1) * 100

    flags = []
    if saturated_at is not None:
        flags.append("reverse_dcf_saturated")
    if actual_cagr is not None and implied_growth > actual_cagr * 1.5:
        flags.append("reverse_dcf_overpriced")
    elif actual_cagr is not None and implied_growth < actual_cagr * 0.7:
        flags.append("reverse_dcf_underpriced")

    return MetricResult(
        value=float(implied_growth),
        flags=flags,
        metadata={
            "actual_cagr": actual_cagr,
            "avg_fcf": avg_fcf,
            "saturated_at": saturated_at,
            "search_bounds_pct": [-10.0, 50.0],
        },
    )


# ── Re-rating headroom (Phase 2, zero weight) ──────────────────────────────
#
# Every default below is mirrored in `price.yaml` params, which is the
# authoritative copy — tuning happens there, as a config edit, exactly as the
# metric registry does everywhere else. These exist so the function still
# computes when called directly (tests, a future simulator) rather than
# silently taking a middle band from nowhere.
#
# They are STARTING POINTS awaiting Phase 5 simulator evidence, in the same
# spirit as the lifecycle trigger thresholds. Under KTD8's two-hash split,
# tuning them moves `forward_signal_hash` and leaves `registry_hash` — and so
# every ticker's momentum baseline — untouched.
DEFAULT_ROCE_BANDS = [12.0, 18.0, 25.0]
DEFAULT_GROWTH_BANDS = [8.0, 15.0, 25.0]
# Rows are RoCE bands (low to high), columns growth bands (low to high).
DEFAULT_JUSTIFIED_MULTIPLE = [
    [10.0, 13.0, 16.0, 19.0],
    [13.0, 17.0, 22.0, 27.0],
    [16.0, 22.0, 29.0, 36.0],
    [19.0, 26.0, 35.0, 45.0],
]
# Bands on the count of years RoCE cleared its threshold, and the multiplier
# each earns. A business that has held its returns deserves a longer runway of
# them priced in; one that has not does not.
DEFAULT_LONGEVITY_BANDS = [3.0, 6.0, 8.0]
DEFAULT_LONGEVITY_MULTIPLIERS = [0.85, 1.0, 1.10, 1.20]

DEFAULT_FAVOURABLE_PCT = 25.0
DEFAULT_STRETCHED_PCT = -25.0


def _band_index(value: float, boundaries: list) -> int:
    """Which band `value` falls in: 0 below the first boundary, N at/above the last."""
    index = 0
    for boundary in boundaries:
        if value < float(boundary):
            break
        index += 1
    return index


def _current_multiple(data: dict) -> tuple[float | None, dict, str]:
    """Latest traded close over latest as-reported annual EPS.

    Deliberately *not* `metadata["Stock P/E"]`. The backtest omits Stock P/E on
    purpose — the stored closes are split- and dividend-adjusted, so a rebuilt
    ratio is not the one anyone saw — and a metric reading it excludes itself
    from the walk-forward check. Price, financials and ratios are all
    truncatable, so building the multiple here keeps this metric inside the
    backtest (A2), where at zero weight it costs neither the correlations nor
    the coverage floor.

    Returns `(multiple, metadata, error)`; exactly one of the first and last
    is meaningful.
    """
    price = data.get("price")
    if price is None or len(price) == 0:
        return None, {}, "No price history for the current multiple"

    price_basis = _price_basis(price)
    close = pd.to_numeric(price["close"], errors="coerce").dropna()
    if close.empty:
        return None, {}, "No usable closes for the current multiple"

    fin = _get_annual_rows(data.get("financials", pd.DataFrame()), 1)
    if fin.empty or "eps" not in fin.columns:
        return None, {}, "Financials lack an annual EPS row for the current multiple"

    eps = pd.to_numeric(fin["eps"], errors="coerce").dropna()
    if eps.empty:
        return None, {}, "No numeric EPS for the current multiple"

    latest_eps = float(eps.iloc[-1])
    if latest_eps <= 0:
        return None, {}, "Non-positive EPS — a re-rating multiple is undefined"

    latest_close = float(close.iloc[-1])
    if latest_close <= 0:
        # The same guard `_close_on_or_before` applies to this column. A zero
        # raises deep inside the ratio and surfaces as an opaque exception
        # string; a negative one produces a negative multiple that reads as a
        # perfectly ordinary headroom figure. Both are scraping glitches, and
        # neither may become a signal.
        return None, {}, "Non-positive traded close — a re-rating multiple is undefined"

    return (
        latest_close / latest_eps,
        {
            "price_basis": price_basis,
            "latest_close": latest_close,
            "latest_eps": latest_eps,
            "eps_period": str(fin["year"].iloc[-1]) if "year" in fin.columns else None,
        },
        "",
    )


def compute_rerating_headroom(data: dict, params: dict) -> MetricResult:
    """How much multiple expansion the company's own fundamentals would justify.

    `headroom_pct = (justified_multiple / current_multiple - 1) x 100`.

    A **ratio expressed as a percentage**, not a difference in multiple points,
    so +40 means "fundamentals justify a multiple 40% above what is being paid"
    and reads the same for a company on 15x as on 60x. Positive means room to
    re-rate up, matching the metric's name.

    The justified multiple is built only from the company's **own** readings —
    a banded lookup on (5yr RoCE x 5yr PAT CAGR), scaled by a longevity
    multiplier off the RoCE-above-threshold year count. Never a sector-relative
    anchor: v05 §14.5 keeps out the peer comparison v04 deliberately removed,
    because "cheap for a chemicals company" is a statement about chemicals, not
    about this business.

    Those three readings are produced by calling the *existing* metric
    functions rather than reimplementing their windows. A second definition of
    the same company's RoCE would leave a reader reconciling two numbers that
    should be one, and the bands are meant to be anchored to figures already
    visible in `scores["details"]`.

    There is no default justified multiple. A missing RoCE, growth or price
    reading yields an error, because an unknown quality profile silently
    receiving the middle band is exactly the confident-but-empty number this
    phase exists to avoid.

    Emits no `raw_series` (KTD6): a series of multiples sitting behind a ratio
    value is the unit mismatch that trapped Phase 1's `persist_years` rules.
    """
    from boundless100x.compute_engine.metrics.builtin.growth import compute_cagr
    from boundless100x.compute_engine.metrics.builtin.longevity import (
        compute_threshold_consistency,
    )
    from boundless100x.compute_engine.metrics.builtin.profitability import (
        compute_roce_avg,
    )

    current, price_meta, error = _current_multiple(data)
    if current is None:
        return MetricResult(error=error)

    roce = compute_roce_avg(data, {"years": params.get("roce_years", 5)})
    if not roce.ok:
        return MetricResult(error=f"No RoCE for the justified multiple: {roce.error}")

    growth = compute_cagr(data, {
        "field": params.get("growth_field", "pat"),
        "years": params.get("growth_years", 5),
    })
    if not growth.ok:
        return MetricResult(error=f"No growth for the justified multiple: {growth.error}")

    longevity = compute_threshold_consistency(data, {
        "field": "roce",
        "years": params.get("longevity_years", 10),
        "threshold": params.get("longevity_threshold", 15),
    })
    if not longevity.ok:
        return MetricResult(
            error=f"No RoCE consistency for the justified multiple: {longevity.error}"
        )

    roce_bands = params.get("roce_bands", DEFAULT_ROCE_BANDS)
    growth_bands = params.get("growth_bands", DEFAULT_GROWTH_BANDS)
    table = params.get("justified_multiple", DEFAULT_JUSTIFIED_MULTIPLE)
    longevity_bands = params.get("longevity_bands", DEFAULT_LONGEVITY_BANDS)
    multipliers = params.get("longevity_multipliers", DEFAULT_LONGEVITY_MULTIPLIERS)

    if len(table) != len(roce_bands) + 1 or any(
        len(row) != len(growth_bands) + 1 for row in table
    ):
        return MetricResult(
            error=(
                f"justified_multiple must be {len(roce_bands) + 1}x"
                f"{len(growth_bands) + 1} for the declared bands"
            )
        )
    if len(multipliers) != len(longevity_bands) + 1:
        return MetricResult(
            error=f"longevity_multipliers must have {len(longevity_bands) + 1} entries"
        )

    roce_index = _band_index(float(roce.value), roce_bands)
    growth_index = _band_index(float(growth.value), growth_bands)
    longevity_index = _band_index(float(longevity.value), longevity_bands)

    base_multiple = float(table[roce_index][growth_index])
    multiplier = float(multipliers[longevity_index])
    justified = base_multiple * multiplier

    headroom_pct = (justified / current - 1) * 100

    favourable = float(params.get("favourable_pct", DEFAULT_FAVOURABLE_PCT))
    stretched = float(params.get("stretched_pct", DEFAULT_STRETCHED_PCT))

    if headroom_pct >= favourable:
        band, flags = "favourable", ["rerating_headroom_favourable"]
    elif headroom_pct <= stretched:
        band, flags = "stretched", ["rerating_headroom_stretched"]
    else:
        band, flags = "fair", []

    return MetricResult(
        value=float(headroom_pct),
        flags=flags,
        metadata={
            "current_multiple": float(current),
            "justified_multiple": float(justified),
            "base_multiple": base_multiple,
            "longevity_multiplier": multiplier,
            "roce_5yr": float(roce.value),
            "growth_cagr": float(growth.value),
            "growth_field": params.get("growth_field", "pat"),
            "roce_years_above_threshold": float(longevity.value),
            "band": band,
            "direction": "higher_is_better",
            "bands_pct": {"favourable_at": favourable, "stretched_at": stretched},
            **price_meta,
        },
    )


def compute_earnings_yield_spread(data: dict, params: dict) -> MetricResult:
    """Earnings Yield (1/PE) minus the India 10yr G-Sec yield."""
    meta = data.get("metadata", {})
    pe = meta.get("Stock P/E")
    if pe is None or pe <= 0:
        return MetricResult(error="No P/E for earnings yield")

    earnings_yield = 100.0 / pe
    gsec_yield = float(params.get("gsec_yield_pct", 7.0))
    spread = earnings_yield - gsec_yield

    flags = []
    if spread > 0:
        flags.append("earnings_yield_above_gsec")
    else:
        flags.append("gsec_more_attractive")

    return MetricResult(
        value=float(spread),
        flags=flags,
        metadata={"earnings_yield": earnings_yield, "gsec_yield": gsec_yield},
    )
