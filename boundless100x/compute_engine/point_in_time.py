"""One shared statement of "what was knowable on date D."

Lifted out of `backtest.WalkForwardBacktest._truncate` (KTD2) so the backtest
and a later simulator can both rewind the same fetched corpus to an arbitrary
historical date without maintaining two copies of the leakage discipline:

  * Period-end cuts, never positional — a row is kept by comparing its actual
    reporting period end against the cutoff, derived through `period_end_date`
    (the one label parser; see `builtin/_helpers.py`).
  * A reporting lag before a period counts as public. Indian filing rules give
    a different clock to each frame — SEBI LODR's ~6-month practical window
    for audited annuals is not the 45-day quarterly-results clock or the
    21-day shareholding-pattern clock — so the lag is per frame, not one
    constant borrowed for all three.
  * `NON_TRUNCATABLE_INPUTS` stripped belt-and-braces from the output, because
    some fetched artefacts (shareholding history today, analyst coverage
    always) cannot be rewound at all for a *particular* caller's purposes even
    though the frame itself is technically cuttable — see the module-level
    note on `shareholding` below.

`truncate_to_date(data, cutoff, ...)` is the one entry point. The backtest's
split-half policy computes its own cutoff exactly as before and delegates
here; a later simulator will call the same function with an arbitrary date.
Both need identical answers to "what was knowable," because a policy that
silently differed between the diagnostic and the tool making it would let one
validate the other for nothing.

## KTD0 — the point-in-time valuation rebuild

`_point_in_time_metadata` (the backtest's original name) omitted `Market Cap`
and `Stock P/E` outright: today's market cap is the single worst leak
available, and a P/E built from a split-adjusted close is not the ratio
anyone saw. That is *still* the default here — `rebuild_valuation=False`,
matching the backtest's own call — but the capability now exists, gated
behind that flag, because a later caller (the simulator) needs both fields to
evaluate the eligibility gates and lifecycle triggers it replays (KTD0,
settled with the owner 2026-08-07):

  * **`Market Cap` = `equity_capital × raw_close ÷ face_value`.** All three
    read at the cutoff: `equity_capital` off the truncated balance sheet's
    last row, `raw_close` off the truncated `close` column's last row — never
    `adj_close`, because a split moves equity capital and the adjusted price
    in the same direction and the two would double-count each other — and
    `face_value` off the raw (untruncated) metadata, which is static.
  * **`Stock P/E`** is rebuilt by literally calling
    `valuation._current_multiple` — raw close over the latest annual EPS — not
    a second implementation of the same formula, and not a reconstruction of
    Screener's stored TTM `Stock P/E`, which is a different multiple
    entirely (KTD0 measured −14% to +1169% divergence trying to compare the
    two). It is written to the `Stock P/E` key regardless, because
    `trailing_peg`, `peg_ratio` and `pe_vs_historical` all read that key
    directly and a renamed key would leave every one of them erroring
    exactly as before this decision. The basis rides beside it as
    `_stock_pe_basis`, never in the key name.
  * **A two-level reconciliation guard**, because a reconstructed market cap
    that fails silently is the whole objection to rebuilding it at all:
      (a) against the stored `Market Cap` in the raw metadata, but only when
          the cutoff genuinely sits at the corpus's latest fetched date —
          that figure exists nowhere else, and 5% tolerance is what the
          corpus measured (20 of 22 tickers within 2%);
      (b) against an independent share count, `pat ÷ eps` off the same
          truncated annual row, checkable at *every* cutoff rather than only
          the one date the simulator never actually scores on.
    A divergence beyond tolerance on either check excludes `Market Cap` at
    that date, and the failure names both figures rather than emitting a bare
    "no market cap."

  Whichever way a field resolves, the reason it is absent travels with the
  view rather than being re-derived by a downstream metric that only sees
  `meta.get("Stock P/E") is None`. Three exclusions read alike from there and
  are not alike at all: `withheld_to_prevent_leak` (this caller declined the
  rebuild), `never_fetched` (an input the formula needs was never available),
  and `reconciliation_failed` (it was computed and did not check out). A
  fourth, `non_positive_input`, is kept distinct from `never_fetched` for the
  one case `_current_multiple` already refuses on its own terms — EPS present
  but not usable — because "the data was never fetched" and "the data exists
  and is nonsensical" are different claims about the corpus.

## `shareholding` is truncatable, and stays withheld from the backtest anyway

`shareholding.csv` is a genuine labelled quarterly series and this module
truncates it like any other frame *when a caller includes it in `data`*. But
`WalkForwardBacktest._load` never reads `shareholding.csv` into `data` in the
first place, so the frame is never there to truncate for the backtest's own
calls, and `non_truncatable_inputs` (default: `NON_TRUNCATABLE_INPUTS`) strips
the key from the output belt-and-braces regardless, so the backtest's
published correlations do not move. A later simulator caller that *does* want
a truncated shareholding view passes its own `non_truncatable_inputs` with
`"shareholding"` removed.
"""

import pandas as pd

from boundless100x.compute_engine.metrics.builtin._helpers import period_end_date

# Frames that carry one row per financial year and can be truncated by their
# own period-end label.
ANNUAL_FRAMES = ("financials", "balance_sheet", "cashflow", "ratios")

# Fetched artefacts that a caller may declare cannot be rewound to a
# historical date. Stripped from the output by default; a caller opts a
# member out by passing its own tuple to `truncate_to_date`.
NON_TRUNCATABLE_INPUTS = ("shareholding", "analyst_coverage", "shareholding_bse")

# Indian annual results are filed within months of the year end; scoring at
# the year end itself would use figures that were not yet public.
ANNUAL_REPORTING_LAG_MONTHS = 6

# SEBI LODR Regulation 33 gives a listed company 45 days from quarter-end to
# file its quarterly financial results — a fraction of the annual lag above,
# so reusing that lag here would withhold figures for months after they were
# actually public.
QUARTERLY_REPORTING_LAG_MONTHS = 2

# SEBI LODR Regulation 31 gives 21 days from quarter-end to file the
# shareholding pattern — the shortest of the three filing clocks.
SHAREHOLDING_REPORTING_LAG_MONTHS = 1

# KTD0: measured against the stored `Market Cap` across the cached corpus,
# 20 of 22 tickers land within 2% and the two outliers (EDELWEISS −2.4%,
# KFINTECH +2.1%) are still inside 5% — tight enough to pass today's corpus
# while still catching a formula gone wrong (partly-paid shares, a face-value
# change, an unusual capital structure). The plan does not give a separate
# number for the second, share-count-based guard, so the same tolerance is
# reused there rather than inventing an unmeasured second constant.
DEFAULT_RECONCILIATION_TOLERANCE = 0.05


def _annual_rows(frame: pd.DataFrame, period_column: str = "year") -> pd.DataFrame:
    """Drop a trailing TTM column Screener appends to the P&L.

    Generalised from `backtest._annual_rows` to take the period column as a
    parameter so the same helper reads `quarterly`/`shareholding`'s `quarter`
    column too. A frame without the column is returned unchanged — safe as a
    no-op for a frame that was never going to carry a TTM row in the first
    place.
    """
    if period_column not in frame.columns:
        return frame
    labels = frame[period_column].astype(str)
    return frame[~labels.str.contains("TTM", case=False, na=False)]


def _truncate_frame(
    frame: pd.DataFrame,
    period_column: str,
    cutoff: pd.Timestamp,
    lag_months: int,
    fallback_rows: int | None,
) -> pd.DataFrame:
    """Keep only rows whose period end, plus this frame's reporting lag, is
    on or before `cutoff`.

    Cutting on the real period-end date (never a bare row position, never a
    bare calendar year) is what stops a trailing part-year column — e.g. a
    "Sep 2025" balance-sheet column Screener appends — from leaking in just
    because it shares a calendar year with the cutoff row.

    A frame whose period labels cannot be parsed at all falls back to
    `frame.head(fallback_rows)` when the caller supplies one (the backtest's
    own split-half row count) and to an empty frame otherwise — guessing a
    row count for an arbitrary cutoff date has no principled answer, so the
    safer default is to exclude rather than guess.
    """
    frame = _annual_rows(frame, period_column)
    period_ends = (
        frame[period_column].map(period_end_date)
        if period_column in frame.columns
        else None
    )
    if period_ends is not None and period_ends.notna().any():
        boundary = period_ends.map(
            lambda pe: (pe + pd.DateOffset(months=lag_months))
            if pd.notna(pe) else pd.NaT
        )
        frame = frame[boundary <= cutoff]
    elif fallback_rows is not None:
        frame = frame.head(fallback_rows)
    else:
        frame = frame.iloc[0:0]
    return frame.reset_index(drop=True)


def _last_numeric(frame: pd.DataFrame | None, column: str) -> float | None:
    """The last non-NaN value in `column`, or None if there isn't one."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _exclusion(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def _withheld_exclusion() -> dict:
    return _exclusion(
        "withheld_to_prevent_leak",
        "valuation reconstruction not requested by this caller",
    )


def _rebuild_market_cap(
    raw: dict,
    truncated: dict,
    at_corpus_latest: bool,
    tolerance: float,
) -> tuple[float | None, dict | None]:
    """`equity_capital × raw_close ÷ face_value`, guarded two ways (KTD0).

    Returns `(market_cap, exclusion)`; exactly one is not None.
    """
    equity_capital = _last_numeric(truncated.get("balance_sheet"), "equity_capital")
    if equity_capital is None or equity_capital <= 0:
        return None, _exclusion(
            "never_fetched", "no positive equity_capital in the truncated balance sheet"
        )

    face_value = raw.get("Face Value")
    if not isinstance(face_value, (int, float)) or face_value <= 0:
        return None, _exclusion("never_fetched", "no positive Face Value in metadata")
    face_value = float(face_value)

    raw_close = _last_numeric(truncated.get("price"), "close")
    if raw_close is None or raw_close <= 0:
        return None, _exclusion(
            "never_fetched", "no usable raw close to rebuild the market cap"
        )

    shares_from_equity = equity_capital / face_value
    rebuilt = shares_from_equity * raw_close

    # Guard (b): an independent share count from the same truncated annual
    # row, checkable at every replay date — unlike guard (a) below, which
    # only has anything to check against at the corpus's one latest date.
    pat = _last_numeric(truncated.get("financials"), "pat")
    eps = _last_numeric(truncated.get("financials"), "eps")
    if pat is None or eps is None or eps == 0:
        return None, _exclusion(
            "never_fetched",
            "no independent pat/eps share count available to cross-check the rebuild",
        )
    shares_from_earnings = pat / eps

    share_divergence = abs(shares_from_equity - shares_from_earnings) / abs(shares_from_equity)
    if share_divergence > tolerance:
        return None, _exclusion(
            "reconciliation_failed",
            f"share count from equity capital ({shares_from_equity:.4f} cr shares) vs "
            f"from pat/eps ({shares_from_earnings:.4f} cr shares) diverge "
            f"{share_divergence * 100:.1f}% against a {tolerance * 100:.0f}% tolerance",
        )

    # Guard (a): only checkable where a fetched Market Cap actually exists —
    # the corpus's latest date, which a genuine historical replay cutoff
    # never is.
    if at_corpus_latest:
        stored = raw.get("Market Cap")
        if isinstance(stored, (int, float)) and stored:
            cap_divergence = abs(rebuilt - float(stored)) / abs(float(stored))
            if cap_divergence > tolerance:
                return None, _exclusion(
                    "reconciliation_failed",
                    f"stored Market Cap ({float(stored):.1f} cr) vs rebuilt "
                    f"({rebuilt:.1f} cr) diverge {cap_divergence * 100:.1f}% "
                    f"against a {tolerance * 100:.0f}% tolerance",
                )

    return rebuilt, None


def _rebuild_stock_pe(truncated: dict) -> tuple[float | None, dict | None, dict | None]:
    """`_current_multiple`'s own formula, called directly rather than
    reimplemented — the exact raw-close-over-annual-EPS basis KTD0 requires,
    with `_current_multiple`'s own non-positive-EPS refusal intact.

    Returns `(stock_pe, price_meta, exclusion)`; `price_meta` is the basis
    metadata `_current_multiple` already returns (price_basis, latest_close,
    latest_eps, eps_period) and is folded into the rebuilt metadata so a
    reader can see exactly what was used.
    """
    # Imported lazily: `valuation.py` is a metrics module and this keeps the
    # dependency direction the same as every other cross-module reuse in
    # `builtin/` (e.g. `valuation._current_multiple` itself imports
    # `profitability._get_annual_rows`) rather than importing a metrics
    # module at `compute_engine` package load time.
    from boundless100x.compute_engine.metrics.builtin.valuation import _current_multiple

    financials = truncated.get("financials")
    if financials is None:
        financials = pd.DataFrame()
    multiple, price_meta, error = _current_multiple(
        {"price": truncated.get("price"), "financials": financials}
    )
    if multiple is None:
        code = "non_positive_input" if "Non-positive" in error else "never_fetched"
        return None, None, _exclusion(code, error)
    return multiple, price_meta, None


def _rebuild_metadata(
    raw: dict,
    truncated: dict,
    full_price: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    rebuild_valuation: bool,
    reconciliation_tolerance: float,
) -> dict:
    """Rebuild only what is genuinely knowable at the cutoff.

    `name`, `sector` and `Face Value` are carried over as static-enough
    metadata (Face Value literally is static — see the module docstring).
    `Current Price` is the truncated price series' own last close, unrelated
    to KTD0's guarded fields.
    """
    past_price = truncated.get("price")
    close = _last_numeric(past_price, "close")
    meta = {
        "name": raw.get("name"),
        "sector": raw.get("sector"),
        "Face Value": raw.get("Face Value"),
        "Current Price": close,
    }

    if not rebuild_valuation:
        meta["_market_cap_exclusion"] = _withheld_exclusion()
        meta["_stock_pe_exclusion"] = _withheld_exclusion()
        return meta

    # "At the corpus's latest date" operationalised: the cutoff excluded no
    # price bar at all relative to the full, untruncated series. Guard (a)
    # only has a stored figure to check against there.
    at_corpus_latest = bool(
        full_price is not None and not full_price.empty
        and full_price["date"].max() <= cutoff
    )

    market_cap, mc_exclusion = _rebuild_market_cap(
        raw, truncated, at_corpus_latest, reconciliation_tolerance
    )
    if market_cap is not None:
        meta["Market Cap"] = market_cap
    else:
        meta["_market_cap_exclusion"] = mc_exclusion

    stock_pe, pe_meta, pe_exclusion = _rebuild_stock_pe(truncated)
    if stock_pe is not None:
        meta["Stock P/E"] = stock_pe
        meta["_stock_pe_basis"] = "annual_eps_reconstructed"
        if pe_meta:
            meta["_stock_pe_source"] = pe_meta
    else:
        meta["_stock_pe_exclusion"] = pe_exclusion

    return meta


def truncate_to_date(
    data: dict,
    cutoff: pd.Timestamp,
    *,
    annual_lag_months: int = ANNUAL_REPORTING_LAG_MONTHS,
    quarterly_lag_months: int = QUARTERLY_REPORTING_LAG_MONTHS,
    shareholding_lag_months: int = SHAREHOLDING_REPORTING_LAG_MONTHS,
    annual_fallback_rows: int | None = None,
    rebuild_valuation: bool = False,
    reconciliation_tolerance: float = DEFAULT_RECONCILIATION_TOLERANCE,
    non_truncatable_inputs: tuple = NON_TRUNCATABLE_INPUTS,
) -> tuple[dict | None, str]:
    """Rebuild a point-in-time view of `data` as it was knowable at `cutoff`.

    `data` carries whatever frames the caller loaded — `price` and
    `_metadata_raw` are required, everything else (`financials`,
    `balance_sheet`, `cashflow`, `ratios`, `quarterly`, `shareholding`) is
    truncated by its own period label when present and simply absent from the
    output when the caller never loaded it. This is what lets one function
    serve both callers: the backtest's `data` never contains `quarterly` or
    `shareholding` (its own `_load` never reads those files), so those frames
    are never produced for it regardless of the lag constants below; a
    simulator that does load them gets them truncated on the same terms as
    everything else.

    `cutoff` plays the role the backtest calls `truncation_date` — the actual
    calendar date being simulated, already past whatever reporting lag
    applies to the frame the caller derived it from. Price is compared to it
    directly (a traded price is public the day it prints); every other frame
    compares `period_end + its own lag <= cutoff` row by row, which is
    mathematically the same test as the backtest's original single-lag
    `period_end <= cutoff_period_end` when `cutoff` is itself
    `cutoff_period_end + lag` — shifting both sides of an inequality by the
    same fixed number of months preserves the ordering, including at the
    boundary row itself.

    Returns `(truncated, reason)`. `truncated` is None with a `reason` naming
    why when nothing can be scored at this cutoff (currently: the price
    series starts after it); otherwise `reason` is empty.
    """
    price = data["price"]
    past_price = price[price["date"] <= cutoff]
    if past_price.empty:
        return None, f"price history starts after {cutoff.date()}"

    truncated: dict = {"price": past_price.reset_index(drop=True)}

    for name in ANNUAL_FRAMES:
        if name not in data:
            continue
        truncated[name] = _truncate_frame(
            data[name], "year", cutoff, annual_lag_months, annual_fallback_rows
        )

    if "quarterly" in data:
        truncated["quarterly"] = _truncate_frame(
            data["quarterly"], "quarter", cutoff, quarterly_lag_months, None
        )

    if "shareholding" in data:
        truncated["shareholding"] = _truncate_frame(
            data["shareholding"], "quarter", cutoff, shareholding_lag_months, None
        )

    truncated["metadata"] = _rebuild_metadata(
        data.get("_metadata_raw", {}),
        truncated,
        price,
        cutoff,
        rebuild_valuation=rebuild_valuation,
        reconciliation_tolerance=reconciliation_tolerance,
    )

    # Belt and braces: a caller's `non_truncatable_inputs` names frames it
    # declares itself unable to rewind for its own purposes, regardless of
    # whether this function just produced a genuinely truncated one above.
    for leaky in non_truncatable_inputs:
        truncated.pop(leaky, None)

    return truncated, ""
