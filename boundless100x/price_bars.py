"""Bar-selection hygiene every `price_volume.csv` reader needs (Phase 4, U6/U7).

Two documented hazards recur wherever a raw price frame is read:

  * `adj_close_is_estimated` marks a jugaad-data fetch where `adj_close` is
    just `close` under another name. Pricing off it unadjusted turns a 1:5
    split into what reads like an 80% loss.
  * The adjusted column trails the raw one by a bar — the source publishes
    today's close before today's adjusted close — so a freshly fetched
    series routinely ends in an empty `adj_close`. Selecting the last row
    unconditionally turns that into a total loss.

`lifecycle/friction.py`'s `_usable_bars` solved this first, and
`simulator/ledger.py` and `simulator/outputs.py` each grew their own copy
solving the same two hazards for the simulator's own callers. That was two
independent duplicates of the identical "drop bad bars, pick a direction"
logic within one diff, with a third (`friction.py`) already on record — and
`json_store.py`'s own module docstring names exactly why that is worth
stopping rather than letting a fourth appear when U7's per-replay-date loop
lands: reaching into another module's underscore-private helper across a
package boundary is "a note to the reader that says 'do not depend on
this' beside code that does," and two copies of a rule is a standing
invitation for the third to read a bar differently.

This is a **leaf module, importing nothing from this project** — the same
shape `json_store.py` and `forward_growth_schema.py` already take, and for
the same reason: a shared dependency two-plus modules read is only safe to
share if it cannot itself pull either of them (or anything else) into its
own import graph. `pandas` is the one dependency, because cleaning a price
frame is what this module is for; nothing here is project-specific.

**`lifecycle/friction.py`'s `_usable_bars` is deliberately left alone and
NOT folded into this module.** It answers a related but not identical
question — it also names *which column* (`close` vs `adj_close`) ended up
usable and carries a row-wise reason string, because
`compute_position_return` puts both into the sentences it returns on a
failure (`"no trading bar on or after the entry date ... the usable
{column} series ends {last}"`). Forcing that third field onto every caller
here would bloat this module's contract with a value `ledger.py` and
`outputs.py` never read, or force `friction.py` to discard information its
own callers use — either way the wrong trade for genuinely small, stable,
already-tested production code. `friction.py` stays a third, independent
implementation of the same two hazards; this module exists for the two
duplicates that appeared *within* the simulator, not to chase down every
implementation of this idiom in the codebase.
"""

from __future__ import annotations

import pandas as pd


def is_estimated(value) -> bool:
    """Whether a bar's adjusted close is an alias for its raw close.

    `adj_close_is_estimated` reaches a caller through a CSV round trip on one
    path and a DataFrame on another, so it arrives as a bool, as the string
    "True", or as nothing at all. A missing value is not a claim that the bar
    is estimated — the flag postdates part of the corpus — so it reads False.
    """
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def clean_price_bars(price_df) -> pd.DataFrame | None:
    """A `{"date", "price"}` frame, sorted by date, with unusable rows
    dropped — or `None` if nothing survives.

    Prefers `adj_close` over `close` when the column exists, and (only for
    `adj_close`) drops every row `adj_close_is_estimated` marks as an
    unadjusted-close alias. A file predating the adjusted schema entirely
    (a single `close`, no alias flag) falls back to the raw close, exactly
    as every caller of this hygiene already did before it was shared.
    """
    if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return None
    if "date" not in price_df.columns:
        return None

    column = "adj_close" if "adj_close" in price_df.columns else "close"
    if column not in price_df.columns:
        return None

    frame = pd.DataFrame({
        "date": pd.to_datetime(price_df["date"], errors="coerce"),
        "price": pd.to_numeric(price_df[column], errors="coerce"),
    })
    usable = frame["date"].notna() & frame["price"].notna()
    if column == "adj_close" and "adj_close_is_estimated" in price_df.columns:
        aliased = price_df["adj_close_is_estimated"].map(is_estimated)
        usable &= ~aliased.to_numpy(dtype=bool)

    frame = frame[usable].sort_values("date")
    return frame if not frame.empty else None


def bar_on_or_before(price_df, as_of) -> dict | None:
    """The last usable bar on or before `as_of`: `{"date": date, "price":
    float}`, or `None` if there isn't one.

    `price_df` may be a RAW per-ticker price frame (the `price_volume.csv`
    shape, carrying `close`/`adj_close`) or an already-cleaned frame (this
    module's own `clean_price_bars` output, carrying `price`). Detected by
    column name — a raw frame is never going to carry a `price` column of
    its own — rather than a second parameter, because a caller holding a
    frame it already cleaned and cached (the point of pairing this function
    with `clean_price_bars` at all: a caller doing hundreds of lookups
    against the same static frame should clean it once, not once per
    lookup) should not have to track and pass a second flag alongside it.
    The minor redundancy of re-running the `"price" in columns` check on an
    already-cleaned frame is the trade-off, and it is a column-name check,
    not a re-clean.

    `as_of` accepts anything `pandas.Timestamp` accepts (a `datetime.date`,
    a `pandas.Timestamp`, an ISO string) — this module takes no project
    dependency, so it cannot call the project's own `as_date`; callers that
    already hold a parsed date pass it straight through.
    """
    if isinstance(price_df, pd.DataFrame) and "price" in price_df.columns:
        frame = price_df
    else:
        frame = clean_price_bars(price_df)
    if frame is None or frame.empty:
        return None

    try:
        as_of_ts = pd.Timestamp(as_of)
    except (TypeError, ValueError):
        return None
    if pd.isna(as_of_ts):
        return None

    dates = frame["date"].dt.normalize()
    at_or_before = frame[dates <= as_of_ts]
    if at_or_before.empty:
        return None

    row = at_or_before.iloc[-1]
    return {"date": row["date"].date(), "price": float(row["price"])}
