"""Friction: what a modeled position keeps after tax and slippage.

v05 §8.2 asks for a position's return net of the two costs that actually
separate a backtest from a brokerage statement — capital-gains tax and the
spread paid getting in and out. Its own framing ("6–10 points more per cycle")
is an annualized-return comparison, and the watchlist tracks no invested
amount, so this is a **return-percentage transform, not a cash-flow ledger**.
Nothing here holds rupees, because nothing upstream ever recorded any.

**Every figure this module produces is a modeled estimate, and the language is
part of the contract** (KTD7). The holding period runs from a `probe`
confirmation date, not a fill. The prices are market bars, not trade prices.
There is no cost basis anywhere in the system. So a reader must never meet a
number from here that reads as a statement about trades that happened — the
word for that kind of return appears nowhere a reader could see it, and a test
enforces that against every string literal in this file.

Each reading therefore carries a `basis`:

  ``estimate``  an in-flight reading — the exit date is still moving
  ``recorded``  taken at a confirmed exit, so the dates are fixed

`recorded` means the dates stopped moving, not that the figure stopped being a
model.

Two conventions are shared with the rest of the lifecycle layer. Settings are
owner config rather than literals, exactly as §8.2 instructs — the tax regime
here is India's as of 2026 and will change without asking this codebase. And an
unreadable input is **unavailable with its reason**, never a zero: a zero
return means the position went nowhere, no return means nobody could tell, and
in a table those look identical.
"""

import logging
import math
from datetime import date, datetime

import pandas as pd

logger = logging.getLogger(__name__)

# India's regime as of 2026, and STARTING POINTS: equity STCG at 20%, LTCG at
# 12.5% beyond a ~365-day holding, plus a round-trip slippage allowance. These
# mirror `config.yaml`'s `friction:` block so a caller that supplies no config
# (a test, a future simulator) computes with the same numbers the CLI does.
# A statute change is a config edit, never a code change.
DEFAULT_STCG_PCT = 20.0
DEFAULT_LTCG_PCT = 12.5
DEFAULT_LTCG_HOLDING_DAYS = 365
DEFAULT_SLIPPAGE_BPS = 100

# Round trip: entry *and* exit, which is how slippage is actually paid. Split
# into two half-legs it would be the same number; stated once it cannot drift.
_BPS_PER_PCT = 100.0

BASIS_ESTIMATE = "estimate"
BASIS_RECORDED = "recorded"

# Percentages are carried to four places. The inputs are proxies, so more
# precision would be false confidence and less would make a hand-checked
# arithmetic test unreproducible.
_PLACES = 4


def config_from(config: dict | None) -> dict:
    """Owner settings for the friction model, with the shipped defaults.

    Accepts either the whole pipeline config (the `pace.config_from` idiom,
    `config_from(service.config)`) or the `friction:` block on its own, because
    both call sites are natural and a caller who passes the wrong one would
    otherwise get silent defaults — a wrong *tax rate* presented as the owner's
    own setting, which is exactly the failure this file cannot afford.
    """
    config = config or {}
    section = config.get("friction") if "friction" in config else config
    section = section or {}
    return {
        "stcg_pct": section.get("stcg_pct", DEFAULT_STCG_PCT),
        "ltcg_pct": section.get("ltcg_pct", DEFAULT_LTCG_PCT),
        "ltcg_holding_days": section.get(
            "ltcg_holding_days", DEFAULT_LTCG_HOLDING_DAYS
        ),
        "slippage_bps": section.get("slippage_bps", DEFAULT_SLIPPAGE_BPS),
    }


def unavailable(reason: str) -> dict:
    """The one shape for "this could not be read, and here is why".

    Shared by both halves so a renderer never has to ask which kind of gap it
    is holding — it asks `available`, and if the answer is False there is
    always a reason to show.
    """
    return {"available": False, "reason": reason}


def _as_date(value) -> date | None:
    """A calendar date from whatever the caller or the store happened to hold.

    The same two formats `lifecycle.evaluator._as_date` reconciles: `as_of` is
    a `date`, while a `state_history` record's `at` is a full ISO datetime
    because `watchlist._now()` writes `datetime.now()`. Timestamps normalize to
    dates here because a market bar has no time of day, and pretending
    otherwise would put a spurious few hours into a holding period that decides
    a tax bracket.
    """
    if isinstance(value, datetime):  # checked first — datetime subclasses date
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def _is_finite_number(value) -> bool:
    """A real number this module can compute with.

    `bool` is excluded because it is an `int` in Python and `True` as a tax
    rate would compute silently as 1%. NaN and infinity are excluded because
    every comparison against NaN is False, which is how a non-finite reading
    slips past a threshold check and comes out the far side as a figure.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _number(value, default, name: str):
    """A finite number, or the shipped default with a warning.

    A malformed setting must not silently become a *different* tax rate, and it
    must not take down an advance run either. Falling back loudly is the
    middle: the reading still states the rate it actually applied, so the
    number a reader sees is always the number that was used.
    """
    if not _is_finite_number(value):
        logger.warning(
            f"Friction: {name} {value!r} is not a finite number — using {default}"
        )
        return default
    return value


def _is_estimated(value) -> bool:
    """Whether a bar's adjusted close is an alias for its raw close.

    `adj_close_is_estimated` reaches this module through a CSV round trip on
    one path and a DataFrame on another, so it arrives as a bool, as the string
    "True", or as nothing at all. A missing value is not a claim that the bar
    is estimated — the flag postdates part of the corpus — so it reads False.
    """
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _usable_bars(price_df) -> tuple[pd.DataFrame | None, str, str]:
    """Bars a position may be priced against, and which column priced them.

    Two documented `price_volume.csv` hazards are dropped here, **before** any
    bar is selected rather than after:

      * `adj_close_is_estimated` marks a jugaad-data fetch where `adj_close` is
        just `close` under another name. A modeled return read off an
        unadjusted series would report a 1:5 split as an 80% loss.
      * The adjusted column trails the raw one by a bar — the source publishes
        today's close before today's adjusted close — so a freshly fetched
        series routinely ends in an empty `adj_close`. Selecting the last row
        unconditionally turns that into a total loss.

    Row-wise rather than whole-series, unlike the backtest's refusal: a series
    that is *wholly* aliased ends up with no usable bars and reads unavailable
    anyway, so this is the same guarantee with the partial case handled.

    A file predating the adjusted schema entirely (a single `close`, no alias
    flag) falls back to the raw close, exactly as it did before.
    """
    if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return None, "", "no price series is available for this position"
    if "date" not in price_df.columns:
        return None, "", "the price series carries no date column"

    column = "adj_close" if "adj_close" in price_df.columns else "close"
    if column not in price_df.columns:
        return None, "", "the price series carries no close column"

    frame = pd.DataFrame({
        "date": pd.to_datetime(price_df["date"], errors="coerce"),
        "price": pd.to_numeric(price_df[column], errors="coerce"),
    })
    usable = frame["date"].notna() & frame["price"].notna()
    if column == "adj_close" and "adj_close_is_estimated" in price_df.columns:
        aliased = price_df["adj_close_is_estimated"].map(_is_estimated)
        usable &= ~aliased.to_numpy(dtype=bool)

    frame = frame[usable].sort_values("date")
    if frame.empty:
        return None, column, (
            f"no usable {column} bars — every bar is empty or an unadjusted-close "
            f"alias (adj_close_is_estimated), which cannot price a position"
        )
    return frame, column, ""


def compute_position_return(price_df, entry_date, exit_date) -> dict:
    """A position's modeled gross return and holding period, off market bars.

    **Bar selection is specified, because lifecycle timestamps rarely land on
    trading days.** A `probe` confirmation written on a Saturday has no
    Saturday bar to read. Each end rounds in the conservative direction:

      * the **entry** bar is the first usable bar *on or after* the entry date,
        because a confirmed buy cannot predate its own confirmation — rounding
        back would price the position at a bar from before it existed, and on a
        rising series that flatters the return;
      * the **exit** bar is the last usable bar *on or before* the exit date,
        because a position cannot be sold into a bar that has not printed.

    An empty range — an entry date past the end of the series, an exit before
    its start, or an exit bar that lands before the entry bar — is unavailable
    with its reason. Never a nearest-neighbour guess: the guess would be a
    number, and a number is indistinguishable from a measurement once it is in
    a table.

    Returns `{available: True, gross_return_pct, holding_days, ...}` or
    `{available: False, reason}`. Nothing here is a statement about trades that
    happened; see the module docstring.
    """
    entry = _as_date(entry_date)
    if entry is None:
        return unavailable(f"entry date {entry_date!r} could not be read")
    settled = _as_date(exit_date)
    if settled is None:
        return unavailable(f"exit date {exit_date!r} could not be read")

    frame, column, reason = _usable_bars(price_df)
    if frame is None:
        return unavailable(reason)

    dates = frame["date"].dt.normalize()
    at_or_after = frame[dates >= pd.Timestamp(entry)]
    if at_or_after.empty:
        last = frame["date"].iloc[-1].date()
        return unavailable(
            f"no trading bar on or after the entry date {entry} — the usable "
            f"{column} series ends {last}"
        )

    at_or_before = frame[dates <= pd.Timestamp(settled)]
    if at_or_before.empty:
        first = frame["date"].iloc[0].date()
        return unavailable(
            f"no trading bar on or before the exit date {settled} — the usable "
            f"{column} series starts {first}"
        )

    entry_bar, exit_bar = at_or_after.iloc[0], at_or_before.iloc[-1]
    entry_bar_date = entry_bar["date"].date()
    exit_bar_date = exit_bar["date"].date()
    if exit_bar_date < entry_bar_date:
        return unavailable(
            f"no usable bars between {entry} and {settled} — the first bar on "
            f"or after entry ({entry_bar_date}) is later than the last bar on "
            f"or before exit ({exit_bar_date})"
        )

    start, end = float(entry_bar["price"]), float(exit_bar["price"])
    if start <= 0 or end <= 0:
        return unavailable(
            f"a non-positive {column} at a window endpoint ({start} → {end}) "
            f"cannot express a return"
        )

    return {
        "available": True,
        "gross_return_pct": round((end / start - 1) * 100, _PLACES),
        # Measured between the **bars**, not between the requested dates: the
        # bars are what supplied the prices, so a tax bracket decided on any
        # other span would not be the span the return was measured over. Both
        # pairs of dates travel in the reading so the rounding is inspectable.
        "holding_days": (exit_bar_date - entry_bar_date).days,
        "entry_date": str(entry_bar_date),
        "exit_date": str(exit_bar_date),
        "requested_entry_date": str(entry),
        "requested_exit_date": str(settled),
        "entry_price": round(start, _PLACES),
        "exit_price": round(end, _PLACES),
        "price_series": column,
    }


def compute_net_return(gross_return_pct, holding_days, config=None) -> dict:
    """Gross return less round-trip slippage, then capital-gains tax.

    **Order matters and matches how the costs are actually borne**: slippage is
    paid on the way in and on the way out whatever the position does, so it
    comes off the gross return first; tax then applies to what is left. Taxing
    first and deducting slippage after would tax a gain the owner never had.

    Slippage is a flat round-trip deduction in basis points — 100bps is one
    percentage point off the return — which is a transform on the return rather
    than a reconstruction of two fills, and that is the whole modeling claim
    being made.

    **A loss is not taxed.** That is realistic capital-gains behaviour rather
    than a floor bolted on: a position that lost money pays slippage and
    nothing else. It falls out of applying tax only to a positive post-slippage
    figure, so a gain that slippage wipes out is untaxed too, without a second
    rule.

    The bracket is chosen by `holding_days` against the configured
    `ltcg_holding_days` — at or beyond the line is long-term. The line is
    config because the regime is a statute, not a property of this system.
    """
    settings = config_from(config)
    stcg_pct = _number(settings["stcg_pct"], DEFAULT_STCG_PCT, "stcg_pct")
    ltcg_pct = _number(settings["ltcg_pct"], DEFAULT_LTCG_PCT, "ltcg_pct")
    slippage_bps = _number(
        settings["slippage_bps"], DEFAULT_SLIPPAGE_BPS, "slippage_bps"
    )
    ltcg_days = _number(
        settings["ltcg_holding_days"], DEFAULT_LTCG_HOLDING_DAYS, "ltcg_holding_days"
    )

    if not _is_finite_number(gross_return_pct):
        return unavailable(
            f"gross return {gross_return_pct!r} is not a finite number — there "
            f"is nothing to apply friction to"
        )
    if not _is_finite_number(holding_days):
        return unavailable(
            f"holding period {holding_days!r} is not a finite number — the tax "
            f"bracket cannot be chosen without it"
        )

    long_term = holding_days >= ltcg_days
    tax_pct = ltcg_pct if long_term else stcg_pct

    after_slippage = gross_return_pct - (slippage_bps / _BPS_PER_PCT)
    taxed = after_slippage > 0
    net = after_slippage * (1 - tax_pct / 100.0) if taxed else after_slippage

    return {
        "available": True,
        "gross_return_pct": round(float(gross_return_pct), _PLACES),
        "holding_days": int(holding_days),
        # Stated even on a loss. Which bracket it *would* have fallen in is
        # part of reading the estimate, and an absent regime looks like a gap
        # in the model rather than a deliberate zero-tax outcome.
        "tax_regime": "ltcg" if long_term else "stcg",
        "tax_pct": tax_pct,
        "taxed": taxed,
        "ltcg_holding_days": ltcg_days,
        "slippage_bps": slippage_bps,
        "after_slippage_pct": round(after_slippage, _PLACES),
        "net_return_pct": round(net, _PLACES),
    }


def model_exit(
    price_df, entry_date, exit_date, config=None, basis: str = BASIS_ESTIMATE
) -> dict:
    """One reading for an exit: gross beside net, with the basis it rests on.

    The two halves are separately testable and separately useful — a caller
    that already knows a return can price the friction on it alone — but a
    reader must never receive one without the other, so the composed reading is
    what every surface renders. `basis` is the caller's to state: `estimate`
    while an exit is only proposed, `recorded` once its dates are fixed.
    """
    position = compute_position_return(price_df, entry_date, exit_date)
    if not position["available"]:
        return {**position, "basis": basis}

    net = compute_net_return(
        position["gross_return_pct"], position["holding_days"], config
    )
    if not net["available"]:
        return {**net, "basis": basis}

    return {**position, **net, "basis": basis}


def describe(reading: dict | None) -> str:
    """One line a person can read, with the modeling claim attached.

    Gross and net always appear together (R5), and the line names itself an
    estimate every time it is rendered. A caption elsewhere on the screen is
    not enough: this string gets copied into evidence, logs and history, where
    the caption does not follow it.
    """
    if not reading:
        return "net of friction: no reading"
    if not reading.get("available"):
        reason = reading.get("reason", "no reason given")
        return f"net of friction: unavailable — {reason}"

    regime = str(reading.get("tax_regime", "")).upper()
    basis = reading.get("basis", BASIS_ESTIMATE)
    return (
        f"gross {reading['gross_return_pct']:+.1f}% / "
        f"net {reading['net_return_pct']:+.1f}% "
        f"(modeled {basis}: {regime} at {reading['tax_pct']}% + "
        f"{reading['slippage_bps']}bps round-trip slippage over "
        f"{reading['holding_days']} days)"
    )
