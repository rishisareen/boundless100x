"""The capital ledger: modeled cash, per-tranche lots, mark-to-market (U4, KTD4).

Production's watchlist tracks no rupees (Phase 3's "no capital to take a
percentage of" boundary, `lifecycle/portfolio.py`). The simulator's CAGR,
drawdown and cash-drag outputs (§10, U6) cannot be produced without a real
cash pool, so this module is where KTD4 gives that boundary its one
exception: **modeled capital lives here, and only here.** Every dict this
module returns that could be mistaken for a real, actionable figure — a
tranche size, a mark, a settlement — carries `"basis": "modeled_capital"`
(`BASIS_MODELED_CAPITAL`), mirroring `lifecycle/portfolio.py`'s
`BASIS_COUNTS` discipline: a reader (including a future Phase 5 sweep
reading raw JSON) must never mistake a rupee-shaped simulator number for
either production's count-based guardrail or a real trade.

**State.** `cash` starts at `simulator.starting_pool` (`owner.config_from`,
default 100 "capital units" — unitless by construction, per `config.yaml`'s
own comment, so no simulated figure can be mistaken for a real rupee
amount). `positions` is `{ticker: [lot, ...]}`; a lot is exactly
`{qty, entry_bar_date, entry_price, lane, tranche_index}` (KTD4's own
words) — a core position built in thirds holds three lots with three
independent holding periods and, potentially, three different tax
brackets at exit. `tranche_index` is the count of prior lots for that
ticker at buy time (0, 1, 2, ...), and list order is chronological (oldest
first) by construction — every buy appends, nothing ever reorders — which
is what lets `sell`'s FIFO consumption walk the list front-to-back with no
separate sort.

**Tranche-notional sizing — a judgment call, stated explicitly.** The plan
names the input ("the sleeve's *deployable* share of pool-and-accrued-value
per the `portfolio:` config") but not a formula. This module computes:

    total_value    = cash + sum(qty * last_known_mark for every open lot)
    sleeve_target  = portfolio.sleeve_split[lane] * total_value
    lane_deployed  = sum(lot.qty * last_known_mark(lot.ticker)
                          for every open lot whose lane == lane)
    headroom       = sleeve_target - lane_deployed
    tranche        = min(portfolio.tranche_size_pct[lane] * sleeve_target,
                          headroom)

    refuse (no lot opened) when headroom <= 0.

Worked example at the shipped defaults (`starting_pool=100`,
`sleeve_split.core=0.7`, `tranche_size_pct.core=0.33`), an empty core
sleeve: `total_value=100`, `sleeve_target=70`, `lane_deployed=0`,
`headroom=70`, so the first tranche is `min(0.33*70, 70) = 23.1`. A second
tranche (prices unchanged) sees `lane_deployed=23.1`, `headroom=46.9`,
`tranche=min(23.1, 46.9)=23.1`; a third sees `headroom=23.8`,
`tranche=min(23.1, 23.8)=23.1`; a fourth is capped by headroom alone
(`~0.7`) and finishes the sleeve. Three-ish tranches of ~23.1 filling a
70-unit sleeve is what "a core position built in thirds" (`tranche_size_pct
= 0.33`) should mean.

**Why the `min(..., headroom)` cap, rather than the more literal reading
"`tranche_size_pct` of the *remaining* headroom every time."** That
literal reading was tried first and rejected: it produces exactly the
degenerate case the plan's own Approach section warns about — a shrinking
geometric series (23.1, 15.5, 10.4, 7.0, ...) that asymptotically
approaches the sleeve target but never actually finishes filling a
position, however many tranches are bought. `tranche_size_pct: 0.33` is
supposed to describe "three tranches, roughly equal, filling the sleeve" —
not "an infinite geometric tail." Capping a *constant* per-tranche target
(`tranche_size_pct * sleeve_target`) at whatever headroom remains keeps the
same self-limiting property the plan asks for (a tranche can never push the
sleeve over its target) while actually converging in about
`1 / tranche_size_pct` tranches, matching the "built in thirds / built in
halves" framing `portfolio.py`'s own `DEFAULT_TRANCHE_SIZE_PCT` comment
uses. Both readings reduce to the same number for the very first tranche in
an empty sleeve (`headroom == sleeve_target`, so the cap does not bind).

`lane_deployed` sums across *every ticker* in the lane, not just the one
being bought — this is deliberately the plan's own framing ("the sleeve's
deployable share"), a lane-wide budget, not a per-position one. A ticker's
own tranche count (`tranche_index`) is a record-keeping fact about that
position; it does not gate that position's own sizing.

**Mark tracking.** `_last_mark: {ticker: price}` is ledger state, updated
by `buy`, `sell` (the traded price is itself a fresh observation) and
`mark_to_market` (when a usable bar is found). It is what lets sizing use
"accrued value" for tickers not being traded on a given call, and what lets
a ticker with no usable bar on a given `mark_to_market` date still be
priced at *something* rather than dropped from the total.

**Bar-selection hygiene is duplicated locally, not imported.**
`lifecycle/friction.py`'s `_usable_bars` implements the same two
`price_volume.csv` hazards this module must also avoid (an
`adj_close_is_estimated` alias, a trailing-empty adjusted column) — but it
is underscore-prefixed, and `json_store.py`'s own module docstring names
exactly why that matters: reaching into another module's private helper
across a package boundary is "a note to the reader that says 'do not
depend on this' beside code that does." This is the data-cleaning idiom
`json_store.py`'s docstring calls out as *not* the KTD1 violation that
duplicating trigger/gate *evaluation* logic would be — "drop bad bars, pick
a direction" is a few lines, not a rule a production surface could
disagree with this module about. `_clean_price_bars` and
`_bar_on_or_before` below are that duplicate, used only by
`mark_to_market` (the one method here that takes a raw multi-ticker price
frame rather than an already-resolved `bar`). Keep them in sync with
`friction.py`'s hygiene if that logic ever changes.

**`buy`/`sell` take an already-resolved `bar` (`{"date", "price"}`), not a
price frame.** `mark_to_market` is the one method that resolves bars itself
— it is explicitly handed `price_frames` (plural, raw) because it must
answer the question for every open ticker at once, on a date nobody
committed to in advance. A `buy`/`sell` call targets one ticker at one
already-known decision point (later, U7's replay loop resolving the
confirmed trade's own bar via the same hygiene before calling in), so
threading a whole price frame through those two calls would just move the
same "which bar" question one layer down for no benefit. This is a
deliberate asymmetry with the plan's own Approach section, which names
`price_frames` only on `mark_to_market`.

**No rounding anywhere in this module**, mirroring `friction_cash`'s own
discipline and for the identical reason: the plan requires the equity
curve's point at any date to equal `cash + sum of every position's mark`
*exactly*, and rounding cash or marks here would make that summation
inexact for no reader's benefit — nothing in this module is rendered
directly (U6's job, downstream of the ledger).

**FIFO is the only lot-selection policy** (KTD4: "the only one that needs
no owner input" — India's de-facto accounting convention). `sell` walks
`positions[ticker]` front-to-back, consuming the oldest lot first, computing
each consumed chunk's `holding_days` from that *lot's own* `entry_bar_date`
to the exit bar's own date (not the caller's nominal sell date) —
mirroring `friction.compute_position_return`'s "measured between the bars,
not the requested dates" reasoning, because the bars are what actually
supplied the prices a tax bracket is chosen against.

This module never reimplements `friction_cash`'s tax/slippage arithmetic —
`cost_of_buy`/`settle_sale` are the only place a rate is applied, this
module only decides *how much* notional/quantity moves and *when*.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd

from boundless100x.lifecycle import portfolio
from boundless100x.lifecycle.states import as_date
from boundless100x.simulator import friction_cash, owner

logger = logging.getLogger(__name__)

# Stated on every reading that could be mistaken for a real, actionable
# figure — mirrors `portfolio.py`'s `BASIS_COUNTS` idiom (see module
# docstring). Not a shared constant with that module: the two bases mean
# different things and must never compare equal by accident.
BASIS_MODELED_CAPITAL = "modeled_capital"

# Floating-point dust guard for lot *quantities* only (never applied to a
# cash figure — see the module docstring's no-rounding rule). Consuming a
# lot down to 1e-10 units left over is a float artifact, not a real residual
# holding, and left alone it would linger in `positions` forever as a lot
# too small for any future sell to round-trip cleanly.
_QTY_EPSILON = 1e-9


# ── local bar-selection hygiene (mark_to_market only — see module docstring) ──


def _is_estimated(value) -> bool:
    """Local duplicate of `friction._is_estimated` — identical three-shape read."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _clean_price_bars(price_df) -> tuple[pd.DataFrame | None, str]:
    """The subset of `friction._usable_bars` this module needs: drop bad
    bars, keep the rest sorted by date. No direction is chosen here — that
    is `_bar_on_or_before`'s job — because a future caller inside this
    module might one day need the other direction and re-deriving the
    cleaning step from scratch is exactly the drift this duplication risks
    if the two are not kept in one place locally.
    """
    if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return None, "no price series is available for this ticker"
    if "date" not in price_df.columns:
        return None, "the price series carries no date column"

    column = "adj_close" if "adj_close" in price_df.columns else "close"
    if column not in price_df.columns:
        return None, "the price series carries no close column"

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
        return None, (
            f"no usable {column} bars — every bar is empty or an "
            f"unadjusted-close alias (adj_close_is_estimated)"
        )
    return frame, ""


def _bar_on_or_before(price_df, as_of: date) -> float | None:
    """The last usable close on or before `as_of`, or `None` if there isn't
    one — the exit/mark-to-market-style direction `friction.py`'s
    `compute_position_return` documents for its own exit bar.
    """
    frame, _reason = _clean_price_bars(price_df)
    if frame is None:
        return None
    dates = frame["date"].dt.normalize()
    at_or_before = frame[dates <= pd.Timestamp(as_of)]
    if at_or_before.empty:
        return None
    return float(at_or_before.iloc[-1]["price"])


# ── bar contract for buy/sell ──────────────────────────────────────────


def _bar_price(bar: dict) -> float:
    if not isinstance(bar, dict) or "price" not in bar:
        raise ValueError(
            f"Ledger: bar {bar!r} must be a dict carrying a 'price' key — the "
            f"caller resolves which trading bar prices this trade (the same "
            f"bar-selection hygiene mark_to_market uses) before calling in"
        )
    price = bar["price"]
    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(price)
        or price <= 0
    ):
        raise ValueError(f"Ledger: bar price {price!r} is not a positive finite number")
    return float(price)


def _bar_date(bar: dict) -> date:
    if not isinstance(bar, dict) or "date" not in bar:
        raise ValueError(f"Ledger: bar {bar!r} must be a dict carrying a 'date' key")
    parsed = as_date(bar["date"])
    if parsed is None:
        raise ValueError(f"Ledger: bar date {bar['date']!r} could not be read")
    return parsed


class Ledger:
    """Modeled cash + per-tranche lots for one simulated replay run.

    `config` is the **whole pipeline config** dict (or `None` for shipped
    defaults) — never a single block. This is deliberate and differs from
    `friction.config_from`/`portfolio.config_from`/`owner.config_from`'s own
    "either the whole config or just my block" idiom: this class needs all
    three of those blocks (`simulator:` for the starting pool, `portfolio:`
    for sleeve/tranche sizing, `friction:` for slippage/tax) at once, and
    passing only one block through would make the other two read as
    entirely absent rather than as the owner's actual settings. Store the
    whole config once at construction and every method call is free to
    still take its own `config` override for a call that genuinely needs a
    different reading (a Phase 5 sweep varying one parameter mid-run); when
    a call's `config` is left as `None`, the ledger's own stored config is
    used, not the shipped defaults.
    """

    def __init__(self, config: dict | None = None, starting_pool: float | None = None):
        self._config = config or {}
        self.cash: float = (
            float(starting_pool)
            if starting_pool is not None
            else float(owner.config_from(config)["starting_pool"])
        )
        self.positions: dict[str, list[dict]] = {}
        # Last observed price per ticker — from a buy, a sell, or a
        # mark_to_market bar that was actually found. Persists across calls
        # so a ticker with no usable bar on one date is still priced (at its
        # last known mark) on every subsequent one, per the plan's own words.
        self._last_mark: dict[str, float] = {}

    # ── internal readings used by sizing/mark-to-market ────────────────

    def _resolve_config(self, config: dict | None) -> dict | None:
        return config if config is not None else self._config

    def _position_qty(self, ticker: str) -> float:
        return sum(lot["qty"] for lot in self.positions.get(ticker, []))

    def _mark_or_cost(self, ticker: str) -> float:
        """The best price this ledger currently knows for `ticker`: its last
        observed mark, or (only if the ticker has never once been priced,
        which should not happen for any ticker holding a lot) the
        quantity-weighted average of its lots' own entry prices.
        """
        mark = self._last_mark.get(ticker)
        if mark is not None:
            return mark
        lots = self.positions.get(ticker) or []
        qty = sum(lot["qty"] for lot in lots)
        if qty <= 0:
            return 0.0
        return sum(lot["qty"] * lot["entry_price"] for lot in lots) / qty

    def _total_positions_value(self) -> float:
        return sum(
            self._position_qty(ticker) * self._mark_or_cost(ticker)
            for ticker in self.positions
            if self._position_qty(ticker) > 0
        )

    def _lane_deployed_value(self, lane: str) -> float:
        total = 0.0
        for ticker, lots in self.positions.items():
            price = self._mark_or_cost(ticker)
            total += sum(lot["qty"] * price for lot in lots if lot["lane"] == lane)
        return total

    def _tranche_notional(self, lane: str, config: dict | None) -> tuple[float | None, str]:
        """See the module docstring's "Tranche-notional sizing" section for
        the formula and the worked example behind it.
        """
        settings = portfolio.config_from(config)
        sleeve_split = settings["sleeve_split"]
        tranche_pct = settings["tranche_size_pct"]

        if lane not in sleeve_split:
            return None, (
                f"lane {lane!r} has no configured portfolio.sleeve_split "
                f"entry — cannot size a tranche without knowing the sleeve's "
                f"target share"
            )
        if lane not in tranche_pct:
            return None, (
                f"lane {lane!r} has no configured portfolio.tranche_size_pct "
                f"entry — cannot size a tranche without knowing its fraction "
                f"of the sleeve"
            )

        total_value = self.cash + self._total_positions_value()
        sleeve_target = sleeve_split[lane] * total_value
        deployed = self._lane_deployed_value(lane)
        headroom = sleeve_target - deployed
        if headroom <= 0:
            return None, (
                f"the {lane} sleeve is already at or above its target modeled "
                f"allocation ({deployed!r} of {sleeve_target!r} modeled "
                f"capital units) — no headroom for another tranche"
            )

        target_tranche = tranche_pct[lane] * sleeve_target
        return min(target_tranche, headroom), ""

    # ── public operations ───────────────────────────────────────────────

    def buy(self, ticker: str, lane: str, bar: dict, config: dict | None = None) -> dict:
        """Size and open (or add a tranche to) a position. Never partially
        mutates state: a refusal (no headroom, or insufficient cash) leaves
        `cash` and `positions` exactly as they were.

        Returns a dict always carrying `filled: bool` and
        `basis: "modeled_capital"`; a refusal carries `reason` instead of
        the trade fields.
        """
        effective_config = self._resolve_config(config)
        price = _bar_price(bar)
        entry_date = _bar_date(bar)

        notional, reason = self._tranche_notional(lane, effective_config)
        if notional is None or notional <= 0:
            logger.info(f"Ledger.buy: {ticker} ({lane}) refused — {reason}")
            return {
                "filled": False,
                "ticker": ticker,
                "lane": lane,
                "reason": reason,
                "basis": BASIS_MODELED_CAPITAL,
            }

        slippage = friction_cash.cost_of_buy(notional, effective_config)
        total_cost = notional + slippage
        if total_cost > self.cash:
            reason = (
                f"insufficient modeled cash: needs {total_cost!r} (notional "
                f"{notional!r} + entry slippage {slippage!r}) against "
                f"{self.cash!r} cash on hand"
            )
            logger.info(f"Ledger.buy: {ticker} ({lane}) refused — {reason}")
            return {
                "filled": False,
                "ticker": ticker,
                "lane": lane,
                "reason": reason,
                "basis": BASIS_MODELED_CAPITAL,
            }

        qty = notional / price
        tranche_index = len(self.positions.get(ticker, []))
        lot = {
            "qty": qty,
            "entry_bar_date": entry_date,
            "entry_price": price,
            "lane": lane,
            "tranche_index": tranche_index,
        }
        self.positions.setdefault(ticker, []).append(lot)
        self.cash -= total_cost
        self._last_mark[ticker] = price

        return {
            "filled": True,
            "ticker": ticker,
            "lane": lane,
            "tranche_index": tranche_index,
            "notional": notional,
            "slippage": slippage,
            "qty": qty,
            "price": price,
            "entry_bar_date": str(entry_date),
            "cash_after": self.cash,
            "basis": BASIS_MODELED_CAPITAL,
        }

    def sell(
        self, ticker: str, fraction: float, bar: dict, reason: str, config: dict | None = None
    ) -> list[dict]:
        """Close `fraction` of `ticker`'s total held quantity, FIFO across
        its lots. Returns one settlement dict per lot (or partial lot)
        touched, each carrying the caller's `reason` and
        `basis: "modeled_capital"`; an empty list means there was nothing to
        sell (no open lots for `ticker`).

        `fraction` must be in `(0, 1]`. `1.0` fully closes the position —
        every lot, entirely consumed, and the ticker's entry is removed from
        `positions`. A smaller fraction is the seam Phase 5's `"reduce"`
        severity will use once decision 4 settles a fraction (U7's job, not
        this module's) — handled FIFO-consistently here regardless of who
        calls it.
        """
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not (0 < fraction <= 1.0):
            raise ValueError(f"Ledger.sell: fraction must be in (0, 1], got {fraction!r}")

        effective_config = self._resolve_config(config)
        lots = self.positions.get(ticker) or []
        total_qty = sum(lot["qty"] for lot in lots)
        if total_qty <= 0:
            logger.info(f"Ledger.sell: {ticker} has no open lots — nothing to sell")
            return []

        price = _bar_price(bar)
        exit_bar_date = _bar_date(bar)
        qty_to_close = fraction * total_qty

        settlements: list[dict] = []
        survivors: list[dict] = []
        remaining = qty_to_close

        for lot in lots:
            if remaining <= _QTY_EPSILON:
                survivors.append(lot)
                continue

            consume = min(lot["qty"], remaining)
            entry_bar_date = as_date(lot["entry_bar_date"])
            # Measured bar-to-bar, not against the caller's nominal sell
            # date — see the module docstring's FIFO section.
            holding_days = (exit_bar_date - entry_bar_date).days

            settled = friction_cash.settle_sale(
                {"qty": consume, "entry_price": lot["entry_price"], "holding_days": holding_days},
                price,
                effective_config,
            )
            settled = {
                **settled,
                "ticker": ticker,
                "lane": lot["lane"],
                "tranche_index": lot["tranche_index"],
                "entry_bar_date": str(entry_bar_date),
                "exit_bar_date": str(exit_bar_date),
                "reason": reason,
                "basis": BASIS_MODELED_CAPITAL,
            }
            settlements.append(settled)
            self.cash += settled["proceeds"]
            remaining -= consume

            leftover = lot["qty"] - consume
            if leftover > _QTY_EPSILON:
                survivors.append({**lot, "qty": leftover})

        if survivors:
            self.positions[ticker] = survivors
        else:
            self.positions.pop(ticker, None)

        self._last_mark[ticker] = price
        return settlements

    def mark_to_market(self, as_of, price_frames: dict[str, pd.DataFrame] | None) -> dict:
        """Portfolio value as of `as_of`: every open position's mark plus
        cash.

        For each ticker with open lots, looks up the last usable bar on or
        before `as_of` in `price_frames[ticker]`. A ticker with no usable
        bar on or before the date (missing from `price_frames`, or every bar
        after/on the cutoff is unreadable) is carried at its last known mark
        (`_last_mark`, ledger state persisting across calls) and listed in
        `stale_marks` — never dropped from the total, per the plan's own
        words. A ticker that has *never* been priced at all (should not
        happen — `buy` always sets a mark before a lot can exist) falls back
        to its lots' cost basis rather than being silently excluded.

        Returns `{date, cash, positions_value, total_value, marks,
        stale_marks, basis}`. `total_value == cash + positions_value`
        exactly — nothing here is rounded (module docstring).
        """
        resolved_date = as_date(as_of)
        if resolved_date is None:
            raise ValueError(f"Ledger.mark_to_market: as_of {as_of!r} could not be read")

        price_frames = price_frames or {}
        marks: dict[str, float] = {}
        stale_marks: list[str] = []
        positions_value = 0.0

        for ticker, lots in self.positions.items():
            total_qty = sum(lot["qty"] for lot in lots)
            if total_qty <= 0:
                continue

            price = _bar_on_or_before(price_frames.get(ticker), resolved_date)
            if price is not None:
                self._last_mark[ticker] = price
            else:
                stale_marks.append(ticker)
                price = self._mark_or_cost(ticker)
                self._last_mark[ticker] = price

            marks[ticker] = price
            positions_value += total_qty * price

        return {
            "date": str(resolved_date),
            "cash": self.cash,
            "positions_value": positions_value,
            "total_value": self.cash + positions_value,
            "marks": marks,
            "stale_marks": stale_marks,
            "basis": BASIS_MODELED_CAPITAL,
        }
