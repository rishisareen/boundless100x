"""Cash-level friction: the `friction:` regime applied to traded notional.

`boundless100x.lifecycle.friction` is documented as "a return-percentage
transform, not a cash-flow ledger" (its own module docstring) — the watchlist
tracks no invested amount, so it deducts basis points off a *return
percentage*. The simulator's ledger (U4, not yet built) holds actual cash, so
it needs the same regime expressed as a charge on *traded notional*: slippage
on each leg's cash amount, tax on a closing lot's post-slippage cash gain when
positive, bracket chosen by that lot's own bar-to-bar holding days. This
module is that arithmetic. U4 is the only caller; it supplies the `lot` shape
documented on `settle_sale` and receives cash figures back, plain floats and
a plain dict — no pandas, no dates, no dependency on anything but
`friction.config_from`.

**KTD5 — reuse the rates, not the transform** (plan lines 445-478). This
module calls `friction.config_from` for every rate — STCG/LTCG percentages,
the LTCG holding-day boundary, slippage bps — and never re-declares a
default. A `stcg_pct: 25` edit in `config.yaml`'s `friction:` block reaches
both this module and `friction.compute_net_return` with zero code changes
anywhere.

**The two models agree on regime and differ on level, by construction.**
Both share: the same rates, the same `ltcg_holding_days >= ` boundary
(long-term is *at or beyond* the line, not beyond it), slippage deducted
before tax, and a loss going untaxed. They cannot agree on *magnitude*:
`compute_net_return` deducts bps off a return percentage (a fixed 1.0
percentage point at the default 100bps, regardless of how large the return
is), while this module charges bps on notional, and the exit leg's notional
is the *grown* position — so the cash cost, read back as percentage points of
the original return, grows with the return. On a +100% gross return the
notional path costs 1.5pp against the transform's flat 1.0pp (worked out in
`tests/test_simulator_friction_cash.py`); the two converge only near a 0%
return, where the grown and original notionals are the same number. Asserting
the two ever produce the *same net figure* would be asserting one of them
stopped doing what it says it does — see the plan's Verification Contract,
"Friction regime consistency".

**`slippage_bps` is a round trip, so a single leg carries half of it.**
`config.yaml`'s own comment says so in capitals, and `friction.py` states the
same fact once, precisely, so it cannot drift: "Round trip: entry *and*
exit... Split into two half-legs it would be the same number." Charging the
full configured bps on both the buy and the sell would silently double the
regime the rest of the system runs under — the one arithmetic mistake this
module is built to make structurally impossible: `_leg_slippage_cost` is the
single formula both `cost_of_buy` and `settle_sale`'s exit leg call, so a
future change to the halving logic cannot fix one call site and miss the
other.

No figure in this module is rounded. `friction.py` rounds its percentages to
four places because they are read by a person; these are cash amounts a
ledger sums across many lots, and the plan requires the equity curve's final
point to equal cash plus marks *exactly* — rounding here would make that
summation inexact for no reader's benefit, since nothing in this module is
rendered directly (that is U6's job, downstream of the ledger).
"""

from __future__ import annotations

from boundless100x.lifecycle.friction import config_from

# The configured `slippage_bps` prices a ROUND TRIP — the entry leg and the
# exit leg together — not a single leg. A buy and the sell half of a settle
# each therefore charge HALF the configured figure. Named and commented once
# so `cost_of_buy` and `settle_sale` cannot drift apart on the halving.
_LEG_SHARE_OF_ROUND_TRIP = 0.5

# Basis points to a fraction: 100 bps == 1.00%.
_BPS_TO_FRACTION = 10_000.0

REGIME_LTCG = "ltcg"
REGIME_STCG = "stcg"


def _leg_slippage_cost(notional: float, slippage_bps: float) -> float:
    """Cash slippage for ONE leg (a buy, or the sell half of a settle).

    The single formula behind both public functions — see the module
    docstring's KTD5 section for why sharing it, rather than writing `/ 2` at
    two call sites, is the point.
    """
    return notional * (slippage_bps * _LEG_SHARE_OF_ROUND_TRIP) / _BPS_TO_FRACTION


def cost_of_buy(notional: float, config: dict | None = None) -> float:
    """The cash slippage cost of opening a position (the entry leg only).

    `notional` is the traded amount (`qty * price`) in whatever unit the
    caller's ledger uses — the simulator's `starting_pool` is unitless
    "capital units" (`config.yaml`'s `simulator:` block), so nothing here
    assumes rupees. `config` is whatever `friction.config_from` accepts: the
    whole pipeline config, the `friction:` block alone, or `None` for the
    shipped defaults.

    Returns the cash cost as a positive float — the caller deducts it from
    cash, this function does not touch a ledger.
    """
    settings = config_from(config)
    return _leg_slippage_cost(notional, settings["slippage_bps"])


def settle_sale(lot: dict, exit_price: float, config: dict | None = None) -> dict:
    """Close (all or part of) a position: exit-leg slippage, then tax, in cash.

    `lot` is a plain dict — the minimal, explicit contract U4 (the capital
    ledger, not yet built) will conform to when it calls this function. Three
    keys are read, and nothing else:

      ``qty``           units being sold. A partial close (fewer units than
                         the original lot) is valid — this function has no
                         notion of "the whole lot" and does no bookkeeping
                         beyond the units it is handed.
      ``entry_price``   the price the lot was bought at. The cost basis is
                         `qty * entry_price`; the *entry* leg's slippage was
                         already charged at `cost_of_buy` time when the lot
                         was opened, so it is never re-charged here — only
                         the exit leg is priced by this function.
      ``holding_days``  a plain `int`, already computed bar-to-bar by the
                         caller. U4's own Approach makes "per-lot holding
                         days" the ledger's job, derived from its own
                         price-frame bar selection — this function receives
                         the number and brackets on it, it does not compute
                         it from dates.

    Mirrors `friction.compute_net_return`'s two conventions, restated in cash
    rather than on a return percentage:

      * **slippage before tax** — the exit leg's slippage comes off gross
        proceeds (`qty * exit_price`) first, and tax applies only to what is
        left;
      * **a loss goes untaxed** — tax applies only to a *positive*
        post-slippage gain, so a gain that slippage wipes out to a loss is
        untaxed too, without a second rule. Slippage only ever subtracts, so
        the only sign flip possible is gain-to-loss, never the reverse.

    The tax bracket is `holding_days >= ltcg_holding_days` — at or beyond the
    line is long-term, the exact boundary `compute_net_return` uses (`>=`,
    never `>`).

    Returns a dict:

      ``proceeds``      net cash the ledger receives — gross proceeds, less
                         this leg's slippage, less tax (`0.0` tax on a loss).
      ``slippage``      this leg's cash slippage cost (always >= 0).
      ``tax``            cash tax paid; `0.0` on a loss.
      ``gain``           post-slippage proceeds less cost basis. May be
                         negative even when the price did not fall, because
                         exit-leg slippage alone can push a flat trade to a
                         small loss.
      ``regime``         `"ltcg"` or `"stcg"` — the same two literal strings
                         `compute_net_return` uses for its `tax_regime`
                         field, so a reader who knows one reads the other for
                         free.
      ``tax_pct``        the rate actually applied (`0` conceptually on a
                         loss, but the rate the bracket *would* have used is
                         still reported — mirrors `compute_net_return`,
                         which states its bracket even on a loss).
      ``taxed``          `bool` — whether `tax` is nonzero, mirroring
                         `compute_net_return`'s own `taxed` field.
      ``holding_days``, ``qty``, ``entry_price``, ``exit_price``
                          echoed back from the inputs, so the reading is
                         self-explanatory without re-deriving it from the
                         caller's lot.
    """
    settings = config_from(config)

    qty = lot["qty"]
    entry_price = lot["entry_price"]
    holding_days = lot["holding_days"]

    gross_proceeds = qty * exit_price
    slippage = _leg_slippage_cost(gross_proceeds, settings["slippage_bps"])
    post_slippage_proceeds = gross_proceeds - slippage

    cost_basis = qty * entry_price
    gain = post_slippage_proceeds - cost_basis

    long_term = holding_days >= settings["ltcg_holding_days"]
    regime = REGIME_LTCG if long_term else REGIME_STCG
    tax_pct = settings["ltcg_pct"] if long_term else settings["stcg_pct"]

    taxed = gain > 0
    tax = gain * (tax_pct / 100.0) if taxed else 0.0
    proceeds = post_slippage_proceeds - tax

    return {
        "proceeds": proceeds,
        "slippage": slippage,
        "tax": tax,
        "gain": gain,
        "regime": regime,
        "tax_pct": tax_pct,
        "taxed": taxed,
        "holding_days": holding_days,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }
