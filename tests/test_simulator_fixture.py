"""The plan's own mandated hand-computed two-name fixture (§12 Phase 4's
stated validation, plan lines 1134-1137 and 1152-1180's Verification
Contract): "two synthetic tickers with scripted fundamentals and prices, a
scripted expected equity curve (every tranche, tax line, and idle day
computed by hand), asserted to match **exactly**."

## Which of the two drivers this fixture uses, and why

The plan names two ways to drive the fixture: through `simulate()` end to
end, or through `replay.run_replay()` directly with hand-built
`stores`/`calendar_result`/`universe_result`/`assignments`/`engines`. This
file uses the second. `simulate()` would additionally exercise
`calendar.compute_calendar`'s own fiscal-grid derivation and
`universe.build_universe`'s own KTD8 candidacy scan — real machinery, but
machinery this fixture does not need to re-prove (both are covered by their
own test files) and machinery that would force the replay calendar to be
whatever the corpus's own quarterly grain produces rather than the four
specific dates this fixture needs to control precisely. Handing `run_replay`
a hand-built four-date calendar and a hand-built `{ticker: first_eligible_
date}` map keeps every other piece of the loop real (`ComputeEngine.run_all`,
`SQGLPScorer`, `EligibilityEvaluator`, `TriggerEvaluator`,
`advance.decide()`, `owner.decide()`/`owner.route()`, the real `Ledger`,
`friction_cash`, `lifecycle.exit.confirm_exit`) while making the four
replay dates and the two tickers' `raw_data/`-shaped fixtures the only two
things this file has to control.

## The one deliberate shortcut, and why it is safe

Both tickers are seeded directly into `watch` (via `watchlist.add()` +
two manual `watchlist.transition()` calls) rather than earning their way
there through `screen -> qualify -> watch`'s own composite-gated triggers
(`qualification_passed`/`awaiting_entry_price`, `score: composite >= 5.5`).
This is the fixture's one departure from "every transition earned through
`decide()`", and it is deliberate: those three pre-position transitions
**move no money** (`lifecycle.states.moves_money` — `AUTO_APPLICABLE`
excludes none of them from that set) and are not part of what this fixture
exists to prove, which is KTD4's claim about the *ledger* — "the equity
curve's final point equals cash + Σ marks exactly" — at every replay point.
Computing the full 51-metric SQGLP composite by hand to clear a `>= 5.5`
threshold would be substantial, disproportionate effort spent proving
something this fixture was never asked to prove (composite scoring has its
own extensive test coverage in `tests/test_scorer.py` and friends). The two
transitions this fixture's whole claim rests on — `watch -> probe`
(`valuation_buy_zone`) and `probe -> exit_review -> exited`
(`capital_efficiency_break` + `lifecycle.exit.confirm_exit`) — are both
driven for real, through the real `TriggerEvaluator` and the real
`advance.decide()`, against real computed metrics from the fixture's own
financials. So is every eligibility gate (`compute_engine.eligibility.
EligibilityEvaluator` against the shipped `DEFAULT_GATES`/`registry.yaml`
gates) and the KTD0 valuation rebuild (`point_in_time.py`) feeding them —
both fully hand-verified below, not skipped.

## Every hand-worked number, and how it was derived

**KTD0's rebuild** (`compute_engine/point_in_time.py::_rebuild_market_cap`/
`_rebuild_stock_pe`), at every replay date (`ALPHA`'s `equity_capital=100`,
`Face Value=10`, `raw_close` from the truncated price series' own last row):

    Market Cap = equity_capital / face_value * raw_close = (100/10) * raw_close
    Stock P/E  = raw_close / latest_annual_eps  (`valuation._current_multiple`)

At every replay date used below, `raw_close=100.0` except at D4 (`2021-12-03`,
`raw_close=150.0`, matching the settlement bar). `latest_annual_eps` is
FY2020's `eps=24` throughout (no later annual row is ever added), so
`Stock P/E = 100/24 = 4.1666...7` at D1-D3, `150/24 = 6.25` at D4's own
(pre-settlement) reading. `Market Cap = 10 * raw_close`: 1000 at D1-D3,
1500 at D4. The reconciliation guard's independent share count
(`pat/eps = 240/24 = 10`) matches `equity_capital/face_value = 100/10 = 10`
exactly (0% divergence, `_rebuild_market_cap`'s guard (b)) — deliberately
designed that way (`eps := pat/10` for every fiscal year, by construction
below) rather than merely falling inside the 5% tolerance. Guard (a) never
applies at any replay date used here: the raw price series extends to
2022-06-01, well past every cutoff, so `full_price["date"].max() <= cutoff`
is always False.

**The three `DEFAULT_GATES` eligibility gates** (`compute_engine/
eligibility.py`), evaluated at D1 (and unchanged through D3, since no new
annual `financials`/`balance_sheet` row is ever added):

  * `size`: `market_cap(1000) < 30000` -> True.
  * `price` (mode `any`, veto `reverse_dcf_overpriced` from
    `reverse_dcf_growth`): `trailing_peg < 2.0` is checked first (see
    below, ~0.246 -- True), so the gate passes regardless of `peg_ratio`;
    the veto source `reverse_dcf_growth` is available (computed, see
    below) and does not carry `reverse_dcf_overpriced`, so the veto does
    not disqualify.
  * `reinvestment`: `roiic(~42.86) >= 15.0` -> True.

  All three pass -> `verdict = "eligible"`. (`roiic`'s formula, `compute_
  engine/metrics/builtin/profitability.py::compute_roiic`: capital =
  equity_capital + reserves + borrowings per row; nopat = operating_profit
  * (1 - 0.25) [no `tax_pct` column supplied, so the function's own 25%
  default applies]; 5 annual rows visible at D1 is below `SMOOTHING_MIN_
  POINTS=6`, so `smoothed_endpoints` returns *plain* endpoints — no
  averaging — `cap_start=capital[FY2016]=500`, `cap_end=capital[FY2020]=780`,
  `nopat_start=nopat[FY2016]=120`, `nopat_end=nopat[FY2020]=240`. `roiic =
  (240-120)/(780-500)*100 = 120/280*100 = 42.857142857142854`.)

**`valuation_buy_zone`'s three conditions** (`lifecycle/triggers.yaml`),
also at D1, mode `all`:

  * `pe_vs_historical <= 60`: the percentile of `Stock P/E` (4.1666...7)
    among the five historical year-end P/Es `[100/12, 100/15, 100/18,
    100/21, 100/24] = [8.333, 6.667, 5.556, 4.762, 4.167]` (year-end close
    is flat 100.0 for every one of the five FY-ends, `valuation.
    compute_pe_percentile`'s own `close_on_or_before / eps` construction).
    Exactly one historical value (FY2020's own, computed identically) is
    `<= 4.1666...7`, so `percentile = 1/5*100 = 20.0 <= 60` -> True.
  * `trailing_peg <= 2.0`: `valuation.compute_trailing_peg` on the last 4
    annual `pat` rows `[150, 180, 210, 240]` (FY2017-FY2020):
    `pat_cagr = ((240/150)**(1/3) - 1) * 100 = 16.9607...%`;
    `trailing_peg = 4.1666.../16.9607... = 0.24566...` -> True.
  * `flag_absent reverse_dcf_overpriced` (source `reverse_dcf_growth`):
    `valuation.compute_reverse_dcf` on `avg_fcf = mean([100,110,120,130,
    140]) = 120` (no MAD outliers — the median-absolute-deviation check
    finds every year within 2 std of the median) against `mcap=1000`; the
    50-iteration bisection search (`discount_rate=0.12`,
    `terminal_growth=0.04`, `projection_years=10`, bounded [-10%, 50%])
    converges to `implied_growth ~= -2.428%` — a *cheap* reading (a modest
    FCF-generator priced at a low multiple needs little growth to justify
    it), well under `1.5 * actual_cagr` (`actual_cagr` = 5-year revenue
    CAGR `[400..800] = 18.92%`, `1.5x = 28.38%`) so `reverse_dcf_
    overpriced` is never carried (the metric instead carries `reverse_dcf_
    underpriced`, which this gate/trigger does not read). All three
    numbers were independently derived by re-implementing `compute_
    reverse_dcf`'s own documented bisection loop against these inputs
    (not by calling the real function) and cross-checked against the real
    `Ledger` class's own output for the ledger arithmetic below (see the
    next section) — the reverse-DCF figure itself is not part of the
    equity curve's exact-match claim, only its *boolean* absence-of-flag
    outcome is, which a wide margin like -2.4% vs a 28.4% threshold does
    not leave in doubt.

  All three True (mode `all`) -> `valuation_buy_zone` fires, proposing
  `watch -> probe`.

**Why `capital_efficiency_break` is the chosen kill switch, and why it
alone fires.** `ALPHA`'s `ratios.csv` carries RoCE `[25, 25, 25, 5]` for
FY2016-FY2019 at D1/D2/D3's *own* candidacy for `fundamentals_deteriorated`
(only `qualify`/`watch` states ask that question; `ALPHA` is `watch` only
through D1-D2's evaluation) — the last two visible entries at that point
are `[FY2018=25, FY2019=5]`, not both `< 15`, so the `persist_years: 2`
series check does not fire and `ALPHA` is never dropped before its buy. A
sixth row, `FY2021 = 5` (period end 2021-03-31, six-month reporting lag to
2021-09-30), is invisible until D3 (`2021-12-01 >= 2021-09-30`) — at which
point the visible window becomes all five rows `[FY2016=25, FY2017=25,
FY2018=25, FY2019=5, FY2021=5]` (`compute_roce_avg`'s own `_get_annual_
rows` has no notion of a missing FY2020 row — it reads whatever annual
rows exist by their own labels) and the last two entries `[FY2019=5,
FY2021=5]` are *both* `< 15`, firing `capital_efficiency_break` (`from:
[probe, scale]`, universal, no `lane` key) the first date it is readable.
Every other kill switch is independently confirmed never to fire at D3/D4:
`incremental_return_break` (`roiic` unchanged at ~42.86%, well above 12),
`growth_quality_degradation` (`growth_quality_risky` is *structurally*
impossible here — `eps := operating_profit * 0.075` for every fiscal year
by construction, so `%delta EPS == %delta operating_profit` exactly every
year and `_mean_yoy_ratio(eps, op)`'s financial-leverage ratio is exactly
1.0, under the `>= 1.3` threshold `_grade_growth_quality` requires before
"Financial leverage" is even a driver, and "risky" requires that driver
*alone*), `valuation_saturation` (`pe_vs_historical` stays 20 through D3,
nowhere near `> 95`), `governance_event` (`promoter_pledge`'s only data
path reads `shareholding_bse`, which `simulator.replay._SIMULATOR_NON_
TRUNCATABLE_INPUTS` always strips — this metric is *always* unavailable in
the simulator, so this condition is *always* indeterminate, never fired,
for any simulator fixture), `checkpoints_failed` (no checkpoints are ever
recorded). `BETA` carries no `ratios.csv` at all, so `roce_5yr_avg` is
"missing input" for it at every date — indeterminate, never fired — and it
is never touched again after its own buy.

**Trading-day arithmetic** (`owner.py::_advance_trading_days`, `pandas.
bdate_range`, Mon-Fri, no holiday calendar): `D1 = 2020-10-01` (a
Thursday). `D2 = D1 + 5 trading days = 2020-10-08` (entry lag, `owner.
DEFAULT_CONFIRMATION_LAG_DAYS["entry"] = 5`, the shipped default — no
`simulator.*` override is used anywhere in this fixture; `config={}` is
handed to every `config_from` in this system precisely so every reading is
its own shipped default, verified in `config.yaml` and re-derived here
independently rather than assumed). `D3 = 2021-12-01` (a Wednesday, chosen
so `ratios.csv`'s sixth `FY2021` row is already readable —
`2021-12-01 >= 2021-09-30`). `D4 = D3 + 2 trading days = 2021-12-03` (exit
lag, `owner.DEFAULT_CONFIRMATION_LAG_DAYS["exit"] = 2`). Holding period for
the tax bracket is measured bar-to-bar (`Ledger.sell`'s own rule, entry
lot's `entry_bar_date` to the exit bar's own date, never the caller's
nominal dates): `(2021-12-03 - 2020-10-08).days = 421 >= friction.
DEFAULT_LTCG_HOLDING_DAYS (365)` -> LTCG, `12.5%`, chosen deliberately far
from the 365-day boundary (421 vs 365, a 56-day margin) so the exact
trading-day count is never in doubt.

**The tranche-notional formula** (`ledger.py::Ledger._tranche_notional`,
quoted from its own module docstring):

    total_value    = cash + sum(qty * last_known_mark for every open lot)
    sleeve_target  = portfolio.sleeve_split[lane] * total_value
    lane_deployed  = sum(lot.qty * last_known_mark(lot.ticker)
                          for every open lot whose lane == lane)
    headroom       = sleeve_target - lane_deployed
    tranche        = min(portfolio.tranche_size_pct[lane] * sleeve_target,
                          headroom)

No override touches `portfolio:` here, so `sleeve_split.core=0.7`,
`tranche_size_pct.core=0.33` (`config.yaml`'s own shipped defaults — the
exact numbers `ledger.py`'s own module docstring already works through for
an empty sleeve's first tranche: "the first tranche is `min(0.33*70, 70) =
23.1`"). `ALPHA` buys first (alphabetical settlement order, both entries
scheduled from the same D1 proposal, confirmed on the same D2): `cash=100`,
no open lots -> `total_value=100`, `sleeve_target=0.7*100=70`,
`lane_deployed=0`, `headroom=70`, `tranche=min(0.33*70, 70)=23.1`.
`slippage = friction_cash.cost_of_buy(23.1, {}) = 23.1 * (100*0.5)/10000 =
0.1155` (`friction.DEFAULT_SLIPPAGE_BPS=100`, halved per leg — KTD5).
`total_cost=23.2155`, `qty=23.1/100=0.231`, `cash_after=76.7845`.

`BETA` buys second, from the *already-updated* ledger: `total_value =
76.7845 + (0.231*100) = 99.8845` (positions_value uses `ALPHA`'s own
just-set mark, 100.0). `sleeve_target = 0.7*99.8845 = 69.91915`.
`lane_deployed = 23.1` (`ALPHA`'s own value, same lane). `headroom =
69.91915 - 23.1 = 46.81915`. `tranche = min(0.33*69.91915, 46.81915) =
min(23.0733195, 46.81915) = 23.0733195`. `slippage = 23.0733195*0.005 =
0.115366...`. `qty = 23.0733195/100 = 0.230733195`. `cash_after =
53.5958139025`.

These two tranche notionals and every downstream figure below were
independently re-derived by re-implementing `Ledger.buy`/`_tranche_
notional`/`mark_to_market` and `friction_cash.cost_of_buy`/`settle_sale`
formula-for-formula against these inputs, then cross-checked line-for-line
against the **real** `Ledger`/`friction_cash` classes from this repository
(not merely restated from them) before being written into this file as
literal expected values — every float below is bit-identical to what that
cross-check produced, which is why no `pytest.approx` is used anywhere in
this file (see "On floating-point exactness" below).

**The equity curve, at every one of the four replay points:**

    D1 (2020-10-01, before any settlement — only a proposal exists):
        cash=100.0, positions_value=0.0, total_value=100.0
    D2 (2020-10-08, both tranches settle, in order ALPHA then BETA):
        cash=53.5958139025, positions_value=46.173319500000005,
        total_value=99.7691334025
    D3 (2021-12-01, the kill switch proposes exit_review; nothing settles
        yet — prices and positions are unchanged from D2):
        cash=53.5958139025, positions_value=46.173319500000005,
        total_value=99.7691334025
    D4 (2021-12-03, ALPHA's sale settles):
        cash=86.65047015249999, positions_value=23.0733195,
        total_value=109.72378965249999

**The sale settlement** (`friction_cash.settle_sale`, `Ledger.sell`, full
exit — `capital_efficiency_break` resolves to severity `"review"`
(`owner.SEVERITY_MAP`), and `"review"`/`"full_exit"` both sell fraction
`1.0`): `gross_proceeds = 0.231*150 = 34.65`. `slippage = 34.65*(100*0.5)/
10000 = 0.17325` (the *exit* leg's own half-share, computed on the grown
notional — this is exactly what `friction_cash.py`'s own module docstring
means by "the cash cost, read back as percentage points of the original
return, grows with the return"). `post_slippage_proceeds = 34.65 - 0.17325
= 34.47675`. `cost_basis = 0.231*100 = 23.1`. `gain = 34.47675 - 23.1 =
11.37675` (a positive gain — taxed). `regime="ltcg"` (421 >= 365),
`tax_pct=12.5`, `tax = 11.37675*0.125 = 1.4220937500000001` (float
literal: `1.4220937499999993`, see below). `proceeds =
34.47675 - 1.4220937499999993 = 33.054656249999994`.
`cash_after_sale = 53.5958139025 + 33.054656249999994 = 86.65047015249999`.

**On floating-point exactness.** Every literal above (and every one
asserted below) is the *exact* float this arithmetic produces in IEEE 754
double precision — Python's own, deterministic for a fixed sequence of
operations. `ledger.py`'s own module docstring states plainly that nothing
in it is rounded, precisely so that "the equity curve's point at any date
equals `cash + sum of every position's mark` exactly" is a claim that can
be checked bit-for-bit rather than approximately. Reproducing the *same*
sequence of floating-point operations (not merely an algebraically
equivalent one) is therefore what "hand-computed" has to mean here for the
match to be exact rather than approximate — an earlier, less careful pass
at this derivation computed `BETA`'s tranche using `total_value ~= 99.9845`
(subtracting only `ALPHA`'s slippage from the starting pool, forgetting the
notional itself was also spent) and got `23.0964195`, not `23.0733195`; the
mistake was caught by the cross-check against the real `Ledger` class
*before* being written into this file as an expected value, which is the
whole reason that cross-check step exists — see the module docstring's
"Every hand-worked number" section above for the general point (an
independent reimplementation, verified against the real classes once, is
what "computed by hand" means for un-rounded floating-point arithmetic;
copying the pipeline's own emergent output instead would prove nothing).

## A genuine bug found while building this fixture — fixed in lifecycle/exit.py

`lifecycle/exit.py::confirm_exit`'s own `EXITED` transition write used to
never pass `at=`, unlike the `EXIT_REVIEW` transition written immediately
before it in the same `simulator.replay.settle_exit` call (which *does*
pass `at=at_date.isoformat()`). `watchlist.transition` defaults an omitted
`at` to `_now()` — real wall-clock time — so every `EXITED` `state_history`
record this simulator (or the `watchlist exit` CLI path) wrote was stamped
with the moment the code ran, not `as_of`/`exit_date`, even though
`confirm_exit` is explicitly handed `as_of` and already uses it for the
exit-date string and the friction reading two lines above.
`lifecycle/states.py::transition`'s own docstring names this exact failure
mode as the reason the `at` parameter exists ("a replay moves a company
through this exact machinery on a historical `as_of` that is never 'now'
... Left at the default, every transition a replay writes would be
stamped with the date the *code ran*"), which was strong circumstantial
evidence this was an oversight in a Phase 3 (KTD10) function never
revisited when Phase 4 started calling it from a replay context, rather
than intentional.

Confirmed directly against the real `lifecycle.exit.confirm_exit` before
being fixed there: `at=exit_date` (the same value already passed to
`record_exit` two lines above) now reaches the `EXITED` transition too.
Harmless for a live `watchlist exit` — `as_of` defaults to
`date.today()`, the same instant either way — so no production caller's
behaviour changed; verified against the full `test_confirm_exit.py`/
`test_watchlist_lifecycle.py`/`test_reinvestment_queue.py`/
`test_lifecycle_advance.py` suites, all unchanged. This also means
`queue.record_confirmation(exit_id, at=record["at"])` (the line
immediately after, in `confirm_exit`) now inherits the correct timestamp
for the queue's own completion stamp, with no further change needed there.

It never affected this fixture's own exact-match claim: the ledger's own
cash/tax/date figures are computed from the settlement `bar` (`{date,
price}`) `run_replay` resolves and passes to `Ledger.sell` directly, never
from the watchlist transition's own timestamp, so the equity curve was
unaffected before or after the fix. The test below now asserts the
corrected behaviour at the `EXITED` record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from boundless100x.lifecycle.states import (
    APPLIED_AUTO,
    EXIT_REVIEW,
    EXITED,
    PROBE,
    QUALIFY,
    WATCH,
    as_date,
)
from boundless100x.simulator import calendar as calendar_module
from boundless100x.simulator import replay as replay_module
from boundless100x.simulator import universe as universe_module
from boundless100x.watchlist import CORE_LANE

# ── the four replay dates, derived and cross-checked in the module docstring ──

D1 = pd.Timestamp("2020-10-01")  # watch -> the valuation_buy_zone proposal fires
D2 = pd.Timestamp("2020-10-08")  # D1 + 5 trading days: both tranches settle
D3 = pd.Timestamp("2021-12-01")  # capital_efficiency_break becomes readable and fires
D4 = pd.Timestamp("2021-12-03")  # D3 + 2 trading days: ALPHA's sale settles

# ── fundamentals shared by ALPHA and BETA (identical fiscal years) ──────────

_YEARS = ["Mar 2016", "Mar 2017", "Mar 2018", "Mar 2019", "Mar 2020"]
_REVENUE = [400.0, 500.0, 600.0, 700.0, 800.0]
_OPERATING_PROFIT = [160.0, 200.0, 240.0, 280.0, 320.0]
# pat := operating_profit * 0.75 (a debt-free company, flat 25% tax, so PAT
# and NOPAT coincide); eps := operating_profit * 0.075 (10cr shares
# outstanding throughout) -- both exact multiples of operating_profit, which
# is what pins the growth-quality financial-leverage ratio at exactly 1.0
# (see the module docstring's kill-switch section).
_PAT = [op * 0.75 for op in _OPERATING_PROFIT]
_EPS = [op * 0.075 for op in _OPERATING_PROFIT]
_CFO = [100.0, 110.0, 120.0, 130.0, 140.0]
_CFI = [0.0, 0.0, 0.0, 0.0, 0.0]
_EQUITY_CAPITAL = 100.0
_RESERVES = [400.0, 470.0, 540.0, 610.0, 680.0]
_BORROWINGS = [0.0, 0.0, 0.0, 0.0, 0.0]
_FACE_VALUE = 10.0

assert _PAT == [120.0, 150.0, 180.0, 210.0, 240.0]
assert _EPS == [12.0, 15.0, 18.0, 21.0, 24.0]


def _write_annual_csv(path: Path, years: list[str], **columns) -> None:
    pd.DataFrame({"year": years, **columns}).to_csv(path, index=False)


def _write_price_csv(path: Path, *, step_date: pd.Timestamp | None, price_after: float) -> None:
    """A business-day close series, 2015-06-01 through 2022-06-01 -- flat
    100.0 throughout, except from `step_date` onward (inclusive) where it
    steps to `price_after`. `step_date=None` never steps (BETA: flat 100.0
    for its entire history). The wide date range guarantees a bar within
    `PRICE_LOOKBACK_DAYS=45` of every FY-end used by `pe_vs_historical`, and
    that the raw series' own max date is always well past every cutoff this
    fixture evaluates (so KTD0's guard (a), the "at the corpus's latest
    date" reconciliation check, never applies here -- see the module
    docstring).
    """
    dates = pd.bdate_range(start="2015-06-01", end="2022-06-01")
    close = pd.Series(100.0, index=range(len(dates)))
    if step_date is not None:
        close[dates >= step_date] = price_after
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": close.values,
        "volume": 100_000,
    }).to_csv(path, index=False)


def _write_ticker(
    root: Path, ticker: str, *, roce_rows: list[tuple[str, float]] | None,
    step_date: pd.Timestamp | None, price_after: float = 100.0,
) -> Path:
    ticker_dir = root / ticker
    ticker_dir.mkdir(parents=True)

    _write_annual_csv(
        ticker_dir / "financials.csv", _YEARS,
        revenue=_REVENUE, operating_profit=_OPERATING_PROFIT, pat=_PAT, eps=_EPS,
    )
    _write_annual_csv(
        ticker_dir / "balance_sheet.csv", _YEARS,
        equity_capital=[_EQUITY_CAPITAL] * 5, reserves=_RESERVES, borrowings=_BORROWINGS,
    )
    _write_annual_csv(ticker_dir / "cashflow.csv", _YEARS, cfo=_CFO, cfi=_CFI)

    if roce_rows is not None:
        years, roce = zip(*roce_rows)
        _write_annual_csv(ticker_dir / "ratios.csv", list(years), roce=list(roce))

    _write_price_csv(ticker_dir / "price_volume.csv", step_date=step_date, price_after=price_after)

    (ticker_dir / "metadata.json").write_text(
        json.dumps({"name": ticker, "Face Value": _FACE_VALUE})
    )
    return ticker_dir


@pytest.fixture
def fixture_root(tmp_path) -> Path:
    root = tmp_path / "raw_data"
    _write_ticker(
        root, "ALPHA",
        # FY2016-FY2019 healthy, then FY2019 alone breaks (5.0) -- not yet a
        # persist_years:2 pair -- and a sixth row, FY2021 (period end
        # 2021-03-31), also 5.0 and invisible until its own six-month
        # reporting lag passes (2021-09-30), pairs with FY2019 once visible.
        roce_rows=[
            ("Mar 2016", 25.0), ("Mar 2017", 25.0), ("Mar 2018", 25.0),
            ("Mar 2019", 5.0), ("Mar 2021", 5.0),
        ],
        step_date=D4, price_after=150.0,
    )
    _write_ticker(root, "BETA", roce_rows=None, step_date=None)
    return root


def _build_watchlist_at_watch(stores, ticker: str) -> None:
    """Pre-seed one entry directly into `watch`, core lane -- the fixture's
    one deliberate shortcut, argued in the module docstring's "one
    deliberate shortcut" section. Both written transitions are pre-position
    (`AUTO_APPLICABLE`) and move no money.
    """
    stores.watchlist.add(ticker, lane=CORE_LANE)
    stores.watchlist.transition(
        ticker, QUALIFY, "fixture_seed",
        evidence="seeded directly for the hand-computed fixture -- composite "
                 "scoring is out of this fixture's scope, see the module docstring",
        applied_by=APPLIED_AUTO, at="2020-01-01T00:00:00",
    )
    stores.watchlist.transition(
        ticker, WATCH, "fixture_seed",
        evidence="seeded directly for the hand-computed fixture",
        applied_by=APPLIED_AUTO, at="2020-01-02T00:00:00",
    )


def test_two_name_fixture_matches_hand_computation_exactly(fixture_root):
    engines = replay_module.build_engines({})
    calendar_result = calendar_module.ReplayCalendar(
        dates=[D1, D2, D3, D4], start=D1, end=D4,
        end_basis="fixture", dominant_fiscal_month=3, lag_months=6,
    )
    universe_result = universe_module.UniverseResult(
        eligible={"ALPHA": D1, "BETA": D1},
        excluded={},
        ticker_dirs={"ALPHA": fixture_root / "ALPHA", "BETA": fixture_root / "BETA"},
    )

    stores = replay_module.build_stores()
    try:
        _build_watchlist_at_watch(stores, "ALPHA")
        _build_watchlist_at_watch(stores, "BETA")

        result = replay_module.run_replay(
            stores, calendar_result, universe_result, assignments={}, engines=engines, config={},
        )

        # ── no per-ticker failures, no reconciliation failures, nothing left
        #    scheduled and unsettled by the end of the window ──
        assert result["errors"] == []
        assert result["reconciliation_failures"] == []
        assert result["unsettled_confirmations"] == []

        # ── the equity curve, exactly, at every one of the four points ──
        curve = result["equity_curve"]
        assert [point["date"] for point in curve] == [
            "2020-10-01", "2020-10-08", "2021-12-01", "2021-12-03",
        ]

        d1, d2, d3, d4 = curve

        assert d1["cash"] == 100.0
        assert d1["positions_value"] == 0.0
        assert d1["total_value"] == 100.0
        assert d1["marks"] == {}

        assert d2["cash"] == 53.5958139025
        assert d2["positions_value"] == 46.173319500000005
        assert d2["total_value"] == 99.7691334025
        assert d2["marks"] == {"ALPHA": 100.0, "BETA": 100.0}

        # D3: the kill switch only proposes -- nothing settles, nothing
        # about the ledger has moved since D2.
        assert d3["cash"] == 53.5958139025
        assert d3["positions_value"] == 46.173319500000005
        assert d3["total_value"] == 99.7691334025
        assert d3["marks"] == {"ALPHA": 100.0, "BETA": 100.0}

        assert d4["cash"] == 86.65047015249999
        assert d4["positions_value"] == 23.0733195
        assert d4["total_value"] == 109.72378965249999
        assert d4["marks"] == {"BETA": 100.0}

        # cash + positions_value == total_value exactly, at every point --
        # the same reconciliation property proven separately elsewhere,
        # confirmed here on this fixture's own exact literals.
        for point in curve:
            assert point["total_value"] == point["cash"] + point["positions_value"]

        # ── the trade log: two buys (ALPHA then BETA, same settlement date,
        #    BETA's tranche computed from ALPHA's already-updated cash/
        #    positions), one sell (ALPHA, full exit) ──
        buys = [t for t in result["trade_log"] if t["kind"] == "buy"]
        sells = [t for t in result["trade_log"] if t["kind"] == "sell"]
        assert [b["ticker"] for b in buys] == ["ALPHA", "BETA"]
        assert len(sells) == 1

        alpha_buy, beta_buy = buys
        assert alpha_buy["notional"] == 23.1
        assert alpha_buy["slippage"] == 0.1155
        assert alpha_buy["qty"] == 0.231
        assert alpha_buy["price"] == 100.0
        assert alpha_buy["entry_bar_date"] == "2020-10-08"
        assert alpha_buy["cash_after"] == 76.7845

        assert beta_buy["notional"] == 23.0733195
        assert beta_buy["slippage"] == 0.11536659750000001
        assert beta_buy["qty"] == 0.230733195
        assert beta_buy["price"] == 100.0
        assert beta_buy["entry_bar_date"] == "2020-10-08"
        assert beta_buy["cash_after"] == 53.5958139025

        sale = sells[0]
        assert sale["ticker"] == "ALPHA"
        assert sale["qty"] == 0.231
        assert sale["entry_price"] == 100.0
        assert sale["exit_price"] == 150.0
        assert sale["entry_bar_date"] == "2020-10-08"
        assert sale["exit_bar_date"] == "2021-12-03"
        assert sale["holding_days"] == 421
        assert sale["regime"] == "ltcg"
        assert sale["tax_pct"] == 12.5
        assert sale["taxed"] is True
        assert sale["slippage"] == 0.17325
        assert sale["gain"] == 11.376749999999994
        assert sale["tax"] == 1.4220937499999993
        assert sale["proceeds"] == 33.054656249999994

        # ── state history: the two money-moving transitions this fixture
        #    exists to prove, both fired by the real TriggerEvaluator /
        #    advance.decide() against real computed metrics ──
        alpha_history = stores.watchlist.get("ALPHA")["state_history"]
        probe_record = next(r for r in alpha_history if r["to"] == PROBE)
        review_record = next(r for r in alpha_history if r["to"] == EXIT_REVIEW)
        exited_record = next(r for r in alpha_history if r["to"] == EXITED)

        assert probe_record["trigger_id"] == "valuation_buy_zone"
        assert as_date(probe_record["at"]) == D2.date()

        assert review_record["trigger_id"] == "capital_efficiency_break"
        assert as_date(review_record["at"]) == D4.date()

        assert exited_record["trigger_id"] == "capital_efficiency_break"
        # This fixture's own hand-verification caught a real bug:
        # lifecycle/exit.py::confirm_exit's `EXITED` transition write omitted
        # `at=`, so this record was stamped with real wall-clock time rather
        # than D4 -- harmless for a live `watchlist exit` (as_of defaults to
        # date.today(), the same instant either way) but silently wrong for
        # any backdated replay caller. Fixed directly in exit.py (`at=
        # exit_date`, the same value already passed to `record_exit`), so
        # this now matches every other transition this fixture writes.
        assert as_date(exited_record["at"]) == D4.date()

        beta_history = stores.watchlist.get("BETA")["state_history"]
        beta_probe = next(r for r in beta_history if r["to"] == PROBE)
        assert beta_probe["trigger_id"] == "valuation_buy_zone"
        assert not any(r["to"] in (EXIT_REVIEW, EXITED) for r in beta_history)
        assert stores.watchlist.get("BETA")["state"] == PROBE

        # KTD10's confirm_exit trio, complete.
        exits = stores.queue.exits()
        assert len(exits) == 1 and exits[0]["ticker"] == "ALPHA"
        assert stores.queue.find_confirmation(exits[0]["exit_id"]) is not None
    finally:
        stores.close()
