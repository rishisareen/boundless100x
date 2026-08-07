---
title: Phase 4 Strategy Simulator - Plan
type: feat
date: 2026-08-07
artifact_contract: ce-unified-plan/v1
artifact_readiness: draft — owner decisions pending (see Session-settled decisions)
product_contract_source: Design/Financial Model v05 - Phased Growth Roadmap.md (§8.2, §10, §12 Phase 4, §14)
execution: code
---

# Phase 4 Strategy Simulator - Plan

## Goal Capsule

- **Objective:** Build the strategy simulator v05 §10 specifies: a
  phase-replay backtest that answers "do gates + lifecycle + sizing + exits
  produce portfolio CAGR above buy-and-hold of the same names?" — the only
  honest test of the fast lane's "accelerated" claim — by replaying the
  production lifecycle machinery over point-in-time truncations of the cached
  corpus, with a modeled cash ledger, per-lane tax/friction, and the
  reinvestment queue's idle-capital drag all simulated, and with every
  exclusion and limitation stated the way the existing backtest states them.
- **Authority:** v05 §10 (simulator spec and the quarterly-depth constraint),
  §8.2 (friction honesty), §12 Phase 4 (validation: "hand-computed two-name
  fixture reproduces simulator equity curve exactly; limitations block
  present; exclusions listed"), §14.1–.3 (owner-policy placeholders the
  simulator exists to set from evidence), §14.4 (propose-and-owner-confirms,
  which the simulator must *simulate*, not bypass), §14.6 (survivorship
  universe accepted, results read as an upper bound). §13 Non-Goals binds
  absolutely: no SQGLP scoring change, no new data source, no automated
  execution. `CLAUDE.md` governs style. Where this plan conflicts with
  observed code reality, surface it rather than guessing — every prior phase
  plan corrected the roadmap this way, and the corrections were the valuable
  part.
- **Stop conditions:** Stop and surface if (a) any change moves a composite,
  element score, coverage ratio, eligibility verdict, lifecycle transition,
  queue event, or report for an unchanged ticker — the simulator is
  additive-only, exactly as Phases 2 and 3 were, and it must not write to
  `score_history.jsonl`, `watchlist.json`, or `reinvestment_queue.json`
  under any code path; (b) replaying a transition turns out to require
  reimplementing trigger, lane-gate, or checkpoint logic rather than calling
  the production evaluators — a second statement of the rules would drift
  from `triggers.yaml`/`lane_gates.yaml` with nothing to say which one the
  money followed; (c) the LLM stages (Pass 1/2, forward-growth extraction)
  turn out to be load-bearing for a transition the roadmap expects the
  simulator to replay — they are unreplayable (cost, non-determinism, and a
  point-in-time corpus), and the honest answer is an exclusion count, not a
  stub; or (d) a "point-in-time" reading is found to depend on anything
  fetched or written after the truncation date — look-ahead leakage is the
  central correctness risk (§10) and a leak invalidates every number the
  simulator produces.
- **Execution profile:** Code with unit tests per unit, on synthetic fixtures
  per `tests/conftest.py` and the fixture builders in
  `tests/test_fixture_builders.py` — never `raw_data/` in unit tests, never
  live scraping. The one `raw_data/` consumer is the full-corpus validation
  run, mirroring the backtest's own discipline.
- **Tail ownership:** Implementer owns commit hygiene and the end-of-phase
  validation named in the Verification Contract: the hand-computed two-name
  fixture reproducing the simulator equity curve exactly, and a full-corpus
  run with its exclusions list and limitations block.

---

## Product Contract

### Summary

One additive package — `boundless100x/simulator/` — that replays history.
At each replay date it truncates every universe ticker's cached data to what
was public that day (generalizing the backtest's point-in-time discipline
from one split-half cut to an arbitrary date), scores with the production
engine, evaluates the production trigger and lane-gate registries against a
simulated watchlist, applies the transitions a documented *simulated owner*
would confirm, moves modeled cash through a per-tranche ledger with
per-lane tax and slippage, routes exit proceeds through a simulated
reinvestment queue, and marks the whole portfolio to market — producing the
equity curve and the six readings §10 names: portfolio CAGR, max drawdown,
turnover, per-lane net-vs-gross, the fast-lane friction break-even, and
cash-drag from the idle reinvestment queue, each beside the same
buy-and-hold of the same names. Everything the simulator cannot know —
checkpoint contents, catalyst names, the owner's actual confirmations — is
either simulated by an explicit, config-stated policy or counted as an
exclusion; nothing is silently assumed. The output is callable as a function
with config overrides, because Phase 5's sensitivity sweeps are a loop over
that call, not a new mode.

### Problem Frame

The existing backtest (refinements U9) validates that **scores correlate
with returns** — a diagnostic about the engine. Nothing in the system yet
validates that **the rules make money**: that the lifecycle's entries,
kill-switches, tranche sizing, lane gates, and friction-adjusted exits,
composed over years, produce a portfolio outcome worth having. That gap
matters most for the fast lane, whose entire premise is that monetizing
re-ratings on a 6–18 month horizon beats holding through them — and whose
round-trip friction (§8.2) can erase the edge it exists to capture. Phase 3
deliberately shipped the break-even question unanswered ("no hurdle is
computed here... the Phase 4 simulator derives one"). This phase is also the
evidence engine Phase 5 stands on: §14.1–.3's owner-policy placeholders
(sleeve split, tranche sizing, kill-switch severity) are explicitly
"config now, evidence later," and the later is this simulator's sensitivity
sweeps. A simulator that reimplements the rules would prove nothing about
the shipped ones, so the central design constraint is that the replay calls
the production evaluators — the same `triggers.yaml`, `lane_gates.yaml`,
thresholds, and scoring the owner actually runs.

### Requirements

- R1. **Point-in-time truncation at an arbitrary date.** The backtest's
  leakage discipline (period-end cuts, reporting lag, point-in-time
  metadata, leaky-input stripping) generalized from its single split-half
  cut to any replay date, shared so the backtest and the simulator cannot
  diverge on what "knowable that day" means.
- R2. **Replay through the production machinery.** Scoring is
  `engine.run_all` on truncated data (the backtest's own idiom — never
  `service.analyze`, which fetches, writes score history, and calls LLMs);
  transitions are the production `TriggerEvaluator` and `LaneGateEvaluator`
  reading the shipped YAML registries; the state machine is the production
  `WatchlistManager` and `ReinvestmentQueue` pointed at temp-dir stores, so
  schema validation and the append-only disciplines are exercised, not
  approximated.
- R3. **A simulated owner, stated as config.** §14.4's
  propose-and-owner-confirms is preserved in shape: money-moving
  transitions are still proposed by the machinery and then confirmed by a
  documented policy — acceptance rules and a confirmation **lag** in
  trading days, because an instantaneous simulated owner would make cash
  drag and deployment pace read as zero, two of the readings this phase
  exists to produce. Every owner input the production system takes —
  confirmation, catalyst recording, exit confirmation, routing decision —
  has an explicit policy entry, and each is named in the limitations block
  as simulated human judgement.
- R4. **A modeled capital ledger, confined to the simulator.** Phase 3's
  scope boundary holds for production: the watchlist never learns rupees.
  The simulator alone carries a cash pool, per-tranche lots (a core
  position built in thirds has three holding periods and three tax
  brackets), tranche sizing from the `portfolio:` config, and lane/sector
  count caps applied as the simulated owner's deployment rule.
- R5. **Cash-level friction, rate-consistent with `friction.py`.** Slippage
  is charged on traded notional per leg; capital-gains tax is charged on
  each closing lot's gain, STCG/LTCG by that lot's holding days against the
  same `friction:` config the production model reads — so the simulator and
  the per-position model cannot disagree about the regime, only about the
  arithmetic level (cash flows vs. return transform) each is documented to
  operate at. A loss is not taxed, in both.
- R6. **Kill-switch severity as config (§14.3).** The placeholder mapping —
  governance = full exit, valuation saturation = reduce (a fractional sale,
  itself config), everything else = exit review followed by simulated
  confirmation — applied at replay and recorded per exit, so Phase 5's
  sweep can vary it.
- R7. **The six §10 outputs, plus the benchmark.** Portfolio CAGR, max
  drawdown, turnover, per-lane net-vs-gross, fast-lane break-even (the
  friction cost per fast-lane round trip, stated per §8.2 as the
  annualized-return gap the lane must clear), and cash-drag (idle days
  between exit and redeployment, from the simulated queue's own events) —
  each beside an equal-weight buy-and-hold of the same names over the same
  window, funded from the same pool.
- R8. **Exclusion and limitation discipline, inherited from the backtest.**
  Every ticker or transition the replay cannot honestly simulate is counted
  with its reason — insufficient history, unreplayable checkpoint-driven
  transitions (R9), unreadable bars — and the limitations block restates
  the survivorship/upper-bound caveat (§14.6), the quarterly-depth
  constraint (§10 rev 2026-08-06b), and every simulated-owner policy.
- R9. **No LLM stages in the replay.** Pass 1/2 and forward-growth
  extraction are unreplayable; their downstream effects are handled two
  ways and never guessed: transitions whose conditions need LLM-produced
  inputs (checkpoint-driven transitions above all) are counted as
  exclusions, and forward-growth-derived *flags* the lane gates read are
  computed from the deterministic metrics only — see KTD6, which this
  requirement constrains.
- R10. **Additive to production, and callable for Phase 5.** No production
  module changes behavior for an unchanged ticker; the simulator's stores
  are temp-dir; the run is one CLI command and one importable function
  (`simulate(config_overrides) -> metrics dict`) that a Phase 5 sweep loops
  over without subprocesses.

### Scope Boundaries

- **No SQGLP scoring changes** — weights, thresholds, element membership,
  gate logic untouched (v05 §13). The simulator *reads* the scoring regime;
  it never writes it.
- **No point-in-time universe** (§14.6, deferred in Phase 3 and unchanged
  here). The universe is the survivorship-selected `raw_data/` corpus,
  accepted per the roadmap's own resolution; results are read as an upper
  bound and the caveat is restated in every output.
- **No production-state mutation.** The simulator must be safe to run on a
  live watchlist directory: temp-dir stores, read-only corpus access, no
  score-history appends, no report generation into production paths.
- **No rupee tracking in production stores.** The ledger is simulator
  state, serialized into the simulator's own output artifact only.
- **No automated execution, no broker integration** (§13). The simulator
  changes no production posture: it measures what the rules would have
  done; the owner still disposes.
- **No intraday mechanics.** Replay cadence is quarterly-grain evaluation
  with daily-grain pricing (mark-to-market and bar selection), matching
  `advance --quarterly`'s 90-day staleness rhythm; the fast lane's shortest
  cadence stays monthly (§13).
- **No SQGLP calibration output.** The simulator's evidence feeds
  *lifecycle* parameter calibration (Phase 5); SQGLP scoring calibration is
  the separate workstream §12 Phase 5 spawns, out of scope here.

#### Deferred to Follow-Up Work

- **Phase 5's sweeps themselves** — the sweep harness, the
  statistical-humility clause's minimum transition counts, and the
  documented before/after retuning workflow. This phase delivers the
  callable; the loop is Phase 5.
- **Checkpoint-driven transition replay** if organic quarterly history
  accumulates (§10 rev note) and a deterministic checkpoint source emerges.
  Counted as exclusions until then.
- **Consensus-estimate feeds** (§13 flags them as a possible later
  addition); nothing here anticipates them.
- **A portfolio-level dashboard** (already deferred by Phase 3 for the same
  reason: it wants the simulator's outputs to exist first).

---

## Planning Contract

### Key Technical Decisions

- **KTD1 — The replay calls the production evaluators; nothing is
  reimplemented.** The backtest's KTD3 ("reuse the production engine on
  truncated inputs") applies one level up: a simulator with its own copy of
  the trigger rules would prove something about *those* rules, and the two
  statements would drift the first time `triggers.yaml` changed. Concretely:
  `TriggerEvaluator` and `LaneGateEvaluator` are constructed per replay run
  from the shipped YAML (validated against the engine's metric ids, exactly
  as `advance()` does), and `evaluate()` is called per ticker per replay
  date with the same keyword contract `advance_ticker` uses — metrics,
  scores, eligibility, checkpoint summary, lane-gate result, catalyst,
  state history, lane, `as_of`. What the simulator writes itself is the
  *orchestration* `advance_ticker` cannot provide, because
  `advance_ticker` calls `service.analyze` (network, LLM, score-history
  append) — the replay loop is the thin layer that scores truncated data
  and hands readings to the same evaluators. The divergence is stated
  plainly in the limitations block: the replay reproduces `advance()`'s
  *decisions*, not its fetch pipeline.
- **KTD2 — Truncation generalizes from split-half to arbitrary date, in one
  shared module.** `compute_engine/backtest.py._truncate` carries the
  leakage discipline: period-end cuts (never positional) against
  `period_end_date`, a reporting lag before accounts count as public,
  point-in-time metadata rebuilt from truncated frames, and
  `NON_TRUNCATABLE_INPUTS` stripped belt-and-braces. That logic lifts into
  `compute_engine/point_in_time.py` parameterized by cutoff date, and the
  backtest's split-half policy becomes one caller computing its cutoff and
  delegating. The byte-for-byte proof that the refactor changed nothing is
  part of the Verification Contract, because a silent change to the
  backtest's notion of "knowable" would invalidate its published results
  and the simulator's at once. The reporting lag moves with the logic —
  observing at the fiscal period end would score on figures nobody could
  have read.
- **KTD3 — The simulated owner is explicit config, and the lag is the
  point.** §14.4 gives money-moving transitions a human confirmer; the
  simulator substitutes a policy block (`simulator:` in config, owner
  editable) stating, per decision kind: accept-when (proposal evidence
  complete), confirm-after (trading-day lag — entries, exits, and routing
  each carry one), and reject-when (cap breach, unreadable pricing). Zero
  lag is deliberately not the default: an instantaneous owner makes the
  reinvestment queue's idle reading and the deployment pace read as
  frictionless, which is the one flattering assumption §10's cash-drag
  output exists to catch. Every policy is recorded into the output
  artifact, so a result can be re-derived from its own record — the same
  standard `evaluator.evaluate()`'s recorded `lane` set in Phase 3.
- **KTD4 — Modeled capital lives in the simulator only, in per-tranche
  lots.** Production's "no rupees" boundary (Phase 3) is what makes the
  production guardrails count-based; the simulator's CAGR, drawdown, and
  cash-drag require the rupees production declines to track. The ledger
  holds: a starting pool (config), cash, and per-position **lots** — one
  per tranche, because a core position built in thirds has three entry
  dates, three holding periods, and potentially three tax brackets at exit.
  Lot selection at exit is FIFO, stated as policy (India's de-facto
  accounting convention and the only one that needs no owner input).
  Sleeve splits and count caps are applied as the simulated owner's
  deployment rule from the `portfolio:` config — the percentages that have
  no denominator in production have one here, and the two readings must
  never be conflated: the simulator's sleeve math is *modeled capital*,
  labeled as such wherever printed.
- **KTD5 — Cash-level friction reuses the rates, not the transform.**
  `friction.py` is documented as "a return-percentage transform, not a
  cash-flow ledger"; the simulator needs the ledger. It therefore imports
  `friction.config_from` (one statement of the regime: STCG/LTCG rates,
  holding threshold, slippage bps) and `friction._usable_bars`-equivalent
  bar hygiene through the shared truncation path, but applies the costs in
  cash: slippage charged on each leg's traded notional (entry and exit,
  matching the round-trip semantics), tax charged per closing lot on
  post-slippage gain when positive, bracket by that lot's bar-to-bar
  holding days — the same two conventions `compute_net_return` documents
  (slippage before tax, a loss untaxed), so the per-position model and the
  simulator disagree about nothing except the arithmetic level each is
  documented to operate at. The per-lane net-vs-gross output is computed
  from the ledger's own cash flows, not inferred.
- **KTD6 — Unreplayable inputs are exclusions or stated policies, never
  stubs.** Three production inputs have no point-in-time replay source.
  **Checkpoints** are Pass 2 output: none exist historically, so
  checkpoint-driven transitions (`checkpoints_failed`, and any checkpoint
  condition) are not replayed — every affected ticker-date is counted, and
  the roadmap's "validated on the annual grain" is corrected to "excluded
  and counted": with no LLM in the replay there are no checkpoint
  *definitions* to evaluate at any grain (see Corrections). **Catalysts**
  are owner judgement the lane gate requires: the simulated-owner policy
  auto-records a synthetic catalyst (config window) for a fast-lane
  candidate that has cleared the other five gates — the gate machinery
  stays live and the input is named as fabricated in the limitations block;
  catalyst-spent exits then fire from the window passing, which is the one
  catalyst dynamic that *is* replayable. **Forward-growth LLM flags** the
  lane gates might read (`rerating_headroom_*` are deterministic and
  replay fine; LLM-extracted forward-growth flags do not exist on
  truncated data) — conditions sourcing them read indeterminate, the
  house rule, and the count of gate-indeterminate readings is reported.
- **KTD7 — Replay cadence is quarterly evaluation over daily pricing.**
  Production's rhythm is `advance --quarterly` (90-day staleness); the
  replay evaluates transitions on quarter boundaries derived from the
  corpus's own fiscal calendar, while mark-to-market, bar selection, and
  holding-period measurement use daily bars. The quarterly-depth
  constraint (§10 rev 2026-08-06b) is inherited: Screener's quarterly
  table covers only recent quarters, so quarterly-grain *metrics* (the TTM
  gate, the accumulation streak) read indeterminate deep in truncated
  history — the gates say so rather than passing silently, and the count
  is reported; this is the indeterminate rule working as designed, not a
  simulator defect.
- **KTD8 — The universe enters at `screen` on first sufficient history.**
  There is no point-in-time universe and no simulated "owner adds a name"
  input, so the candidacy rule is: every `raw_data/` ticker joins the
  simulated watchlist at `screen` on the first replay date whose truncated
  financials meet the engine's minimum-years requirement, with tickers
  failing that bar at every replay date listed in the exclusions. This is
  the survivorship assumption §14.6 resolved to accept, made visible as a
  count rather than hidden as an omission.
- **KTD9 — The benchmark is the same names, bought once.** §10's
  comparison is "portfolio CAGR above buy-and-hold of the same names": an
  equal-weight position per universe ticker (as each reaches sufficient
  history), funded from the same starting pool, marked on the same bars,
  charged the same slippage on entry, never sold, taxed nothing (no
  taxable event occurs) — so the benchmark isolates exactly what the rules
  add or subtract: timing, sizing, exits, and friction. The comparison is
  stated per lane as well as in aggregate, because the fast lane's premise
  is the claim under test.
- **KTD10 — The output is an artifact, and the run is a function.**
  `simulate(config, overrides) -> dict` returns the full result — equity
  curve, metrics, exclusions, limitations, policy record — and the CLI
  command is a thin renderer over it writing
  `output/simulations/<date>/simulation.json` beside the backtest's own
  output directory. Phase 5's sweeps are `for params in grid:
  simulate(config, params)`; nothing in this phase's design may assume a
  human reads the console.

### Session-settled decisions

Drafted defaults, **pending owner confirmation before implementation** —
the same posture Phase 3's plan took toward its owner-policy placeholders.
Each is config-shaped so confirmation is an edit, not a refactor.

1. **Starting pool and its unit.** The ledger needs a modeled starting
   amount to make CAGR and cash drag concrete. Draft: a configurable
   `starting_pool` defaulting to 100 (unitless "capital units"), since
   every output is a ratio; an owner who wants rupee-shaped numbers edits
   one value. No production figure is implied either way.
2. **Confirmation lags.** Draft: entries 5 trading days, exits 2, routing
   5 — placeholders in the §14.1–.3 spirit ("config now, evidence later"),
   deliberately non-zero (KTD3). Sweep candidates for Phase 5.
3. **Catalyst policy window.** Draft: synthetic catalysts carry a
   9-month expected window (mid §9.1's 6–18), after which the simulated
   owner marks them spent unless the position has already exited — the
   replayable half of the catalyst dynamic (KTD6).
4. **Reduction fraction for §14.3's "reduce" severity.** Draft: sell half
   the position's lots (FIFO) — config, sweep candidate.
5. **Cap enforcement posture.** Draft: the simulated owner *obeys* the
   lane/sector count caps (skips and records the skip) rather than
   breaching-and-reporting, because the sweep question is "what do the caps
   do to returns" — a posture the advisory production system cannot take.
   Config-switchable to advisory-only if the owner wants the baseline
   measured uncapped.

### Assumptions

- **A1.** The corpus's cached frames (`financials`, `price_volume`,
  `shareholding`, `quarterly`, `metadata`) are sufficient for the engine to
  score truncated views without any fetch — the backtest already proves
  this for its split-half cut; arbitrary dates are the same reads with a
  different cutoff. *If wrong:* tickers whose truncated view cannot score
  join the exclusions with their reason, and the count is reported.
- **A2.** `adj_close` history is internally consistent across the corpus
  window (the `adj_close_is_estimated` hygiene already applied in
  `friction.py` and the backtest carries over unchanged through the shared
  truncation path). *If wrong:* the same unavailable-with-reason reading
  the production model produces, counted per event.
- **A3.** The universe is static: no ticker in `raw_data/` was ever absent
  from it mid-window (delisting/survivorship is the accepted §14.6 caveat,
  restated in every output).
- **A4.** Quarterly cadence approximates production behavior closely
  enough: an owner advancing monthly would catch transitions a quarterly
  replay sees late by up to one quarter. Stated in limitations; monthly
  cadence is a config knob if evidence later justifies the runtime.

### Starting-point defaults

| Parameter | Value | Status |
|---|---|---|
| Replay start | earliest date with ≥ engine minimum years of truncated financials for ≥ 2 tickers | derived |
| Replay cadence | quarterly, on the corpus fiscal calendar | starting point (A4) |
| Starting pool | 100 capital units | placeholder (decision 1) |
| Confirmation lag: entry / exit / route | 5 / 2 / 5 trading days | placeholder (decision 2) |
| Synthetic catalyst window | 9 months | placeholder (decision 3) |
| Reduce-severity fraction | 50% of lots, FIFO | placeholder (decision 4) |
| Cap posture | enforced | placeholder (decision 5) |
| Friction regime | `friction:` config, unchanged | shipped (Phase 3) |
| Tranche sizing / sleeve split | `portfolio:` config, unchanged | shipped (Phase 3, §14.1–.2) |

### Risks

| Risk | Mitigation |
|---|---|
| Look-ahead leakage through a shared frame or a metadata field nobody truncated | KTD2's single truncation module; belt-and-braces `NON_TRUNCATABLE_INPUTS` strip; a dedicated leak test that scores the same ticker at two cutoffs and asserts no output cell references post-cutoff data |
| The replay drifts from production semantics as the lifecycle evolves | KTD1 — evaluators, registries, stores are the production objects; the thin replay layer is the only simulator-specific logic, and `advance()`'s integration tests pin the evaluator contract it shares |
| The simulated owner flatters the strategy (instant confirmation, perfect catalysts) | Non-zero lags by default (KTD3); every policy recorded into the artifact and named in limitations; catalyst fabrication explicitly labeled |
| Thin transition counts make lane comparisons noise | §10's own requirement: transition counts per lane printed beside every rate; the statistical-humility clause (§12 Phase 5 rev) is quoted in the limitations block even though the sweeps are Phase 5's |
| Runtime: full-corpus scoring × replay dates is hours | Score caching keyed on (ticker, cutoff, registry hash); quarterly cadence by default; `--tickers` subset flag for iteration; the full run is a validation event, not a dev loop |
| A shared-truncation refactor silently changes the published backtest | Byte-identical backtest proof in the Verification Contract, run before any simulator unit lands |

### High-Level Technical Design

```
boundless100x/simulator/
  __init__.py            # simulate(config, overrides) -> dict — the Phase 5 seam
  point_in_time.py       # lives in compute_engine/ instead: see U1
  universe.py            # discovery + sufficient-history candidacy (KTD8)
  calendar.py            # replay dates from the corpus fiscal calendar (KTD7)
  owner.py               # the simulated-owner policy block (KTD3, KTD6)
  ledger.py              # cash, lots, tranches, caps, mark-to-market (KTD4)
  friction_cash.py       # cash-level application of friction.config_from rates (KTD5)
  replay.py              # the loop: truncate → score → evaluate → propose → confirm → settle
  outputs.py             # equity curve, six §10 readings, benchmark, exclusions, limitations
```

The replay loop per replay date, stated once so every unit hangs off the
same skeleton:

1. truncate each active ticker's corpus to the date (U1),
2. score with `engine.run_all` + scorer + eligibility (never
   `service.analyze`), reading the pace modulator once per date off the
   truncated corpus exactly as production reads it once per run,
3. evaluate lane gates (fast-lane tickers) and triggers with
   `as_of` = replay date against the simulated stores,
4. hand money-moving proposals to the simulated owner (U3), which
   schedules confirmations after their lags,
5. settle due confirmations through the ledger (U4): buys in tranches,
   sells per severity, tax and slippage in cash (U5), exit proceeds into
   the simulated queue, routes due into the queue's own top proposal,
6. mark to market and append the equity-curve point.

---

## Implementation Units

### Phase A: Point-in-Time Mechanics

### U1. Arbitrary-date truncation (`compute_engine/point_in_time.py`)

- **Goal:** One shared statement of "what was knowable on date D," consumed
  by the backtest and the simulator alike.
- **Requirements:** R1, R2. **Dependencies:** none.
- **Files:** `boundless100x/compute_engine/point_in_time.py` (new),
  `boundless100x/compute_engine/backtest.py`,
  `tests/test_point_in_time.py` (new), `tests/test_backtest.py`
- **Approach:**
  1. Lift `_truncate`'s logic — period-end cuts via `period_end_date`,
     reporting lag, `_point_in_time_metadata`, the
     `NON_TRUNCATABLE_INPUTS` strip — into
     `truncate_to_date(data, cutoff, reporting_lag_months)` in the new
     module, verbatim semantics.
  2. `backtest.py` computes its split-half cutoff exactly as today and
     delegates; its exclusion reasons and return shape are unchanged.
  3. The quarterly and shareholding frames are cut by their own period
     labels with the same reporting-lag rule — the backtest's
     `ANNUAL_FRAMES` handling extends to the frames Phase 0/3 added, since
     the lane gates read them.
- **Patterns to follow:** `_truncate` itself; `period_end_date`'s handling
  of non-March fiscal years.
- **Test scenarios:**
  - A ticker truncated at an arbitrary date exposes no frame row, price
    bar, shareholding row, or quarterly row whose period ends after the
    cutoff minus reporting lag (the leak test, per column).
  - A trailing part-year column sharing a calendar year with the cutoff
    is excluded (the case `_truncate`'s comment records).
  - The refactored backtest on a fixed fixture produces a byte-identical
    `backtest.json` to the pre-refactor run.
  - A cutoff before the price series starts yields the same exclusion
    reason shape the backtest uses today.
- **Verification:** the byte-identical backtest proof, recorded with its
  diff output (empty) in the Implementation Record.

### U2. Replay calendar, universe, and simulated stores

- **Goal:** The skeleton the loop hangs off: replay dates, candidacy, and
  temp-dir production stores.
- **Requirements:** R2, R8, R10. **Dependencies:** U1.
- **Files:** `boundless100x/simulator/calendar.py` (new),
  `boundless100x/simulator/universe.py` (new),
  `boundless100x/simulator/replay.py` (new, skeleton),
  `tests/test_simulator_calendar.py` (new),
  `tests/test_simulator_universe.py` (new)
- **Approach:**
  1. `calendar.py`: replay dates quarterly from the corpus's fiscal
     calendar (dominant period-end month + reporting lag), first date
     where ≥ 2 tickers have sufficient truncated history, last date the
     price corpus supports.
  2. `universe.py`: `raw_data/` discovery (the backtest's
     `discover_candidates` idiom), per-ticker first-eligible date under
     KTD8, exclusion reasons for never-eligible tickers.
  3. `replay.py` skeleton: constructs `WatchlistManager` and
     `ReinvestmentQueue` on `tempfile.TemporaryDirectory()` paths —
     production schema validation and append-only discipline included,
     production files untouchable by construction — and `add`s each
     ticker at `screen` on its first-eligible date. Lane assignment is
     the simulated owner's: a config rule (draft: every candidate enters
     the core lane; a candidate moves to the fast lane's *evaluation*
     when its fast-lane gate battery is consulted — both lanes simulated
     in parallel watchlists is an anti-goal; one watchlist, lane chosen
     per the config rule, recorded per ticker).
- **Test scenarios:**
  - Replay dates never fall before the earliest sufficient-history date
    or after the last priced date.
  - A ticker with too few years is excluded with its reason and never
    enters the simulated watchlist.
  - Stores are temp-dir: running the skeleton against a fixture leaves
    `boundless100x/watchlist.json`, `score_history.jsonl`, and
    `reinvestment_queue.json` byte-identical (asserted by hash).
  - Lane assignment follows the config rule and is recorded per ticker.

### Phase B: The Simulated Portfolio

### U3. The simulated owner (`owner.py`)

- **Goal:** Every human input to the lifecycle, stated as policy.
- **Requirements:** R3, R6, R8. **Dependencies:** U2.
- **Files:** `boundless100x/simulator/owner.py` (new),
  `boundless100x/config.yaml` (`simulator:` block),
  `tests/test_simulator_owner.py` (new)
- **Approach:**
  1. `config.yaml` gains a commented `simulator:` block: starting pool,
     confirmation lags (entry/exit/route), catalyst window, reduce
     fraction, cap posture — each labeled a placeholder awaiting Phase 5
     evidence, beside the `friction:`/`portfolio:` blocks it consults.
  2. `owner.py`: pure policy functions — `decide(proposal, portfolio,
     config) -> scheduled confirmation | skip with reason`;
     `catalyst_for(candidate, gate_result, config) -> catalyst dict |
     None`; `route(exit_event, queue_view, config) -> scheduled routing |
     hold`. Skips are recorded with reasons (the production queue's
     blocked-with-reasons idiom).
  3. Severity mapping (§14.3) lives here as config: trigger id →
     `full_exit | reduce | review`, with `review` resolving to a
     scheduled exit confirmation after the exit lag.
- **Test scenarios:**
  - A proposal is never confirmed before its lag elapses; the scheduled
    date is recorded.
  - A cap-blocked entry under the enforced posture is skipped with the
    cap's reason and the capital stays idle (and drags).
  - The catalyst policy fabricates a catalyst only for a candidate that
    has cleared the other five gates, and marks it spent when the window
    passes.
  - Every decision lands in the run's policy record verbatim.

### U4. The capital ledger (`ledger.py`)

- **Goal:** Modeled cash, per-tranche lots, mark-to-market — the numbers
  CAGR and drawdown come from.
- **Requirements:** R4, R7. **Dependencies:** U3.
- **Files:** `boundless100x/simulator/ledger.py` (new),
  `tests/test_simulator_ledger.py` (new)
- **Approach:**
  1. State: cash, `{ticker: [lots]}`; a lot is `{qty, entry_bar_date,
     entry_price, lane, tranche_index}`.
  2. `buy(ticker, tranche_size_fraction, bar, config)`: tranche notional
     from the sleeve's *deployable* share of pool-and-accrued-value per
     the `portfolio:` config — stated as modeled capital wherever printed
     (KTD4).
  3. `sell(ticker, fraction, bar, reason) -> realized cash flows`: FIFO
     lots, per-lot holding days, delegating costs to `friction_cash` (U5).
  4. `mark_to_market(date, price_frames) -> portfolio value`: last usable
     bar on or before the date per holding (the friction module's bar
     direction), positions with no usable bar carried at last mark and
     counted.
- **Test scenarios:**
  - A two-tranche position holds two lots with independent holding
    periods; a partial exit consumes FIFO.
  - Slippage reduces cash on both legs; a gain under the LTCG threshold
    is taxed STCG, over it LTCG, a loss untaxed (the U5 contract, seen
    from the ledger).
  - Mark-to-market on a holiday uses the prior trading bar.
  - The equity curve's final point equals cash + Σ marks exactly.

### U5. Cash-level friction (`friction_cash.py`)

- **Goal:** The `friction:` regime applied to traded notional.
- **Requirements:** R5. **Dependencies:** none (leaf; U4 consumes).
- **Files:** `boundless100x/simulator/friction_cash.py` (new),
  `tests/test_simulator_friction_cash.py` (new)
- **Approach:**
  1. `cost_of_buy(notional, config) -> slippage`; `settle_sale(lot,
     exit_price, config) -> {proceeds, tax, slippage, regime}` — bracket
     by the lot's bar-to-bar holding days against
     `ltcg_holding_days`; tax on post-slippage gain when positive; a loss
     untaxed (the `compute_net_return` conventions, restated in cash).
  2. Rates come only from `friction.config_from` — a statute change
     remains one config edit for the whole system.
- **Test scenarios:**
  - Hand-computed: a known lot at a known price delta and holding period
    settles to exactly the expected proceeds, tax, and regime.
  - The round trip's total cost equals the per-position model's
    `compute_net_return` applied to the same return, within rounding —
    the consistency check that the transform and the ledger tell one
    story.
  - Config edits (e.g. `stcg_pct: 25`) flow through with no code change.

### Phase C: Evidence

### U6. Outputs and the benchmark (`outputs.py`)

- **Goal:** The six §10 readings, the benchmark, and the honesty blocks.
- **Requirements:** R7, R8, R9, R10. **Dependencies:** U4, U5.
- **Files:** `boundless100x/simulator/outputs.py` (new),
  `tests/test_simulator_outputs.py` (new)
- **Approach:**
  1. Equity curve → CAGR, max drawdown, turnover (traded notional over
     mean portfolio value, annualized); per-lane net-vs-gross from the
     ledger's cash flows; fast-lane break-even stated per §8.2 as the
     annualized gap a fast-lane round trip must clear, derived from the
     *measured* friction per simulated cycle — this is the number Phase 3
     declined to compute; cash drag as mean/median idle days per exit
     from the simulated queue's events plus the pool share idle over
     time.
  2. Benchmark (KTD9) computed alongside from the same pool, bars, and
     slippage; comparison table strategy-vs-benchmark, aggregate and per
     lane.
  3. Exclusions and limitations blocks, following
     `backtest._describe_exclusions`/`_limitations`: never-eligible
     tickers, checkpoint-driven transitions excluded (counted), gate-
     indeterminate readings deep in history (counted), stale-mark events
     (counted); limitations restate survivorship/upper-bound (§14.6),
     quarterly depth (§10 rev), every simulated-owner policy by name, and
     the statistical-humility clause.
  4. The full result dict is the artifact; a thin console renderer for
     the CLI reads it.
- **Test scenarios:**
  - A hand-computed micro run (fixture ledger) yields exactly the
    expected CAGR/drawdown/turnover.
  - Every exclusion kind is counted and rendered; the limitations block
    names each policy.
  - The artifact round-trips through JSON unchanged (the Phase 5 consumer
    reads it programmatically).

### U7. Replay loop integration, CLI, and validation runs

- **Goal:** Wire A+B into the loop of the High-Level Design; ship the
  command; prove it on the hand-computed fixture and the real corpus.
- **Requirements:** all. **Dependencies:** U1–U6.
- **Files:** `boundless100x/simulator/replay.py` (fleshed out),
  `boundless100x/simulator/__init__.py`, `boundless100x/cli.py`
  (`simulate` command), `tests/test_simulator_replay.py` (new),
  `tests/test_simulator_fixture.py` (new)
- **Approach:**
  1. The six-step loop per the High-Level Design, with per-date score
     caching keyed on (ticker, cutoff, registry hash) and the pace
     modulator resolved once per date off the truncated corpus (the
     production once-per-run idiom at replay grain).
  2. CLI: `boundless100x simulate [--tickers A,B] [--start DATE]
     [--end DATE] [--set key=value ...]` writing
     `output/simulations/<date>/simulation.json` and printing the
     renderer's summary; `--set` feeds the `overrides` seam Phase 5 uses.
  3. The hand-computed two-name fixture: two synthetic tickers with
     scripted fundamentals and prices, a scripted expected equity curve
     (every tranche, tax line, and idle day computed by hand), asserted
     to match **exactly** — §12 Phase 4's stated validation.
  4. Full-corpus run recorded into the Implementation Record with its
     exclusions and limitations — evidence, not a test.
- **Test scenarios:**
  - The two-name fixture's equity curve matches the hand computation at
    every replay point (not just the final one).
  - A replay with a kill-switch firing mid-window exits on schedule
    (post-lag), books tax by lot, and drags cash until the route lag
    elapses.
  - `simulate(config, {"simulator.entry_lag_days": 0})` runs without
    subprocesses and returns a different, recorded policy block (the
    Phase 5 seam, exercised).
  - Production non-mutation: hashes of the three production stores are
    unchanged after any replay.

## Verification Contract

- Full suite green via `venv/bin/python -m pytest tests/` (network tests
  remain deselected).
- **Byte-identical backtest.** The U1 refactor leaves `backtest.json` on a
  fixed fixture byte-identical; the diff (empty) is recorded in the
  Implementation Record.
- **Leak test.** U1's per-column assertion that no truncated frame exposes
  post-cutoff-minus-lag data, plus a two-cutoff scoring diff demonstrating
  no output cell references the future.
- **Hand-computed two-name fixture reproduces the simulator equity curve
  exactly** — §12 Phase 4's stated validation, asserted at every replay
  point with the hand computation checked into the test as literal
  expected values.
- **Production non-regression.** A cached ticker scored before and after
  the phase diffs clean on `scores.json` and `eligibility.json` (the
  Phase 2/3 proof shape), and the three production stores are hash-stable
  across a full simulated run.
- **Full-corpus validation run.** `simulate` over `raw_data/` completes
  with its exclusions list non-empty-or-explained, limitations block
  present, transition counts per lane printed beside every rate, and the
  benchmark beside every strategy figure — recorded in the Implementation
  Record with its actual numbers, not just "passed."
- **Friction consistency.** The U5 round-trip check (ledger cash costs vs.
  `compute_net_return` on the same return, within rounding) passes,
  recorded with its values.

## Definition of Done

- All seven units merged with tests green.
- The byte-identical backtest proof and the production non-regression
  proof recorded with their actual outputs.
- The hand-computed fixture proof recorded — §12 Phase 4's own validation.
- The full-corpus run's numbers, exclusions, and limitations recorded in
  an Implementation Record section appended to this plan (the Phase 0–3
  convention), including any correction this phase forced against the
  roadmap text — starting with the checkpoint-grain correction KTD6
  names, which §10's rev note anticipated only in part.
- The `simulate(config, overrides)` seam demonstrated with one override,
  so Phase 5 starts from a proven callable.
