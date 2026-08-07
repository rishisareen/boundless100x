---
title: Phase 4 Strategy Simulator - Plan
type: feat
date: 2026-08-07
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
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
- R6. **Kill-switch severity as config (§14.3), with "reduce" deferred.**
  The placeholder mapping — governance = full exit, everything else =
  exit review followed by simulated confirmation — applied at replay and
  recorded per exit, so Phase 5's sweep can vary it. §14.3's third value,
  valuation saturation = reduce, has no settled fraction (decision 4 is
  owner-deferred): the partial-sale mechanics ship built but inactive
  behind config with no default, and a saturation trigger in the baseline
  resolves through the same review-then-confirm full exit as every other
  switch, counted separately so the affected sample is visible.
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

- **KTD0 — What a point-in-time *valuation* reading may be. SETTLED with the
  owner on 2026-08-07: rebuild both market cap and the multiple from
  truncatable inputs.** Numbered 0 rather than 11 because U1 implements it and
  the units below already name the other decisions by number — renumbering
  would break those references for no gain.

  Measured on 2026-08-07 by running `backtest._truncate` → `engine.run_all`
  → `EligibilityEvaluator` → `LaneGateEvaluator` over eight cached tickers at
  their own split-half cutoffs. Every input the lifecycle opens a position on
  errors, identically on all eight:

  | Metric | Reading under truncation | What it gates |
  |---|---|---|
  | `market_cap` | `No market cap in metadata` | `size` eligibility gate |
  | `trailing_peg`, `peg_ratio` | `No P/E` | `price` eligibility gate; core `valuation_buy_zone` |
  | `reverse_dcf_growth` | `No market cap for reverse DCF` | the `price` gate's `veto_sources` |
  | `pe_vs_historical` | `No current P/E` | core `valuation_buy_zone`; `valuation_discount` gate |
  | `ttm_growth_vs_cagr` | `Missing input(s): quarterly` | `growth_intact` gate |
  | `institutional_accumulation_streak` | `Missing input(s): shareholding` | `institutional_accumulation` gate |
  | `daily_turnover_ratio` | `No market cap for turnover calculation` | `liquidity_floor` gate |

  Traced through the shipped registries that is not a coverage dent, it is a
  closed door. The eligibility verdict can never read `eligible` — `size` and
  `price` are both permanently indeterminate — so core `qualification_passed`
  never fires and the only core transition the replay can reach is
  `qualification_failed → dropped`. `valuation_buy_zone` needs two of the
  errored metrics. Four of the fast lane's six gates read indeterminate, so
  `lane_verdict` is never `qualifies` and `fast_lane_buy_zone` never fires.
  **The replay buys nothing, in either lane, at any date** — a flat equity
  curve, and all six of R7's readings undefined rather than poor. The fast
  lane's "accelerated" claim, which the Goal Capsule names as the whole point
  of the phase, would be untestable.

  This is neither the quarterly-depth constraint KTD7 inherits nor a defect:
  `_point_in_time_metadata` omits market cap and Stock P/E **on purpose**,
  because today's market cap is the single worst leak available and a P/E
  rebuilt off split-adjusted closes is not the ratio anyone saw. R1's leakage
  discipline and R7's six outputs are, as written, mutually exclusive. The
  decision is which of them bends, and it is the owner's:

  a. **Rebuild both from truncatable inputs.** The precedent is already
     shipped and already argued: `valuation._current_multiple` builds the
     current multiple from price and as-reported annual EPS *specifically* to
     keep `rerating_headroom` inside the backtest — and it is the one
     valuation metric that came back `OK` above. Extending the construction to
     a point-in-time market cap needs a share count the pipeline does not
     store reliably, which is the objection `_point_in_time_metadata` records;
     deriving one from equity capital and face value is possible and would
     have to be proven against the **raw** close, never the adjusted one.
  b. **Accept a documented reduced gate set.** The backtest already does this
     — it drops the `size` gate outright rather than scoring around it, and
     states the divergence in its own `note`. The simulator would state the
     same, per lane, and every output would carry which gates were and were
     not consulted. Cheapest, and the most honest about what it did not test;
     it also weakens KTD1, because the replayed rules are then not quite the
     shipped ones, and that cost must be named in the limitations block.
  c. **Narrow the window to where the corpus supports the gates.** Defensible
     for the fast lane and nearly useless for the core one — see U1's measured
     frame depths.

  **Owner chose (a) on 2026-08-07**, over a recommendation of (b). The
  objection to (a) was that a reconstructed market cap fails quietly, so it
  was measured before being written down — and it holds up:

  ```
  market_cap_cr = equity_capital_cr × raw_close ÷ face_value
  ```

  `equity_capital` is paid-up capital on the balance sheet, `face_value` is
  in `metadata.json`, and the shares term cancels. Checked against the stored
  `Market Cap` at the corpus's latest date, where the stored figure is the
  truth: **20 of 22 tickers within 2%, 14 within 0.5%**; the two outliers are
  EDELWEISS (−2.4%) and KFINTECH (+2.1%). So the objection is answered on the
  market-cap side, and it is answered *by a check the implementation can keep
  running* — see the reconciliation guard in U1, which is what converts the
  quiet failure mode into a loud one.

  **The multiple is a different matter, and the measurement changed what this
  decision says about it.** Rebuilding `close ÷ latest annual EPS` and
  comparing it to Screener's stored `Stock P/E` gives errors from −14% to
  **+1169%** (RAIN), because `Stock P/E` is a TTM figure and annual EPS is
  not — they are two different multiples, not one multiple measured twice. So
  the rebuilt figure is **not** a reconstruction of `Stock P/E` and must never
  be validated against it. It is the `_current_multiple` definition
  (`valuation.py`), which already exists, is already argued, and already
  refuses a non-positive EPS — the guard RAIN's +1169% is exactly what
  motivates.

  That turns out to be the *more* consistent choice rather than a compromise:
  `pe_vs_historical` builds its band from each year-end close over that
  year's EPS, so the band is already on the annual-EPS basis. Feeding it a
  current multiple on the same basis is more internally coherent than the
  production path's stored TTM P/E measured against an annual-EPS band. The
  consequence is stated in the limitations block rather than buried: the
  replay's `pe_vs_historical` is basis-consistent where production's is not,
  so a replayed percentile is **not** the number production would have shown
  that day. That is a divergence in the replay's favour, and it is still a
  divergence.

  **It goes into the `Stock P/E` key regardless, and the honesty travels
  beside it rather than in the key name.** An earlier draft of this KTD said
  the rebuilt multiple would carry a *different* key so nobody could mistake
  it for the fetched TTM figure. Measured, that choice is self-defeating:
  `trailing_peg`, `peg_ratio` and `pe_vs_historical` all read
  `meta.get("Stock P/E")` directly, so a renamed key leaves all three
  erroring, the `price` eligibility gate indeterminate, and the core lane
  frozen exactly as it was before this decision — the phase dies of its own
  naming convention. Changing those three metrics to read a new key is a
  production scoring change, which Scope Boundaries forbid.

  So the key is populated and the basis rides alongside as provenance
  (`_stock_pe_basis: "annual_eps_reconstructed"`), which is the pattern this
  repo already uses for exactly this problem — `price_basis` in valuation
  metadata, `adj_close_is_estimated` on a price series. Provenance beside the
  value travels where a key name cannot: into the limitations block, the
  exclusions, and any artifact a Phase 5 sweep reads. Zero production metrics
  change, and no consumer can mistake a reconstruction for a fetch.

  Spiked on 2026-08-08 with both fields populated, across all 15 scorable
  tickers at their split-half cutoffs: the 100x verdict goes from **0
  `eligible`** to **3**, so `qualification_passed` fires and the core path is
  live. One of those three is RAIN, whose two share counts disagree by 221%
  and whose multiple was the +1169% outlier — a reconstruction artifact that
  the guard above excludes. Without the guard one eligible verdict in three
  would have been garbage, which is the clearest argument for it available.

  Both reconstructions are marked as reconstructed wherever they surface, so
  no consumer can mistake one for a fetched figure.

  **The share count has a second, independent derivation, and that closes the
  hole in the guard.** `pat ÷ eps` off the annual frame gives shares
  outstanding without touching face value or the balance sheet. Measured
  across the corpus it is the *worse* estimator on its own — **16 of 22
  within 2% against the equity-capital route's 20 of 22**, with RAIN at
  +221% and IDEA's implied count moving 152% across five years, because EPS
  is rounded, and basic-versus-diluted and consolidated-versus-standalone do
  not always pair. So it does not replace the formula above.

  What it does is fix a real weakness in the reconciliation guard as first
  specified: **the stored `Market Cap` exists only at the corpus's latest
  date, which is the one replay date the simulator never scores on.**
  Everywhere the reconstruction is actually *used* there was nothing to check
  it against. Two independent derivations of the same quantity are checkable
  against each other at **every** annual row, so the guard runs at every
  replay date rather than once at the end: agreement is evidence, and a
  divergence beyond tolerance excludes the ticker at that date with both
  figures named. The tickers where the two disagree today — RAIN, EDELWEISS,
  IDEA — are exactly the ones a silent reconstruction would have scored
  wrongest, and RAIN is already the outlier in the multiple check, which is
  the corroboration worth having.
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
  state history, lane, `as_of`.

  **Calling the evaluators is not enough on its own, and that gap is what
  U0 closes.** `advance_ticker` cannot be reused whole — it opens with
  `service.analyze`, which fetches, calls the LLM path, and appends to
  `score_history.jsonl`. But everything *after* the readings are in hand is
  also production rule: the precedence sort (`_rank`, and `_exit_rank`'s
  universal-before-lane-scoped tiebreak at an exit review), the kill-switch
  status derivation, the deployment-pace evidence clause and the rule that
  it attaches by trigger id rather than destination state, the friction
  estimate on an `exit_review` proposal, the concentration gate asked only
  when a transition would add a name, and `moves_money → should_apply`.
  Restating those in the replay loop would be a second statement of exactly
  the rules this decision forbids duplicating — and the drift would be
  invisible in the worst way, because a simulator that ranked a buy-zone
  above a kill-switch would report a *better* result for it. So the decision
  core is extracted once (U0) and both callers use it; the replay loop is
  then genuinely thin. The residual divergence is stated plainly in the
  limitations block: the replay reproduces `advance()`'s *decisions*, not
  its fetch pipeline.
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
  cash: slippage on each leg's traded notional, tax charged per closing lot
  on post-slippage gain when positive, bracket by that lot's bar-to-bar
  holding days — the same two conventions `compute_net_return` documents
  (slippage before tax, a loss untaxed).

  **`slippage_bps` is a round trip, so each leg carries half of it.**
  `config.yaml` says so in capitals and `friction.py` states it once
  precisely so it cannot drift ("Round trip: entry *and* exit... Split into
  two half-legs it would be the same number"). Charging the configured bps on
  entry *and* again on exit would silently double the regime the rest of the
  system runs under — the same class of error as reading a per-position cap
  as a per-lane one. So the ledger charges `slippage_bps / 2` per leg, and
  the halving is stated in the code rather than inferred from the name.

  **The two models still cannot agree on magnitude, and that is arithmetic,
  not a defect to fix.** `compute_net_return` deducts bps off the *return
  percentage*; the ledger charges bps on *notional*, and the exit leg's
  notional is the grown position. On a +100% gross the notional path costs
  1.5pp of return where the transform costs a flat 1.0pp, and the gap widens
  with the return. They converge only near zero. So the claim this KTD makes
  is narrower than "the same number": the two share the **regime** — the same
  rates from `friction.config_from`, the same `ltcg_holding_days` bracket
  boundary, slippage before tax, a loss untaxed — and differ by construction
  on level, because a transform on a return and a charge on a traded amount
  are different quantities. U5 asserts the regime and documents the level
  difference with a worked example; it does not assert equality. The per-lane
  net-vs-gross output is computed from the ledger's own cash flows, not
  inferred.
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
  constraint (§10 rev 2026-08-06b) is inherited, and measurement made it
  sharper than the roadmap states it: Screener renders ~11–13 quarters, so
  `quarterly.csv` begins Mar/Jun 2023 and `shareholding.csv` Sep 2023, and
  **no refetch can deepen either** — this is a property of the source, not
  of when the corpus was last pulled. The two quarterly-grain gates (the
  TTM gap, the accumulation streak) are therefore indeterminate before
  ~Q2 2024 whatever the replay start is. The gates say so rather than
  passing silently and the count is reported — the indeterminate rule
  working as designed — but the consequence for §10's per-lane comparison
  is a real limit rather than a rounding error, and it is what moved the
  replay start to 2023 (see U2) and what the per-lane battery-complete
  dates exist to expose.
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

Settled with the owner on 2026-08-07. Each is config-shaped, so a later
reconsideration is an edit, not a refactor.

**One is open, and one was reversed.** Decision 4 remains deferred by the
owner. Decision 5 was **reopened on review and resettled the other way**: it
had been settled on a claim about production behaviour that a later commit
had already made false, and a decision is only as settled as the premise
under it. Both are marked in place rather than rewritten — what a decision
was reversed *from* is the part a reader needs, and dropping it would leave
the same wrong premise available to be reasoned from again. KTD0, which
review opened and the owner then settled, follows the same convention: the
recommendation it was settled *against* stays visible beside the choice.

1. **Starting pool and its unit — SETTLED: 100 unitless "capital
   units."** Every output is a ratio, so the unit cancels, and no simulated
   figure can be mistaken for a real rupee amount. An owner who later wants
   rupee-shaped outputs edits one config value.
2. **Confirmation lags — SETTLED: 5 trading days to confirm an entry, 2
   to confirm an exit, 5 to route exit proceeds.** Deliberately non-zero
   (KTD3): an instantaneous owner would flatter cash drag and deployment
   pace to zero, the exact readings this phase exists to produce. Sweep
   candidates for Phase 5.
3. **Catalyst policy window — SETTLED: 6 months.** The aggressive end of
   §9.1's 6–18 month horizon, not the midpoint: the synthetic catalyst's
   window is what drives catalyst-spent exits (KTD6), and the owner prefers
   the lane's discipline tested at its tightest stated bound. A candidate
   whose re-rating has not come six months after its fabricated catalyst
   meets the exit rule a real thesis of that shape would have met.
4. **Reduction fraction for §14.3's "reduce" severity — DEFERRED by the
   owner.** No fraction is settled, so the baseline gives "reduce" nowhere
   to land: a valuation-saturation trigger resolves through the same
   review-then-confirm path as every other kill-switch (a full exit after
   the exit lag), and the partial-sale mechanics ship built but inactive
   behind config with no default. The consequence is stated plainly in the
   limitations block: the baseline measures a strategy that always exits
   valuation saturation in full, and the reduce-fraction sweep is Phase 5
   work that starts only once the owner settles the number. Simulated
   valuation-saturation events are counted separately so the affected
   sample is visible.
5. **Cap enforcement posture — REOPENED 2026-08-07: the premise it was
   settled on is stale.** It read "SETTLED: advisory — the simulated owner
   breaches-and-reports, exactly as production behaves (Phase 3's documented
   residual: caps gate the advisory router, never the transition path)."
   Production stopped behaving that way in `72a509f feat(portfolio): a
   concentration cap that prevents rather than reports`. Today
   `advance_ticker` consults `concentration_gate` **before** any transition
   that would add a positioned name, sets `concentration_withheld`, and
   computes `should_apply = (apply and not withheld)` — so a breaching
   transition is withheld even under `--apply`. `--override-caps` is an
   explicit per-run escape hatch that writes the breach into the append-only
   evidence, and `portfolio.would_breach` fails *closed* on an unavailable
   reading, an undescribed lane, an unconfigured cap, and a sector that
   cannot be shown to fit.

   So "the strategy as it is actually run" is **enforced, with the breach
   recorded and an explicit per-run override** — the opposite of what the
   decision concluded from it. The baseline posture is therefore
   **enforced-by-default** until the owner says otherwise: a baseline
   measuring a posture production does not run answers a question nobody
   asked. What survives unchanged is the reason the decision gave for
   recording breaches — every withheld transition is logged with the cap it
   would have broken, so a Phase 5 enforced-vs-advisory sweep can still price
   what obedience cost or saved. Advisory becomes the config switch, and the
   simulated owner needs a **third** value, `override`, the analogue of
   `--override-caps`, so all three postures production can be run in are
   reachable from config.

   **RESETTLED with the owner on 2026-08-07: enforced.** The baseline
   measures what `advance --apply` actually does — a breaching transition
   withheld and recorded. `advisory` and `override` ship as config values so
   Phase 5 can sweep all three and price what the guardrail cost or saved.

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
| Replay start | **2023-01-01** — both lanes on comparable data, not the earliest scorable date | settled (owner, 2026-08-08) |
| Reporting lag | per frame: annual 6mo, quarterly results 2mo, shareholding 1mo — **not** one lag for all three | settled (see U1) |
| Replay cadence | quarterly, on the corpus fiscal calendar | starting point (A4) |
| Starting pool | 100 capital units | settled (decision 1) |
| Confirmation lag: entry / exit / route | 5 / 2 / 5 trading days | settled (decision 2) |
| Synthetic catalyst window | 6 months | settled (decision 3) |
| Reduce-severity fraction | none — reduce ships inactive; saturation exits in full | deferred (decision 4) |
| Cap posture | enforced — breaching transition withheld and recorded; `advisory` and `override` are config values | settled (decision 5, resettled) |
| Point-in-time market cap | rebuilt: `equity_capital × raw_close ÷ face_value`, reconciled against the stored figure at the latest date | settled (KTD0) |
| Point-in-time multiple | rebuilt: `_current_multiple`'s raw close ÷ annual EPS — **not** Screener's TTM `Stock P/E` | settled (KTD0) |
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

### U0. Split `advance_ticker` into decision and effects — **LANDED (`2911b79`)**

- **Status:** Shipped ahead of the rest of the phase. `decide()` is
  `advance.py:376`, `advance_ticker` keeps its signature and performs the
  writes, and the purity assertion below exists as
  `TestTheDecisionCoreIsPure::test_it_decides_without_touching_a_store`.
  Suite green at 1741 with no edit to any pre-existing test, which is the
  behaviour-preservation evidence. **Still owed:** the verification line —
  a cached ticker advanced before and after the split producing an identical
  outcome dict. The unit description below is kept as the record of what was
  built, not as work to do.
- **Goal:** One statement of "given these readings, what should happen to
  this company next," callable by production and by the replay. Numbered 0
  for KTD0's reason — it precedes the simulator package and renumbering
  U1–U7 would break the references above.
- **Requirements:** R2 (and it is what makes KTD1 true rather than
  intended). **Dependencies:** none — this is a production-side refactor and
  is not blocked by KTD0.
- **Files:** `boundless100x/lifecycle/advance.py`,
  `tests/test_lifecycle_advance.py`
- **Approach:**
  1. Extract `decide(...) -> dict` from `advance_ticker`: everything between
     the analysis being in hand and the first write. It takes the entry, the
     state and lane read off it, the readings (`metrics`, `scores`,
     `eligibility`, the price frame, the sector), `as_of`, the run's
     `evaluator`, `lane_gates`, `pace`, `concentration_gate` and
     `override_caps`, plus `config` for the friction rates. It returns
     `{evaluation, checkpoint_summary, checkpoint_outcomes,
     lane_gate_result, proposal, friction_estimate, routing_safety}` and
     **performs no I/O and no writes** — that is the property that makes it
     replayable and the one a test should pin directly.
  2. `advance_ticker` becomes: `service.analyze` → re-read `entry` →
     `decide(...)` → the three writes in their existing order
     (`set_kill_switch_status`, `transition` when `should_apply`,
     `record_snapshot` **last**) → `lane_context` off the re-read entry →
     the outcome dict. The snapshot-last ordering is load-bearing and
     documented in place; the extraction must not disturb it, because
     `get_stale(90)` reads that timestamp and an earlier write is what made
     a failed run look freshly scored.
  3. No behaviour change, and no signature change to `advance_ticker` or
     `advance` — every existing caller (CLI, tests) is untouched.
- **Patterns to follow:** `friction.reading_for_exit` and
  `portfolio.would_breach` — both are the same move already made in this
  layer, a rule with two callers stated once, and both docstrings argue why.
- **Test scenarios:**
  - `decide` is called with a stub watchlist entry and hand-built readings
    and returns the expected proposal, with **no** watchlist mutation — the
    purity assertion, made against a store that raises on any write.
  - Precedence survives the move: a kill-switch and a buy-zone firing in the
    same run still resolve to the kill-switch; at an exit review a universal
    switch still outranks a lane-scoped one for the displayed rationale.
  - The existing `test_lifecycle_advance.py` suite passes unchanged — that
    it needed no edit is the evidence the refactor was behaviour-preserving.
- **Verification:** a cached ticker advanced before and after the split
  produces an identical outcome dict (minus timestamps), recorded in the
  Implementation Record.

### U1. Arbitrary-date truncation (`compute_engine/point_in_time.py`)

- **Goal:** One shared statement of "what was knowable on date D," consumed
  by the backtest and the simulator alike.
- **Requirements:** R1, R2. **Dependencies:** KTD0 must be settled first —
  this unit is what implements whatever it decides.
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
  3. Implement KTD0 in `_point_in_time_metadata`'s successor, the only place
     the valuation inputs can come from. It gains two rebuilt fields and a
     reconciliation guard:

     - **`Market Cap` = `equity_capital × raw_close ÷ face_value`**, all three
       read at the cutoff: `equity_capital` off the truncated balance sheet,
       `raw_close` off the **`close`** column (never `adj_close` — a split
       moves equity capital and the adjusted price in the same direction and
       would be counted twice), `face_value` off metadata, which is static.
     - **`Stock P/E` = `_current_multiple`'s raw close ÷ latest annual EPS**,
       written to that exact key so `trailing_peg`, `peg_ratio` and
       `pe_vs_historical` read it unchanged and no production metric moves.
       Non-positive EPS refuses, as `_current_multiple` already does. The
       basis rides beside it as `_stock_pe_basis`, never in the key name —
       see KTD0 for why the honest-rename version kills the core lane.
     - **The reconciliation guard, at two levels.** *Against stored truth*,
       at the corpus's latest date, where the fetched `Market Cap` exists —
       measured baseline 20 of 22 within 2% (worst: EDELWEISS −2.4%,
       KFINTECH +2.1%), so a 5% tolerance passes today while still catching a
       formula gone wrong on partly-paid shares, a face-value change or an
       unusual capital structure. *Against the independent `pat ÷ eps` share
       count*, *at every replay date* — because the stored figure exists only
       at the latest date, which is the one date the simulator never scores
       on, so a latest-date-only guard checks the reconstruction exactly
       where it is not used. A divergence beyond tolerance **fails the ticker
       into the exclusions at that date with both figures named**. Together
       these are what make KTD0(a) safe: without them the reconstruction
       fails silently, which was the whole objection to it.

     Whichever way a field resolves, the *reason* it is absent travels with
     the truncated view rather than being re-derived by each consumer — a
     metric erroring with "No P/E" cannot say whether the field was withheld
     to prevent a leak, was never fetched, or failed reconciliation, and
     those are three different exclusions.
  4. **The `quarterly` frame is cut by its own period labels**, extending the
     backtest's `ANNUAL_FRAMES` handling to a frame Phase 0 added and the
     `growth_intact` gate reads.

     **The reporting lag becomes per frame, and with a 2023 start that is not
     a nicety.** `REPORTING_LAG_MONTHS = 6` is calibrated for annual accounts
     and is right there. Applied unchanged to the quarterly frames it is
     simply wrong: SEBI LODR gives 45 days for quarterly results and 21 for
     the shareholding pattern, so a six-month lag withholds figures that were
     public four to five months earlier. On the old ~24-date window that cost
     precision; on a 12-quarter series inside a 2023-start window it deletes
     two of the fast lane's few evaluable dates outright. So: annual 6
     months, quarterly results 2, shareholding 1 — each a named constant with
     the filing rule beside it, and each still a *lag*, never zero.

     **`shareholding` is a separate question and this plan previously
     contradicted itself on it** — step 1 lifts a `NON_TRUNCATABLE_INPUTS`
     strip that contains `"shareholding"`, and the step this replaces then
     truncated the frame that strip removes. Both cannot hold. Resolve it in
     the direction the data supports: `shareholding.csv` is in fact a
     labelled quarterly series and *is* cuttable, but the measured corpus
     depth is **12 quarters starting Sep 2023** (ASTRAL, CONCOR, 2026-08-07),
     and `institutional_accumulation_streak` needs three adjacent quarters to
     see two rises. So truncating it buys the `institutional_accumulation`
     gate roughly the last 8 of ~24 replay dates and nothing before. Cut it,
     because a real reading late in the window beats none; drop it from
     `NON_TRUNCATABLE_INPUTS` **only** in the simulator's caller, leaving the
     backtest's strip byte-identical, since changing what the backtest
     withholds would move its published correlations.
- **Patterns to follow:** `_truncate` itself; `period_end_date`'s handling
  of non-March fiscal years.
- **Measured frame depths** (2026-08-07, feeding KTD7's revised constraint):
  `financials` 13–14 annual rows; `price_volume` from 2016-08-09;
  `quarterly` 13 rows from Mar/Jun 2023; `shareholding` 12 rows from
  Sep 2023. At `MIN_TOTAL_YEARS=8` the first eligible cutoff observed is
  2020-09-30, so the replay window is ~24 quarterly dates and the two
  quarterly-grain gates are computable across roughly the last third of it.
- **Test scenarios:**
  - A ticker truncated at an arbitrary date exposes no frame row, price
    bar, shareholding row, or quarterly row whose period ends after the
    cutoff minus reporting lag (the leak test, per column).
  - The backtest's own truncated view is unchanged by the shareholding
    decision above — it still receives no `shareholding` key at all.
  - A quarterly or shareholding frame too shallow to reach the cutoff
    yields an absent-with-reason reading, never an empty frame that a
    metric would read as "no rises".
  - **The rebuilt market cap reconciles at the latest date** across the
    cached corpus, within the configured tolerance, and a ticker that does
    not reconcile lands in the exclusions with its error rather than
    scoring on a fabricated size.
  - The rebuild uses `close`, not `adj_close`: a fixture carrying a split
    partway through produces the same market cap on both sides of it,
    which is the double-count this choice exists to avoid.
  - A non-positive EPS refuses the multiple rather than emitting one — the
    RAIN case, where annual EPS near zero puts the rebuilt figure three
    orders of magnitude away from anything meaningful.
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
     calendar (dominant period-end month + the per-frame reporting lags of
     U1.4), from the configured start to the last date the price corpus
     supports.

     **The start is 2023-01-01 by owner decision, not the earliest scorable
     date.** The earliest would be ~2020-09, and the five quarters it buys
     are ones in which the fast lane is *structurally* unable to qualify —
     `growth_intact` and `institutional_accumulation` have no data before
     Mar and Sep 2023 respectively, and no refetch can change that because
     Screener renders only ~11–13 quarters. A window whose first fifth
     admits only core-lane entries would have made the per-lane comparison
     an artifact of the corpus rather than a finding about the rules.

     Measured at three cutoffs (2026-08-08): the scorable universe is a
     stable **15 tickers** whether the cut is 2023-01, 2024-06 or 2025-06,
     so the later start costs no companies. The eligible population does
     thin over the window — 3 → 1 → 0 — which means core-lane entries
     cluster early and the outputs must say so rather than let a quiet
     second half read as a strategy declining to buy.

     **State the residual honestly: 2023 narrows the gap, it does not close
     it.** The fast lane's full battery needs three adjacent shareholding
     quarters (Sep 2023 + two) and a four-quarter TTM, so it completes
     around Q2 2024 — roughly the last 9 of ~14 replay dates. `calendar.py`
     therefore computes and records a **per-lane battery-complete date**,
     and U6's comparison is reported both over the whole window and over
     the sub-window where both lanes are fully evaluable. One of those two
     answers §10's question; the other shows what it cost to ask it.
  2. `universe.py`: `raw_data/` discovery (the backtest's
     `discover_candidates` idiom), per-ticker first-eligible date under
     KTD8, exclusion reasons for never-eligible tickers.
  3. `replay.py` skeleton: constructs `WatchlistManager` and
     `ReinvestmentQueue` on `tempfile.TemporaryDirectory()` paths —
     production schema validation and append-only discipline included,
     production files untouchable by construction — and `add`s each
     ticker at `screen` on its first-eligible date. Lane assignment is
     the simulated owner's, stated as a config rule: every candidate is
     screened against the fast-lane gate battery each replay date it has
     the readings for, and a candidate whose five computable gates clear
     (the sixth being the fabricated catalyst, KTD6) enters at `screen`
     in the `rerating` lane; all others enter `core`. One watchlist —
     parallel lanes in parallel watchlists is an anti-goal — and the
     lane is recorded per ticker per assignment, so the per-lane outputs
     can attribute every transition to the lane that produced it.
- **Test scenarios:**
  - Replay dates never fall before the configured start (2023-01-01) or
    after the last priced date, and never before a ticker's own
    sufficient-history date.
  - The per-lane battery-complete date is computed and recorded, and the
    fast lane's lands after the core lane's — the asymmetry the 2023 start
    narrows but does not remove.
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
     confirmation lags (entry/exit/route), catalyst window, cap posture —
     each labeled with its settled value or deferral (see Session-settled
     decisions), beside the `friction:`/`portfolio:` blocks it consults.
     The reduce fraction is *absent*, not defaulted: decision 4 is
     owner-deferred, and a config that invents one would read as settled.
  2. `owner.py`: pure policy functions — `decide(proposal, portfolio,
     config) -> scheduled confirmation | skip with reason`;
     `catalyst_for(candidate, gate_result, config) -> catalyst dict |
     None`; `route(exit_event, queue_view, config) -> scheduled routing |
     hold`. Skips are recorded with reasons (the production queue's
     blocked-with-reasons idiom).
  3. Severity mapping (§14.3) lives here as config: trigger id →
     `full_exit | reduce | review`, with `review` resolving to a
     scheduled exit confirmation after the exit lag. In the baseline no
     trigger maps to `reduce` — the fraction is unsettled (decision 4) —
     so a valuation-saturation trigger resolves as `review` → full exit,
     and each such event is counted separately for the limitations block.
- **Test scenarios:**
  - A proposal is never confirmed before its lag elapses; the scheduled
    date is recorded.
  - A cap-breaching entry under the **enforced** baseline posture is
    withheld and the cash drags, matching what `advance --apply` does
    (decision 5, as resettled). Under the config-switched `advisory` and
    `override` postures the buy proceeds and the breach is recorded with
    its cap — all three postures production can be run in, all three
    reachable, and the baseline is the one production actually runs.
  - A valuation-saturation trigger resolves as review → full exit after
    the exit lag, and the event lands in the saturation count.
  - The catalyst policy fabricates a catalyst only for a candidate that
    has cleared the other five gates, with the settled 6-month window,
    and marks it spent when the window passes.
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
  2. **Each leg charges `slippage_bps / 2`**, because the configured figure
     is a round trip (KTD5). The halving is a named constant with the reason
     beside it, not an inline `/ 2`: charging the full figure twice is the
     one error here that produces plausible numbers.
  3. Rates come only from `friction.config_from` — a statute change
     remains one config edit for the whole system.
- **Test scenarios:**
  - Hand-computed: a known lot at a known price delta and holding period
    settles to exactly the expected proceeds, tax, and regime.
  - **Regime consistency, not equality.** A round trip and
    `compute_net_return` on the same gross return agree on tax regime,
    `tax_pct`, `ltcg_holding_days` boundary behaviour (at the boundary and
    one day either side), slippage-before-tax ordering, and a loss going
    untaxed. They are asserted to **differ** on level in the documented
    direction — the notional path costs more as the return grows, because
    the exit leg's notional is the grown position — with a worked case
    (+100% gross: 1.5pp of return against the transform's flat 1.0pp)
    checked in as a literal so a future change to either side has to
    confront the arithmetic rather than silently reconcile it.
  - Total round-trip slippage on a flat position equals exactly
    `slippage_bps` of notional, never twice it — the double-charge guard.
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
  3. **Gate coverage, reported per gate per window — not per run.** A
     run-level "which gates were consulted" hides the shape of the problem,
     because the gates do not go dark together. Even with KTD0's rebuild in
     place, two of the fast lane's six are depth-bound rather than
     metadata-bound: `growth_intact` needs `quarterly` (from Mar/Jun 2023)
     and `institutional_accumulation` needs three adjacent shareholding
     quarters (from Sep 2023). Even with the start moved to 2023-01 that
     leaves the fast lane's battery incomplete until ~Q2 2024 — **four of
     six gates over the first third of the window, and one of those four is
     the fabricated catalyst (KTD6), so three are genuinely evaluated.**

     §9.2's battery is conjunctive and *is* the claim under test, so a
     fast-lane entry taken on three real gates is not the entry the shipped
     lane would have taken, and a reader must be able to see which. Every
     `qualifies` verdict therefore carries the gates that actually decided
     it, and the outputs carry a per-gate-per-replay-date coverage matrix.
     This is named in the limitations block **from day one**, not derived
     later from the indeterminate counts.
  4. Exclusions and limitations blocks, following
     `backtest._describe_exclusions`/`_limitations`: never-eligible
     tickers, checkpoint-driven transitions excluded (counted), gate-
     indeterminate readings (counted, and now attributable per gate per
     window by 3), stale-mark events (counted), reconciliation failures
     (KTD0's guard, counted with both share counts); limitations restate
     survivorship/upper-bound (§14.6), quarterly depth (§10 rev), the
     fast-lane gate-coverage caveat above, the rebuilt-multiple basis
     divergence (KTD0), every simulated-owner policy by name, and the
     statistical-humility clause.
  5. The full result dict is the artifact; a thin console renderer for
     the CLI reads it.
- **Test scenarios:**
  - A hand-computed micro run (fixture ledger) yields exactly the
    expected CAGR/drawdown/turnover.
  - Every exclusion kind is counted and rendered; the limitations block
    names each policy.
  - **A fast-lane entry taken while two gates read indeterminate records
    which four decided it**, and the coverage matrix shows those two dark
    at that date — the assertion that a thin battery cannot be mistaken
    for a full one.
  - A window in which no fast-lane gate is fully computable reports the
    lane as unmeasured with its reason, never as a lane with no
    qualifying candidates — the same distinction `lane_gates.py` draws.
  - The artifact round-trips through JSON unchanged (the Phase 5 consumer
    reads it programmatically).

### U7. Replay loop integration, CLI, and validation runs

- **Goal:** Wire A+B into the loop of the High-Level Design; ship the
  command; prove it on the hand-computed fixture and the real corpus.
- **Requirements:** all. **Dependencies:** U0–U6.
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
- **Friction regime consistency.** The U5 check passes: ledger cash costs and
  `compute_net_return` agree on regime, rates, bracket boundary, ordering and
  loss handling, and differ on level in the documented direction. Recorded
  with its values, including the worked +100% case — a level *agreement*
  would mean one of the two stopped doing what it says it does.

## Definition of Done

- All eight units merged with tests green (U0 plus U1–U7). **U0 landed
  early, in `2911b79`**; only its cached-ticker verification is outstanding.
- **The fast lane's measured gate count is stated, not implied.** The
  coverage matrix (U6.3) is in the artifact and the caveat is in the
  limitations block, so no §9.2 result is read as resting on six gates when
  three of them were evaluated.
- **The decision core has exactly one statement.** No trigger precedence,
  kill-switch derivation, pace-evidence, friction-on-exit, concentration-gate
  or `moves_money` rule is written twice — the replay reaches them by calling
  `decide`, not by restating them (U0, KTD1).
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
