---
title: Phase 2 Engine Enhancements - Plan
type: feat
date: 2026-08-06
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: Design/Financial Model v05 - Phased Growth Roadmap.md (§7.1, §7.2, §7.3, §11, §12 Phase 2)
execution: code
---

# Phase 2 Engine Enhancements - Plan

> **Rev 2026-08-06b (external review):** six contracts hardened. Two were
> errors rather than gaps — Stage 1.5 gated cache *hydration* behind `use_llm`
> so the cache could never serve the paths it existed for (U4), and
> `quarterly_momentum` claimed to measure acceleration while specifying a
> single YoY figure, which is a growth level (U5). Also: promises-kept
> semantics and the required-section map are now settled rather than left to
> the implementer (KTD4, U5); extraction validation grounds each quoted
> sentence against the submitted text instead of only type-checking it (KTD3);
> the registry hash splits into scoring-regime and forward-signal hashes so
> Phase 5 calibration cannot destroy the trajectory evidence it needs (KTD8,
> new); and U3's headroom formula, sign convention, and display bands are
> specified because R8's interpretation band depends on them.
>
> **Rev 2026-08-06c (second external review):** two further changes, both from
> measurements against the real corpus. **`found` provenance is only ~56%
> reliable** — 8 of the 18 `found` MD&A slices are actually auditor's reports,
> governance, CSR or HR text, because the heading regex matched a
> cross-reference; KTD9 (new) adds a content gate and a third `suspect`
> provenance value, since none of the existing guards — including KTD3's
> substring grounding — can distinguish genuine text in the wrong section.
> A1's coverage figures are restated in three layers as a result. And the pace
> input becomes the **corpus-median** spread rather than an owner-set value
> (KTD7, U6), which removes the staleness machinery entirely at the cost of a
> two-pass `advance()`.

## Goal Capsule

- **Objective:** Add *forward* evidence and *delta* signal to a system that
  currently measures only backward and only at a point in time — score
  trajectory, re-rating headroom, a forward-growth module extracted from
  annual reports, and a deployment-pace modulator — without moving a single
  composite score.
- **Authority:** v05 §7.1, §7.2, §7.3, §11 and §12 Phase 2 govern behavior;
  §13 Non-Goals binds absolutely ("No changes to SQGLP elements, weights,
  thresholds, or gate logic in any phase"); this plan's Key Technical
  Decisions govern mechanism; `CLAUDE.md` governs style. Where the plan
  conflicts with observed code reality, surface it rather than guessing —
  Phases 0 and 1 each corrected the roadmap this way and the corrections were
  the valuable part.
- **Stop conditions:** Stop and surface if (a) any change moves a composite,
  element score, coverage ratio, or eligibility verdict for an unchanged
  ticker — this phase is additive-only by contract; (b) an LLM-extracted
  sub-metric cannot be made to read indeterminate on `fallback` provenance,
  since mining a chairman's letter for guidance it does not contain is worse
  than having no metric; or (c) the extraction pass cannot be built without
  the compute engine importing from `llm_layer/`, which would invert the
  dependency direction and break the backtest.
- **Execution profile:** Code with unit tests per unit, on synthetic frames
  per `tests/conftest.py` — never `raw_data/`, never live scraping. The
  extraction pass is tested against recorded response fixtures, not live API
  calls.
- **Tail ownership:** Implementer owns commit hygiene and the end-of-phase
  validation: a before/after `scores.json` diff on a cached ticker proving
  the composite is byte-identical, plus the found / suspect / fallback split
  for the forward-growth sub-metrics across the fetched corpus.

---

## Product Contract

### Summary

Four additive capabilities: momentum diffs computed over the append-only score
history Phase 0 began accumulating; a re-rating-headroom metric comparing
today's multiple against one justified by the company's own RoCE, growth and
longevity; a forward-growth module whose three text-derived sub-metrics come
from a new LLM extraction pass and whose fourth is computed offline from the
quarterly series; and a deployment-pace modulator that tightens buy-zone entry
when the equity risk premium is compressed. Every new metric enters at zero
weight and is surfaced in its own report section, so none of it touches the
SQGLP composite.

### Problem Frame

The engine answers "how good is this company, today, on ten years of
history" — and nothing else. Growth carries 25% of the composite but is
measured entirely backward (CAGR, streaks), so a company whose growth is
decelerating *right now* scores identically to one accelerating. Valuation is
treated only as a risk (percentile, reverse-DCF veto), never constructively —
there is no measure of how much re-rating room a company has, which is the
actual accelerator the roadmap is built around. And a score is a photograph:
the pipeline has recorded history since Phase 0 but nothing reads it, so
improving fundamentals — the thing that precedes re-rating — are invisible.

Phase 0 built the data (quarterly series, multi-year annual-report sections,
score history) and Phase 1 built the layer that consumes decisions. Nothing
yet turns that data into forward signal.

### Requirements

- R1. Momentum over score history: per-ticker deltas in composite and element
  scores, computed only within a single scoring regime, with the interval each
  figure spans stated explicitly.
- R2. A re-rating-headroom metric expressing today's multiple against a
  quality-justified multiple derived from the company's **own** fundamentals
  (RoCE, growth, longevity) — never a sector-relative anchor (v05 §14.5).
- R3. A forward-growth module of four sub-metrics: promises-kept ratio, capex
  pipeline, TAM runway, and quarterly momentum.
- R4. The three text-derived sub-metrics are produced by an LLM extraction
  pass whose output is validated at the boundary against a closed vocabulary;
  anything unparseable or out-of-vocabulary is discarded with its reason
  recorded, never stored as a value a consumer will trust.
- R5. A sub-metric whose required annual-report section carried `fallback`
  provenance evaluates **indeterminate**, never a value (v05 §7.2).
- R6. A deployment-pace modulator that tightens buy-zone entry thresholds when
  the earnings-yield-over-G-Sec spread is compressed, and records that it did
  so. It modulates entry pace only: it never fires, suppresses, or overrides a
  kill-switch or an eligibility gate (v05 §11).
- R7. **No composite, element score, coverage ratio, or eligibility verdict
  changes for any ticker.** New metrics carry zero weight and are excluded
  from element scoring, the composite, and the coverage denominator.
- R8. The new signals are visible to the reader in their own report section,
  presented so that nothing implies they contributed to the SQGLP score — and
  each renders with its **direction of goodness and interpretation band**, so
  a bare number can be read as favourable or not without recomputing it. Zero
  weight means these metrics never receive a score, so without a declared band
  the section would ship numbers rather than signal.

### Scope Boundaries

- **No SQGLP scoring changes** — weights, thresholds, element membership for
  scoring purposes, and gate logic are untouched (v05 §13).
- **No synthetic backfill.** Owner-confirmed: the registry-hash change resets
  the momentum baseline and momentum accumulates fresh from here. The
  backtest's truncation machinery is not used to manufacture history.
- **No trigger-threshold calibration** — starting values only; evidence comes
  from the Phase 4 simulator (v05 §12 Phase 5).
- **No fast lane, no portfolio layer, no reinvestment queue** — Phase 3.
- **No new data sources** (v05 §13). Extraction reads annual reports already
  fetched; momentum reads history already recorded.
- **Threshold tightening only, no tranche step-down.** §11 names two pace
  levers; tranche sizing depends on the portfolio layer (§8.1, §14.2) and
  lands with Phase 3. This phase modulates buy-zone entry thresholds only.
- **Quarterly momentum reads the results series only.** §7.2 also names
  latest-quarter shareholding as an input; shareholding-derived momentum stays
  with the Phase 1 checkpoint path that already reads it.
- **No lifecycle trigger consumes a Phase 2 metric.** Deferred until the
  found / suspect / fallback yield is measured. Note that `advance` re-scores with
  `use_llm=False`: under U4's hydration/extraction split the sub-metrics *are*
  available there once a sidecar exists, but they are absent on a ticker never
  analysed with the LLM. Declaring a trigger on them now would read indeterminate
  forever, the exact silent failure `validate_triggers` exists to prevent.

#### Deferred to Follow-Up Work

- **`indeterminate` escalation policy.** Phase 1 deferred "a trigger
  indeterminate for several consecutive runs deserves attention rather than
  silence" to Phase 2. It stays deferred: answering it needs trajectory data
  showing how often it actually happens, and this phase is what starts
  producing that data. Revisit once several quarters of history exist.
- **Feeding TAM runway into Longevity's CAP proxy** (v05 §7.2 suggests it).
  That would change a scored metric's inputs, which R7 forbids. Revisit when
  SQGLP recalibration is opened as its own workstream.
- **Consensus-estimate feeds** as a forward-growth input — a new data source,
  explicitly out of scope until after Phase 3.

---

## Planning Contract

### Key Technical Decisions

- **KTD1 — New metrics enter at zero weight, and forward-growth gets an
  element outside `element_weights`.** §12 says headroom "lands in the price
  element" while §13 forbids changing SQGLP scoring; those reconcile only at
  zero weight. The scorer's display-only path (`scorer.py`, the `weight == 0`
  branch) `continue`s before weighted accumulation, so such a metric
  contributes nothing to the element mean, the composite, or the coverage
  denominator — an errored one cannot even depress coverage. Headroom goes
  into `price.yaml` at weight 0.0 as §12 asks; the four forward-growth
  sub-metrics go into a new `elements/forward_growth.yaml` whose element name
  is **absent from `element_weights`**, mirroring how `composite` already sits
  outside the scored set. Belt and braces: zero weight makes them non-scoring,
  and an unweighted element makes them structurally incapable of scoring.
- **KTD2 — LLM extraction is a separate pass writing into `data`, never a
  metric that calls the LLM.** `compute_engine/` imports nothing from
  `llm_layer/` and no metric has ever made a network call; the seam is the
  `data` dict, exactly as `annual_report_sections` already demonstrates. This
  is not stylistic. The backtest re-runs **every** registered metric inside a
  per-ticker loop — a metric that called an API would issue one call per
  ticker per backtest with no rate limiting, and worse, would read *today's*
  annual-report text against *truncated* financials, which is precisely the
  look-ahead leak the backtest exists to prevent.
- **KTD3 — Extraction output is validated at the boundary against a closed
  vocabulary.** `_parse_json_response` performs no schema validation of any
  kind, so a malformed, truncated, or older response arrives unchecked. The
  recorder mirrors `checkpoints.record_from_pass2`: every field type-checked,
  every field name matched against a declared set, anything failing discarded
  with its reason logged rather than stored. Phase 1 learned the prompt half
  too — "asked for an id without a menu, a model invents plausible ones" — so
  the extraction prompt carries the closed field list.
  **Shape validation is not enough: the claim must be grounded (review
  finding).** A well-typed entry citing a sentence that was never in the
  document is indistinguishable from a real one under type-checking alone, and
  it would become a forward signal. Because U4 already requires a verbatim
  source sentence, the recorder additionally verifies that sentence is a
  literal substring of the section text actually submitted for that year, and
  that the extracted value and period appear within it. That turns "auditable
  if a reader bothers" into "verified before storage", costs one string search,
  and closes fabrication and any instruction-like text embedded in a filing in
  the same move.
- **KTD4 — Provenance gates extraction *and* is carried per section through to
  consumption.** A section whose provenance is `fallback` is never sent to the
  extractor — cheaper, and no tokens spent on a chairman's letter that cannot
  answer the question.
  **Gating the input is necessary but not sufficient (review finding, P0).**
  Provenance is tagged per *section* while a report year usually has a mix:
  measured across the 29 section sidecars in the fetched corpus, **16 years
  carry mixed provenance and 10 have `mdna: fallback` alongside another
  `found` section**. Extraction therefore still runs for those years, and
  output keyed only by year would let a promises-kept reading be built from a
  chairman's letter while MD&A was never read — exactly Stop condition (b).
  So every extracted entry carries **the section it came from**, and every
  text-derived sub-metric declares **the section it requires** and reads
  indeterminate when that section's provenance for the year is `fallback`,
  regardless of what other sections that year yielded. The required-section
  map is declared, not left to the implementer:

  | Sub-metric | Required section | Why |
  |---|---|---|
  | `promises_kept_ratio` | `mdna` | Numeric guidance lives in MD&A; a chairman's letter offers aspiration, not targets a financials row can settle |
  | `capex_pipeline` | `mdna` | Announced capacity and commissioning dates sit in the operational review |
  | `tam_runway` | `mdna`, else `chairman` | Market-size statements appear in either; a chairman's framing of addressable market is usable evidence where a guidance number would not be |

  `tam_runway` is the one ranked fallback, and it is deliberate: it widens
  coverage for the sub-metric whose claim is qualitative anyway. The two
  sub-metrics that settle against numbers accept no substitute section.

- **KTD9 — `found` is not trustworthy on its own; a content gate stands between
  provenance and extraction.** Measured across all 18 `found` MD&A sections in
  the fetched corpus, **8 are not MD&A at all** — they open on audit-committee
  terms of reference, a Board's Report dividend clause, an auditor's key-audit-
  matters table, CSR activities, or HR safety programmes. The heading regex
  matched a *cross-reference* ("…as detailed in the Management Discussion and
  Analysis") and sliced from there. ASTRAL fails this way in all three retained
  years, so the failure is per-filer and persistent, not random.
  Phase 0 could tolerate it — the fallback text was only ever fed to Pass 1 as
  background. Phase 2 cannot: the extractor mines whatever it is handed, and
  audit-committee prose yields confident, well-formed, **wrong** guidance.
  Critically, **none of the other guards catch this.** KTD3's substring check
  confirms a quoted sentence really appears in the submitted text — and it
  does, because the submitted text is genuinely the auditor's report. Shape
  validation, closed vocabulary, and grounding all pass. Only content can
  distinguish it.
  So a section tagged `found` must additionally **look like the section it
  claims to be** — MD&A markers (economic or industry review, outlook, segment
  performance language) near the start of the slice — or it is downgraded to
  **`suspect`** and treated exactly as `fallback`: never sent to the extractor,
  sub-metric reads indeterminate. Provenance becomes three-valued
  (`found` / `suspect` / `fallback`), and the phase reports all three so its
  real yield is visible rather than flattered by a tag that is right about
  half the time.
- **KTD8 — Two hashes: a scoring regime and a forward-signal regime**
  *(session-settled: user-directed — chosen over excluding only a zero-weight
  metric's `params` from the single existing hash.)* The registry hash covers
  everything today, so tuning U3's justified-multiple band would move it and
  reset every ticker's momentum baseline — unrecoverably, since history is
  append-only and this phase writes no synthetic rows. That is circular in the
  worst place: **Phase 5 needs trajectory evidence to calibrate, and
  calibrating would destroy it.** So `registry_hash` narrows to what can move
  a composite — element weights, thresholds, gates, macro, `history_waiver_mcap`,
  and the definitions of scored metrics — and a second `forward_signal_hash`
  covers zero-weight metric definitions and the extraction schema. Score-history
  rows carry both; momentum groups on the scoring hash alone. This states
  KTD1's own logic rather than contradicting it: a metric that provably cannot
  move a score has no business in the hash that describes scoring.
  **Consequence for the settled reset (surfaced before the decision):** Phase 2's
  metrics no longer move the scoring hash, so the momentum baseline **does not
  reset** and the existing rows under `715479102494` stay comparable. The
  earlier "accept the reset" decision is not overridden — the reset simply
  stops being necessary, which was the outcome that decision was settling for
  in the first place.
- **KTD5 — Momentum refuses to diff across regimes and derives its cadence
  from row dates.** History rows carry `config_hash`; two rows under different
  hashes were produced by different rulers and are not comparable — the diff
  groups by hash and never spans a boundary. Interval comes from the actual
  dates, and every figure states the span it covers, so a wide gap can never
  read as quarterly-fresh momentum. Owner-confirmed: the baseline resets when
  this phase's metrics change the hash, and the diff reports insufficient
  history honestly rather than emitting a misleading zero.
- **KTD6 — Any new metric either emits no `raw_series` or documents its
  units, and none joins `SERIES_SAFE_METRICS`.** Phase 1's trap: `roiic`
  returns capital employed beside a percentage, `pe_vs_historical` returns P/E
  multiples beside a percentile, and a `persist_years` rule on either compares
  incompatible units and silently never fires. Headroom is the same shape of
  hazard — a ratio value with a multiples series behind it. Emitting no series
  is the default; a series requires a unit note in the docstring and does not
  earn allowlist membership in this phase.
  **The same open-contract hazard applies to flags (review finding).** The
  report attributes any unrecognised flag to the composite
  (`report_generator`'s `FLAG_ELEMENT_MAP.get(flag, "composite")`), and
  `checklist.build_flags_context` feeds every metric's flags to both LLM
  passes unfiltered. A new metric's flag would therefore render as an SQGLP
  signal and could move Pass 2's `suggested_action` on an otherwise unchanged
  ticker — R7's four listed quantities would all still hold. So every flag a
  Phase 2 metric emits is registered against the forward-signals section, and
  R7's non-regression check extends to the rendered flag list.
- **KTD7 — The pace modulator reads the corpus-median spread, not a single
  company's, and adjusts trigger thresholds by injection.**
  **Correcting the roadmap (review finding, P0).** v05 §11 says to wire
  `earnings_yield_vs_gsec` as the pace input, but that metric is
  **per-company**: `compute_earnings_yield_spread` reads
  `data["metadata"]["Stock P/E"]` — that one ticker's multiple — against the
  macro G-Sec yield. Using it would tighten entry when the *company* is
  expensive, which is the inverse of the modulator's purpose and a second
  per-name valuation test on a buy-zone trigger that already tests valuation.
  No index-level multiple exists in the pipeline, and adding a feed for one is
  a new data source §13 forbids.
  **The regime signal is computed, not hand-maintained.** The pace input is the
  **median `earnings_yield_vs_gsec` across the cached corpus** — a breadth
  reading assembled from per-name metrics that already exist, needing no new
  source and, decisively, no number anyone has to remember to refresh. An
  owner-set value would have needed an `as_of` date and stale-as-unset handling
  precisely because it decays in silence; a computed median cannot go stale.
  Two limits are recorded rather than hidden: the "market" here is ~20
  survivorship-selected names, so this reads *the corpus's* valuation rather
  than the market's; and the median is taken over the **cached corpus**, not
  the watchlist, so adding or dropping a tracked company does not move the
  signal underneath a decision. Only the floor and the tightening factor stay
  owner config (§14.1–.3 family), outside the hashed `macro:` block.
  Mechanism otherwise unchanged: `TriggerEvaluator` accepts an injected trigger dict and
  `advance()` an injected evaluator, so the modulator derives a
  threshold-tightened copy of the buy-zone trigger and passes it in. Adding a
  spread *condition* to the trigger instead would make macro a gate — a
  company blocked from entry by the market's valuation rather than its own,
  which §11 forbids. A run-level input also keeps the single-evaluator seam
  valid: `advance()` builds one evaluator before the ticker loop, which a
  per-company reading could not have supplied — the median is computed once
  per run, ahead of that construction.
  Modulation applies only to entry (`→ probe`, already owner-confirmed) and is
  recorded in the proposal evidence so a tightened threshold is never invisible.

### Session-settled decisions

Three forks were put to the owner at planning time and closed. They are
recorded here as settled input, not open questions.

- **Momentum baseline resets rather than being backfilled**
  *(session-settled: user-directed — chosen over regenerating comparable
  history from cached data via the backtest's truncation machinery: accepts
  that momentum produces nothing usable for several quarters in exchange for
  a simpler phase and no synthetic rows in the log.)* Governs the
  no-synthetic-backfill scope boundary.
  **Superseded in part by KTD8.** The concern that prompted this decision — a
  recurring reset, since Phase 3's lane gates and Phase 5's calibration each
  move the hash — is largely designed out by the two-hash split: Phase 2 no
  longer resets the baseline at all, and later phases reset it only when they
  change something that can actually move a composite (Phase 5's calibration
  genuinely does; Phase 3's lane gates do not touch SQGLP scoring). The
  no-backfill half stands: this phase still writes no synthetic rows, and
  organic rows remain frozen under their hash rather than regenerable.
- **The full forward-growth module ships in this phase**
  *(session-settled: user-directed — chosen over shipping only the
  deterministic quarterly-momentum sub-metric, and over running extraction as
  a separate opt-in command: accepts one extra LLM call per analysis to
  address what the roadmap calls the biggest conceptual weakness.)* Governs
  R3, R4 and units U4–U5.
- **New signals get their own report section**
  *(session-settled: user-directed — chosen over leaving them visible only to
  Pass 2 and the lifecycle triggers.)* Governs R8 and U7.

### Assumptions

- A1. **Real MD&A availability is far below the raw `found` rate, and the
  phase must be judged against the real one.** Three figures, each smaller than
  the last:
  1. `mdna` provenance is `found` on 12 of 20 reports (~60%) — the Phase 0
     measurement.
  2. But only **10 of the 18** `found` MD&A slices in the corpus are actually
     MD&A; the other 8 are auditor's reports, governance, CSR or HR (KTD9). So
     usable MD&A is nearer **~55% of `found`**, i.e. roughly a third of reports.
  3. `promises_kept_ratio` needs **two consecutive years** of usable MD&A,
     which on today's corpus is likely **1–2 tickers**, not the 3 a
     `found`-only count suggests.
  This is the provenance contract plus the content gate working as designed.
  Read the sweep against these numbers, or a correctly-functioning phase will
  look like a failure — and conversely, a yield markedly *above* them means
  the content gate is too permissive.
- A2. The backtest will auto-exclude **U5's four forward-growth sub-metrics**,
  because `_truncate` never loads `quarterly` or `annual_report_sections`;
  excluded metrics are already reported in its exclusion list.
  `rerating_headroom` is **not** excluded — it reads only truncatable frames
  (financials, ratios, price) and will compute inside the backtest. That is
  harmless: at zero weight it enters neither the correlations nor the coverage
  floor.
- A3. Momentum has no usable history the day this phase lands (A2 of the
  owner decision). Its first useful output is several runs away.
- A4. No organically eligible ticker exists in the fetched corpus (Phase 1
  finding). Validation that needs a post-qualification state will again seed
  states in a scratch store so triggers evaluate real metrics.
- A5. Only **5 of 20** BSE codes currently retain ≥2 annual reports at all
  (Phase 0's `max_reports: 3` applies from its landing forward, not
  retroactively). Refetching is the prerequisite for wider promises-kept
  coverage; until then the sub-metric reads indeterminate for most tickers.

### Risks

| Risk | Why it matters here | Mitigation |
|---|---|---|
| A new metric silently moves the composite | R7 is the phase's hardest constraint, and the scorer's divisor is the sum of weights that *actually scored* — so a weighted metric re-weights every sibling. A drift here corrupts every stored score and every comparison against them | Zero weight plus an unweighted element (KTD1); a per-unit non-regression test on U3 and U5; an empirical before/after `scores.json` diff in the Verification Contract |
| Extraction produces confident nonsense | The upstream parser validates nothing, and a chairman's letter will happily yield plausible "guidance" that was never guidance | Provenance gates the input (KTD4) so bad text never reaches the model; boundary validation discards what fails (KTD3); every entry retains its verbatim source sentence for audit |
| LLM cost rises per analysis | One extra call on every `analyze` run, against a phase whose value is unproven until history accumulates | Config-driven char budget mirroring `pass1_ar_char_budget`; no call at all when a ticker has no `found` sections; the pass is skippable with the existing `--no-llm` path |
| Momentum reads as flat rather than absent | The common case at landing is *no usable history* (A3). A zero delta and an unknown delta look identical in a table and mean opposite things | Insufficient-history is a distinct outcome, tested against a genuine zero delta (U2) |
| Macro leaks into per-name decisions | A pace modulator that tightened exits or gates would let market valuation override company evidence — explicitly forbidden by §11 | Modulation applies to entry thresholds by injection only (KTD7); U6 tests assert kill-switches and gates are identical under a compressed spread |
| Phase lands with no live signal | The day-one yield compounds four assumptions in the same direction: no momentum history (A3), promises-kept realistically available on 1–2 of 20 tickers once the content gate is applied (A1/A5, KTD9), and no organically eligible ticker for the modulator to slow (A4). Every stated check could pass while the phase produces almost nothing actionable | The provenance sweep carries a **minimum-yield bar** set before the sweep runs: `rerating_headroom` and `quarterly_momentum` must produce values on a stated majority of the corpus, and at least one text-derived sub-metric must produce a value on at least one ticker. Below that, the phase is not done |
| Backtest sample silently shrinks | A weighted metric that errors drops rows below the coverage floor, quietly reducing the backtest's usable N | Zero weight keeps the new metrics out of the coverage denominator entirely; A2 expects them in the exclusion list, and the Verification Contract checks the backtest still runs |

### High-Level Technical Design

**The extraction seam.** The load-bearing shape of this phase is that LLM work
happens *upstream* of the compute engine and reaches it only as data:

```mermaid
flowchart LR
  subgraph fetch["Stage 1 — fetch"]
    AR["annual_report_sections<br/>{year: {section: text, provenance}}"]
    Q["quarterly.csv"]
    F["financials / ratios / price<br/>(existing frames)"]
  end
  subgraph llm["Stage 1.5 — extraction pass (NEW)"]
    EX["forward-growth extractor<br/>closed field vocabulary"]
    V["boundary validator<br/>discard + log on failure"]
  end
  subgraph compute["Stage 2 — compute engine (offline, deterministic)"]
    M1["promises_kept_ratio"]
    M2["capex_pipeline"]
    M3["tam_runway"]
    M4["quarterly_momentum"]
    M5["rerating_headroom"]
  end
  subgraph out["Stage 3+ — consumers"]
    S["scores.json<br/>weight 0 — composite unchanged"]
    R["report: Forward Signals"]
    T["lifecycle triggers"]
  end

  AR -->|provenance == found only| EX
  EX --> V
  V -->|data#91;'forward_growth'#93;| M1 & M2 & M3
  Q --> M4
  F --> M5
  M1 & M2 & M3 & M4 & M5 --> S --> R
  S -->|offline metrics only| T

  classDef new fill:#fff3cd,stroke:#856404
  class EX,V,M1,M2,M3,M4,M5 new
```

`watchlist advance` re-scores with `use_llm=False`, so the extraction pass
never runs on the lifecycle path — only `quarterly_momentum` and
`rerating_headroom` are reachable by triggers, and this phase declares no
trigger on either (see Scope Boundaries).

Sections whose provenance is `fallback` never enter the extractor, so their
sub-metrics have no input and read indeterminate — R5 holds structurally.

**Momentum regime partitioning.** History rows are comparable only within a
`config_hash`. The diff walks each regime independently and never bridges:

```
history:  [h=715…] [h=715…] [h=715…] │ [h=9a2…] [h=9a2…]
           Jun-06    Jul-14    Aug-06 │  Aug-20    Sep-30
              └────────┴─────────┘    │     └────────┘
              momentum, 61d span      │  momentum, 41d span
                                      │
                          regime boundary — never diffed across
                          (this phase's new metrics move the hash)
```

---

## Implementation Units

### U1. Test fixture builders for the new data shapes

- **Goal:** Give every later unit synthetic inputs that mirror production
  column names, so no Phase 2 test reaches for `raw_data/`.
- **Requirements:** enables R1–R6.
- **Dependencies:** none.
- **Files:** `tests/conftest.py`
- **Approach:** add builders mirroring the real fetched schemas —
  1. `make_quarterly(periods, **overrides)` with the exact columns the Phase 0
     parser writes (`quarter, revenue, expenses, operating_profit, opm_pct,
     other_income, interest, depreciation, pbt, tax_pct, pat, eps`).
  2. `make_ar_sections(years, provenance)` producing the
     `{year: {section: {text, provenance, start_page}}}` shape, able to emit
     both `found` and `fallback` so provenance tests are cheap.
  3. `make_history_rows(...)` producing score-history rows across one or more
     `config_hash` values and arbitrary dates.
  4. `adj_close` on `make_price`, which today emits `close` only.
- Wire the new keys into `make_data` so a default `data` dict carries them.
- **Patterns to follow:** the existing builders' docstring convention and the
  module docstring's rule about mirroring real column names.
- **Test scenarios:**
  - `make_quarterly` output contains every column `checkpoint_vocabulary.yaml`
    references under `source: quarterly` (`revenue`, `opm_pct`,
    `operating_profit`, `pat`, `eps`), so a vocabulary entry cannot silently
    drift from the fixture.
  - `make_ar_sections(provenance="fallback")` yields sections whose provenance
    is uniformly `fallback`; `"found"` yields the converse.
  - `make_history_rows` spanning two `config_hash` values produces rows that
    `load_history` returns without collapsing.
  - `make_price` still emits `close` unchanged for existing callers, and
    `adj_close` when asked.
- **Verification:** the existing suite passes untouched — these are additive
  builders, and no existing test's fixture output changes.

### U2. Score trajectory momentum

- **Goal:** Turn the append-only history into per-ticker momentum, honestly
  labelled.
- **Requirements:** R1.
- **Dependencies:** U1.
- **Files:** `boundless100x/trajectory.py` (new), `boundless100x/service.py`,
  `tests/test_trajectory.py` (new)
- **Approach:** a reader over `score_history.load_history` that
  1. partitions rows by `config_hash` and never diffs across a boundary
     (KTD5), and separately never mixes `synthetic: true` rows with organic
     ones in a single figure;
  2. computes composite and per-element deltas between consecutive rows within
     a regime;
  3. derives the interval from the actual row dates and returns it alongside
     every figure, so the caller cannot present an annual gap as quarterly
     momentum;
  4. returns an explicit insufficient-history outcome — not zero, not `None`
     silently — when a regime holds fewer than two rows. Momentum has no
     usable history the day this lands (A3), so this is the *common* path at
     first and must read as "not enough history yet", never as "flat".
- Keep this out of `score_history.py`: that module's contract is append and
  read: adding interpretation there muddies an append-only store with a
  consumer.
- **Wire it to the result.** Add a `momentum` field to `AnalysisResult` and
  populate it at a new Stage 4.7 in `service.analyze()`, reading through
  `self.history_path`. Without this the module has no caller and the report
  has nothing to render. Routing through the service also preserves the
  per-caller history redirect that the test-isolation fixture depends on —
  having `ReportGenerator` read history directly would bypass it.
- **Patterns to follow:** `score_history.load_history`'s existing
  regime-preserving dedupe; the indeterminate-vs-value discipline used
  throughout `lifecycle/`.
- **Test scenarios:**
  - Two rows in one regime produce a composite delta and an element delta
    matching hand-computed values.
  - Rows spanning two `config_hash` values produce **two** momentum series,
    never one bridging figure.
  - A regime with one row reports insufficient-history, and that outcome is
    distinguishable from a genuine zero delta between two equal scores.
  - The reported interval equals the actual day gap between the rows used —
    a 365-day gap is labelled as such, not assumed quarterly.
  - Synthetic and organic rows in the same regime are not silently averaged
    into one figure.
  - An empty history reads as insufficient rather than raising.
  - An element present in the later row but absent in the earlier one yields
    no delta for that element rather than treating the absence as zero.
- **Verification:** momentum figures reproduce from stored rows without
  re-scoring; the composite of any ticker is untouched (this unit adds no
  metric).

### U3. Re-rating headroom metric

- **Goal:** Measure constructively what valuation metrics currently only veto:
  how much multiple expansion the company's own fundamentals would justify.
- **Requirements:** R2, R7.
- **Dependencies:** U1.
- **Files:** `boundless100x/compute_engine/metrics/builtin/valuation.py`,
  `boundless100x/compute_engine/metrics/elements/price.yaml`,
  `tests/test_rerating_headroom.py` (new)
- **Approach:** a quality-justified multiple built from **own-fundamentals
  bands** (v05 §14.5 — never sector-relative, which would reintroduce the peer
  comparison v04 deliberately removed): map the company's RoCE, growth rate,
  and longevity/consistency readings onto a justified-multiple band declared in
  the metric's YAML `params`, then express headroom as the relationship between
  that justified multiple and the multiple actually being paid. Register at
  **`weight: 0.0`** in `price.yaml` (KTD1). Record `price_basis` in metadata,
  matching the existing convention that valuation metrics use raw close against
  as-reported EPS. Emit **no `raw_series`** (KTD6) — a series of multiples
  behind a ratio value is exactly the unit mismatch that trapped Phase 1.
- **Settled shape** — R8 requires each signal to render with an interpretation
  band, which cannot be specified until the number's shape is fixed, so these
  are decided here rather than left to the implementer:
  - *Output:* `headroom_pct = (justified_multiple / current_multiple - 1) × 100`.
    A **ratio expressed as a percentage**, not a difference in multiple points —
    so +40 means "fundamentals justify a multiple 40% above what is being paid"
    and reads the same for a company on 15× as on 60×. Sign convention:
    **positive means room to re-rate up**, matching the metric's name.
  - *Inputs and windows:* RoCE and growth on the same 5-year window the
    existing `roce_5yr_avg` and `*_cagr_5yr` metrics use, so the bands are
    anchored to numbers already in `scores["details"]`; longevity from
    `roce_consistency`. Reusing the existing windows keeps a reader from having
    to reconcile two definitions of the same company's RoCE.
  - *Justified-multiple table:* a banded lookup in YAML `params` mapping
    (RoCE band × growth band), adjusted by a longevity multiplier. Starting
    values are declared in the YAML as **starting points awaiting Phase 5
    simulator evidence**, in the same spirit as the trigger thresholds.
  - *Error cases:* non-positive earnings, a missing RoCE or growth input, or a
    missing price each yield `MetricResult(error=...)`. There is no default
    justified multiple — an unknown quality profile must not silently receive
    the middle band.
  - *Display bands (feeds R8):* headroom ≥ +25% reads favourable, −25% to
    +25% reads fair, ≤ −25% reads stretched. Owner-editable in the same
    `params` block.
- **Technical design** *(directional, not specification)*: the band table lives
  in YAML `params` rather than in code, so tuning it is a config edit matching
  the metric-registry pattern — and under KTD8 that tuning no longer disturbs
  score history.
- **Patterns to follow:** `compute_pe_percentile`'s construction of a
  historical band from each year-end close over that year's EPS — and its
  docstring, which documents why the naive alternative is wrong;
  `_get_annual_rows` for windowing.
- **Test scenarios:**
  - A high-RoCE, high-growth, consistent company trading below its justified
    multiple yields positive headroom.
  - The same company trading above it yields negative headroom.
  - A low-quality company gets a *lower* justified multiple, so an identical
    traded multiple produces less headroom than the high-quality case — the
    band responds to fundamentals, not just to price.
  - Missing RoCE, missing growth, or missing price each yield
    `MetricResult(error=...)`, never a default multiple.
  - Negative or zero earnings yield an error rather than a nonsensical ratio.
  - A missing RoCE or growth input errors rather than falling back to a middle
    band — an unknown quality profile must not receive a default multiple.
  - Sign convention holds: a company trading below its justified multiple
    reports **positive** headroom.
  - The percentage form is scale-free: two companies with the same
    justified/current ratio but very different absolute multiples report the
    same headroom.
  - The metric emits no `raw_series`.
  - **Non-regression:** scoring a fixture before and after this metric is
    registered leaves `composite`, every element score, and
    `coverage["composite"]` identical.
- **Verification:** the metric appears in `scores["details"]` with
  `"weight": 0` and `"score": None`; the composite for a cached ticker is
  byte-identical to its pre-change value.

### U4. Forward-growth extraction pass

- **Goal:** Get guidance, capex, and TAM statements out of annual-report prose
  and into structured data the compute engine can read offline.
- **Requirements:** R3, R4, R5.
- **Dependencies:** U1.
- **Files:** `boundless100x/llm_layer/prompts/forward_growth_extraction.txt`
  (new), `boundless100x/llm_layer/orchestrator.py`,
  `boundless100x/llm_layer/forward_growth.py` (new — the boundary validator),
  `boundless100x/config.yaml`, `boundless100x/service.py` (Stage 1.5 call
  site), `tests/test_forward_growth_extraction.py` (new)
- **Approach:**
  0. **A content gate ahead of the prompt (KTD9).** Each `found` section is
     checked for markers of the section it claims to be before anything is
     sent; a slice that fails is downgraded to `suspect` and excluded exactly
     as `fallback` is. This runs on the extraction path rather than in the
     fetcher, so Phase 0's sidecars stay as they are and the gate can be tuned
     without a refetch.
  1. A prompt that receives only gated `found`-provenance sections (KTD4, KTD9)
     and asks
     for a closed set of fields per report year — guidance statements with
     their metric and target, announced capex/capacity with commissioning
     dates, and addressable-market statements — each with the verbatim source
     sentence so a reader can audit the extraction, **and the section it came
     from** (KTD4), since a year's sections rarely share one provenance.
  2. A validator in `forward_growth.py` mirroring
     `checkpoints.record_from_pass2`: type-check every field, match names
     against the declared set, discard anything failing with its reason
     logged. It must tolerate a string where a list belongs, a partial object,
     a null entry, and the key being absent entirely — `_parse_json_response`
     validates nothing (KTD3).
  3. Config: model and char budget alongside the existing
     `pass1_ar_char_budget` precedent.
  4. Output lands in `data["forward_growth"]`, keyed by year, each entry
     carrying its source section.
  5. **Call site is `service.analyze()` as a new Stage 1.5**, between Stage 1's
     fetch and Stage 2's compute. `DataFetcherSuite.fetch_all()` takes no
     `use_llm` argument, so wiring the pass there would fire a paid call on
     every `--no-llm` run — including `screen`'s per-candidate `analyze_quick`
     and every `watchlist advance`.
  6. **Hydration and extraction are separately gated — this is the load-bearing
     split.** Stage 1.5 *always* reads a valid sidecar into
     `data["forward_growth"]`, on every run including `use_llm=False`. Only
     *creating or refreshing* a sidecar calls the model, and that half is gated
     on `use_llm and self._llm`. Gating the whole stage would mean the cache is
     never read on the very paths it exists to serve: `watchlist advance` would
     re-read as indeterminate forever, and the scope boundary's promise that
     the sub-metrics become available "once extraction output is cached across
     runs" could never come true.
  7. **Cache validated output in a per-ticker sidecar** keyed by report year,
     mirroring the `.sections.json` cache in
     `download_annual_reports.extract_sections()`. An annual report does not
     change after filing; without this the corpus-wide validation sweep and
     every re-analysis pay again for identical text.
  8. **Version the sidecar** against the source text, the field schema, the
     prompt, and the model id — the same invalidation discipline
     `extract_sections()` applies when its requested section set changes. A
     prompt or model change that silently reused stale extractions would be
     indistinguishable from the new prompt working.
- **Execution note:** build the validator against recorded malformed responses
  before wiring the live call — the failure modes are the point of this unit,
  and they are cheaper to get right without an API in the loop.
- **Patterns to follow:** `checkpoints.record_from_pass2` for defensive
  boundary validation and demotion-with-reasons;
  `checkpoints.vocabulary_prompt_block` for injecting a closed vocabulary into
  a prompt; the quadruple-brace `{{{{` JSON-escaping convention in prompt
  templates.
- **Test scenarios:**
  - A well-formed response for two report years yields structured entries for
    both.
  - A response naming a field outside the declared set has that field
    discarded with a logged reason, while valid sibling fields survive.
  - A string where a list belongs, a partial object, a null entry, and an
    entirely absent key each degrade to empty output without raising.
  - A `parse_error` response (the `_parse_json_response` failure shape) yields
    empty output without raising.
  - Sections carrying `fallback` provenance are **not** included in the prompt
    payload — assert on what was sent, not only on what came back.
  - A year with mixed provenance sends only its `found` sections, and every
    returned entry is tagged with the section it came from.
  - A ticker with no `found` sections produces no extraction call at all.
  - Extracted entries retain the verbatim source sentence for audit.
  - A second run over an unchanged report reads the sidecar and makes no API
    call.
  - **A `use_llm=False` run with a valid sidecar still populates
    `data["forward_growth"]`** — hydration is not gated — and makes no API
    call. This is the `watchlist advance` path.
  - A `use_llm=False` run with **no** sidecar leaves the key absent and makes
    no API call.
  - A quoted sentence that does not appear in the submitted section text is
    discarded with its reason logged, even when every field is well-typed.
  - A changed prompt, model id, or field schema invalidates the sidecar rather
    than reusing it.
  - **A `found` section whose text is actually an auditor's report or
    governance prose is downgraded to `suspect` and never reaches the prompt**
    (KTD9) — use the real ASTRAL slice ("terms of reference of the Audit
    Committee…") as the fixture, since it is a real failure from the corpus
    rather than an invented one.
  - A genuine MD&A slice ("MANAGEMENT DISCUSSION AND ANALYSIS ECONOMIC
    REVIEW…") passes the gate — the check must not reject the 10 of 18 that
    are correct.
- **Verification:** an end-to-end run on a cached ticker with `found` MD&A
  produces populated `data["forward_growth"]`; a ticker whose sections all fell
  back produces an empty one with no API call made.

### U5. Forward-growth sub-metrics

- **Goal:** Four registered metrics turning the extraction output and the
  quarterly series into forward signal.
- **Requirements:** R3, R5, R7.
- **Dependencies:** U1, U4.
- **Files:**
  `boundless100x/compute_engine/metrics/builtin/forward_growth.py` (new),
  `boundless100x/compute_engine/metrics/elements/forward_growth.yaml` (new),
  `tests/test_forward_growth_metrics.py` (new)
- **Approach:** four metrics, all `weight: 0.0` in an element deliberately
  **absent from `element_weights`** (KTD1):
  1. `promises_kept_ratio` — guidance extracted from report year N against
     what the financials actually delivered in year N+1. Requires ≥2 report
     years with usable guidance; fewer reads indeterminate (A5).
     **Settled semantics** — these are decisions, not implementation detail,
     because each changes what the number means:
     - *What counts as a promise:* a guidance statement carrying a **named
       financial quantity and a target value or range** that a later
       `financials`/`ratios` row can settle — revenue, PAT, margin, capex
       spend. Directional prose with no number ("we expect strong growth") is
       recorded by the extractor but is **not** a promise and does not enter
       the denominator; counting unfalsifiable statements would let vague
       management score perfectly.
     - *What counts as kept:* delivered ≥ **95%** of the guided value
       (owner-set tolerance in YAML `params`). Indian annual-report guidance is
       frequently a range or a rounded target, so exact matching would read
       rounding as broken credibility. For a range, the **lower bound** is the
       promise.
     - *Fiscal-period mapping:* guidance in the report **for** FY N is settled
       against the financials row whose period ends in FY N — Screener's
       `Mar YYYY` column. The report *published* in year N typically guides
       FY N+1, so the settling row is one column to the right of the report's
       own year. A guidance statement whose target period cannot be resolved to
       a column is discarded, not guessed.
     - *Denominator:* promises that came due (their target period has a
       settling row). A promise whose period has not yet arrived is pending and
       enters neither numerator nor denominator — the same
       due-versus-not distinction Phase 1's checkpoints already draw.
  2. `capex_pipeline` — announced capacity and commissioning dates as forward
     runway for volume growth.
  3. `tam_runway` — whether the stated addressable market leaves arithmetic
     room for the growth rate the thesis assumes.
  4. `quarterly_momentum` — fully offline from `data["quarterly"]`: is growth
     accelerating or decelerating now. **This is a second difference, not a
     growth rate.** Each year-over-year figure compares against the same
     quarter four periods back, never the previous quarter, so seasonality does
     not read as a trend (Phase 1's checkpoint rule); momentum is then the
     change between *consecutive YoY figures* — YoY(t) minus YoY(t−1). A single
     YoY number is a growth **level** and would answer "is it growing", not "is
     growth speeding up", which is what this sub-metric claims to measure. Two
     YoY figures therefore need **at least six quarters**; eight gives a
     reading not dominated by one quarter's noise. Screener renders ~11–13, so
     both are reachable.
- Each text-derived metric declares `forward_growth` in its `inputs`, which
  catches the key being **wholly absent** (a `--no-llm` run). It does **not**
  catch an empty one: the engine's input check only treats a value as missing
  when it has a truthy `.empty` attribute, i.e. a DataFrame — a present-but-
  empty dict passes straight through (`engine.py`, "Allow dicts … even if
  empty"). So each sub-metric additionally returns `MetricResult(error=...)`
  when its own year-keyed entry is absent, empty, or sourced from a section
  whose provenance was `fallback` (KTD4). No unit may delegate its
  indeterminate to the engine.
- **Patterns to follow:** `checkpoints._series_value`'s YoY implementation and
  its `_QUARTERS_PER_YEAR` constant; the `MetricResult(error=...)` convention
  for unavailable rather than defaulted.
- **Test scenarios:**
  - Guidance met in the following year yields a high promises-kept ratio;
    guidance missed yields a low one.
  - One report year only yields indeterminate, not a perfect score.
  - A sub-metric whose section was `fallback` (so extraction produced nothing)
    reads indeterminate — asserted per sub-metric, since this is R5.
  - **Mixed provenance:** a year with `mdna: fallback` but `chairman: found`
    still yields indeterminate promises-kept — the case that occurs in 10 of
    the 29 real report-years, and the one a year-keyed check would miss.
  - A present-but-empty `data["forward_growth"]` yields an error per
    sub-metric, not a computed value.
  - `quarterly_momentum` compares against four quarters back: a series with
    strong seasonality yields the YoY figure, not the sequential one.
  - **Steady growth yields momentum near zero, not a large positive number** —
    a company growing a constant 20% YoY for eight quarters is not
    accelerating. This is the assertion that distinguishes a second difference
    from a growth rate, and it fails against a single-YoY implementation.
  - Decelerating growth (YoY falling 30% → 20% → 12%) yields negative momentum
    even though every individual YoY figure is positive.
  - A quarterly series shorter than six periods yields indeterminate — two YoY
    figures cannot be formed.
  - An absent `quarterly` frame yields an error, not zero.
  - **Non-regression:** registering all four leaves `composite`, every element
    score, and `coverage["composite"]` identical on a fixture.
  - The `forward_growth` element does not appear in `scores["elements"]`.
- **Verification:** all four appear in `scores["details"]` at weight 0; the
  backtest lists them as excluded metrics (A2) rather than failing.

### U6. Deployment-pace modulator

- **Goal:** Deploy more cautiously when the whole market is expensive, without
  letting macro veto a company.
- **Requirements:** R6.
- **Dependencies:** U1.
- **Files:** `boundless100x/lifecycle/pace.py` (new),
  `boundless100x/lifecycle/advance.py`, `boundless100x/config.yaml`,
  `tests/test_pace_modulator.py` (new)
- **Approach:** compute the **median `earnings_yield_vs_gsec` across the cached
  corpus** as the regime reading (KTD7 — a single company's spread is
  per-name and would invert the unit's purpose). When that median sits below a
  configured floor, derive a **threshold-tightened copy** of the buy-zone
  trigger and pass it to the evaluator via the injection seam
  `TriggerEvaluator` and `advance()` already expose. Only the floor and the
  tightening factor are owner config (§14.1–.3 family), outside the hashed
  `macro:` block — the spread itself is computed, so there is no value to keep
  current and no staleness to handle.
  **`advance()` becomes two-pass:** score every ticker, compute the median,
  build the (possibly modulated) evaluator once, then evaluate. Today it
  constructs the evaluator before the ticker loop and evaluates inline; the
  median is not knowable until scoring finishes, so the loop splits. Report how
  many tickers contributed to the median alongside the reading — a median over
  three names is not a regime signal, and the count is what tells a reader
  that. When too few tickers carry a usable spread, **do not modulate**: an
  unknown macro reading must not tighten entry any more than it may loosen it.
  When modulation applies, say so in the proposal evidence, including the
  median and its contributor count.
- **Execution note:** assert on kill-switch behavior explicitly; the failure
  mode that matters is macro leaking into exit logic.
- **Patterns to follow:** `advance()`'s existing evaluator injection;
  `_PRECEDENCE` for how proposals are ranked.
- **Test scenarios:**
  - A wide spread leaves buy-zone thresholds untouched — the unmodulated
    trigger set is used.
  - A compressed spread tightens them, and a company that would have entered
    the buy zone at the standard threshold no longer proposes entry.
  - Modulation is named in the proposal evidence when it applied.
  - **Kill-switches are unaffected under a compressed spread** — an
    `exit_review` proposal fires identically either way.
  - **Eligibility gates are unaffected** — the verdict is identical either way.
  - Too few tickers carrying a usable spread leaves thresholds unmodified.
  - The median and its contributor count appear in the proposal evidence
    whenever modulation applied.
  - **The median is taken over the cached corpus, not the watchlist** — adding
    or removing a watchlist entry does not shift the reading, so a decision
    cannot move underneath you for reasons unrelated to valuation.
  - **One expensive company does not modulate the run** — a corpus where a
    single ticker's spread is deeply negative but the median is wide leaves
    thresholds untouched. This is the assertion that distinguishes a regime
    reading from the per-name check the roadmap originally implied.
  - Modulation applies to entry only: an `→ exit_review` transition is not
    threshold-adjusted. (`triggers.yaml` declares no `→ scale` transition.)
- **Verification:** with a compressed spread configured, `watchlist advance`
  proposes strictly fewer entries and exactly as many exit reviews.

### U7. Forward-signals report section

- **Goal:** Make the phase's output visible to the reader, without implying it
  moved the score.
- **Requirements:** R8.
- **Dependencies:** U2, U3, U5.
- **Files:** `boundless100x/output/report_generator.py`,
  `boundless100x/output/templates/sqglp_report.html.j2`,
  `boundless100x/output/templates/sqglp_report.md.j2`,
  `tests/test_report_forward_signals.py` (new)
- **Approach:** a new report section rendering headroom, the four
  forward-growth sub-metrics, and score momentum. It must be **separate from
  the SQGLP score drilldown**, which today skips `weight == 0` metrics and
  reads display names from a hardcoded map — reusing that path would either
  require faking a weight or would silently drop the metrics. Each signal
  renders with its direction of goodness and interpretation band (R8). Momentum
  renders from `result.momentum` (U2) — the report never reads score history
  itself. The section carries a one-line statement that these signals inform
  the thesis but do not contribute to the composite, so a reader is never left
  inferring whether the score already includes them. Indeterminate sub-metrics
  render as unknown with their reason, matching how eligibility gates already
  render.
- **Patterns to follow:** `_build_eligibility_badge`'s shape — a builder
  returning a dict the template renders, with reasons carried through.
- **Test scenarios:**
  - A result with all forward signals present renders them in both HTML and
    Markdown.
  - An indeterminate sub-metric renders as unknown with its reason, not as
    zero or blank.
  - A result with no forward signals at all (a ticker predating this phase)
    renders the report without the section and without raising.
  - The SQGLP drilldown is unchanged — the same metrics and scores render as
    before this unit.
  - The section states that these signals do not contribute to the composite.
  - Each rendered signal carries its direction of goodness and band (R8).
  - No flag emitted by a Phase 2 metric renders under an SQGLP element (KTD6).
  - An insufficient-history momentum outcome renders as "not enough history
    yet", never as a zero delta.
- **Verification:** reports generate for a cached ticker in all three formats;
  the score drilldown output is identical to its pre-change form.

---

## Verification Contract

- Full suite green via `venv/bin/python -m pytest tests/` (network tests
  remain deselected).
- **R7 regression — the central proof of this phase.** Phase 0 could prove
  additive-ness *structurally* ("no metric declares `quarterly` or
  `annual_report_*` among its inputs"). Phase 2 is the first phase to consume
  those keys, so that proof no longer holds and must be replaced with an
  empirical one: score a cached ticker before and after the phase and diff
  `scores.json` — `composite`, every element score, and `coverage` must be
  byte-identical, with `details` gaining entries at `"weight": 0` and
  `"score": null`. Also confirm `eligibility.json` is unchanged, **and that no
  new flag renders under an SQGLP element in the report** (KTD6) — flags reach
  both the report's element attribution and the LLM prompts, so a leak there
  moves `suggested_action` while all four listed quantities still hold.
- **Three-bucket provenance split reported.** Run the forward-growth module
  across the fetched corpus and report, per sub-metric, the counts of
  **found / suspect / fallback** — not the two buckets Phase 0 used. The
  `suspect` bucket is the point: it makes visible how often a `found` tag was
  wrong, which a two-bucket report would hide inside an inflated success
  count. Read the result against A1's layered rates and the minimum-yield bar
  in Risks. A phase that produces all indeterminates has not been validated;
  a phase whose `suspect` count is near zero has a content gate that is not
  actually gating.
- **Momentum honesty check.** With fewer than two rows in the current regime,
  the diff reports insufficient history rather than zero. This is the expected
  state at landing (A3), so it must be verified, not assumed.
- Backtest still runs, listing **U5's four forward-growth sub-metrics** as
  excluded (A2) rather than erroring, with `rerating_headroom` computing
  normally inside it at zero weight.

## Definition of Done

All seven units merged with tests; the R7 before/after diff performed and its
result recorded in this plan; the found / suspect / fallback split reported; the
momentum honesty check performed (insufficient history reads as insufficient,
not as zero); the backtest confirmed still running with the forward-growth
sub-metrics listed as excluded;
`CLAUDE.md` updated with the Phase 2 contracts (extraction seam and its
dependency direction, zero-weight/unweighted-element rule, momentum regime
partitioning, pace-modulation boundary); v05 roadmap Phase 2 checked off with
a pointer here.
