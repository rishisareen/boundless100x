---
title: Phase 1 Lifecycle Foundation - Plan
type: feat
date: 2026-08-06
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: Design/Financial Model v05 - Phased Growth Roadmap.md (§12 Phase 1)
execution: code
---

# Phase 1 Lifecycle Foundation - Plan

> **Rev 2026-08-06a (fresh-start decision, owner-directed):** old outputs are
> discarded rather than migrated. `output/reports/` (141 MB of stale reports)
> was deleted and `watchlist.json` reset to empty at commit `4334156`, whose
> history still holds the two prior entries (ASTRAL "Pipe sector leader",
> SUPREMEIND "Closest peer") should they be wanted back. **`raw_data/` was
> deliberately kept** — it is fetched input, not output, and re-acquiring
> 261 MB including 22 BSE annual-report PDFs would cost network time and
> rate-limited scraping for no benefit. Consequences: R3 (lossless migration)
> is struck, KTD5 becomes a fresh-schema decision, and U3 loses its migration
> logic and tests.

## Goal Capsule

- **Objective:** Turn the watchlist into the persistence layer for the v05
  investment lifecycle: declared state transitions in YAML, a deterministic
  evaluator that mirrors the eligibility gates, structured checkpoints the
  code can actually check, and a `watchlist advance` command that proposes
  transitions with the evidence that caused them.
- **Authority:** v05 roadmap §4 (lifecycle), §5 (trigger registry), §6.1
  (core-lane kill-switches), §12 Phase 1, and §14.4 (owner confirms anything
  that moves money) govern behavior; this plan's Key Technical Decisions
  govern mechanism; `CLAUDE.md` governs style. Where the plan conflicts with
  observed code reality, surface it rather than guessing — Phase 0 corrected
  four planning assumptions this way and the corrections were the valuable
  part.
- **Stop conditions:** Stop and surface if (a) a declared trigger cannot be
  expressed against existing metric ids, flags, or series without inventing a
  metric — Phase 1 adds no metrics, so the trigger must be reshaped or
  deferred; (b) the watchlist migration cannot preserve every existing entry
  losslessly; or (c) any change alters composite scores, eligibility verdicts,
  or the resolved action — the lifecycle sits *after* those and consumes them.
- **Execution profile:** Code with unit tests per unit; trigger and checkpoint
  tests run on synthetic `MetricResult`/DataFrame fixtures per
  `tests/conftest.py`, never on `raw_data/`. The end-of-phase replay uses
  refetched real tickers.
- **Tail ownership:** Implementer owns commit hygiene and the replay
  validation: refetch the validation tickers, run `watchlist advance`, and
  confirm every proposed transition cites the trigger evidence that caused it.

---

## Product Contract

### Summary

Six additive capabilities: a `triggers.yaml` registry and evaluator built on
the eligibility-gate precedent; a checkpoint vocabulary bound to the columns
Phase 0's quarterly parser actually produces; a watchlist that stores lane,
state, checkpoints and an append-only transition history; Pass 2 emitting
monitorables in structured form validated at recording time; the core-lane
kill-switches declared in YAML; and `watchlist advance`, which re-scores,
evaluates triggers, and proposes transitions with evidence — auto-applying
only those that move no money.

### Problem Frame

The pipeline ends at a verdict. Everything after it — when a qualified company
becomes buyable, whether the thesis is holding, what would end it — is manual
and unrecorded. The inputs all exist: gates, flags, valuation percentiles,
growth-quality grades, quarterly financials (Phase 0 U1), and score history
(Phase 0 U3). What does not exist is anything that reads them on a schedule
and says what changed.

Two specific gaps make the current state worse than "not built yet". Pass 2
emits `key_monitorables` as free-text strings that **no Python code reads** —
they are rendered as bullets in both report templates and never checked again,
so the thesis is written down and then abandoned. And the LLM response passes
through `_parse_json_response` with **no schema validation of any kind**, so
nothing today would notice if `key_monitorables` came back malformed, absent,
or as a string instead of a list.

### Requirements

- R1. Transitions are declared in `lifecycle/triggers.yaml`, auto-discovered
  and validated at startup, and evaluated by deterministic code. A trigger
  whose inputs are missing evaluates **indeterminate — never a silent pass**,
  matching the eligibility-gate contract.
- R2. Each watchlist entry carries `lane`, `state`, `checkpoints`,
  `kill_switch_status`, `last_score_snapshot`, and an append-only
  `state_history` recording every transition with the evidence that caused it.
- R3. ~~Existing watchlist entries migrate losslessly~~ **struck (rev a)** —
  the store starts empty and is written only in the new schema. Its
  replacement obligation: a watchlist entry is created only by `watchlist
  add` or by an `advance` transition, and no entry is ever placed in a state
  it did not earn.
- R4. Pass 2 emits each monitorable in two forms: the existing prose for the
  reader, plus a structured `{metric_id, comparator, threshold, due_date}`
  object. A structured checkpoint whose `metric_id` is outside the checkpoint
  vocabulary is demoted to prose-only with a log line — never silently
  accepted, never silently dropped.
- R5. The checkpoint vocabulary contains only series the pipeline can actually
  evaluate quarterly, derived from the columns Phase 0 produces.
- R6. The core-lane kill-switches of v05 §6.1 that are expressible against
  existing metrics are declared in YAML; any that are not are recorded as
  deferred with the reason, rather than approximated.
- R7. `watchlist advance` re-scores each entry, evaluates triggers, and reports
  proposed transitions with evidence. Transitions into `probe`, `scale`,
  `exit_review` or `exited` are **proposed only** and require explicit owner
  confirmation; `screen → qualify → watch` and `→ dropped` may apply
  automatically, since no capital is involved (v05 §14.4).
- R8. Scores, eligibility verdicts, growth decomposition and the resolved
  action are unchanged by this phase.

### Scope Boundaries

- **No new metrics.** Every trigger references an existing metric id, an
  existing flag, or an existing `raw_series`. A rule that cannot be expressed
  is deferred with its reason, not approximated.
- **No fast lane.** Lane parameters exist as a `lane` field with the core-lane
  set only; re-rating gates are Phase 3.
- **No position sizing, no reinvestment queue, no portfolio math.** Phase 3.
- **No simulator.** Phase 4.
- **No automated execution, ever** (v05 §13).
- **No trigger-threshold calibration.** Starting values are declared and
  labelled as such; tuning waits for the Phase 4 simulator (v05 §12 Phase 5).

---

## Planning Contract

### Key Technical Decisions

- **KTD1 — The trigger evaluator mirrors `EligibilityEvaluator`, and reuses
  its comparators.** Same condition shape (`metric`/`comparator`/`threshold`),
  same `mode: all|any`, same `COMPARATORS` table (`lt`, `lte`, `gt`, `gte` —
  imported, not redefined), same three-valued result (`True`/`False`/`None`),
  same per-condition `detail` strings feeding a human-readable reason. One
  comparator vocabulary and one indeterminate semantic across the system: a
  reader who understands a gate reason understands a transition reason.
- **KTD2 — Consecutive-year rules read `raw_series`, not a new metric.** v05's
  sketch spells the capital-efficiency kill-switch as
  `{metric: roce_latest, ..., persist_years: 2}`, but **`roce_latest` does not
  exist** — the registry has `roce_5yr_avg`, a five-year mean, which cannot
  express "below 15% for two consecutive years". It does carry
  `raw_series` of the yearly RoCE values. A condition with `persist_years: N`
  therefore reads the metric's last N `raw_series` entries and requires the
  comparator to hold for all of them; a series absent or shorter than N is
  **indeterminate**, never a pass. This keeps the kill-switch's real semantics
  without inventing a metric.
- **KTD3 — Flag conditions carry sources, mirroring `veto_sources`.** A
  `flag_absent` condition names the metric expected to emit that flag; if the
  source metric is unavailable, the flag's absence proves nothing and the
  condition is indeterminate. This is the existing price-gate contract
  (`reverse_dcf_overpriced` / `reverse_dcf_growth`) generalised.
- **KTD4 — The checkpoint vocabulary is bound to real columns.** Phase 0's
  `quarterly.csv` carries `revenue, expenses, operating_profit, opm_pct,
  other_income, interest, depreciation, pbt, tax_pct, pat, eps`, and
  `shareholding.csv` carries `promoter_pct, fii_pct, dii_pct, govt_pct,
  public_pct`. The vocabulary exposes those plus derived year-over-year
  variants, and nothing else. Registry metrics are almost all annual-grain and
  therefore *not* checkpoint-evaluable at quarterly cadence — admitting them
  would produce checkpoints that can never come due.
- **KTD5 — The store starts empty; there is no migration path (rev a).**
  Old outputs were discarded by owner decision, so `watchlist.py` reads and
  writes exactly one schema and needs no version detection, no defaulting of
  absent fields, and no idempotent-migration tests. Entries are created by
  `watchlist add` at state `screen`, and `advance` promotes them through
  `qualify` and `watch` on real evaluation — so the "never grant an unearned
  state" property comes free from the entry point rather than from migration
  care. A file containing an unrecognised entry shape is a **loud error**,
  not something to repair silently: with one schema in existence, an odd
  entry means something is wrong, and guessing at it is how a company ends up
  in a state nobody assigned it.
- **KTD6 — Proposals are data, not prose.** `advance` returns a structured
  proposal list (`ticker, from, to, trigger_id, evidence, auto_applied`);
  the CLI renders it and the future GUI reads the same objects. Evidence is
  the evaluator's per-condition detail, so a proposal can always answer "which
  number caused this".
- **KTD7 — Lifecycle state lives beside the watchlist, not inside
  `AnalysisResult`.** `analyze()` stays a pure function of fetched data; the
  lifecycle is a separate store that *consumes* an `AnalysisResult`. This
  keeps R8 structurally true (nothing in the scoring path can read lifecycle
  state) and keeps `analyze()` usable for one-off research on a ticker that is
  not on the watchlist.

### Assumptions

- A1. The validation tickers must be refetched before the replay: **only 1 of
  23** `raw_data/` directories currently holds a `quarterly.csv`, because the
  rest were fetched before Phase 0 U1. Checkpoint evaluation is inert without
  it. This is a run-cost, not a code dependency.
- A2. `growth_quality_grade` emits flags `growth_quality_{grade}` with grades
  `high_quality | moderate | low_quality | risky`, so `growth_quality_risky`
  as used in the v05 sketch is real.
- A3. Pass 2 already receives the eligibility verdict as context; adding a
  structured monitorables field is a prompt and validation change, not a
  pipeline restructure.

### High-Level Technical Design

```
lifecycle/
  triggers.yaml            declared transitions + core-lane kill-switches
  checkpoint_vocabulary.yaml   quarterly-evaluable series only
  evaluator.py             TriggerEvaluator — mirrors EligibilityEvaluator
  checkpoints.py           vocabulary validation + quarterly series lookup
  store.py                 lifecycle state on the watchlist entry

watchlist.py    entry gains lane/state/checkpoints/state_history
service.py      unchanged scoring path; advance() consumes AnalysisResult
cli.py          `watchlist advance [--apply] [--dry-run]`
llm_layer/      pass2 prompt emits structured monitorables; recorded via
                checkpoints.validate() — out-of-vocabulary demotes to prose
```

---

## Implementation Units

### U1. Trigger registry and evaluator

- `lifecycle/triggers.yaml` + `lifecycle/evaluator.py`. Trigger spec:
  `label`, `rationale`, `from: [state,...]`, `to: state`, `mode: all|any`,
  `conditions: [...]`. Conditions are `{metric, comparator, threshold}`,
  optionally `persist_years: N` (KTD2), or `{flag_present|flag_absent,
  sources: [metric_id,...]}` (KTD3).
- Import `COMPARATORS` from `eligibility.py`; do not redefine. Unknown
  comparator → indeterminate with a warning, exactly as gates do.
- Result shape mirrors the gate result: per-trigger
  `{label, rationale, fired: True|False|None, reason, conditions: [...]}`
  with per-condition `{..., passed, detail}`.
- Startup validation: every `from`/`to` is a known state; every `metric` is a
  known metric id; every comparator is known. A typo must fail loudly at
  startup, not silently never fire.
- *Tests:* all/any combination; indeterminate on missing, errored, and
  non-numeric metrics; `persist_years` satisfied, violated, and
  series-too-short; flag present/absent with source available and unavailable;
  unknown comparator; unknown metric id rejected at validation.

### U2. Checkpoint vocabulary and quarterly series access

- `lifecycle/checkpoint_vocabulary.yaml` mapping each checkpoint metric id to
  its source frame and column, restricted to KTD4's real columns plus derived
  `*_yoy_pct` variants.
- `lifecycle/checkpoints.py`: `validate(checkpoint)` against the vocabulary,
  and `evaluate(checkpoint, data)` reading the quarterly/shareholding frame.
  A checkpoint whose series is absent (a ticker fetched before Phase 0)
  evaluates **indeterminate**, never missed — a company must not be dropped
  for a data gap.
- *Tests:* every vocabulary entry resolves against a synthetic frame with the
  real column names; out-of-vocabulary id rejected; absent frame →
  indeterminate; a met and a missed checkpoint; `*_yoy_pct` computed against
  the same quarter a year earlier, not the previous quarter.

### U3. Watchlist becomes the lifecycle store

*(Simplified by rev a — no migration path.)*

- One schema. An entry carries `ticker`, `added`, `notes`, `lane` (`core`),
  `state`, `checkpoints`, `kill_switch_status`, `last_score_snapshot`, and
  `state_history` (append-only: `{at, from, to, trigger_id, evidence,
  applied_by}`). `watchlist add` creates it at `screen`.
- `last_run`/`last_composite` are absorbed into `last_score_snapshot`, which
  also carries the `config_hash` — so the regime that produced a stored
  composite is visible without cross-referencing `score_history.jsonl`.
- An entry missing required keys raises with the ticker named (KTD5); the
  loader never repairs, defaults, or drops it.
- *Tests:* `add` creates a well-formed entry at `screen`; `state_history` is
  append-only across several transitions and never rewrites an earlier row; a
  malformed entry raises naming the ticker; an empty store loads cleanly;
  `show` renders lane and state.

### U4. Structured monitorables from Pass 2

- Prompt: `key_monitorables` keeps its prose array and gains
  `structured_monitorables: [{metric_id, comparator, threshold, due_date}]`,
  with the checkpoint vocabulary injected into the prompt as the only valid
  `metric_id` values.
- Recording: each structured monitorable is validated against the vocabulary
  (U2); failures are demoted to prose-only with a log line. Because
  `_parse_json_response` performs **no schema validation**, the recorder must
  itself tolerate a missing field, a wrong type, or the whole key being
  absent — an old cached response must not raise.
- *Tests:* a fixture with one valid and one hallucinated `metric_id` yields
  exactly one evaluable checkpoint and one logged demotion; absent key, wrong
  type, and malformed entries each degrade to prose-only without raising;
  prose monitorables still render in both report templates.

### U5. Core-lane kill-switches

- Declare in `triggers.yaml`, using existing metrics only:
  capital-efficiency break (`roce_5yr_avg` + `persist_years`),
  growth-quality degradation (`growth_quality_risky` flag),
  incremental-return break (`roiic` + `persist_years`),
  valuation saturation (`pe_vs_historical` — **note the real id, not the
  `pe_percentile_10y` of the v05 sketch** — combined with
  `reverse_dcf_overpriced`), and governance (`promoter_pledge`).
- Checkpoints-failed is expressed against U2 checkpoint outcomes.
- Any v05 §6.1 switch not expressible against existing metrics is recorded in
  this plan as deferred with its reason (R6), not approximated.
- *Tests:* each switch fires on a synthetic breach and stays silent on a
  healthy fixture; each is indeterminate when its metric is unavailable.

### U6. `watchlist advance`

- `advance(service, apply: bool)` per ticker: re-score via
  `service.analyze(use_llm=False)`, evaluate triggers against the fresh
  metrics and the entry's state, and produce proposals (KTD6).
  Money-moving transitions are proposed only; `screen → qualify → watch` and
  `→ dropped` auto-apply (R7).
- CLI `watchlist advance [--apply] [--quarterly] [-v]` renders proposals with
  their evidence; without `--apply`, nothing is written.
- Score-history rows continue to be written by the existing Stage 4.6 hook —
  `advance` gets trajectory persistence for free.
- *Tests:* a fixture entering the buy zone proposes `watch → probe` and does
  **not** auto-apply; a fired kill-switch proposes `→ exit_review`; a
  qualifying screen entry auto-applies to `watch`; `--apply` writes
  `state_history` with evidence while the default run leaves the file
  untouched; an indeterminate trigger proposes nothing and is reported as
  unknown.

---

## Verification Contract

- Full suite green via `venv/bin/python -m pytest tests/`.
- **R8 regression, structural:** nothing under `compute_engine/` or
  `action_policy.py` imports from `lifecycle/`, and `AnalysisResult` gains no
  lifecycle field — so scores, verdicts and actions cannot move. Confirm by
  re-running a cached ticker and diffing `scores.json` and `eligibility.json`.
- **Replay validation (v05 §12 Phase 1):** refetch CDSL, RAIN and VBL (A1),
  `watchlist add` each, run `watchlist advance`, and confirm every proposed
  transition cites the trigger evidence that caused it, with at least one
  buy-zone and one kill-switch case exercised on real data. The store is now
  empty (rev a), so this doubles as the fresh-schema end-to-end test.
- Report the fired/indeterminate split across the replay tickers, the way U4
  of Phase 0 reported found/fallback — a phase that silently produces all
  indeterminates has not been validated.

## Completion (2026-08-06)

All six units merged: U1 `1405b86`, U2 `f7105f6`, U3 `bb2c6ec`, U4 `0893cd1`,
U5 `93f708f`, U6 with this note. Suite 501 → 522 tests, green.

### Replay validation

Refetched CDSL, RAIN and VBL, added them plus ZYDUSLIFE, and ran
`watchlist advance`. Every proposal cited the trigger evidence that caused it.

**All four dropped at `screen` as `not_eligible`** — honest, and a finding in
itself: nothing in the fetched corpus currently passes the 100x gates, so the
post-qualification paths could not be reached by qualification alone. They were
therefore exercised by seeding real companies into `watch` and `scale` in a
scratch store, so the triggers evaluated **real fetched metrics** even though
the states were assigned.

| Case | Result on real data |
|---|---|
| Kill-switch fires | RAIN at `scale` → `exit_review`, evidence `roce_5yr_avg last 2 periods [4.00, 8.00] all lt 15`; `incremental_return_break` fired too, and was reported as superseded |
| Buy zone refuses | CDSL at `watch`, `fired=False` — P/E at the 100th percentile of its own band, trailing PEG 3.25 |
| Buy zone abstains | VBL at `watch`, `fired=None` — P/E 10th percentile and PEG 1.74 both attractive, but `reverse_dcf_growth` was unavailable so `reverse_dcf_overpriced`'s absence could not be confirmed |
| Money-moving guard | Every `exit_review` proposal came back `needs_confirmation=True, applied=False`; only `→ dropped` auto-applied |

VBL is the clearest vindication of KTD3. Two attractive ratios would have
proposed an entry; the system refused because it could not confirm the market
was not already pricing in growth the company has never delivered. An
"indeterminate, never a silent pass" rule stopped a real buy proposal on real
data — which is the whole reason the rule exists.

### Corrections this phase forced

1. **`roiic` cannot take `persist_years`.** Its `raw_series` is *capital
   employed* (INR Cr) beside an incremental-return *value* (%);
   `pe_vs_historical` likewise carries P/E multiples beside a 0–100
   percentile. Either rule would validate, run, compare incompatible units,
   and silently never fire. `persist_years` is now restricted to an audited
   `SERIES_SAFE_METRICS` allowlist and naming anything else is a startup
   error. The incremental-return switch uses ROIIC's plain value instead,
   since ROIIC is already an incremental measure across a multi-year window.
2. **Zero misses out of zero due checkpoints is not a clean bill of health.**
   Caught by a test: an unmonitored position read `clear`, identical to a
   verified one. The checkpoint condition is now indeterminate unless at least
   one checkpoint has actually come due.
3. **Verdict evidence read as a double negative** ("verdict is not_eligible
   (wanted not_eligible)"); the detail now only names the expectation when it
   differs from the outcome.

### Deferred (unchanged from the plan)

Auditor-resignation governance remains out of reach — no metric or flag
carries auditor changes, so `governance_event` covers the pledge limb only.
Trigger thresholds remain starting points; calibration waits on the Phase 4
simulator.

## Definition of Done

All six units merged with tests; replay validation performed and its results
recorded in this plan; `CLAUDE.md` updated with the lifecycle contract
(triggers registry, checkpoint vocabulary, state machine, propose-vs-apply
rule); v05 roadmap Phase 1 checked off with a pointer here.

## Deferred, with reasons

- **Auditor resignation / adverse governance read** (v05 §6.1 governance
  switch): no metric or flag carries auditor changes; `promoter_pledge` covers
  only the pledge limb. Needs a data source Phase 1 does not add.
- **Trigger threshold calibration:** starting values only, per scope
  boundaries; evidence comes from the Phase 4 simulator.
- **`indeterminate` escalation policy:** a trigger that is indeterminate for
  several consecutive runs arguably deserves attention rather than silence.
  Recorded as a Phase 2 question once trajectory data exists to say how often
  it happens.
