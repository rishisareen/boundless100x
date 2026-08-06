# Boundless100x — Phased Growth & Dual-Lane Roadmap
## Design Addendum to System Design v04 (Roadmap Proposal)

> **Status:** Reviewed and reconciled (2026-08-06). No SQGLP scoring, element
> weights, gates, or thresholds are changed by this document. Everything here
> is an **additive layer** on top of the v04 + Aug-2026 refinements
> architecture.
>
> **Supersedes:** nothing. **Depends on:** v04 design, and the refinements in
> `docs/plans/2026-08-03-001-feat-sqglp-refinements-plan.md` (eligibility gates,
> ROIIC/reinvestment, backtest, action policy).
>
> **Rev 2026-08-06 (design review reconciliation):** added **Phase 0 — Data
> Enablers** to the roadmap (quarterly results parser from the already-cached
> Screener page, score-run persistence, multi-year AR retention with
> section-targeted extraction) — three previously unstated data dependencies
> of Phases 1–2. Resolved open decision points 4–6; reframed 1–3 as owner
> config validated by simulator sensitivity sweeps. Noted the follow-up
> SQGLP-scoring calibration workstream spawned from Phase 5.
>
> **Rev 2026-08-06b (review amendments):** score-history rows carry a
> config/registry version stamp and a synthetic marker (§12 Phase 0, §7.1);
> partial backfill of score history via U9 truncation machinery noted as an
> option (§7.1); Screener quarterly-depth limit recorded as a simulator
> limitation (§10, §12 Phase 0); Pass 2 monitorables become structured
> checkpoints evaluable by code (§4.3, §12 Phase 1); AR section extraction
> gains a first-N-pages fallback and per-section cost caps (§12 Phase 0);
> sensitivity sweeps gain a statistical-humility clause (§12 Phase 5).
>
> **Rev 2026-08-06c (residual-gap fixes):** structured checkpoints constrained
> to a **checkpoint-evaluable vocabulary** of quarterly-derivable series,
> validated at recording time with prose-only demotion on failure (§4.3,
> §12 Phase 1); AR section extraction carries per-section **provenance**
> (`found` / `fallback`), and section-dependent sub-metrics evaluate
> indeterminate on `fallback` (§7.2, §12 Phase 0); trajectory **diff cadence
> is derived from actual row dates** — annual-grain synthetic backfill is
> never read as quarterly momentum (§7.1).

---

## Table of Contents

1. [Purpose & Objective](#1-purpose--objective)
2. [The Gap This Addendum Closes](#2-the-gap-this-addendum-closes)
3. [Design Principles](#3-design-principles)
4. [The Phased Investment Lifecycle](#4-the-phased-investment-lifecycle)
5. [Trigger Rule Registry (`triggers.yaml`)](#5-trigger-rule-registry-triggersyaml)
6. [Exit Policy — The Symmetric Gate System](#6-exit-policy--the-symmetric-gate-system)
7. [Engine Enhancements (Lane-Agnostic)](#7-engine-enhancements-lane-agnostic)
8. [Dual-Lane Portfolio Structure](#8-dual-lane-portfolio-structure)
9. [Re-Rating Lane Specification (Fast Lane)](#9-re-rating-lane-specification-fast-lane)
10. [Strategy Simulator — Backtest Evolution](#10-strategy-simulator--backtest-evolution)
11. [Deployment-Pace Modulator](#11-deployment-pace-modulator)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Non-Goals](#13-non-goals)
14. [Decision Points (resolved)](#14-decision-points-resolved-2026-08-06-except-where-noted)

---

## 1. Purpose & Objective

The v04 pipeline answers: *"Is this company a potential long-term compounder,
and is it 100x-eligible?"* The owner's stated objective is: **make money at an
accelerated rate — either by holding long-term compounders, or by investing
short-term and reinvesting proceeds.**

Between the verdict the pipeline produces today and that objective sits an
entire missing layer: **when** to enter, **how much** to deploy, **in how many
steps**, **what confirms** the thesis after entry, **what kills** the position
quantitatively, and **where proceeds go** next. This addendum designs that
layer as a **phased lifecycle** that serves both the long-term and short-term
approaches from one state machine.

---

## 2. The Gap This Addendum Closes

Today's pipeline terminates in a verdict: composite score, eligibility badge,
suggested action (capped by the action-policy guard). Everything after the
verdict is currently manual and unrecorded:

| Missing capability | Consequence without it |
|---|---|
| Entry timing relative to valuation state | Great companies bought at 95th-percentile P/E produce mediocre returns |
| Position sizing & staged deployment | Full-size entries on unconfirmed theses; no defined add-on-confirmation behaviour |
| Post-entry confirmation loop | LLM `key_monitorables` are written, never systematically re-checked |
| Quantitative exit rules | Exits driven by narrative drift, not by the same rigour as entries |
| Reinvestment routing | Fast-lane proceeds sit idle or get redeployed ad hoc — the "accelerated" part of the objective |
| Strategy-level validation | The backtest proves scores correlate with returns; it does not prove the *rules* make money |

Every input needed to close these gaps already exists in the pipeline
(monitorables, flags, gates, reverse-DCF implied growth, P/E percentile,
growth-quality grade, quarterly shareholding, price history). This addendum
assembles them into a decision system. It introduces **no new data sources**
in its first iteration.

---

## 3. Design Principles

1. **SQGLP is not distorted for the fast lane.** The SQGLP framework measures
   *duration* (compounding durability). The short-term approach monetizes
   *change* (re-rating, earnings surprise, institutional accumulation). These
   are different instruments. The fast lane gets its own lane definition,
   its own gates, and its own exit rules — sharing the engine, never
   re-weighting SQGLP. (Owner-confirmed 2026-08-06.)
2. **Additive layers, not surgery.** Like the eligibility-gates refinement
   (KTD1), the lifecycle sits *after* Stage 3.6 and consumes the metrics,
   scores, gates, and LLM output. Existing reports remain comparable.
3. **Rules in YAML, judgment in LLM, decisions in code.** Mirrors the
   established pattern: metric registry for computation, action policy in
   code (not prompts) for decisions. The lifecycle's triggers are
   declarative YAML; state transitions are deterministic code; the LLM
   informs but never transitions a state.
4. **The same lifecycle serves both lanes.** Phases are identical; only
   trigger thresholds, sizing parameters, and holding-period expectations
   differ per lane. One state machine, two parameter sets.
5. **Exits are first-class.** The exit system mirrors the entry gate system
   in structure and rigour. For an acceleration objective, exits matter more
   than entries.
6. **Friction honesty.** Any fast-lane claim is stated net of Indian tax
   asymmetry (STCG vs LTCG) and small-cap impact cost, with the break-even
   computed by the simulator rather than assumed.

---

## 4. The Phased Investment Lifecycle

### 4.1 State Machine

```
        ┌─────────────────────────────────────────────────────────┐
        │                   LIFECYCLE STATES                       │
        │                                                          │
        │  SCREEN ──▶ QUALIFY ──▶ WATCH ──▶ PROBE ──▶ SCALE ──▶  │
        │     │          │          │        │        │   HARVEST  │
        │     ▼          ▼          ▼        ▼        ▼      │     │
        │  (dropped)  (dropped)  EXIT ◀──── EXIT ◀── EXIT ◀──┘     │
        │                              │                         │
        │                              ▼                         │
        │                     REINVESTMENT QUEUE                   │
        └─────────────────────────────────────────────────────────┘
```

### 4.2 Phase Definitions

| Phase | Entry condition (computable today) | Behaviour | Exit / transition |
|---|---|---|---|
| **0. Screen** | Preset filters (`compounders`, `hidden_gems_100x`) over the universe | Periodic universe sweep | Non-qualifiers dropped; qualifiers → Qualify |
| **1. Qualify** | Eligibility gates pass (lane-specific gate set) + composite ≥ lane threshold | Full pipeline run; thesis + monitorables recorded | Fails → dropped; passes → Watch |
| **2. Watch** | Qualified, but Price element unfavourable (e.g. P/E above own-history band, `reverse_dcf_overpriced`, or `pe_above_historical_75th` flag) | Quarterly re-score; valuation tracked | Valuation enters buy zone → Probe; fundamentals deteriorate → dropped |
| **3. Probe** | Valuation favourable + no active kill-switches | **Tranche 1** (e.g. 25–33% of intended position); LLM Pass 2 `key_monitorables` become written checkpoints with review dates | Thesis confirmed at checkpoint → Scale; kill-switch → Exit |
| **4. Scale** | Checkpoint reviews confirm monitorables; growth quality remains Volume/OpLev-driven | **Tranches 2–3** on confirmation; adds permitted on price weakness *only if* fundamentals unchanged (no new red flags since last review) | Kill-switch or valuation ceiling → Harvest/Exit |
| **5. Harvest / Exit** | Any quantitative kill-switch fires (§6), or target-multiple / time stop reached (fast lane) | Rule-based exit; proceeds logged | Proceeds → Reinvestment Queue |

### 4.3 What Replaces the Current Watchlist

The watchlist becomes the persistence layer for this state machine. Each entry
gains: `lane` (core | rerating), `state`, `entry_date`, `tranches_taken`,
`checkpoints` (structured `{metric_id, comparator, threshold, due_date}`
objects — see below), `last_score_snapshot`,
`kill_switch_status`, and `state_history` (append-only transition log with the
trigger evidence that caused each transition).

**Checkpoints are structured, not prose.** Pass 2 today emits
`key_monitorables` as free text; free text cannot be evaluated by
deterministic code, and Principle 3 puts decisions in code. Pass 2 therefore
emits each monitorable in two forms — the existing prose for the human
reader, plus a structured `{metric_id, comparator, threshold, due_date}`
object the lifecycle can check against the quarterly series. This is what
makes the `monitorable_missed` trigger (§5) evaluable at all; a monitorable
the LLM cannot express in structured form is recorded as prose-only and
reviewed manually, never auto-evaluated.

**Checkpoint-evaluable vocabulary (rev 2026-08-06c).** A structured
checkpoint's `metric_id` may only reference the **checkpoint vocabulary**: an
explicit whitelist of quarterly-derivable series (e.g. quarterly revenue YoY,
quarterly OPM, promoter pledge %, FII+DII holding %), maintained as a small
YAML file alongside `triggers.yaml`. Nearly all registry metrics are
annual-grain and therefore *not* checkpoint-evaluable at quarterly cadence —
without this constraint, Pass 2 would emit ids that are hallucinated or
annual-only, and `monitorable_missed` would drown in indeterminates instead
of firing. Enforcement is two-sided: the Pass 2 prompt lists the vocabulary
as the only valid `metric_id` values, and the recording step **validates
every structured checkpoint against the vocabulary** — an id outside it
demotes that checkpoint to prose-only with a log line, never a silent accept.
Flat add/remove CRUD remains,
but `watchlist update` becomes `watchlist advance`: re-score, evaluate triggers,
propose transitions with evidence. Transitions that move money (Probe → Scale,
any → Exit) are **proposed with reasons and confirmed by the owner**, never
auto-executed — the system advises, the owner decides.

### 4.4 Lane Parameters (same machine, two parameter sets)

| Parameter | Core Lane (long-term) | Re-Rating Lane (fast) |
|---|---|---|
| Gate set | 100x-eligibility gates (existing) | Lane gates (§9.2) |
| Intended holding | 5–10 yr+ | 6–18 months |
| Probe tranche | ~33% of intended size | ~50% (shorter confirmation cycle) |
| Checkpoint cadence | Quarterly results | Quarterly + monthly price/valuation scan |
| Exit basis | Fundamentals kill-switches (§6.1) | Target multiple, time stop, or fundamentals break (§6.2) |
| Sizing cap | e.g. 10–15% of sleeve per name | e.g. 5% of sleeve per name |

---

## 5. Trigger Rule Registry (`triggers.yaml`)

Consistent with the metric-registry centerpiece: a trigger is **one YAML entry
+ one evaluation primitive**, auto-discovered, validated at startup. The
lifecycle evaluator reads it the way the scorer reads the metric registry.

Illustrative shape (design sketch, not final schema):

```yaml
# lifecycle/triggers.yaml — declarative state-transition rules
triggers:
  valuation_buy_zone:
    phase: watch_to_probe
    all:
      - { metric: pe_percentile_10y, comparator: "<=", threshold: 60 }
      - { flag_absent: reverse_dcf_overpriced }
      - { metric: trailing_peg, comparator: "<=", threshold: 2.0 }

  growth_quality_break:
    phase: any_to_exit_review
    any:
      - { flag_present: growth_quality_risky }
      - { metric: roce_latest, comparator: "<", threshold: 15, persist_years: 2 }

  monitorable_missed:
    phase: probe_to_exit_review
    source: llm_checkpoints          # checkpoint overdue OR failed at review
    comparator: "=="
    threshold: true
```

Kill-switch and checkpoint semantics follow the eligibility-gate precedent:
a trigger whose inputs are missing evaluates **indeterminate, never a silent
pass**, and indeterminate transitions surface for owner review rather than
failing open.

---

## 6. Exit Policy — The Symmetric Gate System

### 6.1 Core Lane Kill-Switches (fundamentals-driven)

| Kill-switch | Rule (starting point; tuned by simulator) | Rationale |
|---|---|---|
| Capital-efficiency break | RoCE < 15% for 2 consecutive years | Compounding engine stalled |
| Growth-quality degradation | Growth decomposition flips to Financial-Lever dominant | Low-quality growth; the Risky grade in the existing 4-lever framework |
| Incremental-return break | ROIIC < cost of capital for 2 consecutive years | Growth now destroys value even if EPS rises |
| Valuation saturation | P/E > 95th percentile of own 10-yr history **and** reverse-DCF implied growth > 1.5–2× demonstrated achievable growth | Price has outrun any plausible fundamentals |
| Governance event | Promoter pledge crosses threshold, auditor resignation, adverse LLM governance read | Thesis-level invalidation |
| Checkpoints failed | Majority of Pass 2 monitorables missed at two consecutive reviews | The thesis as written is not happening |

Kill-switches move a position to **Exit Review**, not auto-sell: the report
states which switch fired, with evidence, and the owner confirms.

### 6.2 Fast-Lane Exit Rules (change-driven)

| Rule | Starting point |
|---|---|
| Target reached | Entry thesis multiple attained (e.g. re-rating to quality-justified multiple per §7.3) |
| Time stop | 18 months without the anticipated re-rating — the catalyst did not materialize |
| Fundamentals break | Any core-lane kill-switch (§6.1) — never "trade through" a fundamentals break |
| Catalyst spent | The identified catalyst (results, commissioning, order win) occurred and was fully priced |

---

## 7. Engine Enhancements (Lane-Agnostic)

These strengthen the shared engine; both lanes consume them.

### 7.1 Score Trajectory (priority: high, cost: low)

A single SQGLP score is a photograph; returns come from **deltas** — improving
fundamentals precede re-rating. The pipeline already writes `scores.json` per
run. Enhancement: persist element scores with run dates and compute quarterly
diffs per watchlist name — **score momentum** as a first-class signal (feeds
both Scale decisions in the core lane and candidate surfacing in the fast
lane). This exists today only as a deferred GUI-roadmap item ("historical
analysis"); pull it into the engine.

**Sequencing note (rev 2026-08-06):** diffs need 2–3 stored runs per name
before they say anything, and history cannot be backfilled. The *persistence*
half therefore ships in **Phase 0** (append-only
`{ticker, date, elements, composite, verdict, config_hash, synthetic}` per
run) so a series exists by the time this section's diff computation lands in
Phase 2. The `config_hash` stamps the registry/threshold version on every row,
so momentum diffs never silently mix scoring regimes when Phase 5 calibration
changes weights or thresholds.

**Partial backfill option (rev 2026-08-06b):** "cannot be backfilled" applies
to organic runs — but the U9 backtest machinery can re-score truncated cached
`raw_data/` to synthesize 1–2 historical score points per ticker, under the
same leakage exclusions the backtest already enforces. Such rows are written
with `synthetic: true` and are never mixed with organic runs in a momentum
read without that marker visible. This buys usable diffs quarters earlier
than organic accumulation alone; it is an option, not a requirement.

**Diff cadence is derived, not assumed (rev 2026-08-06c):** U9 truncation
operates on annual history, so synthetic points land roughly one per year —
a momentum read spanning synthetic rows is *annual* momentum, while organic
rows accumulate quarterly. The diff computation therefore derives its cadence
from the actual row dates and labels every momentum figure with the interval
it spans; a diff over backfilled history must never present itself as
quarterly-fresh signal. (A regenerability bonus of the synthetic path: if
Phase 5 changes the config, frozen organic rows keep their old `config_hash`,
but synthetic points can be cheaply re-scored under the new regime for
cross-regime comparison.)

### 7.2 Forward-Growth Module (priority: high — addresses the biggest conceptual weakness)

G carries 25% of the composite but is measured entirely backward (CAGR,
streaks). "Grow more than 20% consistently" is a *forward* claim; the module
adds forward evidence without touching the backward scores:

| Sub-metric | Source (already fetched) | What it measures |
|---|---|---|
| Promises-kept ratio | Annual report guidance vs. subsequent delivery (LLM-assisted extraction) | Management forecast credibility — a proven predictor of guidance reliability |
| Capacity/capex pipeline | AR text: announced capacity, commissioning dates, capex plans | Physical runway for volume growth (Lever 1) |
| TAM/runway sizing | AR + sector context (LLM-assisted) | Whether 20%+ growth is arithmetically possible for N more years (feeds Longevity's CAP proxy) |
| Quarterly momentum | Quarterly results series (Phase 0 parser); latest-quarter shareholding | Is growth accelerating or decelerating *now* |

**Data prerequisite (rev 2026-08-06):** promises-kept and capex-pipeline need
*multi-year* annual reports and MD&A-depth text; the current fetch is one
report, first 30 pages, capped at 5,000 chars — essentially the chairman's
letter. Phase 0 raises retention to 2–3 years and switches to
section-targeted extraction (MD&A, guidance, capex, RPT schedule). Without
that, these sub-metrics evaluate indeterminate. Quarterly momentum likewise
depends on the Phase 0 quarterly parser, not on the single TTM column.

**Provenance rule (rev 2026-08-06c):** section-dependent sub-metrics
(promises-kept, capex pipeline, TAM) read the extraction provenance tags
(§12 Phase 0) and evaluate **indeterminate when their required section is
`fallback`** — first-N-pages text must never be mined for guidance it does
not contain. This extends the indeterminate-never-silent-pass principle from
triggers down to the extraction layer.

### 7.3 Re-Rating Headroom Metric (priority: high — elevate the deferred item)

Expected return ≈ earnings growth × multiple change. The model measures the
first term well and treats the second only as a risk (P/E percentile, reverse
DCF). Add the constructive form, deferred in the refinements plan: **entry
multiple vs. quality-justified multiple** — the multiple defensible given
RoCE, growth, and longevity vs. the multiple paid today. This is the actual
accelerator in both lanes and the fast lane's primary exit target.

### 7.4 Capital Allocation Extension

ROIIC and reinvestment rate (refinements U5) measure *that* capital is
redeployed well; extend to *how* management chooses among reinvestment, M&A,
buybacks, and dividends — the promises-kept ratio (§7.2) doubles as the
credibility input here. Small weight; LLM-assisted.

---

## 8. Dual-Lane Portfolio Structure

### 8.1 Core-Satellite Allocation

```
┌──────────────────────────────────────────────────────────────┐
│                     PORTFOLIO (one pool)                     │
│                                                              │
│  ┌────────────────────────────┐  ┌────────────────────────┐  │
│  │  CORE LANE  (~70%)         │  │  RE-RATING LANE (~30%) │  │
│  │  Phased lifecycle, LT      │  │  Same lifecycle, fast  │  │
│  │  parameters, 5–10yr holds  │  │  parameters, 6–18mo    │  │
│  └─────────────┬──────────────┘  └───────────┬────────────┘  │
│                │  exits feed                  │  exits feed    │
│                ▼                              ▼              │
│              REINVESTMENT QUEUE  ── routes to best-qualified │
│              candidate in EITHER lane by current triggers    │
└──────────────────────────────────────────────────────────────┘
```

- Sleeve proportions are owner-set config, not hardcoded.
- **Reinvestment policy is explicit:** proceeds enter a queue and are routed
  to the highest-priority candidate across *both* lanes as defined by current
  trigger state (e.g. a Watch name entering its buy zone beats a fresh screen
  candidate). Idle cash drag is reported.
- **Concentration guardrails:** per-name caps per lane (§4.4), per-sector cap
  (e.g. 25–30%), and a correlation note when two holdings share a macro
  driver (both are lender-financed, both export-cyclical, etc.).

### 8.2 Friction Honesty

The fast lane must beat the core lane **net**, or it fails its purpose:

- STCG 20% vs LTCG 12.5% (India, current regime — config values, not literals)
- Impact cost / slippage on small caps (the lane's natural habitat — the same
  low-liquidity that creates mispricing taxes exits)
- Break-even: the fast lane must gross roughly **6–10 points more per cycle**
  than the core lane's annualized return just to tie after tax and friction
  (exact figure computed by the simulator with owner-set cost assumptions —
  stated here as an estimate, not a verified number)

This is stated in every fast-lane report header, so "accelerated" is never
confused with "busier."

---

## 9. Re-Rating Lane Specification (Fast Lane)

### 9.1 What It Monetizes

Change, not duration: earnings surprises, multiple re-rating toward
quality-justified levels, institutional discovery, sector rotation, capacity
commissioning. Holding period 6–18 months.

### 9.2 Lane Gates (entry) — own gate set, engine-shared

| Gate | Starting threshold | Data (already fetched) |
|---|---|---|
| Quality floor | Composite ≥ lane threshold (never trade junk) | SQGLP composite |
| Valuation discount | P/E ≤ 40–50th percentile of own 10-yr history, or re-rating headroom ≥ threshold (§7.3) | price element |
| Growth intact | Latest TTM growth ≥ historical CAGR; growth quality not FinLev-driven | growth element |
| Institutional accumulation | FII+DII rising for 2+ consecutive quarters | shareholding.csv (already quarterly) |
| Catalyst identified | Named catalyst with expected window (results, commissioning, order inflow) — recorded, not scored | AR text / owner input |
| Liquidity floor | Average daily turnover sufficient for position size (owner-set) | price_volume.csv |

### 9.3 What the Fast Lane Explicitly Does NOT Do

- No changes to SQGLP weights, elements, or thresholds
- No entry without the quality floor — the lane is a valuation/catalyst
  overlay on proven quality, not a momentum screen over junk
- No averaging down — adds only on thesis confirmation, as in the core lane
- No holding through a fundamentals kill-switch, regardless of price action

---

## 10. Strategy Simulator — Backtest Evolution

The existing backtest (refinements U9) validates that **scores correlate with
returns** — a diagnostic. The simulator validates that **the rules make
money** — the only honest test of "accelerated."

| | Current backtest | Strategy simulator (this proposal) |
|---|---|---|
| Question | Do high scores predict returns? | Do gates + lifecycle + sizing + exits produce portfolio CAGR above buy-and-hold of the same names? |
| Unit of analysis | Per-ticker score vs. forward return | Portfolio equity curve over time |
| Mechanics | Truncate history, score, correlate | Replay phases: entries in tranches on trigger dates, adds on confirmations, exits on kill-switches; apply tax/friction per lane |
| Outputs | Spearman correlations, exclusions | Portfolio CAGR, max drawdown, turnover, per-lane net-vs-gross, fast-lane break-even (§8.2), cash-drag from idle reinvestment queue |
| Universe caveat | Survivorship-selected (stated) | Same caveat, inherited and re-stated — plus phase-transition counts so thin samples are visible |

Implementation follows the backtest's own precedent (KTD3): reuse the
production engine on truncated inputs, report every exclusion, carry a
limitations block, and treat look-ahead leakage as the central correctness
risk. The limitations block also records the **quarterly-depth constraint**
(rev 2026-08-06b): Screener's quarterly table covers only recent quarters, so
quarterly-grain checkpoints cannot be replayed deep into truncated history —
checkpoint-driven transitions are validated on the annual grain until organic
quarterly history accumulates, and every affected transition is counted in
the exclusion report. The simulator is Phase 3 of the roadmap (§12) because it depends on the
lifecycle and triggers existing to be simulated.

---

## 11. Deployment-Pace Modulator

`earnings_yield_vs_gsec` already exists as a metric. Wire the macro block's
spread as a **pace modulator**, not a gate: when equity earnings-yield spread
over G-Sec is wide, Probe tranches deploy on schedule; when compressed,
Watch-phase entries require the stricter end of buy-zone thresholds and
tranche sizes step down. Cheap (config + one rule), keeps the owner from
deploying full pace at market-level valuation extremes. This modulates *pace*
only — it never overrides a per-name kill-switch or gate.

---

## 12. Implementation Roadmap

Sequenced so each phase is independently useful and testable, matching the
repo's phase conventions. No phase touches SQGLP scoring.

**Phase 0 — Data Enablers** — ✅ **complete 2026-08-06**, see
`docs/plans/2026-08-06-002-feat-phase0-data-enablers-plan.md` for the
implementation record, measured hit rates, and the corrections it forced
(quarterly section present on all cached pages; MD&A needs a 150-page search
window, not 30; registry hash extended to macro and the history waiver).
- **Quarterly results parser**: parse the quarterly results table from the
  *already-cached* Screener company page → `quarterly.csv` in the
  `raw_data/{TICKER}/` data contract. Not a new data source — a new parser on
  an existing fetch. Without it, the lifecycle's quarterly checkpoint cadence
  ("OPM below X for 2 consecutive quarters") has nothing to evaluate.
  Two caveats (rev 2026-08-06b): the Screener quarterly table is shallow
  (recent quarters only) — sufficient for consecutive-quarter checks,
  insufficient for deep historical replay, recorded as a simulator limitation
  (§10); and tickers cached before the Screener page-caching fix need a
  re-fetch for the table to exist in cache at all (the same caveat the
  residual review recorded for sector metadata).
- **Score-run persistence**: append-only history per ticker
  (`{ticker, date, elements, composite, verdict, config_hash, synthetic}`)
  written on every scored run. Ships first because history cannot be
  backfilled — every run before this lands is a lost data point for §7.1's
  momentum diffs. `config_hash` stamps the registry/threshold version so
  trajectory diffs never mix scoring regimes after Phase 5 calibration;
  `synthetic: true` marks U9-backfilled rows (§7.1) apart from organic runs.
- **Annual-report depth**: retain 2–3 years of ARs (config `max_reports`),
  section-targeted extraction (MD&A, guidance, capex, RPT schedule) replacing
  first-N-pages. Prerequisite for §7.2 promises-kept / capex pipeline / TAM.
  Two guards (rev 2026-08-06b): section detection is heuristic — Indian AR
  PDFs lack reliable anchors — so first-N-pages remains the fallback when a
  section is not found; and each section carries its own character cap, so
  2–3 reports × deeper extraction does not multiply Pass 1 token cost beyond
  the existing budget controls (`max_text_chars` becomes per-section).
  **Provenance guard (rev 2026-08-06c):** every extracted section is tagged
  `found` or `fallback` in the extraction output. Fallback text is a
  chairman's letter, not MD&A — downstream consumers must be able to tell
  the difference (§7.2), otherwise the fallback silently converts "section
  not found" into "wrong answer."
- *Validation:* quarterly.csv reproduces Screener's displayed quarters for the
  reference companies; two consecutive scored runs produce two history rows;
  extracted AR sections contain MD&A text beyond page 30.

**Phase 1 — Lifecycle Foundation** — ✅ **complete 2026-08-06**, see
`docs/plans/2026-08-06-003-feat-phase1-lifecycle-foundation-plan.md` for the
implementation record, replay results, and the corrections it forced (no
`roce_latest` metric exists; `persist_years` must be allowlisted because
`raw_series` has no unit contract; an unmonitored position must not read as
checkpoint-clear).
- Watchlist → state machine (states, lane field, state_history, checkpoints)
- `triggers.yaml` registry + evaluator (indeterminate-on-missing semantics)
- Exit kill-switch definitions (core lane) in YAML
- `watchlist advance` command: re-score, evaluate triggers, propose transitions with evidence
- **Structured checkpoints** (rev 2026-08-06b): Pass 2 emits
  `key_monitorables` in structured form (`{metric_id, comparator, threshold,
  due_date}`) alongside the existing prose (§4.3) — free-text monitorables
  cannot be evaluated by deterministic code (Principle 3), and this is what
  makes the `monitorable_missed` trigger evaluable. Prompt change only; no
  pipeline restructuring.
- **Checkpoint vocabulary + recording-time validation** (rev 2026-08-06c):
  ship the whitelist of quarterly-derivable series (§4.3) as
  `lifecycle/checkpoint_vocabulary.yaml`, inject it into the Pass 2 prompt,
  and validate every structured checkpoint against it at recording time —
  out-of-vocabulary ids demote to prose-only with a log line.
  *Validation:* a Pass 2 fixture emitting one valid and one hallucinated
  `metric_id` produces exactly one evaluable checkpoint and one logged
  prose-only demotion.
- Checkpoint evaluation reads the Phase 0 quarterly series, so quarterly-grain
  monitorables are checkable at the promised cadence from day one
- *Validation:* replay transitions for existing reports (CDSL, RAIN, VBL) and confirm each proposed transition cites the correct trigger evidence

**Phase 2 — Engine Enhancements** — ✅ **complete 2026-08-06**, see
`docs/plans/2026-08-06-004-feat-phase2-engine-enhancements-plan.md` for the
implementation record, the R7 non-regression proof, and the corrections it
forced (the pace input had to become a corpus median because the metric §11
names is per-company; the registry hash had to split in two or Phase 5 would
destroy the trajectory evidence it needs to calibrate; and the content gate
had to be rebuilt against how MD&A actually reads rather than the structure
the statute requires of it).
- Score trajectory quarterly diffs (§7.1) — persistence itself landed in Phase 0; this phase computes momentum over the accumulated history
- Re-rating headroom metric (§7.3)
- Forward-growth module: promises-kept, capex pipeline, TAM runway, quarterly momentum (§7.2) — LLM-assisted sub-metrics behind the existing cost controls; consumes Phase 0's multi-year AR sections and quarterly series
- Deployment-pace modulator (§11)
- **Two-hash split** (implementation decision): `registry_hash` narrows to what
  can move a composite, and a second `forward_signal_hash` covers zero-weight
  metric definitions and the extraction schema. Without it, Phase 5 could not
  calibrate a forward signal without resetting every ticker's momentum
  baseline — the evidence it needs to calibrate with.
- **Three-valued provenance** (implementation decision, extending §7.2's
  `found`/`fallback`): a located section must also *look like* the section it
  claims to be, or it is downgraded to `suspect` and excluded from extraction
  exactly as `fallback` is. Phase 0 could tolerate a wrong-section slice
  because its text was only Pass 1 background; an extractor mines whatever it
  is handed and yields well-formed, confident, wrong guidance.
- *Validation:* metrics appear in scores.json via the registry; trajectory diffs reproduce from stored runs; headroom metric lands in the price element without altering composite weights

**Phase 3 — Fast Lane + Portfolio Layer**
- Lane gates (§9.2) + lane parameter sets (§4.4)
- Core-satellite allocation config + reinvestment queue with explicit routing policy (§8.1)
- Friction model (STCG/LTCG, slippage) as config; net-vs-gross reporting
- *Validation:* a fast-lane candidate passes all lane gates end-to-end; exit routing into the reinvestment queue produces the documented next action

**Phase 4 — Strategy Simulator**
- Phase-replay mechanics over truncated history (§10), per-lane tax/friction
- Portfolio outputs: CAGR, drawdown, turnover, break-even, cash drag
- *Validation:* hand-computed two-name fixture reproduces simulator equity curve exactly; limitations block present; exclusions listed

**Phase 5 — Calibration Loop (ongoing)**
- Simulator output informs trigger thresholds and lane parameters (documented as config changes with before/after simulator evidence — never silent retuning)
- Quarterly: lifecycle review report — transitions taken vs. proposed, override rate, hit rate of checkpoints
- Simulator sensitivity sweeps over the owner-policy parameters (§14.1–.3): sleeve split, tranche sizing, kill-switch severity — so those configs are set from evidence, not priors
- **Statistical-humility clause (rev 2026-08-06b):** sweep outcomes over the small survivorship-selected universe are **directional only** — a sweep may suggest a parameter change, but acting on one requires a minimum transition count and follows the same documented before/after evidence rule as any other Phase 5 retune. Sweeps that would fit noise are reported as noise, not as settings.
- **Spawns a separate workstream: SQGLP scoring calibration.** Out of scope for this addendum (Non-Goals §13), but once the simulator can supply before/after portfolio evidence, the inherited MOSL-derived element weights and thresholds should face the same evidence standard as the lifecycle triggers. Tracked as its own follow-up under the refinements plan's scope boundary.

---

## 13. Non-Goals

- **No changes to SQGLP elements, weights, thresholds, or gate logic** in any phase (calibration of *lifecycle triggers* is in scope; calibration of *SQGLP scoring* remains follow-up work under the refinements plan's scope boundary)
- **No automated execution.** The system proposes; the owner disposes. No broker integration.
- **No new data sources in Phases 0–3.** The Phase 0 quarterly parser reads the Screener company page already fetched and cached — a new parser on an existing source, not a new source. Forward-growth extraction uses annual reports already downloadable from BSE (retention raised, same source). (Consensus-estimate feeds are a possible later addition, flagged separately.)
- **No intraday/technical trading.** The fast lane's shortest cadence is monthly; it is a re-rating instrument, not a trading system.
- **No GUI work** beyond what v04 already defers.

---

## 14. Decision Points (resolved 2026-08-06 except where noted)

### Owner-policy parameters — config now, evidence later (1–3)

These are personal risk-policy choices, not methodology questions. Decision:
**ship the placeholders as owner-editable config**, and let Phase 5's
simulator sensitivity sweeps set them from evidence rather than priors. None
of them block Phase 1.

1. **Sleeve split:** placeholder 70/30 core/fast as config; simulator sweeps
   60/40 vs 70/30 vs 80/20.
2. **Tranche sizing:** placeholder Probe at 33% (core) / 50% (fast) as
   config; simulator sweeps 25/33/50%.
3. **Kill-switch severity:** placeholder mapping as config — governance =
   full exit; valuation saturation = reduce; others = exit review — pending
   simulator evidence on which switches historically preceded losses vs.
   recoveries.

### Methodology decisions — resolved (4–6)

4. **Transition autonomy: RESOLVED — propose-and-owner-confirms** for any
   transition that moves money (Probe, Scale, Exit, reinvestment routing).
   Auto-transition is permitted only for Screen → Qualify → Watch, where no
   capital is involved. This matches the safety posture of the existing
   action-policy guard: the system advises, deterministic code caps, the
   owner decides.
5. **Re-rating headroom methodology: RESOLVED — own-fundamentals bands**
   (quality-justified multiple derived from the company's RoCE / growth /
   longevity), not sector-relative. Sector-relative would quietly
   reintroduce the peer comparison v04 deliberately removed, and would
   inherit sector-wide mispricing into the "justified" anchor.
6. **Simulator universe: RESOLVED — accept the survivorship-selected
   `raw_data/` universe for the first simulator run**, with the caveat block,
   exactly as the existing backtest precedent (KTD3) does. All simulator
   results are read as an **upper bound** until a point-in-time universe
   exists; building one is real work and does not gate Phase 4.
