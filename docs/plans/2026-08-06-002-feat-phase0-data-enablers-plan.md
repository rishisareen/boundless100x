---
title: Phase 0 Data Enablers - Plan
type: feat
date: 2026-08-06
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: Design/Financial Model v05 - Phased Growth Roadmap.md (§12 Phase 0)
execution: code
---

# Phase 0 Data Enablers - Plan

> **Rev 2026-08-06 (review amendments):** R7's byte-identical guarantee scoped
> to `--no-llm` artifacts, with KTD6 requiring the compatibility string to
> preserve today's extraction prefix (finding: concatenated-found-sections ≠
> first-30-pages text); U2's hash extended to the evaluator's *effective*
> gates, closing the `DEFAULT_GATES` code-fallback gap; U1's smoke adds a
> ticker lacking the `quarters` section to prove graceful absence on real
> data; the smoke DoD now names the expected `indeterminate` verdict for
> ASTRAL (no `price_volume.csv` → price gates indeterminate by design).

## Goal Capsule

- **Objective:** Land the three data enablers the v05 lifecycle depends on —
  a quarterly results parser over the already-cached Screener page, append-only
  score-run persistence with a config hash, and multi-year annual-report
  retention with section-targeted, provenance-tagged extraction. Small,
  independent units; nothing touches SQGLP scoring.
- **Authority:** v05 roadmap §12 Phase 0 (with §4.3, §7.1, §7.2, §10 cross
  references) governs behavior; this plan's Key Technical Decisions govern
  mechanism; `CLAUDE.md` conventions govern style. Where the plan conflicts
  with observed code reality, surface it rather than guessing.
- **Stop conditions:** Stop and surface if (a) the Screener quarterly section
  is absent or structurally different from the annual tables across all three
  reference tickers, (b) score persistence cannot obtain a stable registry
  hash without engine changes deeper than exposing a property, or (c) any
  change alters existing composite scores or report content (Phase 0 is
  additive data plumbing only).
- **Execution profile:** Code with unit tests per unit; parser tests use
  inline HTML literals per the `tests/test_financials_fetch.py` pattern —
  never live scraping, never `raw_data/` dependence.
- **Tail ownership:** Implementer owns commit hygiene and a final smoke run:
  refetch + `python -m boundless100x compute ASTRAL` twice, then verify
  `quarterly.csv` exists and `score_history.jsonl` gained two rows with
  identical `config_hash` and `verdict: "indeterminate"` — ASTRAL has no
  `price_volume.csv`, so price-dependent gates are indeterminate by design
  and that value is the expected smoke assertion, not a failure
  (rev 2026-08-06).

---

## Product Contract

### Summary

Three additive capabilities: (1) `quarterly.csv` joins the per-ticker data
contract, parsed from the same cached Screener page the annual tables come
from; (2) every scored run appends one row of score history stamped with a
registry-config hash; (3) the AR fetcher retains 2–3 years of reports and
extracts named sections with `found`/`fallback` provenance instead of a blind
first-30-pages string.

### Problem Frame

The v05 lifecycle (Phase 1+) evaluates quarterly checkpoints, computes score
momentum, and mines annual reports for guidance — none of which today's data
layer supports. There is no quarterly results parser (`fetch_financials.py`
parses only annual P&L/BS/CF/ratios plus shareholding). There is no run
history: `scores.json` is overwritten per report dir, the watchlist keeps only
`last_run`/`last_composite`, and nothing records which scoring config produced
a number. The AR pipeline keeps one report, extracts the first 30 pages
(~chairman's letter), truncates twice (5,000 chars in the suite, then a
hard-coded 3,000 in the orchestrator), and has zero section awareness. Every
week without persistence is history that can never be recovered.

### Requirements

- R1. `quarterly.csv` is written to `raw_data/{TICKER}/` on fetch, containing
  the Screener quarterly results table (period, sales, expenses, operating
  profit, OPM%, other income, interest, depreciation, PBT, tax%, net profit,
  EPS as available), and is exposed to the compute engine as
  `data["quarterly"]`.
- R2. Every completed `service.analyze()` run (LLM or not) appends exactly one
  row to an append-only score-history store:
  `{schema_version, ticker, date, composite, elements, verdict, coverage,
  flags, config_hash, synthetic}`. Runs whose scoring failed (`scores == {}`)
  append nothing and log why.
- R3. `config_hash` deterministically identifies the metric-registry state
  (element weights, eligibility gates, per-metric configs) such that any
  change to weights, thresholds, or metric definitions changes the hash.
- R4. The AR fetcher retains up to `max_reports: 3` reports (download already
  skips existing files; extraction extends to all retained PDFs, not just the
  newest) and produces per-report, per-section extraction output where each
  section carries `provenance: found | fallback`.
- R5. Section extraction targets at minimum: `mdna` (Management Discussion &
  Analysis), `chairman` (chairman/MD letter), and `governance` (related-party
  / auditor / board report signals). Each section has its own character cap;
  a section not located falls back to first-N-pages text tagged `fallback`.
- R6. The LLM layer's Pass 1 text budget becomes config-driven (the existing
  hard-coded `[:3000]` in the orchestrator is replaced by a config value), so
  per-section caps are not silently overridden downstream.
- R7. Existing behavior is preserved: `data["annual_report_text"]` (combined
  most-recent-report string) still exists for current consumers, and annual
  parsing, scoring, and eligibility are unchanged for the same inputs.
  **Byte-identical is scoped (rev 2026-08-06):** the guarantee covers
  `--no-llm` artifacts — `scores.json`, `eligibility.json`,
  `raw_metrics.json`, and rendered reports for a compute-only run. AR-text-
  dependent content is excluded: when sections are `found`, the combined
  string differs from today's first-30-pages extraction by design (KTD6
  bounds that drift), and LLM output is nondeterministic in any case.

### Scope Boundaries

- **No lifecycle machinery.** States, triggers, checkpoints, and the
  checkpoint vocabulary are Phase 1 (v05 §12). Phase 0 only makes the data
  exist.
- **No synthetic backfill runs.** The `synthetic` field ships in the schema
  (default `false`); actually generating U9-backfilled rows is a Phase 2
  option (v05 §7.1).
- **No quarterly-derived metrics.** No metric YAML or scoring change of any
  kind; `data["quarterly"]` is plumbed but unconsumed until Phase 2.
- **No new data sources.** The quarterly parser reads the already-cached page;
  AR retention raises a count against the same BSE API.
- **Known limitation, recorded not fixed:** Screener's quarterly table is
  shallow (~12 recent quarters) — sufficient for consecutive-quarter checks,
  insufficient for deep simulator replay (v05 §10). Tickers whose page was
  cached before the page-caching fix need one refetch for the table to be in
  cache.

---

## Planning Contract

### Key Technical Decisions

- **KTD1 — Reuse the generic table parser.** The quarterly table is a
  standard Screener `data-table` under section id `quarters`. Extend
  `_parse_table(soup, section_id, label_map)` with an optional
  `period_col="year"` parameter and call it with `("quarters", QTR_LABEL_MAP,
  period_col="quarter")`. No new parsing machinery; a missing section returns
  an empty frame exactly as the existing tables do.
- **KTD2 — History is one git-tracked JSONL file.** 
  `boundless100x/score_history.jsonl`, one JSON object per line, append-only.
  Rationale: `watchlist.json` sets the precedent for git-tracked state;
  `output/` and `raw_data/` are gitignored and would silently lose history.
  Same-day reruns append duplicate-dated rows by design — readers take the
  last row per (ticker, date, config_hash); no in-place rewrites ever.
- **KTD3 — Hash the loaded registry, not the YAML files.** `ComputeEngine`
  already assembles the authoritative dict (element weights + eligibility
  gates + discovered metric configs). Compute
  `sha256(canonical_json(registry))[:12]` once at engine init and expose it
  as `engine.registry_hash`. Hashing the in-memory dict (sorted keys,
  `default=str`) catches custom-metric drop-ins and survives YAML
  reformatting; hashing file bytes would do neither.
- **KTD4 — The persistence hook lives in `service.analyze()` after action
  resolution.** The one point where scores, eligibility verdict, and
  final action all exist on the result (currently between the
  `resolve_action` assignment and `return`). It runs for `--no-llm` and
  `analyze_quick` for free. Report generation stays out of it — history is a
  service concern, not an output format.
- **KTD5 — Section detection is heading-regex over per-page text, honest on
  failure.** Extract per-page text once; locate section starts by
  case-insensitive heading patterns ("management discussion", "chairman's /
  MD's letter", "directors' report / board's report", "related party");
  a section runs to the next detected heading or its page budget. Any section
  not found → that section's slot carries first-N-pages text with
  `provenance: fallback`. Detection quality is expected to be imperfect —
  provenance is the contract that makes imperfection safe (v05 §7.2 rule:
  section-dependent sub-metrics read provenance and go indeterminate on
  `fallback`).
- **KTD6 — Extraction output is a dict, compatibility string preserved.**
  `download_and_extract` returns
  `{year: {section: {"text": str, "provenance": "found"|"fallback"}}}` for
  all retained reports; the suite continues to publish
  `data["annual_report_text"]` (most recent report, sections concatenated,
  existing char cap) and adds `data["annual_report_sections"]` for Phase 2
  consumers. No current consumer changes behavior.
  **Drift bound (rev 2026-08-06):** the concatenated compatibility string is
  assembled in page order from the same first-N-pages extraction window used
  today, so the `found`-case string preserves today's prefix as far as
  section boundaries allow, and the all-`fallback` case reproduces today's
  string exactly. Byte-identical where possible, bounded drift elsewhere —
  per the scoped R7.

### Assumptions

- A1. The Screener consolidated company page includes the quarterly section
  for the reference tickers (Astral, Bajaj Finance, TCS); banks/financials
  may use variant row labels — the label map tolerates missing labels the
  same way the annual maps do.
- A2. `AnalysisResult.eligibility` exposes a final verdict string usable as
  the history row's `verdict`; absent/failed eligibility records
  `"indeterminate"`.
- A3. BSE serves ≥2 prior-year annual reports for most covered tickers; where
  it serves fewer, retention simply holds what exists (promises-kept will
  evaluate indeterminate in Phase 2 — acceptable).

### High-Level Technical Design

```
fetch_financials.py      _parse_all() ── + quarterly ──▶ raw_data/{T}/quarterly.csv
                                                          └▶ data["quarterly"]

engine.py                registry ──▶ registry_hash (sha256[:12])
service.py               analyze() ──▶ score_history.append_run(result, hash)
score_history.py (new)   append_run() / load_history(ticker)   ──▶ score_history.jsonl

download_annual_reports.py   download(max_reports=3) ──▶ {year}_annual_report.pdf ×N
                             extract_sections(pdf) ──▶ {section: {text, provenance}}
suite.py                 data["annual_report_text"] (unchanged)
                         data["annual_report_sections"] (new)
orchestrator.py          [:3000] literal ──▶ config llm.pass1_ar_char_budget
```

---

## Implementation Units

### U1. Quarterly results parser

- Add `QTR_LABEL_MAP` and a `_parse_table` call for section id `quarters`
  with `period_col="quarter"` in `fetch_financials.py`; wire into
  `_parse_all` (`"quarterly"` key) and `_save_to_disk` (`quarterly.csv`).
- Suite: pass `data["quarterly"]` through `fetch_all` like the other frames.
- Tests: extend the inline `SCREENER_HTML` literal with a `quarters` section;
  assert frame shape, `quarter` column naming, numeric coercion, and that a
  page *without* the section yields an empty frame and still writes the
  other CSVs (graceful absence, no exception).
- **Done when:** refetched reference tickers produce `quarterly.csv` whose
  rows match the quarters visible on Screener; graceful absence is proven by
  the synthetic no-section fixture; and old-vs-new parser output is identical
  on every pre-existing key across the cached pages (additive-only proof).
- **Correction (2026-08-06, implementation):** the earlier claim that "2 of
  the 7 currently cached pages" lack the `quarters` section is **false** —
  all 7 carry the section with a populated `data-table` (Rain, CDSL,
  Edelweiss, Control Print, CAMS, Zydus, IGI; 11–13 quarters each). No cached
  ticker exercises the absent-section path, so graceful absence is proven by
  the synthetic fixture only, and the real-data criterion is replaced by the
  old-vs-new equivalence check above. Stop condition (a) does not trigger:
  the section is present and structurally identical to the annual tables.

### U2. Registry hash

- `ComputeEngine`: compute `registry_hash` at init from the canonical-JSON
  sha256 of the merged registry dict; expose as a property.
- Tests: hash is stable across two engine instances; changes when an element
  weight, a metric threshold, or a custom-metric drop-in changes; unaffected
  by dict ordering.
- **Effective-gates coverage (rev 2026-08-06):** the hashed dict must include
  the gates the evaluator *actually uses* — when `EligibilityEvaluator` falls
  back to its code-level `DEFAULT_GATES` (registry YAML section absent), the
  hash input is those effective defaults, not the empty YAML section. A code
  edit changing `DEFAULT_GATES` then flips the hash like any gate change,
  and history rows never mix gate regimes under one hash.
- **Done when:** two engines over the same registry agree; a temporary custom
  YAML flips the hash.

### U3. Score-run persistence

- New `boundless100x/score_history.py`: `append_run(result, config_hash,
  path=DEFAULT)` writing the R2 row (`schema_version: 1`, `synthetic: false`,
  ISO date, elements, composite, verdict per A2, coverage + flags from
  scores); `load_history(ticker, path=DEFAULT)` returning parsed rows
  (Phase 2 consumer, trivial now).
- Hook in `service.analyze()` per KTD4; guard: empty `scores` → log +
  no-append; history-write failure appends to `result.errors`, never raises.
- Tests: two runs append two rows; failed scoring appends none; row
  round-trips through `load_history`; file is append-only (existing lines
  untouched byte-for-byte).
- **Done when:** `compute ASTRAL` twice yields two rows with identical
  `config_hash` and `verdict: "indeterminate"` (expected for ASTRAL — no
  `price_volume.csv`; rev 2026-08-06), and the smoke run in the Goal Capsule
  passes.

### U4. AR retention + section extraction + provenance

- Config: `max_reports: 3`; new `annual_reports.sections` block with
  per-section char caps (`mdna`, `chairman`, `governance`); keep
  `max_text_chars` as the combined-string cap for the compatibility path.
- `download_annual_reports.py`: extract for all retained PDFs (not
  `pdf_paths[0]` only); add `extract_sections(pdf_path, sections_cfg)` per
  KTD5 returning KTD6's shape; `.txt` sidecar cache becomes per-section JSON
  sidecar (`{year}_annual_report.sections.json`).
- Suite: publish `annual_report_sections` (all years) + unchanged
  `annual_report_text` (most recent, concatenated, capped).
- Tests: synthetic multi-page PDF fixtures (built in-test with PyMuPDF) — one
  with recognizable headings (assert `found` + correct text slice + cap
  enforcement), one without (assert `fallback` + first-N-pages text); newest
  vs. multi-year extraction.
- **Done when:** a real refetched ticker yields ≥2 retained reports and a
  sections file where at least `mdna` is `found` for a company with a
  standard AR, and `fallback` never raises.

### U5. Config-driven Pass 1 budget

- Replace the orchestrator's hard-coded `annual_report_text[:3000]` with
  `llm.pass1_ar_char_budget` (default 3000 — behavior-preserving).
- Test: orchestrator respects an injected config value.
- **Done when:** grep finds no bare `3000` truncation literal in `llm_layer/`.

---

## Verification Contract

- Full suite green via `venv/bin/python -m pytest tests/` (network tests
  remain deselected).
- v05 §12 Phase 0 validation gates, verified on refetched reference tickers:
  `quarterly.csv` reproduces Screener's displayed quarters; two consecutive
  scored runs produce two history rows; extracted AR sections contain MD&A
  text beyond page 30 (or are honestly tagged `fallback`).
- Regression: for an unchanged cached ticker, `scores.json`,
  `eligibility.json`, and the rendered reports are unchanged by Phase 0
  (additive-only proof).
- No unit depends on another's merge order except U3 → U2 (hash must exist
  before persistence stamps it).

## Definition of Done

All five units merged with tests; smoke run from the Goal Capsule passes;
`CLAUDE.md` data-contract section updated (`quarterly.csv`,
`score_history.jsonl`, sections sidecar); v05 roadmap Phase 0 items checked
off with a pointer to this plan.
