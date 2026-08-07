# Boundless100x — SQGLP Financial Research System

## Project Overview
Deep company analysis system for long-term investment in Indian markets, using the SQGLP framework (Size, Quality, Growth, Longevity, Price). Computes 51 financial metrics locally, uses Claude API for qualitative analysis.

## Design Document
`Design/Financial Model v04.md` — The complete system design. Always reference this for architecture, metric definitions, data flows, and implementation details.

## Architecture
4-stage pipeline + service layer:
- **Stage 1**: Data fetch (Screener.in, yfinance, BSE, Trendlyne) → `data_fetcher/`
- **Stage 2**: Compute engine with YAML-driven metric registry (51 metrics) → `compute_engine/`
- **Stage 3**: SQGLP scoring + growth decomposition → `compute_engine/scorer.py`
- **Stage 3.6**: 100x eligibility gates (conjunctive; separate from the additive composite) → `compute_engine/eligibility.py`
- **Stage 1.5**: Forward-growth extraction from annual-report prose → `llm_layer/forward_growth.py`
- **Stage 4**: LLM analysis (2-pass: qualitative, synthesis) → `llm_layer/`
- **Stage 4.7**: Score-trajectory momentum over the append-only history → `trajectory.py`
- **Stage 5**: Report generation (HTML/Plotly + Markdown) → `output/`
- **Service layer**: `service.py` orchestrates everything; CLI and future GUI call it

### LLM 2-Pass Analysis
- **Pass 1** (Sonnet/Opus): Qualitative analysis from annual report text — management, moat, risks
- **Pass 2** (Sonnet/Opus): Investment thesis synthesis combining metrics + qualitative output
- `--deep` flag uses Opus instead of Sonnet for both passes
- Pass 1 is skipped if no annual report is available (`skip_pass1_if_no_ar: true`)

## Project Structure
```
boundless100x/
├── __init__.py
├── __main__.py
├── config.yaml                   # All pipeline settings (cache TTL, LLM models)
├── service.py                    # Central API (GUI-ready) — orchestrates full pipeline
├── action_policy.py              # Deterministic cap on the displayed action
├── trajectory.py                 # Score momentum over the append-only history
├── forward_growth_schema.py      # Closed extraction contract; leaf module both
│                                 # llm_layer and compute_engine read
├── cli.py                        # typer CLI (analyze, compute, screen, watchlist)
├── watchlist.py                  # Watchlist management
├── watchlist.json                # Persisted watchlist data
├── data_fetcher/
│   ├── base.py                   # BaseFetcher (retry, rate limit, cache)
│   ├── suite.py                  # FetcherSuite — runs all fetchers for a ticker
│   ├── corpus_snapshot.py        # Snapshot/restore raw_data/ outside the repo
│   ├── refetch.py                # Refresh every cached ticker, cache bypassed
│   ├── corpus_audit.py           # What a refetch changed, read off the disk
│   ├── cache/
│   │   └── cache_manager.py      # TTL-based disk cache
│   ├── fetch_financials.py       # Screener.in scraper (P&L, BS, CF, ratios)
│   ├── fetch_price_volume.py     # yfinance (price history, volume)
│   ├── fetch_shareholding.py     # BSE quarterly shareholding
│   ├── fetch_corporate_actions.py
│   ├── fetch_analyst_coverage.py # Trendlyne analyst data
│   ├── download_annual_reports.py # BSE annual report PDF → text extraction
│   ├── sector_context.yaml       # Sector-specific metric thresholds
│   └── raw_data/{TICKER}/        # Fetched data (JSON/CSV/TXT per ticker)
├── compute_engine/
│   ├── engine.py                 # Auto-discovery metric runner
│   ├── scorer.py                 # SQGLP weighted scoring (threshold, trend, range, percentile)
│   ├── screener.py               # Preset-based universe screening
│   └── metrics/
│       ├── registry.yaml         # SQGLP element weights (S:10, QB:20, QM:10, G:25, L:20, P:15)
│       ├── base.py               # MetricResult dataclass
│       ├── validator.py          # Metric validation rules
│       ├── elements/             # Per-SQGLP element YAML definitions
│       │   ├── size.yaml
│       │   ├── quality_business.yaml
│       │   ├── quality_management.yaml
│       │   ├── forward_growth.yaml   # Phase 2 — element absent from element_weights
│       │   ├── growth.yaml
│       │   ├── longevity.yaml
│       │   ├── price.yaml
│       │   └── composite.yaml
│       ├── builtin/              # Python metric implementations
│       │   ├── _helpers.py       # Shared utilities (MAD-based FCF outlier detection)
│       │   ├── forward_growth.py # Phase 2 sub-metrics (all weight 0)
│       │   ├── size.py
│       │   ├── profitability.py
│       │   ├── efficiency.py
│       │   ├── leverage.py
│       │   ├── growth.py         # CAGR, dilution (bonus/split-aware)
│       │   ├── longevity.py      # FCF consistency (outlier-aware)
│       │   ├── valuation.py      # DCF, reverse DCF (outlier-aware)
│       │   └── composite.py
│       ├── custom/               # User drop-in metrics (empty — add YAML + .py)
│       └── presets/              # Screening presets
│           ├── compounders.yaml
│           └── hidden_gems_100x.yaml
├── lifecycle/                    # Phase 1 — the layer after the verdict
│   ├── states.py                 # screen → qualify → watch → probe → scale;
│   │                             # also owns `as_date` and `last_record_into`,
│   │                             # the history rules that must agree with
│   │                             # themselves across the layer
│   ├── triggers.yaml             # Declared transitions + kill-switches, lane-scoped
│   ├── evaluator.py              # TriggerEvaluator (mirrors EligibilityEvaluator)
│   ├── checkpoints.py            # Machine-checkable half of Pass 2 monitorables
│   ├── checkpoint_vocabulary.yaml
│   ├── advance.py                # Re-score, evaluate, propose
│   ├── pace.py                   # Phase 2 — corpus-median entry-pace modulator
│   ├── lane_gates.py             # Phase 3 — LaneGateEvaluator, the third gate
│   ├── lane_gates.yaml           # evaluator beside eligibility + triggers
│   ├── friction.py               # Phase 3 — modelled net-of-tax-and-slippage return
│   ├── portfolio.py              # Phase 3 — count-based concentration guardrails
│   ├── exit.py                   # Phase 3 — confirm_exit, the ONLY path to `exited`
│   ├── reinvestment.py           # Phase 3 — append-only queue + routing proposal
│   ├── reinvestment_queue.json   # Persisted queue (tracked, sibling to watchlist.json)
│   └── lane_view.py              # Phase 3 — pure lane context both surfaces share
├── llm_layer/
│   ├── orchestrator.py           # LLM pipeline (2 passes + extraction call)
│   ├── checklist.py              # Pre-flight data quality checks
│   ├── forward_growth.py         # Phase 2 — content gate, validator, sidecar
│   ├── sweep.py                  # Priced extraction sweep (dry run, ceiling)
│   └── prompts/
│       ├── pass1_qualitative.txt # Annual report deep dive
│       ├── pass2_synthesis.txt   # Investment thesis
│       └── forward_growth_extraction.txt  # Phase 2 extraction (Stage 1.5)
└── output/
    ├── report_generator.py       # HTML dashboard + Markdown report
    ├── templates/
    │   ├── sqglp_report.html.j2
    │   └── sqglp_report.md.j2
    └── reports/{TICKER}_{DATE}/  # Generated reports (HTML, JSON)
```

## Key Patterns
- **Metric registry**: YAML defines metrics in `elements/*.yaml`, Python functions in `builtin/*.py`. Engine auto-discovers both and rejects duplicate metric ids at startup. Adding a metric = 1 YAML entry + 1 function.
- **Two outputs per company**: the additive SQGLP composite answers "is this a quality compounder?"; the conjunctive eligibility gates in `registry.yaml` answer "could this plausibly 100x?". A company can score well and still fail a gate — that is the point, not a bug. Gates declare `veto_sources`: if the metric that would emit a veto flag (e.g. `reverse_dcf_growth`) is unavailable, the gate reads `indeterminate` rather than passing on ratios alone.
- **Action guard** (`action_policy.py`): Pass 2 returns a `suggested_action`, but the action any surface *displays* is resolved in deterministic code. When the 100x verdict is `not_eligible`/`indeterminate`/missing, or the score carries `low_data_coverage`, a `buy`/`strong_buy` is capped to `watchlist` and the reason travels with it. Capping is not overriding — failing a gate makes a company an unlikely hundred-bagger, not a bad investment, so the action is lowered rather than flipped to `avoid`, and the model's original is always preserved as `llm_action`. The verdict is also given to Pass 2 as context, but prompt compliance is never the guard.
  - **`resolve_for_result(result)` is the single derivation**, called fresh by the service (Stage 4.5), `ReportGenerator._resolve_action`, and the CLI. It reads only `llm_analysis`, `eligibility` and `scores`. `AnalysisResult.final_action` is an **output** of that function and must never become an input to it — it is mutable and gets serialised into reports, so anything that rescores or re-evaluates eligibility afterwards leaves it stale, and a stale decision is as dangerous as an absent one. The render boundary logs a warning when a stored decision disagrees with the recomputed one.
- **Macro assumptions** (inflation, G-Sec yield, discount rate, terminal growth) live in `config.yaml` under `macro:` and reach metrics as parameter defaults; a metric's own YAML params override them. **Owner *policy* blocks sit deliberately outside the hashed `macro:` block** — `deployment_pace:`, `friction:` (STCG/LTCG rates, holding-period boundary, slippage) and `portfolio:` (sleeve split, tranche sizing, per-lane and per-sector name caps) are preferences, not assumptions a metric computes with, so tuning them must not move a scoring hash. Every threshold in them is a starting point awaiting Phase 4/5 simulator evidence.
- **Lifecycle** (`lifecycle/`, Phase 1 of the v05 roadmap): the layer *after* the verdict. States are `screen → qualify → watch → probe → scale`, with `exit_review → exited` and `dropped`; `states.py` is the definition. Transitions are declared in `lifecycle/triggers.yaml` and evaluated by `TriggerEvaluator`, which **mirrors `EligibilityEvaluator`** — same imported `COMPARATORS`, same three-valued outcome, same per-condition `detail` strings, same "indeterminate, never a silent pass" rule. Registry validation runs at construction: unknown states, comparators, and metric ids are startup errors, because a trigger naming a nonexistent metric would read indeterminate forever and a kill-switch that never fires looks exactly like a thesis that never broke.
  - **`persist_years` is allowlisted** (`SERIES_SAFE_METRICS`). `raw_series` has no declared contract — `roiic` returns *capital employed* beside a percentage value, `pe_vs_historical` returns P/E multiples beside a percentile — so a consecutive-year rule on either would compare incompatible units and silently never fire. Adding a metric to the allowlist means reading its implementation first.
  - **Checkpoints** (`lifecycle/checkpoints.py`, vocabulary in `checkpoint_vocabulary.yaml`) are the machine-checkable half of Pass 2's monitorables; the prose half is unchanged. The vocabulary is closed to quarterly-readable series (Phase 0's `quarterly.csv` columns plus quarterly shareholding) so a checkpoint can always come due; an id outside it is demoted to prose at recording time. A data gap is `indeterminate`, never `missed` — and zero misses out of zero *due* checkpoints is indeterminate too, so an unmonitored position never reads like a verified one.
  - **`watchlist advance`** re-scores, evaluates, and proposes. Transitions that move money (`probe`, `scale`, `exit_review`, `exited`) are proposed and wait for `--apply`; pre-position transitions (`qualify`, `watch`, `dropped`) auto-apply. When several triggers fire, the most protective wins — a kill-switch outranks a buy-zone, so a company never gets bought into on the quarter its thesis broke.
- **Annual report sections** (`download_annual_reports.py`): extraction is section-targeted, not first-N-pages. `extract_sections()` returns `{section: {text, provenance, start_page}}` for `mdna`/`chairman`/`governance` (caps in `config.yaml` under `annual_reports.sections`), cached in a `{year}_annual_report.sections.json` sidecar; `download_and_extract()` returns `{year: {...}}` across the retained reports (`max_reports: 3`, so promises-kept can compare guidance to delivery). **`provenance` is the contract**: `found` means the section was located, `fallback` means the slot holds first-N-pages text instead — Phase 2 sub-metrics must evaluate indeterminate on `fallback` rather than mine a chairman's letter for guidance (v05 §7.2). Detection is heuristic heading-matching over `scan_pages: 150` (MD&A starts at pages 20–147 in the fetched corpus, so the 30-page fallback window is far too shallow to search); two adversaries, both real. The **contents page** lists every section name in heading form, and is rejected by three guards (Contents/Index title, bulk page-number entries, ≥2 distinct section names on one page). The **cross-reference** is subtler and was the cause of an 8-of-18 wrong-section rate before it was fixed: a report saying "provided in the Management Discussion and Analysis" scored as a heading under the original short-line-mostly-the-name test, so slices opened on auditor's reports, governance and CSR. `_is_heading_like` now requires the match to *open* its line (leading numbering/bullets stripped) and to not be continued by a lowercase word — position and continuation separate the two, where coverage ratio cannot. Detection also records the heading's **line**, not just its page: a heading low on a page (one real report has it at line 40 of 62) otherwise drags the preceding section's tail into the slice. Measured across the refreshed corpus of **54 report-years** (2026-08-07): raw detection finds mdna 28, chairman 20, governance 49; after the content gate 24 / 18 / 33 survive, so mdna precision holds at ~86% (it was 85% on the 29-report-year corpus, and the rate holding while absolute reach doubled is what says it is a property of the detector rather than of the sample). A residual wrong-section rate remains and consumers must still treat provenance as a claim rather than a guarantee. `combined_text()` builds the single-string view older consumers read — found sections in page order, or, when none were found, the fallback text once, which keeps `annual_report_text` byte-identical to pre-section behaviour. Pass 1's own cap is `llm.pass1_ar_char_budget` (default 3000): it sits downstream of the per-section caps and is the binding limit on what reaches the prompt, so raising a section cap without raising this one changes nothing.
- **Score history** (`score_history.py` → `boundless100x/score_history.jsonl`): every scored `service.analyze()` run appends one row (`schema_version, ticker, date, composite, elements, verdict, coverage, flags, config_hash, synthetic`) at Stage 4.6. Git-tracked and **append-only by contract** — a score not written when the run happened cannot be recovered, so nothing ever rewrites a line. Same-day re-runs append duplicates; `load_history` resolves them at read time (last row wins per ticker/date/`config_hash`), while rows under different hashes are both kept because they are different scoring regimes. A run whose scoring failed appends nothing; a write failure lands in `result.errors` and never costs the caller the analysis. Tests never touch the real file — an autouse `conftest` fixture redirects the module default, and `service.history_path` (config `output.score_history_path`) redirects per caller. The backtest bypasses `service.analyze()`, so truncated-history scores never enter the organic log; synthetic backfill rows (v05 §7.1, Phase 2) must set `synthetic: true`.
- **Two regime hashes** (Phase 2, KTD8). `engine.registry_hash` is a 12-char sha256 over everything that can move a score — the whole `registry.yaml` (element weights, declared gates, `history_waiver_mcap`), the *effective* gates, the definitions of **scored** metrics, and the macro block. `engine.forward_signal_hash` covers the **zero-weight** metric definitions plus the extraction schema. Score-history rows carry both; momentum groups on `config_hash` (the scoring hash) alone. The split is not tidiness: with one hash, tuning a zero-weight forward signal would reset every ticker's momentum baseline unrecoverably, and **Phase 5 needs trajectory evidence to calibrate those signals while calibrating them would destroy it**. Macro sits in both, correctly — it reaches every metric as a parameter default. The hash covers the loaded registry rather than YAML bytes, so custom drop-ins count and reformatting does not; `_`-prefixed provenance keys are excluded so a file rename does not fragment history. `effective_gates()` in `eligibility.py` is the single statement of the "no declared gates falls back to `DEFAULT_GATES`" rule — both the hash and the service's evaluator resolve through it, so the regime recorded always equals the regime enforced.
- **Forward signals** (Phase 2, `v05 §7.1–7.3`, `§11`): four additive capabilities, none of which may move a score.
  - **Zero weight, and an unweighted element.** `rerating_headroom` sits in `price.yaml` at `weight: 0.0`; the four forward-growth sub-metrics sit in `elements/forward_growth.yaml`, whose element name is **absent from `element_weights`**. Belt and braces: the scorer's `weight == 0` branch returns before weighted accumulation (so they reach neither an element mean, the composite, nor the coverage denominator), and an element the scorer never iterates makes them structurally incapable of scoring. The validator excuses a zero-weight metric from mode-specific scoring config, because the scorer would never read it.
  - **The extraction seam is the `data` dict, and its direction is load-bearing** (KTD2). `compute_engine/` imports nothing from `llm_layer/`. Extraction runs at **Stage 1.5** in `service.analyze()` and reaches metrics only as `data["forward_growth"]`. A metric that could call an API would issue one call per ticker per backtest, reading *today's* report text against *truncated* financials — the exact look-ahead leak the backtest exists to prevent. The shared contract (closed field vocabulary, required-section map) lives in `forward_growth_schema.py`, a dependency-free leaf both layers read.
  - **Hydration and extraction are separately gated.** Stage 1.5 reads a valid sidecar on *every* run including `use_llm=False`; only creating or refreshing one calls the model. Gating the whole stage would mean the cache is never read on the paths it exists for — `watchlist advance` re-scores with `use_llm=False`. Three outcomes stay distinguishable: key absent (could not look), `{}` (looked, nothing readable — free), populated. The sidecar versions against source text, field schema, prompt digest and model id.
  - **Provenance is three-valued** (KTD9): `found` / `suspect` / `fallback`. `found` is a claim, not a guarantee — heading detection over arbitrary filer layouts is never exact. A `found` section must also *look like* the section it claims to be or it is downgraded to `suspect` and excluded exactly as `fallback` is. The gate asks two questions: does the slice open like a *different* section (an audit opinion's canonical opening, a governance philosophy, a pointer to an annexure), and is it about the right subject (its own heading opens the slice, or subject markers appear early). **Write markers against how a section actually reads, not the structure a statute requires of it** — a first design using SEBI LODR Sch V(B)'s mandated MD&A contents rejected 11 of 13 real slices, 11 of which were genuine. Corpus rates after the rebuild: mdna 11/13 kept, chairman 9/11, governance 22/26. `suspect` is reported separately from `fallback` because how often the tag was wrong is what says whether the gate works.
  - **Boundary validation is grounding, not just shape** (KTD3). Type-checking cannot tell a well-formed reading from a well-formed fabrication. Every entry's quoted sentence must be a literal substring of the text actually submitted, and its own value and period must appear inside that sentence. Note the gap that keeps the content gate non-redundant: grounding *passes* on a suspect slice, because the quoted sentence really is in the submitted text.
  - **Guidance carries its `subject`, and only `company` is a promise.** Market, industry and economy growth rates far outnumber company-subject ones in the same MD&A sections and the same sentence shape — the corpus scan put it near four to one, and the live extraction sweep came out starker still at 29 market-subject guidance statements against 3 company-subject. A percentage is the one figure where subject and quantity cannot be told apart by type, grounding, or unit: one real filing reads "Company expects market to grow by 4-5%", naming the company and promising nothing. `promises_kept_ratio` counts only `company`; market entries are stored, because a grounded reading is worth keeping even when nothing reads it. Growth guidance (`revenue_growth_pct`, `pat_growth_pct`) settles against the change between **two consecutive annual rows, both looked up by label** — settling FY2026 against FY2024 because FY2025 is missing would read a kept promise as spectacularly beaten.
  - **A figure is stored in the unit the filing stated, never converted** (KTD5). `unit` is a closed set: `inr_cr`/`inr`/`pct` settle against the accounts, `usd_mn`/`usd_bn`/`usd_tn`/`inr_lakh`/`inr_mn`/`inr_lakh_cr` are kept and skipped. Grounding asks whether the numeral is denominated the way the entry claims, so a USD figure grounds *as* a USD figure and the reading metrics refuse it with the reason. **Every INR-comparable metric must check** — the trap being a field name like `amount_inr_cr` that asserts a unit `unit` makes variable, which is how the now-retired `capex_pipeline` came to sum a USD figure into a rupee total. Discarding instead of storing (the pre-2026-08-07 rule) lost the reading *and* hid the gap, which then surfaced as an absent signal indistinguishable from a filing that said nothing. `usd_tn` and `inr_lakh_cr` were added only after a live sweep met them; a unit vocabulary is built from filings, not from first principles.
  - **Momentum refuses to diff across regimes** (`trajectory.py`, KTD5). Rows partition by `config_hash`, and separately by `synthetic`; a backfilled row never supplies the headline. Every figure states the actual day gap it spans. **Insufficient history is a distinct outcome, never a zero** — a zero delta means flat, no delta means unknown, and they look identical in a table. Kept out of `score_history.py`, whose contract is append and read.
  - **Pace modulation is entry-only** (`lifecycle/pace.py`, KTD7). §11's named input `earnings_yield_vs_gsec` is per-*company*, so wiring it in would tighten entry when the company is expensive — the inverse of the purpose. The input is the **median across the cached corpus**, computed once per run before the ticker loop, needing no new source and no number anyone must refresh. It derives a threshold-tightened *copy* of the `→ probe` triggers and passes it through the existing evaluator-injection seam; a spread *condition* would make macro a gate. Kill-switches, drops, and eligibility gates are untouched by construction. Too few contributors leaves thresholds alone: an unknown macro reading must not tighten entry any more than it may loosen it.
  - **Every zero-weight metric's flag is registered** against `FORWARD_SIGNALS_ELEMENT` in `report_generator.FLAG_ELEMENT_MAP` (KTD6). That map falls back to `"composite"`, so an unregistered flag would render as an SQGLP signal on a ticker whose score did not move. Phase 3 shipped one unregistered and proved the point; the rule is now **mechanical rather than remembered** — the test derives the flag set from the registry (every metric at `weight: 0`, every flag literal its implementation can emit) instead of matching hardcoded id prefixes, which is what let the gap through. No zero-weight metric joins `SERIES_SAFE_METRICS`.
  - **R8 makes presentation load-bearing.** Zero weight means these metrics never receive a score, so the number is all a reader gets — and a bare number is not signal. The Forward Signals report section renders each with its direction of goodness and an interpretation band, states that nothing in it touched the composite, and shows an indeterminate signal as unknown *with its reason*.
- **Two lanes** (Phase 3, `v05 §4.4`, `§8`, `§9`): the same state machine, two parameter sets. `LANES` is `core` (the compounder path) and `rerating` (the fast lane, 6–18 months, monetizing re-rating rather than duration).
  - **A lane with its own gate set must not also be gated by the other's.** Four core triggers carry `lane: [core]` — without that, `qualification_failed` would drop any re-rating candidate the *100x* gates reject before its own gates were ever consulted, and `awaiting_entry_price` would strand it. The fast lane has a complete path of its own, including its own drop rule. The **six fundamentals kill-switches and `fundamentals_deteriorated` stay universal**, deliberately: §6.2 is explicit that the fast lane never trades through a fundamentals break. A trigger's absent `lane` key means "every lane"; `evaluate()` must forward `lane` to `applicable()` or the filter is dead code.
  - **`lane_gates.yaml` is the third gate registry**, beside `registry.yaml`'s eligibility gates and `triggers.yaml`'s transitions. `LaneGateEvaluator` is `EligibilityEvaluator`'s second sibling — same comparators, same three-valued outcome, same indeterminate-never-a-silent-pass rule — asking the lifecycle's question ("does this qualify for the fast lane now?") rather than the compute engine's ("could this plausibly 100x?"). Construct it with `known_metric_ids` or its unknown-metric check silently never runs.
  - **`exited` is reachable only through `watchlist exit`** (KTD10), never a trigger: no metric can observe that the owner sold, and a trigger firing on price would record a sale that may not have happened. `validate_triggers` rejects `to: exited` structurally. Two JSON files cannot be written atomically, so the window is made *recoverable*: validate everything first, write the queue event first keyed by an `exit_id` derived from the entry's own `exit_review` timestamp (so a retry computes the same one and the append is idempotent), write the transition second. **A retry adopts the stored date and payload rather than re-pricing** — it must complete the original sale, not a different one. The reverse order is unrecoverable by construction: an exited position with no event, and a state check that then refuses the very retry that could repair it.
  - **Both stores commit copy-on-write** through a shared `_JsonStore` base in `watchlist.py`: stage on a deep copy, write via `atomic_write_json` (temp file + `os.replace` + directory fsync), adopt only on success. A failed save leaves `self.data` equal to disk, so no phantom state survives for a same-process retry. Each commit bumps a monotonic `revision`, and `_commit` re-reads the on-disk counter first — a lost update is a loud refusal rather than a silent clobber. `get()` returns a detached object after any commit; re-read rather than holding one across a mutator.
  - **Friction is a modelled estimate, never a realized return** (KTD7). Every input is a proxy — a `probe` confirmation date rather than a fill, a market bar rather than a trade price, no cost basis — so `basis` distinguishes an in-flight `estimate` from a `recorded` reading, where "recorded" means the dates stopped moving, not that the figure stopped being a model. Slippage is a flat round-trip deduction applied before tax; STCG/LTCG is chosen by holding period against a configured boundary. §8.2's break-even is stated as the roadmap's rough 6–10pp with the assumptions listed, and **no hurdle is computed**: a tax-rate spread is a rate applied to gains, not a number of return points.
  - **Concentration counts names, because nothing counts rupees** (KTD8). The watchlist tracks no invested capital, so §8.1's percentage caps have no denominator; the config says plainly that the counts are proxies. Counts seed from the **live watchlist**, not from successful outcomes — a positioned ticker whose analysis errored must still count toward its lane.
  - **Routing safety is fail-closed per lane** (KTD11) and cannot reuse `action_policy.resolve_for_result`, which returns `None` on the `use_llm=False` path `advance` takes and would pass every candidate silently. Core candidates need an `eligible` 100x verdict; re-rating candidates need a `qualifies` **lane-gate** verdict — applying the 100x question there would reimpose the gate set §9.2 exists to replace. Anything other than the exact positive verdict blocks with its reason.
  - **Only a `Current` routing snapshot may render a proposal.** Freshness is tracked by **revision counters, not clocks** (`as_of` may be a historical business date, and a clock comparison misses every non-scoring mutation). A `--quarterly` run advances a subset and never overwrites the canonical snapshot. Routing records **deployment, not intent**: a route is refused unless the candidate holds an owner-applied position transition dated after the exit, and the idle reading closes at `deployed_at` rather than when someone typed the command.
- **MetricResult**: Every compute function returns `MetricResult(value, raw_series, flags, metadata, error)`. Flags communicate data quality issues (e.g., `insufficient_history`, `possible_bonus_split`, `cfi_dominated_by_acquisitions`).
- **Scoring**: Threshold-based (higher/lower_is_better), range_optimal, categorical, sector_relative_percentile, trend_direction modes. All defined in YAML. Scorer receives full MetricResult for trend analysis.
- **Data contract**: Fetchers write to `raw_data/{TICKER}/` in standardized CSV/JSON. Compute engine reads from there. BSE codes auto-detected from Screener.in metadata. `quarterly.csv` (Screener's quarterly results, `quarter` period column) is parsed from the same cached company page as the annual tables and shares `_parse_table` with them; it feeds Phase 1's checkpoints and Phase 2's `quarterly_momentum`. Screener renders only ~11–13 recent quarters, which is enough for consecutive-quarter checks but not for deep historical replay. The corpus was refetched on **2026-08-07** (`corpus refetch`), so all 22 cached tickers now carry a quarterly series — `quarterly_momentum` computes on 21 of 22, the exception being SPLPETRO, whose Screener page renders only 5 quarters against the 6 a second difference needs. `max_reports: 3` likewise applies from its landing forward: the corpus now holds 54 annual-report years across 20 BSE codes, up from 29. **A ticker fetched before either landed still has neither** — refetch to upgrade.
- **Screener page cache**: The company page HTML is cached via the TTL cache (`txt` entries), so repeat runs within the window do not re-scrape Screener. Parsing stays deterministic on the cached HTML.
- **Price series**: `price_volume.csv` carries both `close` (raw traded) and `adj_close` (split/dividend-adjusted). Valuation metrics use the raw close against as-reported EPS and record `price_basis` in metadata; the backtest's realized return prefers `adj_close`. When the fetch source has no real adjusted series (jugaad-data fallback), `adj_close` is aliased to `close` and `adj_close_is_estimated=True` marks the alias — the backtest refuses to score a realized return off it rather than risk reading a split as a crash. All 22 cached tickers carry the adjusted schema after the 2026-08-07 refetch; a file fetched before that holds a single legacy close with no alias flag at all, so refetch to upgrade. The adjusted column can also **trail the raw one by a bar** — the source publishes today's close before today's adjusted close — so anything reading the latest adjusted price must drop empty rows first rather than take the last one.
- **Growth quality**: `_grade_growth_quality` in `builtin/growth.py` is the single grader for both the scored `growth_quality_grade` metric and the report/LLM lever table. YoY leverage ratios share one helper, `_mean_yoy_ratio`.
- **FCF outlier detection**: MAD-based (Median Absolute Deviation) to identify M&A-dominated years. Applied in valuation.py, longevity.py, profitability.py via `_helpers.py`.
- **Bonus/split detection**: YoY equity capital spikes >50% flagged as structural events. Organic dilution computed separately in growth.py.
- **LLM prompt templates**: rendered through a single `.format()` call, so a literal brace in the JSON schema block is escaped by **doubling** it (`{{` / `}}`). This entry previously said quadruple — it was wrong, and no prompt file in the repo has ever done it: quadrupling renders a literal `{{` into the prompt and corrupts the schema the model is shown. Match the sibling prompt files, not a remembered rule.

## Tech Stack
- Python 3.11+
- Data: requests, beautifulsoup4, yfinance, pandas, numpy, scipy
- PDF: PyMuPDF (fitz)
- Viz: Plotly, Jinja2
- LLM: anthropic SDK (Sonnet/Opus for Pass 1-2)
- CLI: typer
- Config: PyYAML
- Environment: python-dotenv (.env for ANTHROPIC_API_KEY)

## Conventions
- All financial data is in INR Crores unless noted
- 10-year analysis window for financials, 5-year for most averages
- Scoring scale: 1-10 per element, 0-10 weighted composite
- SQGLP weights: Size 10%, Quality Business 20%, Quality Management 10%, Growth 25%, Longevity 20%, Price 15%
- LLM outputs are strict JSON with defined schemas
- Cache with TTL to avoid repeated scraping (2s rate limit between requests)
- BSE codes used for annual report downloads and shareholding data
- Test with: Astral, Bajaj Finance, TCS as reference companies
- `.env` file at project root for `ANTHROPIC_API_KEY` (loaded by python-dotenv)

## Known Issues (as of Aug 2026)
- **BSE code**: resolved from BSE's own active-equity scrip master
  (`data_fetcher/bse_codes.py`), cached for a week — Screener stopped rendering
  bseindia.com links. Matching is by BSE symbol, then normalised company name.
  Companies genuinely not listed on BSE (CDSL, BSE Ltd — both NSE-only) record
  `metadata.bse_listing = not_listed_on_bse` and skip BSE fetches; that is a
  fact, not a failure. A lookup that could not run reports `lookup_failed`.
- **Sector metadata**: only tickers fetched after the breadcrumb fix carry `metadata.sector` (extracted from the `/market/` breadcrumb, Broad Industry preferred; study-bucket matching is plural-tolerant). Older `raw_data/*/metadata.json` files lack it — refetch, or the Trendlyne analyst-coverage merge may backfill.
- **Reverse DCF bounds**: the implied-growth search is bounded to [-10%, +50%]; pinned results now carry the `reverse_dcf_saturated` flag with `saturated_at` in metadata instead of silently returning 50.0/-10.0.

## Commands
```bash
python -m boundless100x analyze ASTRAL          # Full pipeline (fetch + compute + LLM + report)
python -m boundless100x analyze ASTRAL --no-llm # Compute only (no LLM passes)
python -m boundless100x analyze ASTRAL --deep   # Use Opus instead of Sonnet for LLM
python -m boundless100x compute ASTRAL          # Metrics only (no fetch, no LLM)
python -m boundless100x screen --preset compounders       # Screen universe against preset
python -m boundless100x watchlist show          # View watchlist (lane, state, checkpoints)
python -m boundless100x watchlist add ASTRAL    # Track a company — starts at `screen`
python -m boundless100x watchlist add X --lane rerating   # Track in the fast lane
python -m boundless100x watchlist catalyst X \  # Record the one input no metric
    --description "Capacity commissioning" \    # can derive; §9.2 gates entry on it
    --expected-by 2027-06-30
python -m boundless100x watchlist catalyst X --spent   # Flip it; fires the exit rule
python -m boundless100x watchlist advance       # Re-score, evaluate triggers,
                                                # propose transitions with evidence.
                                                # Money-moving ones are proposals only.
python -m boundless100x watchlist advance --apply   # Confirm and record them
                                                # Reports the deployment-pace
                                                # reading first: when the corpus
                                                # is expensive, entry thresholds
                                                # are tightened and say so.

# ── The exit door and the reinvestment queue (Phase 3) ──
# `exited` has exactly one producer, and it is this command. No trigger can
# reach it: no metric can observe that the owner sold.
python -m boundless100x watchlist exit ASTRAL   # Refuses unless the entry is in
                                                # `exit_review`. Records the
                                                # transition, the modelled
                                                # friction reading, and exactly
                                                # one queue event. Interrupted?
                                                # Re-run it — same exit id,
                                                # adopts the stored figures,
                                                # completes the transition.
python -m boundless100x watchlist queue         # Pure read: routing snapshot with
                                                # its state (Unavailable/Partial/
                                                # Stale/Current — only Current
                                                # names a candidate), blocked
                                                # candidates with reasons, and
                                                # each unrouted exit's idle days.
python -m boundless100x watchlist queue route <exit_id> <ticker>
                                                # Records where the proceeds
                                                # actually went. Refuses unless
                                                # the candidate holds an
                                                # owner-applied position dated
                                                # after the exit — deployment
                                                # closes the timer, intent does not.

python -m boundless100x backtest                # Walk-forward self-check: score on
                                                # the first half of each cached ticker's
                                                # history, compare to realized returns

# ── Corpus maintenance ──
# raw_data/ is gitignored and is the only copy of everything ever fetched, so
# a refetch has no revert. Run these in order; refetch refuses to start without
# a snapshot, and the audit is what says whether the refetch helped or hurt.
python -m boundless100x corpus snapshot         # Copy raw_data/ outside the repo
python -m boundless100x corpus refetch          # Refresh every cached ticker,
                                                # fetch cache bypassed so it
                                                # reaches the network. Per-ticker
                                                # isolation; resumes if interrupted.
python -m boundless100x corpus audit            # Gains and regressions, counted
                                                # off the disk — the pipeline
                                                # cannot answer this about itself
python -m boundless100x corpus restore          # Undo a refetch that went wrong

# ── Forward-growth extraction (costs money) ──
python -m boundless100x sweep --all --dry-run   # Price it without calling the API
python -m boundless100x sweep --tickers A,B     # Extract a named list
python -m boundless100x sweep --all --ceiling 1 # Stop at a USD ceiling, naming
                                                # what was not reached

python -m pytest tests/                         # Unit tests (the live-network Screener
                                                # test is deselected by pytest.ini;
                                                # run it with `-m network`)
```

**Environment note**: the checked-in `venv/` works (Python 3.11.15). Run
everything through `venv/bin/python`; the suite is green
(`venv/bin/python -m pytest tests/`). Don't hardcode the test count here —
it drifts every time a test is added and nobody remembers to update it.

## GitHub
- **Repo**: https://github.com/rishisareen/boundless100x (private)
- **Branch strategy**: work directly on `main` (owner-confirmed 2026-08-06). Commit each unit as it lands with its tests green, and push when a phase completes. No feature branches — the earlier `claude/` prefix convention is retired, and its stale branches were deleted once their content was confirmed superseded on `main`.
