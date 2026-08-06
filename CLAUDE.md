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
- **Stage 4**: LLM analysis (2-pass: qualitative, synthesis) → `llm_layer/`
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
├── cli.py                        # typer CLI (analyze, compute, screen, watchlist)
├── watchlist.py                  # Watchlist management
├── watchlist.json                # Persisted watchlist data
├── data_fetcher/
│   ├── base.py                   # BaseFetcher (retry, rate limit, cache)
│   ├── suite.py                  # FetcherSuite — runs all fetchers for a ticker
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
│       │   ├── growth.yaml
│       │   ├── longevity.yaml
│       │   ├── price.yaml
│       │   └── composite.yaml
│       ├── builtin/              # Python metric implementations
│       │   ├── _helpers.py       # Shared utilities (MAD-based FCF outlier detection)
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
├── llm_layer/
│   ├── orchestrator.py           # 2-pass LLM pipeline with JSON parsing
│   ├── checklist.py              # Pre-flight data quality checks
│   └── prompts/
│       ├── pass1_qualitative.txt # Annual report deep dive
│       └── pass2_synthesis.txt   # Investment thesis
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
- **Macro assumptions** (inflation, G-Sec yield, discount rate, terminal growth) live in `config.yaml` under `macro:` and reach metrics as parameter defaults; a metric's own YAML params override them.
- **Lifecycle** (`lifecycle/`, Phase 1 of the v05 roadmap): the layer *after* the verdict. States are `screen → qualify → watch → probe → scale`, with `exit_review → exited` and `dropped`; `states.py` is the definition. Transitions are declared in `lifecycle/triggers.yaml` and evaluated by `TriggerEvaluator`, which **mirrors `EligibilityEvaluator`** — same imported `COMPARATORS`, same three-valued outcome, same per-condition `detail` strings, same "indeterminate, never a silent pass" rule. Registry validation runs at construction: unknown states, comparators, and metric ids are startup errors, because a trigger naming a nonexistent metric would read indeterminate forever and a kill-switch that never fires looks exactly like a thesis that never broke.
  - **`persist_years` is allowlisted** (`SERIES_SAFE_METRICS`). `raw_series` has no declared contract — `roiic` returns *capital employed* beside a percentage value, `pe_vs_historical` returns P/E multiples beside a percentile — so a consecutive-year rule on either would compare incompatible units and silently never fire. Adding a metric to the allowlist means reading its implementation first.
  - **Checkpoints** (`lifecycle/checkpoints.py`, vocabulary in `checkpoint_vocabulary.yaml`) are the machine-checkable half of Pass 2's monitorables; the prose half is unchanged. The vocabulary is closed to quarterly-readable series (Phase 0's `quarterly.csv` columns plus quarterly shareholding) so a checkpoint can always come due; an id outside it is demoted to prose at recording time. A data gap is `indeterminate`, never `missed` — and zero misses out of zero *due* checkpoints is indeterminate too, so an unmonitored position never reads like a verified one.
  - **`watchlist advance`** re-scores, evaluates, and proposes. Transitions that move money (`probe`, `scale`, `exit_review`, `exited`) are proposed and wait for `--apply`; pre-position transitions (`qualify`, `watch`, `dropped`) auto-apply. When several triggers fire, the most protective wins — a kill-switch outranks a buy-zone, so a company never gets bought into on the quarter its thesis broke.
- **Annual report sections** (`download_annual_reports.py`): extraction is section-targeted, not first-N-pages. `extract_sections()` returns `{section: {text, provenance, start_page}}` for `mdna`/`chairman`/`governance` (caps in `config.yaml` under `annual_reports.sections`), cached in a `{year}_annual_report.sections.json` sidecar; `download_and_extract()` returns `{year: {...}}` across the retained reports (`max_reports: 3`, so promises-kept can compare guidance to delivery). **`provenance` is the contract**: `found` means the section was located, `fallback` means the slot holds first-N-pages text instead — Phase 2 sub-metrics must evaluate indeterminate on `fallback` rather than mine a chairman's letter for guidance (v05 §7.2). Detection is heuristic heading-matching over `scan_pages: 150` (MD&A starts at pages 20–147 in the fetched corpus, so the 30-page fallback window is far too shallow to search); two adversaries, both real. The **contents page** lists every section name in heading form, and is rejected by three guards (Contents/Index title, bulk page-number entries, ≥2 distinct section names on one page). The **cross-reference** is subtler and was the cause of an 8-of-18 wrong-section rate before it was fixed: a report saying "provided in the Management Discussion and Analysis" scored as a heading under the original short-line-mostly-the-name test, so slices opened on auditor's reports, governance and CSR. `_is_heading_like` now requires the match to *open* its line (leading numbering/bullets stripped) and to not be continued by a lowercase word — position and continuation separate the two, where coverage ratio cannot. Detection also records the heading's **line**, not just its page: a heading low on a page (one real report has it at line 40 of 62) otherwise drags the preceding section's tail into the slice. Measured across 29 report-years: mdna 13 found, chairman 11, governance 26 — mdna precision ~85% (up from 56%), so a residual wrong-section rate remains and consumers must still treat provenance as a claim rather than a guarantee. `combined_text()` builds the single-string view older consumers read — found sections in page order, or, when none were found, the fallback text once, which keeps `annual_report_text` byte-identical to pre-section behaviour. Pass 1's own cap is `llm.pass1_ar_char_budget` (default 3000): it sits downstream of the per-section caps and is the binding limit on what reaches the prompt, so raising a section cap without raising this one changes nothing.
- **Score history** (`score_history.py` → `boundless100x/score_history.jsonl`): every scored `service.analyze()` run appends one row (`schema_version, ticker, date, composite, elements, verdict, coverage, flags, config_hash, synthetic`) at Stage 4.6. Git-tracked and **append-only by contract** — a score not written when the run happened cannot be recovered, so nothing ever rewrites a line. Same-day re-runs append duplicates; `load_history` resolves them at read time (last row wins per ticker/date/`config_hash`), while rows under different hashes are both kept because they are different scoring regimes. A run whose scoring failed appends nothing; a write failure lands in `result.errors` and never costs the caller the analysis. Tests never touch the real file — an autouse `conftest` fixture redirects the module default, and `service.history_path` (config `output.score_history_path`) redirects per caller. The backtest bypasses `service.analyze()`, so truncated-history scores never enter the organic log; synthetic backfill rows (v05 §7.1, Phase 2) must set `synthetic: true`.
- **Registry hash** (`engine.registry_hash`): a 12-char sha256 over everything that can move a score — the whole `registry.yaml` (element weights, declared gates, `history_waiver_mcap`), the *effective* gates, the metric definitions, and the macro block. It stamps the scoring regime onto score-history rows so a later trajectory diff cannot mistake a threshold edit for fundamental momentum. It hashes the loaded registry rather than YAML bytes, so custom drop-ins count and reformatting does not; `_`-prefixed provenance keys are excluded so a file rename does not fragment history. `effective_gates()` in `eligibility.py` is the single statement of the "no declared gates falls back to `DEFAULT_GATES`" rule — both the hash and the service's evaluator resolve through it, so the regime recorded always equals the regime enforced.
- **MetricResult**: Every compute function returns `MetricResult(value, raw_series, flags, metadata, error)`. Flags communicate data quality issues (e.g., `insufficient_history`, `possible_bonus_split`, `cfi_dominated_by_acquisitions`).
- **Scoring**: Threshold-based (higher/lower_is_better), range_optimal, categorical, sector_relative_percentile, trend_direction modes. All defined in YAML. Scorer receives full MetricResult for trend analysis.
- **Data contract**: Fetchers write to `raw_data/{TICKER}/` in standardized CSV/JSON. Compute engine reads from there. BSE codes auto-detected from Screener.in metadata. `quarterly.csv` (Screener's quarterly results, `quarter` period column) is parsed from the same cached company page as the annual tables and shares `_parse_table` with them; it is plumbed to `data["quarterly"]` but not yet consumed by any metric — the v05 lifecycle's checkpoint grain lands in Phase 1+. Screener renders only ~11–13 recent quarters, which is enough for consecutive-quarter checks but not for deep historical replay.
- **Screener page cache**: The company page HTML is cached via the TTL cache (`txt` entries), so repeat runs within the window do not re-scrape Screener. Parsing stays deterministic on the cached HTML.
- **Price series**: `price_volume.csv` carries both `close` (raw traded) and `adj_close` (split/dividend-adjusted). Valuation metrics use the raw close against as-reported EPS and record `price_basis` in metadata; the backtest's realized return prefers `adj_close`. When the fetch source has no real adjusted series (jugaad-data fallback), `adj_close` is aliased to `close` and `adj_close_is_estimated=True` marks the alias — the backtest refuses to score a realized return off it rather than risk reading a split as a crash. Files fetched before Aug 2026 hold a single legacy close with no alias flag at all — refetch to upgrade.
- **Growth quality**: `_grade_growth_quality` in `builtin/growth.py` is the single grader for both the scored `growth_quality_grade` metric and the report/LLM lever table. YoY leverage ratios share one helper, `_mean_yoy_ratio`.
- **FCF outlier detection**: MAD-based (Median Absolute Deviation) to identify M&A-dominated years. Applied in valuation.py, longevity.py, profitability.py via `_helpers.py`.
- **Bonus/split detection**: YoY equity capital spikes >50% flagged as structural events. Organic dilution computed separately in growth.py.
- **LLM prompt templates**: Use `.format()` with quadruple-braces `{{{{` for JSON schema escaping in prompt files.

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
python -m boundless100x watchlist advance       # Re-score, evaluate triggers,
                                                # propose transitions with evidence.
                                                # Money-moving ones are proposals only.
python -m boundless100x watchlist advance --apply   # Confirm and record them

python -m boundless100x backtest                # Walk-forward self-check: score on
                                                # the first half of each cached ticker's
                                                # history, compare to realized returns

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
