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
- **Macro assumptions** (inflation, G-Sec yield, discount rate, terminal growth) live in `config.yaml` under `macro:` and reach metrics as parameter defaults; a metric's own YAML params override them.
- **MetricResult**: Every compute function returns `MetricResult(value, raw_series, flags, metadata, error)`. Flags communicate data quality issues (e.g., `insufficient_history`, `possible_bonus_split`, `cfi_dominated_by_acquisitions`).
- **Scoring**: Threshold-based (higher/lower_is_better), range_optimal, categorical, sector_relative_percentile, trend_direction modes. All defined in YAML. Scorer receives full MetricResult for trend analysis.
- **Data contract**: Fetchers write to `raw_data/{TICKER}/` in standardized CSV/JSON. Compute engine reads from there. BSE codes auto-detected from Screener.in metadata.
- **Screener page cache**: The company page HTML is cached via the TTL cache (`txt` entries), so repeat runs within the window do not re-scrape Screener. Parsing stays deterministic on the cached HTML.
- **Price series**: `price_volume.csv` carries both `close` (raw traded) and `adj_close` (split/dividend-adjusted). Valuation metrics use the raw close against as-reported EPS and record `price_basis` in metadata; the backtest's realized return prefers `adj_close`. Files fetched before Aug 2026 hold a single legacy close — refetch to upgrade.
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
python -m boundless100x watchlist show          # View watchlist
python -m boundless100x watchlist add ASTRAL    # Add ticker to watchlist

python -m boundless100x backtest                # Walk-forward self-check: score on
                                                # the first half of each cached ticker's
                                                # history, compare to realized returns

python -m pytest tests/                         # Unit tests (the live-network Screener
                                                # test is deselected by pytest.ini;
                                                # run it with `-m network`)
```

**Environment note**: the checked-in `venv/` works (Python 3.11.15). Run
everything through `venv/bin/python`; the suite is 185 tests green
(`venv/bin/python -m pytest tests/`).

## GitHub
- **Repo**: https://github.com/rishisareen/boundless100x (private)
- **Branch strategy**: `main` is default; feature branches via `claude/` prefix
