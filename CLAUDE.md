# Boundless100x — SQGLP Financial Research System

## Project Overview
Deep company analysis system for long-term investment in Indian markets, using the SQGLP framework (Size, Quality, Growth, Longevity, Price). Computes 44 financial metrics locally, uses Claude API for qualitative analysis and peer validation.

## Design Document
`Design/Financial Model v03.md` — The complete system design. Always reference this for architecture, metric definitions, data flows, and implementation details.

## Architecture
4-stage pipeline + service layer:
- **Stage 1**: Data fetch (Screener.in, yfinance, BSE, Trendlyne) → `data_fetcher/`
- **Stage 1.5**: Peer discovery (4-layer: sector → size → financial similarity → LLM validation) → `data_fetcher/peer_discovery.py`
- **Stage 2**: Compute engine with YAML-driven metric registry (44 metrics) → `compute_engine/`
- **Stage 3**: LLM analysis (3-pass: qualitative, synthesis, comparative) → `llm_layer/`
- **Stage 4**: Report generation (HTML/Plotly + Markdown) → `output/`
- **Service layer**: `service.py` orchestrates everything; CLI and future GUI call it

### Peer Discovery (4 Layers)
1. **Layer 1 — Sector**: Screener.in sector classification → candidate list
2. **Layer 2 — Size**: Filter to 0.2x–5x market cap range
3. **Layer 3 — Financial Similarity**: RoCE, OPM, D/E cosine similarity scoring
4. **Layer 4 — LLM Validation**: Haiku classifies peers as true_competitor/tangential/irrelevant, suggests cross-sector alternatives. Enabled via `config.yaml` → `peer_discovery.use_llm_validation`

### LLM 3-Pass Analysis
- **Pass 1** (Sonnet/Opus): Qualitative analysis from annual report text
- **Pass 2** (Sonnet/Opus): Investment thesis synthesis combining metrics + qualitative
- **Pass 3** (Sonnet/Opus): Comparative analysis vs peers, with peer quality context from Layer 4
- `--deep` flag uses Opus instead of Sonnet for all passes

## Project Structure
```
boundless100x/
├── __init__.py
├── __main__.py
├── config.yaml                   # All pipeline settings (cache TTL, LLM models, peer config)
├── service.py                    # Central API (GUI-ready) — orchestrates full pipeline
├── cli.py                        # typer CLI (analyze, compute, peers, screen, watchlist)
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
│   ├── fetch_sector_peers.py     # Screener.in sector peers
│   ├── download_annual_reports.py # BSE annual report PDF → text extraction
│   ├── peer_discovery.py         # 4-layer peer discovery (incl. LLM Layer 4)
│   ├── sector_context.yaml       # Sector-specific metric thresholds
│   └── raw_data/{TICKER}/        # Fetched data (JSON/CSV/TXT per ticker)
├── compute_engine/
│   ├── engine.py                 # Auto-discovery metric runner
│   ├── scorer.py                 # SQGLP weighted scoring (threshold, trend, range, percentile)
│   ├── peer_comparison.py        # Cross-peer metric comparison
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
│   ├── orchestrator.py           # 3-pass LLM pipeline with JSON parsing
│   ├── checklist.py              # Pre-flight data quality checks
│   └── prompts/
│       ├── pass1_qualitative.txt # Annual report deep dive
│       ├── pass2_synthesis.txt   # Investment thesis
│       ├── pass3_comparative.txt # Peer comparison (with quality context)
│       └── peer_validation.txt   # Layer 4 peer validation prompt
└── output/
    ├── report_generator.py       # HTML dashboard + Markdown report
    ├── templates/
    │   ├── sqglp_report.html.j2
    │   └── sqglp_report.md.j2
    └── reports/{TICKER}_{DATE}/  # Generated reports (HTML, JSON)
```

## Key Patterns
- **Metric registry**: YAML defines metrics in `elements/*.yaml`, Python functions in `builtin/*.py`. Engine auto-discovers both. Adding a metric = 1 YAML entry + 1 function.
- **MetricResult**: Every compute function returns `MetricResult(value, raw_series, flags, metadata, error)`. Flags communicate data quality issues (e.g., `insufficient_history`, `possible_bonus_split`, `cfi_dominated_by_acquisitions`).
- **Scoring**: Threshold-based (higher/lower_is_better), range_optimal, categorical, sector_relative_percentile, trend_direction modes. All defined in YAML. Scorer receives full MetricResult for trend analysis.
- **Data contract**: Fetchers write to `raw_data/{TICKER}/` in standardized CSV/JSON. Compute engine reads from there. BSE codes auto-detected from Screener.in metadata.
- **FCF outlier detection**: MAD-based (Median Absolute Deviation) to identify M&A-dominated years. Applied in valuation.py, longevity.py, profitability.py via `_helpers.py`.
- **Bonus/split detection**: YoY equity capital spikes >50% flagged as structural events. Organic dilution computed separately in growth.py.
- **LLM prompt templates**: Use `.format()` with quadruple-braces `{{{{` for JSON schema escaping in prompt files.

## Tech Stack
- Python 3.11+
- Data: requests, beautifulsoup4, yfinance, pandas, numpy, scipy
- PDF: PyMuPDF (fitz)
- Viz: Plotly, Jinja2
- LLM: anthropic SDK (Sonnet/Opus for Pass 1-3, Haiku for peer validation)
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

## Commands
```bash
python cli.py analyze ASTRAL                    # Full pipeline (fetch + compute + LLM + report)
python cli.py analyze ASTRAL --no-llm           # Compute only (no LLM passes)
python cli.py analyze ASTRAL --deep             # Use Opus instead of Sonnet for LLM
python cli.py peers ASTRAL                      # Peer discovery only (all 4 layers)
python cli.py compute ASTRAL                    # Metrics only (no fetch, no LLM)
python cli.py screen --preset compounders       # Screen universe against preset
python cli.py watchlist show                    # View watchlist
python cli.py watchlist add ASTRAL              # Add ticker to watchlist
```

## GitHub
- **Repo**: https://github.com/rishisareen/boundless100x (private)
- **Branch strategy**: `main` is default; feature branches via `claude/` prefix
