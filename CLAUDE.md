# Boundless100x — SQGLP Financial Research System

## Project Overview
Deep company analysis system for long-term investment in Indian markets, using the SQGLP framework (Size, Quality, Growth, Longevity, Price). Computes financial metrics locally, uses Claude API only for qualitative analysis.

## Design Document
`Design/Financial Model v03.md` — The complete system design. Always reference this for architecture, metric definitions, data flows, and implementation details.

## Architecture
4-stage pipeline + service layer:
- **Stage 1**: Data fetch (Screener.in, NSE, BSE, Trendlyne) → `data_fetcher/`
- **Stage 1.5**: Peer discovery (sector, size, financial similarity) → `data_fetcher/peer_discovery.py`
- **Stage 2**: Compute engine with YAML-driven metric registry → `compute_engine/`
- **Stage 3**: LLM analysis (3-pass: qualitative, synthesis, comparative) → `llm_layer/`
- **Stage 4**: Report generation (HTML/Plotly + Markdown) → `output/`
- **Service layer**: `service.py` orchestrates everything; CLI and future GUI call it

## Project Structure
```
boundless100x/
├── config.yaml
├── service.py                    # Central API (GUI-ready)
├── cli.py                        # typer CLI
├── data_fetcher/
│   ├── base.py                   # BaseFetcher (retry, rate limit, cache)
│   ├── cache/cache_manager.py
│   ├── fetch_financials.py       # Screener.in scraper
│   ├── fetch_price_volume.py     # NSE via jugaad-data
│   ├── fetch_shareholding.py     # BSE quarterly
│   ├── fetch_corporate_actions.py
│   ├── fetch_analyst_coverage.py
│   ├── fetch_sector_peers.py
│   ├── download_annual_reports.py
│   ├── peer_discovery.py
│   └── sector_context.yaml
├── compute_engine/
│   ├── engine.py                 # Auto-discovery metric runner
│   ├── scorer.py                 # SQGLP weighted scoring
│   ├── peer_comparison.py
│   └── metrics/
│       ├── registry.yaml         # Element weights only
│       ├── base.py               # MetricResult dataclass
│       ├── validator.py
│       ├── elements/             # Per-SQGLP element YAML definitions
│       ├── custom/               # User drop-in metrics
│       ├── presets/              # Screening presets
│       └── builtin/              # Python metric implementations
├── llm_layer/
│   ├── orchestrator.py
│   ├── checklist.py
│   └── prompts/
└── output/
    ├── report_generator.py
    └── templates/
```

## Key Patterns
- **Metric registry**: YAML defines metrics in `elements/*.yaml`, Python functions in `builtin/*.py`. Engine auto-discovers both. Adding a metric = 1 YAML entry + 1 function.
- **MetricResult**: Every compute function returns `MetricResult(value, raw_series, flags, metadata, error)`.
- **Scoring**: Threshold-based (higher/lower_is_better), range_optimal, categorical, sector_relative_percentile, trend_direction modes. All defined in YAML.
- **Data contract**: Fetchers write to `raw_data/{TICKER}/` in standardized CSV/JSON. Compute engine reads from there.

## Tech Stack
- Python 3.11+
- Data: requests, beautifulsoup4, jugaad-data, nsetools, bsedata, pandas, numpy, scipy
- PDF: PyMuPDF (fitz)
- Viz: Plotly, Jinja2
- LLM: anthropic SDK (Sonnet for Pass 1-2, Haiku for Pass 3)
- CLI: typer
- Config: PyYAML

## Conventions
- All financial data is in INR Crores unless noted
- 10-year analysis window for financials, 5-year for most averages
- Scoring scale: 1-10 per element, 0-10 weighted composite
- LLM outputs are strict JSON with defined schemas
- Cache with TTL to avoid repeated scraping (2s rate limit between requests)
- Test with: Astral, Bajaj Finance, TCS as reference companies

## Commands
```bash
python cli.py analyze ASTRAL                    # Full pipeline
python cli.py analyze ASTRAL --no-llm           # Compute only
python cli.py peers ASTRAL                      # Peer discovery only
python cli.py compute ASTRAL                    # Metrics only
python cli.py screen --preset compounders       # Screen universe
python cli.py watchlist show                    # View watchlist
```
