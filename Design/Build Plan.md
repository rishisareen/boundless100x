## Build Plan

The design document is thorough and well-structured. I'll follow the 6-phase roadmap it defines, building bottom-up so each layer is testable before the next depends on it.

### Phase 1: Foundation — Data Fetching

1. Project scaffolding (`config.yaml`, `requirements.txt`, package structure)
2. `data_fetcher/base.py` — BaseFetcher with retry, rate limiting, session management
3. `data_fetcher/cache/cache_manager.py` — TTL-based local file cache
4. `fetch_financials.py` — Screener.in scraper for 10yr P&L, Balance Sheet, Cash Flow
5. `fetch_price_volume.py` — NSE OHLCV via jugaad-data
6. `fetch_shareholding.py` — BSE quarterly shareholding patterns
7. `fetch_sector_peers.py` — Screener.in sector page scraping
8. Remaining fetchers (corporate actions, analyst coverage, annual report download)
9. Validate by fetching data for Astral, Bajaj Finance, TCS

### Phase 2: Compute Engine

1. `metrics/base.py` — MetricResult dataclass
2. `metrics/validator.py` — YAML schema validation
3. `metrics/registry.yaml` + all `elements/*.yaml` files (size, quality_business, quality_management, growth, longevity, price, composite)
4. `engine.py` — Auto-discovery metric runner
5. All `builtin/*.py` modules (profitability, growth, valuation, leverage, efficiency, size, longevity, composite)
6. `scorer.py` — SQGLP weighted scoring
7. `peer_comparison.py` — Run engine across target + peers

### Phase 3: Peer Discovery

1. Layer 1: Sector peers from Screener.in
2. Layer 2: Size filtering (market cap/revenue bands)
3. Layer 3: Financial similarity (z-score euclidean distance)
4. `peer_discovery.py` orchestrating all layers

### Phase 4: LLM Integration

1. Prompt templates (pass1_qualitative, pass2_synthesis, pass3_comparative)
2. `checklist.py` — QGLP Q1-Q25 mapper
3. `orchestrator.py` — 3-pass LLM orchestration with JSON parsing
4. Annual report PDF text extraction via PyMuPDF
5. `service.py` — Central API orchestrating all stages

### Phase 5: Reporting & CLI

1. Plotly chart functions (radar, P/E band, RoCE trend, growth decomposition)
2. `sqglp_report.html.j2` — Interactive HTML dashboard template
3. `sqglp_report.md.j2` — Markdown report template
4. `report_generator.py`
5. `cli.py` with typer (analyze, peers, compute, screen, watchlist)

### Phase 6: Screening & Watchlist

1. Universe screener using registry-defined filters + presets
2. Watchlist CRUD + quarterly update logic

------

### Key Considerations

- **Screener.in scraping** is the riskiest part — their HTML structure may change and they may rate-limit aggressively. I'll build robust selectors with fallback logic.
- **Each phase produces testable output** — Phase 1 produces CSVs you can inspect, Phase 2 produces JSON metrics, etc.
- **The metric registry pattern** is the architectural centerpiece. Getting this right early means adding metrics later is trivial.