# SQGLP Financial Research System — Complete Design Document

## Deep Company Analysis for Long-Term Investment (Indian Markets)

**Version:** 2.0 — Final Consolidated Design **Framework:** SQGLP (Size, Quality, Growth, Longevity, Price) **Reference:** Motilal Oswal Wealth Creation Study + 100 Baggers (Christopher Mayer)

------

## Table of Contents

1. [System Architecture Overview](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#1-system-architecture-overview)
2. [Data Sources & Fetching (Stage 1)](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#2-data-sources--fetching-stage-1)
3. [Automated Peer Discovery (Stage 1.5)](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#3-automated-peer-discovery-stage-15)
4. [Extensible Compute Engine (Stage 2)](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#4-extensible-compute-engine-stage-2)
5. [SQGLP Metrics — Detailed Specification](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#5-sqglp-metrics--detailed-specification)
6. [LLM Analysis Layer (Stage 3)](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#6-llm-analysis-layer-stage-3)
7. [Output & Report Generation (Stage 4)](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#7-output--report-generation-stage-4)
8. [Service Layer — The GUI-Ready API](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#8-service-layer--the-gui-ready-api)
9. [Project Structure](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#9-project-structure)
10. [Configuration Reference](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#10-configuration-reference)
11. [Technology Stack](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#11-technology-stack)
12. [CLI Reference](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#12-cli-reference)
13. [Implementation Roadmap](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#13-implementation-roadmap)
14. [Key Design Decisions & Rationale](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#14-key-design-decisions--rationale)
15. [Future Work: Interactive GUI](https://claude.ai/chat/01838701-fcfc-4fbf-b3a0-c7a0dca120d1#15-future-work-interactive-gui)

------

## 1. System Architecture Overview

### 1.1 Core Principle

**Compute locally, analyze with LLM.** All number-crunching, data fetching, ratio calculations, peer discovery, and scoring happen in offline Python scripts at zero cost. The LLM is invoked only for qualitative judgment, pattern recognition, and final synthesis — minimizing paid API token usage by ~90%.

### 1.2 Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RESEARCH PIPELINE                                  │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────┐ │
│  │ STAGE 1  │─▶│STAGE 1.5 │─▶│   STAGE 2    │─▶│  STAGE 3  │─▶│ STAGE 4  │ │
│  │ Data     │  │ Peer     │  │  Compute     │  │  LLM      │  │ Output   │ │
│  │ Fetch    │  │ Discovery│  │  Engine      │  │  Analysis  │  │ Reports  │ │
│  └──────────┘  └──────────┘  └──────────────┘  └───────────┘  └──────────┘ │
│   Python        Python        Python            Claude API     HTML/MD/     │
│   Scripts       Scripts       Scripts           (Paid calls)   JSON         │
│   FREE          FREE          FREE              OPTIMIZED      FREE         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  SERVICE LAYER (service.py) — Orchestrates all stages                   │ │
│  │  Called by: CLI today, Streamlit/FastAPI tomorrow                        │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow

```
                    ┌─────────────┐
                    │  config.yaml │
                    └──────┬──────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1: DATA FETCH                            │
│  Screener.in → financials.csv                   │
│  NSE/BSE     → price_volume.csv, shareholding   │
│  Trendlyne   → analyst_coverage                 │
│  BSE Filings → annual_reports/*.pdf             │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1.5: PEER DISCOVERY                      │
│  Screener.in peers → sector_peers               │
│  Size filter       → size_matched_peers         │
│  Financial dist.   → financial_similarity_peers  │
│  (Optional LLM)    → validated_peers            │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: COMPUTE ENGINE                        │
│  registry.yaml defines all metrics              │
│  engine.py runs each metric function            │
│  scorer.py computes weighted SQGLP scores       │
│  peer_comparison.py runs engine on all peers    │
│                                                 │
│  Output: sqglp_metrics.json                     │
│          peer_comparison.json                   │
│          computed_flags[]                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3: LLM ANALYSIS (3 focused passes)       │
│  Pass 1: Qualitative (mgmt, moat, risks)        │
│  Pass 2: Synthesis (thesis, conviction, action)  │
│  Pass 3: Comparative (peer ranking, edges)       │
│                                                  │
│  Output: llm_analysis.json                       │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STAGE 4: REPORT GENERATION                     │
│  Jinja2 + Plotly → sqglp_dashboard.html         │
│  Jinja2          → sqglp_report.md              │
│  Raw data        → raw_metrics.json             │
└─────────────────────────────────────────────────┘
```

------

## 2. Data Sources & Fetching (Stage 1)

### 2.1 Primary Data Sources

| Data Type                              | Source                        | Method         | Notes                                          |
| -------------------------------------- | ----------------------------- | -------------- | ---------------------------------------------- |
| **Financial Statements** (P&L, BS, CF) | Screener.in                   | Web scrape     | 10-year data, already normalized, consolidated |
| **Shareholding Patterns**              | BSE India / Trendlyne         | API / Scrape   | FII, DII, Promoter — quarterly                 |
| **Daily Price & Volume**               | NSE (via `jugaad-data`)       | Python library | OHLCV, splits/bonus adjusted                   |
| **Corporate Actions**                  | BSE / Moneycontrol            | Scrape         | Splits, bonuses, dividends                     |
| **Sector Peer Lists**                  | Screener.in sector pages      | Scrape         | For auto peer discovery                        |
| **Annual Reports / Con-calls**         | BSE filings / company website | PDF download   | For LLM qualitative analysis                   |
| **Analyst Coverage Count**             | Trendlyne / Tickertape        | Scrape         | For "unknown-ness" metric                      |

### 2.2 Python Libraries

```
jugaad-data        — NSE historical price/volume (free, no API key needed)
nsetools           — Live NSE quotes and metadata
bsedata            — BSE corporate data and filings
requests + bs4     — Web scraping (Screener.in, Trendlyne, Moneycontrol)
pandas             — All data manipulation and computation
numpy              — Numerical operations (z-scores, distances)
yfinance           — Fallback for price data (append .NS to ticker)
PyMuPDF (fitz)     — PDF text extraction from annual reports
scipy              — Euclidean distance for peer financial similarity
```

### 2.3 Data Fetch Module Structure

```
data_fetcher/
├── __init__.py
├── base.py                      # BaseFetcher class, retry logic, rate limiting
├── fetch_financials.py          # Screener.in → 10yr P&L, BS, CF
├── fetch_shareholding.py        # BSE → quarterly shareholding patterns
├── fetch_price_volume.py        # NSE → daily OHLCV via jugaad-data
├── fetch_corporate_actions.py   # BSE → splits, bonuses, dividends
├── fetch_analyst_coverage.py    # Trendlyne → analyst count, targets
├── fetch_sector_peers.py        # Screener.in → sector company list with key metrics
├── download_annual_reports.py   # BSE filings → PDF
└── cache/                       # Local cache with TTL (avoid repeated scraping)
    ├── cache_manager.py
    └── cached_data/
```

### 2.4 Data Fetcher Design Principles

**Rate Limiting:** All scrapers include configurable delays (default 2s between requests) to avoid being blocked. Configurable per source in `config.yaml`.

**Caching:** Fetched data is cached locally with a TTL. Financial statements refresh monthly, shareholding quarterly, price data daily. The cache avoids redundant scraping when re-running analysis.

```python
# base.py — All fetchers inherit from this

class BaseFetcher:
    def __init__(self, cache_ttl_hours: int = 24):
        self.cache = CacheManager(ttl_hours=cache_ttl_hours)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (research tool)"
        })

    def fetch_with_cache(self, key: str, fetch_fn: callable) -> pd.DataFrame:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        data = fetch_fn()
        self.cache.set(key, data)
        return data

    def _rate_limit(self):
        time.sleep(self.config.get("rate_limit_seconds", 2))
```

**Output Format:** Every fetcher writes to a standardized CSV/JSON format in `raw_data/{TICKER}/`. Column names are normalized across sources.

```
raw_data/
└── ASTRAL/
    ├── financials.csv          # Columns: year, revenue, ebitda, ebit, pat, eps, ...
    ├── balance_sheet.csv       # Columns: year, total_assets, equity, debt, ...
    ├── cashflow.csv            # Columns: year, cfo, capex, fcf, ...
    ├── shareholding.csv        # Columns: quarter, promoter_pct, fii_pct, dii_pct, ...
    ├── price_volume.csv        # Columns: date, open, high, low, close, volume
    ├── corporate_actions.csv   # Columns: date, type, details
    ├── analyst_coverage.json   # {count, names, avg_target, ...}
    └── annual_reports/
        ├── 2024_annual_report.pdf
        └── 2023_annual_report.pdf
```

------

## 3. Automated Peer Discovery (Stage 1.5)

### 3.1 Why Automate Peer Discovery

"Competitors" is not a single concept. For Astral Ltd (pipes & fittings):

- **Direct product competitors:** Supreme Industries, Prince Pipes, Finolex Industries — make similar pipes
- **Sector peers by classification:** All BSE "Building Materials" companies — useful for valuation benchmarking
- **Financial profile peers:** Companies with similar RoCE, margin, growth — could be from different sectors but answer "what P/E does the market assign to this quality level?"
- **Value chain adjacents:** Pidilite (adhesives), APL Apollo (steel tubes) — affected by same housing cycle

The system auto-discovers at least the first three categories, with optional LLM-assisted value chain mapping.

### 3.2 Multi-Layer Peer Discovery Pipeline

```
┌──────────────────────────────────────────────────────────┐
│              PEER DISCOVERY PIPELINE                      │
│                                                          │
│  Layer 1: Industry Classification (FREE, deterministic)  │
│  ├── Screener.in "Peer Comparison" table (5-10 peers)   │
│  ├── Screener.in sector page (all companies in sector)  │
│  ├── Fallback: BSE industry group classification        │
│  └── Output: 15-40 raw sector peers                     │
│                                                          │
│  Layer 2: Size-Filtered Cohort (FREE, computed)          │
│  ├── Filter Layer 1 by market cap band (0.3x to 3x)    │
│  ├── Filter by revenue band (0.2x to 5x)               │
│  ├── Remove companies with < 5yr listing history        │
│  └── Output: 5-15 size-matched peers                    │
│                                                          │
│  Layer 3: Financial Similarity (FREE, computed)          │
│  ├── Compute z-score normalized vector for each company │
│  │   Dimensions: [RoCE, OPM, Rev CAGR, D/E, log MCap] │
│  ├── Euclidean distance from target to each candidate   │
│  ├── Rank by closest distance                           │
│  └── Output: Top 5 most financially similar companies   │
│                                                          │
│  Layer 4: LLM Peer Validation (OPTIONAL, ~$0.02)        │
│  ├── Input: Layer 2+3 candidates + business description │
│  ├── Task: "Which are true competitors vs tangential?"  │
│  └── Output: Final 4-6 validated direct competitors     │
│                                                          │
│  Layer 5: Value Chain Mapping (OPTIONAL, ~$0.02)         │
│  ├── Input: Company business description                │
│  ├── Task: "Who are upstream/downstream/adjacent?"      │
│  └── Output: Adjacent companies to monitor              │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Implementation

```python
# data_fetcher/peer_discovery.py

from dataclasses import dataclass

@dataclass
class PeerResult:
    direct_competitors: list[str]     # Layer 2+3 (or Layer 4 if LLM used)
    sector_peers: list[str]           # Layer 1: full sector list
    financial_peers: list[str]        # Layer 3: cross-sector similarity
    value_chain: list[str]            # Layer 5: upstream/downstream (if LLM used)
    discovery_metadata: dict          # Similarity scores, filtering stats

class PeerDiscovery:
    """
    Multi-layer peer identification.
    Layers 1-3: fully offline, zero LLM cost.
    Layers 4-5: optional LLM refinement.
    """

    def discover(self, ticker: str, use_llm: bool = False) -> PeerResult:
        # Layer 1: Sector classification from Screener.in
        sector_peers = self._get_sector_peers(ticker)

        # Layer 2: Size filtering
        size_filtered = self._filter_by_size(ticker, sector_peers)

        # Layer 3: Financial similarity scoring
        similarity_ranked = self._rank_by_financial_similarity(ticker, size_filtered)

        # Layers 4-5 (optional): LLM validation
        if use_llm:
            validated = self._llm_validate_peers(ticker, similarity_ranked)
            value_chain = self._llm_map_value_chain(ticker)
        else:
            validated = similarity_ranked[:5]
            value_chain = []

        return PeerResult(
            direct_competitors=validated,
            sector_peers=sector_peers,
            financial_peers=similarity_ranked[:5],
            value_chain=value_chain,
            discovery_metadata={"candidates_evaluated": len(sector_peers)},
        )

    def _get_sector_peers(self, ticker: str) -> list[str]:
        """
        Scrape Screener.in company page:
          1. Extract "Peer Comparison" table → 5-10 direct peers
          2. Follow "Industry" link → full sector company list with key metrics
        Fallback: BSE industry group, Moneycontrol "Peers" tab.
        """
        ...

    def _filter_by_size(self, ticker: str, candidates: list[str]) -> list[str]:
        """
        Filter candidates by:
          - Market cap within 0.3x to 3x of target
          - Revenue within 0.2x to 5x of target
          - Minimum 5 years listing history
        """
        ...

    def _rank_by_financial_similarity(self, ticker: str, candidates: list[str]) -> list[str]:
        """
        For each candidate, compute normalized euclidean distance from target
        across: [RoCE_5yr, OPM_5yr, RevCAGR_5yr, Debt/Equity, log(MarketCap)]
        Return sorted by closest distance (most similar first).
        """
        SIMILARITY_DIMENSIONS = [
            'roce_5yr_avg', 'operating_margin_5yr_avg',
            'revenue_cagr_5yr', 'debt_equity', 'log_market_cap',
        ]
        # z-score normalize each dimension across all candidates
        # compute euclidean distance from target
        # return sorted by distance ascending
        ...
```

### 3.4 Peer Discovery Output

```json
{
  "target": "ASTRAL",
  "discovery": {
    "direct_competitors": ["SUPREMEIND", "PRINCEPIPE", "FINOLEX", "APLLTD", "ASTEC"],
    "sector_peers": ["SUPREMEIND", "PRINCEPIPE", "FINOLEX", "APLLTD", "...40 more"],
    "financial_peers": ["PIDILITIND", "BERGERPAINTS", "SUPREMEIND"],
    "value_chain": [],
    "metadata": {
      "sector": "Building Materials - Plastic Pipes",
      "candidates_evaluated": 43,
      "size_filtered_to": 12,
      "similarity_scores": {
        "SUPREMEIND": 0.82,
        "PRINCEPIPE": 1.23,
        "FINOLEX": 1.45
      }
    }
  }
}
```

------

## 4. Extensible Compute Engine (Stage 2)

### 4.1 Design Problem & Solution

Adding a new financial ratio in a hardcoded system means editing 4+ files (compute function, JSON schema, scoring logic, report template). The extensible architecture reduces this to **1 YAML entry + 1 Python function**. Everything else auto-discovers.

### 4.2 Architecture: Metric Registry Pattern

```
compute_engine/
├── metrics/
│   ├── registry.yaml          # THE source of truth for all metrics
│   ├── base.py                # MetricResult dataclass + MetricDefinition
│   ├── builtin/               # Shipped metric implementations
│   │   ├── __init__.py
│   │   ├── profitability.py   # RoCE, RoE, OPM, NPM, Cash Conversion
│   │   ├── growth.py          # CAGR, 4-lever decomposition, consistency
│   │   ├── valuation.py       # P/E, PEG, EV/EBITDA, DCF, reverse DCF
│   │   ├── leverage.py        # D/E, Interest Coverage, Financial Lever
│   │   ├── efficiency.py      # Working Capital Days, Asset Turnover, CCC
│   │   ├── size.py            # Market cap, institutional holding, turnover
│   │   └── longevity.py       # Consistency streaks, margin stability
│   └── custom/                # User drop-in metrics (auto-discovered)
│       ├── __init__.py
│       └── my_metrics.py      # Your custom ratios
├── engine.py                  # Reads registry, dynamically runs all metrics
├── scorer.py                  # Reads registry weights, computes SQGLP scores
└── peer_comparison.py         # Runs engine on target + all peers
```

### 4.3 The Metric Registry (`registry.yaml`)

Every metric in the system is declared here. The engine, scorer, peer comparison, and report generator all read this file — no metric is hardcoded anywhere else.

```yaml
# metrics/registry.yaml — Single source of truth for all metrics
# To add a metric: (1) add entry here, (2) write the function, (3) done.

# ═══════════════════════════════════════════════════════════════
#  ELEMENT WEIGHTS — Controls the composite SQGLP score
# ═══════════════════════════════════════════════════════════════
element_weights:
  size: 0.10
  quality_business: 0.20
  quality_management: 0.10
  growth: 0.25
  longevity: 0.20
  price: 0.15

# ═══════════════════════════════════════════════════════════════
#  METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════
metrics:

  # ─────────────── S: SIZE ───────────────
  market_cap:
    name: "Market Cap (₹ Cr)"
    element: "size"
    module: "builtin.size"
    function: "compute_market_cap"
    inputs: ["price", "financials"]
    scoring:
      thresholds: [200000, 100000, 50000, 20000, 5000, 1000]
      direction: "lower_is_better"     # Smaller = more room to grow
      weight: 0.25
    display:
      format: "₹{:,.0f} Cr"
      section: "size_analysis"
      peer_compare: true

  institutional_holding:
    name: "FII + DII Holding (%)"
    element: "size"
    module: "builtin.size"
    function: "compute_institutional_holding"
    inputs: ["shareholding"]
    scoring:
      # Sweet spot: 1-10%. Too high = already discovered. Zero = no credibility.
      mode: "range_optimal"
      optimal_range: [1, 10]
      weight: 0.25
    display:
      format: "{:.1f}%"
      section: "size_analysis"
      peer_compare: true

  analyst_coverage:
    name: "Analyst Coverage Count"
    element: "size"
    module: "builtin.size"
    function: "compute_analyst_count"
    inputs: ["analyst_coverage"]
    scoring:
      thresholds: [20, 15, 10, 5, 3, 1]
      direction: "lower_is_better"
      weight: 0.25
    display:
      format: "{:.0f} analysts"
      section: "size_analysis"
      peer_compare: false

  daily_turnover_ratio:
    name: "Daily Turnover Ratio (%)"
    element: "size"
    module: "builtin.size"
    function: "compute_turnover_ratio"
    inputs: ["price"]
    scoring:
      mode: "range_optimal"
      optimal_range: [0.02, 0.1]      # Active but quiet
      weight: 0.25
    display:
      format: "{:.3f}%"
      section: "size_analysis"
      peer_compare: false

  # ─────────────── Q: QUALITY OF BUSINESS ───────────────
  roce_5yr_avg:
    name: "RoCE (5yr Avg)"
    element: "quality_business"
    module: "builtin.profitability"
    function: "compute_roce_avg"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      thresholds: [5, 10, 15, 20, 25, 30]
      direction: "higher_is_better"
      weight: 0.20
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
      peer_compare: true

  roe_5yr_avg:
    name: "RoE (5yr Avg)"
    element: "quality_business"
    module: "builtin.profitability"
    function: "compute_roe_avg"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      thresholds: [5, 10, 15, 18, 22, 28]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
      peer_compare: true

  operating_margin_5yr:
    name: "Operating Margin (5yr Avg)"
    element: "quality_business"
    module: "builtin.profitability"
    function: "compute_opm_avg"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      thresholds: [5, 8, 12, 18, 25, 35]
      direction: "higher_is_better"
      weight: 0.12
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
      peer_compare: true

  cash_conversion:
    name: "Cash Conversion (OCF/EBITDA)"
    element: "quality_business"
    module: "builtin.profitability"
    function: "compute_cash_conversion"
    inputs: ["financials", "cashflow"]
    params:
      years: 5
    scoring:
      thresholds: [30, 45, 55, 65, 75, 85]
      direction: "higher_is_better"
      weight: 0.10
    display:
      format: "{:.0f}%"
      section: "quality_scorecard"
      peer_compare: true

  fcf_yield:
    name: "FCF Yield"
    element: "quality_business"
    module: "builtin.profitability"
    function: "compute_fcf_yield"
    inputs: ["cashflow", "price"]
    scoring:
      thresholds: [-2, 0, 1, 2, 4, 6]
      direction: "higher_is_better"
      weight: 0.08
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
      peer_compare: true

  debt_equity:
    name: "Debt / Equity"
    element: "quality_business"
    module: "builtin.leverage"
    function: "compute_debt_equity"
    inputs: ["financials"]
    scoring:
      thresholds: [2.0, 1.0, 0.5, 0.3, 0.1, 0.0]
      direction: "lower_is_better"
      weight: 0.10
    display:
      format: "{:.2f}x"
      section: "quality_scorecard"
      peer_compare: true

  interest_coverage:
    name: "Interest Coverage"
    element: "quality_business"
    module: "builtin.leverage"
    function: "compute_interest_coverage"
    inputs: ["financials"]
    scoring:
      thresholds: [1, 2, 3, 5, 8, 15]
      direction: "higher_is_better"
      weight: 0.08
    display:
      format: "{:.1f}x"
      section: "quality_scorecard"
      peer_compare: true

  working_capital_days_trend:
    name: "Working Capital Days Trend"
    element: "quality_business"
    module: "builtin.efficiency"
    function: "compute_wc_days_trend"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      # Negative slope = improving (fewer days locked up)
      mode: "trend_direction"
      direction: "declining_is_better"
      weight: 0.07
    display:
      format: "{:.0f} days (Δ{:+.0f})"
      section: "quality_scorecard"
      peer_compare: true

  # ─────────────── Q: QUALITY OF MANAGEMENT ───────────────
  promoter_holding_trend:
    name: "Promoter Holding Trend (5yr)"
    element: "quality_management"
    module: "builtin.size"
    function: "compute_promoter_trend"
    inputs: ["shareholding"]
    params:
      quarters: 20
    scoring:
      mode: "trend_direction"
      direction: "stable_or_rising_is_better"
      weight: 0.25
    display:
      format: "{:.1f}% (Δ{:+.1f}pp)"
      section: "management_assessment"
      peer_compare: true

  promoter_pledge:
    name: "Promoter Pledge %"
    element: "quality_management"
    module: "builtin.size"
    function: "compute_promoter_pledge"
    inputs: ["shareholding"]
    scoring:
      thresholds: [50, 30, 20, 10, 5, 0]
      direction: "lower_is_better"
      weight: 0.25
    display:
      format: "{:.1f}%"
      section: "management_assessment"
      peer_compare: true
      red_flag_threshold: 10

  equity_dilution:
    name: "Shares Outstanding Growth (10yr)"
    element: "quality_management"
    module: "builtin.growth"
    function: "compute_share_dilution"
    inputs: ["financials"]
    params:
      years: 10
    scoring:
      thresholds: [50, 30, 20, 10, 5, 0]
      direction: "lower_is_better"
      weight: 0.20
    display:
      format: "{:.1f}% dilution"
      section: "management_assessment"
      peer_compare: true

  dividend_consistency:
    name: "Dividend Payout Consistency"
    element: "quality_management"
    module: "builtin.longevity"
    function: "compute_dividend_consistency"
    inputs: ["financials"]
    params:
      years: 10
    scoring:
      thresholds: [2, 4, 5, 6, 8, 9]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{}/10 years"
      section: "management_assessment"
      peer_compare: false

  effective_tax_rate_variance:
    name: "Tax Rate Consistency (StdDev)"
    element: "quality_management"
    module: "builtin.profitability"
    function: "compute_tax_rate_variance"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      thresholds: [15, 10, 8, 5, 3, 1]
      direction: "lower_is_better"
      weight: 0.15
    display:
      format: "σ = {:.1f}pp"
      section: "management_assessment"
      peer_compare: false

  # ─────────────── G: GROWTH ───────────────
  revenue_cagr_5yr:
    name: "Revenue CAGR (5yr)"
    element: "growth"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params: { field: "revenue", years: 5 }
    scoring:
      thresholds: [3, 8, 12, 18, 25, 35]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.1f}%"
      section: "growth_decomposition"
      peer_compare: true

  pat_cagr_5yr:
    name: "PAT CAGR (5yr)"
    element: "growth"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params: { field: "pat", years: 5 }
    scoring:
      thresholds: [3, 8, 15, 20, 28, 40]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.1f}%"
      section: "growth_decomposition"
      peer_compare: true

  eps_cagr_5yr:
    name: "EPS CAGR (5yr)"
    element: "growth"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params: { field: "eps", years: 5 }
    scoring:
      thresholds: [3, 8, 15, 20, 28, 40]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.1f}%"
      section: "growth_decomposition"
      peer_compare: true

  operating_leverage:
    name: "Operating Leverage (avg)"
    element: "growth"
    module: "builtin.growth"
    function: "compute_operating_leverage"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      thresholds: [0.5, 0.8, 1.0, 1.3, 1.5, 2.0]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.2f}x"
      section: "growth_decomposition"
      peer_compare: true

  financial_leverage_ratio:
    name: "Financial Leverage (avg)"
    element: "growth"
    module: "builtin.growth"
    function: "compute_financial_leverage"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      # Moderate is best. Too high = risky. Too low = no benefit.
      mode: "range_optimal"
      optimal_range: [0.8, 1.3]
      weight: 0.10
    display:
      format: "{:.2f}x"
      section: "growth_decomposition"
      peer_compare: true

  growth_quality_grade:
    name: "Growth Quality Grade"
    element: "growth"
    module: "builtin.growth"
    function: "compute_growth_quality"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      # Based on primary drivers: Volume+OpLev=High, FinLev+Price=Low
      mode: "categorical"
      categories: { "high_quality": 10, "moderate": 6, "low_quality": 3, "risky": 1 }
      weight: 0.20
    display:
      format: "{}"
      section: "growth_decomposition"
      peer_compare: true

  revenue_growth_consistency:
    name: "Revenue Growth Consistency"
    element: "growth"
    module: "builtin.growth"
    function: "compute_growth_consistency"
    inputs: ["financials"]
    params: { field: "revenue", years: 10 }
    scoring:
      # Lower std dev of YoY growth = more consistent
      thresholds: [30, 25, 20, 15, 10, 5]
      direction: "lower_is_better"
      weight: 0.10
    display:
      format: "σ = {:.1f}%"
      section: "growth_decomposition"
      peer_compare: true

  # ─────────────── L: LONGEVITY ───────────────
  roce_consistency:
    name: "RoCE > 15% Count (10yr)"
    element: "longevity"
    module: "builtin.longevity"
    function: "compute_threshold_consistency"
    inputs: ["financials"]
    params: { field: "roce", years: 10, threshold: 15 }
    scoring:
      thresholds: [3, 5, 6, 7, 8, 9]
      direction: "higher_is_better"
      weight: 0.25
    display:
      format: "{}/10 years"
      section: "longevity_assessment"
      peer_compare: true

  revenue_growth_streak:
    name: "Max Consecutive Growth Years (>10%)"
    element: "longevity"
    module: "builtin.longevity"
    function: "compute_growth_streak"
    inputs: ["financials"]
    params: { field: "revenue", threshold_pct: 10 }
    scoring:
      thresholds: [1, 2, 3, 4, 6, 8]
      direction: "higher_is_better"
      weight: 0.20
    display:
      format: "{} consecutive years"
      section: "longevity_assessment"
      peer_compare: true

  gross_margin_stability:
    name: "Gross Margin Stability (10yr StdDev)"
    element: "longevity"
    module: "builtin.longevity"
    function: "compute_margin_stability"
    inputs: ["financials"]
    params: { field: "gross_margin", years: 10 }
    scoring:
      thresholds: [15, 10, 8, 5, 3, 1.5]
      direction: "lower_is_better"
      weight: 0.20
    display:
      format: "σ = {:.1f}pp"
      section: "longevity_assessment"
      peer_compare: true

  reinvestment_rate:
    name: "Reinvestment Rate (Capex/Depreciation)"
    element: "longevity"
    module: "builtin.longevity"
    function: "compute_reinvestment_rate"
    inputs: ["financials", "cashflow"]
    scoring:
      thresholds: [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.1f}x"
      section: "longevity_assessment"
      peer_compare: true

  fcf_consistency:
    name: "FCF Positive Count (10yr)"
    element: "longevity"
    module: "builtin.longevity"
    function: "compute_fcf_consistency"
    inputs: ["cashflow"]
    params: { years: 10 }
    scoring:
      thresholds: [3, 5, 6, 7, 8, 9]
      direction: "higher_is_better"
      weight: 0.20
    display:
      format: "{}/10 years"
      section: "longevity_assessment"
      peer_compare: true

  # ─────────────── P: PRICE / VALUATION ───────────────
  pe_ttm:
    name: "P/E (TTM)"
    element: "price"
    module: "builtin.valuation"
    function: "compute_pe_ttm"
    inputs: ["financials", "price"]
    scoring:
      mode: "sector_relative_percentile"
      direction: "lower_is_better"
      weight: 0.15
    display:
      format: "{:.1f}x"
      section: "valuation_analysis"
      peer_compare: true

  peg_ratio:
    name: "PEG Ratio"
    element: "price"
    module: "builtin.valuation"
    function: "compute_peg"
    inputs: ["financials", "price"]
    scoring:
      thresholds: [3.0, 2.5, 2.0, 1.5, 1.0, 0.7]
      direction: "lower_is_better"
      weight: 0.20
    display:
      format: "{:.2f}x"
      section: "valuation_analysis"
      peer_compare: true

  ev_ebitda:
    name: "EV/EBITDA"
    element: "price"
    module: "builtin.valuation"
    function: "compute_ev_ebitda"
    inputs: ["financials", "price"]
    scoring:
      mode: "sector_relative_percentile"
      direction: "lower_is_better"
      weight: 0.15
    display:
      format: "{:.1f}x"
      section: "valuation_analysis"
      peer_compare: true

  pe_vs_historical:
    name: "P/E Percentile (10yr range)"
    element: "price"
    module: "builtin.valuation"
    function: "compute_pe_percentile"
    inputs: ["financials", "price"]
    params: { years: 10 }
    scoring:
      thresholds: [90, 75, 60, 50, 35, 20]
      direction: "lower_is_better"
      weight: 0.15
    display:
      format: "{:.0f}th percentile"
      section: "valuation_analysis"
      peer_compare: false

  earnings_yield_vs_gsec:
    name: "Earnings Yield Spread vs G-Sec"
    element: "price"
    module: "builtin.valuation"
    function: "compute_earnings_yield_spread"
    inputs: ["financials", "price"]
    scoring:
      thresholds: [-3, -1, 0, 1, 2, 4]
      direction: "higher_is_better"
      weight: 0.10
    display:
      format: "{:+.1f}pp"
      section: "valuation_analysis"
      peer_compare: false

  dcf_margin_of_safety:
    name: "DCF Margin of Safety"
    element: "price"
    module: "builtin.valuation"
    function: "compute_dcf_margin"
    inputs: ["financials", "cashflow", "price"]
    params:
      projection_years: 10
      terminal_growth: 0.04
      discount_rate: 0.12
    scoring:
      thresholds: [-30, -10, 0, 10, 20, 35]
      direction: "higher_is_better"
      weight: 0.15
    display:
      format: "{:.0f}%"
      section: "valuation_analysis"
      peer_compare: false

  reverse_dcf_growth:
    name: "Reverse DCF Implied Growth"
    element: "price"
    module: "builtin.valuation"
    function: "compute_reverse_dcf"
    inputs: ["financials", "cashflow", "price"]
    scoring:
      # Compare implied growth to actual historical growth
      mode: "comparison_to_actual"
      weight: 0.10
    display:
      format: "{:.1f}% implied vs {:.1f}% actual"
      section: "valuation_analysis"
      peer_compare: false
```

### 4.4 MetricResult Contract

Every metric function follows the same interface:

```python
# metrics/base.py

from dataclasses import dataclass, field

@dataclass
class MetricResult:
    value: float | None                        # The computed number (None if data unavailable)
    raw_series: list[float] = field(default_factory=list)  # Optional: yearly/quarterly values
    flags: list[str] = field(default_factory=list)         # Qualitative flags for LLM context
    metadata: dict = field(default_factory=dict)           # Debug info, years used, etc.
    error: str | None = None                               # Error message if computation failed

# Example implementation:
# metrics/builtin/profitability.py

def compute_roce_avg(data: dict, params: dict) -> MetricResult:
    """
    All metric functions receive:
      data:   dict of DataFrames keyed by input type
              {"financials": pd.DataFrame, "price": pd.DataFrame, ...}
      params: dict from registry.yaml params section

    Must return: MetricResult
    """
    df = data["financials"]
    years = params.get("years", 5)
    roce_values = df["roce"].tail(years).dropna()

    if len(roce_values) < 3:
        return MetricResult(value=None, error="Insufficient data")

    avg = roce_values.mean()
    return MetricResult(
        value=avg,
        raw_series=roce_values.tolist(),
        flags=["consistently_high"] if (roce_values > 15).all() else [],
        metadata={"years_used": len(roce_values)},
    )
```

### 4.5 The Compute Engine

```python
# compute_engine/engine.py

import importlib
import yaml
from metrics.base import MetricResult

class ComputeEngine:
    def __init__(self, registry_path: str = "metrics/registry.yaml"):
        with open(registry_path) as f:
            self.registry = yaml.safe_load(f)
        self.metrics = self.registry["metrics"]
        self.element_weights = self.registry["element_weights"]

    def run_all(self, ticker: str, data: dict) -> dict[str, MetricResult]:
        """Run every registered metric for a company."""
        results = {}
        for metric_id, config in self.metrics.items():
            results[metric_id] = self._run_metric(metric_id, config, data)
        return results

    def run_element(self, element: str, data: dict) -> dict[str, MetricResult]:
        """Run only metrics belonging to a specific SQGLP element."""
        return {
            mid: self._run_metric(mid, cfg, data)
            for mid, cfg in self.metrics.items()
            if cfg["element"] == element
        }

    def _run_metric(self, metric_id: str, config: dict, data: dict) -> MetricResult:
        # Check required data sources
        required = set(config.get("inputs", []))
        available = set(data.keys())
        if not required.issubset(available):
            missing = required - available
            return MetricResult(value=None, error=f"Missing data: {missing}")

        # Dynamically import and call the function
        try:
            module = importlib.import_module(f"compute_engine.metrics.{config['module']}")
            func = getattr(module, config["function"])
            return func(data, config.get("params", {}))
        except Exception as e:
            return MetricResult(value=None, error=str(e))

    def get_display_config(self, metric_id: str) -> dict:
        """Return display formatting info for report generation."""
        return self.metrics[metric_id].get("display", {})

    def get_peer_comparable_metrics(self) -> list[str]:
        """Return metric IDs that should appear in peer comparison tables."""
        return [
            mid for mid, cfg in self.metrics.items()
            if cfg.get("display", {}).get("peer_compare", False)
        ]
```

### 4.6 The Scorer

```python
# compute_engine/scorer.py

class SQGLPScorer:
    def __init__(self, registry: dict):
        self.metrics_config = registry["metrics"]
        self.element_weights = registry["element_weights"]

    def score(self, results: dict[str, MetricResult]) -> dict:
        """
        Compute per-element scores (0-10) and weighted composite.
        Returns: {
            "elements": {"size": 7.2, "quality_business": 8.1, ...},
            "composite": 7.6,
            "details": {metric_id: {"value": X, "score": Y, "weight": Z}, ...}
        }
        """
        element_weighted_scores = {}
        element_total_weights = {}
        details = {}

        for metric_id, result in results.items():
            if result.value is None:
                continue

            config = self.metrics_config[metric_id]
            element = config["element"]
            scoring_config = config["scoring"]
            weight = scoring_config.get("weight", 0.1)

            raw_score = self._compute_raw_score(result.value, scoring_config)

            details[metric_id] = {
                "value": result.value,
                "score": raw_score,
                "weight": weight,
                "flags": result.flags,
            }

            element_weighted_scores.setdefault(element, 0)
            element_total_weights.setdefault(element, 0)
            element_weighted_scores[element] += raw_score * weight
            element_total_weights[element] += weight

        # Normalize element scores to 0-10
        elements = {}
        for el in self.element_weights:
            if element_total_weights.get(el, 0) > 0:
                elements[el] = (
                    element_weighted_scores[el] / element_total_weights[el]
                ) * 10
            else:
                elements[el] = None

        # Weighted composite
        composite = sum(
            elements.get(el, 0) * w
            for el, w in self.element_weights.items()
            if elements.get(el) is not None
        )

        return {"elements": elements, "composite": composite, "details": details}

    def _compute_raw_score(self, value: float, config: dict) -> float:
        """Map a metric value to a 0-1 score using the configured method."""
        mode = config.get("mode", "threshold")
        direction = config.get("direction", "higher_is_better")

        if mode == "threshold" or mode not in ("range_optimal", "categorical",
                                                "sector_relative_percentile",
                                                "trend_direction",
                                                "comparison_to_actual"):
            return self._threshold_score(value, config["thresholds"], direction)
        elif mode == "range_optimal":
            return self._range_score(value, config["optimal_range"])
        elif mode == "categorical":
            return config["categories"].get(value, 0) / 10
        # ... other modes
```

### 4.7 Adding a New Metric — Complete Workflow

**Step 1:** Add entry to `registry.yaml`:

```yaml
  cash_conversion_cycle:
    name: "Cash Conversion Cycle (days)"
    element: "quality_business"
    module: "custom.my_metrics"
    function: "compute_ccc"
    inputs: ["financials"]
    scoring:
      thresholds: [120, 90, 60, 45, 30, 15]
      direction: "lower_is_better"
      weight: 0.05
    display:
      format: "{:.0f} days"
      section: "quality_scorecard"
      peer_compare: true
```

**Step 2:** Write the function:

```python
# metrics/custom/my_metrics.py

from compute_engine.metrics.base import MetricResult

def compute_ccc(data: dict, params: dict) -> MetricResult:
    df = data["financials"]
    receivable_days = df["receivable_days"].iloc[-1]
    inventory_days = df["inventory_days"].iloc[-1]
    payable_days = df["payable_days"].iloc[-1]
    ccc = receivable_days + inventory_days - payable_days
    return MetricResult(
        value=ccc,
        raw_series=[], # Could add trend here
        flags=["negative_ccc_excellent"] if ccc < 0 else [],
    )
```

**Step 3:** Done. No other files need changes. The engine discovers it, scoring includes it, peer comparison includes it (`peer_compare: true`), and the report template renders it in the `quality_scorecard` section.

### 4.8 Future Plug-in Metrics

| Metric                     | Module                     | Why Useful                             |
| -------------------------- | -------------------------- | -------------------------------------- |
| Piotroski F-Score          | `custom/quality.py`        | Aggregate financial health (0-9 score) |
| Altman Z-Score             | `custom/risk.py`           | Bankruptcy risk assessment             |
| DuPont Decomposition       | `builtin/profitability.py` | RoE = Margin × Turnover × Leverage     |
| Sustainable Growth Rate    | `builtin/growth.py`        | RoE × (1 - Payout Ratio)               |
| EVA (Economic Value Added) | `custom/quality.py`        | NOPAT - (Capital × WACC)               |
| Insider Buy/Sell Ratio     | `custom/sentiment.py`      | Management conviction signal           |
| Revenue per Employee Trend | `custom/efficiency.py`     | Operational leverage signal            |
| Benford's Law Analysis     | `custom/forensics.py`      | Accounting fraud detector              |
| Capex/Revenue Trend        | `builtin/growth.py`        | Investment intensity                   |
| Tax Rate Consistency       | `custom/forensics.py`      | Aggressive accounting flag             |

------

## 5. SQGLP Metrics — Detailed Specification

### 5.1 The SQGLP Framework

```
SQGLP Score = (S × 0.10) + (Q_biz × 0.20) + (Q_mgmt × 0.10) + (G × 0.25) + (L × 0.20) + (P × 0.15)
```

| Element                       | Weight | Rationale                                                    |
| ----------------------------- | ------ | ------------------------------------------------------------ |
| **S** — Size                  | 10%    | A filter more than a driver; small size enables discovery    |
| **Q** — Quality of Business   | 20%    | Most important: high-RoCE, cash-generative businesses compound |
| **Q** — Quality of Management | 10%    | Integrity + competence; partially quantitative, partially LLM |
| **G** — Growth                | 25%    | The superstructure: 4-lever decomposition reveals quality of growth |
| **L** — Longevity             | 20%    | Sustainability of quality + growth = moat durability         |
| **P** — Price                 | 15%    | Entry valuation matters, but less than business quality      |

### 5.2 Growth: The 4-Lever Framework

The framework decomposes EPS growth into its constituent drivers using the chain rule:

```
ΔEPS = ΔVolume × (ΔSales/ΔVolume) × (ΔEBIT/ΔSales) × (ΔEPS/ΔEBIT)
       ────────   ─────────────────   ───────────────   ─────────────
       Volume     Price Lever          Operating Lever   Financial Lever
       Growth     (Pricing Power)      (Scale Benefits)  (Debt Amplification)
```

**Growth Quality Grading:**

| Quality Grade    | Primary Drivers                  | Description                                                |
| ---------------- | -------------------------------- | ---------------------------------------------------------- |
| **High Quality** | Volume + Operating Leverage      | Selling more units, achieving economies of scale           |
| **Moderate**     | Volume + Price                   | Growing demand with pricing power                          |
| **Low Quality**  | Financial Leverage + Price Hikes | Debt-driven or aggressive pricing; unsustainable           |
| **Risky**        | Financial Leverage dominant      | Amplified returns in good times, accelerated losses in bad |

### 5.3 Valuation Models (All Computed Offline)

Three intrinsic value models, all running in `builtin/valuation.py`:

**DCF (Discounted Cash Flow):** Project 10yr FCF using historical growth, discount at WACC (12% default for Indian mid-caps), add terminal value at 4% perpetual growth.

**Earnings Power Value:** Normalize last 5yr earnings, divide by cost of equity. Represents value with zero growth — a floor estimate.

**Reverse DCF:** Solve for the growth rate the market is currently pricing in. Compare to actual historical growth. If market implies 30% growth but company has grown at 18%, the market may be overpricing it.

### 5.4 Computed Flags (Auto-Generated, No LLM)

The engine generates qualitative flags that serve two purposes: (1) immediate red/green flags for the human reader, and (2) structured context for the LLM passes.

```python
# Examples of auto-generated flags:
"consistently_high_roce"          # RoCE > 15% in 8+ of 10 years
"improving_margins"               # OPM expanding for 3+ consecutive years
"cash_cow"                        # FCF positive in 8+ of 10 years
"debt_risk"                       # D/E > 1.0 or Interest Coverage < 2x
"promoter_pledge_red_flag"        # Pledge > 10%
"high_dilution"                   # Shares outstanding grew > 30% in 10yr
"under_researched"                # Analyst count <= 3
"low_institutional_ownership"     # FII+DII < 5%
"negative_ccc_excellent"          # Cash Conversion Cycle < 0 days
"growth_quality_high"             # Volume + OpLev driven
"growth_quality_risky"            # FinLev driven
"pe_above_historical_75th"        # Expensive vs own history
"reverse_dcf_overpriced"          # Market implies growth > 1.5x actual
```

------

## 6. LLM Analysis Layer (Stage 3)

### 6.1 Design Principle

Instead of feeding raw financial statements to the LLM (expensive, unfocused), we feed **pre-computed JSON with flags** and ask focused questions. This reduces input tokens by ~90%.

### 6.2 Three-Pass Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM CALL STRATEGY                            │
│                                                                 │
│  Pass 1: QUALITATIVE ANALYSIS (~2K input, ~1K output tokens)   │
│  ├── Model: Claude Sonnet                                      │
│  ├── Input: Annual report excerpts, con-call highlights,        │
│  │          promoter holding trend, computed flags              │
│  ├── Task: Assess management quality, competitive moat,         │
│  │         business model risks, sector tailwinds               │
│  └── Output: Structured qualitative assessment JSON             │
│                                                                 │
│  Pass 2: SYNTHESIS (~3K input, ~2K output tokens)               │
│  ├── Model: Claude Sonnet                                      │
│  ├── Input: All SQGLP computed metrics + flags + Pass 1 output │
│  ├── Task: Investment thesis, conviction, kill-the-thesis risks,│
│  │         what the market is missing                           │
│  └── Output: Investment thesis + risk assessment JSON           │
│                                                                 │
│  Pass 3: COMPARATIVE JUDGMENT (~2K input, ~1K output tokens)    │
│  ├── Model: Claude Haiku (cheaper; simpler task)               │
│  ├── Input: Peer comparison table + sector context             │
│  ├── Task: Rank target vs peers, relative edge analysis        │
│  └── Output: Competitive positioning assessment JSON           │
│                                                                 │
│  ESTIMATED COST: ~$0.05–$0.15 per company                      │
│  (vs ~$2–5 if sending raw data without pre-processing)          │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Prompt Templates

#### Pass 1: Qualitative Analysis

```
SYSTEM: You are an equity research analyst specializing in Indian
mid-cap companies. Analyze the following management and business
quality indicators. Be specific and cite evidence from the data.

INPUT:
- Company: {company_name} ({ticker})
- Sector: {sector}
- Management commentary excerpts: {extracted_annual_report_text}
- Promoter holding trend: {promoter_data}
- Related party transactions summary: {rpt_data}
- Key risks from annual report: {risk_section_text}
- Computed flags: {flags_list}

OUTPUT FORMAT (strict JSON):
{
  "management_integrity_score": <1-10>,
  "management_competence_score": <1-10>,
  "growth_mindset_score": <1-10>,
  "moat_type": "<brand|cost|network|switching|regulatory|none>",
  "moat_strength": <1-10>,
  "moat_evidence": "<specific evidence>",
  "key_risks": ["<risk1>", "<risk2>", "..."],
  "sector_tailwinds": ["<tailwind1>", "..."],
  "sector_headwinds": ["<headwind1>", "..."],
  "red_flags": ["<flag1>", "..."],
  "reasoning": "<2-3 paragraph assessment>"
}
```

#### Pass 2: Synthesis

```
SYSTEM: You are a senior investment analyst. Given the pre-computed
SQGLP metrics and qualitative assessment below, synthesize an
investment thesis. Focus on:
1. Is this a potential long-term compounder (10yr+)?
2. What could go wrong? (kill-the-thesis risks)
3. What is the market missing? (if anything)
4. Does the growth quality justify the valuation?

INPUT:
- Company: {company_name} ({ticker})
- SQGLP Metrics: {sqglp_metrics_json}
- SQGLP Scores: {scores_json}
- Computed Flags: {flags_list}
- Qualitative Assessment (Pass 1): {pass1_output}

OUTPUT FORMAT (strict JSON):
{
  "thesis": "<one paragraph investment thesis>",
  "conviction_level": "<high|medium|low>",
  "bull_case": "<best case scenario in 2-3 sentences>",
  "bear_case": "<worst case scenario in 2-3 sentences>",
  "kill_the_thesis": ["<scenario that would invalidate the thesis>", "..."],
  "key_monitorables": ["<metric or event to track quarterly>", "..."],
  "suggested_action": "<strong_buy|buy|hold|watchlist|avoid>",
  "target_holding_period": "<3-5yr|5-10yr|10yr+>",
  "what_market_is_missing": "<insight or nothing>",
  "reasoning": "<3-4 paragraph detailed reasoning>"
}
```

#### Pass 3: Competitive Judgment

```
SYSTEM: Compare the target company against its sector peers using
the pre-computed metrics below. Identify where it has a clear edge
and where it lags. Be quantitative — reference specific numbers.

INPUT:
- Target: {ticker}
- Peer Comparison Table: {peer_comparison_json}
- Target Investment Thesis: {pass2_thesis}
- Peer discovery metadata: {discovery_metadata}

OUTPUT FORMAT (strict JSON):
{
  "competitive_advantages": ["<specific advantage with evidence>", "..."],
  "competitive_disadvantages": ["<specific weakness>", "..."],
  "best_in_class_metrics": ["<metric_name>", "..."],
  "worst_in_class_metrics": ["<metric_name>", "..."],
  "preferred_pick_in_sector": "<TICKER>",
  "preferred_pick_reasoning": "<why this company over target>",
  "valuation_relative_assessment": "<overvalued|fair|undervalued vs peers>",
  "reasoning": "<2-3 paragraph comparative analysis>"
}
```

### 6.4 LLM Orchestrator

```python
# llm_layer/orchestrator.py

class LLMOrchestrator:
    def __init__(self, api_key: str, config: dict):
        self.client = anthropic.Client(api_key=api_key)
        self.config = config

    def run_analysis(
        self,
        sqglp_metrics: dict,
        scores: dict,
        peer_comparison: dict,
        annual_report_text: str | None = None,
    ) -> dict:
        results = {}

        # Pass 1: Qualitative (skip if no annual report)
        if annual_report_text:
            results["pass1"] = self._call(
                model="claude-sonnet-4-5-20250929",
                prompt_template="pass1_qualitative.txt",
                context={...},
            )
        else:
            results["pass1"] = {"skipped": True, "reason": "No annual report available"}

        # Pass 2: Synthesis (always runs)
        results["pass2"] = self._call(
            model="claude-sonnet-4-5-20250929",
            prompt_template="pass2_synthesis.txt",
            context={
                "sqglp_metrics": sqglp_metrics,
                "scores": scores,
                "pass1": results["pass1"],
                ...
            },
        )

        # Pass 3: Comparative (use cheaper model)
        results["pass3"] = self._call(
            model="claude-haiku-4-5-20251001",
            prompt_template="pass3_comparative.txt",
            context={"peer_comparison": peer_comparison, ...},
        )

        return results

    def _call(self, model: str, prompt_template: str, context: dict) -> dict:
        """Load template, render with context, call API, parse JSON response."""
        template = self._load_template(prompt_template)
        prompt = template.format(**context)

        response = self.client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_json_response(response.content[0].text)
```

### 6.5 Cost Optimization Strategies

| Strategy                   | Savings                       | How                                                          |
| -------------------------- | ----------------------------- | ------------------------------------------------------------ |
| **Use Haiku for Pass 3**   | ~60% on that pass             | Comparative ranking is a simpler task                        |
| **Cache sector context**   | Avoid re-computation          | Same sector prompt prefix for all peers                      |
| **Batch peers in Pass 3**  | Reduce system prompt overhead | Analyze 3-4 peers in one call                                |
| **Skip Pass 1 if no AR**   | Save one full call            | Only run when annual report PDF is available                 |
| **Structured JSON output** | Fewer output tokens           | JSON schema forces concise responses                         |
| **Pre-computed flags**     | Reduce LLM reasoning load     | Flags like `consistently_high_roce` mean LLM doesn't re-derive them |

------

## 7. Output & Report Generation (Stage 4)

### 7.1 Output Files

```
reports/
└── {TICKER}_{DATE}/
    ├── sqglp_dashboard.html      # Self-contained interactive HTML (Plotly charts)
    ├── sqglp_report.md           # Markdown summary for quick reading
    ├── raw_metrics.json          # All computed metric results
    ├── peer_comparison.json      # Peer metrics side-by-side
    ├── peer_discovery.json       # How peers were discovered
    ├── llm_analysis.json         # All 3 LLM pass outputs
    └── scores.json               # SQGLP element scores + composite
```

### 7.2 Report Sections

1. **Executive Summary** — One-paragraph thesis, conviction level, SQGLP radar chart
2. **SQGLP Score Dashboard** — Composite score + element breakdown + flags
3. **Size Analysis** — Market cap, institutional ownership, discovery potential
4. **Quality Scorecard** — Business quality ratios (table with sparklines for trends)
5. **Management Assessment** — Quantitative signals + LLM qualitative assessment
6. **Growth Decomposition** — 4-lever analysis chart, growth quality grade
7. **Longevity Assessment** — Consistency metrics, moat analysis
8. **Valuation Analysis** — P/E band chart, DCF, PEG, reverse DCF
9. **Peer Comparison** — Side-by-side table (auto-generated from registry)
10. **Peer Discovery** — How peers were identified, similarity scores
11. **Investment Thesis** — Bull/bear case, kill-the-thesis scenarios
12. **Risk Register** — Red flags, computed warnings, LLM-identified risks
13. **Monitorables Checklist** — What to track quarterly
14. **Appendix** — Raw data tables, methodology, data sources

### 7.3 HTML Dashboard (GUI Lite)

The primary output format. Uses Jinja2 templates with embedded Plotly charts to produce a self-contained `.html` file with interactive visualizations.

```python
# output/report_generator.py

from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.io as pio

class ReportGenerator:
    def __init__(self, template_dir: str = "output/templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate_html(self, result: "AnalysisResult") -> str:
        template = self.env.get_template("sqglp_report.html.j2")
        return template.render(
            company=result.company,
            scores=result.scores,
            metrics=result.metrics,
            flags=result.flags,
            radar_chart=self._radar_chart(result.scores),
            growth_chart=self._growth_decomposition_chart(result.metrics),
            pe_band_chart=self._pe_band_chart(result.metrics),
            roce_trend_chart=self._trend_chart(result.metrics, "roce"),
            peer_table=result.peer_comparison,
            peer_discovery=result.peer_discovery,
            llm_analysis=result.llm_analysis,
            generation_date=datetime.now().isoformat(),
        )

    def generate_markdown(self, result: "AnalysisResult") -> str:
        template = self.env.get_template("sqglp_report.md.j2")
        return template.render(...)

    def _radar_chart(self, scores: dict) -> str:
        """SQGLP radar chart as embedded Plotly HTML."""
        categories = ["Size", "Quality\n(Business)", "Quality\n(Mgmt)",
                      "Growth", "Longevity", "Price"]
        values = [scores["elements"].get(el, 0) for el in
                  ["size", "quality_business", "quality_management",
                   "growth", "longevity", "price"]]
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])), showlegend=False)
        return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
```

------

## 8. Service Layer — The GUI-Ready API

### 8.1 Purpose

`service.py` is the central orchestrator that the CLI calls today, and a future GUI would call tomorrow. **All business logic lives here** — not in CLI scripts, not in GUI routes.

### 8.2 Interface

```python
# service.py

from dataclasses import dataclass

@dataclass
class AnalysisResult:
    company: dict                  # Ticker, name, sector
    metrics: dict                  # All MetricResult objects
    scores: dict                   # SQGLP element + composite scores
    flags: list[str]               # All computed flags
    peer_discovery: "PeerResult"   # How peers were found
    peer_comparison: dict          # Side-by-side metric table
    llm_analysis: dict | None      # 3-pass LLM output (None if --no-llm)

class ResearchService:
    """
    The single API for all research operations.
    CLI calls it. Future Streamlit/FastAPI calls it.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.fetcher_suite = DataFetcherSuite(self.config)
        self.peer_discovery = PeerDiscovery(self.config)
        self.engine = ComputeEngine()
        self.scorer = SQGLPScorer(self.engine.registry)
        self.llm = LLMOrchestrator(self.config)
        self.reporter = ReportGenerator()

    def analyze_company(
        self,
        ticker: str,
        peers: list[str] | None = None,
        use_llm: bool = True,
        llm_peers: bool = False,
    ) -> AnalysisResult:
        """Full pipeline: fetch → discover peers → compute → score → LLM → report."""

        # Stage 1: Fetch data
        data = self.fetcher_suite.fetch_all(ticker)

        # Stage 1.5: Discover peers (auto or manual)
        if peers:
            peer_result = PeerResult(direct_competitors=peers, ...)
        else:
            peer_result = self.peer_discovery.discover(ticker, use_llm=llm_peers)

        # Stage 2: Compute metrics
        metrics = self.engine.run_all(ticker, data)
        scores = self.scorer.score(metrics)
        flags = self._collect_flags(metrics)

        # Stage 2: Compute peer metrics
        peer_metrics = {}
        for peer_ticker in peer_result.direct_competitors:
            peer_data = self.fetcher_suite.fetch_all(peer_ticker)
            peer_metrics[peer_ticker] = self.engine.run_all(peer_ticker, peer_data)

        peer_comparison = self._build_comparison_table(ticker, metrics, peer_metrics)

        # Stage 3: LLM analysis (optional)
        llm_analysis = None
        if use_llm:
            annual_report_text = self._extract_annual_report_text(ticker)
            llm_analysis = self.llm.run_analysis(
                sqglp_metrics=self._metrics_to_json(metrics),
                scores=scores,
                peer_comparison=peer_comparison,
                annual_report_text=annual_report_text,
            )

        result = AnalysisResult(
            company={"ticker": ticker, "sector": data.get("sector")},
            metrics=metrics,
            scores=scores,
            flags=flags,
            peer_discovery=peer_result,
            peer_comparison=peer_comparison,
            llm_analysis=llm_analysis,
        )

        # Stage 4: Generate reports
        self.reporter.generate_and_save(result)

        return result

    def discover_peers(self, ticker: str, use_llm: bool = False) -> "PeerResult":
        """Standalone peer discovery."""
        return self.peer_discovery.discover(ticker, use_llm=use_llm)

    def screen_universe(self, filters: dict) -> list[dict]:
        """
        Filter NSE universe by criteria.
        Uses registry-defined metrics as filter dimensions.
        e.g., filters={"roce_5yr_avg": {"min": 15}, "market_cap": {"max": 50000}}
        """
        ...

    def get_watchlist(self, watchlist_path: str) -> list[dict]:
        """Return current watchlist with latest SQGLP scores."""
        ...

    def run_quarterly_update(self, watchlist_path: str) -> list["AnalysisResult"]:
        """Re-run analysis on all watchlist companies. Returns updated results."""
        ...
```

------

## 9. Project Structure

```
sqglp_research/
│
├── config.yaml                          # Pipeline configuration
│
├── data_fetcher/                        # ── STAGE 1: Data Acquisition ──
│   ├── __init__.py
│   ├── base.py                          # BaseFetcher (retry, rate limit, caching)
│   ├── cache/
│   │   ├── cache_manager.py             # TTL-based local cache
│   │   └── cached_data/
│   ├── fetch_financials.py              # Screener.in → P&L, BS, CF
│   ├── fetch_shareholding.py            # BSE → quarterly shareholding
│   ├── fetch_price_volume.py            # NSE → OHLCV via jugaad-data
│   ├── fetch_corporate_actions.py       # BSE → splits, bonuses, dividends
│   ├── fetch_analyst_coverage.py        # Trendlyne → analyst count
│   ├── fetch_sector_peers.py            # Screener.in → sector list
│   ├── download_annual_reports.py       # BSE → annual report PDFs
│   ├── peer_discovery.py                # Multi-layer peer identification
│   └── raw_data/                        # Cached raw data per ticker
│       └── {TICKER}/
│           ├── financials.csv
│           ├── balance_sheet.csv
│           ├── cashflow.csv
│           ├── shareholding.csv
│           ├── price_volume.csv
│           └── annual_reports/*.pdf
│
├── compute_engine/                      # ── STAGE 2: Offline Computation ──
│   ├── __init__.py
│   ├── metrics/
│   │   ├── registry.yaml                # Single source of truth for all metrics
│   │   ├── base.py                      # MetricResult dataclass
│   │   ├── builtin/                     # Shipped metric implementations
│   │   │   ├── __init__.py
│   │   │   ├── profitability.py         # RoCE, RoE, OPM, NPM, Cash Conversion
│   │   │   ├── growth.py               # CAGR, 4-lever, consistency, dilution
│   │   │   ├── valuation.py            # P/E, PEG, EV/EBITDA, DCF, reverse DCF
│   │   │   ├── leverage.py             # D/E, Interest Coverage
│   │   │   ├── efficiency.py           # Working Capital Days, CCC, Asset Turnover
│   │   │   ├── size.py                 # Market cap, institutional, turnover, promoter
│   │   │   └── longevity.py            # Consistency, streaks, stability, reinvestment
│   │   └── custom/                      # User drop-in metrics
│   │       ├── __init__.py
│   │       └── my_metrics.py
│   ├── engine.py                        # Generic registry-driven metric runner
│   ├── scorer.py                        # SQGLP scoring from registry weights
│   └── peer_comparison.py               # Run engine on target + peers
│
├── llm_layer/                           # ── STAGE 3: LLM Analysis ──
│   ├── __init__.py
│   ├── orchestrator.py                  # 3-pass LLM orchestration
│   └── prompts/
│       ├── pass1_qualitative.txt        # Management + moat assessment
│       ├── pass2_synthesis.txt          # Investment thesis generation
│       └── pass3_comparative.txt        # Peer ranking
│
├── output/                              # ── STAGE 4: Report Generation ──
│   ├── __init__.py
│   ├── report_generator.py              # Jinja2 + Plotly HTML/MD generation
│   ├── templates/
│   │   ├── sqglp_report.html.j2         # Interactive HTML dashboard template
│   │   └── sqglp_report.md.j2           # Markdown report template
│   └── reports/                         # Generated reports
│       └── {TICKER}_{DATE}/
│           ├── sqglp_dashboard.html
│           ├── sqglp_report.md
│           ├── raw_metrics.json
│           ├── peer_comparison.json
│           ├── peer_discovery.json
│           ├── llm_analysis.json
│           └── scores.json
│
├── service.py                           # Central API layer (GUI-ready)
├── cli.py                               # Command-line interface (typer)
├── requirements.txt                     # Python dependencies
├── .env                                 # API keys (ANTHROPIC_API_KEY)
└── README.md
```

------

## 10. Configuration Reference

```yaml
# config.yaml — Complete configuration

# ── Target Company ──
target:
  ticker: "ASTRAL"
  bse_code: "532830"
  nse_symbol: "ASTRAL"

# ── Peer Discovery ──
# Option A: Auto-discover (recommended)
peer_discovery:
  enabled: true
  use_llm_validation: false          # Layer 4-5 (adds ~$0.02)
  max_peers: 5
  size_band_multiplier: 3.0          # Market cap: 0.3x to 3x of target
  revenue_band_multiplier: 5.0       # Revenue: 0.2x to 5x of target
  sector_source: "screener"          # "screener" | "bse" | "moneycontrol"
  include_financial_peers: true      # Cross-sector similarity matches
  min_listing_years: 5               # Exclude recently listed companies

# Option B: Manual peers (overrides auto-discovery if specified)
# peers:
#   - { ticker: "SUPREMEIND", name: "Supreme Industries" }
#   - { ticker: "PRINCEPIPE", name: "Prince Pipes" }

# ── Analysis Period ──
analysis_period:
  financials_years: 10
  price_history_years: 10
  shareholding_quarters: 20          # 5 years quarterly

# ── Data Fetching ──
fetching:
  rate_limit_seconds: 2              # Delay between HTTP requests
  cache_ttl_hours: 24                # Local cache expiry
  screener_base_url: "https://www.screener.in"
  retry_count: 3
  retry_delay_seconds: 5

# ── LLM Configuration ──
llm:
  provider: "anthropic"
  pass1_model: "claude-sonnet-4-5-20250929"
  pass2_model: "claude-sonnet-4-5-20250929"
  pass3_model: "claude-haiku-4-5-20251001"
  max_tokens: 2000
  enabled: true                      # Set false for compute-only runs
  skip_pass1_if_no_ar: true          # Skip qualitative if no annual report

# ── Scoring ──
scoring:
  # Override element weights (defaults from registry.yaml)
  # element_weights:
  #   size: 0.10
  #   quality_business: 0.20
  #   ...

# ── Output ──
output:
  formats: ["html", "md", "json"]    # Which report formats to generate
  report_dir: "output/reports"
  include_raw_data: true
```

------

## 11. Technology Stack

| Component             | Technology                                               | Cost                 |
| --------------------- | -------------------------------------------------------- | -------------------- |
| **Language**          | Python 3.11+                                             | Free                 |
| **Data Fetching**     | requests, beautifulsoup4, jugaad-data, nsetools, bsedata | Free                 |
| **Data Storage**      | CSV/JSON files (SQLite if scaling to 500+ companies)     | Free                 |
| **Computation**       | pandas, numpy, scipy                                     | Free                 |
| **PDF Extraction**    | PyMuPDF (fitz)                                           | Free                 |
| **Visualization**     | Plotly (embedded in HTML)                                | Free                 |
| **LLM Analysis**      | Claude API (Sonnet + Haiku)                              | ~$0.05–$0.15/company |
| **Report Generation** | Jinja2 templates → HTML/Markdown                         | Free                 |
| **CLI Framework**     | typer (or click)                                         | Free                 |
| **Config**            | PyYAML                                                   | Free                 |
| **API Client**        | anthropic Python SDK                                     | Free                 |

### Python Dependencies (`requirements.txt`)

```
# Data fetching
requests>=2.31
beautifulsoup4>=4.12
jugaad-data>=0.25
nsetools>=1.0
bsedata>=0.5
yfinance>=0.2        # Fallback

# Computation
pandas>=2.1
numpy>=1.25
scipy>=1.11

# PDF processing
PyMuPDF>=1.23

# Visualization & reporting
plotly>=5.18
jinja2>=3.1

# LLM
anthropic>=0.39

# CLI & config
typer>=0.9
pyyaml>=6.0

# Dev
pytest>=7.4
```

------

## 12. CLI Reference

```bash
# ── Full Analysis ──
# Auto-discover peers, compute everything, run LLM, generate reports
python cli.py analyze ASTRAL

# Manual peers, skip LLM
python cli.py analyze ASTRAL --peers SUPREMEIND,PRINCEPIPE --no-llm

# Auto-discover with LLM peer validation
python cli.py analyze ASTRAL --llm-peers

# ── Peer Discovery Only ──
python cli.py peers ASTRAL
python cli.py peers ASTRAL --llm        # With LLM validation

# ── Compute Only (no LLM, no reports) ──
python cli.py compute ASTRAL            # Outputs JSON metrics

# ── Screening ──
python cli.py screen --min-roce 15 --max-mcap 50000 --min-rev-cagr 15
python cli.py screen --config screen_filters.yaml

# ── Watchlist ──
python cli.py watchlist show
python cli.py watchlist add ASTRAL
python cli.py watchlist update          # Re-run analysis on all
python cli.py watchlist update --quarterly  # Only if last run > 90 days ago
```

------

## 13. Implementation Roadmap

### Phase 1: Foundation (Week 1–2)

- [ ] Initialize project structure, `config.yaml`, `requirements.txt`
- [ ] Implement `base.py` (BaseFetcher with caching & rate limiting)
- [ ] Implement `fetch_financials.py` (Screener.in → 10yr P&L, BS, CF)
- [ ] Implement `fetch_price_volume.py` (NSE via jugaad-data)
- [ ] Implement `fetch_shareholding.py` (BSE quarterly)
- [ ] Implement `fetch_sector_peers.py` (Screener.in sector page)
- [ ] **Validation:** Fetch data for 3 known companies (e.g., Astral, Bajaj Finance, TCS). Manually verify 10 data points per company against Screener.in.

### Phase 2: Compute Engine (Week 3–4)

- [ ] Implement `metrics/base.py` (MetricResult dataclass)
- [ ] Create `registry.yaml` with initial ~30 metrics
- [ ] Implement `engine.py` (generic registry-driven metric runner)
- [ ] Implement `builtin/profitability.py` (RoCE, RoE, OPM, cash conversion)
- [ ] Implement `builtin/growth.py` (CAGR, 4-lever decomposition, quality grade)
- [ ] Implement `builtin/valuation.py` (P/E, PEG, EV/EBITDA, DCF, reverse DCF)
- [ ] Implement `builtin/leverage.py` (D/E, interest coverage)
- [ ] Implement `builtin/efficiency.py` (working capital days, CCC)
- [ ] Implement `builtin/size.py` (market cap, institutional, turnover, promoter)
- [ ] Implement `builtin/longevity.py` (consistency, streaks, stability)
- [ ] Implement `scorer.py` (SQGLP scoring from registry weights)
- [ ] Implement `peer_comparison.py` (engine on target + peers)
- [ ] **Validation:** Run on 3 companies, compare scores to manual analysis. Verify DCF outputs against a spreadsheet model.

### Phase 3: Peer Discovery (Week 4–5)

- [ ] Implement Layer 1: `_get_sector_peers()` from Screener.in
- [ ] Implement Layer 2: `_filter_by_size()` (market cap + revenue band)
- [ ] Implement Layer 3: `_rank_by_financial_similarity()` (euclidean distance)
- [ ] **Validation:** Given "ASTRAL", verify it discovers Supreme, Prince, Finolex.
- [ ] Optional: Layer 4 LLM validation prompt

### Phase 4: LLM Integration (Week 5–6)

- [ ] Design and test Pass 1 prompt (qualitative analysis)
- [ ] Design and test Pass 2 prompt (synthesis + thesis)
- [ ] Design and test Pass 3 prompt (peer comparison)
- [ ] Implement `orchestrator.py` with retry logic + JSON parsing
- [ ] Implement annual report PDF text extraction (PyMuPDF)
- [ ] Implement `service.py` (central API orchestrating all stages)
- [ ] **Validation:** End-to-end pipeline on 5 companies. Review LLM output quality.

### Phase 5: Reporting & CLI (Week 6–7)

- [ ] Implement `report_generator.py` with Plotly chart functions
- [ ] Build `sqglp_report.html.j2` (interactive HTML dashboard)
- [ ] Build `sqglp_report.md.j2` (markdown summary)
- [ ] Implement `cli.py` (analyze, peers, compute, screen, watchlist commands)
- [ ] **Validation:** Generate reports for 5 companies. Review readability, chart quality.

### Phase 6: Screening & Watchlist (Week 8+)

- [ ] Build universe screener using registry-defined metric filters
- [ ] Implement watchlist management (add/remove/show/update)
- [ ] Implement quarterly update logic
- [ ] Shortlist top 20 candidates → run full pipeline

------

## 14. Key Design Decisions & Rationale

**Why Screener.in instead of raw BSE XBRL filings?** Screener.in normalizes financial data across companies, handles consolidated vs standalone, and provides 10-year history in a clean format. Parsing raw XBRL is 10x more effort for the same data. The tradeoff is dependency on a third-party site — mitigated by caching and fallback sources.

**Why JSON intermediate format between stages?** JSON acts as a contract between compute and LLM layers. You can inspect, validate, and version-control computed metrics independently. If LLM pricing changes, swap models without touching computation. If a metric formula is wrong, fix it without re-running LLM calls. This also enables running compute-only mode (`--no-llm`) for quick screening.

**Why 3 LLM passes instead of 1 big call?** Each pass has a focused scope with a specific output schema, producing more reliable and structured output. A single large prompt with all data tends to produce generic analysis. Splitting also lets you use cheaper models for simpler tasks (Pass 3 with Haiku) and skip passes when data isn't available (Pass 1 without annual reports).

**Why a metric registry instead of hardcoded modules?** Adding a metric should be a 5-minute task (YAML entry + function), not a multi-file refactor. The registry pattern also makes scoring, peer comparison, and report generation self-configuring — they read the registry and auto-adapt.

**Why not a database?** For personal research on 20-50 companies, CSV/JSON files are simpler to inspect, debug, and version-control with git. If you scale to 500+ companies or need time-series queries, migrate to SQLite (one-file database, zero setup) or PostgreSQL.

**Why the service layer even without a GUI?** It enforces clean separation. Without it, business logic leaks into CLI argument parsing, making it impossible to reuse. With it, adding a Streamlit frontend is literally `result = service.analyze_company("ASTRAL")` — zero refactoring of the pipeline.

------

## 15. Future Work: Interactive GUI

### 15.1 Why Defer

The GUI is deliberately deferred from the core build for three reasons:

1. **Pipeline instability:** Data sources, metric definitions, scoring weights, and LLM prompts will all evolve significantly in the first 2-3 months. GUI changes during this period are wasted effort.
2. **Value concentration:** The quality of research depends on correct RoCE calculations and good LLM prompts, not on having a button to click. CLI + HTML reports serve the analytical workflow well.
3. **Scope explosion:** A useful GUI is not just "display the report" — it's interactive filtering, drill-downs, watchlist management, alerts, and charting. That's a 4-6 week project that doesn't improve the analytical output.

### 15.2 Progression Path

```
Phase 1-5 (Now)              Phase 6-7 (Month 3-4)           Phase 8 (Month 5+)
────────────────────         ──────────────────────────       ──────────────────────
CLI + JSON output        →   Static HTML dashboards      →   Interactive Web GUI
                             (Jinja2 + Plotly charts)         (Streamlit or Dash)
                             ↑                                ↑
                             Might be sufficient              Only if analyzing 50+
                             indefinitely for personal        companies, sharing with
                             research                         others, or wanting alerts
```

### 15.3 GUI-Readiness Already Built In

The `service.py` layer ensures the backend is GUI-ready from day one. A future GUI would call the exact same methods the CLI calls:

```python
# Future: app_streamlit.py (entire GUI wrapper)

import streamlit as st
from service import ResearchService

service = ResearchService()

st.title("SQGLP Research Dashboard")

ticker = st.text_input("Enter Ticker", "ASTRAL")
if st.button("Analyze"):
    with st.spinner("Running pipeline..."):
        result = service.analyze_company(ticker)

    # SQGLP Radar
    st.plotly_chart(create_radar(result.scores))

    # Metrics table
    st.dataframe(metrics_to_df(result.metrics))

    # LLM Thesis
    if result.llm_analysis:
        st.markdown(f"**Thesis:** {result.llm_analysis['pass2']['thesis']}")
        st.markdown(f"**Conviction:** {result.llm_analysis['pass2']['conviction_level']}")

    # Peer comparison
    st.dataframe(peer_comparison_to_df(result.peer_comparison))
```

### 15.4 Technology Options When Ready

| Option                    | Best For                         | Effort        | Tradeoffs                                                    |
| ------------------------- | -------------------------------- | ------------- | ------------------------------------------------------------ |
| **Static HTML (Current)** | Personal research, email sharing | Already built | No interactivity beyond Plotly charts                        |
| **Streamlit**             | Quick personal dashboard         | 1-2 weeks     | Pythonic, fast. Reruns on every interaction (can be slow for heavy pipelines). Fine for personal use. |
| **Plotly Dash**           | Chart-heavy financial dashboard  | 2-3 weeks     | Better charting control than Streamlit. More code but more customizable. Still Python-only. |
| **FastAPI + React**       | Multi-user product, polished UX  | 5-8 weeks     | Full separation of concerns. REST API + SPA frontend. Only justified if sharing as a tool with others. |

### 15.5 GUI Feature Roadmap (When Implemented)

**Phase A — Dashboard (Week 1-2):**

- Single-company analysis view with SQGLP radar
- Metric tables with conditional formatting (green/amber/red)
- Plotly charts: P/E band, RoCE trend, growth decomposition
- LLM thesis display

**Phase B — Comparison & Discovery (Week 2-3):**

- Peer comparison view (side-by-side tables and charts)
- Peer discovery explorer (see how peers were identified)
- Sector heatmap (all sector companies colored by SQGLP score)

**Phase C — Screening & Watchlist (Week 3-4):**

- Interactive screener with filter sliders (RoCE, growth, P/E, etc.)
- Watchlist management (add/remove, track score changes over time)
- Score change alerts (notify when a watchlist company's score changes significantly)

**Phase D — Research Workflow (Week 4+):**

- Side-by-side comparison of any two companies
- Historical analysis (how did the SQGLP score change over time?)
- Export to PDF/DOCX for sharing
- Notes and tags on companies
- Integration with annual report viewer (embedded PDF with highlights)

### 15.6 Data Layer Evolution for GUI

If moving to a GUI, the data layer likely evolves:

```
Phase 1-5:  CSV/JSON files   → Simple, inspectable, git-friendly
Phase 6-7:  SQLite            → Queryable, still single-file, no setup
Phase 8+:   PostgreSQL        → Multi-user, concurrent access, full SQL
```

The `service.py` abstracts this — swap the data layer without touching business logic or GUI code.