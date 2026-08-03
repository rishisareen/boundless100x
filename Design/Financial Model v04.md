# Boundless100x — SQGLP Research Pipeline
## System Design v04

> **v04 change:** Peer comparison feature removed. The system is now focused exclusively on absolute quality assessment of a single company. Pipeline is 4 stages + report generation, LLM layer is 2-pass.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Fetching Layer](#2-data-fetching-layer)
3. [Compute Engine](#3-compute-engine)
4. [SQGLP Metrics — Detailed Specification](#4-sqglp-metrics--detailed-specification)
5. [Sector & Macro Context](#5-sector--macro-context)
6. [LLM Analysis Layer](#6-llm-analysis-layer)
7. [Output & Report Generation](#7-output--report-generation)
8. [Service Layer — The GUI-Ready API](#8-service-layer--the-gui-ready-api)
9. [Project Structure](#9-project-structure)
10. [Configuration Reference](#10-configuration-reference)
11. [Technology Stack](#11-technology-stack)
12. [CLI Reference](#12-cli-reference)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Key Design Decisions & Rationale](#14-key-design-decisions--rationale)
15. [Future Work: Interactive GUI](#15-future-work-interactive-gui)

---

## 1. Architecture Overview

### 1.1 The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BOUNDLESS100X PIPELINE                         │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ STAGE 1  │   │ STAGE 2  │   │ STAGE 3  │   │    STAGE 4       │ │
│  │  Data    │──▶│ Compute  │──▶│  SQGLP   │──▶│  LLM Analysis   │ │
│  │  Fetch   │   │ Engine   │   │ Scoring  │   │  (2-pass)       │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
│       │               │               │                │            │
│  Screener.in     44 metrics       Element +        Pass 1:          │
│  yfinance        MetricResult     composite        Qualitative       │
│  BSE             dataclass        scores           (annual report)   │
│  Trendlyne       auto-discovered  + growth         Pass 2:          │
│                  from YAML        decomposition    Investment thesis  │
│                                                                     │
│                              ▼                                      │
│                   ┌─────────────────────┐                          │
│                   │  REPORT GENERATION  │                          │
│                   │  HTML + Markdown    │                          │
│                   └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Philosophy

**Absolute quality assessment, not relative ranking.** The SQGLP framework is designed to evaluate a single company on 44 dimensions of quality, growth, and valuation. It answers: "Is this a potential long-term compounder?" — not "Is it better than its sector peers?"

This distinction matters:
- Peer comparison adds latency, API cost, and complexity without improving the core question
- A mediocre company doesn't become investable because it's the least bad in a weak sector
- The 44 SQGLP metrics already capture all dimensions a long-term investor needs

**Compute locally, use LLM for judgment.** All quantitative metrics run offline on fetched CSV/JSON data. The LLM is reserved for the ~10 questions in the QGLP 25-question checklist that genuinely require qualitative reading of management commentary, annual reports, and business context.

### 1.3 Data Flow

```
ticker (e.g., "ASTRAL")
    │
    ▼
[Stage 1] FetcherSuite.fetch_all(ticker)
    ├── FinancialsFetcher  → raw_data/ASTRAL/financials.csv
    ├── PriceVolumeFetcher → raw_data/ASTRAL/price_volume.csv
    ├── ShareholdingFetcher → raw_data/ASTRAL/shareholding.csv
    ├── CorporateActionsFetcher → raw_data/ASTRAL/corporate_actions.csv
    ├── AnalystCoverageFetcher → raw_data/ASTRAL/analyst_coverage.json
    └── AnnualReportFetcher → raw_data/ASTRAL/annual_reports/*.pdf
    │
    ▼
[Stage 2] ComputeEngine.run_all(ticker, data)
    ├── Auto-discovers 44 metrics from elements/*.yaml + custom/*.yaml
    ├── Imports and calls each metric function
    └── Returns {metric_id: MetricResult} dict
    │
    ▼
[Stage 3] SQGLPScorer.score(metrics)
    ├── Per-metric threshold scoring → 0-1 raw score
    ├── Per-element weighted aggregation → 0-10 element scores
    ├── Weighted composite → 0-10 composite score
    └── GrowthDecomposer → 4-lever breakdown + quality grade
    │
    ▼
[Stage 4] LLMOrchestrator.run_analysis(metrics, scores, annual_report_text)
    ├── Pass 1: Qualitative (management, moat, risks) ← skipped if no annual report
    └── Pass 2: Investment thesis (conviction, bull/bear, kill-the-thesis)
    │
    ▼
[Report] ReportGenerator.generate(result, formats=["html", "md", "json"])
    ├── sqglp_report.html  ← interactive Plotly dashboard
    ├── sqglp_report.md    ← markdown summary
    ├── raw_metrics.json
    ├── llm_analysis.json
    └── scores.json
```

---

## 2. Data Fetching Layer

### 2.1 Fetcher Architecture

All fetchers inherit from `BaseFetcher` which provides:

```python
# data_fetcher/base.py

class BaseFetcher:
    """
    Base class providing:
    - Rate limiting (configurable delay between requests)
    - Retry logic with exponential backoff
    - TTL-based disk caching
    - Structured logging
    """

    def __init__(self, config: dict):
        self.rate_limit = config["fetching"]["rate_limit_seconds"]
        self.cache_ttl = config["fetching"]["cache_ttl_hours"]
        self.cache = CacheManager(ttl_hours=self.cache_ttl)

    def fetch(self, ticker: str) -> dict:
        """Override in subclasses. Returns normalized dict."""
        raise NotImplementedError

    def _get(self, url: str, **kwargs) -> requests.Response:
        """Rate-limited, retrying HTTP GET with caching."""
        cache_key = self._cache_key(url, kwargs)
        if cached := self.cache.get(cache_key):
            return cached

        time.sleep(self.rate_limit)
        for attempt in range(self.retry_count):
            try:
                response = requests.get(url, timeout=30, **kwargs)
                response.raise_for_status()
                self.cache.set(cache_key, response)
                return response
            except requests.RequestException as e:
                if attempt == self.retry_count - 1:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))
```

### 2.2 Data Sources

| Fetcher | Source | Data | Format |
|---------|--------|------|--------|
| `fetch_financials.py` | Screener.in | P&L, Balance Sheet, Cash Flow (10yr) | CSV |
| `fetch_price_volume.py` | yfinance | Daily OHLCV, market cap (10yr) | CSV |
| `fetch_shareholding.py` | BSE | Quarterly promoter/FII/DII/public | CSV |
| `fetch_corporate_actions.py` | BSE | Bonuses, splits, dividends | CSV |
| `fetch_analyst_coverage.py` | Trendlyne | Analyst count, coverage | JSON |
| `download_annual_reports.py` | BSE | Annual report PDFs → extracted text | TXT |

### 2.3 FetcherSuite Orchestration

```python
# data_fetcher/suite.py

class FetcherSuite:
    """Runs all fetchers for a ticker, returns combined data dict."""

    def fetch_all(self, ticker: str) -> dict:
        data = {}

        # Stage 1a: Core financials
        data["financials"] = self.financials.fetch(ticker)       # P&L, BS, CF
        data["price"] = self.price.fetch(ticker)                 # OHLCV history
        data["shareholding"] = self.shareholding.fetch(ticker)   # Quarterly %
        data["corporate_actions"] = self.corp_actions.fetch(ticker)

        # Stage 1b: Supplementary data
        try:
            data["analyst_coverage"] = self.analyst.fetch(ticker)
        except Exception as e:
            logger.warning(f"Analyst coverage unavailable: {e}")
            data["analyst_coverage"] = {}

        # Stage 1c: Annual report text (best-effort)
        try:
            data["annual_report_text"] = self.annual_reports.fetch(ticker)
        except Exception as e:
            logger.warning(f"Annual report unavailable: {e}")
            data["annual_report_text"] = None

        return data
```

### 2.4 Screener.in Parser

The financial data source. Normalizes Screener.in's table format:

```python
# data_fetcher/fetch_financials.py

def _parse_profit_loss(self, soup) -> pd.DataFrame:
    """
    Parse Screener.in P&L table into a normalized DataFrame.

    Screener columns (years run right-to-left, oldest first):
      Mar 2015 | Mar 2016 | ... | Mar 2024 | TTM

    Output columns (standardized):
      year | revenue | expenses | operating_profit | opm_pct |
      other_income | interest | depreciation | profit_before_tax |
      tax | net_profit | eps
    """
```

**Key normalization decisions:**
- Revenues, expenses in ₹ Crores
- Percentages stored as decimals (0.25 = 25%)
- "TTM" column handled separately
- Non-numeric characters stripped from scraped values

### 2.5 Annual Report Processing

```python
# data_fetcher/download_annual_reports.py

class AnnualReportFetcher(BaseFetcher):
    """
    Downloads annual report PDFs from BSE and extracts text.

    Flow:
    1. Fetch filing list from BSE API (bseindia.com)
    2. Download most recent annual report PDF
    3. Extract text from first N pages (config: annual_reports.max_pages)
    4. Cap at max_text_chars to control LLM token cost
    """

    def fetch(self, ticker: str) -> str | None:
        bse_code = self._get_bse_code(ticker)
        if not bse_code:
            return None

        filing_url = self._get_latest_annual_report_url(bse_code)
        if not filing_url:
            return None

        pdf_bytes = self._download_pdf(filing_url)
        text = self._extract_text(pdf_bytes, max_pages=self.max_pages)
        return text[:self.max_text_chars]

    def _extract_text(self, pdf_bytes: bytes, max_pages: int) -> str:
        """PyMuPDF text extraction from first N pages."""
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = min(max_pages, len(doc))
        return "\n".join(doc[i].get_text() for i in range(pages))
```

---

## 3. Compute Engine

### 3.1 Design

The compute engine is entirely **registry-driven**. Adding a metric requires:
1. One YAML entry in the appropriate `elements/*.yaml` file
2. One Python function that accepts `(data: dict, params: dict) → MetricResult`

No changes to engine, scorer, or report templates required.

### 3.2 Registry Structure

```
metrics/
├── registry.yaml                    # Element weights + global config
└── elements/
    ├── size.yaml                    # S element metrics
    ├── quality_business.yaml        # Q(biz) metrics
    ├── quality_management.yaml      # Q(mgmt) metrics
    ├── growth.yaml                  # G metrics
    ├── longevity.yaml               # L metrics
    ├── price.yaml                   # P metrics
    └── composite.yaml               # Derived: Quality-Growth Matrix
```

#### registry.yaml (master weights)

```yaml
# compute_engine/metrics/registry.yaml

element_weights:
  size: 0.10
  quality_business: 0.20
  quality_management: 0.10
  growth: 0.25
  longevity: 0.20
  price: 0.15

config:
  min_years_required: 3              # Minimum data points to score a metric
  missing_data_penalty: 0.5         # Apply 50% weight if data incomplete
```

#### Per-element YAML anatomy

```yaml
# elements/quality_business.yaml

element: quality_business

metrics:
  roce_avg:
    name: "RoCE 5-Year Average"
    module: "builtin.profitability"
    function: "compute_roce_avg"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      mode: "threshold"
      direction: "higher_is_better"
      thresholds: [5, 10, 15, 20, 25, 30]   # Maps to 0/2/4/6/8/10
      weight: 0.15
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"

  opm_avg:
    name: "Operating Profit Margin (5yr avg)"
    module: "builtin.profitability"
    function: "compute_opm_avg"
    inputs: ["financials"]
    params:
      years: 5
    scoring:
      mode: "threshold"
      direction: "higher_is_better"
      thresholds: [5, 8, 12, 15, 20, 25]
      weight: 0.10
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
```

**Scoring modes:**

| Mode | Description | Example |
|------|-------------|---------|
| `threshold` | Stepped thresholds → 0/2/4/6/8/10 | RoCE: [5,10,15,20,25,30] |
| `range_optimal` | Target range (too low or too high penalized) | D/E: optimal [0, 0.5] |
| `categorical` | String/enum → fixed score | B2C=10, B2B=7, B2G=4 |
| `trend_direction` | Improving trend scores higher | OPM improving 3yr+ |
| `comparison_to_actual` | Compare computed to reference | Reverse DCF vs actual CAGR |

### 3.3 ComputeEngine — Auto-Discovery

```python
# compute_engine/engine.py

class ComputeEngine:
    def __init__(self):
        self.registry = self._load_registry()
        self._validate_registry()

    def _load_registry(self) -> dict:
        """
        Auto-discover all YAML files in:
        - metrics/elements/*.yaml    (built-in metric definitions)
        - metrics/custom/*.yaml      (user drop-ins)

        Merge into unified {metric_id: config} dict.
        """
        metrics = {}
        for yaml_path in Path("metrics/elements").glob("*.yaml"):
            data = yaml.safe_load(yaml_path.read_text())
            element = data["element"]
            for metric_id, config in data["metrics"].items():
                config["element"] = element
                config["_source_file"] = yaml_path.name
                metrics[metric_id] = config

        # Custom metrics (same format, override element)
        for yaml_path in Path("metrics/custom").glob("*.yaml"):
            data = yaml.safe_load(yaml_path.read_text())
            element = data.get("element", "custom")
            for metric_id, config in data.get("metrics", {}).items():
                config["element"] = element
                config["_source_file"] = yaml_path.name
                metrics[metric_id] = config

        return {"metrics": metrics, "element_weights": self._load_element_weights()}

    def run_all(self, ticker: str, data: dict) -> dict[str, MetricResult]:
        """Run all registered metrics. Returns {metric_id: MetricResult}."""
        results = {}
        for metric_id, config in self.registry["metrics"].items():
            try:
                func = self._import_function(config["module"], config["function"])
                inputs = {k: data[k] for k in config.get("inputs", [])}
                params = config.get("params", {})
                results[metric_id] = func(inputs, params)
            except Exception as e:
                results[metric_id] = MetricResult(value=None, error=str(e))
        return results

    def _import_function(self, module: str, function: str):
        """Dynamically import compute function from module path."""
        mod = importlib.import_module(f"compute_engine.metrics.{module}")
        return getattr(mod, function)
```

### 3.4 MetricResult Contract

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

### 3.5 The Scorer

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
```

### 3.6 Adding a New Metric — Complete Workflow

**Step 1:** Add entry to the appropriate element file (e.g., `elements/quality_business.yaml`):

```yaml
  cash_conversion_cycle:
    name: "Cash Conversion Cycle (days)"
    module: "builtin.efficiency"
    function: "compute_ccc"
    inputs: ["financials"]
    scoring:
      thresholds: [120, 90, 60, 45, 30, 15]
      direction: "lower_is_better"
      weight: 0.05
    display: { format: "{:.0f} days", section: "quality_scorecard" }
```

**Step 2:** Write the function:

```python
# metrics/builtin/efficiency.py

from compute_engine.metrics.base import MetricResult

def compute_ccc(data: dict, params: dict) -> MetricResult:
    df = data["financials"]
    receivable_days = df["receivable_days"].iloc[-1]
    inventory_days = df["inventory_days"].iloc[-1]
    payable_days = df["payable_days"].iloc[-1]
    ccc = receivable_days + inventory_days - payable_days
    return MetricResult(
        value=ccc,
        flags=["negative_ccc_excellent"] if ccc < 0 else [],
    )
```

**Step 3:** Done. No other files need changes. The engine discovers it automatically.

### 3.7 Future Plug-in Metrics

| Metric | Module | Why Useful |
|--------|--------|------------|
| Piotroski F-Score | `custom/quality.py` | Aggregate financial health (0-9 score) |
| Altman Z-Score | `custom/risk.py` | Bankruptcy risk assessment |
| DuPont Decomposition | `builtin/profitability.py` | RoE = Margin × Turnover × Leverage |
| Sustainable Growth Rate | `builtin/growth.py` | RoE × (1 - Payout Ratio) |
| EVA (Economic Value Added) | `custom/quality.py` | NOPAT - (Capital × WACC) |
| Insider Buy/Sell Ratio | `custom/sentiment.py` | Management conviction signal |
| Benford's Law Analysis | `custom/forensics.py` | Accounting fraud detector |
| Tax Rate Consistency | `custom/forensics.py` | Aggressive accounting flag |

---

## 4. SQGLP Metrics — Detailed Specification

### 4.1 The SQGLP Framework

```
SQGLP Score = (S × 0.10) + (Q_biz × 0.20) + (Q_mgmt × 0.10) + (G × 0.25) + (L × 0.20) + (P × 0.15)
```

| Element | Weight | Rationale |
|---------|--------|-----------|
| **S** — Size | 10% | A filter more than a driver; small size enables discovery |
| **Q** — Quality of Business | 20% | Most important: high-RoCE, cash-generative businesses compound |
| **Q** — Quality of Management | 10% | Integrity + competence; partially quantitative, partially LLM |
| **G** — Growth | 25% | The superstructure: 4-lever decomposition reveals quality of growth |
| **L** — Longevity | 20% | Sustainability of quality + growth = moat durability |
| **P** — Price | 15% | Entry valuation matters, but less than business quality |

### 4.2 Growth: The 4-Lever Framework

The framework decomposes EPS growth into its constituent drivers using the chain rule:

```
ΔEPS = ΔVolume × (ΔSales/ΔVolume) × (ΔEBIT/ΔSales) × (ΔEPS/ΔEBIT)
       ────────   ─────────────────   ───────────────   ─────────────
       Volume     Price Lever          Operating Lever   Financial Lever
       Growth     (Pricing Power)      (Scale Benefits)  (Debt Amplification)
```

**Growth Quality Grading:**

| Quality Grade | Primary Drivers | Description |
|:---|:---|:---|
| **High Quality** | Volume + Operating Leverage | Selling more units, achieving economies of scale |
| **Moderate** | Volume + Price | Growing demand with pricing power |
| **Low Quality** | Financial Leverage + Price Hikes | Debt-driven or aggressive pricing; unsustainable |
| **Risky** | Financial Leverage dominant | Amplified returns in good times, accelerated losses in bad |

### 4.3 Valuation Models (All Computed Offline)

Three intrinsic value models, all running in `builtin/valuation.py`:

**DCF (Discounted Cash Flow):** Project 10yr FCF using historical growth, discount at WACC (12% default for Indian mid-caps), add terminal value at 4% perpetual growth.

**Earnings Power Value:** Normalize last 5yr earnings, divide by cost of equity. Represents value with zero growth — a floor estimate.

**Reverse DCF:** Solve for the growth rate the market is currently pricing in. Compare to actual historical growth. If market implies 30% growth but company has grown at 18%, the market may be overpricing it.

### 4.4 Computed Flags (Auto-Generated, No LLM)

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

---

## 5. Sector & Macro Context

### 5.1 Sector Tailwind Classification

Based on the Dec 2025 Wealth Creation Study's analysis of NTD era compounders, the system includes a sector context file used by both the screening engine and LLM prompts.

```yaml
# data_fetcher/sector_context.yaml

mtd_consideration_sectors:
  strong_tailwind:
    - "Banks - Private Sector"
    - "Capital Market"
    - "Finance"
    - "Insurance"
    - "Autos - Cars/UVs"
    - "Autos - 2-3 Wheelers"
    - "Capital Goods / Engg"
    - "Consumer - Durables"
    - "Healthcare"
    - "Realty"
    - "Telecom"
    - "E-commerce"
  moderate_tailwind:
    - "IT"
    - "Pharma"
    - "Cables"
    - "Airlines"
  non_consideration:
    - "Fertilizers"
    - "Oil & Gas"
    - "Metals & Mining"
    - "Power Generation"
    - "Sugar"
    - "Chemicals"

business_type_preference:
  # Dec 2025 finding: 60% of compounders were B2C
  preferred: "B2C"
  acceptable: "B2B"
  caution: "B2G"

market_leadership:
  # Dec 2025 finding: 77% of compounders were market leaders (top 3)
  preferred_rank: 3
```

### 5.2 Screening Presets

Pre-built screening configurations derived from the source studies:

```yaml
# metrics/presets/compounders.yaml
# Based on Motilal Oswal Dec 2025 compounder identification methodology
name: "Compounders (MOSL Dec 2025)"
description: |
  Top 500 → RoE > 12% → PAT CAGR 3yr > 20% → Trailing PEG < 2x
  In the NTD era, only 7% of top 500 stocks qualified as compounders.
filters:
  roe_5yr_avg: { min: 12 }
  pat_cagr_3yr: { min: 20 }
  trailing_peg: { max: 2.0 }
rankings:
  primary: "trailing_peg"
  secondary: "roe_5yr_avg"
```

```yaml
# metrics/presets/hidden_gems_100x.yaml
# Based on Motilal Oswal 2014 SQGLP 100x hunting criteria
name: "Hidden Gems - 100x (MOSL 2014)"
description: |
  Market cap < INR 30b, P/E < 25x, low institutional holding.
  Value migration or niche opportunity businesses.
filters:
  market_cap: { max: 30000 }
  pe_ttm: { max: 25 }
  institutional_holding: { max: 10 }
  analyst_coverage: { max: 10 }
rankings:
  primary: "sqglp_composite"
```

### 5.3 Quality-Growth Matrix

From the Dec 2025 report. A 2x2 classification used to quickly categorize any company:

```
                    QUALITY (RoCE > 15%)
                    Low              High
              ┌──────────────┬──────────────┐
         High │ GROWTH TRAP  │ TRUE WEALTH  │
GROWTH        │ Transitory   │ CREATOR      │
(PAT CAGR     │ multi-bagger │ Enduring     │
 > 15%)       ├──────────────┼──────────────┤
         Low  │ WEALTH       │ QUALITY TRAP │
              │ DESTROYER    │ Under-       │
              │ Capital loss │ performer    │
              └──────────────┴──────────────┘
```

Computed automatically as a metric in `elements/composite.yaml` and displayed prominently in the Executive Summary of every report.

---

## 6. LLM Analysis Layer (Stage 4)

### 6.1 Design Principle

Instead of feeding raw financial statements to the LLM (expensive, unfocused), we feed **pre-computed JSON with flags** and ask focused questions. This reduces input tokens by ~90%.

The **QGLP 25-Question Checklist** (from Dec 2025 Wealth Creation Study) structures the LLM's qualitative analysis. Of the 25 questions, ~15 are partially or fully answered by computed metrics. The LLM focuses on the remaining ~10 that require genuine qualitative judgment from annual report text.

### 6.2 QGLP Checklist Integration

```
llm_layer/
├── orchestrator.py              # 2-pass LLM orchestration
├── checklist.py                 # Maps QGLP Q1-Q25 to available data
└── prompts/
    ├── pass1_qualitative.txt    # Annual report qualitative analysis
    └── pass2_synthesis.txt      # Investment thesis synthesis
```

The checklist mapper (`checklist.py`) automatically determines which questions can be pre-answered:

| Questions | Category | Data Source | LLM Needed? |
|-----------|----------|-------------|-------------|
| Q3-Q6 | Business Quality | Computed metrics (RoCE, margins, DuPont, cash flow) | Partial — metrics provided, LLM interprets |
| Q11-Q14 | Management | Computed (promoter, pledge, dilution, tax rate) | Partial — numbers provided, LLM assesses |
| Q18-Q19 | Skin in Game | Computed (promoter holding, pledge %) | No — fully computed |
| Q20-Q24 | Price | Computed (P/E, PEG, DCF, liquidity) | No — fully computed |
| Q1-Q2, Q7-Q10, Q15-Q17, Q25 | Qualitative | Annual report text | **Yes — requires LLM** |

### 6.3 Two-Pass Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM CALL STRATEGY                            │
│                                                                 │
│  Pass 1: QUALITATIVE ANALYSIS (~2K input, ~1K output tokens)   │
│  ├── Model: Claude Sonnet (Opus with --deep)                   │
│  ├── Input: Annual report excerpts, promoter holding trend,     │
│  │          computed flags                                      │
│  ├── Task: Assess management quality, competitive moat,         │
│  │         business model risks, sector tailwinds               │
│  └── Output: Structured qualitative assessment JSON             │
│  Note: Skipped if no annual report available                    │
│                                                                 │
│  Pass 2: SYNTHESIS (~3K input, ~2K output tokens)               │
│  ├── Model: Claude Sonnet (Opus with --deep)                   │
│  ├── Input: All SQGLP computed metrics + flags + Pass 1 output │
│  ├── Task: Investment thesis, conviction, kill-the-thesis risks,│
│  │         what the market is missing                           │
│  └── Output: Investment thesis + risk assessment JSON           │
│                                                                 │
│  ESTIMATED COST: ~$0.03–$0.10 per company (Sonnet)             │
│  (vs ~$2–5 if sending raw data without pre-processing)          │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Prompt Templates

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

### 6.5 LLM Orchestrator

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
        annual_report_text: str | None = None,
    ) -> dict:
        results = {}

        # Pass 1: Qualitative (skip if no annual report)
        if annual_report_text and not self.config["llm"].get("skip_pass1_if_no_ar", True):
            results["pass1"] = self._call(
                model=self.config["llm"]["pass1_model"],
                prompt_template="pass1_qualitative.txt",
                context={...},
            )
        elif annual_report_text:
            results["pass1"] = self._call(
                model=self.config["llm"]["pass1_model"],
                prompt_template="pass1_qualitative.txt",
                context={...},
            )
        else:
            results["pass1"] = {"skipped": True, "reason": "No annual report available"}

        # Pass 2: Synthesis (always runs)
        results["pass2"] = self._call(
            model=self.config["llm"]["pass2_model"],
            prompt_template="pass2_synthesis.txt",
            context={
                "sqglp_metrics": sqglp_metrics,
                "scores": scores,
                "pass1": results["pass1"],
            },
        )

        return results

    def _call(self, model: str, prompt_template: str, context: dict) -> dict:
        """Load template, render with context, call API, parse JSON response."""
        template = self._load_template(prompt_template)
        prompt = template.format(**context)

        response = self.client.messages.create(
            model=model,
            max_tokens=self.config["llm"]["max_tokens"],
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_json_response(response.content[0].text)
```

### 6.6 Cost Optimization Strategies

| Strategy | Savings | How |
|----------|---------|-----|
| **Skip Pass 1 if no AR** | Save one full call | Only run when annual report PDF is available |
| **Structured JSON output** | Fewer output tokens | JSON schema forces concise responses |
| **Pre-computed flags** | Reduce LLM reasoning load | Flags like `consistently_high_roce` mean LLM doesn't re-derive them |
| **Cap annual report text** | Control Pass 1 input tokens | `max_text_chars: 5000` in config |

---

## 7. Output & Report Generation

### 7.1 Output Files

```
reports/
└── {TICKER}_{DATE}/
    ├── sqglp_report.html      # Self-contained interactive HTML (Plotly charts)
    ├── sqglp_report.md        # Markdown summary for quick reading
    ├── raw_metrics.json       # All computed metric results
    ├── llm_analysis.json      # Both LLM pass outputs
    └── scores.json            # SQGLP element scores + composite
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
9. **Investment Thesis** — Bull/bear case, kill-the-thesis scenarios (LLM only)
10. **Risk Register** — Red flags, computed warnings, LLM-identified risks
11. **Monitorables Checklist** — What to track quarterly
12. **Appendix** — Raw data tables, methodology, data sources

### 7.3 HTML Dashboard

The primary output format. Uses Jinja2 templates with embedded Plotly charts to produce a self-contained `.html` file with interactive visualizations.

```python
# output/report_generator.py

class ReportGenerator:
    def __init__(self, template_dir: str = "output/templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate(self, result: "AnalysisResult", formats: list[str]) -> None:
        if "html" in formats:
            html = self._render_html(result)
            self._write(result, "sqglp_report.html", html)
        if "md" in formats:
            md = self._render_markdown(result)
            self._write(result, "sqglp_report.md", md)
        if "json" in formats:
            self._write_json(result)

    def _render_html(self, result) -> str:
        template = self.env.get_template("sqglp_report.html.j2")
        return template.render(
            ticker=result.ticker,
            scores=result.scores,
            metrics=result.metrics,
            flags=result.flags,
            radar_chart=self._radar_chart(result.scores),
            growth_chart=self._growth_decomposition_chart(result),
            pe_band_chart=self._pe_band_chart(result.metrics),
            llm_analysis=result.llm_analysis,
            generation_date=datetime.now().isoformat(),
        )

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

---

## 8. Service Layer — The GUI-Ready API

### 8.1 Purpose

`service.py` is the central orchestrator that the CLI calls today, and a future GUI would call tomorrow. **All business logic lives here** — not in CLI scripts, not in GUI routes.

### 8.2 Interface

```python
# service.py

from dataclasses import dataclass

@dataclass
class AnalysisResult:
    ticker: str                    # Ticker symbol
    data: dict                     # Raw fetched data
    metrics: dict                  # All MetricResult objects
    scores: dict                   # SQGLP element + composite scores
    growth_decomposition: dict     # 4-lever breakdown + quality grade
    llm_analysis: dict | None      # 2-pass LLM output (None if --no-llm)
    errors: list[str]              # Non-fatal errors during pipeline

class Boundless100xService:
    """
    The single API for all research operations.
    CLI calls it. Future Streamlit/FastAPI calls it.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.fetcher_suite = FetcherSuite(self.config)
        self.engine = ComputeEngine()
        self.scorer = SQGLPScorer(self.engine.registry)
        self.llm = LLMOrchestrator(api_key=os.environ["ANTHROPIC_API_KEY"], config=self.config)
        self.reporter = ReportGenerator()

    def analyze(
        self,
        ticker: str,
        use_llm: bool = True,
    ) -> AnalysisResult:
        """Full pipeline: fetch → compute → score → LLM → report."""
        errors = []

        # Stage 1: Fetch data
        data = self.fetcher_suite.fetch_all(ticker)

        # Stage 2: Compute metrics
        metrics = self.engine.run_all(ticker, data)

        # Stage 3: Score + growth decomposition
        scores = self.scorer.score(metrics)
        growth_decomposition = self._compute_growth_decomposition(metrics)

        # Stage 4: LLM analysis (optional)
        llm_analysis = None
        if use_llm:
            llm_analysis = self.llm.run_analysis(
                sqglp_metrics=self._metrics_to_json(metrics),
                scores=scores,
                annual_report_text=data.get("annual_report_text"),
            )

        result = AnalysisResult(
            ticker=ticker,
            data=data,
            metrics=metrics,
            scores=scores,
            growth_decomposition=growth_decomposition,
            llm_analysis=llm_analysis,
            errors=errors,
        )

        # Report generation
        self.reporter.generate(result, formats=self.config["output"]["formats"])

        return result

    def screen_universe(self, filters: dict) -> list[dict]:
        """
        Filter NSE universe by criteria.
        Uses registry-defined metrics as filter dimensions.
        e.g., filters={"roce_5yr_avg": {"min": 15}, "market_cap": {"max": 50000}}
        """
        ...

    def get_watchlist(self) -> list[dict]:
        """Return current watchlist with latest SQGLP scores."""
        ...
```

---

## 9. Project Structure

```
boundless100x/
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
│   ├── fetch_price_volume.py            # yfinance → OHLCV
│   ├── fetch_corporate_actions.py       # BSE → splits, bonuses, dividends
│   ├── fetch_analyst_coverage.py        # Trendlyne → analyst count
│   ├── download_annual_reports.py       # BSE → annual report PDFs
│   ├── sector_context.yaml              # Tailwind sectors, B2C preference
│   └── raw_data/                        # Cached raw data per ticker
│       └── {TICKER}/
│           ├── financials.csv
│           ├── balance_sheet.csv
│           ├── cashflow.csv
│           ├── shareholding.csv
│           ├── price_volume.csv
│           └── annual_reports/*.pdf
│
├── compute_engine/                      # ── STAGE 2+3: Offline Computation ──
│   ├── __init__.py
│   ├── metrics/
│   │   ├── registry.yaml                # Master: element weights + global config
│   │   ├── elements/                    # Per-element metric definitions
│   │   │   ├── size.yaml                # S: market cap, institutional, turnover
│   │   │   ├── quality_business.yaml    # Q(biz): RoCE, margins, DuPont, cash conv.
│   │   │   ├── quality_management.yaml  # Q(mgmt): promoter, pledge, owner-operator
│   │   │   ├── growth.yaml              # G: CAGR, 4-lever, quality grade
│   │   │   ├── longevity.yaml           # L: CAP proxy, streaks, stability
│   │   │   ├── price.yaml               # P: PE, PEG, trailing PEG, DCF
│   │   │   └── composite.yaml           # Quality-Growth Matrix
│   │   ├── custom/                      # User drop-in metrics
│   │   ├── presets/                     # Screening presets
│   │   │   ├── compounders.yaml         # Dec 2025 methodology
│   │   │   └── hidden_gems_100x.yaml    # 2014 SQGLP 100x criteria
│   │   ├── base.py                      # MetricResult dataclass
│   │   ├── validator.py                 # YAML schema validation on startup
│   │   └── builtin/                     # Metric implementation modules
│   │       ├── __init__.py
│   │       ├── _helpers.py              # MAD-based FCF outlier detection
│   │       ├── profitability.py         # RoCE, RoE, OPM, Cash Conversion, DuPont
│   │       ├── growth.py               # CAGR, 4-lever, quality grade, consistency
│   │       ├── valuation.py            # P/E, PEG, trailing PEG, DCF, reverse DCF
│   │       ├── leverage.py             # D/E, Interest Coverage
│   │       ├── efficiency.py           # Working Capital Days, CCC, Asset Turnover
│   │       ├── size.py                 # Market cap, institutional, owner-operator
│   │       ├── longevity.py            # Consistency, streaks, CAP proxy
│   │       └── composite.py            # Quality-Growth Matrix classification
│   ├── engine.py                        # Auto-discovery metric runner
│   ├── scorer.py                        # SQGLP scoring from registry weights
│   └── screener.py                      # Preset-based universe screening
│
├── llm_layer/                           # ── STAGE 4: LLM Analysis ──
│   ├── __init__.py
│   ├── orchestrator.py                  # 2-pass LLM orchestration
│   ├── checklist.py                     # Maps QGLP Q1-Q25 to available data
│   └── prompts/
│       ├── pass1_qualitative.txt        # Annual report deep dive
│       └── pass2_synthesis.txt          # Investment thesis generation
│
├── output/                              # ── Report Generation ──
│   ├── __init__.py
│   ├── report_generator.py              # Jinja2 + Plotly HTML/MD generation
│   ├── templates/
│   │   ├── sqglp_report.html.j2         # Interactive HTML dashboard
│   │   └── sqglp_report.md.j2           # Markdown report
│   └── reports/                         # Generated reports
│       └── {TICKER}_{DATE}/
│           ├── sqglp_report.html
│           ├── sqglp_report.md
│           ├── raw_metrics.json
│           ├── llm_analysis.json
│           └── scores.json
│
├── service.py                           # Central API layer (GUI-ready)
├── cli.py                               # Command-line interface (typer)
├── watchlist.py                         # Watchlist management
├── watchlist.json                       # Persisted watchlist data
├── requirements.txt                     # Python dependencies
├── .env                                 # API keys (ANTHROPIC_API_KEY)
└── README.md
```

---

## 10. Configuration Reference

```yaml
# config.yaml — Complete configuration

# ── Target Company ──
target:
  ticker: "ASTRAL"
  bse_code: "532830"
  nse_symbol: "ASTRAL"

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

# ── Annual Reports ──
annual_reports:
  enabled: true
  max_reports: 1                     # Only need the most recent report
  max_pages: 30                      # Extract text from first N pages
  max_text_chars: 5000               # Cap extracted text length

# ── LLM Configuration ──
llm:
  provider: "anthropic"
  pass1_model: "claude-sonnet-4-6"
  pass2_model: "claude-sonnet-4-6"
  max_tokens: 4096
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

---

## 11. Technology Stack

| Component | Technology | Cost |
|-----------|------------|------|
| **Language** | Python 3.11+ | Free |
| **Data Fetching** | requests, beautifulsoup4, yfinance | Free |
| **Data Storage** | CSV/JSON files (SQLite if scaling to 500+ companies) | Free |
| **Computation** | pandas, numpy, scipy | Free |
| **PDF Extraction** | PyMuPDF (fitz) | Free |
| **Visualization** | Plotly (embedded in HTML) | Free |
| **LLM Analysis** | Claude API (Sonnet or Opus) | ~$0.03–$0.10/company |
| **Report Generation** | Jinja2 templates → HTML/Markdown | Free |
| **CLI Framework** | typer | Free |
| **Config** | PyYAML | Free |
| **API Client** | anthropic Python SDK | Free |

### Python Dependencies (`requirements.txt`)

```
# Data fetching
requests>=2.31
beautifulsoup4>=4.12
yfinance>=0.2

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
python-dotenv>=1.0

# Dev
pytest>=7.4
```

---

## 12. CLI Reference

```bash
# ── Full Analysis ──
# Fetch data, compute all metrics, run LLM, generate reports
python -m boundless100x analyze ASTRAL

# Skip LLM passes (compute only)
python -m boundless100x analyze ASTRAL --no-llm

# Use Opus instead of Sonnet for deeper analysis
python -m boundless100x analyze ASTRAL --deep

# ── Compute Only (no LLM, no reports) ──
python -m boundless100x compute ASTRAL            # Outputs JSON metrics

# ── Screening ──
python -m boundless100x screen --min-roce 15 --max-mcap 50000 --min-rev-cagr 15
python -m boundless100x screen --preset compounders       # Motilal Oswal Dec 2025 methodology
python -m boundless100x screen --preset hidden_gems_100x  # SQGLP 2014 100x hunting

# ── Watchlist ──
python -m boundless100x watchlist show
python -m boundless100x watchlist add ASTRAL
python -m boundless100x watchlist remove ASTRAL
python -m boundless100x watchlist update          # Re-run analysis on all
python -m boundless100x watchlist update --quarterly  # Only if last run > 90 days ago
```

---

## 13. Implementation Roadmap

### Phase 1: Foundation
- [x] Initialize project structure, `config.yaml`, `requirements.txt`
- [x] Implement `base.py` (BaseFetcher with caching & rate limiting)
- [x] Implement `fetch_financials.py` (Screener.in → 10yr P&L, BS, CF)
- [x] Implement `fetch_price_volume.py` (yfinance)
- [x] Implement `fetch_shareholding.py` (BSE quarterly)
- [x] Validation: Fetch data for 3 known companies (Astral, Bajaj Finance, TCS)

### Phase 2: Compute Engine
- [x] Implement `metrics/base.py` (MetricResult dataclass)
- [x] Create `registry.yaml` with 44 metrics
- [x] Implement `engine.py` (generic registry-driven metric runner)
- [x] Implement all `builtin/*.py` modules
- [x] Implement `scorer.py` (SQGLP scoring from registry weights)
- [x] Validation: Run on 3 companies, compare scores to manual analysis

### Phase 3: LLM Integration
- [x] Design and test Pass 1 prompt (qualitative analysis)
- [x] Design and test Pass 2 prompt (synthesis + thesis)
- [x] Implement `orchestrator.py` with retry logic + JSON parsing
- [x] Implement annual report PDF text extraction (PyMuPDF)
- [x] Implement `service.py` (central API orchestrating all stages)
- [x] Validation: End-to-end pipeline on 5 companies

### Phase 4: Reporting & CLI
- [x] Implement `report_generator.py` with Plotly chart functions
- [x] Build `sqglp_report.html.j2` (interactive HTML dashboard)
- [x] Build `sqglp_report.md.j2` (markdown summary)
- [x] Implement `cli.py` (analyze, compute, screen, watchlist commands)
- [x] Validation: Generate reports for 5 companies

### Phase 5: Screening & Watchlist (In Progress)
- [ ] Build universe screener using registry-defined metric filters
- [ ] Implement watchlist management (add/remove/show/update)
- [ ] Implement quarterly update logic
- [ ] Shortlist top 20 candidates → run full pipeline

---

## 14. Key Design Decisions & Rationale

**Why Screener.in instead of raw BSE XBRL filings?**
Screener.in normalizes financial data across companies, handles consolidated vs standalone, and provides 10-year history in a clean format. Parsing raw XBRL is 10x more effort for the same data. The tradeoff is dependency on a third-party site — mitigated by caching and fallback sources.

**Why JSON intermediate format between stages?**
JSON acts as a contract between compute and LLM layers. You can inspect, validate, and version-control computed metrics independently. If LLM pricing changes, swap models without touching computation. If a metric formula is wrong, fix it without re-running LLM calls. This also enables running compute-only mode (`--no-llm`) for quick screening.

**Why 2 LLM passes instead of 1 big call?**
Each pass has a focused scope with a specific output schema, producing more reliable and structured output. A single large prompt with all data tends to produce generic analysis. Splitting also lets you skip Pass 1 when annual reports aren't available without affecting the core investment thesis synthesis.

**Why absolute assessment rather than peer comparison?**
The SQGLP framework answers "Is this a long-term compounder?" — an absolute question. A mediocre company in a weak sector is not investable just because it outperforms its peers. Peer comparison would add significant complexity (5 extra compute runs, peer discovery logic, additional LLM call) without improving the fundamental investment decision. The 44 metrics already capture every dimension relevant to long-term compounding: quality, growth, moat, valuation.

**Why a metric registry instead of hardcoded modules?**
Adding a metric should be a 5-minute task (YAML entry + function), not a multi-file refactor. The registry pattern also makes scoring and report generation self-configuring — they read the registry and auto-adapt.

**Why split the registry into per-element YAML files?**
A monolithic registry with 40+ metrics becomes error-prone, creates merge conflicts, and is cognitively overwhelming. Per-element files are focused (~80 lines each), independently editable, and validated on startup. Custom metrics drop into `custom/` with zero changes to existing files.

**Why the QGLP 25-question checklist in the LLM layer?**
The Motilal Oswal Dec 2025 report provides a professionally-tested assessment framework. Instead of open-ended LLM prompts that produce variable output, the structured checklist ensures systematic coverage of all quality dimensions. The key insight: ~15 of 25 questions are pre-answered by computed metrics, so the LLM only does genuine qualitative judgment on the remaining ~10 questions.

**Why not a database?**
For personal research on 20-50 companies, CSV/JSON files are simpler to inspect, debug, and version-control with git. If you scale to 500+ companies or need time-series queries, migrate to SQLite (one-file database, zero setup) or PostgreSQL.

**Why the service layer even without a GUI?**
It enforces clean separation. Without it, business logic leaks into CLI argument parsing, making it impossible to reuse. With it, adding a Streamlit frontend is literally `result = service.analyze("ASTRAL")` — zero refactoring of the pipeline.

---

## 15. Future Work: Interactive GUI

### 15.1 Why Defer

The GUI is deliberately deferred for three reasons:

1. **Pipeline instability:** Data sources, metric definitions, scoring weights, and LLM prompts will all evolve significantly in the first 2-3 months. GUI changes during this period are wasted effort.
2. **Value concentration:** The quality of research depends on correct RoCE calculations and good LLM prompts, not on having a button to click. CLI + HTML reports serve the analytical workflow well.
3. **Scope explosion:** A useful GUI is not just "display the report" — it's interactive filtering, drill-downs, watchlist management, alerts, and charting. That's a 4-6 week project that doesn't improve the analytical output.

### 15.2 Progression Path

```
Phase 1-4 (Done)             Phase 5-6 (Now)                  Phase 7+ (Future)
────────────────────         ──────────────────────────        ──────────────────────
CLI + JSON output        →   Static HTML dashboards       →    Interactive Web GUI
                             (Jinja2 + Plotly charts)          (Streamlit or Dash)
                             ↑                                 ↑
                             Might be sufficient               Only if analyzing 50+
                             indefinitely for personal         companies, sharing with
                             research                          others, or wanting alerts
```

### 15.3 GUI-Readiness Already Built In

The `service.py` layer ensures the backend is GUI-ready from day one. A future GUI would call the exact same methods the CLI calls:

```python
# Future: app_streamlit.py (entire GUI wrapper)

import streamlit as st
from boundless100x.service import Boundless100xService

service = Boundless100xService()

st.title("SQGLP Research Dashboard")

ticker = st.text_input("Enter Ticker", "ASTRAL")
if st.button("Analyze"):
    with st.spinner("Running pipeline..."):
        result = service.analyze(ticker)

    # SQGLP Radar
    st.plotly_chart(create_radar(result.scores))

    # Metrics table
    st.dataframe(metrics_to_df(result.metrics))

    # LLM Thesis
    if result.llm_analysis:
        st.markdown(f"**Thesis:** {result.llm_analysis['pass2']['thesis']}")
        st.markdown(f"**Conviction:** {result.llm_analysis['pass2']['conviction_level']}")
```

### 15.4 Technology Options When Ready

| Option | Best For | Effort | Tradeoffs |
|--------|----------|--------|-----------|
| **Static HTML (Current)** | Personal research, email sharing | Already built | No interactivity beyond Plotly charts |
| **Streamlit** | Quick personal dashboard | 1-2 weeks | Pythonic, fast. Reruns on every interaction. Fine for personal use. |
| **Plotly Dash** | Chart-heavy financial dashboard | 2-3 weeks | Better charting control. More code but more customizable. |
| **FastAPI + React** | Multi-user product, polished UX | 5-8 weeks | Full separation of concerns. Only justified if sharing as a tool with others. |

### 15.5 GUI Feature Roadmap (When Implemented)

**Phase A — Dashboard:**
- Single-company analysis view with SQGLP radar
- Metric tables with conditional formatting (green/amber/red)
- Plotly charts: P/E band, RoCE trend, growth decomposition
- LLM thesis display

**Phase B — Screening & Watchlist:**
- Interactive screener with filter sliders (RoCE, growth, P/E, etc.)
- Watchlist management (add/remove, track score changes over time)
- Score change alerts (notify when a watchlist company's score changes significantly)

**Phase C — Research Workflow:**
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
