# Financial Research System — Design Enhancements (v2)

Addendum to the base design, addressing three areas: automated competitor identification, extensible metric architecture, and the GUI question.

------

## Enhancement 1: Automated Competitor Identification

The base design assumed you'd manually specify peers in `config.yaml`. That's a bottleneck. Here's a multi-layer approach to automate it.

### 1.1 The Problem

"Competitors" is not a single concept. Consider Astral Ltd (pipes & fittings):

- **Direct product competitors:** Supreme Industries, Prince Pipes, Finolex Industries — they make similar pipes
- **Sector peers by classification:** All BSE/NSE "Building Materials" companies — useful for valuation benchmarking
- **Financial peers:** Companies with similar RoCE, margin profile, growth rate — could be from completely different sectors but useful for "what's a fair P/E for this quality level?"
- **Value chain adjacents:** Pidilite (adhesives), APL Apollo (steel tubes) — not direct competitors but affected by same housing cycle

A good research system needs at least the first two automatically, and the third as a bonus.

### 1.2 Multi-Layer Peer Discovery

```
┌──────────────────────────────────────────────────────────┐
│              PEER DISCOVERY PIPELINE                      │
│                                                          │
│  Layer 1: Industry Classification (FREE, deterministic)  │
│  ├── BSE/NSE sector + sub-sector classification          │
│  ├── Screener.in "sector" page → all companies in sector │
│  └── Output: 15-40 raw sector peers                      │
│                                                          │
│  Layer 2: Size-Filtered Cohort (FREE, computed)          │
│  ├── Filter Layer 1 by market cap band (0.3x to 3x)     │
│  ├── Filter by revenue band (0.2x to 5x)                │
│  └── Output: 5-15 size-matched peers                     │
│                                                          │
│  Layer 3: Financial Similarity (FREE, computed)          │
│  ├── Compute distance on key metrics:                    │
│  │   [RoCE, OPM, Revenue CAGR, Debt/Equity, P/E]       │
│  ├── Normalized euclidean distance or cosine similarity  │
│  └── Output: Top 5 most financially similar companies    │
│                                                          │
│  Layer 4: Smart Peer Validation (LLM, optional)          │
│  ├── Input: Layer 2 + Layer 3 candidates + company desc  │
│  ├── Task: "Which of these are true competitors?"        │
│  └── Output: Final 4-6 validated peers                   │
│                                                          │
│  Layer 5: Value Chain Mapping (LLM, optional)            │
│  ├── Input: Company business description                 │
│  ├── Task: "Who are upstream/downstream players?"        │
│  └── Output: Adjacent companies to monitor               │
└──────────────────────────────────────────────────────────┘
```

### 1.3 Implementation: `peer_discovery.py`

```python
# peer_discovery.py — Automated competitor identification

class PeerDiscovery:
    """
    Multi-layer peer identification system.
    Layers 1-3 are fully offline (no LLM cost).
    Layer 4-5 are optional LLM-assisted refinement.
    """

    def discover(self, ticker: str, use_llm: bool = False) -> PeerResult:
        # Layer 1: Sector classification
        sector_peers = self._get_sector_peers(ticker)

        # Layer 2: Size filtering
        size_filtered = self._filter_by_size(ticker, sector_peers)

        # Layer 3: Financial similarity scoring
        similarity_ranked = self._rank_by_financial_similarity(
            ticker, size_filtered
        )

        # Layer 4 (optional): LLM validation
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
        )
```

### 1.4 Sector Peer Source: Screener.in

Screener.in groups companies by sector. For any company page, you can extract:

- The sector/industry classification
- The "Peers" section which already lists comparable companies
- The sector page with all companies and their key metrics

```python
def _get_sector_peers(self, ticker: str) -> list[str]:
    """
    Approach 1: Scrape Screener.in company page → extract 'Peer Comparison' table
    Approach 2: Scrape sector page → get all companies in same industry
    Approach 3: Use BSE sector classification API

    Fallback: Moneycontrol 'Peers' tab
    """
    # Screener.in: https://www.screener.in/company/{TICKER}/
    # → "Peer Comparison" section lists 5-10 peers with ratios
    # → "Industry" link leads to full sector page

    # BSE: https://www.bseindia.com/corporates/List_Scrips.html
    # → Filter by industry group
```

### 1.5 Financial Similarity Scoring

This is the most interesting layer — it finds companies that "look like" your target financially, even across sectors. This is useful for valuation benchmarking ("what P/E do the market assign to companies with this quality profile?").

```python
def _rank_by_financial_similarity(self, ticker, candidates):
    """
    Compute normalized distance across key financial dimensions.

    Metrics used for similarity:
    - RoCE (5yr avg)           — quality proxy
    - Operating Margin (5yr)   — business model proxy
    - Revenue CAGR (5yr)       — growth proxy
    - Debt/Equity              — risk proxy
    - Market Cap (log-scaled)  — size proxy

    Each metric is z-score normalized, then euclidean distance computed.
    Closest companies = most similar financial profile.
    """
    SIMILARITY_METRICS = [
        'roce_5yr_avg',
        'operating_margin_5yr_avg',
        'revenue_cagr_5yr',
        'debt_equity',
        'log_market_cap',
    ]

    # Normalize, compute distance, rank
    target_vector = get_metrics(ticker, SIMILARITY_METRICS)
    distances = {}
    for candidate in candidates:
        cand_vector = get_metrics(candidate, SIMILARITY_METRICS)
        distances[candidate] = euclidean_distance(
            normalize(target_vector),
            normalize(cand_vector)
        )

    return sorted(distances, key=distances.get)
```

### 1.6 Updated `config.yaml`

Peers become optional. If not specified, auto-discovery kicks in:

```yaml
target:
  ticker: "ASTRAL"
  nse_symbol: "ASTRAL"

# Optional: manually specify peers (overrides auto-discovery)
# peers:
#   - { ticker: "SUPREMEIND" }
#   - { ticker: "PRINCEPIPE" }

peer_discovery:
  enabled: true                    # Auto-discover if peers not specified
  use_llm_validation: false        # Layer 4-5 (costs ~$0.02)
  max_peers: 5                     # Final peer count
  size_band_multiplier: 3.0        # 0.3x to 3x market cap
  sector_source: "screener"        # "screener" | "bse" | "moneycontrol"
  include_financial_peers: true    # Cross-sector similarity matches
```

### 1.7 Updated CLI

```bash
# Auto-discover peers
python run_analysis.py --ticker ASTRAL

# Auto-discover + LLM validation
python run_analysis.py --ticker ASTRAL --llm-peers

# Still works: manual override
python run_analysis.py --ticker ASTRAL --peers SUPREMEIND,PRINCEPIPE
```

------

## Enhancement 2: Extensible Metric Architecture

The current design has metrics hardcoded in each SQGLP module. That's fine for v1 but becomes painful when you want to add a new ratio — you'd have to modify the compute function, the output JSON schema, the scoring logic, and the report template.

### 2.1 The Problem

Today, adding "Cash Conversion Cycle trend" means editing:

1. `q_quality_business.py` — add computation
2. `scoring.py` — add to scoring formula
3. Report template — add display row
4. Peer comparison — add column

That's 4 files for one metric. It should be 1 configuration change.

### 2.2 Solution: Metric Registry Pattern

Every metric is defined as a declarative config entry. The compute engine reads the registry and runs whatever metrics are registered.

```
compute_engine/
├── metrics/
│   ├── registry.yaml          # THE source of truth for all metrics
│   ├── base.py                # MetricDefinition class
│   ├── builtin/               # Built-in metric implementations
│   │   ├── profitability.py   # RoCE, RoE, OPM, NPM, etc.
│   │   ├── growth.py          # CAGR, 4-lever decomposition
│   │   ├── valuation.py       # P/E, PEG, EV/EBITDA, DCF
│   │   ├── leverage.py        # D/E, Interest Coverage, etc.
│   │   ├── efficiency.py      # Working capital, asset turnover
│   │   ├── size.py            # Market cap, institutional holding
│   │   └── longevity.py       # Consistency scores, streaks
│   └── custom/                # YOUR custom metrics (drop-in)
│       └── my_metrics.py
├── engine.py                  # Reads registry, runs all metrics
└── scorer.py                  # Reads registry weights, computes scores
```

### 2.3 `registry.yaml` — The Single Source of Truth

```yaml
# registry.yaml — Every metric is defined here
# To add a new metric: add an entry here + implement the function

metrics:

  # ──────────────────── QUALITY: BUSINESS ────────────────────
  roce_5yr_avg:
    name: "RoCE (5yr Avg)"
    element: "quality_business"          # SQGLP element
    module: "builtin.profitability"      # Python module path
    function: "compute_roce_avg"         # Function to call
    inputs: ["financials"]               # Which data sources needed
    params:
      years: 5
    scoring:
      thresholds: [5, 10, 15, 20, 25, 30]   # Maps to scores 1-7+
      direction: "higher_is_better"
      weight: 0.15                           # Weight within element
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
      peer_compare: true

  roce_consistency:
    name: "RoCE > 15% Count (10yr)"
    element: "longevity"
    module: "builtin.longevity"
    function: "compute_roce_consistency"
    inputs: ["financials"]
    params:
      years: 10
      threshold: 15
    scoring:
      thresholds: [3, 5, 6, 7, 8, 9]
      direction: "higher_is_better"
      weight: 0.20
    display:
      format: "{}/10 years"
      section: "longevity_assessment"
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
      weight: 0.10
    display:
      format: "{:.1f}%"
      section: "quality_scorecard"
      peer_compare: true

  revenue_cagr_5yr:
    name: "Revenue CAGR (5yr)"
    element: "growth"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params:
      field: "revenue"
      years: 5
    scoring:
      thresholds: [5, 8, 12, 18, 25, 35]
      direction: "higher_is_better"
      weight: 0.12
    display:
      format: "{:.1f}%"
      section: "growth_decomposition"
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
      weight: 0.08
    display:
      format: "{:.2f}x"
      section: "quality_scorecard"
      peer_compare: true

  pe_current:
    name: "P/E (TTM)"
    element: "price"
    module: "builtin.valuation"
    function: "compute_pe_ttm"
    inputs: ["financials", "price"]
    scoring:
      # P/E scoring is relative — uses sector percentile, not absolute
      mode: "sector_relative_percentile"
      direction: "lower_is_better"
      weight: 0.10
    display:
      format: "{:.1f}x"
      section: "valuation_analysis"
      peer_compare: true

  # ... (50+ more metrics follow the same pattern)

  # ──────────────────── CUSTOM EXAMPLE ────────────────────
  # Adding a new metric is just this entry + one function:
  cash_conversion_cycle:
    name: "Cash Conversion Cycle (days)"
    element: "quality_business"
    module: "custom.my_metrics"       # Your custom file
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


# ──────────────────── ELEMENT WEIGHTS ────────────────────
# These control the SQGLP composite score
element_weights:
  size: 0.10
  quality_business: 0.20
  quality_management: 0.10
  growth: 0.25
  longevity: 0.20
  price: 0.15
```

### 2.4 Metric Implementation Contract

Every metric function follows the same interface:

```python
# builtin/profitability.py

def compute_roce_avg(data: dict, params: dict) -> MetricResult:
    """
    Every metric function receives:
      - data: dict of DataFrames keyed by input type
              e.g., {"financials": pd.DataFrame, "price": pd.DataFrame}
      - params: dict from registry.yaml params section

    Must return:
      MetricResult(
          value=22.5,                # The computed number
          raw_series=[18, 20, 22, 25, 27],  # Optional: yearly values
          flags=["consistently_high"],       # Optional: qualitative flags
          metadata={"years_used": 5}         # Optional: debug info
      )
    """
    df = data["financials"]
    years = params.get("years", 5)
    roce_values = df["roce"].tail(years)

    return MetricResult(
        value=roce_values.mean(),
        raw_series=roce_values.tolist(),
        flags=["consistently_high"] if (roce_values > 15).all() else [],
    )
```

### 2.5 The Compute Engine

```python
# engine.py — Generic engine that runs any registered metric

class ComputeEngine:
    def __init__(self, registry_path="metrics/registry.yaml"):
        self.registry = load_yaml(registry_path)
        self.metrics = self.registry["metrics"]

    def run_all(self, ticker: str, data: dict) -> dict:
        """Run all registered metrics for a company."""
        results = {}
        for metric_id, config in self.metrics.items():
            # Dynamically import the function
            module = import_module(f"metrics.{config['module']}")
            func = getattr(module, config["function"])

            # Check required data is available
            required = set(config.get("inputs", []))
            available = set(data.keys())
            if not required.issubset(available):
                results[metric_id] = MetricResult(
                    value=None,
                    flags=["data_unavailable"]
                )
                continue

            # Compute
            results[metric_id] = func(data, config.get("params", {}))

        return results

    def run_element(self, element: str, ticker: str, data: dict) -> dict:
        """Run only metrics belonging to a specific SQGLP element."""
        element_metrics = {
            k: v for k, v in self.metrics.items()
            if v["element"] == element
        }
        # ... same logic as run_all but filtered

    def score(self, results: dict) -> dict:
        """Compute SQGLP scores from metric results using registry weights."""
        element_scores = {}
        for metric_id, result in results.items():
            config = self.metrics[metric_id]
            element = config["element"]
            score = self._apply_threshold(
                result.value,
                config["scoring"]["thresholds"],
                config["scoring"]["direction"]
            )
            weighted = score * config["scoring"]["weight"]
            element_scores.setdefault(element, []).append(weighted)

        # Aggregate per element, then weighted composite
        sqglp = {}
        for element, scores in element_scores.items():
            sqglp[element] = sum(scores) / sum(
                self.metrics[m]["scoring"]["weight"]
                for m in self.metrics
                if self.metrics[m]["element"] == element
                and results.get(m) and results[m].value is not None
            ) * 10  # Scale to 0-10

        composite = sum(
            sqglp.get(el, 0) * w
            for el, w in self.registry["element_weights"].items()
        )

        return {"elements": sqglp, "composite": composite}
```

### 2.6 Adding a New Metric: The Workflow

**Step 1:** Add entry to `registry.yaml` (the YAML block above)

**Step 2:** Write the function (if it doesn't exist):

```python
# custom/my_metrics.py

def compute_ccc(data: dict, params: dict) -> MetricResult:
    df = data["financials"]
    receivable_days = df["receivable_days"].iloc[-1]
    inventory_days = df["inventory_days"].iloc[-1]
    payable_days = df["payable_days"].iloc[-1]
    ccc = receivable_days + inventory_days - payable_days
    return MetricResult(value=ccc)
```

**Step 3:** Done. The engine auto-discovers it, scoring includes it, reports display it, peer comparison includes it. Zero changes to engine, scorer, or report templates.

### 2.7 Future Metrics You Might Add

Here are some metrics that are easy to plug into this architecture later:

| Metric                      | Module                     | Why useful                         |
| --------------------------- | -------------------------- | ---------------------------------- |
| Altman Z-Score              | `custom/risk.py`           | Bankruptcy risk assessment         |
| Piotroski F-Score           | `custom/quality.py`        | Aggregate financial health (0-9)   |
| DuPont Decomposition        | `builtin/profitability.py` | RoE = Margin × Turnover × Leverage |
| Insider Buy/Sell Ratio      | `custom/sentiment.py`      | Management conviction              |
| Capex/Revenue Trend         | `builtin/growth.py`        | Investment intensity               |
| Tax Rate Consistency        | `custom/forensics.py`      | Accounting quality signal          |
| Sustainable Growth Rate     | `builtin/growth.py`        | RoE × (1 - Payout Ratio)           |
| EVA (Economic Value Added)  | `custom/quality.py`        | NOPAT - (Capital × WACC)           |
| Revenue per Employee Trend  | `custom/efficiency.py`     | Operational leverage signal        |
| Benford's Law on Financials | `custom/forensics.py`      | Accounting fraud detector          |

------

## Enhancement 3: GUI — When and How

### 3.1 My Recommendation: Not Yet, But Design for It

Building a GUI now would be premature for three reasons:

**1. The pipeline isn't stable yet.** You'll be iterating on data sources (Screener.in might change, you might switch to a paid API), metric definitions, scoring weights, and LLM prompts. Every GUI change during this period is wasted effort.

**2. The highest-value work is the engine, not the interface.** The quality of your investment research depends on whether your RoCE calculation is correct, not whether there's a pretty button to trigger it. CLI + markdown reports will serve you well for months.

**3. GUI effort is substantial.** A useful GUI is not just "display the report" — it's interactive filtering, drill-down on metrics, watchlist management, alerts, charting. That's a 4-6 week project on its own.

### 3.2 The Right Sequence

```
Phase 1-4 (Now)          Phase 5-6 (Month 2-3)        Phase 7 (Month 4+)
─────────────────         ──────────────────────        ──────────────────
CLI + JSON/MD output  →   Static HTML dashboards   →   Interactive Web GUI
                          (Jinja2 + Plotly)             (if needed)
                          ↑                             ↑
                          This might be                 Only if you're
                          enough forever                analyzing 50+
                                                        companies or
                                                        sharing with others
```

### 3.3 But: Design the Backend to Be GUI-Ready

Even without building a GUI now, structure the code so a frontend can be bolted on later with minimal refactoring:

```python
# The key: a clean service layer that both CLI and GUI can call

class ResearchService:
    """
    This class IS the API. CLI calls it. Future GUI calls it.
    No business logic in CLI scripts or GUI routes.
    """

    def analyze_company(self, ticker: str, config: dict) -> AnalysisResult:
        """Full pipeline: fetch → compute → score → (optional) LLM"""
        ...

    def discover_peers(self, ticker: str) -> PeerResult:
        """Auto-discover competitors"""
        ...

    def compare_peers(self, ticker: str, peers: list) -> ComparisonResult:
        """Side-by-side peer comparison"""
        ...

    def screen_universe(self, filters: dict) -> list[ScreenResult]:
        """Filter NSE universe by criteria"""
        ...

    def get_watchlist(self) -> list[WatchlistEntry]:
        """Return current watchlist with latest scores"""
        ...

    def run_quarterly_update(self, watchlist: str) -> UpdateResult:
        """Re-run analysis on watchlist companies"""
        ...

# CLI usage (now):
# cli.py
service = ResearchService()
result = service.analyze_company("ASTRAL", config)
render_markdown(result)

# Future GUI usage (later):
# app.py (FastAPI/Streamlit)
@app.get("/api/analyze/{ticker}")
def analyze(ticker: str):
    return service.analyze_company(ticker, config)
```

### 3.4 If You Do Build a GUI Later: Technology Choice

| Option                   | Best For                    | Effort    | Notes                                                        |
| ------------------------ | --------------------------- | --------- | ------------------------------------------------------------ |
| **Streamlit**            | Quick personal dashboard    | 1-2 weeks | Pythonic, fast to build, ugly but functional. Best for "I just need to see the data interactively." |
| **Plotly Dash**          | Interactive charts focus    | 2-3 weeks | Better charting than Streamlit, still Python-only. Good for financial dashboards. |
| **FastAPI + React**      | Polished multi-user app     | 4-6 weeks | Full separation of concerns. Only if you plan to share this with others or use it as a product. |
| **Static HTML (Jinja2)** | Reports you can email/share | 3-4 days  | Not really a "GUI" but generates beautiful standalone HTML reports with Plotly charts embedded. **Start here.** |

### 3.5 The Recommended "GUI Lite": Static HTML Dashboard

This gives you 80% of the GUI value with 10% of the effort:

```python
# report_generator.py
# Uses Jinja2 to produce a self-contained HTML file with embedded Plotly charts

def generate_html_report(result: AnalysisResult) -> str:
    template = jinja_env.get_template("sqglp_report.html.j2")
    return template.render(
        company=result.company,
        sqglp_scores=result.scores,
        radar_chart=plotly_radar_chart(result.scores),
        growth_chart=plotly_growth_decomposition(result.growth),
        pe_band_chart=plotly_pe_band(result.valuation),
        peer_table=result.peer_comparison,
        llm_thesis=result.llm_analysis,
    )
```

The output is a single `.html` file you can open in any browser — with interactive Plotly charts (hover, zoom), radar charts for SQGLP scores, and the full analysis. No server needed.

------

## Updated Architecture (v2)

```
financial_research/
├── config.yaml                      # Company + pipeline config
│
├── data_fetcher/                    # STAGE 1: Data Acquisition
│   ├── fetch_financials.py
│   ├── fetch_shareholding.py
│   ├── fetch_price_volume.py
│   ├── peer_discovery.py            # NEW: Auto competitor ID
│   └── raw_data/{TICKER}/
│
├── compute_engine/                  # STAGE 2: Offline Computation
│   ├── metrics/
│   │   ├── registry.yaml            # NEW: Metric definitions
│   │   ├── base.py                  # MetricResult dataclass
│   │   ├── builtin/                 # Built-in metric modules
│   │   │   ├── profitability.py
│   │   │   ├── growth.py
│   │   │   ├── valuation.py
│   │   │   ├── leverage.py
│   │   │   ├── efficiency.py
│   │   │   ├── size.py
│   │   │   └── longevity.py
│   │   └── custom/                  # NEW: Drop-in custom metrics
│   │       └── my_metrics.py
│   ├── engine.py                    # NEW: Generic metric runner
│   ├── scorer.py                    # Registry-driven scoring
│   └── peer_comparison.py
│
├── llm_layer/                       # STAGE 3: LLM Analysis
│   ├── prompts/
│   │   ├── pass1_qualitative.txt
│   │   ├── pass2_synthesis.txt
│   │   └── pass3_comparative.txt
│   └── orchestrator.py
│
├── service.py                       # NEW: Clean API layer (GUI-ready)
│
├── output/                          # STAGE 4: Reports
│   ├── templates/
│   │   ├── sqglp_report.html.j2     # Interactive HTML dashboard
│   │   └── sqglp_report.md.j2       # Markdown report
│   ├── report_generator.py
│   └── reports/{TICKER}_{DATE}/
│
├── cli.py                           # Command-line interface
└── requirements.txt
```

------

## Updated Implementation Roadmap (v2)

### Phase 1: Foundation (Week 1–2) — unchanged

- [ ] Data fetchers for financials, price, shareholding

### Phase 2: Compute Engine (Week 3–4) — restructured

- [ ] Implement `registry.yaml` with initial 20 metrics
- [ ] Implement `engine.py` (generic metric runner)
- [ ] Implement built-in metric modules (profitability, growth, valuation, etc.)
- [ ] Implement `scorer.py` (registry-driven SQGLP scoring)
- [ ] Implement `peer_comparison.py` (uses same engine for all companies)
- [ ] Validate: Run on 3 known companies, compare scores to manual analysis

### Phase 2.5: Peer Discovery (Week 4) — NEW

- [ ] Implement Layers 1-3 of `peer_discovery.py` (sector + size + financial similarity)
- [ ] Test: Given "ASTRAL", does it find Supreme, Prince, Finolex?
- [ ] Optional: Layer 4 LLM validation prompt

### Phase 3: LLM Integration (Week 5) — unchanged

### Phase 4: Reporting (Week 6) — enhanced

- [ ] Build `service.py` (clean API layer)
- [ ] Build Jinja2 HTML template with Plotly charts (the "GUI Lite")
- [ ] Build markdown report template
- [ ] CLI: `python cli.py analyze ASTRAL` → produces HTML + MD reports

### Phase 5: Screening & Watchlist (Week 7+)

- [ ] Universe screener using registry-defined filters
- [ ] Watchlist management + quarterly re-run

### Phase 6 (Future): Interactive GUI

- [ ] Only if static HTML reports feel insufficient
- [ ] Start with Streamlit, backed by `service.py`