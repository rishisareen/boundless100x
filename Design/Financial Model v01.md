# Financial Research System Design

## Deep Company Analysis for Long-Term Investment (Indian Markets)

**Based on the SQGLP Framework (Size, Quality, Growth, Longevity, Price)**

------

## 1. System Architecture Overview

The system is designed around a core principle: **compute locally, analyze with LLM**. All number-crunching, data fetching, and ratio calculations happen in offline Python scripts. The LLM is only invoked for qualitative judgment, pattern recognition, and final synthesis — minimizing token usage and cost.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RESEARCH PIPELINE                            │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐  │
│  │  STAGE 1 │───▶│   STAGE 2    │───▶│  STAGE 3  │───▶│ STAGE 4  │  │
│  │  Data    │    │  Compute     │    │  LLM      │    │ Output   │  │
│  │  Fetch   │    │  (Offline)   │    │  Analysis │    │ Report   │  │
│  └──────────┘    └──────────────┘    └───────────┘    └──────────┘  │
│   Python          Python              Claude API       Markdown/    │
│   Scripts         Scripts             (Paid calls)     HTML/DOCX    │
│   FREE            FREE                OPTIMIZED        FREE         │
└─────────────────────────────────────────────────────────────────────┘
```

------

## 2. Data Sources & Fetching (Stage 1)

### 2.1 Primary Data Sources

| Data Type                              | Source                                | Method            | Notes                            |
| -------------------------------------- | ------------------------------------- | ----------------- | -------------------------------- |
| **Financial Statements** (P&L, BS, CF) | Screener.in                           | Web scrape or API | 10-year data, already normalized |
| **Shareholding Patterns**              | BSE India / Trendlyne                 | API / Scrape      | FII, DII, Promoter quarterly     |
| **Daily Price & Volume**               | NSE (via `jugaad-data` or `nsetools`) | Python library    | OHLCV data                       |
| **Corporate Actions**                  | BSE / Moneycontrol                    | Scrape            | Splits, bonuses, dividends       |
| **Peer Comparison**                    | Screener.in sector pages              | Scrape            | Industry averages                |
| **Annual Reports / Con-calls**         | BSE filings / company website         | Download PDF      | For LLM qualitative analysis     |
| **Analyst Coverage Count**             | Trendlyne / Tickertape                | Scrape            | For "unknown-ness" metric        |

### 2.2 Recommended Python Libraries

```
jugaad-data        — NSE historical data (free, no API key)
nsetools           — Live NSE data
bsedata            — BSE corporate data
requests + bs4     — Web scraping for Screener.in, Trendlyne
pandas             — All data manipulation
yfinance           — Fallback for price data (.NS suffix)
```

### 2.3 Data Fetch Script Structure

```
data_fetcher/
├── config.yaml                  # Company ticker, peers list, date ranges
├── fetch_financials.py          # Screener.in → 10yr P&L, BS, CF
├── fetch_shareholding.py        # BSE → quarterly shareholding
├── fetch_price_volume.py        # NSE → daily OHLCV
├── fetch_peers.py               # Sector peers from Screener
├── fetch_analyst_coverage.py    # Trendlyne → analyst count
├── download_annual_reports.py   # BSE filings → PDF
└── raw_data/
    └── {TICKER}/
        ├── financials.csv
        ├── shareholding.csv
        ├── price_volume.csv
        ├── peers.csv
        └── annual_reports/
```

### 2.4 `config.yaml` Example

```yaml
target:
  ticker: "ASTRAL"
  bse_code: "532830"
  nse_symbol: "ASTRAL"
  sector: "Building Materials"

peers:
  - { ticker: "SUPREMEIND", name: "Supreme Industries" }
  - { ticker: "PRINCEPIPE", name: "Prince Pipes" }
  - { ticker: "FINOLEX", name: "Finolex Industries" }

analysis_period:
  financials_years: 10       # 10 years of annual data
  price_history_years: 10
  shareholding_quarters: 20  # 5 years quarterly
```

------

## 3. Offline Computation Engine (Stage 2)

This is where all the SQGLP metrics are computed locally — **zero LLM cost**.

### 3.1 Module Structure

```
compute_engine/
├── s_size.py              # Element 1: Size metrics
├── q_quality_business.py  # Element 2a: Business quality ratios
├── q_quality_mgmt.py      # Element 2b: Management quality signals
├── g_growth.py            # Element 3: 4-lever growth decomposition
├── l_longevity.py         # Element 4: Consistency & moat metrics
├── p_price.py             # Element 5: Valuation metrics
├── peer_comparison.py     # Cross-company comparative analysis
├── scoring.py             # Composite SQGLP score
└── output/
    └── {TICKER}/
        ├── sqglp_report.json       # Structured data for LLM
        ├── sqglp_summary.md        # Human-readable summary
        └── peer_comparison.json
```

### 3.2 Metrics by SQGLP Element

------

#### S — Size Metrics (`s_size.py`)

| Metric                 | Formula / Source                    | Target for 100x                 |
| ---------------------- | ----------------------------------- | ------------------------------- |
| Market Cap Category    | Current price × shares outstanding  | Mid-cap or below (< ₹50,000 Cr) |
| FII + DII Holding %    | From shareholding data              | 1% – 10%, ideally rising        |
| Analyst Coverage Count | Trendlyne scrape                    | 0 – 3 brokerages                |
| Daily Turnover Ratio   | Avg daily traded value / Market Cap | < 0.1%                          |
| Free Float %           | 100% – Promoter holding             | > 25% (liquidity floor)         |

**Output:**

```json
{
  "size": {
    "market_cap_cr": 45000,
    "category": "mid_cap",
    "fii_holding_pct": 3.2,
    "dii_holding_pct": 4.1,
    "institutional_total_pct": 7.3,
    "analyst_count": 2,
    "daily_turnover_ratio_pct": 0.08,
    "free_float_pct": 45.2,
    "score": 8,       // out of 10
    "flags": ["low_institutional_ownership_positive", "under_researched"]
  }
}
```

------

#### Q — Quality of Business (`q_quality_business.py`)

| Metric                         | Formula                                 | What it reveals         | Benchmark         |
| ------------------------------ | --------------------------------------- | ----------------------- | ----------------- |
| **RoCE** (5yr avg)             | EBIT / Capital Employed                 | Capital efficiency      | > 15%             |
| **RoE** (5yr avg)              | PAT / Equity                            | Shareholder returns     | > 15%             |
| **Operating Margin** (5yr avg) | EBIT / Sales                            | Pricing power           | Sector-dependent  |
| **FCF Yield**                  | Free Cash Flow / Market Cap             | Cash generation         | > 3%              |
| **Debt/Equity**                | Total Debt / Shareholder Equity         | Leverage risk           | < 0.5 preferred   |
| **Interest Coverage**          | EBIT / Interest Expense                 | Debt servicing ability  | > 3x              |
| **Working Capital Days** trend | (Receivable + Inventory - Payable days) | Efficiency              | Declining is good |
| **Profit Pool Position**       | Company PAT / Sector aggregate PAT      | Market share of profits | Rising is good    |
| **Cash Conversion**            | Operating CF / EBITDA                   | Earnings quality        | > 70%             |

**Derived Flags (computed, not LLM):**

- `consistently_high_roce`: RoCE > 15% in 8 of last 10 years
- `improving_margins`: Operating margin expanding for 3+ consecutive years
- `cash_cow`: FCF positive for 8 of last 10 years
- `debt_risk`: Debt/Equity > 1.0 or Interest Coverage < 2x

------

#### Q — Quality of Management (`q_quality_mgmt.py`)

These are partially quantitative signals that can be computed offline:

| Signal                          | How to Compute                          | What it indicates             |
| ------------------------------- | --------------------------------------- | ----------------------------- |
| **Promoter Holding Trend**      | Δ Promoter % over 5 years               | Skin in the game              |
| **Promoter Pledge %**           | From shareholding data                  | Red flag if > 10%             |
| **Dividend Payout Consistency** | # years dividend paid / 10              | Capital allocation discipline |
| **Related Party Transactions**  | Flag from annual report (manual or LLM) | Integrity check               |
| **Equity Dilution**             | Shares outstanding growth over 10 yrs   | Shareholder-friendliness      |
| **Capex as % of CFO**           | Capex / Operating Cash Flow             | Reinvestment rate             |
| **Tax Rate Consistency**        | Effective tax rate variance             | Aggressive accounting flag    |

**Qualitative aspects (marked for LLM analysis):**

- Management commentary consistency (con-call transcripts)
- Capital allocation track record
- Related party transaction review
- Corporate governance red flags

------

#### G — Growth (`g_growth.py`)

**Core: 4-Lever Decomposition**

```python
def four_lever_decomposition(financials_df):
    """
    Decomposes EPS growth into:
    1. Volume Growth (if data available, else Revenue proxy)
    2. Price Lever = Revenue Growth / Volume Growth
    3. Operating Lever = EBIT Growth / Revenue Growth
    4. Financial Lever = EPS Growth / EBIT Growth
    
    Returns quality assessment:
    - HIGH: Driven by Volume + Operating Leverage
    - LOW:  Driven by Financial Leverage + Price Hikes
    """
```

| Metric                     | Formula                            | Target                 |
| -------------------------- | ---------------------------------- | ---------------------- |
| Revenue CAGR (3/5/10 yr)   | (End/Start)^(1/n) - 1              | > 15%                  |
| PAT CAGR (3/5/10 yr)       | Same                               | > 18%                  |
| EPS CAGR (3/5/10 yr)       | Same                               | > 18%                  |
| Operating Lever            | Δ EBIT / Δ Sales (YoY)             | > 1.0 sustained        |
| Financial Lever            | Δ EPS / Δ EBIT (YoY)               | 0.8 – 1.2 (moderate)   |
| Revenue Growth Consistency | Std dev of YoY growth rates        | Low variance preferred |
| Earnings Surprise Trend    | Actual vs consensus (if available) | Positive surprises     |

**Growth Quality Score (computed):**

```
If primary drivers are Volume + Operating Lever → "High Quality Growth"
If primary driver is Financial Lever → "Leveraged Growth (Risky)"
If primary driver is Price Hikes only → "Pricing-Dependent Growth"
```

------

#### L — Longevity (`l_longevity.py`)

| Metric                            | Formula                                | What it measures               |
| --------------------------------- | -------------------------------------- | ------------------------------ |
| **RoCE Consistency**              | # years RoCE > 15% out of 10           | Moat durability                |
| **Revenue Growth Streak**         | Max consecutive years of > 10% growth  | Demand sustainability          |
| **Market Share Trend**            | Company revenue / sector revenue (5yr) | Competitive position           |
| **Gross Margin Stability**        | Std deviation of gross margin (10yr)   | Pricing power durability       |
| **Customer Concentration**        | Top client % of revenue (from AR)      | Risk of key client loss        |
| **Reinvestment Rate**             | Capex / Depreciation                   | Continued investment in growth |
| **R&D / Revenue** (if applicable) | R&D spend / Revenue                    | Innovation investment          |

------

#### P — Price / Valuation (`p_price.py`)

| Metric                  | Formula                                   | Context                         |
| ----------------------- | ----------------------------------------- | ------------------------------- |
| **Current P/E**         | Price / TTM EPS                           | vs 5yr median, vs sector median |
| **P/E to Growth (PEG)** | P/E / EPS CAGR                            | < 1.0 is attractive             |
| **EV/EBITDA**           | Enterprise Value / EBITDA                 | vs sector average               |
| **Price/Book**          | Market Cap / Book Value                   | vs historical                   |
| **FCF Yield**           | FCF / Market Cap                          | > 4% is attractive              |
| **Earnings Yield**      | EPS / Price                               | vs risk-free rate (10yr G-Sec)  |
| **Historical P/E Band** | 10yr P/E range with current position      | Percentile ranking              |
| **Margin of Safety**    | (Intrinsic Value - CMP) / Intrinsic Value | > 20% preferred                 |

**Intrinsic Value Models (all computed offline):**

1. **DCF** — 10yr projected FCF, discount at WACC
2. **Earnings Power Value** — Normalized earnings / cost of capital
3. **Reverse DCF** — What growth rate is the market pricing in?

------

### 3.3 Peer Comparison Engine (`peer_comparison.py`)

For each peer, compute the same Q, G, P metrics above. Output a comparative table:

```json
{
  "comparison": {
    "metrics": ["RoCE_5yr", "PAT_CAGR_5yr", "P/E", "PEG", "Debt_Equity"],
    "companies": {
      "ASTRAL":     [28.5, 22.3, 65.2, 2.9, 0.1],
      "SUPREMEIND": [24.1, 18.7, 42.1, 2.2, 0.2],
      "PRINCEPIPE": [18.3, 25.1, 38.5, 1.5, 0.3],
      "FINOLEX":    [15.2, 12.4, 22.8, 1.8, 0.05]
    },
    "sector_median": [18.0, 16.5, 35.0, 2.0, 0.2],
    "rankings": {
      "ASTRAL": { "quality_rank": 1, "growth_rank": 2, "value_rank": 4 }
    }
  }
}
```

------

### 3.4 Composite SQGLP Scoring (`scoring.py`)

Each element is scored 1–10 based on thresholds. Weighted composite:

```
SQGLP Score = (S × 0.10) + (Q × 0.30) + (G × 0.25) + (L × 0.20) + (P × 0.15)
```

Weighting rationale:

- **Quality (30%)** — most important per your framework
- **Growth (25%)** — the superstructure
- **Longevity (20%)** — sustainability of quality + growth
- **Price (15%)** — entry valuation matters but less than business quality
- **Size (10%)** — a filter more than a driver

------

## 4. LLM Analysis Layer (Stage 3) — Cost Optimized

### 4.1 Design Principle: Structured Input → Focused Output

Instead of feeding raw data to the LLM, we feed **pre-computed JSON** with clear questions. This reduces input tokens by ~80% compared to sending raw financial statements.

### 4.2 LLM Call Architecture

The LLM is called in **3 focused passes**, each with a specific role:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM CALL STRATEGY                            │
│                                                                 │
│  Pass 1: QUALITATIVE ANALYSIS (~2K input, ~1K output tokens)    │
│  ├── Input: Annual report excerpts, con-call highlights         │
│  ├── Task: Assess management quality, competitive moat,         │
│  │         business model risks, sector tailwinds               │
│  └── Output: Structured qualitative assessment JSON             │
│                                                                 │
│  Pass 2: SYNTHESIS (~3K input, ~2K output tokens)               │
│  ├── Input: SQGLP computed metrics + Pass 1 output              │
│  ├── Task: Identify contradictions, hidden risks, key           │
│  │         strengths, and generate investment thesis            │
│  └── Output: Investment thesis + risk assessment                │
│                                                                 │
│  Pass 3: COMPARATIVE JUDGMENT (~2K input, ~1K output tokens)    │
│  ├── Input: Peer comparison table + sector context              │
│  ├── Task: Rank target vs peers, identify relative              │
│  │         advantages/disadvantages                             │
│  └── Output: Competitive positioning assessment                 │
│                                                                 │
│  TOTAL ESTIMATED COST: ~$0.05–$0.15 per company (Sonnet)        │
│  vs ~$2–5 if sending raw data without pre-processing            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Prompt Templates

#### Pass 1: Qualitative Analysis

```
SYSTEM: You are an equity research analyst specializing in Indian 
mid-cap companies. Analyze the following management and business 
quality indicators. Be specific and cite evidence.

INPUT:
- Management commentary excerpts: {extracted_text}
- Promoter holding trend: {data}
- Related party transactions summary: {data}
- Key risks from annual report: {extracted_text}

OUTPUT FORMAT (JSON):
{
  "management_integrity_score": 1-10,
  "management_competence_score": 1-10,
  "growth_mindset_score": 1-10,
  "moat_type": "brand|cost|network|switching|regulatory|none",
  "moat_strength": 1-10,
  "key_risks": ["risk1", "risk2"],
  "sector_tailwinds": ["tailwind1"],
  "red_flags": ["flag1"] or [],
  "reasoning": "..."
}
```

#### Pass 2: Synthesis

```
SYSTEM: You are a senior investment analyst. Given the pre-computed 
SQGLP metrics and qualitative assessment below, synthesize an 
investment thesis. Focus on:
1. Is this a potential long-term compounder?
2. What could go wrong? (kill-the-thesis risks)
3. What is the market missing? (if anything)

INPUT:
- SQGLP Metrics: {sqglp_report.json}
- Qualitative Assessment: {pass_1_output}
- Current SQGLP Score: {composite_score}

OUTPUT FORMAT (JSON):
{
  "thesis": "...",
  "conviction_level": "high|medium|low",
  "bull_case": "...",
  "bear_case": "...",
  "key_monitorables": ["metric1", "metric2"],
  "suggested_action": "buy|hold|watchlist|avoid",
  "target_holding_period": "3-5yr|5-10yr|10yr+",
  "reasoning": "..."
}
```

#### Pass 3: Competitive Judgment

```
SYSTEM: Compare the target company against its peers using the 
data below. Identify where it has a clear edge and where it lags.

INPUT:
- Peer comparison table: {peer_comparison.json}
- Target company thesis: {pass_2_output}

OUTPUT FORMAT (JSON):
{
  "competitive_advantages": ["..."],
  "competitive_disadvantages": ["..."],
  "best_in_class_metrics": ["RoCE", "Growth"],
  "worst_in_class_metrics": ["Valuation"],
  "preferred_pick_in_sector": "TICKER",
  "reasoning": "..."
}
```

### 4.4 Further Cost Optimization Strategies

| Strategy                  | Savings                       | How                                      |
| ------------------------- | ----------------------------- | ---------------------------------------- |
| **Use Haiku for Pass 3**  | ~60% on that pass             | Comparative ranking is simpler           |
| **Cache sector context**  | Avoid re-fetching             | Same sector prompt prefix for all peers  |
| **Batch companies**       | Reduce system prompt overhead | Analyze 3-4 peers in one Pass 3 call     |
| **Skip Pass 1 if no AR**  | Save one full call            | Only run when annual report is available |
| **Use structured output** | Fewer output tokens           | JSON schema forces concise responses     |

------

## 5. Output & Report Generation (Stage 4)

### 5.1 Output Files

```
reports/
└── {TICKER}_{DATE}/
    ├── sqglp_dashboard.html      # Interactive single-page report
    ├── sqglp_report.md           # Markdown summary
    ├── sqglp_report.docx         # Professional Word document
    ├── raw_metrics.json          # All computed data
    └── llm_analysis.json         # All LLM outputs
```

### 5.2 Report Sections

1. **Executive Summary** — One-paragraph thesis + SQGLP score radar chart
2. **Size Analysis** — Market cap, institutional ownership, discovery potential
3. **Quality Scorecard** — Business quality ratios + management assessment
4. **Growth Decomposition** — 4-lever analysis with quality grading
5. **Longevity Assessment** — Moat analysis, consistency metrics
6. **Valuation Analysis** — Current vs historical, DCF, PEG, peer relative
7. **Peer Comparison Table** — Side-by-side metrics
8. **Risk Register** — Red flags, kill-the-thesis scenarios
9. **Monitorables Checklist** — What to track quarterly
10. **Appendix** — Raw data tables, methodology notes

------

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1–2)

- [ ] Set up project structure and `config.yaml`
- [ ] Build `fetch_financials.py` (Screener.in scraper)
- [ ] Build `fetch_price_volume.py` (NSE via jugaad-data)
- [ ] Build `fetch_shareholding.py` (BSE)
- [ ] Test with 2-3 known companies (e.g., Astral, Bajaj Finance)

### Phase 2: Compute Engine (Week 3–4)

- [ ] Implement `q_quality_business.py` (RoCE, RoE, margins, etc.)
- [ ] Implement `g_growth.py` (4-lever decomposition)
- [ ] Implement `p_price.py` (valuation metrics + DCF)
- [ ] Implement `s_size.py` and `l_longevity.py`
- [ ] Implement `peer_comparison.py`
- [ ] Implement `scoring.py` (composite SQGLP score)
- [ ] Validate outputs against known good analyses

### Phase 3: LLM Integration (Week 5)

- [ ] Design and test prompt templates (Pass 1, 2, 3)
- [ ] Build LLM orchestrator script with retry logic
- [ ] Add annual report PDF text extraction (PyMuPDF)
- [ ] Test end-to-end pipeline on 5 companies

### Phase 4: Reporting (Week 6)

- [ ] Build markdown report generator
- [ ] Build HTML dashboard with charts (matplotlib/plotly)
- [ ] Optional: DOCX report generation

### Phase 5: Screening (Week 7+)

- [ ] Build screener that runs Size + basic Quality filters on full NSE universe
- [ ] Shortlist candidates → run full pipeline on top 20
- [ ] Build watchlist tracking (quarterly re-run)

------

## 7. Technology Stack Summary

| Component         | Technology                                         | Cost           |
| ----------------- | -------------------------------------------------- | -------------- |
| Data Fetching     | Python (requests, jugaad-data, bs4)                | Free           |
| Data Storage      | CSV/JSON files (or SQLite for scale)               | Free           |
| Computation       | Python (pandas, numpy)                             | Free           |
| Visualization     | Plotly / Matplotlib                                | Free           |
| LLM Analysis      | Claude API (Sonnet for Pass 1-2, Haiku for Pass 3) | ~$0.10/company |
| Report Generation | Jinja2 templates → MD/HTML                         | Free           |
| Orchestration     | Python CLI (click/typer)                           | Free           |

------

## 8. Usage Example

```bash
# Full pipeline for a single company
python run_analysis.py --ticker ASTRAL --peers SUPREMEIND,PRINCEPIPE,FINOLEX

# Screening mode: filter universe first
python screen.py --min-roce 15 --max-mcap 50000 --min-revenue-cagr 15

# Re-run quarterly update on watchlist
python update_watchlist.py --watchlist my_watchlist.yaml
```

------

## 9. Key Design Decisions & Rationale

**Why scrape Screener.in instead of raw BSE filings?** Screener.in already normalizes financial data across companies, handles consolidated vs standalone, and provides 10-year history in a clean format. Parsing raw XBRL from BSE is significantly more effort for the same data.

**Why JSON intermediate format instead of direct LLM calls?** JSON acts as a contract between compute and LLM layers. You can inspect, validate, and version-control the computed metrics independently. If LLM pricing changes, you swap models without touching computation. If a metric formula is wrong, you fix it without re-running LLM calls.

**Why 3 LLM passes instead of 1 big call?** Each pass has a focused scope, producing more reliable output. A single large prompt with all data tends to produce generic analysis. Splitting also lets you use cheaper models for simpler tasks (Pass 3 with Haiku).

**Why not a database?** For personal research on 20-50 companies, CSV/JSON files are simpler to inspect, debug, and version control with git. If you scale to 500+ companies, migrate to SQLite or PostgreSQL.