# SQGLP Research System — Design Update v4

## Summary of Changes

Two targeted enhancements to the v3 consolidated design:

1. **Peer Comparison: Remove Financial Peers, Keep Industry Peers Only** — Eliminates cross-sector financial similarity matching (Layer 3) from the peer discovery pipeline and from the peer comparison output. The rationale: financial similarity peers (e.g., Berger Paints showing up as a peer for Astral Pipes because they share a similar RoCE/margin profile) are misleading in a competitive analysis context. They make sense for valuation benchmarking, but not for the core peer comparison table.

2. **Report Growth Section: Expanded 4-Lever Earnings Decomposition** — The Growth Decomposition report section is restructured to follow a structured equity analyst template. It now includes an Earnings Growth Profile table, a 4-Lever Decomposition table, a Growth Synthesis narrative, and a Valuation Reality Check (PEG). This makes the report section production-grade for equity research.

---

## Change 1: Peer Comparison — Industry Peers Only

### 1.1 What Changes

**Removed:**
- Layer 3 (Financial Similarity Scoring) is removed from the peer discovery pipeline
- `financial_peers` field is removed from `PeerResult` dataclass
- `include_financial_peers` config option is removed
- Cross-sector companies no longer appear in any peer comparison output

**Retained:**
- Layer 1: Industry Classification (Screener.in sector, BSE classification)
- Layer 2: Size-Filtered Cohort (market cap band, revenue band, listing history)
- Layer 4 (optional): LLM Peer Validation — now operates only on Layer 2 output
- Layer 5 (optional): Value Chain Mapping

### 1.2 Updated Peer Discovery Pipeline

```
┌──────────────────────────────────────────────────────────┐
│              PEER DISCOVERY PIPELINE (v4)                 │
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
│  └── Output: 5-15 size-matched industry peers           │
│                                                          │
│  Layer 3: LLM Peer Validation (OPTIONAL, ~$0.02)        │
│  ├── Input: Layer 2 candidates + business description   │
│  ├── Task: "Which are true competitors vs tangential?"  │
│  └── Output: Final 4-6 validated direct competitors     │
│                                                          │
│  Layer 4: Value Chain Mapping (OPTIONAL, ~$0.02)         │
│  ├── Input: Company business description                │
│  ├── Task: "Who are upstream/downstream/adjacent?"      │
│  └── Output: Adjacent companies to monitor              │
└──────────────────────────────────────────────────────────┘
```

**Key difference from v3:** Layer 3 (Financial Similarity) is gone. Layers 4 and 5 are renumbered to 3 and 4. The pipeline is now purely industry-focused: all peers come from the same sector/industry classification, then filtered by size, then optionally validated by LLM for competitive relevance.

### 1.3 Updated Implementation

```python
# data_fetcher/peer_discovery.py (v4)

from dataclasses import dataclass

@dataclass
class PeerResult:
    direct_competitors: list[str]     # Layer 2 (or Layer 3 if LLM used)
    sector_peers: list[str]           # Layer 1: full sector list
    value_chain: list[str]            # Layer 4: upstream/downstream (if LLM used)
    discovery_metadata: dict          # Filtering stats

class PeerDiscovery:
    """
    Industry-focused peer identification.
    Layers 1-2: fully offline, zero LLM cost.
    Layers 3-4: optional LLM refinement.
    
    v4 CHANGE: Removed cross-sector financial similarity matching.
    Peers are now exclusively from the same industry classification.
    """

    def discover(self, ticker: str, use_llm: bool = False) -> PeerResult:
        # Layer 1: Sector classification from Screener.in
        sector_peers = self._get_sector_peers(ticker)

        # Layer 2: Size filtering (within sector only)
        size_filtered = self._filter_by_size(ticker, sector_peers)

        # Layer 3 (optional): LLM validation of industry peers
        if use_llm:
            validated = self._llm_validate_peers(ticker, size_filtered)
            value_chain = self._llm_map_value_chain(ticker)
        else:
            validated = size_filtered[:5]
            value_chain = []

        return PeerResult(
            direct_competitors=validated,
            sector_peers=sector_peers,
            value_chain=value_chain,
            discovery_metadata={
                "candidates_evaluated": len(sector_peers),
                "size_filtered_to": len(size_filtered),
            },
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

    # REMOVED: _rank_by_financial_similarity() method
    # v4 rationale: Cross-sector financial similarity peers were causing
    # confusion in peer comparison tables. A pipe company being compared
    # to a paint company because they share margin profiles is misleading
    # for competitive analysis.

    def _llm_validate_peers(self, ticker: str, candidates: list[str]) -> list[str]:
        """LLM reviews size-filtered industry peers for competitive relevance."""
        ...

    def _llm_map_value_chain(self, ticker: str) -> list[str]:
        """LLM identifies upstream/downstream/adjacent companies."""
        ...
```

### 1.4 Updated Peer Discovery Output

```json
{
  "target": "ASTRAL",
  "discovery": {
    "direct_competitors": ["SUPREMEIND", "PRINCEPIPE", "FINOLEX", "APLLTD", "ASTEC"],
    "sector_peers": ["SUPREMEIND", "PRINCEPIPE", "FINOLEX", "APLLTD", "...40 more"],
    "value_chain": [],
    "metadata": {
      "sector": "Building Materials - Plastic Pipes",
      "candidates_evaluated": 43,
      "size_filtered_to": 12
    }
  }
}
```

**Removed:** `financial_peers` field and `similarity_scores` from metadata.

### 1.5 Updated Config

```yaml
peer_discovery:
  enabled: true                    # Auto-discover if peers not specified
  use_llm_validation: false        # Layer 3-4 (costs ~$0.02)
  max_peers: 5                     # Final peer count
  size_band_multiplier: 3.0        # 0.3x to 3x market cap
  sector_source: "screener"        # "screener" | "bse" | "moneycontrol"
  # REMOVED: include_financial_peers option
```

### 1.6 Impact on Other Sections

**Peer Comparison Engine (`peer_comparison.py`):** No change to the comparison logic itself — it still runs all registry metrics on each peer. The only difference is the *input list of peers* no longer includes cross-sector matches.

**LLM Pass 3 (Competitive Judgment):** Unchanged in logic, but now receives only industry peers, which makes the competitive analysis more relevant and focused.

**Report Section 9 (Peer Comparison):** The side-by-side table now shows only companies from the same sector. Removed the "Financial Similarity Peers" sub-table.

**Report Section 10 (Peer Discovery):** Simplified — no longer shows cross-sector similarity scores or financial distance metrics.

---

## Change 2: Expanded Growth Report Section — Earnings Quality & 4-Lever Decomposition

### 2.1 What Changes

The report section "Growth Decomposition" (Section 6 in v3 report, previously a brief 4-lever chart + quality grade) is now expanded into a full **Earnings Quality & Valuation Report** subsection. This follows a structured equity analyst template with four explicit sub-sections:

1. **Earnings Growth Profile** — Summary table of 3yr and 5yr PAT CAGR
2. **4-Lever Earnings Decomposition** — Detailed table breaking down each lever with status and analysis
3. **Growth Synthesis** — Bullet-point narrative connecting the levers to the growth quality
4. **Valuation Reality Check** — PEG ratio calculation and verdict

### 2.2 New Metrics Required

The following metrics are needed for the expanded report. Most already exist in the v3 metric registry; new additions are marked:

| Metric | Status | Registry Location |
|--------|--------|-------------------|
| PAT CAGR (3yr) | ✅ Exists | `elements/growth.yaml` → `pat_cagr_3yr` |
| PAT CAGR (5yr) | ✅ Exists | `elements/growth.yaml` → `pat_cagr_5yr` |
| Revenue CAGR (3yr) | 🆕 New | `elements/growth.yaml` → `revenue_cagr_3yr` |
| Revenue CAGR (5yr) | ✅ Exists | `elements/growth.yaml` → `revenue_cagr_5yr` |
| Volume Growth (proxy: unit sales or revenue deflated) | ✅ Exists (proxy) | Derived from revenue vs price index |
| Operating Leverage (avg 5yr) | ✅ Exists | `elements/growth.yaml` → `operating_leverage` |
| Financial Leverage (avg 5yr) | ✅ Exists | `elements/growth.yaml` → `financial_leverage_ratio` |
| Growth Quality Grade | ✅ Exists | `elements/growth.yaml` → `growth_quality_grade` |
| Current P/E | ✅ Exists | `elements/price.yaml` → `current_pe` |
| Trailing PEG Ratio | ✅ Exists | `elements/price.yaml` → `trailing_peg` |
| Revenue Growth Consistency | ✅ Exists | `elements/growth.yaml` → `revenue_growth_consistency` |
| Revenue vs Volume divergence (Price Lever signal) | 🆕 New | `elements/growth.yaml` → `price_lever_signal` |
| EBIT CAGR (5yr) | 🆕 New | `elements/growth.yaml` → `ebit_cagr_5yr` |
| EBIT CAGR (3yr) | 🆕 New | `elements/growth.yaml` → `ebit_cagr_3yr` |

### 2.3 New Metric Definitions for `elements/growth.yaml`

```yaml
  # ─── NEW in v4: Additional growth metrics for expanded report ───

  revenue_cagr_3yr:
    name: "Revenue CAGR (3yr)"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params: { field: "revenue", years: 3 }
    scoring: { thresholds: [5, 10, 15, 20, 30, 50], direction: "higher_is_better", weight: 0.05 }
    display: { format: "{:.1f}%", section: "growth_decomposition", peer_compare: true }

  ebit_cagr_5yr:
    name: "EBIT CAGR (5yr)"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params: { field: "ebit", years: 5 }
    scoring: { thresholds: [3, 8, 15, 20, 28, 40], direction: "higher_is_better", weight: 0.05 }
    display: { format: "{:.1f}%", section: "growth_decomposition", peer_compare: true }

  ebit_cagr_3yr:
    name: "EBIT CAGR (3yr)"
    module: "builtin.growth"
    function: "compute_cagr"
    inputs: ["financials"]
    params: { field: "ebit", years: 3 }
    scoring: { thresholds: [5, 10, 15, 20, 30, 50], direction: "higher_is_better", weight: 0.05 }
    display: { format: "{:.1f}%", section: "growth_decomposition", peer_compare: true }

  price_lever_signal:
    name: "Price Lever Signal"
    module: "builtin.growth"
    function: "compute_price_lever"
    inputs: ["financials"]
    params: { years: 5 }
    scoring:
      mode: "categorical"
      categories: { "strong_pricing_power": 10, "moderate_pricing": 7, "discounting": 3, "unknown": 5 }
      weight: 0.05
    display: { format: "{}", section: "growth_decomposition", peer_compare: false }
```

### 2.4 New Compute Functions in `builtin/growth.py`

```python
# builtin/growth.py — New functions for v4 expanded growth report

def compute_price_lever(financials: pd.DataFrame, years: int = 5) -> MetricResult:
    """
    Detect pricing power by comparing revenue growth to volume proxy.
    
    For companies where unit volume data is available (e.g., from annual reports):
        Price Lever = Revenue CAGR / Volume CAGR
    
    For most companies (volume data unavailable), use proxy signals:
        1. Revenue CAGR vs sector average (if growing faster with stable margins → pricing)
        2. Realization per unit trend (if available from segments)
        3. Revenue growth minus WPI/CPI inflation = real pricing signal
    
    Categories:
        - "strong_pricing_power": Revenue CAGR > Volume proxy + 3pp consistently
        - "moderate_pricing": Revenue CAGR > Volume proxy + 1-3pp
        - "discounting": Revenue CAGR < Volume proxy (selling more but at lower prices)
        - "unknown": Insufficient data
    """
    revenue_cagr = _compute_cagr(financials, "revenue", years)
    
    # Proxy: Use WPI/CPI to deflate revenue → estimate real volume growth
    # Real volume growth = Revenue growth - Inflation
    # Price lever = Revenue growth / Real volume growth
    # If price lever > 1.0 consistently → pricing power
    
    # Alternatively, if segment-wise realization data exists:
    # realization_growth = compute from annual report segments
    
    # For now, use the proxy approach
    inflation_avg = 5.0  # Default assumption; can be parameterized
    real_volume_growth = revenue_cagr - inflation_avg
    
    if real_volume_growth <= 0:
        signal = "discounting" if revenue_cagr < inflation_avg else "unknown"
    elif revenue_cagr > real_volume_growth + 3:
        signal = "strong_pricing_power"
    elif revenue_cagr > real_volume_growth + 1:
        signal = "moderate_pricing"
    else:
        signal = "unknown"
    
    return MetricResult(
        value=signal,
        details={
            "revenue_cagr": revenue_cagr,
            "estimated_volume_growth": real_volume_growth,
            "inflation_assumption": inflation_avg,
        },
    )


def compute_lever_decomposition_table(financials: pd.DataFrame, years: int = 5) -> dict:
    """
    Full 4-lever decomposition for the expanded report section.
    
    Returns a structured dict with:
    - earnings_profile: {pat_cagr_3yr, pat_cagr_5yr}
    - lever_table: [{lever, status, analysis}, ...]
    - growth_synthesis: {primary_driver, quality_flag, narrative}
    - valuation_check: {current_pe, pat_cagr_5yr, trailing_peg, verdict}
    
    This is consumed by the Jinja2 report template.
    """
    # Compute all required CAGRs
    rev_cagr_3 = _compute_cagr(financials, "revenue", 3)
    rev_cagr_5 = _compute_cagr(financials, "revenue", 5)
    pat_cagr_3 = _compute_cagr(financials, "pat", 3)
    pat_cagr_5 = _compute_cagr(financials, "pat", 5)
    ebit_cagr_3 = _compute_cagr(financials, "ebit", 3)
    ebit_cagr_5 = _compute_cagr(financials, "ebit", 5)
    eps_cagr_5 = _compute_cagr(financials, "eps", 5)
    
    # Operating Leverage = EBIT Growth / Revenue Growth (YoY, averaged)
    op_lever_avg = _compute_operating_leverage_avg(financials, years)
    
    # Financial Leverage = EPS Growth / EBIT Growth (YoY, averaged)
    fin_lever_avg = _compute_financial_leverage_avg(financials, years)
    
    # Volume & Price Lever (proxy-based)
    price_lever = compute_price_lever(financials, years)
    
    # ─── 1. Earnings Growth Profile ───
    earnings_profile = {
        "pat_cagr_3yr": pat_cagr_3,
        "pat_cagr_5yr": pat_cagr_5,
    }
    
    # ─── 2. Lever Table ───
    lever_table = [
        {
            "lever": "Volume Growth",
            "status": _classify_volume_status(rev_cagr_5, price_lever),
            "analysis": _generate_volume_analysis(
                rev_cagr_5, price_lever, financials
            ),
        },
        {
            "lever": "Price Lever",
            "status": price_lever.value,
            "analysis": _generate_price_analysis(
                rev_cagr_5, price_lever, financials
            ),
        },
        {
            "lever": "Operating Lever",
            "status": _classify_op_lever(op_lever_avg, ebit_cagr_5, rev_cagr_5),
            "analysis": _generate_op_lever_analysis(
                op_lever_avg, ebit_cagr_5, rev_cagr_5, financials
            ),
        },
        {
            "lever": "Financial Lever",
            "status": _classify_fin_lever(fin_lever_avg),
            "analysis": _generate_fin_lever_analysis(
                fin_lever_avg, eps_cagr_5, ebit_cagr_5, financials
            ),
        },
    ]
    
    # ─── 3. Growth Synthesis ───
    growth_synthesis = _synthesize_growth_quality(
        pat_cagr_3, pat_cagr_5, op_lever_avg, fin_lever_avg, price_lever
    )
    
    # ─── 4. Valuation Reality Check ───
    current_pe = financials.iloc[-1].get("pe_ratio", None)
    trailing_peg = (current_pe / pat_cagr_5) if (current_pe and pat_cagr_5 > 0) else None
    
    valuation_check = {
        "current_pe": current_pe,
        "pat_cagr_5yr": pat_cagr_5,
        "trailing_peg": trailing_peg,
        "verdict": _peg_verdict(trailing_peg, growth_synthesis["quality_flag"]),
    }
    
    return {
        "earnings_profile": earnings_profile,
        "lever_table": lever_table,
        "growth_synthesis": growth_synthesis,
        "valuation_check": valuation_check,
    }


def _classify_volume_status(rev_cagr_5, price_lever_result):
    """Classify volume growth: Strong / Moderate / Weak / Declining."""
    est_volume = price_lever_result.details.get("estimated_volume_growth", 0)
    if est_volume >= 15:
        return "Strong organic volume growth"
    elif est_volume >= 8:
        return "Moderate volume growth"
    elif est_volume >= 0:
        return "Weak volume growth"
    else:
        return "Volume declining"


def _classify_op_lever(op_lever_avg, ebit_cagr, rev_cagr):
    """Classify operating leverage status."""
    if op_lever_avg >= 1.3:
        return "Strong positive operating leverage"
    elif op_lever_avg >= 1.0:
        return "Mild operating leverage"
    elif op_lever_avg >= 0.8:
        return "Neutral — margins stable"
    else:
        return "Negative operating leverage (margin compression)"


def _classify_fin_lever(fin_lever_avg):
    """Classify financial leverage status."""
    if fin_lever_avg >= 1.5:
        return "⚠ High financial leverage — debt-amplified"
    elif fin_lever_avg >= 1.1:
        return "Moderate positive financial leverage"
    elif fin_lever_avg >= 0.8:
        return "Neutral — minimal debt impact"
    else:
        return "Negative financial leverage (deleveraging)"


def _synthesize_growth_quality(pat_3, pat_5, op_lever, fin_lever, price_lever):
    """
    Determine the primary growth driver and flag quality.
    
    High quality: Volume + Operating Leverage
    Moderate: Volume + Price
    Low quality: Financial Leverage + Price Hikes
    Risky: Financial Leverage dominant
    """
    drivers = []
    
    # Check volume
    vol_growth = price_lever.details.get("estimated_volume_growth", 0)
    if vol_growth >= 10:
        drivers.append("Volume expansion")
    
    # Check pricing power
    if price_lever.value in ("strong_pricing_power", "moderate_pricing"):
        drivers.append("Price realization")
    
    # Check operating leverage
    if op_lever >= 1.1:
        drivers.append("Operating leverage")
    
    # Check financial leverage
    if fin_lever >= 1.3:
        drivers.append("Financial leverage")
    
    # Determine quality flag
    if "Volume expansion" in drivers and "Operating leverage" in drivers:
        quality = "high_quality"
    elif "Volume expansion" in drivers and "Price realization" in drivers:
        quality = "moderate"
    elif "Financial leverage" in drivers and len(drivers) == 1:
        quality = "risky"
    elif "Financial leverage" in drivers:
        quality = "low_quality"
    else:
        quality = "moderate"
    
    # Build narrative
    narrative_parts = []
    narrative_parts.append(
        f"3-year PAT CAGR of {pat_3:.1f}% and 5-year PAT CAGR of {pat_5:.1f}%."
    )
    if drivers:
        narrative_parts.append(f"Growth primarily driven by: {', '.join(drivers)}.")
    
    if quality == "high_quality":
        narrative_parts.append(
            "This is high-quality growth — organic volume expansion "
            "amplified by operating scale benefits."
        )
    elif quality == "risky":
        narrative_parts.append(
            "⚠ FLAG: Growth is primarily driven by financial leverage "
            "(debt amplification). This is low-quality, unsustainable growth "
            "that amplifies returns in good times but accelerates losses in downturns."
        )
    elif quality == "low_quality":
        narrative_parts.append(
            "⚠ FLAG: Significant financial leverage contribution detected. "
            "Growth quality is compromised — not purely operating-driven."
        )
    
    return {
        "primary_drivers": drivers,
        "quality_flag": quality,
        "narrative": " ".join(narrative_parts),
    }


def _peg_verdict(trailing_peg, quality_flag):
    """
    One-sentence PEG verdict per the 100-bagger golden rule.
    PEG < 1.0 is attractive. But quality matters — a PEG of 0.5 
    driven by debt-fueled growth is a trap.
    """
    if trailing_peg is None:
        return "PEG cannot be computed (negative or zero earnings growth)."
    
    if trailing_peg < 1.0:
        if quality_flag in ("high_quality", "moderate"):
            return (
                f"Trailing PEG of {trailing_peg:.2f}x is below 1.0 — "
                f"the golden rule for 100-baggers. Combined with "
                f"{quality_flag.replace('_', ' ')} earnings drivers, "
                f"the valuation appears justified and attractive."
            )
        else:
            return (
                f"Trailing PEG of {trailing_peg:.2f}x is below 1.0, but the "
                f"growth quality is flagged as '{quality_flag.replace('_', ' ')}'. "
                f"Low PEG driven by leveraged or unsustainable growth is a "
                f"value trap signal — proceed with caution."
            )
    elif trailing_peg < 2.0:
        return (
            f"Trailing PEG of {trailing_peg:.2f}x is between 1.0-2.0 — "
            f"fairly valued relative to growth. Not a screaming bargain "
            f"but acceptable if growth quality is high."
        )
    else:
        return (
            f"Trailing PEG of {trailing_peg:.2f}x is above 2.0 — "
            f"the market is pricing in significantly higher growth "
            f"than recent history. Risk of valuation correction if "
            f"growth decelerates."
        )
```

### 2.5 Updated Report Template — Growth Decomposition Section

The Jinja2 template for the Growth Decomposition section (Section 6 in the HTML report) is restructured as follows:

```html
<!-- templates/sections/growth_decomposition.html -->

<section id="growth-decomposition">
  <h2>6. Earnings Quality & Growth Decomposition</h2>
  
  <!-- 6.1 Earnings Growth Profile -->
  <h3>6.1 Earnings Growth Profile</h3>
  <table class="metric-table">
    <thead>
      <tr><th>Metric</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>3-Year PAT CAGR</td>
        <td class="{{ 'green' if growth.earnings_profile.pat_cagr_3yr >= 20 else 'amber' if growth.earnings_profile.pat_cagr_3yr >= 10 else 'red' }}">
          {{ "%.1f"|format(growth.earnings_profile.pat_cagr_3yr) }}%
        </td>
      </tr>
      <tr>
        <td>5-Year PAT CAGR</td>
        <td class="{{ 'green' if growth.earnings_profile.pat_cagr_5yr >= 20 else 'amber' if growth.earnings_profile.pat_cagr_5yr >= 10 else 'red' }}">
          {{ "%.1f"|format(growth.earnings_profile.pat_cagr_5yr) }}%
        </td>
      </tr>
    </tbody>
  </table>
  
  <!-- 6.2 The 4-Lever Earnings Decomposition -->
  <h3>6.2 The 4-Lever Earnings Decomposition</h3>
  <p class="methodology-note">
    <em>Decomposes EPS growth: ΔEPS = ΔVolume × (ΔSales/ΔVolume) × (ΔEBIT/ΔSales) × (ΔEPS/ΔEBIT)</em>
  </p>
  <table class="lever-table">
    <thead>
      <tr><th>Lever</th><th>Status / Observation</th><th>Analysis</th></tr>
    </thead>
    <tbody>
      {% for lever in growth.lever_table %}
      <tr>
        <td><strong>{{ lever.lever }}</strong></td>
        <td>{{ lever.status }}</td>
        <td>{{ lever.analysis }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  
  <!-- 6.3 Growth Synthesis -->
  <h3>6.3 Growth Synthesis</h3>
  <div class="growth-synthesis">
    <div class="quality-badge {{ growth.growth_synthesis.quality_flag }}">
      {{ growth.growth_synthesis.quality_flag | replace("_", " ") | title }}
    </div>
    <ul>
      {% for driver in growth.growth_synthesis.primary_drivers %}
      <li><strong>{{ driver }}</strong> is a primary growth driver</li>
      {% endfor %}
    </ul>
    <p>{{ growth.growth_synthesis.narrative }}</p>
  </div>
  
  <!-- 6.4 Valuation Reality Check -->
  <h3>6.4 Valuation Reality Check</h3>
  <table class="metric-table">
    <thead>
      <tr><th>Metric</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Current P/E Ratio</td>
        <td>{{ "%.1f"|format(growth.valuation_check.current_pe) }}x</td>
      </tr>
      <tr>
        <td>5-Year PAT CAGR</td>
        <td>{{ "%.1f"|format(growth.valuation_check.pat_cagr_5yr) }}%</td>
      </tr>
      <tr>
        <td>Trailing PEG Ratio</td>
        <td class="{{ 'green' if growth.valuation_check.trailing_peg and growth.valuation_check.trailing_peg < 1.0 else 'amber' if growth.valuation_check.trailing_peg and growth.valuation_check.trailing_peg < 2.0 else 'red' }}">
          {{ "%.2f"|format(growth.valuation_check.trailing_peg) }}x
        </td>
      </tr>
    </tbody>
  </table>
  <div class="verdict-box">
    <strong>Verdict:</strong> {{ growth.valuation_check.verdict }}
  </div>
</section>
```

### 2.6 Updated Markdown Report Template

The corresponding markdown report section:

````markdown
## 6. Earnings Quality & Growth Decomposition

### 6.1 Earnings Growth Profile

| Metric | Value |
|--------|-------|
| 3-Year PAT CAGR | {{ pat_cagr_3yr }}% |
| 5-Year PAT CAGR | {{ pat_cagr_5yr }}% |

### 6.2 The 4-Lever Earnings Decomposition

*Formula: ΔEPS = ΔVolume × (ΔSales/ΔVolume) × (ΔEBIT/ΔSales) × (ΔEPS/ΔEBIT)*

| Lever | Status / Observation | Analysis |
|-------|---------------------|----------|
| **Volume Growth** | {{ volume_status }} | {{ volume_analysis }} |
| **Price Lever** | {{ price_status }} | {{ price_analysis }} |
| **Operating Lever** | {{ op_lever_status }} | {{ op_lever_analysis }} |
| **Financial Lever** | {{ fin_lever_status }} | {{ fin_lever_analysis }} |

### 6.3 Growth Synthesis

**Growth Quality: {{ quality_grade }}**

{{ growth_narrative }}

### 6.4 Valuation Reality Check

| Metric | Value |
|--------|-------|
| Current P/E Ratio | {{ current_pe }}x |
| 5-Year PAT CAGR | {{ pat_cagr_5yr }}% |
| Trailing PEG Ratio | {{ trailing_peg }}x |

**Verdict:** {{ peg_verdict }}
````

### 2.7 LLM Integration — Growth Section Enhancement

The expanded growth section also changes how the LLM interacts with growth data. In the LLM layer:

**Pass 1 (Qualitative Assessment)** — QGLP Checklist Questions Q9 and Q10 now receive richer pre-computed context:

```yaml
growth_checklist:
  Q9:
    question: "What is the addressable market opportunity and its key drivers?"
    input_sources: ["sector_context", "annual_report"]
    pre_computed_context:
      # v4 NEW: Include lever decomposition for LLM to reason about
      - "lever_decomposition.earnings_profile"
      - "lever_decomposition.lever_table"
    output_type: "text"
  Q10:
    question: "What is the company's growth plan? How sustainable is the growth?"
    input_sources: ["annual_report", "computed_metrics.growth_quality_grade"]
    pre_computed_context:
      # v4 NEW: Include full synthesis for LLM
      - "lever_decomposition.growth_synthesis"
      - "lever_decomposition.valuation_check"
    output_type: "text_with_score"
```

**Pass 2 (Synthesis)** — The investment thesis prompt now includes the growth synthesis and PEG verdict as structured input:

```
INPUT (updated for v4):
- SQGLP Metrics: {sqglp_report.json}
- Qualitative Assessment: {pass_1_output}
- Current SQGLP Score: {composite_score}
- Growth Quality Report: {lever_decomposition}  ← NEW
  - Earnings Profile: {pat_cagr_3yr, pat_cagr_5yr}
  - 4-Lever Decomposition: {lever_table}
  - Growth Synthesis: {quality_flag, primary_drivers, narrative}
  - Valuation Reality Check: {trailing_peg, verdict}
```

### 2.8 Updated Report Section List

The full report section list (v4):

1. **Executive Summary** — One-paragraph thesis, conviction level, SQGLP radar chart
2. **SQGLP Score Dashboard** — Composite score + element breakdown + flags
3. **Size Analysis** — Market cap, institutional ownership, discovery potential
4. **Quality Scorecard** — Business quality ratios (table with sparklines for trends)
5. **Management Assessment** — Quantitative signals + LLM qualitative assessment
6. **Earnings Quality & Growth Decomposition** ← EXPANDED in v4
   - 6.1 Earnings Growth Profile (3yr/5yr PAT CAGR table)
   - 6.2 The 4-Lever Earnings Decomposition (Volume, Price, Operating, Financial)
   - 6.3 Growth Synthesis (primary drivers, quality flag, narrative)
   - 6.4 Valuation Reality Check (P/E, PEG, verdict)
7. **Longevity Assessment** — Consistency metrics, moat analysis
8. **Valuation Analysis** — P/E band chart, DCF, reverse DCF (detailed; Section 6.4 is a quick check)
9. **Peer Comparison** — Side-by-side table, industry peers only ← CHANGED in v4
10. **Peer Discovery** — How peers were identified (industry-only pipeline)
11. **Investment Thesis** — Bull/bear case, kill-the-thesis scenarios
12. **Risk Register** — Red flags, computed warnings, LLM-identified risks
13. **Monitorables Checklist** — What to track quarterly
14. **Appendix** — Raw data tables, methodology, data sources

---

## Summary of All Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `data_fetcher/peer_discovery.py` | **Modified** | Removed `_rank_by_financial_similarity()`, removed `financial_peers` from `PeerResult`, renumbered layers |
| `config.yaml` | **Modified** | Removed `include_financial_peers` option |
| `elements/growth.yaml` | **Modified** | Added `revenue_cagr_3yr`, `ebit_cagr_5yr`, `ebit_cagr_3yr`, `price_lever_signal` |
| `builtin/growth.py` | **Modified** | Added `compute_price_lever()`, `compute_lever_decomposition_table()`, and all helper functions |
| `templates/sections/growth_decomposition.html` | **Rewritten** | New 4-subsection structure with tables, synthesis, and PEG verdict |
| `templates/sqglp_report.md` | **Modified** | Growth section expanded to match HTML template |
| `llm_layer/prompts/pass1_checklist_template.yaml` | **Modified** | Q9, Q10 now receive lever decomposition as pre-computed context |
| `llm_layer/prompts/pass2_synthesis.txt` | **Modified** | Includes growth quality report in synthesis input |

---

## Appendix: Example Report Output (Growth Section)

For illustration, here's what the expanded Growth section would look like for a hypothetical company (Astral Ltd):

### 6.1 Earnings Growth Profile

| Metric | Value |
|--------|-------|
| 3-Year PAT CAGR | 28.3% |
| 5-Year PAT CAGR | 22.7% |

### 6.2 The 4-Lever Earnings Decomposition

*Formula: ΔEPS = ΔVolume × (ΔSales/ΔVolume) × (ΔEBIT/ΔSales) × (ΔEPS/ΔEBIT)*

| Lever | Status / Observation | Analysis |
|-------|---------------------|----------|
| **Volume Growth** | Strong organic volume growth | Unit deliveries grew at ~17% CAGR over 5 years, significantly above industry average of 10%. The company has been gaining market share through geographic expansion and new product categories (CPVC, SWR fittings). |
| **Price Lever** | Moderate pricing power | Revenue CAGR (24.5%) exceeds estimated volume growth (17%), indicating ~7pp of price realization. This is partly driven by product mix shift toward higher-value CPVC products and partly by raw material pass-through. Sustainable pricing power is moderate — the company cannot raise prices arbitrarily as PVC is partially commoditized. |
| **Operating Lever** | Strong positive operating leverage (1.4x) | EBIT grew 1.4x faster than revenue over 5 years. Operating margins expanded from 14% to 18% as fixed manufacturing costs were spread over a larger production base. The Ghiloth plant scale-up was a key driver. This is the hallmark of a high-fixed-cost business hitting scale. |
| **Financial Lever** | Neutral — minimal debt impact (0.95x) | EPS growth tracked EBIT growth closely (0.95x ratio). The company operates with near-zero debt (D/E of 0.08), so there is no financial leverage amplification. This is a positive signal — growth is entirely operating-driven, not debt-fueled. |

### 6.3 Growth Synthesis

**Growth Quality: High Quality**

- 3-year PAT CAGR of 28.3% and 5-year PAT CAGR of 22.7%.
- Growth primarily driven by: **Volume expansion** and **Operating leverage**.
- This is high-quality growth — organic volume expansion amplified by operating scale benefits. No debt amplification detected. The combination of strong volume growth with expanding operating margins is the strongest possible growth profile for a potential 100-bagger.

### 6.4 Valuation Reality Check

| Metric | Value |
|--------|-------|
| Current P/E Ratio | 52.3x |
| 5-Year PAT CAGR | 22.7% |
| Trailing PEG Ratio | 2.30x |

**Verdict:** Trailing PEG of 2.30x is above 2.0 — the market is pricing in significantly higher growth than recent history. Risk of valuation correction if growth decelerates. Despite high-quality earnings drivers, the entry price demands either (a) acceleration in earnings growth, or (b) patience to hold through a potential de-rating phase.
