"""QGLP checklist — maps computed metrics to structured LLM context."""

from boundless100x.compute_engine.metrics.base import MetricResult


def build_quality_metrics_context(
    metrics: dict[str, MetricResult],
    scores: dict,
) -> str:
    """Format computed quality/growth/longevity metrics for LLM prompt."""
    lines = []

    metric_display = {
        "roce_5yr_avg": ("RoCE 5yr avg", "%"),
        "roe_5yr_avg": ("ROE 5yr avg", "%"),
        "operating_margin_5yr": ("OPM 5yr avg", "%"),
        "dupont_margin": ("DuPont: Net Margin", "%"),
        "dupont_turnover": ("DuPont: Asset Turnover", "x"),
        "dupont_equity_multiplier": ("DuPont: Equity Multiplier", "x"),
        "cash_conversion": ("Cash Conversion (CFO/PAT)", "%"),
        "fcf_yield": ("FCF Yield", "%"),
        "debt_equity": ("Debt/Equity", "x"),
        "interest_coverage": ("Interest Coverage", "x"),
        "revenue_cagr_5yr": ("Revenue CAGR 5yr", "%"),
        "pat_cagr_5yr": ("PAT CAGR 5yr", "%"),
        "pat_cagr_3yr": ("PAT CAGR 3yr", "%"),
        "eps_cagr_5yr": ("EPS CAGR 5yr", "%"),
        "operating_leverage": ("Operating Leverage", "x"),
        "financial_leverage_ratio": ("Financial Leverage", "x"),
        "revenue_growth_consistency": ("Revenue Growth StdDev", "%"),
        "roce_consistency": ("RoCE Consistency (yrs >15%)", "yrs"),
        "revenue_growth_streak": ("Revenue Growth Streak", "yrs"),
        "gross_margin_stability": ("Margin Stability (StdDev)", "%"),
        "reinvestment_rate": ("Reinvestment Rate", "%"),
        "fcf_consistency": ("FCF Positive Years", "yrs"),
        "pe_ttm": ("PE TTM", "x"),
        "peg_ratio": ("PEG Ratio", "x"),
        "trailing_peg": ("Trailing PEG (3yr)", "x"),
        "ev_ebitda": ("EV/EBITDA", "x"),
        "earnings_yield_spread": ("Earnings Yield Spread", "%"),
        "market_cap": ("Market Cap", "₹Cr"),
        "institutional_holding": ("Institutional Holding", "%"),
        "promoter_holding_trend": ("Promoter Holding Δ (3yr)", "pp"),
        "promoter_pledge": ("Promoter Pledge", "%"),
        "equity_dilution": ("Equity Dilution (5yr)", "%"),
    }

    for metric_id, (label, unit) in metric_display.items():
        result = metrics.get(metric_id)
        if result and result.ok:
            val = result.value
            if isinstance(val, float):
                val = round(val, 2)
            lines.append(f"- {label}: {val}{unit}")
            if result.flags:
                lines.append(f"  Flags: {', '.join(result.flags)}")

    return "\n".join(lines) if lines else "No computed quality metrics available."


def build_flags_context(metrics: dict[str, MetricResult]) -> str:
    """Extract all computed flags across metrics for LLM context."""
    all_flags = []
    for metric_id, result in metrics.items():
        if result.ok and result.flags:
            for flag in result.flags:
                all_flags.append(f"[{metric_id}] {flag}")

    return "\n".join(all_flags) if all_flags else "No flags raised."


def build_promoter_context(metrics: dict[str, MetricResult]) -> str:
    """Format promoter-related metrics for LLM prompt."""
    lines = []

    promoter = metrics.get("promoter_holding_trend")
    if promoter and promoter.ok:
        lines.append(f"Promoter holding change (3yr): {promoter.value:.2f} pp")
        if promoter.metadata:
            lines.append(f"  Latest: {promoter.metadata.get('latest_holding', 'N/A')}%")
            lines.append(f"  3yr ago: {promoter.metadata.get('earliest_holding', 'N/A')}%")

    pledge = metrics.get("promoter_pledge")
    if pledge and pledge.ok:
        lines.append(f"Promoter pledge: {pledge.value:.1f}%")

    dilution = metrics.get("equity_dilution")
    if dilution and dilution.ok:
        lines.append(f"Equity dilution (5yr): {dilution.value:.1f}%")

    return "\n".join(lines) if lines else "No promoter data available."


def build_scores_summary(scores: dict) -> str:
    """Format SQGLP scores for LLM prompt."""
    elements = scores.get("elements", {})
    composite = scores.get("composite", "N/A")

    element_names = {
        "size": "Size (S)",
        "quality_business": "Quality - Business (Q)",
        "quality_management": "Quality - Management (Q)",
        "growth": "Growth (G)",
        "longevity": "Longevity (L)",
        "price": "Price (P)",
    }

    lines = []
    for el_key, label in element_names.items():
        score = elements.get(el_key)
        if score is not None:
            lines.append(f"- {label}: {score:.1f}/10")
        else:
            lines.append(f"- {label}: N/A")

    lines.append(f"\nComposite SQGLP Score: {composite}/10")
    return "\n".join(lines)


def build_key_metrics_context(
    metrics: dict[str, MetricResult], scores: dict
) -> str:
    """Build condensed key metrics context for Pass 2."""
    lines = []

    key_metrics = [
        ("roce_5yr_avg", "RoCE 5yr avg", "%"),
        ("pat_cagr_5yr", "PAT CAGR 5yr", "%"),
        ("pat_cagr_3yr", "PAT CAGR 3yr", "%"),
        ("revenue_cagr_5yr", "Revenue CAGR 5yr", "%"),
        ("operating_margin_5yr", "OPM 5yr", "%"),
        ("cash_conversion", "Cash Conversion", "%"),
        ("debt_equity", "Debt/Equity", "x"),
        ("pe_ttm", "PE TTM", "x"),
        ("peg_ratio", "PEG", "x"),
        ("trailing_peg", "Trailing PEG", "x"),
        ("ev_ebitda", "EV/EBITDA", "x"),
        ("fcf_yield", "FCF Yield", "%"),
        ("roce_consistency", "RoCE >15% years", "yrs"),
        ("reinvestment_rate", "Reinvestment Rate", "%"),
        ("promoter_holding_trend", "Promoter Δ 3yr", "pp"),
        ("operating_leverage", "Op Leverage", "x"),
        ("market_cap", "Market Cap", "₹Cr"),
    ]

    for metric_id, label, unit in key_metrics:
        result = metrics.get(metric_id)
        if result and result.ok:
            val = result.value
            if isinstance(val, float):
                val = round(val, 2)
            lines.append(f"- {label}: {val}{unit}")

    return "\n".join(lines) if lines else "No key metrics available."


def build_qg_quadrant_context(metrics: dict[str, MetricResult]) -> str:
    """Format Quality-Growth matrix position for LLM."""
    qg = metrics.get("quality_growth_quadrant")
    if not qg or not qg.ok:
        return "Quality-Growth quadrant: not computed"

    quadrant_labels = {
        "true_wealth_creator": "True Wealth Creator (High Quality + High Growth)",
        "quality_trap": "Quality Trap (High Quality + Low Growth)",
        "growth_trap": "Growth Trap (Low Quality + High Growth)",
        "wealth_destroyer": "Wealth Destroyer (Low Quality + Low Growth)",
    }

    label = quadrant_labels.get(qg.value, qg.value)
    meta = qg.metadata or {}

    return (
        f"Quality-Growth Matrix: {label}\n"
        f"  Avg RoCE: {meta.get('avg_roce', 'N/A'):.1f}% "
        f"(threshold: {meta.get('quality_threshold', 15)}%)\n"
        f"  PAT CAGR: {meta.get('pat_cagr', 'N/A'):.1f}% "
        f"(threshold: {meta.get('growth_threshold', 15)}%)"
    )


def build_growth_decomposition_context(growth_decomposition: dict | None) -> str:
    """Format growth decomposition data for LLM context (v4)."""
    if not growth_decomposition:
        return "No growth decomposition data available."

    lines = []

    # Earnings Profile
    ep = growth_decomposition.get("earnings_profile", {})
    pat_3 = ep.get("pat_cagr_3yr")
    pat_5 = ep.get("pat_cagr_5yr")
    lines.append("Earnings Growth Profile:")
    lines.append(f"  3-Year PAT CAGR: {pat_3:.1f}%" if pat_3 is not None else "  3-Year PAT CAGR: N/A")
    lines.append(f"  5-Year PAT CAGR: {pat_5:.1f}%" if pat_5 is not None else "  5-Year PAT CAGR: N/A")

    # Lever Table
    lines.append("\n4-Lever Earnings Decomposition:")
    for lever in growth_decomposition.get("lever_table", []):
        lines.append(f"  {lever['lever']}: {lever['status']}")
        lines.append(f"    {lever['analysis']}")

    # Growth Synthesis
    synthesis = growth_decomposition.get("growth_synthesis", {})
    quality = synthesis.get("quality_flag", "N/A")
    drivers = synthesis.get("primary_drivers", [])
    narrative = synthesis.get("narrative", "")
    lines.append(f"\nGrowth Quality: {quality.replace('_', ' ').title()}")
    if drivers:
        lines.append(f"Primary drivers: {', '.join(drivers)}")
    if narrative:
        lines.append(f"Narrative: {narrative}")

    # Valuation Check
    vc = growth_decomposition.get("valuation_check", {})
    pe = vc.get("current_pe")
    peg = vc.get("trailing_peg")
    verdict = vc.get("verdict", "")
    lines.append("\nValuation Reality Check:")
    lines.append(f"  Current P/E: {pe:.1f}x" if pe is not None else "  Current P/E: N/A")
    lines.append(f"  Trailing PEG: {peg:.2f}x" if peg is not None else "  Trailing PEG: N/A")
    if verdict:
        lines.append(f"  Verdict: {verdict}")

    return "\n".join(lines)
