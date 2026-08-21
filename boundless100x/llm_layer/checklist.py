"""QGLP checklist — maps computed metrics to structured LLM context."""

from functools import lru_cache

from boundless100x.compute_engine.metrics.base import UNSCORABLE_FLAGS, MetricResult
from boundless100x.compute_engine.sector import (
    classify_sector,
    structure_caveat,
    study_findings,
    study_labels,
)


# ── What the score did NOT count ──────────────────────────────────────────
#
# Two ways a computed figure can fail to reach a score, and the model needs to
# be told which, because they call for different reasoning:
#
#   WITHDRAWN — the sector table says the metric measures nothing for a company
#   of this kind. Shown, because the observation can still matter (4x leverage
#   is 4x leverage), but never as a mark out of ten. Left unmarked, Pass 2 read
#   JIOFIN's EV/EBITDA of 78.05x — a ratio the same run had already called
#   meaningless for a lender — and spent a red flag on it.
#
#   NOT A READING — arithmetically correct and not evidence, e.g. a 269% CAGR
#   measured from a post-demerger base of ₹31 Cr. The scorer waives these and
#   the eligibility gates refuse them; the prompt has to say so too, or the
#   model is the only layer still treating the number as a fact.


def _withdrawn_metrics(scores: dict | None) -> dict[str, str]:
    return dict((scores or {}).get("not_applicable") or {})


def _unscorable_metrics(scores: dict | None) -> dict[str, str]:
    details = (scores or {}).get("details") or {}
    return {
        metric_id: detail.get("waived", "")
        for metric_id, detail in details.items()
        if detail.get("waived") == "not_a_reading"
    }


def _score_status(metric_id: str, withdrawn: dict, unscorable: dict) -> str:
    if metric_id in withdrawn:
        return "  [WITHDRAWN — does not measure anything for this kind of company"
    if metric_id in unscorable:
        return "  [NOT A READING — excluded from the score and from the 100x gates"
    return ""


# ── Labels and units come from the registry, never from a list here ───────
#
# **Every metric already declares its own name and unit**, in the YAML the
# engine validates at startup. A second copy in this module is a copy that
# drifts, and it drifted three times before this was written — each time
# putting a confident, wrong sentence into the model's context:
#
#   * `promoter_holding_trend` was labelled "Promoter Holding Δ (3yr) pp" while
#     the metric returns the LEVEL, so Pass 2 was told promoters had bought
#     32pp of Edelweiss and spent a red flag disproving it;
#   * `cash_conversion` was labelled "Cash Conversion (CFO/PAT)" while the
#     metric computes OCF/EBITDA. Caplin Point's thesis therefore led with
#     "CFO/PAT of 61.5% — two-fifths of reported profit has not converted",
#     set a kill-trigger at "CFO/PAT below 70%", and printed a table on the
#     same page showing CFO/PAT at 80% for two straight years;
#   * `reinvestment_rate` was labelled "%" while the metric returns a multiple,
#     so the model reported "heavy_reinvestment attached to a 6.35% rate" as an
#     internal contradiction that did not exist.
#
# The registry is the single statement of what a metric is called and what it
# is denominated in; this reads it rather than restating it.

_UNIT_SUFFIX = {
    "percent": "%",
    "percentage_points": "pp",
    "multiple": "x",
    "inr_crore": " Cr",
    "years": " yrs",
    "days": " days",
    "count": "",
    "percentile": "th percentile",
    "category": "",
}


@lru_cache(maxsize=1)
def _registry_display() -> dict[str, tuple[str, str]]:
    """metric id -> (declared name, unit suffix), read off the metric registry.

    Cached because it constructs a `ComputeEngine`, and the registry cannot
    change inside a run. Degrades to an empty map rather than raising: a
    prompt built with fallback labels is worse than one built from the
    registry and far better than no analysis at all.
    """
    try:
        from boundless100x.compute_engine.engine import ComputeEngine

        engine = ComputeEngine()
    except Exception:  # pragma: no cover - defensive
        return {}

    display = {}
    for metric_id, config in engine.metrics.items():
        presentation = config.get("presentation") or {}
        unit = _UNIT_SUFFIX.get(presentation.get("unit", ""), "")
        display[metric_id] = (config.get("name", metric_id), unit)
    return display


def _label_for(metric_id: str) -> tuple[str, str]:
    """The declared name and unit, or a readable fallback for an unknown id."""
    known = _registry_display().get(metric_id)
    if known:
        return known
    return metric_id.replace("_", " ").title(), ""


def build_quality_metrics_context(
    metrics: dict[str, MetricResult],
    scores: dict,
) -> str:
    """Format computed quality/growth/longevity metrics for LLM prompt."""
    lines = []
    withdrawn = _withdrawn_metrics(scores)
    unscorable = _unscorable_metrics(scores)

    # Ids only — the label and unit come from each metric's own declaration.
    metric_ids = [
        "roce_5yr_avg", "roe_5yr_avg", "roa_5yr_avg", "operating_margin_5yr",
        "dupont_margin", "dupont_turnover", "dupont_equity_multiplier",
        "cash_conversion", "fcf_yield", "debt_equity", "interest_coverage",
        "revenue_cagr_5yr", "pat_cagr_5yr", "pat_cagr_3yr", "eps_cagr_5yr",
        "book_value_cagr_5yr", "operating_leverage", "financial_leverage_ratio",
        "revenue_growth_consistency", "roce_consistency", "roe_consistency",
        "revenue_growth_streak", "gross_margin_stability", "reinvestment_rate",
        "fcf_consistency", "pe_ttm", "peg_ratio", "trailing_peg",
        "price_to_book", "ev_ebitda", "earnings_yield_vs_gsec", "market_cap",
        "institutional_holding", "promoter_holding_trend", "promoter_pledge",
        "equity_dilution",
    ]

    for metric_id in metric_ids:
        label, unit = _label_for(metric_id)
        result = metrics.get(metric_id)
        if result and result.ok:
            val = result.value
            if isinstance(val, float):
                val = round(val, 2)
            lines.append(f"- {label}: {val}{unit}")
            status = _score_status(metric_id, withdrawn, unscorable)
            if status:
                reason = withdrawn.get(metric_id, "")
                lines.append(f"{status}{(': ' + reason) if reason else ''}]")
            elif result.flags:
                lines.append(f"  Flags: {', '.join(result.flags)}")

    return "\n".join(lines) if lines else "No computed quality metrics available."


def build_flags_context(
    metrics: dict[str, MetricResult], scores: dict | None = None
) -> str:
    """Extract all computed flags across metrics for LLM context.

    Flags from a withdrawn metric are dropped unless its table entry kept them
    — the same rule the report's signal list applies, so the model and the
    reader are not shown different evidence about the same company.
    """
    suppressed = set((scores or {}).get("flags_suppressed") or ())
    unscorable = set(_unscorable_metrics(scores))
    all_flags = []
    for metric_id, result in metrics.items():
        if metric_id in suppressed:
            continue
        if result.ok and result.flags:
            for flag in result.flags:
                # A flag derived from an artefact is an artefact; the flag
                # naming the artefact is not.
                if metric_id in unscorable and flag not in UNSCORABLE_FLAGS:
                    continue
                all_flags.append(f"[{metric_id}] {flag}")

    return "\n".join(all_flags) if all_flags else "No flags raised."


def build_promoter_context(metrics: dict[str, MetricResult]) -> str:
    """Format promoter-related metrics for LLM prompt."""
    lines = []

    promoter = metrics.get("promoter_holding_trend")
    if promoter and promoter.ok:
        # `value` is the latest **level**, never the change — see
        # compute_promoter_trend. Labelling it as a delta told the model that
        # promoters had bought 32pp of Edelweiss, which is what a
        # reclassification looks like when the level is read as a change.
        meta = promoter.metadata or {}
        lines.append(f"Promoter holding (latest quarter): {promoter.value:.2f}%")
        change = meta.get("change_pp")
        if change is not None:
            quarters = meta.get("quarters_used")
            window = f" over {quarters} quarters" if quarters else ""
            earliest = meta.get("earliest_pct")
            since = f", from {earliest:.2f}%" if isinstance(earliest, (int, float)) else ""
            lines.append(f"  Change{window}: {change:+.2f} pp{since}")

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
    withdrawn = _withdrawn_metrics(scores)
    unscorable = _unscorable_metrics(scores)

    # Ids only — see `_registry_display`. The Pass 1 list carries the same
    # rule, and both had drifted from the registry in different ways.
    key_metric_ids = [
        "roce_5yr_avg", "roa_5yr_avg", "roiic", "pat_cagr_5yr", "pat_cagr_3yr",
        "revenue_cagr_5yr", "book_value_cagr_5yr", "operating_margin_5yr",
        "cash_conversion", "debt_equity", "pe_ttm", "peg_ratio", "trailing_peg",
        "price_to_book", "ev_ebitda", "fcf_yield", "roce_consistency",
        "reinvestment_rate", "promoter_holding_trend", "operating_leverage",
        "market_cap",
    ]

    for metric_id in key_metric_ids:
        label, unit = _label_for(metric_id)
        result = metrics.get(metric_id)
        if result and result.ok:
            val = result.value
            if isinstance(val, float):
                val = round(val, 2)
            status = _score_status(metric_id, withdrawn, unscorable)
            suffix = ""
            if metric_id in withdrawn:
                suffix = "   [WITHDRAWN — not scored; measures nothing for this kind of company]"
            elif metric_id in unscorable:
                suffix = "   [NOT A READING — an artefact of a tiny base, not scored and not gated]"
            lines.append(f"- {label}: {val}{unit}{suffix}")

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

    rendered = (
        f"Quality-Growth Matrix: {label}\n"
        f"  Avg RoCE: {meta.get('avg_roce', 'N/A'):.1f}% "
        f"(threshold: {meta.get('quality_threshold', 15)}%)\n"
        f"  PAT CAGR: {meta.get('pat_cagr', 'N/A'):.1f}% "
        f"(threshold: {meta.get('growth_threshold', 15)}%)"
    )

    # Pass 2 receives no sector context of its own, and the quadrant is the
    # single reading a group structure most distorts — so the caveat travels
    # here, attached to the claim it qualifies rather than filed somewhere the
    # model has to connect it back.
    caveat = meta.get("structure_caveat")
    if caveat:
        rendered += f"\n\n{caveat}"

    return rendered


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
    # Named by its denominator, because the scored `trailing_peg` metric in
    # `key_metrics` above divides by a different window and the model was
    # shown both under one name.
    peg = vc.get("pe_to_pat_cagr_5yr")
    peg_label = vc.get("peg_label") or "P/E / 5yr PAT CAGR"
    verdict = vc.get("verdict", "")
    lines.append("\nValuation Reality Check:")
    lines.append(f"  Current P/E: {pe:.1f}x" if pe is not None else "  Current P/E: N/A")
    lines.append(
        f"  {peg_label}: {peg:.2f}x" if peg is not None else f"  {peg_label}: N/A"
    )
    if verdict:
        lines.append(f"  Verdict: {verdict}")

    return "\n".join(lines)


def build_eligibility_context(eligibility: dict | None) -> str:
    """The 100x verdict and its gate reasons, for Pass 2's narrative.

    Given so the thesis reads coherently against a verdict the reader will
    see beside it — not as a control. The action a report displays is capped
    in deterministic code (action_policy), so nothing here needs the model to
    comply for the guard to hold.
    """
    if not eligibility or not eligibility.get("verdict"):
        return (
            "100x eligibility: NOT EVALUATED. Do not assert the company is or "
            "is not a hundred-bagger candidate — the gates did not run."
        )

    verdict = eligibility["verdict"]
    gates = eligibility.get("gates", {})
    lines = [f"100x eligibility verdict: {verdict.upper()}"]

    headline = {
        "eligible": "Clears every 100x gate.",
        "not_eligible": (
            "Fails at least one necessary condition for a hundredfold move. "
            "This does not make it a bad investment — only an unlikely "
            "100-bagger. Judge it as a compounder on its merits and say so "
            "plainly."
        ),
        "indeterminate": (
            "A gate could not be evaluated from available data. Treat 100x "
            "potential as unproven rather than absent, and do not assume the "
            "unevaluated gate would have passed."
        ),
    }.get(verdict)
    if headline:
        lines.append(headline)

    for label, gate_ids in (
        ("Failed", eligibility.get("failed", [])),
        ("Could not evaluate", eligibility.get("indeterminate", [])),
    ):
        for gate_id in gate_ids:
            reason = gates.get(gate_id, {}).get("reason")
            if reason:
                lines.append(f"  {label}: {reason}")

    return "\n".join(lines)


def build_sector_context(metadata: dict | None) -> str:
    """Sector classification plus the study findings that frame it, for Pass 1."""
    meta = metadata or {}
    sector = meta.get("sector")
    # Asked with all three breadcrumbs, so the study's bucket is found
    # wherever it happens to be listed — see `classify_sector`.
    classification = classify_sector(study_labels(meta))
    findings = study_findings()

    label = {
        "strong_tailwind": "STRONG TAILWIND — a sector that produced compounders",
        "moderate_tailwind": "MODERATE TAILWIND",
        "non_consideration": "NON-CONSIDERATION — the study ruled this sector out",
        "unknown": "UNKNOWN — sector not available or not in the study's lists",
    }[classification]

    lines = [
        f"Sector: {sector or 'not available'}",
        f"Classification: {label}",
    ]

    business = findings.get("business_type", {})
    if business:
        lines.append(
            f"Business-type finding: 60% of compounders were {business.get('preferred', 'B2C')}; "
            f"{business.get('acceptable', 'B2B')} acceptable, {business.get('caution', 'B2G')} warrants caution."
        )

    leadership = findings.get("leadership", {})
    if leadership:
        lines.append(
            f"Leadership finding: 77% of compounders were market leaders "
            f"(top {leadership.get('preferred_rank', 3)} in their category)."
        )

    if classification == "unknown":
        lines.append(
            "Treat sector as unverified — do not infer a tailwind the data does not support."
        )

    # Pass 1 reads the annual report, which is where segment disclosures
    # actually live — so it is the pass best placed to act on this, provided
    # it is told to look.
    caveat = structure_caveat(meta)
    if caveat:
        lines.append("")
        lines.append(caveat)

    return "\n".join(lines)
