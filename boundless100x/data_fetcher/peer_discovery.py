"""Multi-layer peer discovery pipeline.

Layers 1-2: fully offline, zero LLM cost.
Layer 3: LLM peer validation — validates candidates for competitive relevance.
Layer 4: Value chain mapping (future).

v4 CHANGE: Removed cross-sector financial similarity matching.
Peers are now exclusively from the same industry classification.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.fetch_sector_peers import SectorPeersFetcher
from boundless100x.data_fetcher.fetch_financials import FinancialsFetcher

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "llm_layer" / "prompts"


@dataclass
class PeerResult:
    """Result of multi-layer peer discovery."""

    direct_competitors: list[str] = field(default_factory=list)  # Combined list (backward compat)
    sector_peers: list[str] = field(default_factory=list)
    value_chain: list[str] = field(default_factory=list)
    discovery_metadata: dict = field(default_factory=dict)
    peer_data: dict = field(default_factory=dict)
    peer_categories: dict[str, str] = field(default_factory=dict)  # ticker → "competitor"|"benchmark"


class PeerDiscovery:
    """Industry-focused peer identification.

    Layer 1: Industry classification from Screener.in peer table
    Layer 2: Size-filtered cohort (market cap and revenue bands)
    Layer 3: LLM peer validation — validates competitive relevance (optional)
    Layer 4: Value chain mapping (optional, future)

    v4 CHANGE: Removed cross-sector financial similarity matching.
    Peers are now exclusively from the same industry classification.
    """

    def __init__(self, config: dict):
        pd_config = config.get("peer_discovery", {})
        self.max_peers = pd_config.get("max_peers", 5)
        self.max_benchmarks = pd_config.get("max_benchmarks", 3)
        self.size_band_mult = pd_config.get("size_band_multiplier", 3.0)
        self.revenue_band_mult = pd_config.get("revenue_band_multiplier", 5.0)
        self.min_listing_years = pd_config.get("min_listing_years", 5)

        # Layer 3: LLM peer validation config
        self._llm_validation_enabled = pd_config.get("use_llm_validation", False)
        self._llm_max_suggestions = pd_config.get("max_llm_suggestions", 5)
        llm_config = config.get("llm", {})
        self._peer_validation_model = pd_config.get(
            "peer_validation_model",
            llm_config.get("pass3_model", "claude-haiku-4-5-20251001"),
        )
        self._llm_client = None  # Lazy init

        fetch_config = config.get("fetching", {})
        rate_limit = fetch_config.get("rate_limit_seconds", 2.0)
        cache_ttl = fetch_config.get("cache_ttl_hours", 24)

        self.peers_fetcher = SectorPeersFetcher(
            rate_limit_seconds=rate_limit, cache_ttl_hours=cache_ttl
        )
        self.financials_fetcher = FinancialsFetcher(
            rate_limit_seconds=rate_limit, cache_ttl_hours=cache_ttl
        )

    def discover(
        self,
        ticker: str,
        target_data: dict | None = None,
        use_llm: bool = False,
    ) -> PeerResult:
        """Run multi-layer peer discovery.

        Args:
            ticker: NSE symbol of the target company.
            target_data: Pre-fetched data dict (to avoid re-fetching).
            use_llm: Whether to run Layer 3 LLM validation.

        Returns:
            PeerResult with categorized peer lists.
        """
        logger.info(f"Starting peer discovery for {ticker}")

        # Layer 1: Get raw sector peers from Screener.in
        sector_df = self._layer1_sector_peers(ticker, target_data)
        all_sector_tickers = self._extract_peer_tickers(sector_df, ticker)

        logger.info(f"Layer 1: {len(all_sector_tickers)} sector peers found")

        if not all_sector_tickers:
            return PeerResult(
                discovery_metadata={"error": "No sector peers found", "layers_run": [1]}
            )

        # Layer 2: Filter by size (market cap and revenue bands)
        target_mcap = self._get_target_mcap(sector_df, ticker, target_data)
        target_revenue = self._get_target_revenue(target_data)
        size_filtered = self._layer2_size_filter(
            sector_df, ticker, target_mcap, target_revenue
        )

        logger.info(f"Layer 2: {len(size_filtered)} after size filter")

        # Build set of irrelevant tickers to exclude from backfill too
        irrelevant_tickers = set()
        if not sector_df.empty and "roce" in sector_df.columns and "pe" in sector_df.columns:
            for _, row in sector_df.iterrows():
                pt = row.get("peer_ticker")
                if pt and pt != ticker:
                    roce_val = pd.to_numeric(row.get("roce"), errors="coerce") if row.get("roce") is not None else None
                    pe_val = pd.to_numeric(row.get("pe"), errors="coerce") if row.get("pe") is not None else None
                    roce_ok = roce_val is not None and not pd.isna(roce_val) and roce_val > 0
                    pe_ok = pe_val is not None and not pd.isna(pe_val) and pe_val > 0 and pe_val < 200
                    if not roce_ok and not pe_ok:
                        irrelevant_tickers.add(pt)

        # ── Separate competitors from benchmarks ──
        # Competitors: peers that passed the size filter (same sector + similar size)
        # Benchmarks: backfilled peers from broader sector (for ratio comparison only)
        competitors = size_filtered[: self.max_peers]
        peer_categories = {t: "competitor" for t in competitors}

        # Backfill with benchmarks (NOT competitors) when few pass size filter
        benchmarks = []
        if len(competitors) < self.max_peers:
            mcap_ceiling = 8.0  # Never include peers >8x different in mcap
            rev_ceiling = 15.0  # Never include peers >15x different in revenue
            needed = min(self.max_benchmarks, self.max_peers - len(competitors))
            backfill_candidates = []
            for t in all_sector_tickers:
                if t not in competitors and t not in irrelevant_tickers:
                    peer_row = sector_df[sector_df["peer_ticker"] == t]
                    if peer_row.empty:
                        continue
                    peer_mcap = pd.to_numeric(
                        peer_row["market_cap"].iloc[0], errors="coerce"
                    )
                    if pd.isna(peer_mcap) or peer_mcap <= 0:
                        continue

                    # Hard ceiling on market cap
                    mcap_ratio = 1.0
                    if target_mcap and target_mcap > 0:
                        mcap_ratio = max(peer_mcap, target_mcap) / max(
                            min(peer_mcap, target_mcap), 1
                        )
                        if mcap_ratio > mcap_ceiling:
                            continue

                    # Revenue proximity + hard ceiling
                    rev_ratio = 1.0
                    if target_revenue and target_revenue > 0 and "sales_qtr" in peer_row.columns:
                        peer_rev = pd.to_numeric(
                            peer_row["sales_qtr"].iloc[0], errors="coerce"
                        )
                        if not pd.isna(peer_rev) and peer_rev > 0:
                            rev_ratio = max(peer_rev, target_revenue) / max(
                                min(peer_rev, target_revenue), 1
                            )
                            if rev_ratio > rev_ceiling:
                                continue

                    # Combined proximity score (geometric mean of both ratios)
                    proximity = (mcap_ratio * rev_ratio) ** 0.5
                    backfill_candidates.append((t, proximity))

            # Sort by combined proximity (lowest = closest match)
            backfill_candidates.sort(key=lambda x: x[1])
            for t, _prox in backfill_candidates:
                benchmarks.append(t)
                peer_categories[t] = "benchmark"
                if len(benchmarks) >= needed:
                    break

            if benchmarks:
                logger.info(
                    f"Added {len(benchmarks)} sector benchmarks (for ratio reference, "
                    f"NOT competitors)"
                )

        # Combined list for backward compat (service.py iterates this to fetch data)
        direct = competitors + benchmarks

        if not direct:
            logger.warning(
                f"No suitable peers for {ticker} — all sector peers "
                f"exceed size ceilings"
            )

        # Build peer data dict for downstream use
        peer_data = {}
        for _, row in sector_df.iterrows():
            pt = row.get("peer_ticker")
            if pt and pt != ticker:
                peer_data[pt] = {
                    "name": row.get("name"),
                    "market_cap": row.get("market_cap"),
                    "pe": row.get("pe"),
                    "roce": row.get("roce"),
                    "sales_qtr": row.get("sales_qtr"),
                }

        layers_run = [1, 2]
        competitor_count = len(competitors)
        benchmark_count = len(benchmarks)
        has_competitors = competitor_count > 0

        # Build limitation note when competitors are scarce
        limitation_note = ""
        if not has_competitors:
            limitation_note = (
                f"No direct competitors found among listed Indian companies in "
                f"the same size range. Competitors may be private, international, "
                f"or classified under a different sector. "
                f"The {benchmark_count} companies shown are sector benchmarks for "
                f"financial ratio reference only — they are NOT direct competitors."
            )
        elif competitor_count == 1:
            limitation_note = (
                f"Only 1 direct competitor found. {ticker} may have limited "
                f"listed competition. {benchmark_count} sector benchmark(s) "
                f"included for reference."
            )

        peer_quality_context = (
            f"Peers sourced from Screener.in sector classification. "
            f"{competitor_count} competitor(s) + {benchmark_count} benchmark(s) "
            f"selected from {len(all_sector_tickers)} sector peers after size filtering."
        )

        # Layer 3: LLM Peer Validation (optional)
        run_layer3 = use_llm or self._llm_validation_enabled
        if run_layer3:
            try:
                validated, suggestions, quality_context, assessments = self._layer3_llm_validation(
                    ticker=ticker,
                    target_data=target_data,
                    candidates=direct,
                    peer_data=peer_data,
                )

                # Reclassify based on LLM assessment
                llm_categories = {}
                for a in assessments:
                    t = a.get("ticker", "")
                    rel = a.get("relevance", "")
                    if rel == "true_competitor":
                        llm_categories[t] = "competitor"
                    elif rel == "tangential":
                        llm_categories[t] = "benchmark"
                    # irrelevant → excluded by validated list

                # Process LLM-suggested alternative peers
                if suggestions:
                    suggestion_data = self._process_llm_suggestions(
                        ticker, target_data, suggestions, target_mcap, target_revenue
                    )
                    # Add suggestion data to peer_data
                    peer_data.update(suggestion_data)
                    # LLM specifically found these as competitors
                    for s_ticker in suggestion_data:
                        if s_ticker not in validated:
                            validated.append(s_ticker)
                        llm_categories[s_ticker] = "competitor"

                direct = validated[: self.max_peers]
                # Rebuild peer_categories from LLM classification
                peer_categories = {}
                for t in direct:
                    peer_categories[t] = llm_categories.get(
                        t, "benchmark"  # default to benchmark if LLM didn't classify
                    )

                peer_quality_context = quality_context
                layers_run.append(3)

                # Recount after LLM
                competitor_count = sum(1 for v in peer_categories.values() if v == "competitor")
                benchmark_count = sum(1 for v in peer_categories.values() if v == "benchmark")
                has_competitors = competitor_count > 0

                # Rebuild limitation note after LLM
                if not has_competitors:
                    limitation_note = (
                        f"LLM validation found no true competitors among sector peers. "
                        f"Competitors may be private or international. "
                        f"{benchmark_count} sector benchmarks shown for reference."
                    )
                elif competitor_count == 1:
                    limitation_note = (
                        f"Only 1 direct competitor validated by LLM. "
                        f"{benchmark_count} sector benchmark(s) included for reference."
                    )
                else:
                    limitation_note = ""

                logger.info(
                    f"Layer 3: {competitor_count} competitors + "
                    f"{benchmark_count} benchmarks after LLM validation"
                )
            except Exception as e:
                logger.warning(
                    f"Layer 3 LLM validation failed, using Layer 1-2 peers: {e}"
                )

        result = PeerResult(
            direct_competitors=direct,
            sector_peers=all_sector_tickers,
            value_chain=[],  # Layer 4: future
            discovery_metadata={
                "target": ticker,
                "target_mcap": target_mcap,
                "target_revenue": target_revenue,
                "candidates_evaluated": len(all_sector_tickers),
                "size_filtered_to": len(size_filtered),
                "layers_run": layers_run,
                "peer_quality_context": peer_quality_context,
                "llm_validated": 3 in layers_run,
                "has_competitors": has_competitors,
                "competitor_count": competitor_count,
                "benchmark_count": benchmark_count,
                "limitation_note": limitation_note,
            },
            peer_data=peer_data,
            peer_categories=peer_categories,
        )

        logger.info(
            f"Peer discovery complete: {competitor_count} competitors + "
            f"{benchmark_count} benchmarks, "
            f"{len(all_sector_tickers)} sector peers evaluated"
        )

        return result

    # ── Layer 1: Sector Classification ──

    def _layer1_sector_peers(
        self, ticker: str, target_data: dict | None
    ) -> pd.DataFrame:
        """Get sector peers from Screener.in peer comparison table."""
        warehouse_id = None
        if target_data and "metadata" in target_data:
            warehouse_id = target_data["metadata"].get("warehouse_id")

        return self.peers_fetcher.fetch(ticker, warehouse_id=warehouse_id)

    def _extract_peer_tickers(self, df: pd.DataFrame, exclude: str) -> list[str]:
        """Extract unique peer tickers, excluding the target company."""
        if df.empty or "peer_ticker" not in df.columns:
            return []
        tickers = df["peer_ticker"].dropna().unique().tolist()
        return [t for t in tickers if t != exclude]

    # ── Layer 2: Size + Relevance Filtering ──

    def _layer2_size_filter(
        self,
        sector_df: pd.DataFrame,
        ticker: str,
        target_mcap: float | None,
        target_revenue: float | None,
    ) -> list[str]:
        """Filter peers by market cap, revenue, and business relevance.

        Filters applied:
        1. Market cap band (default 3x of target)
        2. Revenue band (default 5x of target quarterly revenue)
        3. Business relevance: remove companies with negative RoCE or no PE
           (signals misclassification — e.g., real estate in a paper sector)
        """
        if sector_df.empty:
            return []

        candidates = sector_df[sector_df["peer_ticker"] != ticker].copy()
        if candidates.empty:
            return []

        before_count = len(candidates)

        # Business relevance filter: remove clearly misclassified companies
        # Companies with negative RoCE AND no valid PE are likely in a different business
        if "roce" in candidates.columns and "pe" in candidates.columns:
            roce_vals = pd.to_numeric(candidates["roce"], errors="coerce")
            pe_vals = pd.to_numeric(candidates["pe"], errors="coerce")
            # Keep companies that have either: positive RoCE OR valid PE
            relevance_mask = (roce_vals > 0) | (pe_vals.notna() & (pe_vals > 0) & (pe_vals < 200))
            excluded = candidates[~relevance_mask]
            if not excluded.empty:
                excluded_names = excluded["peer_ticker"].tolist()
                logger.info(
                    f"  Filtered {len(excluded_names)} irrelevant peers "
                    f"(negative RoCE + no valid PE): {excluded_names}"
                )
            candidates = candidates[relevance_mask]

        # Market cap filter
        if target_mcap and target_mcap > 0:
            mcap_low = target_mcap / self.size_band_mult
            mcap_high = target_mcap * self.size_band_mult
            candidates = candidates[
                candidates["market_cap"].between(mcap_low, mcap_high, inclusive="both")
            ]

        # Revenue filter (using quarterly sales as proxy)
        if target_revenue and target_revenue > 0:
            rev_low = target_revenue / self.revenue_band_mult
            rev_high = target_revenue * self.revenue_band_mult
            if "sales_qtr" in candidates.columns:
                candidates = candidates[
                    candidates["sales_qtr"].between(rev_low, rev_high, inclusive="both")
                ]

        filtered = candidates["peer_ticker"].dropna().tolist()
        logger.info(f"  Size+relevance filter: {before_count} → {len(filtered)} candidates")
        return filtered

    def _get_target_mcap(
        self, sector_df: pd.DataFrame, ticker: str, target_data: dict | None
    ) -> float | None:
        """Get target market cap from data or sector table."""
        if target_data and "metadata" in target_data:
            mcap = target_data["metadata"].get("Market Cap")
            if mcap:
                return float(mcap)

        # Fallback: find target in sector table
        if not sector_df.empty:
            target_row = sector_df[sector_df["peer_ticker"] == ticker]
            if not target_row.empty:
                return float(target_row["market_cap"].iloc[0])

        return None

    def _get_target_revenue(self, target_data: dict | None) -> float | None:
        """Get target quarterly revenue from data."""
        if target_data is None:
            return None

        fin = target_data.get("financials")
        if fin is not None and not fin.empty and "revenue" in fin.columns:
            # Use latest annual revenue / 4 as quarterly proxy
            annual = fin[~fin["year"].astype(str).str.contains("TTM", case=False, na=False)]
            if not annual.empty:
                latest_rev = pd.to_numeric(annual["revenue"], errors="coerce").iloc[-1]
                if not pd.isna(latest_rev):
                    return float(latest_rev) / 4

        return None

    # ── Layer 3: LLM Peer Validation (optional) ──

    def _get_llm_client(self):
        """Lazy-init anthropic client for Layer 4."""
        if self._llm_client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set — cannot run Layer 3")
            self._llm_client = anthropic.Anthropic(api_key=api_key)
        return self._llm_client

    def _layer3_llm_validation(
        self,
        ticker: str,
        target_data: dict | None,
        candidates: list[str],
        peer_data: dict,
    ) -> tuple[list[str], list[str], str, list[dict]]:
        """Validate peer candidates using LLM for competitive relevance.

        Returns:
            (validated_tickers, suggested_tickers, peer_quality_context, assessments)
        """
        logger.info("[Layer 3] Running LLM peer validation")
        start_time = time.time()

        # Build context from target data
        metadata = (target_data or {}).get("metadata", {})
        company_name = metadata.get("name", ticker)
        sector = metadata.get("sector", "Unknown")
        market_cap = metadata.get("Market Cap", "N/A")
        roce = metadata.get("ROCE %", "N/A")
        opm = metadata.get("OPM", "N/A")
        de = metadata.get("Debt / Equity", "N/A")

        # Build candidate list text
        candidate_lines = []
        for i, ct in enumerate(candidates, 1):
            info = peer_data.get(ct, {})
            candidate_lines.append(
                f"{i}. {info.get('name', ct)} ({ct}) — "
                f"MCap ₹{info.get('market_cap', 'N/A')} Cr, "
                f"PE {info.get('pe', 'N/A')}, "
                f"RoCE {info.get('roce', 'N/A')}%"
            )
        candidate_list = "\n".join(candidate_lines) if candidate_lines else "No candidates"

        # Load and format prompt
        prompt_path = PROMPTS_DIR / "peer_validation.txt"
        with open(prompt_path) as f:
            template = f.read()

        prompt = template.format(
            company_name=company_name,
            ticker=ticker,
            sector=sector,
            market_cap=f"{market_cap:,.0f}" if isinstance(market_cap, (int, float)) else market_cap,
            roce=roce,
            opm=opm,
            de=de,
            candidate_list=candidate_list,
            max_suggestions=self._llm_max_suggestions,
        )

        # Call LLM
        client = self._get_llm_client()
        response = client.messages.create(
            model=self._peer_validation_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        elapsed = time.time() - start_time
        output_text = response.content[0].text

        logger.info(
            f"  Layer 3 LLM: {response.usage.input_tokens}in + "
            f"{response.usage.output_tokens}out tokens, {elapsed:.1f}s"
        )

        # Parse JSON response
        parsed = self._parse_llm_json(output_text)

        # Extract validated peers (true_competitor only)
        validated = []
        assessments = parsed.get("candidate_assessments", [])
        for a in assessments:
            if a.get("relevance") == "true_competitor":
                t = a.get("ticker", "")
                if t in candidates:
                    validated.append(t)

        # If LLM rejected ALL candidates but we need some peers, keep tangential ones
        if not validated:
            for a in assessments:
                if a.get("relevance") == "tangential":
                    t = a.get("ticker", "")
                    if t in candidates:
                        validated.append(t)
            if validated:
                logger.info(
                    f"  No true_competitors found, keeping {len(validated)} tangential peers"
                )

        # Extract suggested alternatives
        suggestions = []
        for s in parsed.get("suggested_alternatives", []):
            st = s.get("ticker", "").strip().upper()
            if st and st != ticker and st not in validated:
                suggestions.append(st)

        # Build peer quality context string for Pass 3
        classification_ok = parsed.get("classification_appropriate", True)
        classification_reason = parsed.get("classification_reasoning", "")
        ideal = parsed.get("ideal_peer_characteristics", "")

        context_parts = []
        if not classification_ok:
            context_parts.append(
                f"WARNING: Screener.in sector classification '{sector}' is NOT appropriate "
                f"for {company_name}. {classification_reason}"
            )
        else:
            context_parts.append(
                f"Screener.in sector classification '{sector}' is appropriate. "
                f"{classification_reason}"
            )

        # Summarize assessments
        relevance_counts = {"true_competitor": 0, "tangential": 0, "irrelevant": 0}
        for a in assessments:
            r = a.get("relevance", "")
            if r in relevance_counts:
                relevance_counts[r] += 1
        context_parts.append(
            f"Of {len(assessments)} candidates: {relevance_counts['true_competitor']} true competitors, "
            f"{relevance_counts['tangential']} tangential, {relevance_counts['irrelevant']} irrelevant."
        )

        if suggestions:
            sugg_names = [s.get("ticker", "") for s in parsed.get("suggested_alternatives", [])]
            context_parts.append(
                f"LLM suggested {len(suggestions)} alternative peers: {', '.join(sugg_names)}."
            )

        if ideal:
            context_parts.append(f"Ideal peer characteristics: {ideal}")

        peer_quality_context = " ".join(context_parts)

        logger.info(
            f"  Layer 3 result: {len(validated)} validated, "
            f"{len(suggestions)} suggestions, "
            f"classification_appropriate={classification_ok}"
        )

        return validated, suggestions, peer_quality_context, assessments

    def _process_llm_suggestions(
        self,
        ticker: str,
        target_data: dict | None,
        suggestions: list[str],
        target_mcap: float | None,
        target_revenue: float | None,
    ) -> dict:
        """Fetch data for LLM-suggested peers and validate them.

        Returns dict of {ticker: {name, market_cap, pe, roce, sales_qtr}} for valid suggestions.
        """
        logger.info(f"  Processing {len(suggestions)} LLM-suggested peers")
        valid_suggestions = {}

        for suggested_ticker in suggestions[: self._llm_max_suggestions]:
            try:
                # Fetch the suggested company's Screener.in page for metadata
                data = self.financials_fetcher.fetch_all(suggested_ticker)
                if not data or "metadata" not in data:
                    logger.info(f"    {suggested_ticker}: no data found, skipping")
                    continue

                meta = data["metadata"]
                s_mcap = meta.get("Market Cap")
                s_roce = meta.get("ROCE %")
                s_pe = meta.get("Stock P/E")

                # Get quarterly revenue
                s_qtr_rev = None
                fin = data.get("financials")
                if fin is not None and not fin.empty and "revenue" in fin.columns:
                    annual = fin[
                        ~fin["year"].astype(str).str.contains("TTM", case=False, na=False)
                    ]
                    if not annual.empty:
                        latest_rev = pd.to_numeric(
                            annual["revenue"], errors="coerce"
                        ).iloc[-1]
                        if not pd.isna(latest_rev):
                            s_qtr_rev = float(latest_rev) / 4

                # Basic size check (relaxed — 5x band for suggestions)
                if target_mcap and s_mcap:
                    s_mcap_f = float(s_mcap)
                    if s_mcap_f < target_mcap / 5 or s_mcap_f > target_mcap * 5:
                        logger.info(
                            f"    {suggested_ticker}: MCap ₹{s_mcap_f:,.0f} outside "
                            f"5x band of target ₹{target_mcap:,.0f}, skipping"
                        )
                        continue

                valid_suggestions[suggested_ticker] = {
                    "name": meta.get("name", suggested_ticker),
                    "market_cap": float(s_mcap) if s_mcap else None,
                    "pe": float(s_pe) if s_pe else None,
                    "roce": float(s_roce) if s_roce else None,
                    "sales_qtr": s_qtr_rev,
                }
                logger.info(
                    f"    {suggested_ticker}: valid (MCap ₹{s_mcap} Cr, RoCE {s_roce}%)"
                )

            except Exception as e:
                logger.warning(f"    {suggested_ticker}: fetch failed ({e}), skipping")

        return valid_suggestions

    def _parse_llm_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try code block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

        # Try generic code block
        if "```" in text:
            start = text.index("```") + 3
            newline = text.index("\n", start)
            start = newline + 1
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from Layer 3 LLM response")
        return {}
