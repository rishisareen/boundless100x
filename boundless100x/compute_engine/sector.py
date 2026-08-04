"""Sector tailwind classification from the Dec 2025 Wealth Creation Study.

`data_fetcher/sector_context.yaml` lists the sectors that produced compounders
in the NTD era and the ones the study rules out. This module is the single
reader of that file, used by both the scored metric and the LLM prompt context.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PATH = (
    Path(__file__).parent.parent / "data_fetcher" / "sector_context.yaml"
)

STRONG = "strong_tailwind"
MODERATE = "moderate_tailwind"
NON_CONSIDERATION = "non_consideration"
UNKNOWN = "unknown"


@lru_cache(maxsize=4)
def load_sector_context(path: str | None = None) -> dict:
    """Load the sector lists. Returns empty buckets if the file is unreadable."""
    target = Path(path) if path else DEFAULT_CONTEXT_PATH
    empty = {STRONG: [], MODERATE: [], NON_CONSIDERATION: [], "raw": {}}

    try:
        raw = yaml.safe_load(target.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read sector context at {target}: {exc}")
        return empty

    buckets = raw.get("mtd_consideration_sectors", {}) or {}
    return {
        STRONG: list(buckets.get("strong_tailwind", []) or []),
        MODERATE: list(buckets.get("moderate_tailwind", []) or []),
        NON_CONSIDERATION: list(buckets.get("non_consideration", []) or []),
        "raw": raw,
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _matches(sector: str, listed: str) -> bool:
    """Whole-phrase match, so a short code like 'IT' cannot hit 'Securities'."""
    sector_n, listed_n = _normalise(sector), _normalise(listed)
    if sector_n == listed_n:
        return True
    return re.search(rf"(?<!\w){re.escape(listed_n)}(?!\w)", sector_n) is not None


def classify_sector(sector: str | None, context: dict | None = None) -> str:
    """Map a sector name onto its study bucket, or `unknown` when unlisted."""
    if not sector or not str(sector).strip():
        return UNKNOWN

    ctx = context or load_sector_context()
    for bucket in (STRONG, MODERATE, NON_CONSIDERATION):
        if any(_matches(str(sector), listed) for listed in ctx.get(bucket, [])):
            return bucket
    return UNKNOWN


def study_findings(context: dict | None = None) -> dict:
    """The business-type and leadership findings that sit alongside the lists."""
    raw = (context or load_sector_context()).get("raw", {})
    return {
        "business_type": raw.get("business_type_preference", {}) or {},
        "leadership": raw.get("market_leadership", {}) or {},
    }
