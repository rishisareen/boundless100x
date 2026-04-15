"""Watchlist Manager — track companies for periodic re-analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"


class WatchlistManager:
    """Persistent watchlist with last-run tracking."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else DEFAULT_WATCHLIST_PATH
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"companies": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add(self, ticker: str, notes: str = "") -> bool:
        """Add a company to the watchlist."""
        if ticker in self.data["companies"]:
            logger.info(f"{ticker} already in watchlist")
            return False

        self.data["companies"][ticker] = {
            "added": datetime.now().isoformat(),
            "last_run": None,
            "last_composite": None,
            "notes": notes,
        }
        self._save()
        logger.info(f"Added {ticker} to watchlist")
        return True

    def remove(self, ticker: str) -> bool:
        """Remove a company from the watchlist."""
        if ticker not in self.data["companies"]:
            return False
        del self.data["companies"][ticker]
        self._save()
        logger.info(f"Removed {ticker} from watchlist")
        return True

    def list(self) -> list[dict]:
        """Return all watchlist entries."""
        entries = []
        for ticker, info in self.data["companies"].items():
            entries.append({
                "ticker": ticker,
                "added": info.get("added", ""),
                "last_run": info.get("last_run"),
                "last_composite": info.get("last_composite"),
                "notes": info.get("notes", ""),
            })
        return entries

    def mark_run(self, ticker: str, composite_score: float | None = None):
        """Mark a company as having been analyzed."""
        if ticker in self.data["companies"]:
            self.data["companies"][ticker]["last_run"] = datetime.now().isoformat()
            if composite_score is not None:
                self.data["companies"][ticker]["last_composite"] = composite_score
            self._save()

    def get_stale(self, days: int = 90) -> list[str]:
        """Return tickers that haven't been analyzed in N days."""
        stale = []
        now = datetime.now()
        for ticker, info in self.data["companies"].items():
            last_run = info.get("last_run")
            if last_run is None:
                stale.append(ticker)
            else:
                last_dt = datetime.fromisoformat(last_run)
                if (now - last_dt).days >= days:
                    stale.append(ticker)
        return stale

    def update_all(self, service, quarterly: bool = False):
        """Re-run analysis on all watchlist companies.

        Args:
            service: Boundless100xService instance.
            quarterly: If True, only update stale (90+ days) entries.

        Returns:
            List of (ticker, composite_score) tuples.
        """
        from boundless100x.output.report_generator import ReportGenerator

        tickers = self.get_stale(days=90) if quarterly else list(self.data["companies"].keys())

        if not tickers:
            logger.info("No companies need updating")
            return []

        logger.info(f"Updating {len(tickers)} watchlist companies")
        generator = ReportGenerator()
        results = []

        for ticker in tickers:
            try:
                result = service.analyze(ticker, use_llm=False)
                composite = result.scores.get("composite")
                self.mark_run(ticker, composite)
                generator.generate(result, formats=["json", "md"])
                results.append((ticker, composite))
                logger.info(f"  {ticker}: composite {composite}/10")
            except Exception as e:
                logger.error(f"  {ticker}: failed — {e}")
                results.append((ticker, None))

        return results
