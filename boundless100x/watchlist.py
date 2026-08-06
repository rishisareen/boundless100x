"""The watchlist — persistence for the investment lifecycle.

Each entry is one company's position in the state machine: which lane it is
in, what state it has reached, the checkpoints its thesis is held to, and an
append-only log of every transition with the evidence that caused it.

Two properties are deliberate.

**A state is earned, never granted.** `add` creates an entry at `screen` and
nothing else can set a state directly — the only way forward is
`transition()`, which records the trigger and evidence that justified it. A
company therefore cannot end up in `scale` without a readable trail of why.

**History is append-only.** `state_history` is never rewritten, so a decision
that later looks wrong can still be traced to the evidence available when it
was taken. That is the whole point of recording the evidence rather than just
the outcome.

There is one schema and no migration path: old outputs were discarded at the
start of Phase 1. An entry that does not match is a loud error, because with
one schema in existence an odd entry means something is wrong, and repairing
it silently is how a company ends up in a state nobody assigned it.
"""

from __future__ import annotations  # `list` is a method name here; keep annotations lazy

import json
import logging
from datetime import datetime
from pathlib import Path

from boundless100x.lifecycle import states as lifecycle_states

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"

CORE_LANE = "core"
LANES = (CORE_LANE,)  # The re-rating lane arrives in Phase 3.

APPLIED_AUTO = "auto"
APPLIED_OWNER = "owner"

REQUIRED_KEYS = (
    "added",
    "notes",
    "lane",
    "state",
    "checkpoints",
    "kill_switch_status",
    "last_score_snapshot",
    "state_history",
)


class WatchlistError(ValueError):
    """A stored entry does not match the schema."""


def _now() -> str:
    return datetime.now().isoformat()


def _new_entry(notes: str, lane: str) -> dict:
    return {
        "added": _now(),
        "notes": notes,
        "lane": lane,
        "state": lifecycle_states.INITIAL,
        "checkpoints": [],
        "kill_switch_status": {},
        "last_score_snapshot": None,
        "state_history": [],
    }


class WatchlistManager:
    """Reads and writes lifecycle state for tracked companies."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else DEFAULT_WATCHLIST_PATH
        self.data = self._load()

    # ── persistence ──

    def _load(self) -> dict:
        if not self.path.exists():
            return {"companies": {}}
        with open(self.path) as f:
            data = json.load(f)
        companies = data.get("companies", {})
        for ticker, entry in companies.items():
            self._validate_entry(ticker, entry)
        return {"companies": companies}

    @staticmethod
    def _validate_entry(ticker: str, entry: object) -> None:
        if not isinstance(entry, dict):
            raise WatchlistError(f"{ticker}: entry must be an object")
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        if missing:
            raise WatchlistError(
                f"{ticker}: entry is missing {', '.join(missing)}. The watchlist has a "
                f"single schema and no migration path — fix or remove the entry rather "
                f"than letting it be repaired silently."
            )
        if not lifecycle_states.is_state(entry["state"]):
            raise WatchlistError(f"{ticker}: unknown state {entry['state']!r}")
        if entry["lane"] not in LANES:
            raise WatchlistError(f"{ticker}: unknown lane {entry['lane']!r}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    # ── membership ──

    def add(self, ticker: str, notes: str = "", lane: str = CORE_LANE) -> bool:
        """Track a company. Starts at `screen` — qualification is earned."""
        ticker = ticker.upper()
        if ticker in self.data["companies"]:
            return False
        if lane not in LANES:
            raise WatchlistError(f"unknown lane {lane!r}")
        self.data["companies"][ticker] = _new_entry(notes, lane)
        self._save()
        return True

    def remove(self, ticker: str) -> bool:
        ticker = ticker.upper()
        if ticker not in self.data["companies"]:
            return False
        del self.data["companies"][ticker]
        self._save()
        return True

    def get(self, ticker: str) -> dict | None:
        return self.data["companies"].get(ticker.upper())

    def tickers(self) -> list[str]:
        return list(self.data["companies"].keys())

    def list(self) -> list[dict]:
        """Flat rows for display, newest state first in the history."""
        rows = []
        for ticker, entry in self.data["companies"].items():
            snapshot = entry.get("last_score_snapshot") or {}
            rows.append({
                "ticker": ticker,
                "lane": entry["lane"],
                "state": entry["state"],
                "added": entry["added"],
                "last_run": snapshot.get("at"),
                "last_composite": snapshot.get("composite"),
                "verdict": snapshot.get("verdict"),
                "checkpoints": len(entry.get("checkpoints") or []),
                "notes": entry.get("notes", ""),
            })
        return rows

    # ── lifecycle ──

    def transition(
        self,
        ticker: str,
        to_state: str,
        trigger_id: str,
        evidence: str = "",
        applied_by: str = APPLIED_AUTO,
    ) -> dict:
        """Move a company to a new state, recording why.

        The evidence travels with the transition because a state without its
        reason cannot be reviewed later — and reviewing later is the point.
        """
        entry = self.get(ticker)
        if entry is None:
            raise WatchlistError(f"{ticker} is not on the watchlist")
        if not lifecycle_states.is_state(to_state):
            raise WatchlistError(f"unknown state {to_state!r}")

        record = {
            "at": _now(),
            "from": entry["state"],
            "to": to_state,
            "trigger_id": trigger_id,
            "evidence": evidence,
            "applied_by": applied_by,
        }
        entry["state_history"].append(record)
        entry["state"] = to_state
        self._save()
        logger.info(
            f"{ticker.upper()}: {record['from']} → {to_state} "
            f"({trigger_id}, {applied_by})"
        )
        return record

    def record_snapshot(self, ticker: str, result, config_hash: str | None = None) -> None:
        """Store the latest scoring outcome against the entry.

        The registry hash rides along so the regime behind a stored composite
        is visible without cross-referencing score_history.jsonl.
        """
        entry = self.get(ticker)
        if entry is None:
            raise WatchlistError(f"{ticker} is not on the watchlist")

        scores = result.scores or {}
        eligibility = result.eligibility or {}
        entry["last_score_snapshot"] = {
            "at": _now(),
            "composite": scores.get("composite"),
            "elements": scores.get("elements", {}),
            "verdict": eligibility.get("verdict", "indeterminate"),
            "config_hash": config_hash,
        }
        self._save()

    def set_checkpoints(self, ticker: str, checkpoints: list[dict]) -> None:
        """Replace the recorded checkpoints for a company."""
        entry = self.get(ticker)
        if entry is None:
            raise WatchlistError(f"{ticker} is not on the watchlist")
        entry["checkpoints"] = list(checkpoints or [])
        self._save()

    def set_kill_switch_status(self, ticker: str, status: dict) -> None:
        entry = self.get(ticker)
        if entry is None:
            raise WatchlistError(f"{ticker} is not on the watchlist")
        entry["kill_switch_status"] = dict(status or {})
        self._save()

    # ── scheduling ──

    def get_stale(self, days: int = 90) -> list[str]:
        """Tickers not scored within `days`. Never-scored entries are stale."""
        stale = []
        for ticker, entry in self.data["companies"].items():
            snapshot = entry.get("last_score_snapshot") or {}
            last = snapshot.get("at")
            if not last:
                stale.append(ticker)
                continue
            try:
                age = (datetime.now() - datetime.fromisoformat(last)).days
            except (TypeError, ValueError):
                stale.append(ticker)
                continue
            if age >= days:
                stale.append(ticker)
        return stale
