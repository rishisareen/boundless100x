"""The watchlist — persistence for the investment lifecycle.

Each entry is one company's position in the state machine: which lane it is
in, what state it has reached, the checkpoints its thesis is held to, and an
append-only log of every transition with the evidence that caused it.

Three properties are deliberate.

**A state is earned, never granted.** `add` creates an entry at `screen` and
nothing else can set a state directly — the only way forward is
`transition()`, which records the trigger and evidence that justified it. A
company therefore cannot end up in `scale` without a readable trail of why.

**History is append-only.** `state_history` is never rewritten, so a decision
that later looks wrong can still be traced to the evidence available when it
was taken. That is the whole point of recording the evidence rather than just
the outcome.

**A change is durable before it is visible.** Every mutator stages onto a deep
copy, writes the copy through `atomic_write_json`, and adopts it only once the
write has returned. Two failures are ruled out by that order: a crash mid-write
leaves the previous store rather than truncated JSON, and a failed save leaves
`self.data` describing exactly what is on disk rather than a change that was
never durable. The second is the more dangerous of the two, because nothing
later in the process can tell that memory ran ahead.

There is one schema and no migration path: old outputs were discarded at the
start of Phase 1. An entry that does not match is a loud error, because with
one schema in existence an odd entry means something is wrong, and repairing
it silently is how a company ends up in a state nobody assigned it. `catalyst`
is the exception that proves the shape of the rule: it is optional rather than
required, so a store written before the fast lane existed keeps loading
untouched, and every reader asks for it with `.get("catalyst")`.
"""

from __future__ import annotations  # `list` is a method name here; keep annotations lazy

import copy
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from boundless100x.lifecycle import states as lifecycle_states

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"

CORE_LANE = "core"
RERATING_LANE = "rerating"
LANES = (CORE_LANE, RERATING_LANE)

APPLIED_AUTO = "auto"
APPLIED_OWNER = "owner"

CATALYST_ACTIVE = "active"
CATALYST_SPENT = "spent"
CATALYST_STATUSES = (CATALYST_ACTIVE, CATALYST_SPENT)

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


class StoreConflictError(RuntimeError):
    """The store on disk moved on since this instance loaded it.

    Not a schema fault, which is why it is neither `WatchlistError` nor
    `ReinvestmentError`: the document is fine, the *writer* is stale. Raised
    by `_JsonStore._commit`, and the way out is always the same — reload the
    store and redo the change against what is actually there.
    """


def _now() -> str:
    return datetime.now().isoformat()


def _fsync_directory(directory: Path) -> None:
    """Make a rename in this directory durable, where the platform allows it.

    Its own function so the `except OSError` cannot accidentally swallow a
    failure from the write itself: everything in here is the *extra*
    guarantee, and the file it publishes is already on disk by the time this
    runs. Opening a directory for reading is not portable — Windows refuses,
    and some filesystems do too — so a refusal is logged and the write stands.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError as e:
        logger.debug(f"Could not open {directory} to fsync the rename: {e}")
        return
    try:
        os.fsync(fd)
    except OSError as e:
        logger.debug(f"Could not fsync {directory} after the rename: {e}")
    finally:
        os.close(fd)


def atomic_write_json(path: Path | str, data: object) -> None:
    """Write JSON such that the previous file survives a failed write.

    The write lands on a temp file in the *same directory* — `os.replace` is
    only atomic within one filesystem, so a temp directory elsewhere would
    reintroduce the copy this exists to avoid. The store therefore only ever
    holds a fully written document: either the old one or the new one, never
    the first few hundred bytes of the new one.

    **Both the data and the publication are synced.** Syncing the file proves
    its bytes reached the disk; the `os.replace` that makes them *the store* is
    a directory metadata change, and an unsynced one can still be in the OS
    cache when the power goes. That matters beyond one file: `exit.py`'s
    crash-safety argument is that the queue event is durable before the
    transition is attempted, and two unsynced renames can be made durable in
    the opposite order — producing an `exited` entry with no queue event, the
    one direction the exit protocol calls unrecoverable.

    The directory sync is best-effort by necessity: some platforms and
    filesystems refuse to open a directory for reading at all. A failure there
    costs the extra guarantee and is logged; it must never cost the write,
    which has already completed.

    Shared rather than private because the durability argument is per-file and
    identical wherever the argument applies.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # A replaced file inherits the temp file's private 0600 mode, so a
        # store that was readable before this call must stay readable after it
        # — durability may not quietly change who can open the file.
        if path.exists():
            os.chmod(temp_name, path.stat().st_mode & 0o777)
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        # Leave nothing behind to be mistaken for a store: the caller keeps
        # the previous file, and a stray half-written sibling would only
        # confuse whoever comes to investigate the failure.
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _revision_of(data: dict) -> int:
    """The store's commit counter, defaulting to zero for a store without one.

    Absent on every file written before Phase 3, and hand-editable into
    nonsense like anything else on disk — either way the counter restarts from
    zero rather than raising, because a missing revision is a staleness signal
    nobody can read yet, not a corrupt watchlist.
    """
    revision = data.get("revision", 0)
    if not isinstance(revision, int) or revision < 0:
        return 0
    return revision


class _JsonStore:
    """The commit mechanics every tracked JSON store in this system shares.

    Two stores exist — this module's watchlist and `lifecycle/reinvestment.py`'s
    event queue — and they stay two stores on purpose: different files,
    different schemas, different validation, different questions. **Only the
    commit mechanics are shared**, exactly as `atomic_write_json` already was,
    and for the same reason: the argument is per-file and identical wherever it
    applies, so two copies would be two things to keep in step.

    The revision counter is why "would be" is not hypothetical. `snapshot_state`
    decides whether a routing proposal may be rendered by comparing **both**
    stores' counters against the ones a snapshot captured, so the clamping rule
    — absent or negative restarts at zero — has to mean the same thing in both
    files. Two copies of it, and a snapshot could read current against one store
    and stale against the other, which is a proposal rendered or withheld on the
    strength of which file somebody last edited by hand.

    `_load` stays with the subclass. What a store *contains* and what makes an
    entry valid is the part that is genuinely not shared, and a base class that
    reached into either would be the merge this deliberately is not.
    """

    def __init__(self, path: str | Path | None, default_path: Path):
        self.path = Path(path) if path else default_path
        self.data = self._load()

    def _load(self) -> dict:
        raise NotImplementedError

    def _stage(self) -> dict:
        """A deep copy of the store, safe to mutate before anything is committed."""
        return copy.deepcopy(self.data)

    def _on_disk_revision(self) -> int | None:
        """The counter the file currently holds, or None if there is no file.

        Read fresh rather than remembered, because the whole question this
        answers is what somebody *else* did in the meantime.
        """
        if not self.path.exists():
            return None
        try:
            with open(self.path) as f:
                return _revision_of(json.load(f))
        except (OSError, ValueError) as e:
            # Unreadable is not proof of safety. The constructor validated this
            # file, so something changed it since, and overwriting whatever is
            # there now on the strength of a copy loaded before that happened is
            # exactly the destruction this check exists to prevent.
            raise StoreConflictError(
                f"{self.path} could not be re-read before committing ({e}), so "
                f"this write cannot be shown not to discard someone else's — "
                f"reload the store and repeat the change"
            ) from e

    def _commit(self, staged: dict) -> None:
        """Persist a staged store, then adopt it — never the other way round.

        The revision bumps here and nowhere else, so it counts durable commits
        rather than attempts. A reader comparing revisions to decide whether
        its view is current would otherwise be told a change happened that the
        store never took.

        **A superseded writer is refused rather than allowed to win.** This
        method rewrites the whole document from the copy loaded at
        construction, so two processes that both loaded revision 5 both write
        revision 6 and the loser's change vanishes — with a counter that reads
        perfectly consistent afterwards, which makes the loss invisible to the
        one mechanism built to detect superseded state. The documented workflow
        reaches it: `watchlist advance` holds both stores open across a
        minutes-long fetch loop, and `watchlist exit` is a separate command.
        So the on-disk counter is re-read immediately before the write and must
        still be the one this instance loaded.

        **This closes the lost-update window, not the race.** Two processes can
        still pass the check and reach `os.replace` in either order; what is
        gone is the far wider window between loading a store and committing to
        it. Deliberately not a lock: a lock file is state that outlives the
        process holding it, and a stale one would block the exit command at
        precisely the moment it is needed. A refusal the caller can retry from
        a fresh load is the smaller promise, honestly kept.
        """
        loaded = _revision_of(self.data)
        current = self._on_disk_revision()
        if current is not None and current != loaded:
            raise StoreConflictError(
                f"{self.path} is at revision {current}; this instance loaded "
                f"revision {loaded} and its write would discard everything "
                f"committed since. Nothing was written — reload the store and "
                f"repeat the change against what is on disk."
            )

        staged["revision"] = loaded + 1
        atomic_write_json(self.path, staged)
        self.data = staged


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


class WatchlistManager(_JsonStore):
    """Reads and writes lifecycle state for tracked companies."""

    def __init__(self, path: str | None = None):
        super().__init__(path, DEFAULT_WATCHLIST_PATH)

    # ── persistence ──

    def _load(self) -> dict:
        if not self.path.exists():
            return {"companies": {}, "revision": 0}
        with open(self.path) as f:
            data = json.load(f)
        companies = data.get("companies", {})
        for ticker, entry in companies.items():
            self._validate_entry(ticker, entry)
        return {"companies": companies, "revision": _revision_of(data)}

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

        # Optional, so its absence says "written before the fast lane existed"
        # rather than "broken". Present, it is held to the same loud standard
        # as the rest: a status nothing recognises would read as neither active
        # nor spent wherever the fast lane asks.
        catalyst = entry.get("catalyst")
        if catalyst is not None:
            if not isinstance(catalyst, dict):
                raise WatchlistError(f"{ticker}: catalyst must be an object")
            if catalyst.get("status") not in CATALYST_STATUSES:
                raise WatchlistError(
                    f"{ticker}: unknown catalyst status {catalyst.get('status')!r}"
                )

    def _stage_entry(self, ticker: str) -> tuple[dict, dict]:
        """A staged store and the entry inside it — the pair every setter mutates."""
        key = ticker.upper()
        if key not in self.data["companies"]:
            raise WatchlistError(f"{ticker} is not on the watchlist")
        staged = self._stage()
        return staged, staged["companies"][key]

    # ── membership ──

    def add(self, ticker: str, notes: str = "", lane: str = CORE_LANE) -> bool:
        """Track a company. Starts at `screen` — qualification is earned.

        The lane is the owner's choice of *how* the company will be judged:
        `core` for the compounder path, `rerating` for the fast lane. Neither
        grants a state; both start at `screen`.
        """
        ticker = ticker.upper()
        if ticker in self.data["companies"]:
            return False
        if lane not in LANES:
            raise WatchlistError(f"unknown lane {lane!r} — one of {', '.join(LANES)}")
        staged = self._stage()
        staged["companies"][ticker] = _new_entry(notes, lane)
        self._commit(staged)
        return True

    def remove(self, ticker: str) -> bool:
        ticker = ticker.upper()
        if ticker not in self.data["companies"]:
            return False
        staged = self._stage()
        del staged["companies"][ticker]
        self._commit(staged)
        return True

    def set_lane(self, ticker: str, lane: str, reason: str = "") -> dict:
        """Move a tracked company to the other lane, keeping everything else.

        The lane was settable once, at `add`, and nothing could change it
        afterwards: `add` returns False for a ticker already tracked, so the
        only route was `remove` then re-`add` — which discards the
        append-only `state_history` and with it every piece of evidence behind
        the states the company has already earned. A company whose thesis
        changed from "compounder" to "re-rating" is a normal thing to happen
        and should not cost its record.

        **The state is untouched.** A lane says how a company is judged, not
        how far along it is, and re-deriving one from the other would silently
        promote or demote a company nobody had re-evaluated. The next
        `advance` evaluates it under the new lane's rules, which is where a
        change of lane is supposed to show up.

        **Recorded in `lane_history`, not in `state_history`.** This is not a
        transition — no state moved — and writing it into the state log would
        put a record there with no `to` anybody could read. It is still
        append-only, because which lane's kill-switches applied when is
        exactly the kind of thing a later review needs.

        A no-op change is refused rather than recorded: a log of lane changes
        in which nothing changed is noise in the one place that must stay
        readable.
        """
        staged, entry = self._stage_entry(ticker)
        if lane not in LANES:
            raise WatchlistError(f"unknown lane {lane!r} — one of {', '.join(LANES)}")

        previous = entry["lane"]
        if previous == lane:
            raise WatchlistError(
                f"{ticker.upper()} is already in the {lane} lane — nothing to change"
            )

        record = {
            "at": _now(),
            "from": previous,
            "to": lane,
            "state": entry["state"],
            "reason": reason,
        }
        entry.setdefault("lane_history", []).append(record)
        entry["lane"] = lane
        self._commit(staged)
        logger.info(f"{ticker.upper()}: lane {previous} → {lane}")
        return record

    def get(self, ticker: str) -> dict | None:
        """The stored entry, for reading.

        Not a write path: the next commit replaces `self.data` wholesale, so an
        entry held across one is a detached copy and anything written into it
        goes nowhere. Every change belongs in a mutator, which is also the only
        way it reaches disk.
        """
        return self.data["companies"].get(ticker.upper())

    def tickers(self) -> list[str]:
        return list(self.data["companies"].keys())

    def list(self) -> list[dict]:
        """Flat rows for display, newest state first in the history.

        The catalyst travels because it is owner-recorded state that gates
        fast-lane entry and fires an exit rule, and nothing on any surface
        could read it back: `watchlist catalyst` wrote it, and seeing it again
        meant a full `advance` or opening the JSON by hand. State the system
        acts on and the owner cannot see is state the owner cannot correct.

        `catalyst_due` is derived rather than stored, because "has the window
        passed?" is a question about today and a stored answer would be wrong
        by tomorrow. Three-valued, in the house style: `None` when there is no
        catalyst or its date cannot be read, so an unreadable window never
        renders as one comfortably in the future.
        """
        rows = []
        for ticker, entry in self.data["companies"].items():
            snapshot = entry.get("last_score_snapshot") or {}
            catalyst = entry.get("catalyst") or {}
            expected = lifecycle_states.as_date(catalyst.get("expected_by"))
            rows.append({
                "ticker": ticker,
                "lane": entry["lane"],
                "state": entry["state"],
                "added": entry["added"],
                "last_run": snapshot.get("at"),
                "last_composite": snapshot.get("composite"),
                "verdict": snapshot.get("verdict"),
                "checkpoints": len(entry.get("checkpoints") or []),
                "catalyst": catalyst.get("description") or "",
                "catalyst_status": catalyst.get("status") or "",
                "catalyst_expected_by": catalyst.get("expected_by") or "",
                "catalyst_overdue": (
                    None if expected is None else expected < datetime.now().date()
                ),
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
        details: dict | None = None,
    ) -> dict:
        """Move a company to a new state, recording why.

        The evidence travels with the transition because a state without its
        reason cannot be reviewed later — and reviewing later is the point.

        `details` is the structured half of that same argument. Prose is what a
        person reads, but a report reading a transition back needs the figures
        apart — a confirmed exit's friction payload has a gross return, a
        holding period, a tax regime, a net return and a basis, and no amount of
        parsing recovers those five from a sentence that mentions them. So a
        caller with a payload attaches it whole, and the prose keeps saying the
        same thing in the same line as before.

        **Optional, and absent when not supplied.** A transition that carried no
        payload records no `details` key at all, so every existing caller writes
        exactly the record it wrote before and a stored history is unchanged.
        Readers ask with `.get("details")` and get None either way.

        The payload is deep-copied on the way in: the staged store is adopted as
        `self.data`, and a caller's dict left wired into it would let a later
        mutation of that dict silently edit an append-only record.
        """
        staged, entry = self._stage_entry(ticker)
        if not lifecycle_states.is_state(to_state):
            raise WatchlistError(f"unknown state {to_state!r}")
        if details is not None and not isinstance(details, dict):
            raise WatchlistError(
                f"transition details must be an object, not {type(details).__name__} "
                f"— the point of the field is that a reader can take it apart"
            )

        record = {
            "at": _now(),
            "from": entry["state"],
            "to": to_state,
            "trigger_id": trigger_id,
            "evidence": evidence,
            "applied_by": applied_by,
        }
        if details is not None:
            record["details"] = copy.deepcopy(details)
        entry["state_history"].append(record)
        entry["state"] = to_state
        self._commit(staged)
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
        staged, entry = self._stage_entry(ticker)

        scores = result.scores or {}
        eligibility = result.eligibility or {}
        entry["last_score_snapshot"] = {
            "at": _now(),
            "composite": scores.get("composite"),
            "elements": scores.get("elements", {}),
            "verdict": eligibility.get("verdict", "indeterminate"),
            "config_hash": config_hash,
        }
        self._commit(staged)

    def set_checkpoints(self, ticker: str, checkpoints: list[dict]) -> None:
        """Replace the recorded checkpoints for a company."""
        staged, entry = self._stage_entry(ticker)
        entry["checkpoints"] = list(checkpoints or [])
        self._commit(staged)

    def set_kill_switch_status(self, ticker: str, status: dict) -> None:
        staged, entry = self._stage_entry(ticker)
        entry["kill_switch_status"] = dict(status or {})
        self._commit(staged)

    # ── catalysts ──

    def record_catalyst(self, ticker: str, description: str, expected_by: str) -> dict:
        """Name what this company is waiting on, and by when.

        The fast lane rests on a re-rating happening for a reason, and the
        reason is the one input the pipeline cannot derive: no metric knows
        that a demerger is filed or a plant is due. So it is recorded as owner
        judgement, and both halves are required — an event with no window can
        never come due, and a window with no event cannot be checked when it
        does. Recording again replaces outright rather than appending: a
        catalyst is the current reason to be waiting, not a history of them.
        """
        staged, entry = self._stage_entry(ticker)

        description = (description or "").strip()
        expected_by = (expected_by or "").strip()
        missing = [
            name for name, value in
            (("description", description), ("expected_by", expected_by))
            if not value
        ]
        if missing:
            raise WatchlistError(
                f"{ticker.upper()}: a catalyst needs both a description and an "
                f"expected_by window — missing {', '.join(missing)}"
            )

        catalyst = {
            "description": description,
            "expected_by": expected_by,
            "status": CATALYST_ACTIVE,
            "recorded_at": _now(),
        }
        entry["catalyst"] = catalyst
        self._commit(staged)
        return catalyst

    def mark_catalyst_spent(self, ticker: str) -> dict:
        """Record that the awaited event has happened.

        Spent, not deleted: what the fast lane bought into is still readable
        afterwards, and a position whose catalyst has been spent without the
        re-rating following is exactly the case worth being able to see.
        """
        staged, entry = self._stage_entry(ticker)
        catalyst = entry.get("catalyst")
        if not catalyst:
            raise WatchlistError(f"{ticker.upper()} has no catalyst to spend")

        catalyst["status"] = CATALYST_SPENT
        self._commit(staged)
        return catalyst

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
