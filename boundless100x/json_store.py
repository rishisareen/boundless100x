"""The commit mechanics every tracked JSON store in this system shares.

Two stores exist — `watchlist.py`'s lifecycle record and
`lifecycle/reinvestment.py`'s exit/routing event log — and they stay two stores
on purpose: different files, different schemas, different validation, different
questions. **Only the commit mechanics are shared.**

They live here, in a leaf module that imports nothing from this project,
following `forward_growth_schema.py`'s precedent and for the same reason. The
mechanics were previously defined in `watchlist.py` and reached from the
lifecycle package as `_JsonStore` and `_revision_of` — underscore-private names
crossing a module boundary, which is a note to the reader that says "do not
depend on this" beside code that does. Worse, it made `boundless100x.watchlist`
and the `boundless100x.lifecycle` package mutually dependent, latent only
because `lifecycle/__init__.py` happens to be a bare docstring. A leaf both
sides import from removes the edge and lets the shared names be public, which
is what they always were in practice.

**The revision counter is why "shared" is not merely tidy.**
`reinvestment.snapshot_state` decides whether a routing proposal may be
rendered by comparing *both* stores' counters against the ones a snapshot
captured, so the clamping rule — absent or negative restarts at zero — has to
mean the same thing in both files. Two copies of it, and a snapshot could read
current against one store and stale against the other: a proposal rendered or
withheld on the strength of which file somebody last edited by hand.

`_load` deliberately stays with the subclass. What a store *contains* and what
makes an entry valid is the part that is genuinely not shared, and a base class
that reached into either would be the merge this deliberately is not.
"""

import copy
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class StoreConflictError(RuntimeError):
    """The store on disk moved on since this instance loaded it.

    Not a schema fault, which is why it is neither `WatchlistError` nor
    `ReinvestmentError`: the document is fine, the *writer* is stale. Raised
    by `JsonStore._commit`, and the way out is always the same — reload the
    store and redo the change against what is actually there.
    """


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


def revision_of(data: dict) -> int:
    """The store's commit counter, defaulting to zero for a store without one.

    Absent on every file written before Phase 3, and hand-editable into
    nonsense like anything else on disk — either way the counter restarts from
    zero rather than raising, because a missing revision is a staleness signal
    nobody can read yet, not a corrupt store.
    """
    revision = data.get("revision", 0)
    if not isinstance(revision, int) or revision < 0:
        return 0
    return revision


class JsonStore:
    """Copy-on-write staging, an atomic write, and a lost-update refusal.

    Subclasses supply `_load` and everything above it. See the module docstring
    for why only this much is shared.
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
                return revision_of(json.load(f))
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
        still read the same counter here, both pass, and reach `os.replace` in
        either order — the loser's change vanishes under a counter that reads
        perfectly consistent afterwards. What is gone is the far wider window
        between *loading* a store and committing to it, which is the one the
        documented workflow actually reaches. The remainder is a file read and
        a rename apart, so losing a write to it needs two processes aligned
        within about a millisecond.

        **Deliberately not a lock, and the honest reason is the size of that
        window rather than the cost of locking.** An earlier version of this
        argued that "a lock file is state that outlives the process holding it,
        and a stale one would block the exit command at precisely the moment it
        is needed" — true of an `O_EXCL` lockfile, and **not** true of `flock`,
        which the kernel releases when the process dies, SIGKILL included. The
        two are not interchangeable and the objection only ever applied to one
        of them. If this is closed later, `flock` is the tool.

        **A lock would not address the exposure that is actually plausible
        here.** Both stores sit inside an iCloud-synced directory, so the
        second writer to worry about is the sync daemon or another machine, not
        another process on this one — and `flock` and `os.replace` are
        local-filesystem guarantees that say nothing about a file replaced
        underneath them by a sync. The revision check happens to be the right
        instrument for that case too: a store synced in from elsewhere carries
        a counter this instance did not load, so the next commit refuses
        instead of clobbering. `docs/residual-review-findings/` carries the
        decision this is still waiting on.
        """
        loaded = revision_of(self.data)
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
