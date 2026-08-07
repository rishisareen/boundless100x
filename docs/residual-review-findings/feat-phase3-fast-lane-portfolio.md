# Residual review findings — Phase 3 fast lane + portfolio layer

Findings from the code review of Phase 3 that were **not** fixed in it.
Recorded so they are not lost; none blocks the phase, and the suite is green
at 1583.

Reviewers: correctness, adversarial, reliability, testing, project-standards,
maintainability, api-contract. The cross-model adversarial pass could not run —
no peer CLI is installed on this machine — so the adversarial lens ran
in-process, which is the documented fallback and a mild independence gap.

Nine other findings from the same review **were** fixed (commit `e43b207`), and
nine quality findings before it (`f19f264`, `10ff1c3`). What follows is what
survived triage.

## Design decisions the phase left open

- **~~Removing a watchlist entry orphans its unrouted exit proceeds.~~**
  **Fixed** (Tranche 1a). Completeness is now stamped onto the queue as a
  third event kind — `confirmed`, appended by `confirm_exit` as a fourth
  ordered write — rather than read from live lifecycle state, so a recorded
  exit survives the removal of the company it describes. `exit_is_complete`
  takes the exit event and its stamp; the live-state fallback is kept only for
  the window between the transition and the stamp, and matches on *this*
  exit's own review, which closes the other direction too. `watchlist remove`
  refuses while a ticker holds an unconfirmed exit (fail-closed on an
  unreadable queue), and `unroutable_reason` now decides what every surface
  says, so `NO_PROCEEDS` can no longer be printed over a half-written record.
  An entry already in `exited` whose event is unstamped is reconciled rather
  than refused. The original finding follows.

  Flagged independently by correctness, adversarial, and testing.
  `exit_is_complete` answers a per-*exit* question by reading per-*ticker*
  watchlist state, so `watchlist remove` on a company whose exit was fully
  recorded but not yet routed makes its proceeds permanently unroutable — and
  the queue then reports "No exit proceeds awaiting routing", the exact false
  all-clear the module docstring says it exists to prevent. The recovery
  instruction the display offers (`watchlist exit <ticker>`) can never succeed,
  because the ticker is no longer on the watchlist. Re-adding the ticker, or a
  ticker that later re-reaches `exit_review`, produces the same class of
  confusion in the other direction.

  The fix needs a decision rather than a patch: **should completeness be
  stamped onto the queue event** at `confirm_exit` (a `confirmed` marker, or
  the transition timestamp written back) so it stops depending on live
  lifecycle state? That is the shape all three reviewers converged on. The
  cheaper interim is to refuse `watchlist remove` while a ticker holds an
  unrouted exit, and to stop `propose_routing` emitting `NO_PROCEEDS` when
  `incomplete` is non-empty.

- **~~Concentration caps gate the advisory router but never `advance --apply`.~~**
  **Fixed** (Tranche 1b), as its own commit, since it is a real behaviour
  change to the money-moving path. The "does one more name fit?" question moved
  into `portfolio.would_breach`, so the router and the transition path ask it
  through one function rather than two. `advance()` supplies a gate recomputed
  **per candidate** rather than once before the loop — an applying run changes
  the occupancy it is checking, so two probes into a lane with room for one
  would both pass a single up-front reading. It fires only when a transition
  would *add* a name, so `probe → scale` is untouched. A breach withholds
  `should_apply` even under `--apply`; `--override-caps` proceeds and records
  the breach in the append-only evidence, because the override needing no flag
  is editing the config, and that one leaves no trace. The original finding
  follows.

  `check_concentration` had exactly two consumers: `propose_routing` (inert)
  and the CLI display. Nothing in the transition path read it, and the reading
  was computed *after* the ticker loop — so the first time an owner learned a
  lane or sector was over its cap, the transitions that broke it were already
  durable. A cap could therefore only ever be reported as already breached,
  never prevented.

- **~~Unknown sectors read as sector headroom.~~** **Fixed** (Tranche 3).
  `would_breach` now asks "can one more be shown to fit?" and counts every
  unread sector as though it might be the one at issue — a worst case rather
  than a refusal, because blocking on any unknown at all would let one
  sectorless holding freeze the whole book. A candidate with no sector of its
  own is measured against the fullest group. `check_concentration` also carries
  every sector group including singles (`_MIN_GROUP` now gates only the prose)
  and a top-level `sector_max`, which is what lets a cap of 1 fire at all and a
  cap of 0 reach a sector holding nothing. The original follows.

  Positioned names whose sector could not be read are dropped into
  `unknown_sector` and surfaced only as a human-readable note, which no machine
  consumer reads — so they are invisible to the router's cap check. Not rare:
  only tickers fetched after the breadcrumb fix carry `metadata.sector`, and
  any ticker whose analysis failed arrives sectorless. Relatedly, a sector cap
  of 0 or 1 cannot fire at all, because `check_concentration` only reports
  groups of 2+. Harmless at the shipped cap of 3.

  **The stakes rose in Tranche 1b**, which is why this was worth doing rather
  than deferring: `portfolio.would_breach` had just started gating the
  money-moving path as well as the advisory one, so a sectorless name became a
  silent pass on a *transition* rather than on a recommendation. The lane axis
  failed closed on every gap it met and the sector axis failed open on this
  one. All 22 cached tickers carry a sector today, so the fix changes nothing
  about the current corpus — a ticker whose analysis errors mid-run is the case
  it actually catches.

## Reliability

- **~~A snapshot is written before evaluation.~~** **Fixed** (Tranche 2):
  `record_snapshot` is now the last write in `advance_ticker`, after the
  transition, so "scored" means the run got through rather than that it
  started. An errored ticker stays stale and the next `--quarterly` run picks
  it up. The original follows.

  A snapshot is written before evaluation. `record_snapshot` commits
  `last_score_snapshot.at` immediately after `analyze()`, and `get_stale(90)`
  reads exactly that. A ticker whose advance raises anywhere downstream is
  recorded in `errors` for the run and then reads as freshly scored for 90
  days, so `watchlist advance --quarterly` will not look at it again until the
  quarter is up — a thesis that broke on the one day the ticker errored goes
  unevaluated. The write ordering predates Phase 3, but Phase 3 is what put
  lane gates, routing safety, friction and the sector lookup between the
  snapshot and the return.

- **Concurrency beyond the lost-update guard.** Still open, but the question
  has been sharpened and it is **two questions, not one**. The original
  finding follows the analysis.

  **What the guard actually leaves.** `_commit` re-reads the on-disk counter
  immediately before writing, so the window that remains is one file read and
  a rename wide. Two processes can both read revision 5, both pass, and both
  write revision 6; the second rename wins and the counter reads perfectly
  consistent afterwards, so nothing detects it. Losing a write to that needs
  two writers aligned within roughly a millisecond. The window the *documented
  workflow* reaches — `watchlist advance` holding both stores open across a
  minutes-long fetch loop while `watchlist exit` runs in another terminal — is
  already a loud refusal.

  **`flock` and an `O_EXCL` lockfile are not interchangeable, and the recorded
  rationale conflated them.** `json_store.py` argued against locking on the
  grounds that a lock file outlives the process holding it and a stale one
  would block the exit command exactly when it is needed. That is true of an
  `O_EXCL` lockfile and false of `flock`, which the kernel releases on process
  death, SIGKILL included. The objection only ever applied to one of the two
  options the finding listed. Corrected in the docstring; if this is ever
  closed, `flock` is the tool.

  **The exposure that is actually plausible is not local, and a lock does not
  address it.** Both stores live inside an iCloud-synced directory
  (`~/Library/Mobile Documents/com~apple~CloudDocs/…/boundless100x/`), so the
  second writer to worry about is the sync daemon or another Mac, not another
  process on this one. `flock` and `os.replace` are local-filesystem
  guarantees and say nothing about a file replaced underneath them by a sync.
  (Reasoned from the path and general sync behaviour, not from anything
  verified about iCloud's internals — worth confirming before acting on it.)
  The revision check turns out to be the right instrument for that case
  anyway: a store synced in from elsewhere carries a counter this instance did
  not load, so the next commit refuses rather than clobbering.

  **So the decision is:**
  1. *Local simultaneity* — close it or not? Current read: **not**, and for a
     better reason than "single-owner CLI", which is an assumption about the
     future rather than a fact about the code. The window is sub-millisecond
     and needs exact alignment.
  2. *Sync as a second writer* — the one with real exposure. Locking is not
     the lever. The levers are moving the stores outside the synced tree (they
     are git-tracked, so iCloud adds little) or accepting it and relying on
     the revision refusal. **This is the one that would change if the
     "GUI-ready" service layer in `CLAUDE.md` ever grows a GUI, or if the
     repo is ever cloned onto a second machine.**

  The original finding: `_commit` now refuses a superseded write (fixed in
  `e43b207`), which closes silent data loss. It is not a lock: the
  read-then-write race window remains, and two processes can still interleave.
  Closing it properly needs an `O_EXCL` lockfile or `flock` — a design call,
  and arguably unnecessary for a single-owner CLI.

- **~~One deployment transition can close several exits.~~** **Fixed**
  (Tranche 2): `deployments_consumed_by` records which transitions have
  already closed an exit, and `queue route` refuses one that has, naming the
  exit holding it. Refused rather than made impossible — a system that counts
  names and not rupees cannot rule out one purchase absorbing two sales — so
  `--allow-shared-deployment` covers the real case. The original follows.

  One deployment transition can close several exits. `queue route`
  validates that the candidate holds an eligible position transition after the
  exit, but nothing records that the transition has already been consumed.
  Routing exit A and exit B into the same `probe` both succeed.

## Structure

- **~~`cli.py` crossed 1,000 lines~~** — **fixed** (Tranche 4). It had
  reached 2,030 by the end of Tranche 3. The lifecycle surface moved to
  `cli_lifecycle.py` (785 / 1,287 / 30 across `cli.py`,
  `cli_lifecycle.py` and a new `cli_common.py` holding the shared console
  and `setup_logging` — two modules cannot each own a `Console` without
  owning two wrapping widths and two capture buffers). Registered with
  `add_typer` the way `corpus` already was, with the helpers re-exported
  from `cli.py` so no caller moved. The `_print_routing_result` drift the
  finding named is fixed by a banner every `_print_*` now sits under. The
  original follows.

  It crossed 1,000 lines (953 → ~1,700). Four new command groups and
  six display helpers now sit interleaved with corpus, screen, sweep and
  backtest commands in one flat file, and the drift has already started —
  `_print_routing_result` is defined above the `# ── Display Helpers ──` banner
  every other `_print_*` helper lives under. The natural extraction is a
  `cli_lifecycle.py` holding the watchlist/queue surface, registered with
  `add_typer` the way the corpus group already is.

- **~~`report_generator.py` is now the largest module~~** — **fixed**
  (owner-requested, after Tranche 4). It had reached 2,110. Two things
  inside it were not report *sections*: `report_vocabulary.py` takes the
  ~400 lines of display vocabulary that grow every time a metric or flag
  is added, and `report_charts.py` the ~400 lines of Plotly trace
  assembly — every builder was already `self`-free, so the extraction was
  mechanical rather than a judgement call. 1,337 lines stay: the class
  that decides what a report contains. Names the suite imports are
  re-exported, and three tests pin the boundaries, since a new label is
  easiest to type beside the section that renders it. The original
  follows.

  It was the largest module (1809 → 2095). The new
  Lane & Friction surface is cleanly banner-delimited, so it does not read as a
  regression, but the next report section has no seam to land at.

- **~~The two stores' shared base is underscore-private across a module
  boundary.~~** **Fixed** (Tranche 4): the commit mechanics moved to
  `boundless100x/json_store.py`, a leaf importing nothing from this project,
  with public names. The lifecycle *vocabulary* the other five modules reached
  back for — lanes, `applied_by`, catalyst statuses — moved to
  `lifecycle/states.py`, where its meaning always was; `watchlist.py`
  re-exports every name it published. No lifecycle module imports the
  watchlist now, and a test walks the package's ASTs to keep it that way. The
  original follows.

  `reinvestment.py` imports `reinvestment.py` imports `_JsonStore` and `_revision_of` from
  `watchlist.py`. The coupling is deliberate and documented, but it also makes
  `boundless100x.watchlist` and the `boundless100x.lifecycle` package mutually
  dependent — latent only because `lifecycle/__init__.py` is a bare docstring.
  A dependency-free leaf module (the `forward_growth_schema.py` precedent)
  would remove the edge and let the shared names be public.

- **~~`LANE_VERDICTS` names two incompatible types~~** — **fixed** (Tranche 2):
  the report-layer map is now `LANE_VERDICT_LABELS`, leaving the name meaning
  one thing repo-wide. The original follows.

  `LANE_VERDICTS` names two incompatible types: a tuple of the vocabulary
  in `lane_gates.py`, and a label/sentiment dict in `report_generator.py`.
  Renaming the report-layer map to `LANE_VERDICT_LABELS` would leave the name
  meaning one thing repo-wide.

## Schema and contracts

- **~~`reinvestment_queue.json` carries no `schema_version`~~** — **fixed**
  (Tranche 2), while no queue file existed on disk anywhere, which is the
  cheapest this could ever be. A file without the key reads as version 1; a
  *newer* version is refused rather than read under today's rules; a commit
  preserves the version the file was loaded at, since the store never rewrites
  an existing event. The original follows.

  `reinvestment_queue.json` carries no `schema_version`, unlike
  `score_history.jsonl`. It validates loudly against a per-kind required-key
  set with an explicit no-migration rule, so the first future change that adds
  a required key turns every existing queue into a hard error with no way to
  distinguish "written by an older version" from "corrupt".

- **~~`latest_proposal` is loaded unvalidated~~** — **fixed** (Tranche 2): a
  non-object is now the store's own error naming the file and how to clear it,
  rather than an `AttributeError` from inside a display command. The original
  follows.

  `latest_proposal` is loaded unvalidated while every event is validated. A
  non-dict value loads silently and then raises `AttributeError` inside
  `snapshot_state`, so `watchlist queue` dies with a traceback instead of the
  store's own actionable message.

- **~~Event validation checks key presence, not value shape.~~** **Fixed**
  (Tranche 2): every identifying field must be a usable string and an exit's
  friction payload must be an object, checked at load where the error can
  still name the event and the key. The original follows.

  Event validation checks key presence, not value shape. An event with
  `"ticker": null` loads cleanly and crashes later in `exit_views`.

- **~~`watchlist catalyst` writes state nothing on the CLI can read.~~**
  **Fixed** (Tranche 3): `list()` carries the description, status, window and a
  derived three-valued `catalyst_overdue`, and `watchlist show` has a Catalyst
  column that distinguishes active, overdue, spent and unreadable. The original
  follows.

  The catalyst gates fast-lane entry and fires an exit rule, but
  `WatchlistManager.list()` returns no catalyst field, so `watchlist show` has
  no column for it. An owner cannot see which companies have an active
  catalyst, or which window has passed, without a full `advance` or opening the
  JSON by hand.

- **~~No CLI path changes an entry's lane~~** — **fixed** (Tranche 3):
  `watchlist lane <ticker> <lane> [--reason]` moves it, leaves the state alone,
  and appends to a new `lane_history` rather than to `state_history`, since no
  state moved. A positioned company is moved with a warning naming what changed
  (lane-scoped exits) and what did not (the universal kill-switches). The
  original follows.

  After `watchlist add --lane`, `add`
  returns False for an existing ticker; `remove` + re-`add` discards the
  append-only `state_history`.

## Testing

- **~~Golden normalisation renders an empty chart identically to a full one.~~**
  **Partly fixed, and the finding's conclusion was wrong** (Tranche 4). The
  observation was right: `_CHART` did conflate the two. The conclusion — that
  every chart silently failing would leave the golden green — is false, and
  was checked against the real generator before the normaliser was touched.
  The template guards each container with `{% if chart %}`, so an empty chart
  removes the container *and its card* rather than rendering an empty one: the
  count falls from three to one, which the comparison catches under either
  normaliser. The conflation is closed anyway, since it is one removed `{% if
  %}` away from mattering, and both facts now have tests. The original
  follows.

  `_CHART` replaces
  `_CHART` replaces everything from the container div to end of line, so a
  Plotly payload and `<div class="chart-container"></div>` both normalise to
  the same placeholder. Every chart builder returns `""` on failure, so a data
  contract change that made all seven charts silently stop rendering would
  leave the golden green.

- **~~The exit-event-with-removed-ticker branch is untested~~**, which is how
  the orphaned-proceeds finding above survived. **Fixed** (Tranche 1a):
  `TestCompletenessOutlivesTheEntry` covers both directions, the step 3→4
  crash window has its own class in `test_confirm_exit.py`, and the removal
  refusal is tested at the CLI where it lives.

- **~~`advance()`'s per-run resolutions are never all live in one call.~~**
  **Fixed** (Tranche 4): `TestTheRunsResolutionsAreAllLiveAtOnce` drives pace
  and routing together with no injected evaluator, and shows a tightened
  spread flipping which candidate the router ranks first — against the
  alphabetical order that decided it at a wide spread, which is what says the
  pace caused it. The original follows.

  Every Every
  routing test injects an evaluator (which short-circuits pace), and every pace
  test passes no queue. A tightened pace threshold withholds a `→ probe`
  proposal, which changes the routing ranking — that interaction is uncovered.

- **~~No test drives `advance(apply=True)` with a queue~~** — **fixed**
  (Tranche 4): the run buys the candidate the router would have proposed, and
  the assertions are that it leaves `CANDIDATE_STATES`, is not reported as
  blocked, and the snapshot still persists. The original follows.

  Nothing drove it, so the case where a
  candidate is bought into `probe` during the run and therefore leaves
  `CANDIDATE_STATES` before routing ranks it is uncovered.

- **~~`advance_ticker` is never run against the degraded result the real
  service produces on a fetch failure~~** — **fixed** (Tranche 4): three tests
  drive the empty-scores result `analyze` actually returns, asserting it does
  not raise, that routing safety fails closed on it, and that it flows through
  the loop as an ordinary outcome rather than landing in `errors`. The
  original follows.

  It is never run against that result — `service.analyze` catches its own exceptions
  and returns empty scores rather than raising, while the test stub raises.

## Second-pass review (2026-08-07)

An independent re-review of `5ae71fc..6960a76` after the phase closed. It
confirmed the findings above against the code and the green suite (1583), and
added three more, none blocking.

- **~~The checkpoint past-dating check does not follow the run clock.~~**
  **Fixed** (Tranche 2): `as_of` threads through `record_checkpoints` into
  `record_from_pass2`, so a backdated replay validates `due_date`s against the
  date the rest of the run reads. The original follows.

  The checkpoint past-dating check does not follow the run clock.
  `record_from_pass2` gained an `as_of` parameter (`checkpoints.py`), but its
  only production caller, `record_checkpoints` (`advance.py`), never passes
  one, so the validation falls back to `date.today()` — and the `{today}` the
  pass-2 prompt states is wall-clock too. On a live run these coincide, but a
  backdated `--as-of` replay validates `due_date`s against the wrong "today":
  the one seam where the phase's otherwise-rigorous "same clock the rest of
  the run reads" discipline (threaded correctly through the evaluator,
  friction, and the time stops) does not reach. The fix is one parameter
  through two signatures.

- **~~The accumulation streak fails closed but silently on a reordered file.~~**
  **Fixed** (Tranche 2): the periods must be strictly ascending or the metric
  errors, so an unwalkable frame reads indeterminate and names the pair that
  broke the order instead of returning a zero indistinguishable from a real
  one. The original follows.

  The accumulation streak fails closed but silently on a reordered file.
  `compute_institutional_accumulation_trend` walks backward assuming
  oldest-first `shareholding.csv` order (verified true of the corpus today).
  If a refetch ever wrote newest-first, the adjacency check breaks the walk at
  the first step and the metric returns `0` — a *fail*, not an error — so the
  gate reads "no accumulation" indefinitely rather than indeterminate. The
  safe direction, but inconsistent with the module's own loudness rule; a
  "periods must be ascending, else error" check would close it.

- **~~No decay exit from `watch` in either lane.~~** **Fixed** (Tranche 3):
  both `qualification_failed` and `fast_lane_qualification_failed` now cover
  `watch`. The drop outranks a buy-zone firing in the same run on the core
  side; on the fast lane the two cannot co-fire, because its drop floor (5.0)
  sits below its `quality_floor` gate (5.5) — a gap that is now itself pinned
  by a test. The original follows.

  `fast_lane_qualification_ `fast_lane_qualification_
  failed` covers `screen`/`qualify` only, mirroring the core lane's
  pre-existing gap: a candidate whose composite decays below the floor while
  sitting in `watch` is never dropped — only `fundamentals_deteriorated` can
  remove it. Symmetric rather than a regression, but the fast lane's "complete
  path" claim is slightly overstated: a stalled `watch` entry is
  indistinguishable from a considered one, the same criticism the drop rule's
  own rationale levels at `screen`.
