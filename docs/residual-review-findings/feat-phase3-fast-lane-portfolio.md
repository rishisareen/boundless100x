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

- **Unknown sectors read as sector headroom.** Positioned names whose sector
  could not be read are dropped into `unknown_sector` and surfaced only as a
  human-readable note, which no machine consumer reads — so they are invisible
  to the router's cap check. Not rare: only tickers fetched after the
  breadcrumb fix carry `metadata.sector`, and any ticker whose analysis failed
  arrives sectorless. Relatedly, a sector cap of 0 or 1 cannot fire at all,
  because `check_concentration` only reports groups of 2+. Harmless at the
  shipped cap of 3.

## Reliability

- **A snapshot is written before evaluation.** `record_snapshot` commits
  `last_score_snapshot.at` immediately after `analyze()`, and `get_stale(90)`
  reads exactly that. A ticker whose advance raises anywhere downstream is
  recorded in `errors` for the run and then reads as freshly scored for 90
  days, so `watchlist advance --quarterly` will not look at it again until the
  quarter is up — a thesis that broke on the one day the ticker errored goes
  unevaluated. The write ordering predates Phase 3, but Phase 3 is what put
  lane gates, routing safety, friction and the sector lookup between the
  snapshot and the return.

- **Concurrency beyond the lost-update guard.** `_commit` now refuses a
  superseded write (fixed in `e43b207`), which closes silent data loss. It is
  not a lock: the read-then-write race window remains, and two processes can
  still interleave. Closing it properly needs an `O_EXCL` lockfile or `flock` —
  a design call, and arguably unnecessary for a single-owner CLI.

- **One deployment transition can close several exits.** `queue route`
  validates that the candidate holds an eligible position transition after the
  exit, but nothing records that the transition has already been consumed.
  Routing exit A and exit B into the same `probe` both succeed.

## Structure

- **`cli.py` crossed 1,000 lines** (953 → ~1,700). Four new command groups and
  six display helpers now sit interleaved with corpus, screen, sweep and
  backtest commands in one flat file, and the drift has already started —
  `_print_routing_result` is defined above the `# ── Display Helpers ──` banner
  every other `_print_*` helper lives under. The natural extraction is a
  `cli_lifecycle.py` holding the watchlist/queue surface, registered with
  `add_typer` the way the corpus group already is.

- **`report_generator.py` is now the largest module** (1809 → 2095). The new
  Lane & Friction surface is cleanly banner-delimited, so it does not read as a
  regression, but the next report section has no seam to land at.

- **The two stores' shared base is underscore-private across a module
  boundary.** `reinvestment.py` imports `_JsonStore` and `_revision_of` from
  `watchlist.py`. The coupling is deliberate and documented, but it also makes
  `boundless100x.watchlist` and the `boundless100x.lifecycle` package mutually
  dependent — latent only because `lifecycle/__init__.py` is a bare docstring.
  A dependency-free leaf module (the `forward_growth_schema.py` precedent)
  would remove the edge and let the shared names be public.

- **`LANE_VERDICTS` names two incompatible types**: a tuple of the vocabulary
  in `lane_gates.py`, and a label/sentiment dict in `report_generator.py`.
  Renaming the report-layer map to `LANE_VERDICT_LABELS` would leave the name
  meaning one thing repo-wide.

## Schema and contracts

- **`reinvestment_queue.json` carries no `schema_version`**, unlike
  `score_history.jsonl`. It validates loudly against a per-kind required-key
  set with an explicit no-migration rule, so the first future change that adds
  a required key turns every existing queue into a hard error with no way to
  distinguish "written by an older version" from "corrupt".

- **`latest_proposal` is loaded unvalidated** while every event is validated. A
  non-dict value loads silently and then raises `AttributeError` inside
  `snapshot_state`, so `watchlist queue` dies with a traceback instead of the
  store's own actionable message.

- **Event validation checks key presence, not value shape.** An event with
  `"ticker": null` loads cleanly and crashes later in `exit_views`.

- **`watchlist catalyst` writes state nothing on the CLI can read.** The
  catalyst gates fast-lane entry and fires an exit rule, but
  `WatchlistManager.list()` returns no catalyst field, so `watchlist show` has
  no column for it. An owner cannot see which companies have an active
  catalyst, or which window has passed, without a full `advance` or opening the
  JSON by hand.

- **No CLI path changes an entry's lane** after `watchlist add --lane`. `add`
  returns False for an existing ticker; `remove` + re-`add` discards the
  append-only `state_history`.

## Testing

- **Golden normalisation renders an empty chart identically to a full one.**
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

- **`advance()`'s per-run resolutions are never all live in one call.** Every
  routing test injects an evaluator (which short-circuits pace), and every pace
  test passes no queue. A tightened pace threshold withholds a `→ probe`
  proposal, which changes the routing ranking — that interaction is uncovered.

- **No test drives `advance(apply=True)` with a queue**, so the case where a
  candidate is bought into `probe` during the run and therefore leaves
  `CANDIDATE_STATES` before routing ranks it is uncovered.

- **`advance_ticker` is never run against the degraded result the real service
  produces on a fetch failure** — `service.analyze` catches its own exceptions
  and returns empty scores rather than raising, while the test stub raises.

## Second-pass review (2026-08-07)

An independent re-review of `5ae71fc..6960a76` after the phase closed. It
confirmed the findings above against the code and the green suite (1583), and
added three more, none blocking.

- **The checkpoint past-dating check does not follow the run clock.**
  `record_from_pass2` gained an `as_of` parameter (`checkpoints.py`), but its
  only production caller, `record_checkpoints` (`advance.py`), never passes
  one, so the validation falls back to `date.today()` — and the `{today}` the
  pass-2 prompt states is wall-clock too. On a live run these coincide, but a
  backdated `--as-of` replay validates `due_date`s against the wrong "today":
  the one seam where the phase's otherwise-rigorous "same clock the rest of
  the run reads" discipline (threaded correctly through the evaluator,
  friction, and the time stops) does not reach. The fix is one parameter
  through two signatures.

- **The accumulation streak fails closed but silently on a reordered file.**
  `compute_institutional_accumulation_trend` walks backward assuming
  oldest-first `shareholding.csv` order (verified true of the corpus today).
  If a refetch ever wrote newest-first, the adjacency check breaks the walk at
  the first step and the metric returns `0` — a *fail*, not an error — so the
  gate reads "no accumulation" indefinitely rather than indeterminate. The
  safe direction, but inconsistent with the module's own loudness rule; a
  "periods must be ascending, else error" check would close it.

- **No decay exit from `watch` in either lane.** `fast_lane_qualification_
  failed` covers `screen`/`qualify` only, mirroring the core lane's
  pre-existing gap: a candidate whose composite decays below the floor while
  sitting in `watch` is never dropped — only `fundamentals_deteriorated` can
  remove it. Symmetric rather than a regression, but the fast lane's "complete
  path" claim is slightly overstated: a stalled `watch` entry is
  indistinguishable from a considered one, the same criticism the drop rule's
  own rationale levels at `screen`.
