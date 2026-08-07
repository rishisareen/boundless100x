"""Phase 4 strategy simulator — a replay of the production lifecycle over
truncated historical data (`docs/plans/2026-08-07-007-feat-phase4-strategy-simulator-plan.md`).

The central design constraint (per the plan's Requirements, R2/R10) is that
the replay calls the production evaluators — `ComputeEngine.run_all`,
`SQGLPScorer`, `EligibilityEvaluator`, `TriggerEvaluator`, `LaneGateEvaluator`
— on point-in-time-truncated data, rather than a second statement of the
same rules. A simulator that reimplemented the lifecycle would prove
something about *that* reimplementation, not about the shipped one.

Submodules, in the order the replay loop consumes them:

  * `calendar.py`  — replay dates from the corpus's own fiscal calendar
                      (KTD7), plus the per-lane battery-complete reading.
  * `universe.py`   — `raw_data/` discovery and per-ticker candidacy under
                      KTD8 (first replay date a ticker's truncated financials
                      clear the engine's minimum-years bar).
  * `replay.py`     — the loop skeleton: temp-dir production stores,
                      `add`-and-lane-assign every eligible ticker at
                      `screen`. The full six-step loop (truncate -> score ->
                      evaluate -> propose -> confirm -> settle -> mark to
                      market) is U7; this module stops at "a watchlist
                      populated with lane-assigned candidates at their
                      screen dates."

Not yet built (later units, per the plan's Implementation Units): `owner.py`
(U3, the simulated-owner policy), `ledger.py` (U4, modeled capital),
`friction_cash.py` (U5, cash-level friction), `outputs.py` (U6, the six §10
readings). `simulate(config_overrides) -> dict` — the one importable
entry point R10 requires for a Phase 5 sweep to loop over without
subprocesses — lands in this file once U7 exists to drive it end to end.
"""
