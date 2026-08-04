# Residual review findings — feat/sqglp-refinements

Findings from the code review of this branch that were **not** fixed here.
Recorded so they are not lost; none blocks the branch.

Reviewers: correctness, adversarial, testing. The feasibility reviewer did not
complete (the Anthropic account hit its monthly spend limit mid-run), so its
lens is a coverage gap — the other three verified most of the same ground
against the real cached data.

## Correctness / design

- **The 4-lever decomposition bypasses the macro config.**
  `compute_lever_decomposition_table` calls `compute_price_lever(data, {"years": years})`
  directly, so a changed `macro.inflation_pct` reaches the registry-scored
  `price_lever_signal` metric but not the narrative table in the report. The two
  can disagree about the same company. Fix: thread `macro` through the
  decomposition call in `service.py` and `report_generator`.

- **Veto flags disappear when their source metric errors.**
  `EligibilityEvaluator` scans all metrics for `reverse_dcf_overpriced`. If the
  reverse-DCF metric itself errored, no metric carries the flag and the price
  gate proceeds on the PEG conditions alone. A veto whose source is unavailable
  should read indeterminate, matching how missing conditions are already handled.

- **A single missing year can trigger the small-cap history waiver.**
  `_short_window_flags` counts non-null observations, so one NaN year in a
  ten-year series looks like short history. For a company under the waiver
  threshold that silently drops the metric instead of scoring the gap. Base the
  flag on rows available rather than on the post-`dropna` count.

- **`short_history_smallcap` is emitted but never displayed.**
  The scorer surfaces it in `scores["flags"]`, and nothing reads it. A waived
  composite is presented on the same 0–10 scale as a fully evidenced one with no
  visible marker. Consider a coverage ratio (scored weight / total weight) beside
  the composite whenever the waiver fires.

- **The backtest never exercises the production scorer's history waiver.**
  Market cap is withheld under truncation, so `_waived_for_history` always
  returns False there. The backtest therefore validates a slightly different
  scoring configuration than production runs. Worth stating in the limitations
  block.

- **Reverse DCF saturates silently.** The binary search is bounded to
  [-10%, +50%]; a company pinned at the ceiling returns exactly 50.0 with no
  flag, and that value both feeds scoring and gates the overpriced veto.
  Pre-existing, not introduced here.

## Testing

- **No test covers `service.py` wiring.** Every test constructs
  `ComputeEngine` / `SQGLPScorer` / `EligibilityEvaluator` with explicit kwargs.
  Renaming the `macro:` block in `config.yaml` would silently revert every metric
  to its hardcoded fallback with the suite green.

- **Two ROIIC error-branch tests assert disjunctions that cannot fail**
  (`assert not result.ok or result.value is None`). They should assert the
  specific flag — `capital_base_flat` vs `capital_base_shrinking` — so the two
  branches are distinguishable.

- **`compute_capital_reinvestment_rate`'s `high_capital_redeployment` branch is
  never asserted**, though an existing fixture already crosses its threshold.

- **The CLI `backtest` command and `_print_eligibility` have no tests** — sort
  key, colour thresholds, and the JSON write are unexercised.

- **`test_emitted_quadrant_values_are_the_four_expected` cannot catch a fifth
  value.** It asserts the four known strings appear in the source, not that no
  other quadrant is emitted — the residual of the bug it guards.

## Data quality (pre-existing, surfaced during review)

- **`metadata["sector"]` is absent from every cached ticker.** The Screener
  selector that populates it is not matching, so `sector_tailwind` scores
  `unknown` for all 17 companies until a re-fetch. The wiring is correct; the
  input is missing.

- **The financials fetch bypasses its cache** (dead code in
  `FinancialsFetcher.fetch_all`), so every run re-scrapes Screener.in. Deferred
  by the plan's scope boundaries.
