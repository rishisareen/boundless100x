# Residual review findings — feat/sqglp-refinements

Findings from the code review of this branch that were **not** fixed here.
Recorded so they are not lost; none blocks the branch.

Reviewers: correctness, adversarial, testing. The feasibility reviewer did not
complete (the Anthropic account hit its monthly spend limit mid-run), so its
lens is a coverage gap — the other three verified most of the same ground
against the real cached data.

## Correctness / design

- ~~**The 4-lever decomposition bypasses the macro config.**~~ **Resolved.**
  The table now takes a `macro` argument that the service passes from the
  engine, and it resolves the current P/E from metadata rather than a `pe_ratio`
  column Screener never publishes — so its valuation verdict is real instead of
  permanently "cannot be computed". The report reuses the service's table rather
  than recomputing and patching its own, so the reader and LLM Pass 2 can no
  longer be shown different verdicts for one company.

- ~~**Veto flags disappear when their source metric errors.**~~ **Resolved.**
  Gates now declare `veto_sources` (metrics expected to emit the veto flag).
  When no metric carries `reverse_dcf_overpriced` and its source
  `reverse_dcf_growth` is missing or errored, the price gate reads
  indeterminate instead of proceeding on the PEG conditions alone.

- **A single missing year can trigger the small-cap history waiver.**
  `_short_window_flags` counts non-null observations, so one NaN year in a
  ten-year series looks like short history. For a company under the waiver
  threshold that silently drops the metric instead of scoring the gap. Base the
  flag on rows available rather than on the post-`dropna` count.

- ~~**`short_history_smallcap` is emitted but never displayed.**~~ **Resolved.**
  Scores now carry a coverage block (share of declared weight, per element and
  overall) that the report and CLI render, and waived metrics count as absent.

- **The backtest never exercises the production scorer's history waiver.**
  Market cap is withheld under truncation, so `_waived_for_history` always
  returns False there. The backtest therefore validates a slightly different
  scoring configuration than production runs. Worth stating in the limitations
  block.

- ~~**Reverse DCF saturates silently.**~~ **Resolved.** The search still runs
  bounded to [-10%, +50%], but a pinned result now carries the
  `reverse_dcf_saturated` flag and `saturated_at: ceiling|floor` in metadata,
  so scoring and the overpriced veto can see that the value is an artifact of
  the bound.

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

- ~~**`metadata["sector"]` is absent from every cached ticker.**~~ **Resolved.**
  Root cause: the page carries several `<p class="sub">` blocks (the first is
  a "machine generated" disclaimer), so the first-anchor selector never found
  the sector. Extraction now locates the `/market/` breadcrumb by its link
  titles and prefers Broad Industry; study-bucket matching is plural-tolerant
  ("Capital Market" matches Screener's "Capital Markets"). Live-verified on
  CDSL. Tickers still need a re-fetch for their cached metadata to pick it up.

- ~~**The financials fetch bypasses its cache**~~ **Resolved.** The dead
  `_do_fetch`/`cache_key` code is gone; the Screener company page HTML now
  goes through the TTL cache (`txt` entries), so repeat runs within the
  window parse the cached page instead of re-scraping.
