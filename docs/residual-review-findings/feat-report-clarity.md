# Residual review findings — Report clarity (the research note)

Findings from the code review of the report-clarity work that were **not**
fixed in it. Recorded so they are not lost; neither blocks the phase, and the
suite is green at 2537.

Reviewers: correctness, security, adversarial, testing, maintainability,
reliability, api-contract, project-standards. The cross-model adversarial pass
could not run — no peer CLI is installed on this machine — so the adversarial
lens ran in-process, which is the documented fallback and a mild independence
gap. An independent validator upheld all eight findings it was given; none was
dropped as a false positive.

Thirteen findings from the same review **were** fixed (`edd4f0d`), and the
label-drift finding below was fixed after it. What follows is what survived
triage.

## The design call the phase left open

- **An indeterminate eligibility gate reads as agreement, not as
  indeterminate.** `contradiction.py:546` computes
  `unread = [side for side in sides if side["state"] is None]`, and a gate at
  `GATE_INDETERMINATE` carries a state, so it is treated as read. A pair whose
  gate side could not be evaluated therefore resolves to `agrees` rather than
  `indeterminate` — the one outcome this codebase's spine says an unknown must
  never take.

  It is left open rather than fixed because it is genuinely a decision, and
  the decision is not obvious. `test_an_indeterminate_gate_is_not_a_disagreement`
  deliberately pins today's behaviour, and there is a real argument for it: a
  gate that could not be evaluated has not *disagreed* with anything, and the
  contradiction trigger exists to surface a live conflict rather than to
  announce every gate the run could not compute. The counter-argument is that
  `agrees` is a positive finding — it says two readings were compared and
  found consistent — and nothing was compared here.

  The choice is between a third outcome the caller must handle and a narrower
  fix: treat a gate side at `GATE_INDETERMINATE` as unread **unless** the
  declaration lists `indeterminate` in its `verdict_in`, which keeps the
  current behaviour available to a pair that wants it. Either way the pinning
  test has to change, which is what makes this an owner's call rather than a
  patch.

  Reach for this before adding a second declared pair whose gate side is one
  of the gates that reads indeterminate often — today only one pair is
  declared, against the `price` gate, which limits the blast radius.

## Known bounds, recorded rather than fixed

- **The suppression corpus cannot refuse to compare across scoring regimes.**
  A `scores.json` carries no `config_hash`, so unlike `trajectory.py` the
  corpus behind R8's zero-rate test reads a registry change at face value. The
  exposure is bounded by what is counted — whether a score was *exactly* zero
  is coarse, and a threshold edit moves a score off zero far less often than
  it moves it at all — but it is a real limitation. Writing the hash into
  `scores.json` is the fix when it matters.

- **`load_scored_corpus` treats every subdirectory of the reports directory as
  a report.** A stray folder is counted in the reader-facing "could not be
  read" prose. Harmless today; it would mislead if that directory ever gains
  non-report contents.

- **The two YAML loaders collapse "failed to parse" and "declares nothing"
  into the same empty return.** `load_sector_applicability` and
  `load_contradiction_pairs` both log and return `{}`. The code says this is
  the safe direction and it is — an unreadable table must not read as a table
  that excludes nothing — but only a log line tells anyone the signal is off
  for the whole run. No report surface says so.

- **`_reserved_keys()` lets four single-word grades through as a whole field.**
  `moderate`, `risky`, `unknown` and `discounting` are English words as well as
  grade values, and three shipped band labels read "moderate", so the
  whole-field rule had to narrow or the note could not render a real company.
  Safe because grades route through `CATEGORICAL_VALUE_LABELS`, but
  `Quantity._fallback_text` still returns a raw grade verbatim if it is ever
  reached.

- **The reading-layer subsystem sits inside `ReportGenerator`.** Now eight
  `_reading_*` methods rather than twelve `_clarity_*` ones (the note's own
  subtitle, unscored section and appendix builders went with the note when the
  layer was folded into the dashboard), but the call stands: they sit in a
  class that is ~1,700 lines, where `report_components`, `report_expansion` and
  `report_reading` were extracted in the same work. Extracting them to
  `output/report_clarity.py`, invoked the way `report_charts` is, is the
  obvious next tidy. No defect; a single reviewer's structural call.

- **`possible_bonus_split_year_N` is registered for N in 1..12.** `growth.py`
  indexes by annual row position and the cache already holds 13 annual rows,
  so one more Screener column emits an unregistered `_year_13` — which renders
  as an unknown rather than leaking, but loses its wording.

## Testing gaps left open

- Nothing forces `metric_row`'s or `disclosure_for`'s
  `except ComponentContentError` fallback to execute for `disclosure_for`.
  (`metric_row`'s is now covered.)
- No test pins that `METRIC_DISPLAY_NAMES` agrees with the registry's `name`
  field. The drill-down now prefers the registry, so the table is a fallback
  only — but a future editor could still add a disagreeing entry.
- No end-to-end test drives a `custom/` drop-in metric through the CLI now
  that `presentation` is required of every metric.
