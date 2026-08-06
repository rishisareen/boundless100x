"""The closed contract the forward-growth extraction pass is held to.

A deliberately dependency-free leaf module, because two layers that must not
import each other both read it:

  * `llm_layer/forward_growth.py` renders the closed field list into the
    extraction prompt and validates every response against it (KTD3).
  * `compute_engine/metrics/builtin/forward_growth.py` reads the settling map
    to turn a stored promise into a number.
  * `compute_engine/engine.py` folds the fingerprint below into the
    forward-signal hash, so a schema or prompt change is visible in score
    history rather than silently reusing the old regime's label.

`compute_engine` importing `llm_layer` would invert the dependency direction
the whole extraction seam rests on — the backtest re-runs every registered
metric per ticker, and a metric that could reach the LLM layer would issue a
paid call per ticker per backtest against *today's* report text and
*truncated* financials (KTD2). So the shared vocabulary lives here, in
neither.

Bump `SCHEMA_VERSION` whenever a field, kind, settling rule, **or validation
rule** changes. It invalidates every cached extraction sidecar and moves the
forward-signal hash; a prompt change that silently reused stale extractions
would be indistinguishable from the new prompt working.

The validation half is easy to forget and was: the sidecar version covers the
source text, the field schema, the prompt and the model, but not the
*validator*, and a validator that rejected a whole class of genuine entries
kept serving its own empty results from cache after being fixed. Anything that
changes which entries survive belongs in this number.
"""

# 2 — grounding compares on whitespace-normalised text (PDF line wrapping is
#     typesetting, not part of a claim), so entries a stricter byte-exact
#     comparison discarded must be re-extracted rather than read from cache.
# 3 — figures are taken only when the filing already states them in the target
#     unit. The prompt previously asked for conversion while the validator
#     required the value to appear in the quoted sentence — mutually exclusive,
#     so every foreign-currency statement was extracted and then discarded.
#     A filing that reports its market only in USD now yields nothing, which is
#     the truth for a pipeline holding no FX rate.
SCHEMA_VERSION = 3

# ── Provenance ──
# Three-valued (KTD9). `found` means the section was located and looks like
# what it claims to be; `suspect` means it was located but its content says
# otherwise; `fallback` means the slot holds first-N-pages text instead.
# `suspect` is kept distinct from `fallback` rather than folded into it,
# because how often a `found` tag turned out to be wrong is precisely the
# number that says whether the content gate is doing anything.
FOUND = "found"
SUSPECT = "suspect"
FALLBACK = "fallback"

# ── Entry kinds ──
# `data["forward_growth"][year]` is `{kind: [entry, ...]}` for these kinds.
GUIDANCE = "guidance"
CAPEX = "capex"
TAM = "tam"

ENTRY_KINDS = (GUIDANCE, CAPEX, TAM)

# Every entry carries these two whatever its kind.
#   source_sentence — the verbatim sentence the claim came from. KTD3 verifies
#                     it is a literal substring of the text actually submitted
#                     for that year, so a well-typed fabrication is caught
#                     before storage rather than becoming a forward signal.
#   section         — which section it came from. Provenance is tagged per
#                     section while a report year usually carries a mix, so an
#                     entry keyed only by year would let a promises-kept
#                     reading be built from a chairman's letter while MD&A was
#                     never read (KTD4).
COMMON_FIELDS = ("source_sentence", "section")

FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    GUIDANCE: {
        "required": ("metric", "target_value", "target_period") + COMMON_FIELDS,
        # A range is guided as often as a point target in Indian annual
        # reports. The lower bound is the promise (U5), so the high end is
        # optional context rather than part of the test.
        "optional": ("target_value_high",),
    },
    CAPEX: {
        "required": ("amount_inr_cr", "commissioning_year") + COMMON_FIELDS,
        "optional": ("description",),
    },
    TAM: {
        "required": ("market_size_inr_cr",) + COMMON_FIELDS,
        "optional": ("market", "period"),
    },
}

# Closed set of guided quantities. Each names the fetched column that settles
# it, so a promise whose quantity is outside this set is not a promise this
# system can check and is discarded rather than counted.
#
# `unit` is the unit the *filing* must already state the figure in — never a
# conversion target. Nothing here converts currencies: an Indian filer quoting
# a market in USD billion yields no entry, because a converted figure could not
# be grounded in the sentence it came from and no FX rate exists in this
# pipeline to check it against. Coverage lost, auditability kept.
#
# `capex` settles against the magnitude of investing cash flow, which also
# carries acquisitions and treasury investments — a capex promise settled in
# an M&A year therefore reads generously. Each settled promise records the
# column and both values in metadata so that is auditable rather than hidden.
GUIDANCE_METRICS: dict[str, dict] = {
    "revenue": {"frame": "financials", "column": "revenue", "unit": "inr_cr"},
    "pat": {"frame": "financials", "column": "pat", "unit": "inr_cr"},
    "eps": {"frame": "financials", "column": "eps", "unit": "inr"},
    "operating_margin_pct": {
        "frame": "financials", "column": "opm_pct", "unit": "pct",
    },
    "capex": {
        "frame": "cashflow", "column": "cfi", "unit": "inr_cr", "absolute": True,
    },
}

# Which annual-report section each text-derived sub-metric requires, in
# preference order (KTD4). A sub-metric reads indeterminate when none of its
# sections was usable for that year, regardless of what other sections that
# year yielded.
#
# `tam_runway` is the one ranked fallback and it is deliberate: it widens
# coverage for the sub-metric whose claim is qualitative anyway. The two that
# settle against numbers accept no substitute — a chairman's letter offers
# aspiration, not targets a financials row can settle.
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "promises_kept_ratio": ("mdna",),
    "capex_pipeline": ("mdna",),
    "tam_runway": ("mdna", "chairman"),
}


def is_number(value) -> bool:
    """Whether a value is a usable numeric reading.

    Booleans are excluded deliberately: `isinstance(True, int)` is True in
    Python, so a model returning `true` where a target value belongs would
    otherwise validate and settle as 1. Both sides of the extraction seam need
    the same answer to that, which is why it lives here with the rest of the
    contract rather than being restated in each.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def schema_fingerprint() -> dict:
    """The parts of this contract a regime hash must cover.

    Assembled rather than hashing the module's source: a docstring edit is not
    a regime change, and fragmenting score history on one would be a false
    positive of exactly the kind `_source_file` exclusion already prevents in
    the metric registry.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "fields": {kind: FIELDS[kind] for kind in ENTRY_KINDS},
        "guidance_metrics": GUIDANCE_METRICS,
        "required_sections": REQUIRED_SECTIONS,
    }
