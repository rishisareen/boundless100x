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
# 4 — a numeral must also be denominated as the field claims. Matching the bare
#     digits proved only that they occur: "capex of USD 500 million" grounded an
#     `amount_inr_cr: 500` entry, wrong by two orders of magnitude, and
#     promises-kept settled it against real INR-crore financials. Optional
#     free-text fields are also type- and length-bounded now.
# 5 — guidance can be a growth *rate*, and every guidance entry must now say
#     whose growth it describes. The corpus scan found market and economy
#     growth rates outnumbering company-subject ones roughly four to one, in
#     the same sections and the same sentence shape; a percentage is the one
#     figure where subject and quantity cannot be told apart by type,
#     grounding, or unit. Entries validated before this rule carry no subject
#     and must be re-extracted rather than counted.
# 6 — a figure is stored in the unit the filing stated it in, never converted
#     and no longer discarded for being foreign. Version 3 made a non-INR
#     statement yield nothing at all; that lost the reading *and* hid the
#     coverage gap, which then showed up as an absent signal indistinguishable
#     from a filing that said nothing. Every figure-bearing entry now carries
#     its stated `unit`, grounding checks the numeral against *that* unit, and
#     the metrics needing INR comparability set aside what they cannot use and
#     say so. Which entries survive changes in both directions, so nothing
#     validated under the old rule may be read from cache.
# 7 — `usd_tn` and `inr_lakh_cr`, found by the first live sweep. Both are units
#     the corpus states figures in and the vocabulary had no word for, so five
#     genuinely-present readings were extracted in the nearest available unit
#     and then correctly refused by grounding.
SCHEMA_VERSION = 7

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

# ── Guidance subject (KTD8) ──
# Whose growth a guidance entry describes. Only the company's own is a promise
# management can be held to; a market, industry or economy forecast is not,
# however confidently it is stated and however much it looks like one.
#
# This is the one place in the schema where subject cannot be inferred from
# anything else. "expected to grow by 20%" is a promise or a macro forecast
# depending only on what the sentence is about — the type is the same, the
# grounding is the same, the unit is the same. The corpus bears that out:
# market-subject growth rates outnumber company-subject ones roughly four to
# one, in the same MD&A sections, and one real sentence reads "the Company
# expects the market to grow by 4-5%" — naming the company and still promising
# nothing. So the extractor is asked for the subject explicitly and
# promises-kept counts only `company`.
#
# Market-subject entries are stored rather than dropped, and it is worth being
# honest about what that buys: no current metric reads them. `tam_runway` needs
# a market *size*, not a market *growth rate*. Keeping them is cheap, makes the
# corpus's actual content visible in the data rather than in a one-off scan,
# and follows the same rule as a foreign-currency figure — a grounded reading
# is worth storing even when nothing can yet use it.
SUBJECT_COMPANY = "company"
SUBJECT_MARKET = "market"
GUIDANCE_SUBJECTS = (SUBJECT_COMPANY, SUBJECT_MARKET)

# ── Stated units (KTD5) ──
# The unit the *filing* stated a figure in. Nothing here converts: an exchange
# rate moves constantly and feeds both regime hashes, so every revision of one
# would reset every ticker's momentum baseline — a cost far out of proportion
# to the coverage it buys.
#
# But storing the figure as stated is strictly better than discarding it, which
# is what schema 3 did. A discarded reading is lost twice over: the entry is
# gone, and the coverage gap it represents shows up downstream as an absent
# signal, indistinguishable from a filing that said nothing at all. Stored, the
# entry stays grounded and auditable, the gap is visible in the data, and a
# later FX decision finds the figures already recorded.
#
# The first three are the units metrics settle in. The rest are the foreign and
# mis-scaled ones the corpus actually contains — a pharma exporter states its
# market in USD billion, and Indian filings mix crore, lakh and million freely.
# Anything outside the set is refused, like every other vocabulary here.
# `usd_tn` and `inr_lakh_cr` were added after the first live sweep, which is
# the only way this set can honestly be built. The pilot's discards were not
# defects: ZYDUSLIFE states its markets in USD *trillion* and CAMS states
# industry AUM in *lakh crore*, and with neither unit available the extractor
# reached for the nearest one and the grounding check refused it — correctly,
# since "1.98 trillion" is not 1.98 billion and "40 lakh crore" is not 40
# crore. Five real readings were lost to a vocabulary that had never met the
# filings.
UNIT_INR_CR = "inr_cr"
UNIT_INR = "inr"
UNIT_PCT = "pct"
SETTLING_UNITS = (UNIT_INR_CR, UNIT_INR, UNIT_PCT)
FOREIGN_UNITS = ("usd_mn", "usd_bn", "usd_tn", "inr_lakh", "inr_mn", "inr_lakh_cr")
UNITS = SETTLING_UNITS + FOREIGN_UNITS

FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    GUIDANCE: {
        "required": ("metric", "target_value", "target_period", "subject", "unit")
                    + COMMON_FIELDS,
        # A range is guided as often as a point target in Indian annual
        # reports. The lower bound is the promise (U5), so the high end is
        # optional context rather than part of the test.
        "optional": ("target_value_high",),
    },
    CAPEX: {
        "required": ("amount_inr_cr", "commissioning_year", "unit") + COMMON_FIELDS,
        "optional": ("description",),
    },
    TAM: {
        "required": ("market_size_inr_cr", "unit") + COMMON_FIELDS,
        "optional": ("market", "period"),
    },
}

# `amount_inr_cr` and `market_size_inr_cr` say "INR crore" in their own names,
# so a USD-stated figure in them is only safe because `unit` records the truth
# and every consumer checks it — which `capex_pipeline` did not, summing
# `amount_inr_cr` straight into a rupee total.
_KIND_UNITS = {CAPEX: UNIT_INR_CR, TAM: UNIT_INR_CR}


def settling_unit(kind: str, metric: str | None = None) -> str | None:
    """The stated unit an entry must carry for a metric to be able to read it.

    Guidance settles in whatever unit its guided quantity is measured in; capex
    and TAM settle in INR crore, because that is what their field names claim
    and what the frames they are compared against are denominated in. An entry
    in any other unit is kept and skipped, never converted.
    """
    if kind == GUIDANCE:
        spec = GUIDANCE_METRICS.get(metric)
        return spec["unit"] if spec else None
    return _KIND_UNITS.get(kind)


def is_settleable(kind: str, entry: dict, metric: str | None = None) -> bool:
    """Whether an entry's stated unit is one the reading metric can use."""
    expected = settling_unit(kind, metric)
    return expected is not None and (entry or {}).get("unit") == expected


def partition_by_unit(kind: str, entries) -> tuple[list, list]:
    """Split entries into the ones a metric can settle and the units it cannot.

    Lives here for the same reason `_entries_by_year` factors the structurally
    identical provenance rule in the metric module: all three INR-comparable
    sub-metrics need "drop what this unit cannot express, and remember what the
    unit was", and three hand-rolled copies is how `capex_pipeline` came to sum
    `amount_inr_cr` into a rupee total on the strength of the field name alone.

    Returns `(usable, set_aside_units)` with the units sorted and deduplicated,
    so a caller can name them in an error without repeating that bookkeeping.
    """
    usable, set_aside = [], set()
    for entry in entries or []:
        if is_settleable(kind, entry, (entry or {}).get("metric")):
            usable.append(entry)
        else:
            set_aside.add(str((entry or {}).get("unit")))
    return usable, sorted(set_aside)

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
#
# **`growth: True` settles against the change between two consecutive annual
# rows** rather than against one column (KTD4). This is the shape Indian filers
# actually guide in — the corpus holds five company-subject growth-rate
# statements and not one forward absolute-INR-crore figure — and it needs no
# exchange rate, because a percentage carries its unit in the numeral and so
# grounds in its own sentence whatever currency the accounts are kept in.
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
    "revenue_growth_pct": {
        "frame": "financials", "column": "revenue", "unit": "pct", "growth": True,
    },
    "pat_growth_pct": {
        "frame": "financials", "column": "pat", "unit": "pct", "growth": True,
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
        "guidance_subjects": GUIDANCE_SUBJECTS,
        "units": UNITS,
        "required_sections": REQUIRED_SECTIONS,
    }
