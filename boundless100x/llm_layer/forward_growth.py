"""Forward-growth extraction: the content gate, the boundary validator, the cache.

Guidance, capacity plans and market-size statements live in annual-report
prose, and no amount of arithmetic over the financials will produce them. This
module is what turns that prose into structured data the compute engine can
read offline — and, mostly, what stops it turning the *wrong* prose into
confident nonsense.

Three guards stand between a PDF and a stored forward signal, and each catches
something the others cannot.

**Provenance gates the input** (KTD4). A section tagged `fallback` holds
first-N-pages text rather than the section it names, so it is never submitted.
Cheaper, and no tokens spent on a chairman's letter that cannot answer the
question. Provenance is tagged per *section* while a report year usually
carries a mix — 16 of 29 real report-years do — so every stored entry records
the section it came from, or a promises-kept reading could be built from a
chairman's letter while MD&A was never read.

**A content gate stands behind provenance** (KTD9), because `found` is a claim
rather than a guarantee. Heading detection over arbitrary filer layouts will
never be exact: a review of the corpus found 8 of 18 `found` MD&A slices were
actually auditor's reports, governance, CSR or HR text. Phase 0 fixed the root
cause — a heading must now open its line and not be continued in lowercase —
and precision rose from ~56% to ~85%, but at least one residual survives: a
bare "Management Discussion and Analysis" line immediately followed by
governance prose, structurally indistinguishable from a real heading. Phase 0
could tolerate that residue because its fallback text was only Pass 1
background. This cannot: the extractor mines whatever it is handed, and
governance prose yields well-formed, confident, wrong guidance. So a `found`
section must additionally *look like* the section it claims to be, or it is
downgraded to **`suspect`** and treated exactly as `fallback`. The bucket is
reported rather than folded into `fallback`, because how often the tag was
wrong is the number that tells you whether the gate is working.

**Grounding stands behind shape** (KTD3). Type-checking cannot distinguish a
well-formed reading from a well-formed fabrication — every field is the right
type either way, and the result becomes a forward signal. Because each entry
must carry the verbatim source sentence, the validator additionally verifies
that sentence is a literal substring of the text actually submitted for that
year, and that the entry's own value and period appear inside it. One string
search turns "auditable if a reader bothers" into "verified before storage",
and closes fabrication and any instruction-like text embedded in a filing in
the same move.

Note what none of these can do for the other: KTD3's grounding *passes* on a
suspect slice, because the quoted sentence really is in the submitted text —
the submitted text is genuinely the auditor's report. Only content can tell
those apart, which is why KTD9 exists as its own guard.
"""

import hashlib
import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from boundless100x import forward_growth_schema as schema
from boundless100x.data_fetcher.download_annual_reports import SECTION_PATTERNS

logger = logging.getLogger(__name__)

# Re-exported from the shared contract so this module reads self-contained;
# the compute engine reads the same three values without importing llm_layer.
FOUND = schema.FOUND
SUSPECT = schema.SUSPECT
FALLBACK = schema.FALLBACK

PROMPT_NAME = "forward_growth_extraction.txt"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Only sections some sub-metric can actually read are worth paying for.
# `governance` is extracted by Phase 0 and answers none of the three questions
# below, so it is gated for reporting but never submitted.
EXTRACTABLE_SECTIONS = tuple(sorted(
    {name for names in schema.REQUIRED_SECTIONS.values() for name in names}
))

# Per gated section, per report year. Must be at least the largest per-section
# cap in `annual_reports.sections`, because `build_submission` truncates from
# the *front*: MD&A opens with the economic review and carries its guidance in
# the outlook near the end, so a budget below the section cap reliably submits
# the only half that cannot contain a target.
DEFAULT_CHAR_BUDGET = 12000
# The gate reads the opening of a slice. A wrong-section slice is wrong from
# its first line; reading further would only let a passing mention of "outlook"
# ten pages in rescue an auditor's report.
DEFAULT_SCAN_CHARS = 3000
# How far in the section's own heading may sit and still count as opening the
# slice. Detection already established it *is* a heading; this only confirms
# the slice begins there rather than somewhere else entirely.
_HEADING_WINDOW = 150
# Disqualifiers are canonical *openings* of other sections, so they are looked
# for at the start. Beyond this a mention is commentary — a genuine chairman's
# statement noting that the auditors raised no qualification must not be
# thrown away for using the words.
_DISQUALIFIER_WINDOW = 600

# One marker is the bar, deliberately. An earlier two-marker rule modelled on
# SEBI LODR Schedule V(B)'s mandated MD&A contents rejected 11 of the 13
# `found` MD&A slices in the fetched corpus, of which 11 were genuine — real
# MD&A opens with narrative economy and industry prose and reaches the mandated
# sub-headings pages later, well past any sane scan window.
DEFAULT_MIN_MARKERS = {"mdna": 1, "chairman": 1, "governance": 1}

# Subject-matter markers: what each section is actually *about* in its opening
# paragraphs, measured against the real corpus rather than against the formal
# structure it is supposed to have.
SECTION_MARKERS: dict[str, tuple[re.Pattern, ...]] = {
    "mdna": tuple(re.compile(p, re.I) for p in (
        # Economic review — how most Indian MD&A sections actually open.
        r"(?:global|indian|india[’']?s|domestic|world)\s+econom",
        r"econom(?:y|ic)\s+(?:review|overview|scenario|outlook|environment|landscape)",
        r"\beconomy\s+overview\b",
        r"global\s+growth",
        r"\bgdp\b",
        # Industry review, however the filer phrases the possessive.
        r"industry[’']?s?\s+(?:structure|overview|review|scenario|outlook"
        r"|landscape|trends?|growth|dynamics|developments?)",
        # SEBI LODR Schedule V(B) sub-headings, when they do appear early.
        r"opportunit\w+\s+and\s+threats",
        r"segment[\s–—-]*wise",
        r"risks?\s+and\s+concerns",
        r"internal\s+control\s+systems?",
        r"key\s+financial\s+ratios",
        r"financial\s+performance\s+with\s+respect\s+to",
        r"\boutlook\b",
        # Market and demand commentary.
        r"\bmarket\s+(?:size|share|dynamics|outlook|demand|opportunity)",
        r"\bdemand\s+(?:environment|outlook|drivers?)",
        r"growth\s+drivers?",
    )),
    "chairman": tuple(re.compile(p, re.I) for p in (
        r"dear\s+(?:share\s*holders?|members|stakeholders|friends|colleagues)",
        r"it\s+gives\s+me|it\s+is\s+my\s+(?:privilege|pleasure)",
        r"i\s+am\s+(?:pleased|delighted|happy|privileged)",
        r"on\s+behalf\s+of\s+the\s+board",
        r"your\s+company",
        r"yours\s+sincerely|warm\s+(?:regards|greetings)",
        r"annual\s+report\s+of\s+(?:your|the)\s+company",
    )),
    "governance": tuple(re.compile(p, re.I) for p in (
        r"philosophy\s+on\s+(?:the\s+)?(?:code\s+of\s+)?(?:corporate\s+)?governance",
        r"composition\s+of\s+the\s+board",
        r"board\s+of\s+directors",
        r"audit\s+committee",
        r"nomination\s+and\s+remuneration",
        r"board\s+met\b",
        r"general\s+(?:body\s+)?meeting",
    )),
}

# The canonical *opening* of a different section. Each of these is how another
# statutory section begins, not merely a phrase it contains — that distinction
# matters, because `auditor's report` on its own throws away a real chairman's
# statement noting the auditors raised no qualification, while `we have
# audited` only ever opens an audit opinion.
_OTHER_SECTION_OPENINGS = (
    r"philosophy\s+on\s+(?:the\s+)?(?:code\s+of\s+)?(?:corporate\s+)?governance",
    r"terms\s+of\s+reference\s+of\s+the\s+[\w\s]{0,20}committee",
    r"key\s+audit\s+matters",
    r"we\s+have\s+audited",
    r"independent\s+auditor'?[’']?s?\s+report",
    r"report\s+on\s+the\s+audit\s+of",
    r"corporate\s+social\s+responsibility\s+(?:policy|committee|activities)",
)

# A slice whose opening says the section is *elsewhere* is a pointer, not the
# section. This is the second real failure in the corpus: a Board's Report
# heading "MANAGEMENT DISCUSSION AND ANALYSIS:" followed immediately by "the
# detailed Management Discussion and Analysis forms a part of this report at
# Annexure-A", then governance prose.
_POINTER_PHRASES = (
    r"forms?\s+(?:a\s+)?part\s+of\s+th(?:is|e)\s+report",
    r"forms?\s+an\s+integral\s+part\s+of\s+th(?:is|e)\s+report",
    r"is\s+annexed|annexed\s+(?:to|herewith)",
    r"\bannexure[\s\-–—]?[A-Z0-9]\b",
    r"(?:given|set\s+out|appears?|provided)\s+(?:elsewhere|separately)",
)

SECTION_DISQUALIFIERS: dict[str, tuple[re.Pattern, ...]] = {
    "mdna": tuple(
        re.compile(p, re.I) for p in _OTHER_SECTION_OPENINGS + _POINTER_PHRASES
    ),
    "chairman": tuple(
        re.compile(p, re.I) for p in _OTHER_SECTION_OPENINGS + _POINTER_PHRASES
    ),
    # A governance report legitimately opens on its own philosophy, discusses
    # committee terms of reference, and reports CSR — only an audit opinion
    # masquerading as one, or a pointer, is disqualifying.
    "governance": tuple(re.compile(p, re.I) for p in (
        r"key\s+audit\s+matters", r"we\s+have\s+audited",
        r"report\s+on\s+the\s+audit\s+of",
    ) + _POINTER_PHRASES),
}


# ── The content gate ───────────────────────────────────────────────────────


def _gate_one(name: str, text: str, scan_chars: int, min_markers: int) -> tuple[bool, str]:
    """Whether a slice looks like the section it claims to be, and why not.

    Two independent questions, and both have to be asked. *Does it open like
    another section?* — a canonical opening of an audit opinion, a corporate
    governance report, or a pointer saying the section is at an annexure means
    the slice is that other thing whatever else it contains. *Is it about the
    right subject?* — either the section's own heading opens the slice, or its
    subject-matter markers appear early.

    The heading counts because detection already established it *is* a heading
    rather than a cross-reference; this only confirms the slice begins there.
    On its own it would be nearly a no-op, which is why the disqualifiers carry
    the residual cases KTD9 was written for — in both real failures the heading
    was present and correct, and the prose underneath belonged to another
    section.
    """
    opening = (text or "")[:scan_chars]
    if not opening.strip():
        return False, "empty slice"

    blocked = [
        pattern.pattern
        for pattern in SECTION_DISQUALIFIERS.get(name, ())
        if pattern.search(opening[:_DISQUALIFIER_WINDOW])
    ]
    if blocked:
        return False, (
            f"opens like a different section (or a pointer to one): "
            f"{', '.join(blocked)}"
        )

    heading = SECTION_PATTERNS.get(name)
    if heading is not None and heading.search(opening[:_HEADING_WINDOW]):
        return True, ""

    hits = [
        pattern.pattern
        for pattern in SECTION_MARKERS.get(name, ())
        if pattern.search(opening)
    ]
    if len(hits) < min_markers:
        return False, (
            f"no {name} heading opens the slice, and only {len(hits)} of the "
            f"{min_markers} subject markers a {name} section should show appear "
            f"in its first {scan_chars} chars"
        )
    return True, ""


def gate_sections_with_reasons(
    sections_by_year: dict,
    scan_chars: int = DEFAULT_SCAN_CHARS,
    min_markers: dict | None = None,
    enabled: bool = True,
) -> tuple[dict, dict]:
    """Three-valued provenance per section, plus why anything was downgraded.

    Runs on the extraction path rather than in the fetcher on purpose: Phase 0's
    `.sections.json` sidecars stay exactly as they are, and the gate can be
    tuned — or switched off — without refetching twenty annual reports.
    """
    minimums = {**DEFAULT_MIN_MARKERS, **(min_markers or {})}

    gated: dict[str, dict] = {}
    reasons: dict[str, dict] = {}
    for year, sections in (sections_by_year or {}).items():
        if not isinstance(sections, dict):
            continue
        gated[year], reasons[year] = {}, {}
        for name, section in sections.items():
            provenance = (section or {}).get("provenance", FALLBACK)
            if provenance != FOUND or not enabled:
                # Only a `found` tag is a claim. A `fallback` slot never
                # claimed to be the section, so there is nothing to disprove.
                gated[year][name] = provenance
                continue

            genuine, why = _gate_one(
                name, (section or {}).get("text", ""), scan_chars,
                minimums.get(name, 2),
            )
            gated[year][name] = FOUND if genuine else SUSPECT
            if not genuine:
                reasons[year][name] = why
                logger.info(
                    f"Content gate: {year} {name} downgraded found -> suspect ({why})"
                )
    return gated, reasons


def gate_sections(sections_by_year: dict, **kwargs) -> dict:
    """Three-valued provenance per section. See `gate_sections_with_reasons`."""
    return gate_sections_with_reasons(sections_by_year, **kwargs)[0]


def build_submission(
    sections_by_year: dict,
    gated: dict,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> dict:
    """`{year: {section: text}}` — only what may be sent, capped.

    A year with no usable section is absent rather than present-and-empty, so
    "nothing to submit" is a property of the payload rather than something
    every caller has to check for separately.
    """
    payload: dict[str, dict] = {}
    for year, sections in (sections_by_year or {}).items():
        if not isinstance(sections, dict):
            continue
        usable = {
            name: (sections[name] or {}).get("text", "")[:char_budget]
            for name in EXTRACTABLE_SECTIONS
            if name in sections and gated.get(year, {}).get(name) == FOUND
        }
        usable = {name: text for name, text in usable.items() if text.strip()}
        if usable:
            payload[year] = usable
    return payload


# ── Prompt assembly ────────────────────────────────────────────────────────


def vocabulary_prompt_block() -> str:
    """The closed field list, rendered for the prompt.

    Phase 1's lesson, applied: asked for an id without a menu, a model invents
    plausible ones. Sending the vocabulary is what makes boundary validation
    something other than a rejection machine.
    """
    lines = ["Entry kinds and their fields:"]
    for kind in schema.ENTRY_KINDS:
        fields = schema.FIELDS[kind]
        lines.append(f"  {kind}:")
        lines.append(f"    required: {', '.join(fields['required'])}")
        if fields["optional"]:
            lines.append(f"    optional: {', '.join(fields['optional'])}")

    lines.append("")
    lines.append("Valid values for guidance.metric (nothing else is accepted):")
    for metric_id, spec in sorted(schema.GUIDANCE_METRICS.items()):
        lines.append(f"  {metric_id} — settled against a {spec['unit']} figure")

    lines.append("")
    lines.append("Valid values for unit (nothing else is accepted):")
    lines.append(
        f"  {', '.join(schema.SETTLING_UNITS)} — units this system can settle "
        f"against the accounts"
    )
    lines.append(
        f"  {', '.join(schema.FOREIGN_UNITS)} — report these as stated too; the "
        f"figure is kept and simply not settled"
    )

    lines.append("")
    lines.append(
        f"Valid values for guidance.subject: {', '.join(schema.GUIDANCE_SUBJECTS)}"
    )
    return "\n".join(lines)


def render_report_text(submission: dict) -> str:
    """The submitted sections, labelled so the model can attribute each entry."""
    blocks = []
    for year in sorted(submission, reverse=True):
        blocks.append(f"=== REPORT YEAR {year} ===")
        for name, text in sorted(submission[year].items()):
            blocks.append(f"--- SECTION: {name} ---")
            blocks.append(text)
    return "\n".join(blocks)


def prompt_template() -> str:
    return (PROMPTS_DIR / PROMPT_NAME).read_text()


def build_prompt(ticker: str, company_name: str, submission: dict) -> str:
    """The exact prompt an extraction call sends.

    **One assembly, because the sweep prices what the live call sends.** This
    needs no orchestrator state — it is template plus vocabulary plus report
    text — so it lives here rather than as a method, and the dry run can price
    a real prompt with no API key at all. Two copies would be wrong in the way
    that matters: a new placeholder breaks the unkeyed path with a `KeyError`,
    and a new *block* diverges the estimate silently, which is worse.
    """
    return prompt_template().format(
        ticker=ticker,
        company_name=company_name,
        vocabulary=vocabulary_prompt_block(),
        report_text=render_report_text(submission),
    )


def plan_submission(config: dict, sections_by_year: dict, llm=None) -> dict:
    """Everything that decides *what* an extraction call would be given.

    Shared by Stage 1.5 and the sweep's dry run, which is the whole point: the
    sweep exists to price the call the pipeline would make, and a second copy
    of the gate, the char budget and the model resolution is a copy that can
    drift. It already had: the stage preferred the live orchestrator's model
    and budget while the sweep always read config, so a deep-mode run would
    have priced Sonnet and billed Opus.

    `llm` is optional because pricing a spend is what one does *before*
    configuring the credentials to make it.
    """
    from boundless100x.llm_layer.orchestrator import (
        forward_growth_char_budget,
        forward_growth_model,
    )

    gate_config = (config or {}).get("annual_reports", {}).get("content_gate", {})
    gated, reasons = gate_sections_with_reasons(
        sections_by_year,
        scan_chars=gate_config.get("scan_chars", DEFAULT_SCAN_CHARS),
        min_markers=gate_config.get("min_markers"),
        enabled=gate_config.get("enabled", True),
    )
    budget = (
        llm.forward_growth_char_budget if llm
        else forward_growth_char_budget(config)
    )
    return {
        "gated": gated,
        "reasons": reasons,
        "submission": build_submission(sections_by_year, gated, char_budget=budget),
        "model": llm.forward_growth_model if llm else forward_growth_model(config),
        "char_budget": budget,
    }


@lru_cache(maxsize=1)
def prompt_digest() -> str:
    """Fingerprint of the prompt file.

    Part of the sidecar version: a prompt change that silently reused stale
    extractions would be indistinguishable from the new prompt working.

    Cached for the life of the process. Every cache read calls this — including
    the hydration-only path, once per ticker on a `watchlist advance` run — and
    the prompt file cannot change mid-run, so re-reading and re-hashing it per
    ticker is pure repetition. This caches the digest, never the invalidation
    decision: the submitted text is still digested afresh on every call.
    """
    try:
        return hashlib.sha256(prompt_template().encode()).hexdigest()[:12]
    except OSError:
        return "unavailable"


# ── Boundary validation ────────────────────────────────────────────────────


_is_number = schema.is_number

# Typographic variants a filing uses and a model silently normalises away.
# These are the document's *typesetting*, never part of its claim.
_TYPOGRAPHY = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ", "­": "",
})


def ground_text(text: str) -> str:
    """Text reduced to what a grounding comparison should actually care about.

    **Whitespace is the whole reason this exists.** PDF extraction preserves
    the printed line breaks, so a sentence in a section slice reads
    `"...USD 860.1 billion \\nduring FY2025-26."` — one MD&A slice in the corpus
    carries 320 such breaks. A model asked to quote verbatim returns the
    sentence as it reads, unwrapped, which is the same claim and a different
    byte string. Comparing raw, KTD3's substring check rejected 8 of 8
    genuinely-present statements on the first live run: the guard was not
    catching fabrication, it was catching typesetting.

    Normalising both sides costs the guard nothing. A sentence that was never
    in the document still does not appear once whitespace is collapsed, so
    fabrication and embedded instruction text are caught exactly as before.
    Case is deliberately preserved — that is a real property of a quotation,
    and no PDF extractor changes it.
    """
    return " ".join(
        unicodedata.normalize("NFKC", text or "").translate(_TYPOGRAPHY).split()
    )


# A scale word *follows* its numeral in both English and Indian usage —
# "500 million", "1,500 crore" — so the multiplier is read from the text
# immediately after. `lakh` and `million` are two orders of magnitude from a
# crore, so either makes an `inr_cr` claim wrong.
_WRONG_SCALE_AFTER_NUMBER = re.compile(
    r"^\W{0,3}(?:million|mn|billion|bn|trillion|tn|lakhs?|thousand|k)\b", re.I
)

# A currency marker *precedes* its numeral — "USD 500", "$500", "EUR 12".
_WRONG_CURRENCY_BEFORE_NUMBER = re.compile(
    r"(?:\bus\s*\$|\busd|\beur|\bgbp|\bjpy|[$€£¥])\W{0,3}$", re.I
)

_USD_BEFORE_NUMBER = re.compile(r"(?:\bus\s*\$|\busd|\$)\W{0,3}$", re.I)

# Deliberately tight. A wide window makes any foreign figure elsewhere in the
# sentence condemn a sound one — "revenue grew from USD 100 million to
# Rs 1,500 crore" carries both, and the rupee figure in it is good evidence.
_UNIT_WINDOW_BEFORE = 12
_UNIT_WINDOW_AFTER = 18

# What each declared unit must read like where its numeral sits (U5). The
# check that used to ask "is this INR crore?" now asks "is this the unit the
# entry says it is?" — a strictly wider question, and the one that lets a
# foreign figure be stored honestly instead of being thrown away.
#
# `inr_lakh` carries a negative lookahead because "lakh crore" is its own
# Indian scale, five orders of magnitude from a lakh. Without it a
# 40-lakh-crore industry AUM would ground as 40 lakh.
_SCALE_AFTER = {
    "inr_cr": re.compile(r"^\W{0,3}(?:crores?|cr)\b", re.I),
    "inr_lakh": re.compile(r"^\W{0,3}lakhs?\b(?!\W{0,3}crores?\b)", re.I),
    "inr_lakh_cr": re.compile(r"^\W{0,3}lakhs?\W{0,3}crores?\b", re.I),
    "inr_mn": re.compile(r"^\W{0,3}(?:million|mn)\b", re.I),
    "usd_mn": re.compile(r"^\W{0,3}(?:million|mn)\b", re.I),
    "usd_bn": re.compile(r"^\W{0,3}(?:billion|bn)\b", re.I),
    "usd_tn": re.compile(r"^\W{0,3}(?:trillion|tn)\b", re.I),
}
# A dollar marker has to precede the numeral for a USD claim to stand. Without
# it, "500 million" is as likely to be rupees.
_NEEDS_USD_MARKER = frozenset({"usd_mn", "usd_bn", "usd_tn"})

# Units an Indian filing routinely leaves implicit — a table headed "Rs crore"
# writes the figures bare. For these, and only these, the absence of a
# conflicting unit beside the numeral grounds the claim, which is exactly the
# rule that stood before stated units existed. Everything else has to be
# written out, because nothing in an Indian filing defaults to dollars.
_IMPLICIT_UNITS = frozenset({"inr_cr", "inr"})

# A percent field is exempt from all of it: "18%" carries its unit in the
# numeral, and a guided range writes "4-5%" with the lower bound — the promise
# — nowhere near the sign.
_UNCHECKED_UNITS = frozenset({"pct", None})

# A quotation has to be *prose*. PDF extraction flattens charts and tables into
# runs of bare numerals, and one reached storage: IXIGO's FY2024 sidecar held a
# `tam` entry whose `source_sentence` was "3,808 5,904 8% 6% 7% 12% CAGR
# (FY23-28) 1,365 2,900 1,660" — a chart's axis labels. It passed every guard
# honestly. The figure really is in the submitted text, denominated as claimed,
# beside a plausible period; grounding cannot tell a claim from a scraped axis,
# because on those tests it *is* one. It then fed `tam_runway` a Rs 3,808 crore
# addressable market and produced the only value that metric returned across
# the whole corpus — the well-formed, well-grounded, wrong-thing failure KTD9
# names, one level below the section gate it was written for.
#
# Five is measured, not guessed: across the 60 stored quotations the fragment
# has 2 alphabetic words and the next-lowest genuine sentence has 9, so the
# threshold sits in an empty gap and rejects exactly that one entry.
_MIN_PROSE_WORDS = 5
_PROSE_WORD = re.compile(r"[A-Za-z]{2,}")

# Ceiling on an optional free-text field. These are the only parts of an entry
# nothing else constrains — not grounded, not settled against a number — so a
# bound is what stops a malformed response putting an arbitrary payload into
# the sidecar and into metric metadata.
_MAX_FREE_TEXT = 300


def _number_appears(sentence: str, value, unit: str | None = None) -> bool:
    """Whether a number is written in its own quoted sentence, in the right unit.

    Filings write `1,500` and models return `1500`; both are the same claim, so
    grouping separators are stripped from the haystack rather than guessed at in
    the needle, and both the integer and decimal renderings are tried.

    **The unit check is the load-bearing half.** Matching the bare numeral
    proves only that the digits occur somewhere — and "capex of USD 500 million
    by FY2027" genuinely contains `500`, so an entry claiming 500 in INR crore
    grounds cleanly while being wrong by two orders of magnitude, and
    `compute_promises_kept` then settles it against real INR-crore financials.
    The prompt does state the rule, but a prompt rule nothing enforces is not a
    rule; this is the enforcement.

    Since U5 the entry declares the unit itself, so the question became "is the
    numeral denominated the way this entry says it is?" rather than "is it INR
    crore?". A USD figure now grounds *as a USD figure* — it is the reading
    metrics that refuse it, with the reason, instead of the boundary throwing
    the filing's own words away.

    Checking per occurrence rather than per sentence matters throughout:
    "revenue grew from USD 100 million to Rs 1,500 crore" carries both, and the
    rupee figure in it is perfectly good evidence for an `inr_cr` claim.
    """
    return bool(_number_positions(sentence, value, unit))


def _number_positions(sentence: str, value, unit: str | None = None) -> list[tuple]:
    """Every occurrence of a number that is also denominated as claimed."""
    haystack = re.sub(r"(?<=\d)[,\s](?=\d)", "", sentence or "")
    candidates = {f"{float(value):.10g}"}
    if float(value).is_integer():
        candidates.add(str(int(value)))

    positions = sorted(
        (match.start(), match.end())
        for candidate in candidates
        for match in re.finditer(re.escape(candidate), haystack)
    )
    if unit in _UNCHECKED_UNITS:
        return positions

    return [
        (start, end)
        for start, end in positions
        if _denominated(
            unit,
            haystack[max(0, start - _UNIT_WINDOW_BEFORE):start],
            haystack[end:end + _UNIT_WINDOW_AFTER],
        )
    ]


def _denominated(unit: str, before: str, after: str) -> bool:
    """Whether the text around one numeral denominates it as `unit` claims.

    Two ways to qualify, and they cannot both apply: the unit is *written out*
    beside the numeral, or — for the rupee units an Indian filing routinely
    leaves implicit — nothing beside the numeral contradicts it. No `usd_*`
    unit is implicit, because nothing in an Indian filing defaults to dollars.
    """
    scale = _SCALE_AFTER.get(unit)
    if scale is not None and scale.match(after):
        if unit in _NEEDS_USD_MARKER:
            return bool(_USD_BEFORE_NUMBER.search(before))
        # A rupee unit still may not sit behind a foreign currency marker.
        # Without this, "capex of USD 500 million" grounded an `inr_mn` claim
        # and "USD 1500 crore" grounded an `inr_cr` one — the scale word
        # matched and the check returned before ever reading what came before
        # the numeral. That is the same well-typed-fabrication class the field
        # name used to hide, moved to the `unit` field.
        return not _WRONG_CURRENCY_BEFORE_NUMBER.search(before)
    return (
        unit in _IMPLICIT_UNITS
        and not _WRONG_SCALE_AFTER_NUMBER.match(after)
        and not _WRONG_CURRENCY_BEFORE_NUMBER.search(before)
    )


# How far apart a value and the period it is guided for may sit and still be
# one claim. A clause, roughly — wide enough for "revenue of Rs 1,500 crore for
# the financial year ending March 2026", narrow enough that a historical figure
# at one end of a sentence cannot borrow a target year from the other.
_CLAIM_WINDOW = 120


def _period_positions(sentence: str, period) -> list[int]:
    """Where a target period's year is named in its own quoted sentence.

    Grounds on the year rather than the exact string: a filing writes `FY26`,
    `FY2026`, or `financial year 2026` for one period, and demanding a literal
    match would discard correct readings far more often than fabricated ones.
    """
    years = re.findall(r"\d{2,4}", str(period or ""))
    if not years:
        return []

    haystack = str(sentence or "")
    positions: list[int] = []
    for year in years:
        positions.extend(m.start() for m in re.finditer(re.escape(year), haystack))
        if len(year) == 4:
            positions.extend(
                m.start() for m in re.finditer(re.escape(year[2:]), haystack)
            )
    return sorted(positions)


def _period_appears(sentence: str, period) -> bool:
    return bool(_period_positions(sentence, period))


def _value_and_period_cohere(sentence: str, value, unit, period) -> bool:
    """Whether the value and the period read as *one* claim, not two.

    Checking each independently against the whole sentence is not enough, and
    the failure is not hypothetical: "We delivered 26% revenue growth in FY23
    and separately target a capex commissioning by FY26" satisfies both checks
    for a guidance entry claiming 26 in FY26 — a historical figure married to
    an unrelated year, stored as a target and later settled against real
    financials as though management had promised it.

    So the two must also sit near each other. A number that came from one
    clause and a period that came from another is two facts, not a promise.
    """
    numbers = _number_positions(sentence, value, unit)
    periods = _period_positions(sentence, period)
    if not numbers or not periods:
        return False
    return any(
        abs(period_at - number_at) <= _CLAIM_WINDOW
        for number_at, _ in numbers
        for period_at in periods
    )


def _validate_entry(
    kind: str, entry, year: str, submitted_sections: dict, discarded: list,
    grounded_sections: dict | None = None,
) -> dict | None:
    """One entry, or None with the reason recorded."""
    where = f"{year}.{kind}"

    def drop(reason):
        # The quoted sentence travels with the reason. Reading a discard list
        # without it means going back to the filing to work out whether a
        # refusal was a defect or a genuine gap — which is exactly the question
        # a sweep exists to answer, and the first live sweep had to answer it
        # by hand. Truncated, because this is diagnostics rather than evidence.
        record = {"where": where, "reason": reason}
        if isinstance(entry, dict) and isinstance(entry.get("source_sentence"), str):
            record["source_sentence"] = entry["source_sentence"][:_MAX_FREE_TEXT]
        discarded.append(record)
        logger.warning(f"Forward-growth entry discarded ({where}): {reason}")
        return None

    if not isinstance(entry, dict):
        return drop(f"entry is {type(entry).__name__}, expected an object")

    fields = schema.FIELDS[kind]
    allowed = set(fields["required"]) | set(fields["optional"])
    unknown = sorted(set(entry) - allowed)
    if unknown:
        discarded.append({
            "where": where,
            "reason": f"field(s) outside the declared set stripped: {', '.join(unknown)}",
        })
        logger.warning(f"Forward-growth fields stripped ({where}): {unknown}")
    kept = {key: value for key, value in entry.items() if key in allowed}

    missing = [name for name in fields["required"] if kept.get(name) is None]
    if missing:
        return drop(f"missing required field(s): {', '.join(missing)}")

    section = kept["section"]
    if not isinstance(section, str):
        # An unhashable value here raised TypeError out of the whole
        # validation pass, discarding every entry from a call already paid for.
        return drop(f"section is {type(section).__name__}, expected a string")
    if section not in submitted_sections:
        return drop(
            f"section {section!r} was never submitted for {year} "
            f"(sent: {', '.join(sorted(submitted_sections)) or 'none'})"
        )

    # The submitted section is normalised once per section by the caller, not
    # once per entry: `ground_text` walks the whole 12,000-char slice and
    # always returns the same string for it, while a year's entries can number
    # a dozen. The quoted sentence is normalised once here for the same reason.
    grounded = (grounded_sections or {}).get(section)
    if grounded is None:
        grounded = ground_text(submitted_sections[section])

    sentence = kept["source_sentence"]
    grounded_sentence = ground_text(sentence) if isinstance(sentence, str) else ""
    if grounded_sentence and len(_PROSE_WORD.findall(grounded_sentence)) < _MIN_PROSE_WORDS:
        return drop(
            f"source_sentence is not prose — {len(_PROSE_WORD.findall(grounded_sentence))} "
            f"word(s), needs {_MIN_PROSE_WORDS}. A run of numerals is a chart or "
            f"a table flattened by PDF extraction, and it grounds honestly "
            f"while claiming nothing"
        )
    if not grounded_sentence or grounded_sentence not in grounded:
        return drop(
            f"source_sentence does not appear in the submitted {section} text "
            f"for {year} — the claim is not in the document"
        )

    # Optional free-text fields are the one part of an entry nothing else
    # constrains: they are not grounded (they are genuinely free text) and
    # nothing downstream settles them against a number. Bounding type and
    # length is what stops a hostile or malformed response putting arbitrary
    # content into the sidecar and into MetricResult.metadata, where the
    # module's own "everything is grounded" claim would otherwise vouch for it.
    # Required period fields are free text too — they are grounded only by the
    # *year* inside them, so the rest of the string is unconstrained and a
    # 5,000-character `target_period` was reaching the sidecar and metric
    # metadata. Bound them exactly like the optional fields.
    bounded = [f for f in schema.FIELDS[kind]["optional"] if f != "target_value_high"]
    bounded += [f for f in ("target_period", "commissioning_year") if f in kept]
    for field in bounded:
        if field in kept:
            text = kept[field]
            if not isinstance(text, str) or len(text) > _MAX_FREE_TEXT:
                return drop(
                    f"{field} must be a string of at most "
                    f"{_MAX_FREE_TEXT} characters"
                )

    # The stated unit is closed like every other vocabulary here, and it is
    # checked before the figure so the grounding message can name a real unit.
    unit = kept["unit"]
    if unit not in schema.UNITS:
        return drop(
            f"unit {unit!r} is outside the closed set ({', '.join(schema.UNITS)})"
        )

    if kind == schema.GUIDANCE:
        if not isinstance(kept["metric"], str):
            return drop(f"metric is {type(kept['metric']).__name__}, expected a string")
        spec = schema.GUIDANCE_METRICS.get(kept["metric"])
        if spec is None:
            return drop(
                f"guidance metric {kept['metric']!r} is outside the closed set "
                f"({', '.join(sorted(schema.GUIDANCE_METRICS))})"
            )
        # KTD8. Closed like every other vocabulary here: an invented subject
        # would be silently treated as not-the-company by promises-kept and
        # would look identical to a market forecast, so it is refused at the
        # boundary where the reason can still be recorded.
        if kept["subject"] not in schema.GUIDANCE_SUBJECTS:
            return drop(
                f"guidance subject {kept['subject']!r} is outside the closed set "
                f"({', '.join(schema.GUIDANCE_SUBJECTS)})"
            )
        if not _is_number(kept["target_value"]):
            return drop(f"target_value {kept['target_value']!r} is not a number")
        if "target_value_high" in kept and not _is_number(kept["target_value_high"]):
            return drop(f"target_value_high {kept['target_value_high']!r} is not a number")
        if not _number_appears(sentence, kept["target_value"], unit):
            return drop(
                f"target_value does not appear in its own source_sentence as a "
                f"{unit} figure"
            )
        if not _period_appears(sentence, kept["target_period"]):
            return drop("target_period does not appear in its own source_sentence")
        if not _value_and_period_cohere(
            sentence, kept["target_value"], unit, kept["target_period"]
        ):
            return drop(
                "target_value and target_period appear in the source_sentence but "
                "too far apart to be one claim — a figure from one clause married "
                "to a period from another is two facts, not a promise"
            )

    elif kind == schema.CAPEX:
        if not _is_number(kept["amount_inr_cr"]):
            return drop(f"amount_inr_cr {kept['amount_inr_cr']!r} is not a number")
        if not _number_appears(sentence, kept["amount_inr_cr"], unit):
            return drop(
                f"amount_inr_cr does not appear in its own source_sentence as a "
                f"{unit} figure"
            )
        if not _period_appears(sentence, kept["commissioning_year"]):
            return drop("commissioning_year does not appear in its own source_sentence")
        if not _value_and_period_cohere(
            sentence, kept["amount_inr_cr"], unit, kept["commissioning_year"]
        ):
            return drop(
                "amount_inr_cr and commissioning_year appear in the source_sentence "
                "but too far apart to be one claim"
            )

    elif kind == schema.TAM:
        if not _is_number(kept["market_size_inr_cr"]):
            return drop(f"market_size_inr_cr {kept['market_size_inr_cr']!r} is not a number")
        if not _number_appears(sentence, kept["market_size_inr_cr"], unit):
            return drop(
                f"market_size_inr_cr does not appear in its own source_sentence as "
                f"a {unit} figure"
            )

    return kept


def validate_extraction(raw, submission: dict, gated: dict) -> dict:
    """Turn a raw model response into storable entries, or into nothing.

    Deliberately defensive throughout, exactly as `checkpoints.record_from_pass2`
    is and for the same reason: `_parse_json_response` performs no schema
    validation of any kind, so a malformed, truncated, or simply older response
    reaches here unchecked and every shape must degrade rather than raise.

    Returns `{"years": {...}, "discarded": [...], "call_failed": bool}`. Every
    submitted year is present even when it yielded nothing, carrying the gated
    provenance of each of its sections — that is what lets a sub-metric say
    *why* it is indeterminate rather than only that it is.

    **`call_failed` separates a fourth outcome that otherwise hides inside the
    third.** `_call_api` turns any network, rate-limit or auth failure into
    `{"error": ...}` rather than raising, and an error response reduced to the
    same empty result as "the model read the section and found nothing". They
    are not the same thing: one is a finding, the other is an outage. Folded
    together, a single transient failure would be written to the sidecar and
    served as a confirmed-empty extraction on every later run — permanently, as
    nothing re-extracts until the source text, schema, prompt or model changes.
    """
    discarded: list[dict] = []

    years: dict[str, dict] = {}
    for year, sections in submission.items():
        years[year] = {
            "sections": dict(gated.get(year, {})),
            **{kind: [] for kind in schema.ENTRY_KINDS},
        }

    if isinstance(raw, dict) and raw.get("error"):
        logger.error(
            f"Forward-growth extraction call failed: {raw['error']} — not caching "
            f"this result, so the next run retries rather than reading an outage "
            f"back as a finding"
        )
        return {"years": years, "discarded": discarded, "call_failed": True}

    payload = raw.get("years") if isinstance(raw, dict) else None
    if not isinstance(payload, dict):
        if raw is not None:
            logger.warning(
                "Forward-growth response carried no 'years' mapping — "
                "treating the extraction as empty"
            )
        return {"years": years, "discarded": discarded, "call_failed": False}

    grounded = {
        year: {name: ground_text(text) for name, text in sections.items()}
        for year, sections in submission.items()
    }

    for year, kinds in payload.items():
        year = str(year)
        if year not in years:
            discarded.append({
                "where": year,
                "reason": f"report year {year} was never submitted",
            })
            logger.warning(f"Forward-growth: response named unsubmitted year {year}")
            continue
        if not isinstance(kinds, dict):
            discarded.append({
                "where": year,
                "reason": f"year payload is {type(kinds).__name__}, expected an object",
            })
            continue

        for kind, entries in kinds.items():
            if kind not in schema.ENTRY_KINDS:
                discarded.append({
                    "where": year,
                    "reason": f"entry kind {kind!r} is outside the declared set "
                              f"({', '.join(schema.ENTRY_KINDS)})",
                })
                logger.warning(f"Forward-growth: unknown entry kind {kind!r} in {year}")
                continue
            if not isinstance(entries, list):
                discarded.append({
                    "where": f"{year}.{kind}",
                    "reason": f"expected a list, got {type(entries).__name__}",
                })
                continue

            for entry in entries:
                validated = _validate_entry(
                    kind, entry, year, submission[year], discarded,
                    grounded_sections=grounded[year],
                )
                if validated is not None:
                    years[year][kind].append(validated)

    kept = sum(len(years[y][k]) for y in years for k in schema.ENTRY_KINDS)
    logger.info(
        f"Forward-growth extraction: {kept} entries kept, {len(discarded)} discarded"
    )
    return {"years": years, "discarded": discarded, "call_failed": False}


# ── Sidecar cache ──────────────────────────────────────────────────────────


def _source_digest(submission: dict) -> dict:
    """Per-year fingerprint of the exact text submitted.

    Digesting the submitted payload rather than the raw section text means the
    char budget and the gate's own verdict are covered too: raise the budget or
    tighten the gate and the cache invalidates, because a different question
    was asked.
    """
    digests = {}
    for year, sections in submission.items():
        canonical = json.dumps(sections, sort_keys=True)
        digests[year] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return digests


def _version_block(submission: dict, model: str) -> dict:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "prompt_digest": prompt_digest(),
        "model": model,
        "source_digest": _source_digest(submission),
    }


def read_sidecar(path, submission: dict, model: str) -> dict | None:
    """Cached extraction for exactly this question, or None.

    An annual report does not change after filing, so without this the
    corpus-wide validation sweep and every re-analysis pay again for identical
    text. But "identical" has to mean the whole question — source text, field
    schema, prompt, and model id — or a prompt change would silently reuse the
    old prompt's answers.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Ignoring unreadable forward-growth sidecar {target.name}: {e}")
        return None

    if not isinstance(stored, dict) or stored.get("version") != _version_block(
        submission, model
    ):
        logger.info(
            f"Forward-growth sidecar {target.name} is stale — re-extraction required"
        )
        return None
    years = stored.get("years")
    return years if isinstance(years, dict) else None


def read_sidecar_discards(path) -> list:
    """Why entries were dropped last time this ticker was extracted.

    Diagnostic only — never gates anything, and deliberately not part of the
    version block, so a reader can inspect it without a stale discard list
    invalidating an otherwise-current cache.
    """
    target = Path(path)
    if not target.exists():
        return []
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    discarded = stored.get("discarded") if isinstance(stored, dict) else None
    return discarded if isinstance(discarded, list) else []


def write_sidecar(
    path, years: dict, submission: dict, model: str, discarded: list | None = None
) -> None:
    """Cache one ticker's validated extraction, with why anything was dropped.

    Written whole rather than merged: the version block describes exactly the
    question these `years` answer, so a partial update would leave a cache
    claiming to answer a question half of it never saw.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "version": _version_block(submission, model),
                    "years": years,
                    "discarded": discarded or [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning(f"Could not cache forward-growth extraction at {target}: {e}")
