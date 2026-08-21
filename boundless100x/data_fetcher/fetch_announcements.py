"""Corporate announcements from BSE — the events that postdate the accounts.

**The gap this closes.** Every other source in this pipeline is periodic. The
annual report lands once a year and the retained corpus reaches three years
back; Screener's tables are annual and quarterly; the price series says a
number moved without saying why. So the pipeline's picture of a company ends
at its last filed statement, and a company is not the sum of its filed
statements — it is that plus everything that has happened since.

EDELWEISS is the case that made this concrete. The FY2025 annual report is the
newest document the pipeline held, and the entire investment question about the
company had moved on: its alternatives subsidiary filed a DRHP in January 2026,
received SEBI's observation letter that April, and had a 4.4% stake placed
privately at a valuation implying most of the parent's market capitalisation.
The model produced a careful, well-reasoned thesis that said to wait until the
value crystallised — not knowing that the crystallisation had a regulator's
clearance and a price on it. None of that is inferable from a ratio. All of it
was filed with the exchange, in a feed nothing read.

**Materiality is the whole problem.** A year of one company's filings is a
hundred-odd rows, and the overwhelming majority are compliance chaff —
certificates under Reg. 74(5), newspaper publication intimations, trading
window closures. Handing that to a model unfiltered would bury the two rows
that matter and spend the context budget doing it. So this module's real work
is the classifier below, and its bias is stated: `_NOISE` must only ever match
things that are *definitionally* routine, because a false negative here costs a
line of context and a false positive costs the pipeline the one filing that
changes the answer.

The feed is **evidence, not instruction**. A subject line is text a company
wrote about itself; it says a filing exists and what the filer called it. It is
summarised for a reader and a model to weigh, never parsed into a fact the
score depends on.
"""

import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from boundless100x.data_fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

BSE_ANNOUNCEMENTS_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
)

# The API pages at 50 rows and will happily walk back for ever. A year is the
# window in which "what has happened since the last annual report" is actually
# answerable, and the page cap is a backstop against a filer with unusual
# volume rather than an expected limit.
DEFAULT_LOOKBACK_DAYS = 365
MAX_PAGES = 6

# ── The classifier ───────────────────────────────────────────────────────
#
# Ordered noise-first: a filing that matches `_NOISE` is dropped even if it
# also contains a material word, because the compliance templates routinely
# name SEBI regulations and would otherwise all read as material.
#
# Every entry here is a filing whose *category* is routine, never one whose
# subject merely looks dull. "Certificate under Reg. 74(5)" is a depository
# housekeeping note in every instance; "Disclosure under Reg. 30" is the
# regulation companies announce acquisitions under, and is deliberately absent.
_NOISE = re.compile(
    r"certificate under reg|reg\.?\s*74|reg\.?\s*7\(3\)|newspaper (publication|advertisement)"
    r"|trading window|closure of trading|compliances?-\s*(certificate|reg)"
    r"|half yearly report|share certificate|loss of (share )?certificate"
    r"|duplicate (share )?certificate|investor complaint|grievance redress"
    r"|reconciliation of share capital|record date.*dividend payment"
    r"|intimation of.*newspaper|corporate governance report"
    r"|related party transaction.*half|statement of investor"
    # Diary entries rather than news. "Board Meeting Intimation for
    # Consideration of the Results" says a meeting is scheduled; the outcome
    # filed days later says what was decided, and only the second is worth a
    # line of a model's context. Left in, these were 8 of EDELWEISS's 38
    # material rows and pushed genuine filings down the list.
    r"|board meeting intimation|meet\s*-?\s*intimation|intimation for"
    # Routine monthly ESOP allotments. Dilution matters and is already
    # measured directly by `equity_dilution` off the share count; a filing per
    # tranche adds noise rather than a second reading of it.
    r"|allotment of esop",
    re.I,
)

# What a long-term owner would want to be told about, in rough order of how
# often it changes a thesis. The label travels with the row so a reader can see
# why a filing was kept.
_MATERIAL: tuple[tuple[str, re.Pattern], ...] = (
    ("listing_or_ipo", re.compile(
        r"\b(ipo|drhp|rhp|red herring|offer for sale|initial public offer"
        r"|observation letter|listing of)\b", re.I)),
    ("restructuring", re.compile(
        r"\b(demerger|de-merger|merger|amalgamation|scheme of arrangement"
        r"|spin[- ]?off|hive[- ]?off|slump sale|composite scheme)\b", re.I)),
    ("stake_change", re.compile(
        r"\b(divestment|disinvestment|diversification|stake|acquisition"
        r"|acquire[sd]?|subsidiary|joint venture|strategic partnership"
        r"|open offer)\b", re.I)),
    ("regulatory_action", re.compile(
        r"\b(penalt|show cause|adjudicat|enforcement|prohibit|restrain"
        r"|supervisory action|inspection report|search and seizure"
        r"|non[- ]compliance|suspension)\b", re.I)),
    ("audit_or_restatement", re.compile(
        r"\b(qualified opinion|auditor.{0,20}(resign|qualif)|restat"
        r"|adverse opinion|emphasis of matter|whistle ?blower|fraud)\b", re.I)),
    ("leadership", re.compile(
        r"\b(resignation|cessation|appointment of|change in (management|director"
        r"|key managerial)|managing director|chief executive|chief financial"
        r"|company secretary)\b", re.I)),
    ("capital_raise", re.compile(
        r"\b(fund rais|preferential (issue|allotment)|qip|qualified institution"
        r"|rights issue|allotment of|debenture|ncd|buyback|bonus issue"
        r"|stock split|sub-division)\b", re.I)),
    ("credit_rating", re.compile(r"\b(credit rating|rating (action|revision)|downgrade|upgrade)\b", re.I)),
    ("results_or_guidance", re.compile(
        r"\b(financial results|earnings call|investor (presentation|meet)"
        r"|analyst meet|transcript|outcome of board meeting)\b", re.I)),
    ("operations", re.compile(
        r"\b(commissioning|capacity expansion|new plant|order win|bags order"
        r"|letter of intent|contract award|plant shutdown|fire at|force majeure)\b", re.I)),
)


def classify_announcement(subject: str, headline: str = "") -> str | None:
    """The material category a filing falls in, or None if it is routine.

    Noise is tested first and wins: compliance templates quote the regulations
    that the material categories also match.
    """
    text = f"{subject or ''} {headline or ''}".strip()
    if not text:
        return None
    if _NOISE.search(text):
        return None
    for label, pattern in _MATERIAL:
        if pattern.search(text):
            return label
    return None


# How much a category tends to change a long-term thesis. Used to order the
# rendered list, because a context budget truncates from the end and the
# feed's own order is chronological — under which a quarter's routine results
# filings can push a demerger below the cut.
_CATEGORY_RANK = {
    "restructuring": 0,
    "listing_or_ipo": 1,
    "regulatory_action": 2,
    "audit_or_restatement": 3,
    "stake_change": 4,
    "credit_rating": 5,
    "capital_raise": 6,
    "leadership": 7,
    "operations": 8,
    "results_or_guidance": 9,
}

_CATEGORY_LABELS = {
    "restructuring": "Restructuring",
    "listing_or_ipo": "Listing / IPO",
    "regulatory_action": "Regulatory action",
    "audit_or_restatement": "Audit / restatement",
    "stake_change": "Stake / acquisition",
    "credit_rating": "Credit rating",
    "capital_raise": "Capital raise",
    "leadership": "Leadership change",
    "operations": "Operations",
    "results_or_guidance": "Results / guidance",
}


def build_announcements_context(
    announcements, limit: int = 25, as_of: date | None = None
) -> str:
    """Render the filings for an LLM prompt, most thesis-relevant first.

    Takes the DataFrame `AnnouncementsFetcher.fetch` returns. The empty case
    says plainly that nothing was read rather than returning an empty string,
    because a model given no section cannot tell "no filings" from "this
    pipeline does not look at filings" — and on that distinction rests whether
    it should hedge about recent events or state that there were none.
    """
    if announcements is None or getattr(announcements, "empty", True):
        return (
            "No corporate announcements were retrieved for this company. Treat "
            "recent corporate events as UNKNOWN rather than absent — do not "
            "state that nothing has happened since the last annual report."
        )

    rows = announcements.copy()
    rows["_rank"] = rows["category"].map(lambda c: _CATEGORY_RANK.get(c, 99))
    rows = rows.sort_values(["_rank", "date"], ascending=[True, False])

    shown = rows.head(limit)
    lines = [
        f"Corporate filings with BSE, most thesis-relevant first "
        f"({len(rows)} material filings found"
        + (f", as at {as_of.isoformat()}" if as_of else "")
        + "):"
    ]
    for _, row in shown.iterrows():
        label = _CATEGORY_LABELS.get(row["category"], row["category"])
        subject = str(row["subject"])[:180]
        headline = str(row.get("headline") or "").strip()
        detail = f" — {headline[:120]}" if headline and headline not in subject else ""
        lines.append(f"- {row['date']} [{label}] {subject}{detail}")

    if len(rows) > limit:
        lines.append(f"- ... and {len(rows) - limit} further material filings")

    lines.append(
        "\nThese are filing SUBJECT LINES, not verified facts: they say a "
        "document exists and what the company called it. They are the only "
        "source here that postdates the annual report, so a filing that "
        "changes the investment case outranks a ratio computed from older "
        "accounts — but treat the substance as unread and say what would "
        "confirm it."
    )
    return "\n".join(lines)


class AnnouncementsFetcher(BaseFetcher):
    """Recent BSE corporate filings for one scrip, filtered to the material ones."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session.headers.update(
            {
                "Referer": "https://www.bseindia.com/",
                "Origin": "https://www.bseindia.com",
                "Accept": "application/json, text/plain, */*",
            }
        )

    def fetch(
        self,
        bse_code: str,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        output_dir: str | None = None,
        today: date | None = None,
    ) -> pd.DataFrame:
        """Material announcements for the last `lookback_days`.

        Returns a DataFrame with columns `date, category, subject, headline,
        bse_category, url`, newest first. Empty on any failure — this is
        supplementary context and must never be able to stop an analysis.

        `today` is injectable so a test does not depend on the wall clock, and
        so a future point-in-time replay can ask what was filed as at a past
        date rather than as at now.
        """
        as_of = today or date.today()
        start = as_of - timedelta(days=lookback_days)
        cache_key = f"announcements_raw_{bse_code}_{lookback_days}_{as_of.isoformat()}"

        # **The RAW feed is cached, and the classifier runs on every read.**
        # Caching the filtered frame instead would make `_NOISE` and
        # `_MATERIAL` effectively immutable for the life of a cache entry: a
        # tightened pattern would change nothing until someone paid for a
        # refetch, and the filings it newly kept or dropped would differ
        # between two tickers purely by when each was last fetched. Same
        # reasoning as the Screener page cache — hold the source, parse
        # deterministically on read.
        # Wrapped in a dict because the cache stores DataFrames, dicts and
        # strings — a bare list has no format. The wrapper also leaves room to
        # record what window was fetched alongside the rows.
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict) and "rows" in cached:
            logger.info(f"Cache hit: {cache_key}")
            rows = cached["rows"]
        else:
            rows = self._fetch_pages(bse_code, start, as_of)
            self.cache.set(cache_key, {
                "rows": rows,
                "from": start.isoformat(),
                "to": as_of.isoformat(),
            })

        df = self._to_frame(rows)

        if output_dir and not df.empty:
            path = Path(output_dir) / str(bse_code)
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "announcements.csv", index=False)
            logger.info(f"Saved {bse_code}/announcements.csv ({len(df)} rows)")

        return df

    def _fetch_pages(self, bse_code: str, start: date, end: date) -> list[dict]:
        """Walk the paged feed. A page that fails ends the walk, keeping what came before."""
        collected: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            try:
                resp = self._get(
                    BSE_ANNOUNCEMENTS_URL,
                    params={
                        "pageno": page,
                        "strCat": "-1",
                        "strPrevDate": start.strftime("%Y%m%d"),
                        "strToDate": end.strftime("%Y%m%d"),
                        "strScrip": str(bse_code),
                        "strSearch": "P",
                        "strType": "C",
                        "subcategory": "-1",
                    },
                )
                payload = resp.json()
            except Exception as e:
                logger.warning(
                    f"BSE announcements page {page} failed for {bse_code}: {e}"
                )
                break

            table = (payload or {}).get("Table") or []
            if not table:
                break
            collected.extend(table)
            if len(table) < 50:
                # A short page is the last page; asking for the next one is a
                # request the feed has already answered.
                break

        return collected

    @staticmethod
    def _to_frame(rows: list[dict]) -> pd.DataFrame:
        records = []
        for row in rows:
            subject = (row.get("NEWSSUB") or "").strip()
            headline = (row.get("HEADLINE") or "").strip()
            category = classify_announcement(subject, headline)
            if category is None:
                continue

            stamp = (row.get("NEWS_DT") or row.get("DT_TM") or "")[:10]
            records.append({
                "date": stamp,
                "category": category,
                "subject": subject,
                "headline": headline,
                "bse_category": (row.get("CATEGORYNAME") or "").strip(),
                "url": (row.get("NSURL") or "").strip(),
            })

        if not records:
            return pd.DataFrame(
                columns=["date", "category", "subject", "headline",
                         "bse_category", "url"]
            )

        df = pd.DataFrame(records)
        # Deduplicate: a filing revised the same day appears twice with the
        # same subject, and both rows would reach the prompt.
        df = df.drop_duplicates(subset=["date", "subject"])
        return df.sort_values("date", ascending=False).reset_index(drop=True)
