"""The lifecycle surface: the watchlist, its queue, and what they render.

Everything an owner touches *after* the verdict — tracking a company, changing
its lane, recording a catalyst, advancing the lifecycle, confirming an exit,
and routing the proceeds — plus the display helpers those commands share.

It is a module of its own because `cli.py` had grown past two thousand lines
with four command groups and a dozen display helpers interleaved among the
corpus, screen, sweep and backtest commands, and the drift had already started:
`_print_routing_result` had settled above the `# ── Display Helpers ──` banner
every other `_print_*` helper lived under. The split follows the one seam that
actually exists in that file — pipeline commands on one side, the lifecycle
layer on the other — and `cli.py` registers this group with `add_typer` exactly
as it already did for `corpus`.

`cli.py` re-exports the helpers here that it and the test suite already named,
so the split moves no caller. The three at the top are the lifecycle's
integration with `analyze`: they run after a full analysis, and they live here
rather than beside `analyze` because what they know about is the lifecycle.
"""

import logging
from pathlib import Path

import typer
from rich.table import Table

from boundless100x.cli_common import console, setup_logging

logger = logging.getLogger(__name__)

def _record_checkpoints_if_tracked(ticker: str, result, as_of=None) -> None:
    """Persist Pass 2's structured monitorables for a watchlisted company.

    `as_of` is the run's own date where a caller has one. It reaches the
    recorder's past-dating check, which would otherwise read the wall clock
    while the rest of the run read a supplied date — see `record_checkpoints`.
    """
    from boundless100x.lifecycle.advance import record_checkpoints
    from boundless100x.watchlist import WatchlistManager

    try:
        wm = WatchlistManager()
        if wm.get(ticker) is None:
            return
        recorded = record_checkpoints(wm, ticker, result, as_of=as_of)
        if recorded["checkpoints"]:
            console.print(
                f"[dim]Recorded {len(recorded['checkpoints'])} checkpoint(s) "
                f"for {ticker.upper()}[/dim]"
            )
        if recorded["demoted"]:
            console.print(
                f"[yellow]{len(recorded['demoted'])} monitorable(s) kept as prose "
                f"only — not machine-checkable[/yellow]"
            )
    except Exception as e:
        # Never cost the caller the analysis they just paid for.
        logger.warning(f"Could not record checkpoints for {ticker}: {e}")


def _lane_context_if_tracked(ticker: str, result, service) -> dict | None:
    """The lane, gates and modeled friction for a watchlisted company, or None.

    The same gate as `_record_checkpoints_if_tracked`, deliberately: "is this
    ticker tracked" is one question and it should be asked the same way twice.
    A company that is not on the watchlist has no lane to report, and its
    report renders exactly as it did before this section existed.

    A failure never costs the caller the analysis they just paid for — the
    report is written either way, one section shorter.
    """
    from datetime import date

    from boundless100x.lifecycle.lane_view import build_lane_context
    from boundless100x.watchlist import WatchlistManager

    try:
        entry = WatchlistManager().get(ticker)
        if entry is None:
            return None
        return build_lane_context(
            entry, result, date.today(), config=getattr(service, "config", None)
        )
    except Exception as e:
        logger.warning(f"Could not build the lane context for {ticker}: {e}")
        return None


def _print_lane_status(context: dict | None) -> None:
    """Lane, state and the modeled friction line, for a tracked company.

    Short by design — the full section is in the report this command has just
    written. What earns a terminal line is what an owner would otherwise have
    to open a file to learn: which lane the company is being judged in, whether
    a catalyst window has passed, and, for a position, what it is modeled to
    keep after tax and slippage.

    Gross and net travel together because `friction.describe` renders them
    together, and the line names itself modeled every time (R5, KTD7). An
    unavailable reading prints its reason rather than nothing, for the reason
    every gap in this system is printed: silence and zero look identical.
    """
    from rich.markup import escape

    from boundless100x.lifecycle import friction

    if not context:
        return

    console.print(
        f"\n[bold]Lane:[/bold] {escape(str(context.get('lane')))} "
        f"[dim](lifecycle state: {escape(str(context.get('state')))})[/dim]"
    )

    catalyst = context.get("catalyst") or {}
    if catalyst.get("overdue"):
        # Advisory, and said so in the line itself: §13 keeps the clock feeding
        # the time stop and nothing else, so an owner must not read this as a
        # transition that has been taken on their behalf.
        console.print(
            f"[yellow]Catalyst overdue: "
            f"{escape(str(catalyst.get('description', '')))} — expected by "
            f"{escape(str(catalyst.get('expected_by', '')))}. Advisory only; "
            f"no transition was proposed or taken.[/yellow]"
        )

    reading = context.get("friction")
    if reading:
        colour = "yellow" if reading.get("available") else "dim"
        console.print(
            f"  [{colour}]{escape(friction.describe(reading))}[/{colour}]"
        )


# ── Watchlist Commands ──

watchlist_app = typer.Typer(help="Manage your company watchlist")


# Colour carries the lifecycle's own meaning: green where capital is
# committed, red where a thesis is under review, dim before anything is at
# stake.
STATE_COLOURS = {
    "screen": "dim",
    "qualify": "cyan",
    "watch": "yellow",
    "probe": "green",
    "scale": "green bold",
    "exit_review": "red bold",
    "exited": "red",
    "dropped": "dim",
}


@watchlist_app.command("show")
def watchlist_show():
    """Show all companies in the watchlist.

    The catalyst has a column because the system acts on it — it gates
    fast-lane entry and fires an exit rule — and nothing here could read it
    back before. An owner could record one and then have no way to see which
    companies held one, or whose window had passed, short of running a full
    `advance` or opening the JSON.
    """
    from rich.markup import escape

    from boundless100x.watchlist import WatchlistManager

    wm = WatchlistManager()
    entries = wm.list()

    if not entries:
        console.print("[dim]Watchlist is empty. Add companies with: watchlist add TICKER[/dim]")
        return

    table = Table(title="Watchlist")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Lane", style="dim")
    table.add_column("State", style="bold")
    table.add_column("Last Run", style="dim")
    table.add_column("Composite", justify="right")
    table.add_column("Checks", justify="right")
    table.add_column("Catalyst", max_width=34)
    table.add_column("Notes")

    for e in entries:
        last_run = e["last_run"][:10] if e["last_run"] else "never"
        composite = f"{e['last_composite']}/10" if e["last_composite"] else "—"
        table.add_row(
            e["ticker"],
            e["lane"],
            f"[{STATE_COLOURS.get(e['state'], 'white')}]{e['state']}[/]",
            last_run,
            composite,
            str(e["checkpoints"]) if e["checkpoints"] else "—",
            _catalyst_cell(e),
            escape(e.get("notes", "")),
        )

    console.print(table)


def _catalyst_cell(row: dict) -> str:
    """The catalyst, its window, and whether that window has passed.

    Three states worth telling apart, because they call for different things.
    An **active** catalyst whose window is still open is the ordinary case and
    stays quiet. One **overdue** is the fast lane's own exit rule about to
    fire, so it is red and says so. A **spent** one is history the owner
    deliberately kept — `mark_catalyst_spent` records rather than deletes,
    because a position whose catalyst was spent without the re-rating following
    is exactly the case worth being able to see.

    An unreadable `expected_by` renders as unknown rather than as a date
    comfortably in the future, which is the direction that would matter.
    """
    from rich.markup import escape

    description = row.get("catalyst")
    if not description:
        return "[dim]—[/dim]"

    window = row.get("catalyst_expected_by") or "no date"
    overdue = row.get("catalyst_overdue")
    if row.get("catalyst_status") == "spent":
        tone, suffix = "dim", " (spent)"
    elif overdue is True:
        tone, suffix = "red", f" (window passed {window})"
    elif overdue is None:
        tone, suffix = "yellow", f" (window unreadable: {window})"
    else:
        tone, suffix = "", f" (by {window})"

    text = f"{escape(description)}{escape(suffix)}"
    return f"[{tone}]{text}[/{tone}]" if tone else text


@watchlist_app.command("add")
def watchlist_add(
    ticker: str = typer.Argument(help="NSE symbol to add"),
    notes: str = typer.Option("", help="Optional notes"),
    lane: str = typer.Option(
        "core", help="Lane to track in: core (compounder) or rerating (fast lane)"
    ),
):
    """Add a company to the watchlist."""
    from boundless100x.watchlist import LANES, WatchlistManager

    ticker, lane = ticker.upper(), lane.lower()
    if lane not in LANES:
        raise typer.BadParameter(f"unknown lane {lane!r} — one of: {', '.join(LANES)}")

    wm = WatchlistManager()
    if wm.add(ticker, notes=notes, lane=lane):
        console.print(f"[green]Added {ticker} to the {lane} lane[/green]")
    else:
        console.print(f"[yellow]{ticker} is already in the watchlist[/yellow]")


@watchlist_app.command("catalyst")
def watchlist_catalyst(
    ticker: str = typer.Argument(help="NSE symbol"),
    description: str = typer.Option(
        None, "--description", help="What the re-rating is waiting on"
    ),
    expected_by: str = typer.Option(
        None, "--expected-by", help="When it is expected (e.g. FY2027 Q2)"
    ),
    spent: bool = typer.Option(
        False, "--spent", help="Mark the recorded catalyst as having happened"
    ),
):
    """Record the catalyst a company is waiting on, or mark it spent.

    Owner judgement, never computed — no metric knows that a demerger is
    filed. The two modes are kept apart on purpose: a flip that could also
    rewrite the description would quietly change *which* catalyst it says was
    spent.
    """
    from boundless100x.watchlist import WatchlistError, WatchlistManager

    ticker = ticker.upper()
    if spent and (description or expected_by):
        raise typer.BadParameter(
            "--spent marks the recorded catalyst spent and takes nothing else. "
            "Record a replacement in a separate call."
        )
    if not spent:
        missing = [
            name for name, value in
            (("--description", description), ("--expected-by", expected_by))
            if not value
        ]
        if len(missing) == len(("--description", "--expected-by")):
            raise typer.BadParameter(
                "give --description and --expected-by to record a catalyst, "
                "or --spent to mark one spent"
            )
        if missing:
            raise typer.BadParameter(f"{missing[0]} is required to record a catalyst")

    wm = WatchlistManager()
    try:
        if spent:
            catalyst = wm.mark_catalyst_spent(ticker)
            console.print(
                f"[yellow]{ticker}: catalyst spent — {catalyst['description']}[/yellow]"
            )
        else:
            catalyst = wm.record_catalyst(ticker, description, expected_by)
            console.print(
                f"[green]{ticker}: catalyst recorded — {catalyst['description']} "
                f"(expected by {catalyst['expected_by']})[/green]"
            )
    except WatchlistError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@watchlist_app.command("lane")
def watchlist_lane(
    ticker: str = typer.Argument(help="NSE symbol whose lane is changing"),
    lane: str = typer.Argument(help="Lane to move it to: core or rerating"),
    reason: str = typer.Option(
        "", "--reason", help="Why the lane changed. Recorded with the change."
    ),
):
    """Move a tracked company to the other lane, keeping its history.

    The lane could only be set at `add`, and `add` refuses a ticker already
    tracked — so the only way to change one was `remove` then re-`add`, which
    throws away the append-only `state_history` and every piece of evidence
    behind the states the company had earned.

    The state does not move. A lane says how a company is judged, not how far
    along it is; the next `advance` evaluates it under the new lane's rules,
    which is where the change is meant to show up.

    **A positioned company keeps its capital and changes its exit rules**, so
    the command says which ones. The six fundamentals kill-switches are
    universal and apply in both lanes by design (§6.2) — what changes is the
    lane-scoped exits, and an owner moving a live position deserves to read
    that in the same breath as the confirmation.
    """
    from rich.markup import escape

    from boundless100x.lifecycle import states as lifecycle_states
    from boundless100x.watchlist import WatchlistError, WatchlistManager

    wm = WatchlistManager()
    symbol = ticker.upper()
    entry = wm.get(symbol)
    if entry is None:
        console.print(f"[red]{escape(symbol)} is not on the watchlist[/red]")
        raise typer.Exit(1)

    try:
        record = wm.set_lane(symbol, lane, reason=reason)
    except WatchlistError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]{escape(symbol)}: {record['from']} → {record['to']} lane[/green] "
        f"[dim](state unchanged: {record['state']})[/dim]"
    )
    if record["state"] in lifecycle_states.POSITIONED:
        console.print(
            f"[yellow]{escape(symbol)} holds a position. The lane-scoped exit "
            f"rules that apply to it have changed; the six fundamentals "
            f"kill-switches are universal and are unaffected. Run `watchlist "
            f"advance` to see it evaluated under the {record['to']} lane."
            f"[/yellow]"
        )


@watchlist_app.command("remove")
def watchlist_remove(
    ticker: str = typer.Argument(help="NSE symbol to remove"),
):
    """Remove a company from the watchlist.

    **Refused while the company holds an unconfirmed exit.** That is the one
    state whose repair genuinely needs the lifecycle record — `watchlist exit`
    keys on the entry's `exit_review` transition and completes from its history
    — so deleting the entry underneath it strands the proceeds with no command
    able to reach them, and the queue then reports nothing outstanding.
    Completing the exit first costs one command; the removal is then safe,
    because a confirmed exit carries its own completion stamp and no longer
    depends on the entry existing at all.

    An unreadable queue refuses too. It cannot show that the removal is safe,
    and "could not check" must not resolve to "go ahead" on the one path that
    cannot be undone.
    """
    from rich.markup import escape

    from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
    from boundless100x.watchlist import WatchlistManager

    symbol = ticker.upper()
    try:
        blocking = ReinvestmentQueue().unconfirmed_exits(symbol)
    except Exception as e:
        logger.error(f"The reinvestment queue could not be read: {e}")
        console.print(
            f"[red]The reinvestment queue could not be read ({escape(str(e))}), "
            f"so this removal cannot be shown not to strand recorded exit "
            f"proceeds.[/red]"
        )
        console.print("[dim]Nothing was removed.[/dim]")
        raise typer.Exit(1)

    if blocking:
        listed = ", ".join(event["exit_id"] for event in blocking)
        console.print(
            f"[red]{symbol} holds {len(blocking)} recorded exit(s) whose sale is "
            f"not yet confirmed ({escape(listed)}).[/red]"
        )
        console.print(
            f"[yellow]Run `watchlist exit {symbol}` to complete the record "
            f"first — removing the entry now would leave those proceeds with no "
            f"command able to route them, and `watchlist queue` would stop "
            f"reporting them as outstanding.[/yellow]"
        )
        console.print("[dim]Nothing was removed.[/dim]")
        raise typer.Exit(1)

    wm = WatchlistManager()
    if wm.remove(ticker):
        console.print(f"[green]Removed {ticker} from watchlist[/green]")
    else:
        console.print(f"[yellow]{ticker} not found in watchlist[/yellow]")


@watchlist_app.command("exit")
def watchlist_exit(
    ticker: str = typer.Argument(help="NSE symbol whose exit is being confirmed"),
    as_of: str = typer.Option(
        None, "--as-of", help="Date of the sale (YYYY-MM-DD). Defaults to today."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Record an owner-confirmed exit — the only path to `exited`.

    A command of its own rather than a flag on `advance`, because it moves
    money and no metric can observe that the owner sold. `advance` proposes
    `exit_review`; this is what closes it.

    The output states everything needed to reconcile the sale afterwards: the
    transition, its date, the trigger the review was recorded under, the
    friction reading (or why there is none), and the queue event's `exit_id` —
    the id a retry would recompute, and therefore the one to quote if anything
    about this run needs looking into.
    """
    setup_logging(verbose)

    from datetime import date as date_type

    from rich.markup import escape

    from boundless100x.lifecycle import friction
    from boundless100x.lifecycle.exit import confirm_exit
    from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
    from boundless100x.service import Boundless100xService
    from boundless100x.watchlist import WatchlistManager

    when = None
    if as_of:
        try:
            when = date_type.fromisoformat(as_of)
        except ValueError:
            raise typer.BadParameter(f"--as-of {as_of!r} is not a YYYY-MM-DD date")

    try:
        outcome = confirm_exit(
            WatchlistManager(), ReinvestmentQueue(), ticker,
            Boundless100xService(), when,
        )
    except Exception as e:
        # `confirm_exit` lets an exception escape its third step deliberately:
        # the queue event is already durable at that point, so the situation is
        # exactly the recoverable window and re-running the command completes
        # it. That argument only helps somebody who is told to re-run — and a
        # traceback at the moment two stores disagree is the worst possible
        # thing to hand them. So the operation keeps raising and the surface
        # says what to do about it.
        symbol = escape(ticker.upper())
        logger.error(f"{ticker.upper()}: the exit could not be completed: {e}")
        console.print(
            f"\n[red]{symbol}: the exit could not be completed — "
            f"{escape(str(e))}[/red]"
        )
        console.print(
            f"[yellow]Re-run `watchlist exit {symbol}` — it recomputes "
            f"the same exit id, adopts the queued date and figures rather than "
            f"re-pricing the sale, and completes the transition. "
            f"`watchlist queue` shows whether the queue event landed.[/yellow]"
        )
        raise typer.Exit(1)

    if not outcome["ok"]:
        # The refusal already says which state the entry is in and that nothing
        # was written; after a command that touches two stores, that second half
        # is the part the owner needs.
        console.print(f"[red]{escape(outcome['reason'])}[/red]")
        raise typer.Exit(1)

    # Three things this command can have just done, and they are worth telling
    # apart: recorded a sale, completed a transition an earlier crash left
    # queued, or stamped a record whose transition was already there. All three
    # leave the same state; only the last two mean a previous run was
    # interrupted.
    if outcome["stamp_only"]:
        console.print(
            f"\n[bold green]{outcome['ticker']}: exit record completed[/bold green] "
            f"for the sale of {outcome['exit_date']}"
        )
        console.print(
            "[dim]The transition was already recorded; this run added only the "
            "completion stamp its proceeds needed to be routable.[/dim]"
        )
    else:
        verb = "reconciled" if outcome["adopted"] else "recorded"
        console.print(
            f"\n[bold green]{outcome['ticker']}: exit_review → exited[/bold green] "
            f"on {outcome['exit_date']} ({verb})"
        )
        if outcome["adopted"]:
            console.print(
                "[dim]An earlier run had already queued this exit; its date and "
                "figures were adopted rather than re-priced.[/dim]"
            )
    console.print(f"  trigger: {escape(outcome['trigger_id'] or '—')}")
    reading = outcome["friction"]
    colour = "yellow" if reading.get("available") else "dim"
    console.print(f"  [{colour}]{escape(friction.describe(reading))}[/{colour}]")
    console.print(f"  [dim]queue event: {escape(outcome['exit_id'])}[/dim]")
    console.print(
        "[dim]Holding period is measured from the `probe` confirmation date, "
        "not a broker fill.[/dim]"
    )


# ── Reinvestment queue ──
#
# `queue` renders; `queue route` records. They are one group because they read
# the same two stores, and separate commands because only one of them writes.

queue_app = typer.Typer(
    help="Where exit proceeds should go, and where they went",
    invoke_without_command=True,
    no_args_is_help=False,
)
watchlist_app.add_typer(queue_app, name="queue")


@queue_app.callback(invoke_without_command=True)
def watchlist_queue(ctx: typer.Context):
    """Show the stored routing snapshot and the exit event log.

    **A pure read.** It never calls `advance()` and never builds a service: a
    display command must not re-score the corpus or mutate lifecycle state as a
    side effect of being looked at.

    The snapshot is labelled with one of four states, resolved in precedence
    order, and **only `Current` renders the proposal** — see
    `reinvestment.snapshot_state`. The exits below it are read live rather than
    from the snapshot, because an idle reading grows every day and the stored
    one stopped growing when the run ended.
    """
    if ctx.invoked_subcommand is not None:
        return

    from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
    from boundless100x.watchlist import WatchlistManager

    _print_routing_snapshot(ReinvestmentQueue(), WatchlistManager())


@queue_app.command("route")
def watchlist_queue_route(
    exit_id: str = typer.Argument(help="The exit event whose proceeds were deployed"),
    candidate: str = typer.Argument(help="NSE symbol the proceeds went into"),
    transition_at: str = typer.Option(
        None, "--transition-at",
        help="Which deployment transition moved this capital (its exact `at` "
             "timestamp). Required only when the candidate holds more than one.",
    ),
    allow_shared_deployment: bool = typer.Option(
        False, "--allow-shared-deployment",
        help="Route into a deployment that already closed another exit. Only "
             "when one purchase genuinely absorbed the proceeds of both sales.",
    ),
):
    """Record that an exit's proceeds were deployed into a company.

    **A deployment, not an intention.** The candidate must already hold an
    owner-applied `probe`/`scale` transition dated on or after the exit: the
    idle reading measures exit-to-deployed-capital, and a plan that never
    executed must not close it. The event stores `deployed_at` from that
    transition and `recorded_at` from this command, so entering a route late
    does not inflate the window it closes.

    The candidate need not be the one the snapshot proposed. The proposal
    advises; this records what actually happened.

    Validation runs in a fixed order and every refusal names its cause. The
    first check is whether any routable proceeds exist at all — with none, that
    is the answer, and judging the arguments first would report a smaller
    problem than the one in front of the owner.
    """
    from rich.markup import escape

    from boundless100x.lifecycle.reinvestment import (
        ReinvestmentQueue,
        eligible_deployments,
        exit_is_complete,
        unroutable_reason,
    )
    from boundless100x.watchlist import WatchlistManager

    wm, queue = WatchlistManager(), ReinvestmentQueue()
    candidate = candidate.upper()

    def refuse(message: str) -> None:
        console.print(f"[red]{escape(message)}[/red]")
        console.print("[dim]Nothing was recorded.[/dim]")
        raise typer.Exit(1)

    # 1. Proceeds first, before any argument is judged — and the reason
    #    distinguishes an empty queue from one holding exits only the owner can
    #    complete, because those two call for opposite next steps.
    routable, incomplete = queue.unrouted_views(wm)
    if not routable:
        console.print(f"[yellow]{escape(unroutable_reason(incomplete))}[/yellow]")
        console.print(
            "[dim]A routing event deploys the proceeds of a completed exit.[/dim]"
        )
        raise typer.Exit(1)

    # 2. An exit that exists, and is not already closed.
    event = queue.find_exit(exit_id)
    if event is None:
        refuse(
            f"no exit {exit_id} is recorded — a routing event must reference the "
            f"exit whose proceeds it deploys"
        )
    routed = queue.routing_for(exit_id)
    if routed is not None:
        refuse(
            f"exit {exit_id} was already routed into {routed.get('candidate')} "
            f"on {routed.get('deployed_at')} — the log is append-only and an "
            f"exit's proceeds are deployed once"
        )

    # 3. An exit the watchlist agrees completed. KTD10's crash window is not
    #    routable proceeds, and the display already excludes it — enforced here
    #    so the direct command cannot be a way round that exclusion.
    sold = event["ticker"]
    entry = wm.get(sold)
    if not exit_is_complete(entry=entry, event=event,
                            confirmation=queue.find_confirmation(exit_id)):
        state = entry["state"] if entry else "no watchlist entry"
        refuse(
            f"{sold} is in {state!r} and this exit carries no completion stamp, "
            f"so the sale is only half recorded — run `watchlist exit {sold}` "
            f"to complete it before routing its proceeds"
        )

    # 4. A candidate that actually received capital after the exit.
    candidate_entry = wm.get(candidate)
    if candidate_entry is None:
        refuse(f"{candidate} is not on the watchlist")

    eligible = eligible_deployments(candidate_entry, event["at"])
    if not eligible:
        refuse(
            f"{candidate} holds no owner-applied probe or scale transition dated "
            f"on or after {event['at']} — the idle reading measures "
            f"exit-to-deployed-capital, and a plan that never executed cannot "
            f"close it"
        )

    # 4b. A deployment that has not already closed a different exit. Nothing
    #     recorded that a transition had been consumed, so routing two exits
    #     into the same `probe` both succeeded — and the second exit's idle
    #     reading then closed on a purchase that had nothing to do with it,
    #     which is the one figure this whole store exists to measure. Refused
    #     rather than prevented outright, because a single deployment absorbing
    #     two sales is genuinely possible in a system that counts names and has
    #     never counted rupees.
    consumed = queue.deployments_consumed_by(candidate)
    if not allow_shared_deployment:
        free = [record for record in eligible if record.get("at") not in consumed]
        if not free:
            listed = ", ".join(
                f"{record['at']} (already closed {consumed[record['at']]})"
                for record in eligible if record.get("at") in consumed
            )
            refuse(
                f"every eligible deployment {candidate} holds has already closed "
                f"another exit: {listed}. Routing a second exit into it would "
                f"close this one's idle reading on a purchase that did not fund "
                f"it. If one deployment really did absorb both sales, re-run "
                f"with --allow-shared-deployment"
            )
        eligible = free

    # 5. One recorded date, never a guess between two.
    if transition_at:
        chosen = [record for record in eligible if record.get("at") == transition_at]
        if not chosen:
            refuse(
                f"--transition-at {transition_at} matches no eligible deployment "
                f"for {candidate}. Eligible: "
                f"{', '.join(record['at'] for record in eligible)}"
            )
        eligible = chosen
    elif len(eligible) > 1:
        listed = ", ".join(
            f"{record['to']} at {record['at']}" for record in eligible
        )
        refuse(
            f"{candidate} holds {len(eligible)} eligible deployment transitions "
            f"({listed}). `deployed_at` is a recorded fact, so choose one with "
            f"--transition-at <timestamp> rather than letting it be guessed"
        )

    deployment = eligible[0]
    routing = queue.record_routing(
        exit_id=exit_id, candidate=candidate, deployed_at=deployment["at"]
    )

    view = next(
        (v for v in queue.exit_views(wm) if v["exit_id"] == exit_id), {}
    )
    idle = view.get("idle_days")
    console.print(
        f"\n[bold green]{exit_id} → {candidate}[/bold green] "
        f"(deployed {routing['deployed_at']}, {deployment['to']})"
    )
    console.print(
        f"  [dim]recorded {routing['recorded_at']}; idle "
        f"{'unknown' if idle is None else idle} day(s) between the sale and the "
        f"deployment[/dim]"
    )


@watchlist_app.command("advance")
def watchlist_advance(
    apply: bool = typer.Option(
        False, "--apply",
        help="Confirm and record transitions that move money. Without this, "
             "they are proposed only.",
    ),
    quarterly: bool = typer.Option(
        False, "--quarterly", help="Only advance stale (90+ days) entries"
    ),
    override_caps: bool = typer.Option(
        False, "--override-caps",
        help="Apply a position transition even when it would breach a per-lane "
             "or per-sector concentration cap. The breach is recorded in the "
             "transition's evidence.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Re-score the watchlist, evaluate triggers, and propose transitions.

    **A concentration cap is checked before the transition, not after it.** A
    proposal that would take the portfolio past a per-lane or per-sector cap is
    withheld even under `--apply`, and says which cap and by how much. The
    reading used to be counted only once the loop had finished, which meant an
    owner could learn a lane was over its cap only from the run that had
    already put it there. Every figure is a count of positioned names, never a
    share of capital — see `lifecycle/portfolio.py`.

    `--override-caps` proceeds anyway. It exists because a guardrail with no
    way past it can trap an owner out of their own decision, and it is explicit
    rather than silent: the breach is written into the append-only evidence
    beside the reason the transition fired.
    """
    setup_logging(verbose)

    from boundless100x.lifecycle.advance import advance
    from boundless100x.lifecycle.reinvestment import ReinvestmentQueue
    from boundless100x.service import Boundless100xService
    from boundless100x.watchlist import WatchlistManager

    svc = Boundless100xService()
    wm = WatchlistManager()

    # The production store, so the run's routing view is derived from the real
    # event log and written back to it. Without a queue the run still advances;
    # routing simply reports itself unavailable — see `advance._routing`.
    #
    # Built inside a guard rather than inline as an argument, because `_load`
    # raises on unreadable JSON or an invalid event and an exception there
    # would escape before `advance()` was ever entered — taking the designed
    # degradation path with it. A fault in the *routing* store would then stop
    # every tracked company from being re-scored, kill-switches included, which
    # trades the whole run for the one reading that is allowed to be missing.
    try:
        queue = ReinvestmentQueue()
    except Exception as e:
        from rich.markup import escape

        logger.error(f"The reinvestment queue could not be read: {e}")
        console.print(
            f"[yellow]Reinvestment routing unavailable: the queue could not be "
            f"read ({escape(str(e))}). The advance below is unaffected; fix or "
            f"remove the queue file to restore routing.[/yellow]\n"
        )
        queue = None

    result = advance(
        svc, wm, apply=apply, quarterly=quarterly, queue=queue,
        override_caps=override_caps,
    )
    outcomes, errors = result["outcomes"], result["errors"]

    # Say when the corpus's valuation tightened entry, before showing what did
    # and did not qualify — a proposal withheld by a tightened threshold would
    # otherwise look like a company that simply failed on its own merits.
    pace = result.get("pace") or {}
    if pace.get("applied"):
        console.print(f"[yellow]Deployment pace: {pace['evidence']}[/yellow]\n")
    elif pace.get("reason"):
        console.print(f"[dim]Deployment pace: {pace['reason']}[/dim]\n")

    _print_concentration(result.get("concentration"))

    if not outcomes and not errors:
        console.print("[dim]No companies to advance[/dim]")
        return

    table = Table(title="Lifecycle Advance")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("State", style="bold")
    table.add_column("Composite", justify="right")
    table.add_column("Proposal")
    table.add_column("Evidence", style="dim", max_width=54)

    for o in outcomes:
        proposal = o["proposal"]
        state = f"[{STATE_COLOURS.get(o['state'], 'white')}]{o['state']}[/]"
        composite = f"{o['composite']}/10" if o["composite"] is not None else "—"

        if not proposal:
            unknown = len(o["indeterminate"])
            note = f"[dim]no change ({unknown} unknown)[/dim]" if unknown else "[dim]no change[/dim]"
            table.add_row(o["ticker"], state, composite, note, "")
            continue

        arrow = f"→ [{STATE_COLOURS.get(proposal['to'], 'white')}]{proposal['to']}[/]"
        if proposal.get("concentration_withheld"):
            # Distinct from the ordinary "confirm with --apply", because
            # re-running with --apply will not move this one: the owner has a
            # decision to make about the cap, not a confirmation to give.
            arrow += " [red](cap breached — see below)[/red]"
        elif proposal["needs_confirmation"]:
            arrow += " [yellow](confirm with --apply)[/yellow]"
        elif proposal["applied"]:
            arrow += " [green]applied[/green]"
        table.add_row(
            o["ticker"], state, composite, arrow,
            _evidence_cell(proposal["evidence"]),
        )

    console.print(table)

    _print_exit_friction(outcomes)

    for ticker, message in errors:
        console.print(f"[red]{ticker}: {message}[/red]")

    _print_capped_transitions(outcomes)

    pending = [
        o for o in outcomes
        if o["proposal"] and o["proposal"]["needs_confirmation"]
        and not o["proposal"].get("concentration_withheld")
    ]
    if pending:
        console.print(
            f"\n[yellow]{len(pending)} transition(s) move money and were not "
            f"applied. Review the evidence, then re-run with --apply.[/yellow]"
        )

    _print_routing_result(result.get("routing"))


# ── Display Helpers ──
#
# Every `_print_*` and cell-formatting helper the commands above render
# through, and all of them below this line. The banner is the fix for the drift
# the split was prompted by: `_print_routing_result` had settled *above* the
# equivalent banner in `cli.py`, on the grounds that it was written with the
# command that calls it — which is how a section that means something becomes a
# section nobody can rely on.


def _print_capped_transitions(outcomes) -> None:
    """Which entries a concentration cap held back, and by how much.

    Its own block rather than a line in the table, for `_print_exit_friction`'s
    reason: the cap has to travel with the count it breaches and the basis that
    count is in, and an evidence cell truncated to 54 characters would show
    "the core lane already holds 8 of a maxi…" — which reads as a system that
    refused without saying why.

    It prints the escape hatch too. A guardrail whose only visible face is a
    refusal invites being worked around by editing the config, which is the
    version of the override that leaves no record.
    """
    from rich.markup import escape

    capped = [
        o for o in outcomes
        if o["proposal"] and o["proposal"].get("concentration_withheld")
    ]
    if not capped:
        return

    console.print(
        f"\n[bold red]{len(capped)} transition(s) withheld by a concentration "
        f"cap[/bold red]"
    )
    for outcome in capped:
        proposal = outcome["proposal"]
        console.print(
            f"  [cyan]{escape(str(outcome['ticker']))}[/cyan] "
            f"[dim]{outcome['state']} → {proposal['to']}[/dim]"
        )
        for reason in proposal.get("concentration_reasons") or []:
            console.print(f"    [red]- {escape(str(reason))}[/red]")
    console.print(
        "  [dim]Exit or drop a name to make room, raise the cap in "
        "config.yaml under `portfolio:`, or re-run with --override-caps to "
        "proceed and record the breach in the evidence.[/dim]"
    )


def _print_routing_result(routing) -> None:
    """One line on where this run says exit proceeds should go.

    Deliberately short, and printed last. The full view lives behind
    `watchlist queue`, which can be read without re-scoring anything; repeating
    it here would bury the transitions this command exists to propose.

    **A candidate is named only when the view was durably stored.** A
    `--quarterly` run deliberately does not write one — its ranking came from
    whichever companies happened to be stale — and a full run whose write
    failed has nothing behind the name either. In both cases the line says what
    happened instead, so that what the owner reads here and what
    `watchlist queue` will show them tomorrow cannot disagree.

    **And only when the run finished.** `_routing` stamps `status: partial` the
    moment any ticker's analysis fails, and `reinvestment.snapshot_state`
    refuses to render a partial snapshot's proposal — so naming one here would
    have this line and a `watchlist queue` run seconds later disagree about the
    *same stored bytes*, which is precisely what the paragraph above says
    cannot happen. The condition is read off the snapshot rather than
    hand-written a second time: `status` must be `current` and `errors` must be
    empty, exactly the two questions `snapshot_state` asks before it sets
    `renders_proposal`. A payload carrying no `status` at all cannot prove its
    run finished, so it fails closed with the rest.

    Resolving it through `snapshot_state` itself would be the tidier call, but
    the honest comparison it makes needs the *live* store revisions, and this
    call site holds neither store — feeding it the snapshot's own captured
    counters would compare a number against itself and report `current`
    unconditionally, which is worse than not asking.

    An unavailable view says so with its reason. A run that quietly printed
    nothing would be indistinguishable from one where the queue was empty, and
    those are different facts.
    """
    from rich.markup import escape

    from boundless100x.lifecycle import reinvestment

    if not routing:
        return
    if not routing.get("available"):
        console.print(
            f"\n[dim]Reinvestment routing unavailable: "
            f"{escape(str(routing.get('reason', '')))}[/dim]"
        )
        return
    if not routing.get("persisted"):
        console.print(
            f"\n[dim]Reinvestment: no candidate named — "
            f"{escape(str(routing.get('persist_reason', 'the view was not stored')))}"
            f"[/dim]"
        )
        return

    errored = [str(ticker) for ticker in routing.get("errors") or []]
    proposal = routing.get("proposal")
    if routing.get("status") != reinvestment.SNAPSHOT_CURRENT or errored:
        # Name the tickers that failed, because they are the thing to go and
        # fix — and say which command rebuilds the ranking once they do.
        detail = (
            f"this run could not evaluate {', '.join(errored)}"
            if errored else
            f"this run's view does not record a completed pass "
            f"(status: {routing.get('status')!r})"
        )
        # Two lines rather than one long one: a wrapped console splits mid-word,
        # and the words most worth not splitting are the command to run.
        console.print(
            f"\n[dim]Reinvestment: no candidate named — {escape(detail)}, so "
            f"the ranking was built on an incomplete field.[/dim]"
        )
        console.print(
            "  [dim]`watchlist queue` withholds this proposal too — re-run "
            "`watchlist advance` once every tracked company evaluates.[/dim]"
        )
    elif proposal:
        console.print(
            f"\n[bold]Reinvestment:[/bold] proceeds → "
            f"[cyan]{escape(str(proposal.get('ticker')))}[/cyan] "
            f"[dim]({proposal.get('lane')} lane; advisory — see "
            f"`watchlist queue`)[/dim]"
        )
    else:
        console.print(
            f"\n[dim]Reinvestment: {escape(str(routing.get('reason', '')))}[/dim]"
        )

    blocked = routing.get("blocked") or []
    if blocked:
        console.print(
            f"  [dim]{len(blocked)} candidate(s) blocked "
            f"({', '.join(str(entry.get('ticker')) for entry in blocked)}) — "
            f"reasons in `watchlist queue`[/dim]"
        )


def _evidence_cell(evidence: str) -> str:
    """A trigger's evidence rendered as literal text rather than as markup.

    Evidence is assembled in `lifecycle/advance.py` and appends bracketed
    clauses — `[deployment pace: ...]`, `[gross ... / net ...]`. Rich reads a
    leading `[` as the start of a style tag, so an unescaped cell drops the
    whole clause from the table and shows a sentence that simply stops. The
    figures R5 requires beside every proposed exit vanish silently, in exactly
    the column an owner reads before confirming one.

    Escaping only at the render boundary keeps the stored evidence
    byte-identical — the append-only history holds the sentence that was
    actually reasoned with, not a display-encoded version of it.
    """
    from rich.markup import escape

    return escape(evidence or "")


def _print_concentration(reading) -> None:
    """How crowded the portfolio already is, before the table proposes adding to it.

    Placed with the deployment-pace line and for the same reason: it is context
    for reading the proposals below it. A proposal to enter a lane that is
    already at its cap is a different thing from the same proposal in an empty
    lane, and an owner should not have to scroll past the entry to learn that.

    Every figure here is a COUNT of positioned names — this system records no
    invested amount — so the line says so in the words themselves rather than
    in a caption, and the breaches and same-sector notes are printed beneath it
    where they cannot be missed.
    """
    from rich.markup import escape

    from boundless100x.lifecycle import portfolio

    if not reading:
        return
    # Nothing positioned and nothing to note is not a concentration reading
    # worth a line. A reading that could not be *built* still prints: a gap has
    # to be visible, or it reads as "checked, all clear".
    if reading.get("available") and not reading.get("positioned") and not reading.get(
        "notes"
    ):
        return

    colour = "yellow" if reading.get("breaches") else "dim"
    # Escaped, not trusted: the notes interpolate sector names read from a
    # scraped breadcrumb, and rich would swallow anything bracketed — a cap
    # breach silently truncated is the one line here that must always render.
    console.print(f"[{colour}]{escape(portfolio.describe(reading))}[/{colour}]")
    for note in reading.get("notes") or []:
        console.print(f"  [dim]{escape(note)}[/dim]")
    console.print()


def _snapshot_age(generated_at) -> str:
    """How old the stored view is, in words, or that nobody can tell.

    Display only. Freshness is decided by the revision counters — a clock
    comparison would miss every mutation that does not re-score, and `as_of`
    may be a historical business date in any case.
    """
    from datetime import datetime

    if not generated_at:
        return "generated at an unknown time"
    try:
        when = datetime.fromisoformat(str(generated_at))
    except ValueError:
        return f"generated {generated_at}"
    days = (datetime.now() - when).days
    if days < 0:
        return f"generated {when:%Y-%m-%d %H:%M} (dated ahead of now)"
    if days == 0:
        return f"generated {when:%Y-%m-%d %H:%M} (today)"
    return f"generated {when:%Y-%m-%d %H:%M} ({days} day(s) ago)"


def _print_routing_snapshot(queue, watchlist) -> None:
    """The routing view an owner reads: state first, then what it may show.

    Two sources, deliberately. The **proposal and the blocked list come from
    the snapshot**, because they are the output of a run that ranked candidates
    and cannot be recomputed without one. The **exits come live** from the event
    log, because an idle reading grows every day and a stored one stopped
    growing when the run ended — and because the reconciliation notice for an
    exit stranded in `exit_review` is an instruction about the state of the
    stores *now*.

    Only a `Current` snapshot renders its proposal. `Partial` and `Stale` keep
    every diagnostic and print the refresh instruction where the candidate
    would have gone: a candidate named by incomplete or superseded inputs is a
    recommendation those inputs no longer back.
    """
    from rich.markup import escape

    from boundless100x.lifecycle import friction
    from boundless100x.lifecycle.reinvestment import (
        SNAPSHOT_UNAVAILABLE,
        snapshot_state,
        unroutable_reason,
    )

    snapshot = queue.latest_proposal() or {}
    state = snapshot_state(
        queue.latest_proposal(),
        watchlist.data.get("revision"),
        queue.data.get("revision"),
    )
    missing = state["state"] == SNAPSHOT_UNAVAILABLE

    colour = {
        "current": "green", "stale": "yellow",
        "partial": "yellow", "unavailable": "dim",
    }[state["state"]]
    # No age for a snapshot that does not exist: "generated at an unknown time"
    # invites the reader to wonder which run wrote it.
    age = "" if missing else (
        f" [dim]({_snapshot_age(state['generated_at'])}"
        f"{', as of ' + str(state['as_of']) if state['as_of'] else ''})[/dim]"
    )
    console.print(
        f"\n[bold]Reinvestment queue[/bold] — [{colour}]"
        f"{state['state'].capitalize()}[/{colour}]{age}"
    )
    if state["reason"]:
        console.print(f"  [dim]{escape(state['reason'])}[/dim]")

    if state["renders_proposal"]:
        proposal = snapshot.get("proposal")
        if proposal:
            console.print(
                f"\n[bold]Proposed destination for proceeds:[/bold] "
                f"[cyan]{escape(str(proposal.get('ticker')))}[/cyan] "
                f"({proposal.get('lane')} lane, {proposal.get('state')})"
            )
            console.print(
                f"  [dim]{escape(str(proposal.get('trigger_id') or 'no trigger'))}: "
                f"{escape(str(proposal.get('evidence') or ''))}[/dim]"
            )
            console.print(
                "  [dim]Advisory only — record what you actually deploy with "
                "`watchlist queue route`.[/dim]"
            )
        elif snapshot.get("reason"):
            console.print(f"\n[yellow]{escape(snapshot['reason'])}[/yellow]")
    elif not missing:
        # Where the candidate would have gone. Omitted when there is no
        # snapshot at all, because the state's own reason already says to run
        # the command that would produce one.
        console.print(
            "\n[dim]No candidate is shown for this snapshot — run "
            "`watchlist advance` to refresh it.[/dim]"
        )

    # Printed in every state: which candidates were skipped, and why, is a true
    # statement about the run that produced the snapshot even when the ranking
    # itself has been superseded. Without it, an all-blocked run would render
    # exactly like an empty pipeline.
    blocked = snapshot.get("blocked") or []
    if blocked:
        console.print("\n[bold]Blocked candidates[/bold]")
    for entry in blocked:
        console.print(
            f"  [cyan]{escape(str(entry.get('ticker')))}[/cyan] "
            f"[dim]({entry.get('lane')} lane, {entry.get('state')})[/dim]"
        )
        for reason in entry.get("reasons") or []:
            console.print(f"    [yellow]- {escape(str(reason))}[/yellow]")

    views = queue.exit_views(watchlist)
    if views:
        console.print("\n[bold]Exit events[/bold]")
    for view in views[-8:]:
        idle = view["idle_days"]
        idle_text = "unknown" if idle is None else f"{idle}"
        if view["closed"]:
            line = (
                f"routed into {view['routed_into']} on {view['deployed_at']} "
                f"— idle {idle_text} day(s)"
            )
            tone = "dim"
        else:
            line = f"awaiting routing — idle {idle_text} day(s)"
            tone = "yellow"
        console.print(
            f"  [cyan]{escape(str(view['exit_id']))}[/cyan] "
            f"[dim]{view['ticker']} ({view['lane']} lane), sold "
            f"{view['at']}[/dim]"
        )
        console.print(f"    [{tone}]{escape(line)}[/{tone}]")
        console.print(
            f"    [dim]{escape(friction.describe(view['friction']))}[/dim]"
        )
        if view["note"]:
            console.print(f"    [red]{escape(view['note'])}[/red]")

    # The closing line, and the one place this display could quietly lie. An
    # exit recorded but not confirmed is capital the owner has and cannot yet
    # reach; printing "no proceeds awaiting routing" over it is the false
    # all-clear the queue exists to prevent, so the sentence is chosen by
    # `unroutable_reason` and the incomplete case is yellow, not dim.
    routable, incomplete = queue.unrouted_views(watchlist)
    if not routable:
        tone = "yellow" if incomplete else "dim"
        console.print(f"\n[{tone}]{escape(unroutable_reason(incomplete))}[/{tone}]")


def _print_exit_friction(outcomes) -> None:
    """Net of tax and slippage, beside gross, for every proposed exit (§8.2).

    Printed as its own block rather than squeezed into the table's evidence
    column, because the assumptions have to travel with the figures: a return
    shown without them invites being read as money that was made, and every
    input here is a proxy — a `probe` confirmation date rather than a fill,
    market bars rather than trade prices, no cost basis anywhere.

    A reading that could not be computed is shown *with its reason* rather than
    omitted, for the reason the whole codebase treats gaps this way: a missing
    line and a silent zero look identical, and only one of them is honest.

    Escaped like every other site that renders `describe`. The unavailable
    branch interpolates an arbitrary exception message, so a bracketed fragment
    would be swallowed as markup — the same silent truncation `_evidence_cell`
    exists to prevent — and a closing form would raise `MarkupError` here,
    aborting the run *after* its transitions had already been committed.
    """
    from rich.markup import escape

    from boundless100x.lifecycle import friction

    exits = [
        o for o in outcomes
        if o.get("proposal") and o["proposal"].get("friction") is not None
    ]
    if not exits:
        return

    console.print(
        "\n[bold]Exit friction[/bold] [dim](modeled estimates — no fills, no "
        "cost basis)[/dim]"
    )
    for o in exits:
        reading = o["proposal"]["friction"]
        colour = "yellow" if reading.get("available") else "dim"
        console.print(
            f"  [cyan]{escape(str(o['ticker']))}[/cyan] "
            f"[{colour}]{escape(friction.describe(reading))}[/{colour}]"
        )
    console.print(
        "[dim]Holding period is measured from the `probe` confirmation date, "
        "not a broker fill.[/dim]"
    )


