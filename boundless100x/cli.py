"""Boundless100x CLI — SQGLP Financial Research System."""

import logging
from pathlib import Path

from dotenv import load_dotenv
import typer

load_dotenv()
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="boundless100x",
    help="SQGLP Financial Research System for Indian Markets",
    no_args_is_help=True,
)
console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def analyze(
    ticker: str = typer.Argument(help="NSE symbol (e.g., ASTRAL)"),
    bse_code: str = typer.Option(None, help="BSE scrip code"),
    peers: str = typer.Option(None, help="Comma-separated manual peer list"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM analysis"),
    deep: bool = typer.Option(False, "--deep", help="Use Opus for Pass 1 & 2 (~5x LLM cost, deeper analysis)"),
    max_peers: int = typer.Option(5, help="Max peers to compute"),
    formats: str = typer.Option("html,md,json", help="Output formats (comma-separated)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
):
    """Run full SQGLP analysis pipeline for a company."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService
    from boundless100x.output.report_generator import ReportGenerator

    mode = " [bold magenta](DEEP — Opus)[/bold magenta]" if deep else ""
    console.print(f"\n[bold blue]Boundless100x SQGLP Analysis: {ticker}[/bold blue]{mode}\n")

    svc = Boundless100xService()
    result = svc.analyze(
        ticker=ticker,
        bse_code=bse_code,
        use_llm=not no_llm,
        deep=deep,
        max_peer_compute=max_peers,
    )

    # Print summary to console
    _print_scores(result, svc)

    if result.peers:
        _print_peers(result)

    if result.llm_analysis and not result.llm_analysis.get("skipped"):
        _print_llm_summary(result)

    # Generate reports
    fmt_list = [f.strip() for f in formats.split(",")]
    generator = ReportGenerator()
    report_dir = generator.generate(result, formats=fmt_list)

    console.print(f"\n[bold green]Reports saved to:[/bold green] {report_dir}")

    if result.errors:
        console.print(f"\n[bold yellow]Warnings ({len(result.errors)}):[/bold yellow]")
        for e in result.errors:
            console.print(f"  [yellow]! {e}[/yellow]")


@app.command()
def compute(
    ticker: str = typer.Argument(help="NSE symbol"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Compute metrics only (no peers, no LLM, no reports)."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService

    console.print(f"\n[bold blue]Computing metrics for {ticker}[/bold blue]\n")

    svc = Boundless100xService()
    result = svc.analyze_quick(ticker)

    _print_scores(result, svc)

    # Print all metrics
    table = Table(title="Computed Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Flags", style="yellow")

    for mid, mr in sorted(result.metrics.items()):
        if mr.ok:
            val = mr.value
            if isinstance(val, float):
                val = f"{val:.2f}"
            flags = ", ".join(mr.flags) if mr.flags else ""
            table.add_row(mid, str(val), flags)
        else:
            table.add_row(mid, f"[red]ERR: {mr.error}[/red]", "")

    console.print(table)


@app.command(name="peers")
def discover_peers(
    ticker: str = typer.Argument(help="NSE symbol"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Discover sector peers for a company."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService

    console.print(f"\n[bold blue]Discovering peers for {ticker}[/bold blue]\n")

    svc = Boundless100xService()
    result = svc.analyze(ticker, skip_peers=False, use_llm=False, max_peer_compute=0)

    if result.peers:
        _print_peers(result)
    else:
        console.print("[red]No peers found.[/red]")


@app.command()
def screen(
    tickers: str = typer.Argument(help="Comma-separated NSE symbols to screen"),
    preset: str = typer.Option("compounders", help="Screening preset name"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Screen a list of companies using preset criteria."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService
    from boundless100x.compute_engine.screener import Screener

    ticker_list = [t.strip() for t in tickers.split(",")]
    console.print(
        f"\n[bold blue]Screening {len(ticker_list)} companies "
        f"with preset: {preset}[/bold blue]\n"
    )

    svc = Boundless100xService()
    screener = Screener()

    # Show preset info
    preset_info = screener.presets.get(preset)
    if preset_info:
        console.print(f"[dim]{preset_info.get('name', preset)}[/dim]")
        console.print(f"[dim]{preset_info.get('description', '').strip()}[/dim]\n")

    survivors = screener.screen_quick(ticker_list, svc, preset=preset)

    if not survivors:
        console.print("[yellow]No companies passed the screening criteria.[/yellow]")
        return

    table = Table(title=f"Screening Results — {preset}")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Composite", justify="right")

    # Add filter metric columns
    filter_metrics = list(preset_info.get("filters", {}).keys()) if preset_info else []
    for mid in filter_metrics:
        table.add_column(mid, justify="right")

    for entry in survivors:
        row = [
            str(entry.get("rank", "")),
            entry["ticker"],
            f"{entry.get('sqglp_composite', 'N/A')}/10" if entry.get("sqglp_composite") else "N/A",
        ]
        for mid in filter_metrics:
            val = entry.get(mid)
            if val is not None:
                row.append(f"{val:.2f}")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[green]{len(survivors)} companies passed screening[/green]")


# ── Watchlist Commands ──

watchlist_app = typer.Typer(help="Manage your company watchlist")
app.add_typer(watchlist_app, name="watchlist")


@watchlist_app.command("show")
def watchlist_show():
    """Show all companies in the watchlist."""
    from boundless100x.watchlist import WatchlistManager

    wm = WatchlistManager()
    entries = wm.list()

    if not entries:
        console.print("[dim]Watchlist is empty. Add companies with: watchlist add TICKER[/dim]")
        return

    table = Table(title="Watchlist")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Added", style="dim")
    table.add_column("Last Run", style="dim")
    table.add_column("Composite", justify="right")
    table.add_column("Notes")

    for e in entries:
        last_run = e["last_run"][:10] if e["last_run"] else "never"
        composite = f"{e['last_composite']}/10" if e["last_composite"] else "—"
        table.add_row(
            e["ticker"],
            e["added"][:10] if e["added"] else "",
            last_run,
            composite,
            e.get("notes", ""),
        )

    console.print(table)


@watchlist_app.command("add")
def watchlist_add(
    ticker: str = typer.Argument(help="NSE symbol to add"),
    notes: str = typer.Option("", help="Optional notes"),
):
    """Add a company to the watchlist."""
    from boundless100x.watchlist import WatchlistManager

    wm = WatchlistManager()
    if wm.add(ticker, notes=notes):
        console.print(f"[green]Added {ticker} to watchlist[/green]")
    else:
        console.print(f"[yellow]{ticker} is already in the watchlist[/yellow]")


@watchlist_app.command("remove")
def watchlist_remove(
    ticker: str = typer.Argument(help="NSE symbol to remove"),
):
    """Remove a company from the watchlist."""
    from boundless100x.watchlist import WatchlistManager

    wm = WatchlistManager()
    if wm.remove(ticker):
        console.print(f"[green]Removed {ticker} from watchlist[/green]")
    else:
        console.print(f"[yellow]{ticker} not found in watchlist[/yellow]")


@watchlist_app.command("update")
def watchlist_update(
    quarterly: bool = typer.Option(False, "--quarterly", help="Only update stale (90+ days) entries"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Re-run analysis on all watchlist companies."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService
    from boundless100x.watchlist import WatchlistManager

    svc = Boundless100xService()
    wm = WatchlistManager()

    results = wm.update_all(svc, quarterly=quarterly)

    if not results:
        console.print("[dim]No companies to update[/dim]")
        return

    table = Table(title="Watchlist Update Results")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Composite", justify="right")
    table.add_column("Status")

    for ticker, composite in results:
        if composite is not None:
            color = "green" if composite >= 7 else "yellow" if composite >= 4 else "red"
            table.add_row(ticker, f"[{color}]{composite}/10[/{color}]", "[green]OK[/green]")
        else:
            table.add_row(ticker, "—", "[red]FAILED[/red]")

    console.print(table)


# ── Display Helpers ──

def _print_scores(result, svc):
    summary = svc.get_element_summary(result)

    table = Table(title=f"SQGLP Scores — {result.ticker}")
    table.add_column("Element", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right", style="dim")

    element_order = [
        "size", "quality_business", "quality_management",
        "growth", "longevity", "price",
    ]
    element_names = {
        "size": "Size (S)",
        "quality_business": "Quality - Business (Q)",
        "quality_management": "Quality - Mgmt (Q)",
        "growth": "Growth (G)",
        "longevity": "Longevity (L)",
        "price": "Price (P)",
    }

    for el in element_order:
        info = summary.get(el, {})
        score = info.get("score")
        weight = info.get("weight", "")
        if score is not None:
            color = "green" if score >= 7 else "yellow" if score >= 4 else "red"
            table.add_row(element_names.get(el, el), f"[{color}]{score:.1f}/10[/{color}]", weight)
        else:
            table.add_row(element_names.get(el, el), "[dim]N/A[/dim]", weight)

    composite = summary.get("composite")
    table.add_section()
    table.add_row("[bold]COMPOSITE[/bold]", f"[bold]{composite}/10[/bold]", "100%")

    console.print(table)


def _print_peers(result):
    if not result.peers:
        return

    table = Table(title="Peer Discovery")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Name")
    table.add_column("MCap (₹Cr)", justify="right")
    table.add_column("PE", justify="right")
    table.add_column("RoCE", justify="right")
    table.add_column("Type", style="dim")

    for pt in result.peers.direct_competitors:
        info = result.peers.peer_data.get(pt, {})
        mcap = info.get("market_cap", 0)
        pe = info.get("pe", 0)
        roce = info.get("roce", 0)
        name = info.get("name", "")
        table.add_row(
            pt, name,
            f"{mcap:,.0f}" if mcap else "—",
            f"{pe:.1f}" if pe else "—",
            f"{roce:.1f}%" if roce else "—",
            "Direct",
        )

    for pt in result.peers.financial_peers:
        if pt not in result.peers.direct_competitors:
            info = result.peers.peer_data.get(pt, {})
            table.add_row(
                pt, info.get("name", ""),
                f"{info.get('market_cap', 0):,.0f}",
                f"{info.get('pe', 0):.1f}",
                f"{info.get('roce', 0):.1f}%",
                "Financial",
            )

    console.print(table)


def _print_llm_summary(result):
    llm = result.llm_analysis
    if not llm:
        return

    p2 = llm.get("pass2", {})
    if p2 and not p2.get("error") and not p2.get("skipped"):
        console.print("\n[bold]Investment Thesis:[/bold]")
        console.print(f"  {p2.get('thesis', 'N/A')}")
        console.print(
            f"  Conviction: [bold]{p2.get('conviction_level', 'N/A')}[/bold] | "
            f"Action: [bold]{p2.get('suggested_action', 'N/A')}[/bold] | "
            f"Period: {p2.get('target_holding_period', 'N/A')}"
        )

    usage = llm.get("usage", {})
    if usage:
        console.print(
            f"\n[dim]LLM: {usage.get('total_tokens', 0)} tokens | "
            f"~${usage.get('estimated_cost_usd', 0):.4f} | "
            f"{usage.get('total_seconds', 0)}s[/dim]"
        )


if __name__ == "__main__":
    try:
        app()
    finally:
        # Force exit — jugaad-data / requests can leave background threads alive
        import os
        os._exit(0)
