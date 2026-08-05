"""Pixel Purge CLI (Typer). Phase 1: init, ingest, dedup, stats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config
from .database import Database

app = typer.Typer(
    add_completion=False,
    help="Fully local Google Photos cleanup, dedup, and classification engine.",
    no_args_is_help=True,
)
console = Console()


def _db(config_path: Optional[Path] = None) -> Database:
    cfg = Config.load(config_path)
    return Database(cfg.db_path)


@app.command()
def version() -> None:
    """Print the Pixel Purge version."""
    console.print(f"pixel-purge {__version__}")


@app.command()
def init() -> None:
    """Initialize the manifest database and app directory."""
    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
    console.print(f"[green]Initialized[/green] manifest DB at {cfg.db_path}")


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Takeout .tgz file or (extracted) directory"),
    resume: bool = typer.Option(True, help="Skip files already in the manifest (by path)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only; write nothing"),
    keyframes: bool = typer.Option(True, help="Extract video keyframes (needs ffmpeg)"),
) -> None:
    """Ingest a Google Takeout export: discover media, merge metadata, record."""
    from .ingestion import ingest as run_ingest

    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
        result = run_ingest(
            source, db, resume=resume, dry_run=dry_run, extract_keyframes=keyframes
        )

    table = Table(title="Ingestion Summary", show_header=False)
    table.add_row("Discovered", str(result.discovered))
    table.add_row("Ingested", str(result.ingested))
    table.add_row("Skipped (already present)", str(result.skipped_existing))
    table.add_row("With sidecar metadata", str(result.with_sidecar))
    table.add_row("Filled from EXIF fallback", str(result.from_exif))
    table.add_row("Errors", str(result.errors))
    console.print(table)


@app.command()
def dedup(
    gps_radius: float = typer.Option(None, help="GPS bucket radius in meters"),
    time_window: int = typer.Option(None, help="Temporal session window in minutes"),
    hamming_threshold: int = typer.Option(None, help="Max pHash Hamming distance for a match"),
    tier: Optional[int] = typer.Option(None, help="Run a single tier only (1, 2, or 3)"),
    stats: bool = typer.Option(False, "--stats", help="Show dedup statistics and exit"),
) -> None:
    """Run the hierarchical deduplication pipeline (Tiers 1-3)."""
    from .dedup import run_dedup

    cfg = Config.load()
    d = cfg.dedup
    gps_radius = gps_radius if gps_radius is not None else d.gps_radius_meters
    time_window = time_window if time_window is not None else d.time_window_minutes
    hamming_threshold = (
        hamming_threshold if hamming_threshold is not None else d.hamming_threshold
    )
    tiers = (tier,) if tier else (1, 2, 3)

    with Database(cfg.db_path) as db:
        db.init_schema()
        if stats:
            _print_stats(db)
            return
        run_dedup(
            db,
            gps_radius_m=gps_radius,
            time_window_min=time_window,
            hamming_threshold=hamming_threshold,
            tiers=tiers,
        )
        _print_stats(db)


def _print_stats(db: Database) -> None:
    s = db.dedup_stats()
    reclaimable_mb = (s["reclaimable_bytes"] or 0) / (1024 * 1024)
    table = Table(title="Dedup Statistics", show_header=False)
    table.add_row("Total items", str(s["total"]))
    table.add_row("Duplicates flagged", str(s["duplicates"] or 0))
    table.add_row("  Exact (SHA-256)", str(s["exact_dupes"] or 0))
    table.add_row("  Visual (pHash)", str(s["visual_dupes"] or 0))
    table.add_row("Reclaimable", f"{reclaimable_mb:,.1f} MB")
    console.print(table)


if __name__ == "__main__":
    app()
