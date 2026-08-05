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
            source, db, resume=resume, dry_run=dry_run, extract_keyframes_enabled=keyframes
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
    text_hamming_threshold: int = typer.Option(
        None, help="Stricter Hamming distance for text-heavy images (screenshots/docs)"),
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
    text_hamming_threshold = (
        text_hamming_threshold if text_hamming_threshold is not None
        else d.text_hamming_threshold
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
            text_hamming_threshold=text_hamming_threshold,
            tiers=tiers,
        )
        _print_stats(db)


@app.command()
def classify(
    vision_only: bool = typer.Option(False, "--vision-only", help="Run CLIP vision only"),
    faces_only: bool = typer.Option(False, "--faces-only", help="Run face clustering only"),
    device: str = typer.Option("auto", help="Torch device: auto, mps, cpu"),
    eps: float = typer.Option(0.45, help="DBSCAN eps (cosine) for face clustering"),
    min_samples: int = typer.Option(2, help="DBSCAN min_samples for face clustering"),
    stats: bool = typer.Option(False, "--stats", help="Show classification stats and exit"),
) -> None:
    """Module C: local CLIP vision tagging + InsightFace clustering."""
    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
        if stats:
            _print_classify_stats(db)
            return

        run_vision = not faces_only
        run_faces = not vision_only
        if run_vision:
            from .vision.clip_tagger import run_vision as _run_vision

            _run_vision(db, device=device)
        if run_faces:
            from .vision.faces import run_face_clustering

            run_face_clustering(db, eps=eps, min_samples=min_samples)
        _print_classify_stats(db)


@app.command()
def delta(
    source: Path = typer.Argument(..., help="Incremental Takeout .tgz or export directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report new items only"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Skip dedup vs manifest"),
    stats: bool = typer.Option(False, "--stats", help="Show delta stats and exit"),
    notify: bool = typer.Option(False, "--notify", help="Post a local notification when done"),
) -> None:
    """Module E: local monthly delta — ingest new export, dedup, CLIP-classify new items."""
    from .delta import delta_run

    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
        if stats:
            _print_delta_stats(db)
            return
        result = delta_run(source, db, cfg, dry_run=dry_run, dedup=not no_dedup)
        if not dry_run:
            _print_delta_stats(db)
            if notify:
                _notify(f"Pixel Purge: classified {result.classified} new photo(s)")


@app.command()
def schedule(
    source: Path = typer.Argument(..., help="Export path the monthly delta should process"),
    day: int = typer.Option(1, help="Day of month to run"),
    hour: int = typer.Option(9, help="Hour of day (0-23) to run"),
) -> None:
    """Install a launchd agent that runs `pixel-purge delta` monthly (macOS)."""
    from .delta.scheduling import install_agent, load_hint

    plist = install_agent(source, day=day, hour=hour)
    console.print(load_hint(plist))


@app.command()
def dashboard(
    port: int = typer.Option(8787, help="Port to serve on"),
    host: str = typer.Option("127.0.0.1", help="Bind address (localhost only by default)"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the browser"),
) -> None:
    """Launch the local dashboard (Review / Dedup / Faces) at http://localhost:PORT."""
    import webbrowser

    import uvicorn

    from .dashboard import create_app

    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
    app_ = create_app(cfg.db_path)
    url = f"http://{host}:{port}"
    console.print(f"[green]Pixel Purge dashboard[/green] → {url}  (Ctrl+C to stop)")
    if not no_open:
        webbrowser.open(url)
    uvicorn.run(app_, host=host, port=port, log_level="warning")


def _notify(message: str) -> None:
    """Best-effort macOS notification; silently no-op elsewhere."""
    import shutil
    import subprocess

    if shutil.which("osascript"):
        try:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "Pixel Purge"'],
                capture_output=True, timeout=10,
            )
        except subprocess.SubprocessError:
            pass


def _print_delta_stats(db: Database) -> None:
    v = db.vision_stats()
    table = Table(title="Delta Statistics", show_header=False)
    for bucket in ("ADHOC_PURGE", "TRIP", "FAMILY_KEEP", "OTHER"):
        table.add_row(bucket, str(v.get(bucket, 0)))
    table.add_row("Watermark (epoch)", str(db.get_delta_watermark()))
    console.print(table)


@app.command("eval")
def eval_cmd(
    manifest: Path = typer.Argument(..., help="Labeled-set CSV with columns: path,bucket"),
    device: str = typer.Option("auto", help="Torch device: auto, mps, cpu"),
    target: float = typer.Option(0.90, help="Accuracy target to check against"),
) -> None:
    """Measure vision classifier accuracy against a labeled set (path,bucket CSV) [H2]."""
    from .vision.eval import evaluate, load_eval_manifest, make_clip_classify_fn
    from .vision.taxonomy import BUCKETS

    cases = load_eval_manifest(manifest)
    metrics = evaluate(cases, make_clip_classify_fn(device=device))

    # Confusion matrix (rows = expected, cols = predicted)
    ct = Table(title="Confusion Matrix (row=expected, col=predicted)")
    ct.add_column("expected \\ predicted")
    for b in BUCKETS:
        ct.add_column(b, justify="right")
    for e in BUCKETS:
        ct.add_row(e, *[str(metrics.confusion[e][p]) for p in BUCKETS])
    console.print(ct)

    mt = Table(title="Per-bucket metrics")
    for col in ("bucket", "precision", "recall", "f1", "support"):
        mt.add_column(col, justify="right" if col != "bucket" else "left")
    for b in BUCKETS:
        m = metrics.per_bucket[b]
        mt.add_row(b, f"{m['precision']:.2f}", f"{m['recall']:.2f}",
                   f"{m['f1']:.2f}", str(int(m["support"])))
    console.print(mt)

    ok = metrics.accuracy >= target
    color = "green" if ok else "red"
    console.print(f"[{color}]Overall accuracy: {metrics.accuracy:.1%}[/{color}] "
                  f"({metrics.correct}/{metrics.total}); target {target:.0%} "
                  f"{'met ✅' if ok else 'not met ❌'}")


@app.command()
def review(
    batch_size: int = typer.Option(20, help="Items shown per review batch"),
) -> None:
    """Interactively review flagged duplicates + purge candidates (approve/reject)."""
    from .tui import review as run_review

    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
        run_review(db, batch_size=batch_size)


@app.command()
def export(
    output: Path = typer.Option(Path("manifest.csv"), "--output", "-o", help="CSV output path"),
    deletions_only: bool = typer.Option(False, "--deletions-only",
                                        help="Export only items marked DELETE"),
) -> None:
    """Export the manifest (or just the deletion set) to CSV."""
    from .cleanup.export import export_csv, export_deletion_manifest

    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
        n = (export_deletion_manifest(db, output) if deletions_only
             else export_csv(db, output))
    console.print(f"[green]Exported[/green] {n} row(s) to {output}")


@app.command()
def cleanup(
    strategy: str = typer.Option("export", help="export | stage | clean-slate | browser-auto"),
    staging_dir: Path = typer.Option(Path("~/pixel-purge-staging").expanduser(),
                                     help="Where to stage curated keepers"),
    output: Path = typer.Option(Path("deletions.csv"), "--output", "-o",
                                help="Deletion manifest path (export strategy)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only; change nothing"),
    i_have_a_backup: bool = typer.Option(False, "--i-have-a-backup",
                                         help="Acknowledge you have a verified backup (clean-slate)"),
    confirm: str = typer.Option("", "--confirm",
                                help="Type the confirmation phrase for clean-slate"),
) -> None:
    """Execute cleanup. Default 'export' writes a safe deletion manifest.

    Deletion is never performed by the API (Google removed those scopes and never
    supported media deletion). 'export' is the recommended primary path; 'clean-slate'
    is a gated last-resort; 'browser-auto' is experimental.
    """
    from .cleanup.planner import CONFIRM_PHRASE, clean_slate_allowed, print_plan

    cfg = Config.load()
    with Database(cfg.db_path) as db:
        db.init_schema()
        print_plan(db)

        if strategy == "export":
            from .cleanup.export import export_deletion_manifest

            n = export_deletion_manifest(db, output)
            console.print(f"[green]Deletion manifest:[/green] {n} item(s) -> {output}")
            console.print("Review it, then delete manually in Google Photos, or use "
                          "[cyan]--strategy browser-auto[/cyan] (experimental).")

        elif strategy == "stage":
            from .cleanup.curate import stage_keepers

            r = stage_keepers(db, staging_dir, dry_run=dry_run)
            db.commit()
            console.print(f"[green]Staged[/green] {r.staged}; metadata restored "
                          f"{r.metadata_restored}, skipped {r.metadata_skipped}.")

        elif strategy == "clean-slate":
            if not clean_slate_allowed(confirm, i_have_a_backup):
                console.print(
                    "[bold red]Clean-slate is destructive and irreversible.[/bold red]\n"
                    "It requires BOTH:\n"
                    "  --i-have-a-backup   (verified independent backup)\n"
                    f'  --confirm "{CONFIRM_PHRASE}"\n'
                    "Aborted."
                )
                raise typer.Exit(code=1)
            from .cleanup.curate import stage_keepers

            r = stage_keepers(db, staging_dir, dry_run=dry_run)
            db.commit()
            console.print(f"[green]Staged[/green] {r.staged} keeper(s) to {staging_dir}.")
            console.print("\n[bold red]⚠️  MANUAL STEP:[/bold red] delete all photos at "
                          "photos.google.com and empty trash, then run "
                          "[cyan]pixel-purge cleanup --strategy upload[/cyan].")

        elif strategy == "upload":
            _run_upload(db, staging_dir, dry_run)

        elif strategy == "browser-auto":
            _run_browser_auto(db)

        else:
            console.print(f"[red]Unknown strategy:[/red] {strategy}")
            raise typer.Exit(code=1)


def _run_upload(db: Database, staging_dir: Path, dry_run: bool) -> None:
    if dry_run:
        console.print(f"[cyan]Dry run:[/cyan] would upload staged files from {staging_dir}")
        return
    from .cleanup.google_auth import build_photos_service
    from .cleanup.uploader import upload_curated_set

    service = build_photos_service()
    upload_curated_set(db, service, staging_dir)


def _run_browser_auto(db: Database) -> None:
    import asyncio

    from .cleanup.browser_auto import delete_flagged

    asyncio.run(delete_flagged(db))


def _print_classify_stats(db: Database) -> None:
    v = db.vision_stats()
    f = db.face_cluster_stats()
    table = Table(title="Classification Statistics", show_header=False)
    for bucket in ("ADHOC_PURGE", "TRIP", "FAMILY_KEEP", "OTHER"):
        table.add_row(bucket, str(v.get(bucket, 0)))
    table.add_row("Faces detected", str(f["faces"]))
    table.add_row("People (clusters)", str(f["clusters"]))
    table.add_row("Unclustered faces", str(f["unclustered"]))
    console.print(table)


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
