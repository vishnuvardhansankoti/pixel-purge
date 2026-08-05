"""Interactive Rich TUI for batch-reviewing flagged items.

Review targets: dedup duplicates (keeper_status=REVIEW) and vision purge
candidates (classification_bucket=ADHOC_PURGE). The reviewer confirms deletion
(-> DELETE) or keeps an item (-> KEEP), batch by batch. Nothing is deleted here;
this only records the human decision on the manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ..database import Database
from ..models import MediaRecord

console = Console()


def mark_for_deletion(db: Database, item_id: int) -> None:
    db.set_keeper_status(item_id, "DELETE")


def keep_item(db: Database, item_id: int) -> None:
    db.set_keeper_status(item_id, "KEEP")


def _fmt_ts(ts: int | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _reason(item: MediaRecord) -> str:
    if item.is_duplicate:
        tier = "exact" if item.dedup_tier == "EXACT_HASH" else "visual"
        extra = f" d={item.hamming_distance}" if item.hamming_distance is not None else ""
        return f"duplicate ({tier}{extra}) of #{item.duplicate_of}"
    return "purge candidate (vision)"


def build_review_table(items: list[MediaRecord]) -> Table:
    table = Table(title=f"Review — {len(items)} item(s)")
    table.add_column("ID", justify="right")
    table.add_column("Filename")
    table.add_column("Date")
    table.add_column("Size", justify="right")
    table.add_column("Bucket")
    table.add_column("Why")
    for it in items:
        size_mb = it.file_size / (1024 * 1024)
        table.add_row(
            str(it.id), it.filename, _fmt_ts(it.taken_timestamp),
            f"{size_mb:.1f}MB", it.classification_bucket or "-",
            _reason(it),
        )
    return table


def review(db: Database, batch_size: int = 20) -> dict:
    """Interactive batch review loop. Returns counts of decisions made."""
    items = db.get_items_for_review()
    if not items:
        console.print("[green]Nothing to review.[/green]")
        return {"deleted": 0, "kept": 0, "skipped": 0}

    counts = {"deleted": 0, "kept": 0, "skipped": 0}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        console.print(build_review_table(batch))
        console.print(
            "[bold]Batch action:[/bold] [red]d[/red]=delete all  "
            "[green]k[/green]=keep all  [yellow]s[/yellow]=skip  [cyan]q[/cyan]=quit"
        )
        choice = Prompt.ask("Choice", choices=["d", "k", "s", "q"], default="s")
        if choice == "q":
            break
        for it in batch:
            if choice == "d":
                mark_for_deletion(db, it.id)
                counts["deleted"] += 1
            elif choice == "k":
                keep_item(db, it.id)
                counts["kept"] += 1
            else:
                counts["skipped"] += 1
        db.commit()

    console.print(
        f"[green]Review done:[/green] {counts['deleted']} to delete, "
        f"{counts['kept']} kept, {counts['skipped']} skipped."
    )
    return counts
