"""Cleanup planning + the safety gate for destructive actions [C2].

Clean-slate (wipe the whole cloud library, then re-upload) is irreversible, so it
is demoted to a last-resort path guarded by two independent conditions: an
explicit "I have a verified backup" acknowledgement AND a typed confirmation
phrase. The gate logic is a pure function so it can be unit-tested.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ..database import Database

console = Console()

CONFIRM_PHRASE = "DELETE MY LIBRARY"


def clean_slate_allowed(typed_confirmation: str | None, have_backup: bool) -> bool:
    """Both conditions must hold before a clean-slate wipe may proceed."""
    return bool(have_backup) and (typed_confirmation or "").strip() == CONFIRM_PHRASE


def print_plan(db: Database) -> dict:
    s = db.cleanup_summary()
    delete_mb = (s["delete_bytes"] or 0) / (1024 * 1024)
    table = Table(title="Cleanup Plan", show_header=False)
    table.add_row("Total ingested", str(s["total"]))
    table.add_row("Marked KEEP/pending", str(s["to_keep"] or 0))
    table.add_row("Marked DELETE", str(s["to_delete"] or 0))
    table.add_row("Reclaimable if deleted", f"{delete_mb:,.1f} MB")
    console.print(table)
    return s
