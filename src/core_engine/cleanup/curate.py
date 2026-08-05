"""Stage the curated keeper set to a clean directory, restoring metadata.

Used by the clean-slate re-upload path: copy every keeper to a staging dir and
write GPS/timestamp back into the copy so Google Photos reads the original date.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from ..database import Database
from .metadata_restore import restore_metadata

console = Console()


@dataclass
class StageResult:
    staged: int = 0
    metadata_restored: int = 0
    metadata_skipped: int = 0
    errors: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)


def stage_keepers(
    db: Database, staging_dir: Path, restore: bool = True, dry_run: bool = False
) -> StageResult:
    """Copy keeper files to staging_dir, restoring metadata into each copy."""
    keepers = db.get_keepers()
    result = StageResult()
    staging_dir = Path(staging_dir)

    if dry_run:
        console.print(
            f"[cyan]Dry run:[/cyan] would stage {len(keepers)} keeper(s) to {staging_dir}"
        )
        result.staged = len(keepers)
        return result

    staging_dir.mkdir(parents=True, exist_ok=True)
    with Progress(console=console) as progress:
        task = progress.add_task("Staging keepers...", total=len(keepers))
        for item in keepers:
            try:
                src = Path(item.local_path)
                dest = _unique_dest(staging_dir, item.filename)
                shutil.copy2(src, dest)
                result.staged += 1

                if restore:
                    r = restore_metadata(dest, item)
                    if r.ok and r.reason != "nothing to restore":
                        result.metadata_restored += 1
                    elif not r.ok:
                        result.metadata_skipped += 1
                        key = r.reason.split(":")[0][:40]
                        result.skip_reasons[key] = result.skip_reasons.get(key, 0) + 1
            except OSError as e:
                result.errors += 1
                console.print(f"[yellow]stage skip[/yellow] {item.filename}: {e}")
            progress.advance(task)

    return result


def _unique_dest(staging_dir: Path, filename: str) -> Path:
    """Avoid collisions when keepers from different albums share a basename."""
    dest = staging_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    n = 1
    while (staging_dir / f"{stem}_{n}{suffix}").exists():
        n += 1
    return staging_dir / f"{stem}_{n}{suffix}"
