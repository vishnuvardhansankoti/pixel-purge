"""Module A orchestrator: discover -> merge metadata -> insert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

import json

from ..database import Database
from ..models import MediaRecord
from .exif_extractor import extract_exif_metadata
from .keyframe import extract_keyframes
from .media import classify_media_type
from .sidecar_merger import merge_sidecar_metadata, resolve_sidecar
from .takeout_parser import discover_media_files, resolve_media_root

console = Console()


@dataclass
class IngestResult:
    discovered: int
    ingested: int
    skipped_existing: int
    errors: int
    with_sidecar: int
    from_exif: int


def ingest(
    source_path: Path,
    db: Database,
    resume: bool = True,
    dry_run: bool = False,
    extract_keyframes_enabled: bool = True,
) -> IngestResult:
    """Ingest a Google Takeout export (archive or directory) into the manifest."""
    media_root = resolve_media_root(Path(source_path))
    all_files = discover_media_files(media_root)

    # Resume filters on the FULL PATH [M1], not the basename — Takeout reuses
    # filenames across albums and local_path is the UNIQUE key.
    skipped_existing = 0
    if resume:
        already = db.get_ingested_paths()
        pending = [f for f in all_files if str(f) not in already]
        skipped_existing = len(all_files) - len(pending)
    else:
        pending = all_files

    result = IngestResult(
        discovered=len(all_files),
        ingested=0,
        skipped_existing=skipped_existing,
        errors=0,
        with_sidecar=0,
        from_exif=0,
    )

    if dry_run:
        console.print(
            f"[cyan]Dry run:[/cyan] {len(all_files)} media files discovered, "
            f"{len(pending)} new, {skipped_existing} already ingested."
        )
        return result

    with Progress(console=console) as progress:
        task = progress.add_task("Ingesting media...", total=len(pending))
        for media_file in pending:
            try:
                record = MediaRecord(
                    filename=media_file.name,
                    local_path=str(media_file),
                    file_size=media_file.stat().st_size,
                    media_type=classify_media_type(media_file),
                    ingestion_status="COMPLETE",
                )

                sidecar = resolve_sidecar(media_file)
                if sidecar and merge_sidecar_metadata(record, sidecar):
                    result.with_sidecar += 1

                # Fall back to embedded EXIF when the sidecar left timestamp/GPS empty.
                if record.taken_timestamp is None or record.latitude is None:
                    if extract_exif_metadata(record, media_file):
                        result.from_exif += 1

                if record.media_type == "VIDEO" and extract_keyframes_enabled:
                    kf = extract_keyframes(media_file)
                    if kf.primary is not None:
                        record.keyframe_path = str(kf.primary)
                    if kf.paths:
                        record.keyframe_paths = json.dumps([str(p) for p in kf.paths])
                    if kf.duration is not None:
                        record.duration_seconds = kf.duration

                db.insert_media_record(record)
                result.ingested += 1
            except Exception as e:  # noqa: BLE001 - log-and-continue per PRD §4.4
                result.errors += 1
                _record_error(db, media_file, str(e))
            finally:
                progress.advance(task)

    db.commit()
    return result


def _record_error(db: Database, media_file: Path, message: str) -> None:
    """Persist an ERROR row so a failed file isn't silently retried forever."""
    try:
        record = MediaRecord(
            filename=media_file.name,
            local_path=str(media_file),
            file_size=media_file.stat().st_size if media_file.exists() else 0,
            media_type=classify_media_type(media_file),
            ingestion_status="ERROR",
            error_message=message[:500],
        )
        db.insert_media_record(record)
    except Exception:  # noqa: BLE001 - never let error-logging crash ingestion
        pass
