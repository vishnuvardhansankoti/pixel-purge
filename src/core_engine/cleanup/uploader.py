"""Re-upload the curated set to Google Photos with rate limiting + resume.

Upload is a two-step API dance: POST the bytes to get an upload token, then
`mediaItems:batchCreate` to materialize items. Resume skips anything already
marked UPLOADED in the manifest, so a crash mid-run costs at most one batch.

The raw-bytes upload is factored into `_upload_bytes` (injectable) so the retry /
resume / batching logic is unit-testable without real network or credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import Progress

from ..database import Database
from ..models import MediaRecord

console = Console()

BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.8  # ~75 req/min
MAX_RETRIES = 6


@dataclass
class UploadResult:
    uploaded: int = 0
    failed: int = 0
    skipped: int = 0


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def upload_curated_set(
    db: Database,
    service,
    staging_dir: Path,
    *,
    upload_bytes: Callable[[object, Path], str] | None = None,
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    batch_size: int = BATCH_SIZE,
) -> UploadResult:
    """Upload keeper files from staging_dir. Resumes from manifest upload_status."""
    upload_bytes = upload_bytes or _upload_bytes
    keepers = db.get_keepers()
    already = db.get_uploaded_item_ids()

    # Match staged files back to their manifest records by filename.
    staged = {p.name: p for p in Path(staging_dir).iterdir() if p.is_file()}
    pending: list[tuple[MediaRecord, Path]] = []
    result = UploadResult()
    for rec in keepers:
        if rec.id in already:
            result.skipped += 1
            continue
        path = staged.get(rec.filename)
        if path is not None:
            pending.append((rec, path))

    with Progress(console=console) as progress:
        task = progress.add_task("Uploading...", total=len(pending))
        for batch in _chunked(pending, batch_size):
            new_items = []
            for rec, path in batch:
                try:
                    token = _with_retries(lambda: upload_bytes(service, path))
                    new_items.append((rec, {
                        "simpleMediaItem": {"uploadToken": token, "fileName": path.name}
                    }))
                    db.set_upload_status(rec.id, "UPLOADING")
                except Exception as e:  # noqa: BLE001
                    result.failed += 1
                    db.set_upload_status(rec.id, "FAILED")
                    console.print(f"[yellow]upload token failed[/yellow] {path.name}: {e}")
                time.sleep(rate_limit_delay)

            if not new_items:
                progress.advance(task, advance=len(batch))
                continue

            response = _with_retries(lambda: service.mediaItems().batchCreate(
                body={"newMediaItems": [it for _, it in new_items]}
            ).execute())

            results = response.get("newMediaItemResults", [])
            for (rec, _), item_result in zip(new_items, results):
                status = item_result.get("status", {})
                if status.get("code", 0) == 0 and item_result.get("mediaItem"):
                    db.set_upload_status(rec.id, "UPLOADED",
                                         item_result["mediaItem"].get("id"))
                    result.uploaded += 1
                else:
                    db.set_upload_status(rec.id, "FAILED")
                    result.failed += 1
            db.commit()
            progress.advance(task, advance=len(batch))

    console.print(
        f"[green]Upload:[/green] {result.uploaded} uploaded, "
        f"{result.failed} failed, {result.skipped} already done."
    )
    return result


def _with_retries(fn, max_retries: int = MAX_RETRIES):
    """Exponential backoff for transient API errors (429/5xx)."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _upload_bytes(service, path: Path) -> str:
    """POST raw bytes to the Photos upload endpoint, returning an upload token."""
    import google.auth.transport.requests as tr

    creds = service._http.credentials  # reuse the service's authorized creds
    session = tr.AuthorizedSession(creds)
    headers = {
        "Content-type": "application/octet-stream",
        "X-Goog-Upload-Protocol": "raw",
        "X-Goog-Upload-File-Name": path.name,
    }
    resp = session.post(
        "https://photoslibrary.googleapis.com/v1/uploads",
        data=path.read_bytes(), headers=headers, timeout=120,
    )
    resp.raise_for_status()
    return resp.text
