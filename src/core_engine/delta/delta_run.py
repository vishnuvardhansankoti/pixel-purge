"""Module E orchestrator: local monthly delta over an incremental Takeout/Picker export.

The delta is just the batch pipeline over a small new input, deduplicated against
the existing manifest, then CLIP-classified with a GPS→TRIP override. Fully local;
the only idempotency needed comes from `classification_bucket IS NULL` + a stored
watermark [H5].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.progress import Progress

from ..config import Config
from ..database import Database
from ..dedup import run_dedup as run_dedup_pipeline
from ..ingestion import ingest as run_ingest
from ..ingestion.decode import UnsupportedImageError, open_image
from ..vision import quality
from ..vision.clip_tagger import fuse_classification
from .gps_override import apply_gps_override

console = Console()


class SupportsClassify(Protocol):
    def classify(self, image) -> dict[str, float]: ...


@dataclass
class DeltaResult:
    ingested: int = 0
    classified: int = 0
    errors: int = 0
    watermark: int = 0
    buckets: dict[str, int] = field(default_factory=dict)


def classify_items(
    db: Database,
    items,
    classifier: SupportsClassify,
    home_lat: float,
    home_lon: float,
    trip_distance_miles: float,
) -> DeltaResult:
    """Classify a list of items with CLIP + quality fusion + GPS override."""
    result = DeltaResult()
    max_ts = db.get_delta_watermark()

    with Progress(console=console) as progress:
        task = progress.add_task("Delta classify...", total=len(items))
        for item in items:
            try:
                image = open_image(item.visual_path)
                probs = classifier.classify(image)
                res = fuse_classification(
                    probs, quality.blur_score(image), quality.text_density(image)
                )
                res = apply_gps_override(
                    res, item.latitude, item.longitude,
                    home_lat, home_lon, trip_distance_miles,
                )
                db.update_vision(
                    item.id,
                    ai_caption=max(probs, key=probs.get),
                    ai_label=res.label,
                    blur_score=res.blur_score,
                    ocr_text_ratio=res.text_density,
                    bucket=res.bucket,
                    confidence=res.confidence,
                    reasoning=res.reasoning,
                )
                result.classified += 1
                result.buckets[res.bucket] = result.buckets.get(res.bucket, 0) + 1
                if item.taken_timestamp:
                    max_ts = max(max_ts, item.taken_timestamp)
            except (UnsupportedImageError, OSError, ValueError) as e:
                result.errors += 1
                db.mark_vision_error(item.id, str(e))
            progress.advance(task)

    db.set_delta_watermark(max_ts)
    db.commit()
    result.watermark = max_ts
    return result


def delta_run(
    source_path: Path,
    db: Database,
    config: Config | None = None,
    *,
    resume: bool = True,
    dry_run: bool = False,
    dedup: bool = True,
    classifier: SupportsClassify | None = None,
) -> DeltaResult:
    """Ingest a new export, dedup vs the manifest, and classify new items."""
    config = config or Config()

    ingest_result = run_ingest(source_path, db, resume=resume, dry_run=dry_run)
    if dry_run:
        console.print(f"[cyan]Delta dry run:[/cyan] {ingest_result.discovered} discovered, "
                      f"{ingest_result.discovered - ingest_result.skipped_existing} new.")
        return DeltaResult(ingested=0)

    if dedup:
        run_dedup_pipeline(
            db,
            gps_radius_m=config.dedup.gps_radius_meters,
            time_window_min=config.dedup.time_window_minutes,
            hamming_threshold=config.dedup.hamming_threshold,
            text_hamming_threshold=config.dedup.text_hamming_threshold,
        )

    items = db.get_items_for_delta()
    if not items:
        console.print("[green]Delta:[/green] no new items to classify.")
        return DeltaResult(ingested=ingest_result.ingested, watermark=db.get_delta_watermark())

    if classifier is None:  # lazy: only import torch/open_clip when actually classifying
        from ..vision.clip_tagger import CLIPClassifier

        classifier = CLIPClassifier(model_name=config.delta.model, device=config.delta.device)

    result = classify_items(
        db, items, classifier,
        config.home_latitude, config.home_longitude, config.delta.trip_distance_miles,
    )
    result.ingested = ingest_result.ingested
    console.print(
        f"[green]Delta:[/green] classified {result.classified} new item(s); "
        f"watermark -> {result.watermark}."
    )
    return result
