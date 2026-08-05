"""Local dashboard API (FastAPI). Reads the manifest, streams thumbnails from disk.

Bound to localhost; the OS user boundary is the security boundary, so there is no
auth layer. A new DB connection is opened per request (SQLite + threadpool safety).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..database import Database
from ..ingestion.decode import UnsupportedImageError
from ..models import MediaRecord
from .thumbs import render_thumbnail

STATIC_DIR = Path(__file__).parent / "static"

_ITEM_FIELDS = (
    "id", "filename", "taken_timestamp", "file_size", "latitude", "longitude",
    "is_duplicate", "duplicate_of", "dedup_tier", "hamming_distance",
    "ai_label", "blur_score", "ocr_text_ratio", "face_count",
    "classification_bucket", "classification_confidence", "classification_reasoning",
    "keeper_status",
)


def item_dict(rec: MediaRecord) -> dict:
    d = {f: getattr(rec, f, None) for f in _ITEM_FIELDS}
    d["thumb"] = f"/thumb/{rec.id}"
    return d


def create_app(db_path: Path | str) -> FastAPI:
    app = FastAPI(title="Pixel Purge Dashboard")
    db_path = str(db_path)

    def get_db() -> Database:
        db = Database(db_path)
        try:
            yield db
        finally:
            db.close()

    # ---- app shell ----
    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    # ---- summary ----
    @app.get("/api/summary")
    def summary(db: Database = Depends(get_db)):
        return {
            "dedup": db.dedup_stats(),
            "vision": db.vision_stats(),
            "faces": db.face_cluster_stats(),
            "cleanup": db.cleanup_summary(),
        }

    # ---- review ----
    @app.get("/api/review")
    def review(bucket: str | None = Query(None), db: Database = Depends(get_db)):
        items = [item_dict(r) for r in db.get_items_for_review()]
        if bucket and bucket != "ALL":
            items = [i for i in items if i["classification_bucket"] == bucket]
        return {"items": items, "count": len(items)}

    @app.post("/api/items/{item_id}/status")
    def set_status(item_id: int, payload: dict = Body(...), db: Database = Depends(get_db)):
        status = payload.get("status")
        if status not in ("KEEP", "DELETE", "REVIEW", "PENDING"):
            raise HTTPException(400, "status must be KEEP|DELETE|REVIEW|PENDING")
        if db.get_item(item_id) is None:
            raise HTTPException(404, "item not found")
        db.set_keeper_status(item_id, status)
        db.commit()
        return {"id": item_id, "keeper_status": status}

    # ---- dedup ----
    @app.get("/api/dedup")
    def dedup(db: Database = Depends(get_db)):
        clusters = []
        for c in db.get_dedup_clusters():
            clusters.append({
                "keeper": item_dict(c["keeper"]),
                "members": [item_dict(m) for m in c["members"]],
                "savings_bytes": sum(m.file_size for m in c["members"]),
            })
        return {"clusters": clusters, "count": len(clusters)}

    # ---- faces ----
    @app.get("/api/faces")
    def faces(db: Database = Depends(get_db)):
        clusters = db.get_face_clusters()
        for c in clusters:
            c["thumb"] = f"/thumb/{c['representative_item_id']}"
        return {"clusters": clusters, "count": len(clusters)}

    @app.get("/api/faces/{cluster_id}")
    def face_items(cluster_id: str, db: Database = Depends(get_db)):
        items = [item_dict(r) for r in db.get_face_cluster_items(cluster_id)]
        return {"cluster_id": cluster_id, "name": db.get_face_name(cluster_id),
                "items": items, "count": len(items)}

    @app.post("/api/faces/{cluster_id}/name")
    def name_face(cluster_id: str, payload: dict = Body(...), db: Database = Depends(get_db)):
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        db.set_face_name(cluster_id, name)
        db.commit()
        return {"cluster_id": cluster_id, "name": name}

    @app.post("/api/faces/merge")
    def merge_faces(payload: dict = Body(...), db: Database = Depends(get_db)):
        source, target = payload.get("source"), payload.get("target")
        if not source or not target or source == target:
            raise HTTPException(400, "distinct source and target required")
        moved = db.merge_face_clusters(source, target)
        return {"source": source, "target": target, "moved": moved}

    # ---- thumbnails ----
    @app.get("/thumb/{item_id}")
    def thumb(item_id: int, size: int = Query(256, ge=32, le=1024),
              db: Database = Depends(get_db)):
        item = db.get_item(item_id)
        if item is None:
            raise HTTPException(404, "item not found")
        try:
            data = render_thumbnail(item.visual_path, size)
        except UnsupportedImageError as e:
            raise HTTPException(415, str(e))
        except (OSError, ValueError):
            raise HTTPException(404, "thumbnail unavailable")
        return Response(data, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})

    return app
