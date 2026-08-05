"""Face detection/embedding (InsightFace) + unsupervised clustering (DBSCAN).

InsightFace `buffalo_l` runs via ONNX Runtime (CoreML execution provider on Apple
Silicon) — faster and more accurate than dlib/HOG, and it installs cleanly on M-series
Macs [H4]. Embeddings are 512-d float32, L2-normalized by the model.

`min_samples=2` (not 3) so people who appear in only two photos still form a cluster
instead of being discarded as noise [H4]. The DB write logic is separated from the
model + sklearn so it stays unit-testable.
"""

from __future__ import annotations

import json

import numpy as np
from rich.console import Console
from rich.progress import Progress

from ..database import Database
from ..ingestion.decode import UnsupportedImageError, open_image
from ..models import MediaRecord

console = Console()

EMBED_DTYPE = np.float32


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=EMBED_DTYPE).tobytes()


def embedding_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=EMBED_DTYPE)


class InsightFaceAnalyzer:
    """Lazy-loading InsightFace wrapper returning (embedding, bbox) per detected face."""

    def __init__(self, det_size: int = 640):
        import insightface  # heavy; imported on first use
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CoreMLExecutionProvider" in available:  # Apple Silicon
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def analyze(self, image) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """Return [(embedding, (top, right, bottom, left)), ...] for a PIL image."""
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for InsightFace
        faces = self.app.get(arr)
        out = []
        for f in faces:
            emb = f.normed_embedding.astype(EMBED_DTYPE)
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            out.append((emb, (y1, x2, y2, x1)))  # (top, right, bottom, left)
        return out


def extract_faces(db: Database, analyzer: InsightFaceAnalyzer) -> int:
    """Detect + embed faces for all pending items. Returns total faces stored."""
    items = db.get_items_for_faces()
    total = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Detecting faces...", total=len(items))
        for item in items:
            try:
                faces = analyzer.analyze(open_image(item.visual_path))
                for idx, (emb, bbox) in enumerate(faces):
                    db.add_face_embedding(item.id, idx, embedding_to_bytes(emb), bbox)
                db.mark_faces_done(item.id, len(faces))
                total += len(faces)
            except (UnsupportedImageError, OSError, ValueError) as e:
                db.mark_faces_done(item.id, 0, error=str(e))
            progress.advance(task)
    db.commit()
    return total


def cluster_faces(embeddings: np.ndarray, eps: float = 0.45, min_samples: int = 2):
    """Run DBSCAN over face embeddings. Returns an array of integer labels (-1 = noise)."""
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    return clustering.fit_predict(embeddings)


def assign_person_clusters(
    db: Database, embedding_rows: list[dict], labels
) -> dict:
    """Write cluster labels back to face_embeddings and aggregate onto media_items.

    Pure DB logic (no model / no sklearn) — unit-testable with hand-made labels.
    `labels[i]` corresponds to `embedding_rows[i]`; -1 means noise (unclustered).
    """
    per_item: dict[int, set[str]] = {}
    n_clusters = 0
    seen = set()

    for row, label in zip(embedding_rows, labels):
        label = int(label)
        if label == -1:
            cluster_id = None
        else:
            cluster_id = f"person_{label:04d}"
            if label not in seen:
                seen.add(label)
                n_clusters += 1
            per_item.setdefault(row["media_item_id"], set()).add(cluster_id)
        db.set_face_cluster(row["id"], cluster_id)

    for media_item_id, ids in per_item.items():
        db.set_person_cluster_ids(media_item_id, json.dumps(sorted(ids)))

    db.commit()
    return {"faces": len(embedding_rows), "clusters": n_clusters}


def run_face_clustering(
    db: Database, eps: float = 0.45, min_samples: int = 2
) -> dict:
    """Full face pipeline: extract embeddings, cluster, assign person ids."""
    analyzer = InsightFaceAnalyzer()
    extract_faces(db, analyzer)

    rows = db.get_all_face_embeddings()
    if not rows:
        console.print("[yellow]Faces:[/yellow] no faces detected.")
        return {"faces": 0, "clusters": 0}

    matrix = np.vstack([embedding_from_bytes(r["embedding"]) for r in rows])
    labels = cluster_faces(matrix, eps=eps, min_samples=min_samples)
    stats = assign_person_clusters(db, rows, labels)

    n_noise = int(sum(1 for label in labels if int(label) == -1))
    console.print(
        f"[green]Faces:[/green] {stats['faces']} faces -> {stats['clusters']} "
        f"people ({n_noise} unclustered)."
    )
    return stats
