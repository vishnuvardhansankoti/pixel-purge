"""Accuracy evaluation harness for the vision classifier [H2].

Substantiates the ≥90% (batch) / ≥85% (delta) accuracy targets by measuring the
classifier against a **labeled set** — a CSV of `path,bucket` rows the user
curates from their own library. The metric computation is pure and unit-tested;
running against real photos additionally needs the `[vision]` extra.

Labeled-set format (`eval_manifest.csv`):

    path,bucket
    /photos/receipt_01.heic,ADHOC_PURGE
    /photos/paris_eiffel.jpg,TRIP
    /photos/kids_birthday.jpg,FAMILY_KEEP
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .taxonomy import BUCKETS

# A classifier here is any callable: image path -> predicted bucket string.
ClassifyFn = Callable[[str], str]


@dataclass
class EvalCase:
    path: str
    expected: str


@dataclass
class Metrics:
    total: int
    correct: int
    confusion: dict[str, dict[str, int]]  # expected -> predicted -> count
    per_bucket: dict[str, dict[str, float]]  # bucket -> {precision, recall, f1, support}

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def load_eval_manifest(path: Path) -> list[EvalCase]:
    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bucket = (row.get("bucket") or "").strip()
            if bucket not in BUCKETS:
                raise ValueError(f"invalid bucket {bucket!r} in eval manifest (row {row})")
            cases.append(EvalCase(path=row["path"].strip(), expected=bucket))
    return cases


def compute_metrics(pairs: list[tuple[str, str]]) -> Metrics:
    """pairs: list of (expected_bucket, predicted_bucket). Pure — no I/O, no model."""
    confusion = {e: {p: 0 for p in BUCKETS} for e in BUCKETS}
    correct = 0
    for expected, predicted in pairs:
        confusion[expected][predicted] += 1
        if expected == predicted:
            correct += 1

    per_bucket: dict[str, dict[str, float]] = {}
    for b in BUCKETS:
        tp = confusion[b][b]
        fn = sum(confusion[b][p] for p in BUCKETS) - tp          # expected b, predicted other
        fp = sum(confusion[e][b] for e in BUCKETS) - tp          # predicted b, expected other
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_bucket[b] = {"precision": precision, "recall": recall, "f1": f1,
                         "support": support}

    return Metrics(total=len(pairs), correct=correct, confusion=confusion,
                   per_bucket=per_bucket)


def evaluate(cases: list[EvalCase], classify_fn: ClassifyFn) -> Metrics:
    """Run classify_fn over each case and compute metrics."""
    pairs = [(c.expected, classify_fn(c.path)) for c in cases]
    return compute_metrics(pairs)


def make_clip_classify_fn(device: str = "auto", model_name: str = "ViT-B-32") -> ClassifyFn:
    """Build a real CLIP-backed image-path -> bucket function (needs [vision] extra)."""
    from ..ingestion.decode import open_image
    from . import quality
    from .clip_tagger import CLIPClassifier, fuse_classification

    classifier = CLIPClassifier(model_name=model_name, device=device)

    def _classify(path: str) -> str:
        image = open_image(path)
        probs = classifier.classify(image)
        result = fuse_classification(
            probs, quality.blur_score(image), quality.text_density(image)
        )
        return result.bucket

    return _classify
