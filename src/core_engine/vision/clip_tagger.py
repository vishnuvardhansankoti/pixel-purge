"""CLIP zero-shot classification into the unified taxonomy, fused with the
blur/text-density heuristics.

The pure decision logic (`fuse_classification`) is separated from the model so it
can be unit-tested with mocked scores; the heavy torch/open_clip import lives
inside `CLIPClassifier` and is loaded lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress

from ..database import Database
from ..ingestion.decode import UnsupportedImageError, open_image
from . import quality
from .taxonomy import (
    ADHOC_PURGE,
    BUCKETS,
    FAMILY_KEEP,
    PROMPTS,
    prompt_bucket,
    prompt_label,
)

console = Console()


@dataclass
class ClassificationResult:
    bucket: str
    confidence: float
    label: str
    reasoning: str
    blur_score: float
    text_density: float


def fuse_classification(
    prompt_probs: dict[str, float],
    blur: float,
    density: float,
) -> ClassificationResult:
    """Combine CLIP prompt probabilities with quality signals into one decision.

    Pure function — no model, no I/O — so it is fully unit-testable.
    """
    # Aggregate prompt probabilities up to bucket level for a stable confidence.
    bucket_prob = {b: 0.0 for b in BUCKETS}
    for p, prob in prompt_probs.items():
        bucket_prob[prompt_bucket(p)] += prob

    top_prompt = max(prompt_probs, key=prompt_probs.get)
    clip_bucket = prompt_bucket(top_prompt)
    clip_label = prompt_label(top_prompt)
    clip_conf = bucket_prob[clip_bucket]
    top_prob = prompt_probs[top_prompt]
    base_reason = f"CLIP:{clip_label} p={top_prob:.2f}"

    # Override 1: text-heavy (screenshot/document) beats the CLIP subject.
    if quality.is_text_heavy(density):
        return ClassificationResult(
            ADHOC_PURGE, max(clip_conf, density), "text_heavy",
            f"text-density {density:.2f} > {quality.TEXT_DENSITY_THRESHOLD}; {base_reason}",
            blur, density,
        )

    # Override 2: blurry — unless CLIP is confident this is a person/family shot
    # (we don't want to purge a slightly-soft family photo).
    if quality.is_blurry(blur) and not (clip_bucket == FAMILY_KEEP and top_prob > 0.5):
        return ClassificationResult(
            ADHOC_PURGE, max(0.5, clip_conf), "blurry",
            f"blur {blur:.0f} < {quality.BLUR_THRESHOLD}; {base_reason}",
            blur, density,
        )

    return ClassificationResult(
        clip_bucket, clip_conf, clip_label,
        f"{base_reason}; blur {blur:.0f}, text {density:.2f}",
        blur, density,
    )


class CLIPClassifier:
    """Lazy-loading open_clip wrapper. Instantiating loads the model + text features."""

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: str = "auto"):
        import open_clip  # heavy; imported on first use
        import torch

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._torch = torch

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            tokens = tokenizer(PROMPTS).to(device)
            text_features = self.model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify(self, image) -> dict[str, float]:
        """Return {prompt: probability} for one PIL image."""
        torch = self._torch
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = (100.0 * feats @ self.text_features.T).softmax(dim=-1)
            probs = logits[0].cpu().tolist()
        return dict(zip(PROMPTS, probs))


def run_vision(db: Database, device: str = "auto", model_name: str = "ViT-B-32") -> int:
    """Classify all pending non-duplicate items. Returns count classified."""
    items = db.get_items_for_vision()
    if not items:
        console.print("[green]Vision:[/green] nothing to classify.")
        return 0

    classifier = CLIPClassifier(model_name=model_name, device=device)
    done = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Vision (CLIP)...", total=len(items))
        for item in items:
            try:
                image = open_image(item.visual_path)
                probs = classifier.classify(image)
                result = fuse_classification(
                    probs, quality.blur_score(image), quality.text_density(image)
                )
                db.update_vision(
                    item.id,
                    ai_caption=max(probs, key=probs.get),
                    ai_label=result.label,
                    blur_score=result.blur_score,
                    ocr_text_ratio=result.text_density,
                    bucket=result.bucket,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                )
                done += 1
            except (UnsupportedImageError, OSError, ValueError) as e:
                db.mark_vision_error(item.id, str(e))
            progress.advance(task)
    db.commit()
    console.print(f"[green]Vision:[/green] classified {done} item(s).")
    return done
