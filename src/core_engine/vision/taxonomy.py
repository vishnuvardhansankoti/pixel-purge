"""Unified classification taxonomy (shared by Module C batch and Module E delta).

CLIP zero-shot works by scoring the image against a set of natural-language label
prompts; each prompt maps to one of the four unified buckets. This keeps the
batch pipeline and the delta classifier in lock-step [H2].
"""

from __future__ import annotations

# The four buckets. OTHER is always routed to human review, never auto-purged.
ADHOC_PURGE = "ADHOC_PURGE"
TRIP = "TRIP"
FAMILY_KEEP = "FAMILY_KEEP"
OTHER = "OTHER"

BUCKETS = (ADHOC_PURGE, TRIP, FAMILY_KEEP, OTHER)

# CLIP prompt -> (bucket, short label). Prompts are phrased as "a photo of ..."
# style captions, which is what CLIP's contrastive training expects.
LABEL_PROMPTS: dict[str, tuple[str, str]] = {
    "a screenshot of a phone or computer screen": (ADHOC_PURGE, "screenshot"),
    "a photo of a receipt, invoice, or bill": (ADHOC_PURGE, "receipt"),
    "a scanned document or page of text": (ADHOC_PURGE, "document"),
    "a meme or an image that is mostly text": (ADHOC_PURGE, "meme_or_text"),
    "a QR code or a barcode": (ADHOC_PURGE, "qr_or_barcode"),
    "a portrait photo of a person or a selfie": (FAMILY_KEEP, "person"),
    "a photo of a family, children, or a group of people": (FAMILY_KEEP, "people"),
    "a photo of a pet or an animal": (FAMILY_KEEP, "pet"),
    "a photo of food or a meal": (FAMILY_KEEP, "food"),
    "a celebration, party, or social gathering": (FAMILY_KEEP, "celebration"),
    "a travel photo of a landmark or tourist attraction": (TRIP, "landmark"),
    "a landscape photo of nature, mountains, or a beach": (TRIP, "scenery"),
    "a photo of a city street or building": (TRIP, "cityscape"),
    "an ordinary snapshot of a random object": (OTHER, "object"),
}

# Deterministic prompt ordering so index<->label mapping is stable across runs.
PROMPTS: list[str] = list(LABEL_PROMPTS.keys())


def prompt_bucket(prompt: str) -> str:
    return LABEL_PROMPTS[prompt][0]


def prompt_label(prompt: str) -> str:
    return LABEL_PROMPTS[prompt][1]
