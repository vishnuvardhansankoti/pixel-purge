"""Blur and text-density heuristics — pure numpy (no OpenCV), so they are real and
unit-testable without heavy dependencies.

- blur_score: variance of the Laplacian. Low variance == few sharp edges == blurry.
- text_density: fraction of the image occupied by dense high-frequency edge regions,
  a cheap proxy for "screenshot / document / text-heavy" without running real OCR.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Thresholds (documented so downstream rules and tests share one source of truth).
BLUR_THRESHOLD = 50.0        # Laplacian variance below this == BLURRY
TEXT_DENSITY_THRESHOLD = 0.30  # edge-region fraction above this == text-heavy


def _to_gray(image: Image.Image, max_side: int = 512) -> np.ndarray:
    """Downscale (for speed/consistency) and convert to a float64 grayscale array."""
    img = image.convert("L")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(img, dtype=np.float64)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """Convolve with a 3x3 Laplacian kernel (edge/detail response)."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(gray, 1, mode="reflect")
    # Vectorised 3x3 convolution via shifted slices.
    out = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    return out


def blur_score(image: Image.Image) -> float:
    """Variance of the Laplacian. Higher == sharper."""
    gray = _to_gray(image)
    if gray.size == 0:
        return 0.0
    return float(_laplacian(gray).var())


EDGE_FRACTION_CUTOFF = 0.15  # normalized |Laplacian| above this counts as an edge pixel


def text_density(image: Image.Image) -> float:
    """Estimate text/screenshot-ness as the fraction of the image made of strong edges.

    Screenshots and documents spread sharp edges across most of the frame; a photo
    of a subject on a background concentrates them in a small region. We normalize
    the Laplacian magnitude and return the fraction of pixels above a fixed cutoff.
    This is a cheap proxy for "text-heavy", not real OCR.
    """
    gray = _to_gray(image)
    if gray.size == 0:
        return 0.0
    edges = np.abs(_laplacian(gray))
    peak = edges.max()
    if peak <= 0:
        return 0.0
    return float((edges / peak > EDGE_FRACTION_CUTOFF).mean())


def is_blurry(score: float) -> bool:
    return score < BLUR_THRESHOLD


def is_text_heavy(density: float) -> bool:
    return density > TEXT_DENSITY_THRESHOLD
